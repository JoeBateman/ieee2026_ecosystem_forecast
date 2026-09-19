DATA_DIR = "../data"
GLOB_DIR = DATA_DIR + '/data_global'
STAT_DIR = DATA_DIR + '/data_stats'

TREE_BAND = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_SITE, N_YEAR, N_MONTH, N_FEA, N_OUT = 54152, 40, 12, 136, 7

print('Data dir :', DATA_DIR)
print('Targets  :', TREE_BAND)
print('Seed ages:', AGES)

mask2d = np.load(STAT_DIR + '/mask2d.npy')                # (360, 720), coordinates of land pixels

# Naive evolution of targets with age, based on the mean and slope of each target across age and location.
age_triplet = np.load(STAT_DIR + '/age_triplet.npz')            # (15,), (15,), (15,)
TRI_AGE, TRI_MEAN, TRI_SLOPE = age_triplet['age'], age_triplet['age_mean'], age_triplet['age_slope']

N_MONTH = 12

stats = np.load(STAT_DIR + '/data_stats.npz')
x_mean, x_std = stats['x_mean'], stats['x_std'] # stats for normalization of the input variables (features, 136) 
y_mean, y_std = stats['y_mean'], stats['y_std']  # stats for normalization of the output variables (targets, 7)

def normalize_x(x, x_mean=x_mean, x_std=x_std):
    return (x - x_mean) / (x_std + 1e-10)

def inv_x(x, x_mean=x_mean, x_std=x_std):
    return x * (x_std + 1e-10) + x_mean

class TeacherStudentRollout(nn.Module):
    """
    Model-agnostic teacher/student wrapper for the recursive year-by-year carbon-state forecast.

    Wraps an externally-defined "core" model -- any `nn.Module` whose `forward` maps a batch of
    per-year inputs `(N_site, N_age, N_month, N_fea + N_out + 1)` (climate features ++ previous
    year's target vector `v` ++ scaled seed age) to that year's predicted `(N_site, N_age, N_out)`
    targets, exactly `MonthMetaModel`'s existing signature -- and adds the teacher-forced /
    autoregressive rollout mechanics on top, without knowing anything about the core model's
    internals.

    - `teacher_rollout` feeds the TRUE previous year's `v` at every step (teacher forcing).
    - `student_rollout` feeds the model's OWN previous prediction at every step (full
      autoregressive rollout), seeded only by the true year-0 state -- matching test-time/
      deployment conditions.
    - `compute_losses` runs both and returns the supervised losses plus a distillation loss that
      pulls the student's per-step prediction toward the teacher's (stop-gradient) prediction for
      the same year.
    - `update_teacher` nudges the teacher's weights toward the student's by exponential moving
      average (the "mean teacher" pattern, Tarvainen & Valpola 2017) instead of ever copying the
      student onto the teacher outright -- the teacher is meant to become a slow, smoothed
      trajectory of the student's weights (which tends to generalize better than the raw student at
      any single step, similar to weight averaging/SWA), not to collapse into being the student.
      That smoothed teacher then keeps supervising the student via `distill_loss`, so the two can
      improve together rather than the teacher acting as a fixed ceiling.

    By default `student_model` is omitted and both paths share one set of weights (`core_model`),
    which are trained jointly through both losses. Pass a distinct `student_model` (e.g. a copy of
    an already-converged, frozen teacher) to decouple them -- see `DATASET_NOTES.md` §11's "warm
    start" alternative. When decoupled, the teacher rollout is run under `torch.no_grad()` since
    its output is only ever used detached (as the distillation target), saving the memory of a
    computation graph nobody will backprop through.
    """

    def __init__(self, core_model, student_model=None, ages=AGES, n_months=N_MONTH):
        super().__init__()
        self.teacher_model = core_model
        self.student_model = student_model if student_model is not None else core_model
        self.shares_weights = student_model is None

        self.ages = np.asarray(ages)
        self.n_months = n_months
        self.register_buffer(
            "age_scaled", torch.tensor(self.ages / self.ages.max(), dtype=torch.float32)
        )

    def _build_step_input(self, x_year, v):
        """
        x_year: (N_site, N_age, N_month, N_fea)
        v:      (N_site, N_age, N_out) -- the previous year's target state
        Returns (N_site, N_age, N_month, N_fea + N_out + 1)
        """
        n_site, n_age = x_year.shape[0], x_year.shape[1]
        age_col = self.age_scaled.to(x_year.device).view(1, n_age, 1).expand(n_site, -1, -1)  # (N_site, N_age, 1)
        v_month = v.unsqueeze(2).expand(-1, -1, self.n_months, -1)
        age_month = age_col.unsqueeze(2).expand(-1, -1, self.n_months, -1)
        return torch.cat([x_year, v_month, age_month], dim=-1)

    def rollout(self, model, x, y_true, start_year=0, end_year=N_YEAR, teacher_forcing=True):
        """
        Roll `model` forward year-by-year over [start_year, end_year).

        x:      (N_site, N_year, N_month, N_fea) inputs (whatever units/scale the model expects)
        y_true: (N_site, N_age, N_year+1, N_out) targets, year 0 = init state (same units as x's
                target space). Only `y_true[:, :, start_year, :]` (the init state) is required when
                `teacher_forcing=False` -- later years are read from `y_true` only to supply `v`
                when teacher forcing.

        Returns: predictions, shape (N_site, N_age, end_year - start_year, N_out)
        """
        v = y_true[:, :, start_year, :]
        preds = []
        for t in range(start_year, end_year):
            n_age = v.shape[1]
            x_year = x[:, t, :, :].unsqueeze(1).expand(-1, n_age, -1, -1)  # (N_site, N_age, N_month, N_fea)
            step_in = self._build_step_input(x_year, v)
            pred = model(step_in)  # (N_site, N_age, N_out)
            preds.append(pred)
            v = y_true[:, :, t + 1, :] if teacher_forcing else pred
        return torch.stack(preds, dim=2)  # (N_site, N_age, T, N_out)

    def teacher_rollout(self, x, y_true, start_year=0, end_year=N_YEAR):
        return self.rollout(self.teacher_model, x, y_true, start_year, end_year, teacher_forcing=True)

    def student_rollout(self, x, y_true, start_year=0, end_year=N_YEAR):
        return self.rollout(self.student_model, x, y_true, start_year, end_year, teacher_forcing=False)

    def compute_losses(self, x, y_true, start_year=0, end_year=N_YEAR, distill_weight=1.0):
        """
        Runs both rollouts and returns the loss components for one training step:
          - 'teacher_loss': teacher predictions vs true targets -- supervises the teacher path.
          - 'student_loss': student predictions vs true targets -- the actual objective (matches
            the autoregressive test-time protocol).
          - 'distill_loss': student predictions vs teacher predictions (stop-gradient) -- extra,
            stable gradient signal pulling the harder autoregressive rollout toward the teacher's.
          - 'total': student_loss + distill_weight * distill_loss, plus teacher_loss when the two
            paths share weights (so the shared core model is also directly supervised).
        """
        targets = y_true[:, :, start_year + 1:end_year + 1, :]  # (N_site, N_age, T, N_out)

        if self.shares_weights:
            teacher_pred = self.teacher_rollout(x, y_true, start_year, end_year)
        else:
            # Teacher is a separate, already-trained network here -- its output is only ever used
            # detached (as the distillation target), so skip building its autograd graph entirely.
            with torch.no_grad():
                teacher_pred = self.teacher_rollout(x, y_true, start_year, end_year)

        student_pred = self.student_rollout(x, y_true, start_year, end_year)

        teacher_loss = nn.functional.mse_loss(teacher_pred, targets)
        student_loss = nn.functional.mse_loss(student_pred, targets)
        distill_loss = nn.functional.mse_loss(student_pred, teacher_pred.detach())

        total = student_loss + distill_weight * distill_loss
        if self.shares_weights:
            total = total + teacher_loss

        return {
            "teacher_loss": teacher_loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
            "total": total,
        }

    @torch.no_grad()
    def update_teacher(self, decay=0.999):
        """
        Exponential-moving-average update of the teacher's weights toward the student's:
            teacher <- decay * teacher + (1 - decay) * student
        the "mean teacher" pattern (Tarvainen & Valpola, 2017). `decay` close to 1 (e.g. 0.999)
        keeps the teacher slow-moving -- a smoothed trajectory of the student's weights rather than
        a copy of the student at any single step, which is why the teacher can end up *better* than
        a plain snapshot of the student (analogous to stochastic weight averaging). Call this once
        per optimizer step (after `student_optimizer.step()`), not once per epoch, so the teacher
        tracks the student's weight trajectory smoothly rather than in large jumps.

        Requires a distinct `student_model` -- meaningless (and refused) when the two paths share
        weights, since there the "teacher" and "student" are the same tensors already.
        """
        if self.shares_weights:
            raise RuntimeError(
                "update_teacher() requires a distinct student_model; teacher and student share "
                "weights here, so there is nothing separate to average toward."
            )
        for t_param, s_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            t_param.mul_(decay).add_(s_param, alpha=1 - decay)
        for t_buf, s_buf in zip(self.teacher_model.buffers(), self.student_model.buffers()):
            if torch.is_floating_point(t_buf):
                t_buf.mul_(decay).add_(s_buf, alpha=1 - decay)
            else:
                t_buf.copy_(s_buf)  # e.g. BatchNorm's num_batches_tracked -- an integer count, not averaged