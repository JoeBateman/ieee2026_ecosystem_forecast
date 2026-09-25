import numpy as np
import pandas as pd
import torch
from torch import nn

baseline = pd.DataFrame({
    'RMSE': [3.060, 1.353, 1.371, 0.588, 0.374, 0.186, 0.182],
    'MAE': [1.534, 0.594, 0.724, 0.270, 0.160, 0.079, 0.088],
    'R2': [0.914, 0.935, 0.951, 0.913, 0.921, 0.917, 0.914],
    'delta': [0.440, 0.164, 0.166, 0.414, 0.315, 0.159, 0.182],
    'cumulative': [4.426, 1.751, 2.022, 0.649, 0.447, 0.222, 0.20]
}, index=['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh'])

DATA_DIR = "../data"
GLOB_DIR = DATA_DIR + '/data_global'
STAT_DIR = DATA_DIR + '/data_stats'

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_SITE, N_YEAR, N_MONTH, N_FEA, N_OUT = 54152, 40, 12, 136, 7


def metrics_table_global(true_seq, pred_seq, target_names=None):
    """
    true_seq, pred_seq: (A, S, N_YEAR, N_OUT) physical-unit arrays, A=n_ages, S=n_sites.
    Mirrors the tutorial's `metrics_table`: RMSE/MAE/R2 pooled over (age, site, year), plus the
    paper's delta (year-over-year change) and cumulative (final-year) errors. Shared across
    notebooks (originally from meta_models.ipynb) since every model's rollout eval needs the same
    scoring, whatever produced `true_seq`/`pred_seq`.
    """
    target_names = TREE_BAND if target_names is None else target_names
    t = true_seq.reshape(-1, true_seq.shape[-1])
    p = pred_seq.reshape(-1, pred_seq.shape[-1])
    rmse = np.sqrt(np.mean((t - p) ** 2, axis=0))
    mae = np.mean(np.abs(t - p), axis=0)
    ss_res = np.sum((t - p) ** 2, axis=0)
    ss_tot = np.sum((t - t.mean(0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot
    dt = np.diff(true_seq, axis=2); dp = np.diff(pred_seq, axis=2)
    delta = np.sqrt(np.mean((dt - dp) ** 2, axis=(0, 1, 2)))
    ce = np.sqrt(np.mean((true_seq[:, :, -1, :] - pred_seq[:, :, -1, :]) ** 2, axis=(0, 1)))
    return pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'R2': r2, 'delta': delta, 'cumulative': ce},
                         index=target_names).round(3)

# 136 input feature names (glob_X_fea.npy), in verified on-disk channel order (agent/LOGBOOK.md
# §1.3) -- NOT the contiguous-block order implied by DATASET_NOTES.md's Table-2 reading. The
# hourly humidity/temperature/radiation/wind channels are INTERLEAVED (stride 4: hus, ta_h, rsds,
# sfcWind, repeated per hour-of-day), which was confirmed by their diurnal-cycle shape.
FEA_NAMES = (
    ['ta_m', 'pr'] # Monthly mean temperature and precipitation
    + [f'tsl{i}' for i in range(1, 7)] # Monthly average soil temperature at different depths (tsl1 to tsl6)
    + [f'co2_h{h}' for h in range(24)] # Hourly CO2 concentration
    + ['dst'] # Disturbance rate, to be dropped as it has no variance 
    + [f'{v}_h{h}' for h in range(24) for v in ['hus', 'ta_h', 'rsds', 'sfcWind']] # Monthly average hourly air specific humidity, air temperature, downward surface radiation, and surface wind speed
    + ['k_sat', # Saturated hydraulic conductivity
        's_theta', # Saturated water content in MVG
        'r_theta', # Residual water content in MVG
        'L', # Parameter L in MVG
        'n', # Parameter n in MVG
        'm', # Parameter m in MVG
        'sd' # Soil depth to bedrock
        ]
)

TREE_BAND = ['height', # Vegetation canopy height
            'agb',  # Aboveground biomass
            'soil', # Soil carbon
            'lai', # Leaf area index
            'gpp', # Gross primary productivity (annual)
            'npp', # Net primary productivity (annual)
            'rh'] # Heterotrophic respiration (annual)

def engineering_features(x):
    """
    Features relating to biomass growth based on the inputs:
        - Growing/chilling/frost degree-hours
        - Vapor pressure deficit
        - Light saturation/limitation 
    """
    
    ta_h_indices = [i for i, key in enumerate(FEA_NAMES) if 'ta_h' in key]
    ta_h = x[..., ta_h_indices]
    ta_h_means = x_mean[ta_h_indices]
    ta_h_stds = x_std[ta_h_indices]
    ta_h = inv_x(ta_h, ta_h_means, ta_h_stds)  # Unnormalize the hourly temperature data

    gdd_5c   = np.clip(ta_h - (5 + 273.15), 0, None).mean(-1)        # growing-degree signal, base 5C
    chill_frac = ((ta_h >= 273.15) & (ta_h <= 280.15)).mean(-1)      # fraction of hours in 0-7C chilling band
    frost_frac = (ta_h < 273.15).mean(-1)                             # fraction of hours below freezing
    heat_frac  = (ta_h > 308.15).mean(-1)                             # fraction of hours above ~35C

    hus_h_indices = [i for i, key in enumerate(FEA_NAMES) if 'hus_h' in key]
    hus_h = x[..., hus_h_indices]
    hus_h_means = x_mean[hus_h_indices]
    hus_h_stds = x_std[hus_h_indices]

    hus_h = inv_x(hus_h, hus_h_means, hus_h_stds)  # Unnormalize the hourly specific humidity data

    def vpd_kpa(ta_k, hus, p_kpa=101.325):
        ta_c = ta_k - 273.15
        e_sat = 0.6108 * np.exp(17.27 * ta_c / (ta_c + 237.3))      # Tetens, kPa
        e_act = hus * p_kpa / (0.622 + 0.378 * hus)                  # specific humidity -> actual vapor pressure
        return np.clip(e_sat - e_act, 0, None)

    vpd = vpd_kpa(ta_h, hus_h)                             # vapor pressure deficit
    vpd_stress_frac = (vpd > 1.5).mean(-1)                             # fraction of hours with VPD > 1.5 kPa


    rsds_h_indices = [i for i, key in enumerate(FEA_NAMES) if 'rsds_h' in key]
    rsds_h = x[..., rsds_h_indices]
    rsds_h_means = x_mean[rsds_h_indices]
    rsds_h_stds = x_std[rsds_h_indices]
    rsds_h = inv_x(rsds_h, rsds_h_means, rsds_h_stds)  # Unnormalize the hourly downwelling shortwave radiation data

    par = rsds_h * 2.02  # umol photons m^-2 s^-
    light_limited_frac = ((par > 0) & (par < 20)).mean(-1)   # below compensation point during daylight
    light_saturated_frac = (par > 500).mean(-1)               # at/above sun-leaf saturation

    return (gdd_5c, chill_frac, frost_frac, heat_frac, vpd_stress_frac, light_limited_frac, light_saturated_frac), ('gdd_5c', 'chill_frac', 'frost_frac', 'heat_frac', 'vpd_stress_frac', 'light_limited_frac', 'light_saturated_frac')

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


# ---------------------------------------------------------------------------------------------
# v4 design matrix (agent/LOGBOOK.md Entries 4-8, agent/REPORT.md) -- ported from
# src/features.py + src/dataset.py + src/dataset3.py + src/dataset4.py into the x/y convention
# the other notebooks already use, instead of the src/ scripts' precomputed `cache/*.npy` files.
#
# That convention (see exploring.ipynb / transformer.ipynb):
#   x: (N_SITE, N_YEAR, N_MONTH, N_FEA), NORMALIZED via `normalize_x(x)`.
#   y: (N_SITE, N_AGE, N_YEAR+1, N_MONTH, N_OUT), site-first (after `.transpose(1,0,2,3,4)`),
#      NORMALIZED via `normalize_x(y, y_mean, y_std)`.
#
# Typical usage, right after the existing load-and-normalize cell:
#   x_train, y_train = normalize_x(x_train), normalize_x(y_train, y_mean, y_std)
#   X_train, D_train, y0_train = build_v4_train(x_train, y_train)
#   # ... train a model on (X_train -> D_train), predicting y40 = y0_train + D_hat ...
#   test_x = normalize_x(np.load(".../test_ssp126.npz")["test_x"])
#   test_y0 = np.load(".../test_ssp126.npz")["test_y0"]           # physical units, NOT normalized
#   X_test, y0_test_true = build_v4_test(test_x, test_y0)
# ---------------------------------------------------------------------------------------------

# raw-channel indices used by `annual_features`, derived from FEA_NAMES rather than hardcoded so
# they can't silently drift out of sync with the verified interleaved layout above (LOGBOOK §1.3).
# Leading underscore: these are implementation details, not meant to be used from a notebook.
_TA_IDX = FEA_NAMES.index('ta_m')
_PR_IDX = FEA_NAMES.index('pr')
_TSL1_IDX = FEA_NAMES.index('tsl1'); _TSL3_IDX = FEA_NAMES.index('tsl3'); _TSL6_IDX = FEA_NAMES.index('tsl6')
_CO2_H_IDX = np.array([i for i, k in enumerate(FEA_NAMES) if 'co2_h' in k])
_HUS_H_IDX = np.array([i for i, k in enumerate(FEA_NAMES) if 'hus_h' in k])
_TAH_H_IDX = np.array([i for i, k in enumerate(FEA_NAMES) if 'ta_h_h' in k])
_RSD_H_IDX = np.array([i for i, k in enumerate(FEA_NAMES) if 'rsds_h' in k])
_WND_H_IDX = np.array([i for i, k in enumerate(FEA_NAMES) if 'sfcWind_h' in k])
_STATIC_CH_IDX = np.array([FEA_NAMES.index(n) for n in ['k_sat', 's_theta', 'r_theta', 'L', 'n', 'm', 'sd']])

CLIM_FEAT_NAMES = [
    "ta_mean", "ta_min", "ta_max", "ta_std",
    "pr_mean", "pr_min", "pr_max", "pr_std",
    "tsl1_mean", "tsl3_mean", "tsl6_mean",
    "co2_mean",
    "hus_mean", "hus_std", "hus_amp",
    "tah_mean", "tah_dtr",
    "rsds_mean", "rsds_min", "rsds_max", "rsds_peak",
    "wind_mean", "wind_amp",
    "gdd5", "n_months_gt5c",
    "ta_warmq", "ta_coldq", "pr_wetq", "pr_dryq",
    "ta_pr_corr", "aridity", "vpd_mean",
]
# CO2 excluded: no cross-site variance in training, 100% out of range at test, and the ablation
# confirms dropping it is free in-distribution (LOGBOOK Entry 3.4 / 8.1).
CLIM_DROP = {"co2_mean"}
CLIM_KEEP_IDX = [i for i, n in enumerate(CLIM_FEAT_NAMES) if n not in CLIM_DROP]
CLIM_KEPT_NAMES = [CLIM_FEAT_NAMES[i] for i in CLIM_KEEP_IDX]

STATIC_NAMES = ['log_k_sat', 's_theta', 'r_theta', 'L', 'n', 'm', 'sd']

PROF_AGES = [0, 2, 4, 6, 10, 14]  # indices into AGES -> seed ages 1, 20, 50, 100, 300, 500

# affine map test_y0 = a*true_y0 + b (LOGBOOK Entries 3 & 7): b from exact floors in the
# competition's test_ssp*.npz files, a from the height-agb allometry (the one sampling-invariant
# estimator) with b held fixed. Only relevant to `build_v4_test` -- `res_train4_test8.npz`,
# used elsewhere in these notebooks, is NOT affine-shifted.
AFF_B = np.array([-0.9009, -0.4014, -0.5324, 0.1179, 0.2429, 0.1187, 0.0847], np.float32)
AFF_A = np.array([0.9206, 0.8401, 0.8989, 1.0110, 0.9487, 0.9527, 0.9348], np.float32)


def _cyclic_quarters(x):
    """3-month rolling sums over the month axis (last axis), wrapping December -> January."""
    return x + np.roll(x, -1, axis=-1) + np.roll(x, -2, axis=-1)


def _monthly_reduce(blk):
    """blk: (..., 12, 136) PHYSICAL units -> dict of (..., 12) monthly series."""
    b = blk.astype(np.float32)
    return dict(
        ta=b[..., _TA_IDX], pr=b[..., _PR_IDX],
        tsl1=b[..., _TSL1_IDX], tsl3=b[..., _TSL3_IDX], tsl6=b[..., _TSL6_IDX],
        co2=b[..., _CO2_H_IDX].mean(-1),
        hus=b[..., _HUS_H_IDX].mean(-1), hus_amp=np.ptp(b[..., _HUS_H_IDX], -1),
        tah=b[..., _TAH_H_IDX].mean(-1), tah_dtr=np.ptp(b[..., _TAH_H_IDX], -1),
        rsds=b[..., _RSD_H_IDX].mean(-1), rsds_peak=b[..., _RSD_H_IDX].max(-1),
        wind=b[..., _WND_H_IDX].mean(-1), wind_amp=np.ptp(b[..., _WND_H_IDX], -1),
    )


def annual_features(blk):
    """
    blk: (N, 12, 136) PHYSICAL units, e.g. `inv_x(x[:, year])` -> (N, len(CLIM_FEAT_NAMES))
    float32. Ecologically meaningful annual summaries (growing degree days, warmest/coldest-
    quarter temperature, an aridity index, VPD via Tetens, etc.), ported from src/features.py.

    Deliberately simple: LOGBOOK Entry 5 found a richer set (adding per-year min/max and decade
    anomalies) scored 9% worse, because every training site shares one 40-year weather
    realisation, so those richer features describe *when* in the shared record something
    happened rather than a site property, and transfer worse to the SSP scenarios' different
    climate trajectory.
    """
    m = _monthly_reduce(blk)
    ta, pr = m["ta"], m["pr"]
    q_ta = _cyclic_quarters(ta) / 3.0
    q_pr = _cyclic_quarters(pr)

    # Vapour pressure deficit proxy (Pa): saturation minus actual vapour pressure (Tetens over
    # water; actual from specific humidity at a nominal 101325 Pa).
    esat = 611.0 * np.exp(17.27 * (ta - 273.15) / np.maximum(ta - 35.85, 1.0))
    eact = m["hus"] * 101325.0 / 0.622
    vpd = np.maximum(esat - eact, 0.0)

    pr_ann = pr.sum(-1)
    rsds_ann = m["rsds"].mean(-1)

    tac = ta - ta.mean(-1, keepdims=True)
    prc = pr - pr.mean(-1, keepdims=True)
    corr = (tac * prc).mean(-1) / (ta.std(-1) * pr.std(-1) + 1e-6)

    return np.stack([
        ta.mean(-1), ta.min(-1), ta.max(-1), ta.std(-1),
        pr.mean(-1), pr.min(-1), pr.max(-1), pr.std(-1),
        m["tsl1"].mean(-1), m["tsl3"].mean(-1), m["tsl6"].mean(-1),
        m["co2"].mean(-1),
        m["hus"].mean(-1), m["hus"].std(-1), m["hus_amp"].mean(-1),
        m["tah"].mean(-1), m["tah_dtr"].mean(-1),
        m["rsds"].mean(-1), m["rsds"].min(-1), m["rsds"].max(-1), m["rsds_peak"].mean(-1),
        m["wind"].mean(-1), m["wind_amp"].mean(-1),
        np.maximum(ta - 278.15, 0.0).sum(-1),
        (ta > 278.15).sum(-1).astype(np.float32),
        q_ta.max(-1), q_ta.min(-1), q_pr.max(-1), q_pr.min(-1),
        corr,
        pr_ann / (rsds_ann + 1.0),
        vpd.mean(-1),
    ], axis=-1).astype(np.float32)


def climate_block(x):
    """
    x: (N_SITE, N_YEAR, N_MONTH, N_FEA) NORMALIZED, e.g. `x_train`/`x_test` after `normalize_x`.
    Returns (N_SITE, 3*len(CLIM_KEEP_IDX)): 40-year mean / std / linear trend of the annual
    climate features, CO2 excluded. Ported from src/dataset.py's `climate_block`, adapted to
    compute `annual_features` on the fly from the full cube instead of a precomputed cache --
    fine at the scale these notebooks load (`res_train4_test8.npz`); the src/ scripts cache to
    disk only because they operate on the 14 GB global grid, which these notebooks never touch.
    """
    n_year = x.shape[1]
    feats = np.stack([annual_features(inv_x(x[:, yr])) for yr in range(n_year)], axis=1)
    f = feats[:, :, CLIM_KEEP_IDX]
    t = np.arange(n_year, dtype=np.float32); t = (t - t.mean()) / (t.std() * n_year)
    mean = f.mean(1); std = f.std(1)
    trend = np.tensordot(f - mean[:, None, :], t, axes=([1], [0]))
    return np.concatenate([mean, std, trend], 1)


def climate_block_names():
    return ([f"{n}_mean" for n in CLIM_KEPT_NAMES] + [f"{n}_std" for n in CLIM_KEPT_NAMES]
            + [f"{n}_trend" for n in CLIM_KEPT_NAMES])


def static_block(x):
    """
    x: (N_SITE, N_YEAR, N_MONTH, N_FEA) NORMALIZED. Static soil properties are constant across
    time, so any (year, month) works -- year 0, month 0 is used. Returns (N_SITE, 7), with
    k_sat log10-scaled since it spans 1e4..3e6.
    """
    s = inv_x(x[:, 0, 0, :])[:, _STATIC_CH_IDX].astype(np.float32).copy()
    s[:, 0] = np.log10(np.maximum(s[:, 0], 1.0))
    return s


def profile_block(y0):
    """
    y0: (N_SITE, 15, 7) PHYSICAL units -> (N_SITE, 30) site-level summary of the initial-state-
    vs-age curve. Ported from src/dataset3.py -- the single biggest lever after the delta
    reformulation itself (LOGBOOK Entry 6, +13%): a site's OWN age-500 initial state is a direct
    readout of its carrying capacity, far more informative for a young seed age's eventual
    growth than any climate summary. `test_y0` ships all 15 seed ages per site, so this is
    available at inference time for free.
    """
    h, a = y0[:, :, 0], y0[:, :, 1]
    parts = [
        h[:, PROF_AGES], a[:, PROF_AGES],                             # the curve at 6 anchor ages
        y0.mean(1), y0.max(1),                                        # per-variable mean/max over ages
        (h[:, 14] - h[:, 0])[:, None], (a[:, 14] - a[:, 0])[:, None],  # total span
        (h[:, 6] - h[:, 2])[:, None], (a[:, 6] - a[:, 2])[:, None],    # mid-curve slope
    ]
    return np.concatenate(parts, 1).astype(np.float32)


def profile_names():
    ag = [int(AGES[i]) for i in PROF_AGES]
    return ([f"prof_h_a{x}" for x in ag] + [f"prof_agb_a{x}" for x in ag]
            + [f"prof_mean_{t}" for t in TREE_BAND] + [f"prof_max_{t}" for t in TREE_BAND]
            + ["prof_h_span", "prof_agb_span", "prof_h_midslope", "prof_agb_midslope"])


def assemble_v4(clim, stat, y0):
    """
    clim (N_SITE, C); stat (N_SITE, 7); y0 (N_SITE, 15, 7) physical units -> X (N_SITE*15, F),
    rows ordered site-major then age (row i's seed age is `AGES[i % 15]`, exactly as LOGBOOK
    Entry 6 fixed after the earlier v2-layout bug). Ported verbatim from src/dataset4.py's
    `assemble4`.
    """
    n, na = clim.shape[0], len(AGES)
    prof = profile_block(y0)
    Xc = np.repeat(np.concatenate([clim, stat, prof], 1), na, 0)
    Y0 = y0.reshape(n * na, 7).astype(np.float32)
    ag = np.tile(np.asarray(AGES, np.float32), n)[:, None]
    profr = np.repeat(prof, na, 0)
    o = len(PROF_AGES) * 2 + 7                      # offset of the prof_max block
    hmax, amax = profr[:, o:o + 1], profr[:, o + 1:o + 2]
    extra = np.concatenate([
        ag / 500.0, np.log1p(ag) / np.log(501.0),
        Y0[:, :1] / (np.abs(Y0[:, 1:2]) + 0.5), Y0[:, 1:2] / (np.abs(Y0[:, :1]) + 0.5),
        Y0[:, 3:4] / (np.abs(Y0[:, :1]) + 0.5), Y0[:, 5:6] / (np.abs(Y0[:, 4:5]) + 0.1),
        Y0[:, 6:7] / (np.abs(Y0[:, 4:5]) + 0.1), Y0[:, 4:5] / (np.abs(Y0[:, 3:4]) + 0.1),
        hmax - Y0[:, :1], amax - Y0[:, 1:2],         # headroom to the site's own ceiling
        Y0[:, :1] / (np.abs(hmax) + 0.5), Y0[:, 1:2] / (np.abs(amax) + 0.5),
    ], 1)
    return np.concatenate([Xc, Y0, extra], 1).astype(np.float32)


def v4_feature_names():
    return (climate_block_names() + STATIC_NAMES + profile_names() + [f"y0_{t}" for t in TREE_BAND]
            + ["age_scaled", "log_age", "h_over_agb", "agb_over_h", "lai_over_h", "cue",
               "rh_over_gpp", "lue", "h_headroom", "agb_headroom", "h_frac", "agb_frac"])


def build_v4_train(x, y, month=12):
    """
    Build the v4 design matrix straight from the notebook's own x/y (see the module-level note
    above), instead of the src/ scripts' precomputed `cache/*.npy` files.

    x: (N_SITE, N_YEAR, N_MONTH, N_FEA) NORMALIZED, e.g. `x_train`/`x_test` from
       `res_train4_test8.npz` after `normalize_x(x_train)`.
    y: (N_SITE, N_AGE, N_YEAR+1, N_MONTH, N_OUT) NORMALIZED, e.g. `y_train`/`y_test` after the
       site-first transpose and `normalize_x(y_train, y_mean, y_std)` already used elsewhere in
       these notebooks.

    Returns:
        X  (N_SITE*15, F) design matrix, row i's seed age is `AGES[i % 15]`.
        D  (N_SITE*15, 7) target Delta = y40 - y0, physical units, all 7 variables (only
           height/agb are scored by the competition; the rest are useful multi-task auxiliary
           supervision, REPORT.md §3.2).
        y0 (N_SITE*15, 7) physical units -- the additive term a delta model's prediction gets
           added back onto: y40_hat = y0 + D_hat.
    """
    y0 = inv_x(y[:, :, 0, month - 1, :], y_mean, y_std).astype(np.float32)
    y40 = inv_x(y[:, :, -1, month - 1, :], y_mean, y_std).astype(np.float32)
    clim, stat = climate_block(x), static_block(x)
    X = assemble_v4(clim, stat, y0)
    n, na = clim.shape[0], len(AGES)
    D = (y40 - y0).reshape(n * na, 7).astype(np.float32)
    return X, D, y0.reshape(n * na, 7).astype(np.float32)


def build_v4_test(test_x, test_y0):
    """
    Build the v4 design matrix for the competition's public test files (test_ssp126.npz /
    test_ssp585.npz), which ship only `test_x` and `test_y0` -- no year-40 ground truth -- and
    where `test_y0` sits on an affine-shifted scale relative to training (LOGBOOK Entry 3):
    `test = a*true + b`. This corrects that shift before building features, matching
    src/dataset.py's `build_test`.

    test_x:  (N_SITE, 40, 12, 136) NORMALIZED the same way as `x_train` -- call
             `normalize_x(test_x)` first.
    test_y0: (N_SITE, 15, 7) PHYSICAL units, as shipped (not z-scored, LOGBOOK Entry 1.2).

    Returns:
        X       (N_SITE*15, F) design matrix.
        y0_true (N_SITE*15, 7) physical units in TRAINING space -- add a model's predicted delta
                onto this, then re-apply the affine map to submit: `test_y0 + AFF_A*delta_hat`
                (REPORT.md §3.2's inference recipe).
    """
    y0_true = ((test_y0 - AFF_B) / AFF_A).astype(np.float32)
    clim, stat = climate_block(test_x), static_block(test_x)
    X = assemble_v4(clim, stat, y0_true)
    n, na = clim.shape[0], len(AGES)
    return X, y0_true.reshape(n * na, 7).astype(np.float32)