# CarbonGlobe — Dataset Notes

Working notes from a first read-through of the tutorial notebook, `data/readme.txt`, and the
actual `.npy`/`.npz` files, plus the dataset's GitHub/HF/paper pages where useful. Written to build
intuition before writing a submission, not as a replacement for the paper.

## 1. What the challenge actually asks

The task emulates the **Ecosystem Demography v3.0 (ED)** model — a physics/theory-based forest
carbon model that's accurate but too slow to run at global scale. CarbonGlobe pre-computed ED's
output for the whole globe so we can train a fast ML "emulator" instead.

> Given a year's 12-month environmental inputs **X** and the ecosystem's state at the end of last
> year **v** (7 carbon numbers), predict this year's state. Repeat for 40 years, feeding each
> year's prediction back in as next year's `v` (**autoregressive rollout**).

The 7 target variables (per site, per year, physical units) are:

| Code | Meaning |
|---|---|
| `height` | Canopy / forest height |
| `agb` | Above-ground biomass (carbon in living vegetation) |
| `soil` | Soil carbon |
| `lai` | Leaf area index |
| `gpp` | Gross primary productivity (total carbon fixed by photosynthesis) |
| `npp` | Net primary productivity (`gpp` − plant respiration) |
| `rh` | Heterotrophic respiration (carbon released by decomposition) |

Because the model must run itself forward from its own (imperfect) predictions, errors compound
over the 40-year horizon — that long-horizon drift is the thing the benchmark is actually testing,
not one-step accuracy.

## 2. The core "cube": site × year × month × variable

Everything hangs off a `(54152, 40, 12, …)` shape:

- **54,152 sites** — land grid cells on a global 0.5° lat/lon grid (`mask2d.npy`, shape `(360,720)`,
  is the lookup table that places each of the 54,152 1-D site indices back onto its `(row, col)` on
  that 2D grid — most of the 360×720=259,200 cells are ocean/non-land and are masked out).
- **40 years** — the forecast horizon of one simulation run.
- **12 months** — monthly time resolution within a year (the tutorial mostly reduces this to the
  December value, or an annual mean, for baseline modelling).
- **136 input features** (`glob_X_fea.npy`) — monthly environmental drivers, from the paper's Table
  2 (the NeurIPS paper, `1982_CarbonGlobe_A_Global_Scal.pdf`, was later added to the folder and
  gives the exact breakdown):

  | Variable | # channels | Unit | Description |
  |---|---|---|---|
  | `ta_m` | 1 | K | Monthly-averaged air temperature |
  | `pr` | 1 | mm | Total monthly precipitation |
  | `tsl1`…`tsl6` | 6 | K | Monthly-averaged soil temp. at 6 depths (0.099 m → 10 m) |
  | `co2` | 24 | ppm | Monthly average of **hourly** CO₂, kept at hourly-of-day resolution |
  | `dst` | 1 | — | Disturbance rate |
  | `hus` | 24 | — | Monthly average of hourly air specific humidity |
  | `ta_h` | 24 | K | Monthly average of hourly air temperature |
  | `rsds` | 24 | W/m² | Monthly average of hourly downward shortwave radiation |
  | `sfcWind` | 24 | m/s | Monthly average of hourly wind speed |
  | `k_sat` | 1 | mm/yr | Saturated hydraulic conductivity (soil) |
  | `s_theta`, `r_theta` | 2 | m³/m³ | Saturated / residual water content (soil, MVG model) |
  | `L`, `n`, `m` | 3 | — | MVG soil hydraulic shape parameters |
  | `sd` | 1 | mm | Soil depth to bedrock |

  `1+1+6+24+1+24+24+24+24+1+2+3+1 = 136`. The "24" channels are variables that started as *hourly*
  values and were monthly-averaged **separately for each hour-of-day** (i.e. they keep a diurnal
  cycle: hour 0's monthly mean, hour 1's monthly mean, …, hour 23's), rather than being collapsed
  to one monthly number — this is why humidity/temperature/radiation/wind each occupy 24 of the 136
  slots instead of 1. Sources: meteorology from **MERRA-2**, CO₂ from **NOAA CarbonTracker**, soil
  properties from **ROSETTA**. The 7 target variables are *also* fed back in as input, but only as
  the single initial-condition vector `v` at the start of a forecast step (not part of the 136 —
  they're what the RF baseline concatenates separately to make its 144-dim feature vector).
- **7 output targets** (`glob_Y{age}.npy`, see below) — the ED-simulated carbon variables above,
  at monthly resolution, for 41 years (year 0 = the initial condition, years 1–40 = the forecast).

## 3. What "age" means here — seed tree age, not observation age

This is the part that's easy to misread. **`age` is not a real-world measured stand age** — it's a
**simulation initial condition**. ED needs to know how developed the forest canopy already is
before it can grow it forward, but the *true* age structure of a real forest patch generally isn't
known globally. So instead of guessing one age per site, the dataset runs ED **15 separate times
per site**, each time telling the model "pretend this stand started as a `age`-year-old forest,"
for `age ∈ {1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500}`. Each run then
produces its own independent 40-year trajectory (`glob_Y001.npy` … `glob_Y500.npy`, each
`(54152, 41, 12, 7)`).

So the full dataset is really `54,152 sites × 15 seed ages` = **812,280 independent 40-year
sequences** — a "site" alone doesn't fully specify a training example; you need (site, seed age)
together. The seed age is fed to the RF/DL models as an extra input feature (`age / 500`, scaled to
`[0, 1]`) so the model learns how the forecast should differ depending on how mature the initial
stand is.

## 4. The "triplet" file — `data_stats/age_triplet.npz`

This file is a **precomputed per-age reference/prior**, not raw simulation data. It has three
arrays (hence "triplet"):

| Key | Shape | What it is |
|---|---|---|
| `age` | `(15,)` | the same 15 seed ages listed above |
| `age_mean` | `(15, 7)` | mean value of each of the 7 targets, averaged across all sites/years, for that seed age |
| `age_slope` | `(15, 7)` | the average year-over-year growth rate of each target, for that seed age |

Pulling the actual numbers out of the file shows exactly what you'd expect ecologically:

- `age_mean` for `height` rises quickly from 1‑year-old stands (~6.2 m) to ~11.7–12 m by age 50–500
  and then **plateaus** — canopy height saturates once a stand matures.
- `age_mean` for `agb` (biomass) and `soil` (soil carbon) **keep climbing** all the way to age 500
  (never fully plateauing in this range) — carbon keeps accumulating in wood and soil long after
  height stops changing.
- `age_slope` (the growth rate) **drops sharply as seed age increases** — e.g. `height`'s slope
  falls from ~0.195 m/yr at age 1 to ~0.0003 m/yr by age 500. This is the standard forest-ecology
  pattern: young stands grow fast, old stands are near steady-state.

Practically, this file is a cheap **age-conditioned baseline/prior** — e.g. "if I know nothing else
about a site except its seed age, `age_mean`/`age_slope` is a reasonable guess" — useful as a sanity
check baseline or as an extra engineered feature, rather than something you need to consume in the
core pipeline. The tutorial notebook loads it but never actually uses it in the RF baseline.

## 5. In-situ validation & `lat_lon_age_weights.csv` — mixing the ages back together

`data/insitu_data/` holds real flux-tower-style GPP observations from the **ABoVE** field campaign
at ~20 real-world lat/lon locations, in files like
`lat_48.2167_lon_-82.1556_GPP_comparison.csv` (columns: `latitude, longitude, start_date,
in-situ gpp, ed_gpp`).

The catch: a real forest patch is **not** a pure single-age stand — it's a natural mixture of trees
of many ages. So to compare ED's simulated output against a real observation, you can't just pick
one of the 15 seed-age runs. Instead, `lat_lon_age_weights.csv` gives, **per real site**, a
15-column weight vector (`weight_1` … `weight_15`, one per seed age) that **sums to exactly 1.0**
(verified: every row sums to 1.0). The `ed_gpp` column in each comparison CSV is the weighted sum
of that site's 15 seed-age trajectories using these weights — i.e. `ed_gpp = Σ_a weight_a *
glob_Y{a}[site]`. This reconstructs a "realistic mixed-age" forest carbon estimate that's directly
comparable to the real-world `in-situ gpp` measurement. This is a good pattern to reuse if a
submission wants to validate against real-world data rather than only the held-out ED simulation
sites.

## 6. Two ways to consume the data

The tutorial deliberately offers two entry points:

1. **`data_global/res_train4_test8.npz`** — a pre-bundled, ML-ready `x_train/y_train/x_test/y_test`
   set (train = 3035 train + 338 val sites = 3373; test = a fixed **852-site held-out subset**
   called "test8"). This is what the deep-learning baselines (DeepED, Transformer, Informer, …)
   train and report their headline numbers on. Confirmed shapes: `x_train (3373,40,12,136)`,
   `y_train (15,3373,41,12,7)`, `x_test (852,40,12,136)`, `y_test (15,852,41,12,7)` — note `y_*`
   keeps all 15 seed ages, so this file is really `(sites, ages)`-indexed once you unpack the `y`
   dimension.
2. **The raw global arrays** (`glob_X_fea.npy`, `glob_Y{age}.npy`, all `54,152` sites) sliced by
   `train.csv` (3035 idx) / `val.csv` (338 idx) / `test.csv` (50,779 idx) — these three CSVs
   **partition** all 54,152 sites with no overlap. Use this route to score a model across the
   *entire* planet or make global maps; `test8` is a fixed subset of `test.csv`'s 50,779 sites, so
   both routes share the same held-out sites and neither ever leaks a test site into training.

The split looks close to a **uniform ~1/16 spatial subsample** for train+val (per the notebook's
map plot), i.e. training sites are thinly and evenly scattered across the globe rather than
clustered in one region. The paper confirms this is deliberate: they train on only 1/16 of sites
because a *point* of building an ML emulator is to run it at higher spatial resolution than you
trained on — training on a coarse, uniformly-sampled subset and then predicting at the full-density
grid is "approximately the same as using a 4×-coarser grid for training," and uniform (not
clustered) sampling avoids the model spatially overfitting to one region.

**Both `y_train` and `y_test` in `res_train4_test8.npz` carry full ground truth** — this is not a
Kaggle-style withheld-label test set as shipped in the tutorial data. See §11 below on what that
means for the actual competition submission mechanics, which we couldn't confirm locally.

### The three evaluation scenarios (from the paper)

Beyond the plain train/test split, the paper defines three ways of slicing test performance to
answer different questions:

1. **Overall evaluation** — the standard "all test sites pooled together" numbers (the RF baseline
   notebook does this).
2. **Climate-based evaluation** — test sites are grouped by **Köppen–Geiger climate zone**
   (tropical, arid, temperate, cold, polar) and scored separately, to see if a model is
   systematically worse in, e.g., tropical vs. cold forests.
3. **Forest-age-based evaluation** — test performance is broken out **by seed tree age**, since
   young vs. old stands have very different growth dynamics (§3/§4 above).

These aren't in the tutorial notebook (which only does #1), but they're worth reproducing for a
submission write-up since the paper's own benchmark table is built around exactly these three
views.

## 7. Normalization — `data_stats/data_stats.npz`

Plain per-channel z-score statistics: `x_mean, x_std` (136,) for inputs, `y_mean, y_std` (7,) for
targets. All models train in normalized space (`z = (x - mean) / (std + 1e-10)`) and metrics get
converted back to physical units for reporting (`x * std + mean`).

## 8. How the tutorial's Random Forest baseline is wired together

- **Training** builds a "one-step-ahead" supervised problem: for every (train site × seed age ×
  year), the feature vector is `[136 annual-mean inputs, 7 previous-year Dec outputs (v), 1 scaled
  age]` = 144 features, and the label is that year's 7 December outputs. This uses **teacher
  forcing** — the true previous year's output is given, not the model's own prior guess. On the
  full train set that's `3035 sites × 15 ages × 40 years` ≈ **1.82M rows**.
- **Testing** is autoregressive: only the true year-0 state is given; every subsequent year's `v`
  is the *model's own* prediction fed back in, exactly like the deep baselines are evaluated, so
  errors can accumulate.
- **Metrics** (physical units): RMSE, MAE, R² computed step-wise across all years; plus two
  "problem-driven" metrics from the paper — **cumulative error** `E_C = ||Y_T − Ŷ_T||²` (squared
  error only at the final step `T` of the sequence — pure long-horizon drift) and **delta error**
  `E_Δ = (1/(T−1)) Σ_{t=2..T} ||(Y_t − Y_{t-1}) − (Ŷ_t − Ŷ_{t-1})||²` (mean squared error of the
  year-over-year *change*, i.e. how well the model captures inter-annual variation independent of
  its absolute level). The tutorial takes square roots of both (reports them as RMSE-like), the
  paper's raw formulas are squared-error as written above — worth double-checking which convention
  a submission is scored against.
- The paper also reports a **training trick** worth knowing about: because training uses
  teacher-forcing (true previous-year `v`) while testing is autoregressive (predicted `v`), models
  trained only on true data accumulate error faster at test time. Their fix — injecting small
  Gaussian **random-walk noise** (σ≈1e-4) into the training-time `v` so the model sees slightly
  imperfect inputs during training too — measurably reduced cumulative error (up from being worse
  without it in the paper's own ablation, Table 3 vs Table 1). This is a candidate technique worth
  trying if a submission's autoregressive rollout drifts badly.

## 9. Two data tiers, and how strict the "test" set actually is

**Why two tiers exist.** The full global arrays are large (`glob_X_fea.npy` alone is ~14 GB, plus a
separate ~14 GB-scale file per seed age for `glob_Y*.npy` × 15), so they're memory-mapped and sliced
by site index rather than loaded whole. `res_train4_test8.npz` exists purely as a **convenience
bundle**: it pre-extracts exactly the rows needed for standard model training/benchmarking (all
train+val sites, plus a fixed "test8" subset) into ready-to-load `x_train/y_train/x_test/y_test`
arrays, matching what the paper's own DeepED/Transformer/etc. benchmarks (Table 1 in the PDF) train
and report on. The global-grid tier (`glob_X_fea.npy` + `glob_Y{age}.npy` + the three CSVs) is the
same underlying data at full resolution — needed if you want to score a model on *all* 50,779
`test.csv` sites (not just the 852-site test8 subset), make a global error map, or run the paper's
climate-zone/forest-age breakdown scenarios, which need more than 852 sites to be meaningful per
climate zone. `test8` (852 sites) is a fixed, non-random subset of `test.csv`'s 50,779 sites — same
split, smaller sample for cheaper repeated benchmarking runs. (The "4"/"8" in the filename don't
obviously map to anything in the array shapes — left unexplained rather than guessed.)

**Is the test set blind?** Based on everything available locally: **no, it is not blind.** Both
`x_test` **and `y_test`** are shipped in `res_train4_test8.npz` with real values (not placeholders),
and the global `glob_Y{age}.npy` files contain true outputs for every one of the 54,152 sites,
`test.csv`'s included — there's no separate "hidden labels" mechanism in the data files themselves.
This is a **fixed, published research-benchmark split** (defined by the paper's "Overall
evaluation" scenario: uniformly sample 1/16 of sites for training, the rest for testing), not a
Kaggle-style train-on-visible/submit-for-hidden-scoring split. The tutorial notebook computes its
own RMSE/MAE/R²/delta/cumulative metrics directly against `y_test`, locally, with no submission
step.

**However** — this is being run as a Kaggle competition (IEEE BigData Cup 2026) layered on top of
this research dataset, and competitions conventionally hide the true test labels and score
submissions against a leaderboard instead, sometimes using a further-held-out split the public data
doesn't contain at all. **I could not confirm which of these applies here** — the saved Kaggle page
(`IEEE Bigdata Cup 2026… .html`) is only the React/SPA shell (Kaggle renders Overview/Data/Rules
content client-side via authenticated API calls), so it carries no competition text beyond the
`<title>`/meta description. **This needs to be checked on the live, logged-in Kaggle page's
Data/Rules/Evaluation tabs** before assuming the local `y_test` is the same thing the leaderboard
scores against — it's entirely possible the competition asks for predictions on a site set that
doesn't even appear in these local files, or scores only a subset/metric the local files don't
distinguish.

## 10. Suggested next steps

- **Check the live Kaggle competition page** (Data / Rules / Evaluation tabs, logged in) to confirm
  how submissions are actually scored — whether it's against the local `y_test`/`test.csv` labels
  as-is, or a separate hidden set, and what exact metric/weighting determines the leaderboard.
- Consider reproducing the paper's **climate-zone** and **forest-age** evaluation breakdowns (§9 in
  this doc), not just the pooled "overall" metric the tutorial computes — the paper shows model
  rankings can flip between scenarios (e.g. DeepED wins on height/AGB/soil/LAI, Transformer/Informer
  win on GPP/NPP/Rh), which is useful context for choosing or ensembling a modeling approach.
- Consider whether `age_triplet.npz`'s `age_mean`/`age_slope` are worth using as an extra
  engineered feature or a naive baseline to beat, since they're currently loaded but unused in the
  tutorial's RF baseline.
- Consider the paper's **noise-injection trick** (small random-walk noise added to training-time `v`)
  as a way to reduce autoregressive drift, since it measurably helped all 8 of the paper's benchmark
  models.

## 11. Training convention: teacher forcing vs. full autoregressive rollout

`notebooks/exploring.ipynb`'s custom `SimpleNN` diverges from the tutorial/paper's training
convention in a way worth calling out explicitly, since it changes what's actually being optimized:

- **Tutorial/paper convention (teacher forcing):** during training, the model is given the *true*
  previous year's output as `v` at every step — each of the 40 years is an independent, well-posed
  one-step regression problem. Only at *test* time does the model have to consume its own predicted
  `v` and roll out autoregressively for 40 years.
- **`exploring.ipynb`'s convention (full autoregressive / backprop-through-time):** the same
  `recursive_predict` function is used for *both* training and testing — training also starts from
  only `y0` and feeds the model's own (initially bad) predictions forward for all 40 years, with
  gradients backpropagated through the entire unrolled sequence.

Neither is "wrong," but they trade off different failure modes. Teacher forcing is easy to
optimize (every step sees a realistic, true-valued input) but creates **exposure bias**: the model
never practices recovering from its own mistakes during training, yet that's exactly the condition
it faces at test time, so errors that would never occur during training can compound unchecked at
test time. Full autoregressive training closes that gap by construction (train and test conditions
match exactly), but early in training — when the model is still bad — every step after the first
is conditioned on a already-wrong `v`, so the gradient signal for years 2-40 is contaminated by the
model's own compounding mistakes rather than reflecting the true input-output relationship; it's
also more expensive (the computation graph spans all 40 steps) and more prone to vanishing/exploding
gradients over that many steps. The CarbonGlobe paper's own middle-ground fix — injecting small
random-walk noise into the *true* training-time `v` — is a lighter-weight way of exposing the model
to slightly-imperfect inputs without going fully autoregressive.

**Proposed teacher/student setup to bridge the gap** (per discussion), using the standard
distillation convention — **teacher** = the stronger, privileged-information model that supervises;
**student** = the model that must learn to perform without that privilege, guided by the teacher:

- **Teacher** = a model trained the tutorial's way, with teacher forcing — given the true previous
  year's `v` at every step (this is *literally* what "teacher forcing" already names: the teacher
  supplies ground truth). Solving a much easier, well-posed one-step regression problem, it
  converges faster and more stably, and ends up more accurate than a model forced to bootstrap off
  its own early, bad predictions.
- **Student** = the "production" model, trained the way `exploring.ipynb` currently does: fully
  recursively, rolled out from `y0` alone, exactly matching the real test-time/deployment
  conditions (no access to true `v` beyond year 0).

This matches standard usage elsewhere in the sequence-modeling literature (e.g. "Professor Forcing,"
Lamb et al. 2016, which pushes a free-running network's behavior toward a teacher-forced network's).
Because the teacher converges quickly and stably on the true one-step mapping, it can be used to
**stabilize the student's harder training problem** rather than leaving the student to learn purely
from its own raw, compounding-error rollout gradient. Two concrete ways to wire this together:

1. **Warm start:** train the teacher to convergence first, then initialize the student's weights
   from the teacher's before switching to full autoregressive fine-tuning — avoids trying to solve
   the hard 40-step BPTT problem from a random initialization.
2. **Auxiliary loss:** while unrolling the student during training, at each step also compute what
   the teacher would have predicted from the *true* history, and add a loss term pulling the
   student's per-step prediction toward the teacher's — giving the student extra, stable gradient
   signal on top of its own autoregressive rollout loss, similar in spirit to scheduled sampling /
   knowledge distillation for sequence models.

Worth trying if the current fully-recursive training turns out to converge slower or less stably
than the tutorial's teacher-forced RF/DL baselines.

**Implemented as `TeacherStudentRollout`** in `notebooks/helpers.py` (see §12) — a model-agnostic
`nn.Module` wrapper taking an externally-defined "core" model (any architecture matching
`MonthMetaModel`'s `forward` signature) and adding `teacher_rollout`/`student_rollout`/
`compute_losses` (supervised + distillation loss) plus `update_teacher`, an exponential-moving-average
update (the "mean teacher" pattern, Tarvainen & Valpola 2017) that lets a frozen, pretrained teacher
keep drifting slowly toward the student's weights during student training, rather than staying a
fixed snapshot — the goal being a teacher (and thus a distillation target) that keeps improving
alongside the student instead of capping it.

## 12. Shared code — `notebooks/helpers.py`

Constants, normalization helpers, and model-training utilities that were originally built up inline
in `recursive_models.ipynb` are being extracted into `notebooks/helpers.py` so new model notebooks
can `from helpers import ...` instead of re-pasting/redefining them as more architectures are added.
**Future agents building a new model notebook should add shared, reusable pieces here rather than
duplicating them inline again.**

Current contents:

- **Constants**: `DATA_DIR`/`GLOB_DIR`/`STAT_DIR`, `TREE_BAND`, `AGES`,
  `N_SITE`/`N_YEAR`/`N_MONTH`/`N_FEA`/`N_OUT` (see §2–§3 above for what these mean).
- **Arrays/stats loaded at import time**: `mask2d`, the age-triplet prior
  (`TRI_AGE`/`TRI_MEAN`/`TRI_SLOPE`, see §4), and `data_stats.npz`'s normalization stats
  (`x_mean`/`x_std`/`y_mean`/`y_std`, see §7). **Importing this module has side effects** — it reads
  several files off disk and prints a few lines — the moment it's imported, not lazily.
- **`normalize_x` / `inv_x`** — the z-score normalize/denormalize helpers from §7.
- **`TeacherStudentRollout(nn.Module)`** — the teacher/student rollout wrapper described in §11.

**Caveats for whoever wires this in next** (true as of this writing — check before trusting):

- The file is **missing its own imports** (`numpy as np`, `torch`, `from torch import nn`) — it
  currently only runs if the importing notebook has already imported those names into scope first,
  which is fragile and import-order-dependent. Add explicit imports at the top of `helpers.py`
  before relying on it from a notebook that doesn't happen to import numpy/torch first.
- `DATA_DIR = "../data"` is a **relative path**, only correct when the importing code's current
  working directory is `notebooks/` (as the existing notebooks' cwd is). A script run from elsewhere
  (e.g. a top-level training script, or a notebook in a different folder) will fail to find the data
  unless this is made relative to the module's own location or otherwise made robust.
- **Neither `recursive_models.ipynb` nor `exploring.ipynb` has actually been switched over to import
  from `helpers.py` yet** — both notebooks still carry their own inline, duplicate copies of these
  constants/functions/`TeacherStudentRollout`. Until that migration happens, the copies **can drift
  out of sync**: check whether you're editing the canonical version in `helpers.py` or a stale inline
  duplicate in a notebook before changing shared logic like the constants, `normalize_x`/`inv_x`, or
  `TeacherStudentRollout`.

## 13. Feature engineering idea — borrowing from financial time-series ML

Per discussion: this dataset has a similar shape to what quant/financial ML deals with (a long
univariate-per-channel time series per entity, where you need to predict a future value from noisy,
autocorrelated drivers), so techniques standard in that field are worth trying here even though the
domain is completely different:

- **Rolling statistics of the monthly climate drivers** — rolling mean/std/min/max of `ta_m`, `pr`,
  etc. over trailing windows (e.g. 3, 6, 12, 24 months) as extra input features, not just the current
  annual mean the RF/`MonthMetaModel` baselines currently use (`prep_year`'s `x[:, year, :, :]`,
  §8/DATASET_NOTES `prep_year`). A rolling mean smooths noisy single-year weather into a "climate
  trend" signal; a rolling std captures volatility/variability the model might need to distinguish a
  genuinely-shifting climate from a single unusual year — analogous to how price momentum/volatility
  indicators (moving averages, Bollinger-band-style rolling std) are standard engineered features in
  financial forecasting.
- **Momentum / lag features on the target trajectory itself** — beyond just feeding the single
  previous year's `v` (the current teacher-forcing/autoregressive convention, §8/§11), also feed the
  *change* over the last N years (`v_t - v_{t-N}`) or a short rolling average of recent `v` values.
  This mirrors financial ML's use of lagged returns/moving averages of the target series as features
  for the next prediction, and could help the model separate "currently growing fast" stands from
  "currently plateaued" stands more directly than a single previous value does (the age-conditioned
  `age_slope` in `age_triplet.npz`, §4, is effectively a coarse, age-bucketed version of this idea
  already present in the data).
- **Caveat carried over from finance**: rolling/lag features computed naively can leak future
  information if not shifted correctly, and adding many correlated rolling windows can blow up the
  feature count for comparatively little new signal (the 136 input features already include several
  variables at 24 sub-channels each, e.g. `hus`/`ta_h`/`rsds`/`sfcWind`, so there's a real risk of
  redundant, highly-correlated columns). Worth validating any new rolling feature actually improves
  held-out global-test metrics (§9's `metrics_table_global`) before keeping it, rather than assuming
  more engineered features are automatically better.
