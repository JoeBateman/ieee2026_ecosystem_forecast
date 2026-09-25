# Logbook — IEEE BigData Cup 2026 / CarbonGlobe Ecosystem Forecast

Running log of what was tried, what was measured, and what was decided. Newest entries at the bottom.
Companion documents: `agent/DATASET_NOTES.md` (dataset anatomy, written by an earlier session) and
`agent/REPORT.md` (the final write-up).

---

## 2026-09-19 — Entry 0: environment survey

**Compute.** `E:\miniconda3\envs\torch\python.exe` — torch 2.13.0+rocm10.0.0, AMD Radeon RX 9070,
17.1 GB VRAM, CUDA-API available through ROCm. numpy 2.5.2, pandas 3.0.5, sklearn 1.9.1,
xgboost 3.4.1. **lightgbm is not installed.** Plain `python` is not on PATH (Windows Store stub);
every script must be run with the explicit interpreter path.

**Data on disk.**
- `data/data_global/glob_X_fea.npy` — 14.1 GB, the global input cube.
- `data/data_global/glob_Y{age}.npy` — 15 files, 746 MB each, one per seed age.
- `data/data_global/res_train4_test8.npz` — 3.95 GB convenience bundle.
- `data/public/test_ssp126.npz`, `test_ssp585.npz` — 1.61 GB each, the competition inputs.
- `models/rf_baseline.joblib` — 162 MB, a Random Forest from an earlier session.
- `submissions/meta_monthly_student_submission.csv` — an earlier session's submission.

**Prior work.** Two notebooks (`exploring.ipynb`, `recursive_models.ipynb`) and `notebooks/helpers.py`.
Git history shows an RF baseline, a `SimpleNN` trained fully autoregressively, and a
teacher/student rollout wrapper. No recorded leaderboard score for any of them.

**The one thing that reframes the task.** `data/public/README.md` asks only for the **year-40**
values of **height** and **agb**, scored by MSE on mean-scaled values. The 7-variable, 40-step,
12-month autoregressive rollout that the research benchmark is built around is *a* way to produce
that number, but it is not what is being scored. Two of seven variables, one of forty timesteps.
That mismatch is the first thing worth exploiting — the prior work optimised the research metric.

**Immediate suspicion about the existing submission.** Its first rows read
`ssp126_0_1,0.1356,0.0389`. Unscaling gives height = 0.1356 x 10.90645 = **1.48 m** and
agb = 0.0389 x 3.39567 = **0.13**. The age-triplet prior in the dataset says mean height sits at
~6.2 m even for 1-year-old seed stands and ~12 m for mature ones. A 1.5 m year-40 canopy is not a
plausible forecast; this looks like a rollout that collapsed toward zero, or a double-scaling bug.
Flagged to check against the real distribution before reusing anything from that pipeline.

**Plan for entry 1.** Establish ground truth about the arrays before modelling: confirm shapes,
find where the SSP climate inputs sit relative to the training climate (the extrapolation risk),
and check whether `test_y0` looks like the training year-0 state.

---

## Entry 1: what the arrays actually contain

Six exploration scripts (`src/explore01..06.py`). Findings, in order of how much they change the plan.

### 1.1 The submission scalers are just `y_mean`

`data_stats.npz`'s `y_mean` is `[10.90645, 3.39567, 6.24214, 1.88269, 1.081, 0.52339, 0.489]`. The
README's "divide height by 10.90645, agb by 3.39567" is exactly `y_mean[0]` and `y_mean[1]`.
So the scored quantity is `pred_physical / y_mean`, and the metric is the mean of squared
*relative* errors. Nothing exotic.

### 1.2 `test_y0` is in physical units (checked, not assumed)

The sign of a few entries (`height` min −0.90, `agb` min −0.40) initially suggested z-scored data.
It is not. Comparing per-age means against `glob_Y{age}[:, 0]` settles it:

| variable | test_y0 / glob_Y year-0, ratio by age |
|---|---|
| lai | 1.08, 0.96, 0.93, … 1.00 |
| gpp | 2.07, 1.13, 1.10, … 1.09 |
| npp | 2.07, 1.13, 1.10, … 1.10 |
| rh | 3.27, 1.13, 1.07, … 1.00 |
| height | 0.35, 0.68, 0.72, … **0.81** |
| agb | —, 0.24, 0.49, … **0.71** |
| soil | —, 0.11, 0.53, … **0.79** |

The flux variables and LAI line up with the training distribution to within ~10%; if `test_y0` were
z-scored none of them would. So it is physical — with a small amount of noise that pushes a few
values slightly negative.

**But the three stock variables are systematically low** — height ×0.81, agb ×0.71, soil ×0.79 —
while fluxes match. A different *sample* of sites would move all seven together. Only the stocks
moving suggests either a different spin-up in the SSP runs, or a test-site sample biased toward
low-biomass land. This matters: it means I cannot trust a global prior fitted on training stock
levels, and it is worth resolving directly (see §1.5).

### 1.3 The feature layout in DATASET_NOTES is wrong — channels are interleaved

DATASET_NOTES §2 describes the 136 channels as contiguous blocks (`co2` 24, then `hus` 24, then
`ta_h` 24, …). Per-channel statistics say otherwise. `ta_h` under that reading would have min 0 and
max 1143 — not a temperature. The actual layout is:

```
ch 0        ta_m      air temperature (K)
ch 1        pr        precipitation (mm)
ch 2:8      tsl1..6   soil temperature, 6 depths (K)
ch 8:32     co2       24 hours-of-day (ppm)
ch 32       dst       disturbance
ch 33:129   24 repeats of (hus, ta_h, rsds, sfcWind)   <-- INTERLEAVED, stride 4
ch 129:136  k_sat, s_theta, r_theta, L, n, m, sd       static soil properties
```

Verified by the diurnal cycle: channels `34+4k` trace a smooth temperature curve (282.6 K at hour 0,
peaking 284.5 K at hour 13); channels `35+4k` trace a solar curve (105 → 277 at hour ~8 → 35 at
hour 22, floored at exactly 0 overnight). Contiguous blocks produce no such structure. Any
feature engineering that slices these channels by the old layout would have been silently scrambling
humidity, temperature, radiation and wind together.

### 1.4 Two things that constrain the model choice

**`dst` (channel 32) is a dead feature** — exactly 0.012 at every site, year, and month, in both the
training data and both scenarios. Zero variance, drop it.

**CO₂ in the test scenarios is *inside* the training range, and much flatter.** Training CO₂ runs
357 → 419 ppm over the 40 years (the historical 1980–2020 record), spanning 332–524 across all
sites/months/hours. `ssp126` runs 363.8 → 371.2 (**+7 ppm over 40 years**) and `ssp585` runs
363.9 → 393.1 (+29 ppm), spanning only 363–373 and 362–396. That is far milder than real SSP
forcing, and it is good news — the scenarios interpolate within training support rather than
extrapolating beyond it.

It also raises a hazard that shapes the whole design. CO₂ is very nearly spatially uniform in the
training data, so across training sites it carries **almost no cross-sectional variance** — it is
effectively a single global time series perfectly collinear with the year index. A model cannot
separate "CO₂ fertilisation" from "time trend" from that. At test time the CO₂ trajectory is
different (flatter and ~16 ppm lower), so whatever the model attached to those channels transfers
wrongly. **Action: test CO₂-in vs CO₂-out empirically rather than assuming.**

### 1.5 Next: can the SSP sites be mapped back onto the global grid?

`sites_ssp.csv` is just row indices 0…6170, so the scenarios ship no coordinates. But channels
129–135 are static soil properties, constant in time per site. If those tuples are near-unique they
act as a fingerprint that identifies which global grid cell each SSP site is. That would let me
resolve §1.2's stock-variable puzzle directly, and open the door to site-matched features.

---

## Entry 2: site matching abandoned; the target structure reframes the task

### 2.1 The "noise in the test features" scare was a float32 artifact

`src/noise_probe.py` initially appeared to show the time-invariant soil channels varying in time in
the test set (std up to 1.39), which would have implied deliberately noised test inputs. It does not.
`k_sat` has values of order 1e5–1e6, where float32 spacing is ~0.06–0.5; the apparent "variance" is
rounding. Relative noise is 0.0000 for every static channel, and the *training* data shows the same
artifact (time-std 5.69 on `k_sat`). Static channels are clean, and are **byte-identical between the
two scenarios** (`ssp126` and `ssp585` share the same 6171 sites). `test_y0`, by contrast, differs
slightly between scenarios (max abs diff 0.34), so each scenario carries its own initial state.

### 2.2 SSP sites cannot be mapped back to the global grid — dropped

`sites_ssp.csv` is just row indices 0…6170, so the scenarios ship no coordinates. I tried
fingerprinting sites by their 7 static soil properties. The global fingerprints are unique (54152/54152
distinct), but the test sites do not match them: **zero exact matches**, and the nearest-neighbour
distance ratio `d1/d2` has median **0.843** with only 0.5% of sites below 0.1. Nearest and
second-nearest are essentially equidistant, so the assignment is not identifiable — these are not the
same grid cells, or the soil layer was re-derived. Pursuing it would inject wrong labels into
features, so I dropped it. Climate features already encode geography well enough.

### 2.3 The target month axis is pure replication

`glob_Y{age}` has shape `(54152, 41, 12, 7)`, but checking 1500 random sites × 15 ages × 41 years
gives `max |Y[:,:,m,:] − Y[:,:,0,:]| = 0.000e+00` — **exactly zero**. Every month within a year holds
the identical value. The targets are annual and always were; 11/12 of the ~11 GB of target data is
duplication. This also dissolves a question the submission format raised: since `test_y0` has no
month axis, one had to guess whether "annual" meant December or the mean. It means neither and both —
they are the same number. Targets cached as `cache/Y_glob.npy`, `(15, 54152, 41, 7)`, 932 MB.

### 2.4 The finding that reframes the whole task

`src/target_analysis.py`, 8000 random sites, scored in the **competition metric** (mean squared
error on `value / y_mean`, so directly comparable to the leaderboard):

| predictor | score | height | agb |
|---|---|---|---|
| all zeros | 2.78402 | 2.18006 | 3.38798 |
| global mean | 1.68412 | 1.02029 | 2.34796 |
| age-conditional mean | 1.61426 | 1.01700 | 2.21152 |
| **persistence (ŷ40 = y0)** | **0.29690** | 0.25384 | 0.33996 |
| **per-age linear fit on y0 alone** | **0.16533** | 0.13314 | 0.19751 |

Persistence beats the age-conditional mean by 5.4x. A per-age straight line through `y0` — no
climate, no soil, two coefficients per age per variable — beats it by 9.8x. Supporting structure:

- **Variance is spatial, not developmental.** Of year-40 height variance, 91.1% is across sites and
  0.3% across seed ages (agb: 85.6% / 5.8%).
- **corr(y0, y40)** is 0.91–0.97 for height and 0.95–0.98 for agb at every seed age except age 1
  (agb 0.758 — young stands are the exception, they genuinely grow).
- **Mean trajectories are nearly flat.** Age 100 height drifts 11.78 → 11.90 m across the full 40
  years (+1%); agb goes 3.12 → 2.97 (−5%), dipping mid-record and recovering.

**Consequence for the design: model the residual `Δ = y40 − y0`, not the level.** Predicting `y40`
directly spends most of the model's capacity relearning an identity map that a 2-parameter line
already captures, while the actual skill lives in a small correction term. This is the single
biggest lever identified so far, and it is invisible if you approach the problem as the research
benchmark's 7-variable 40-step rollout.

The corollary is where the remaining difficulty sits: **young seed ages**. Age 1 and 10 stands grow
substantially over 40 years and have the weakest `y0` correlation, so they carry disproportionate
skill-headroom, while ages 100–500 are near steady-state.

---

## Entry 3: the test targets live on a shifted scale (and why it mostly cancels)

### 3.1 The evidence

`test_y0` cannot be compared to training `y0` at face value. Three independent observations:

**Floors.** Training `y0` has an exact floor at **0.0** for height, agb, soil, lai and rh — barren
land where nothing grows — hit by 8.9–10.3% of (site, age) rows. The test set has an exact floor
too, at **−0.9009 / −0.4014 / −0.5324 / +0.1179 / +0.0847**, hit by 9.5–10.2% of rows. The floor
value is **identical at every seed age from 10 to 500** (67,169 training rows sit at exactly 0.0;
8,195 test rows sit at exactly −0.9009).

That rules out the innocent explanations. Additive noise would give each age a different minimum.
A different sample of sites cannot manufacture a negative floor at all — a barren cell has zero
biomass, not −0.4 of it. A deterministic floor shared across ages is a deterministic transform.

**A second landmark.** Training age-1 height has a secondary point mass at exactly 1.5002 m (862
rows) — ED's seeded stand height. The test set has the matching cluster at ≈0.4138.

**Quantile matching.** Fitting `test = a·train + b` on 199 pooled quantiles gives R² of
0.9983–0.9999 for all seven variables.

So `test_y0 = a·true + b` per variable, with `b` read directly off the floor (true 0 → b).

### 3.2 What is and isn't identifiable

`b` is rigorous: it is the image of a physical zero and is immune to how the test sites were sampled.

`a` is not cleanly identifiable. The two estimators disagree for height — the landmark pair
(0 → −0.9009, 1.5 → 0.4138) gives **a = 0.877**, while the full-range quantile fit gives
**a = 0.921**. The landmark pair has a very short lever arm (both points near the bottom of a
0–30 m range), so it is noisy; the quantile fit uses the whole range but absorbs any genuine
difference in the site sample. For agb the two agree well (quantile fit a = 0.840 reproduces every
test quantile from −0.40 to 19.25 to within 0.13).

### 3.3 Why this is survivable — and another argument for modelling Δ

Under an affine map the intercept **cancels out of the difference**:

```
Δ_test = test_y40 − test_y0 = a·(true_y40 − true_y0) = a·Δ_true
```

So the recipe is: map the test initial state back with `true_y0 = (test_y0 − b)/a`, predict
`Δ_true`, and submit `test_y0 + a·Δ_true`. The level term passes through untouched, which is where
the overwhelming majority of the signal lives (Entry 2.4).

And the residual uncertainty in `a` turns out not to matter. `a` multiplies only the correction
term. With `Δ/y_mean` having RMS ≈ 0.5, a 5% error in `a` contributes ≈ (0.05 × 0.5)² = 0.000625 to
the score — negligible beside a plausible model score of ~0.05. **`b`, which shifts the model's
input, is the part that had to be right, and `b` is the part that is rigorously determined.**

Adopted constants (`b` from floors; `a` refit with `b` held fixed):

| var | a | b |
|---|---|---|
| height | ~0.92 | −0.9009 |
| agb | ~0.84 | −0.4014 |
| soil | ~0.90 | −0.5324 |
| lai | ~1.01 | +0.1179 |
| rh | ~0.93 | +0.0847 |
| gpp, npp | quantile fit | quantile fit (no exact floor) |

### 3.4 Feature-space shift, and the CO₂ decision

`src/shift_check.py`, comparing the 40-year mean of each engineered feature against the training
0.5–99.5 percentile range:

- **`co2_mean`: 100.00% of test sites out of range, in both scenarios.** Training site means span
  377.9–391.4 ppm; `ssp126` sits at 368.6 and `ssp585` at 376.2. Combined with the fact that CO₂ has
  almost no cross-site variance in training (so no causal response can be learned from it, only its
  collinearity with the year index), this makes the CO₂ channels actively harmful:
  a model that reads CO₂ ≈ 368 would infer "year 9" and hold the stand there for the whole rollout.
  **Decision: drop CO₂ from the feature set.** Scenario discrimination still works, because `ssp126`
  and `ssp585` differ in temperature (+0.28 K), radiation and VPD (+19 Pa). To be verified by ablation.
- Moderate shift in the radiation and diurnal-range features: `rsds_max` 19% out of range,
  `rsds_peak` 16%, `tah_dtr` 11%. The test sites are sunnier with a wider diurnal swing than
  training — consistent with the lower stock levels, i.e. a sample skewed toward open, dry, sunny
  land. Tree ensembles clamp at the edge of their training range here, which is the safe behaviour;
  an unconstrained MLP would extrapolate linearly, which is not. Worth remembering when weighing
  the two model families.
- Every other feature: under 4% out of range. Static soil properties: under 1.1%.

---

## Entry 4: the approach, and the first models

### 4.1 Design, stated plainly

The competition scores **year-40 height and agb only**, under MSE on mean-scaled values. The
research benchmark this dataset was built for scores a 7-variable, 40-step, 12-month autoregressive
rollout. Those are different problems, and the prior work in this repo optimised the second one.

Given Entry 2.4 (`y40 ≈ y0` + a correction) and Entry 3.3 (the affine distortion cancels out of a
difference), the approach is:

> **Predict `Δ = y40 − y0` directly, in one shot, from a 40-year climate summary + the initial
> state + the seed age. No rollout.**

Three reasons this beats an autoregressive emulator *for this metric*:

1. **No error compounding.** A one-step model applied 40 times accumulates drift — which is exactly
   what the benchmark paper spends its effort fighting (noise injection, teacher forcing). A direct
   model never has that failure mode, because it is never fed its own output.
2. **The identity map is free.** `y0` enters the prediction as an exact additive term, so the model
   only has to learn the correction. A rollout re-derives the level 40 times over.
3. **It sidesteps the affine problem.** The scale distortion of Entry 3 cancels from the delta but
   would contaminate every step of a rollout.

The cost is that the model cannot express year-to-year feedback. Accepting that is the bet, and the
numbers below are what justify it.

**Pipeline.** 14 GB raw cube → `src/features.py` reduces each (site, year) to 32 annual summaries,
including bioclimatic-style ones borrowed from species-distribution modelling (growing degree days
above 5 °C, warmest/coldest quarter temperature, wettest/driest quarter precipitation, a VPD proxy
from Tetens' equation, an aridity index, and the temperature–precipitation phase correlation, which
distinguishes Mediterranean from monsoon climates at equal annual totals). `src/dataset.py` then
aggregates the 40 years and assembles one row per (site, seed age): **812,280 rows × 111 features**.
CO₂ excluded per Entry 3.4.

**Validation.** Held-out **sites**, not rows — 10,830 of 54,152 sites (20%), so all 15 seed ages of a
held-out site are unseen together. Scored in the exact competition metric.

A note on why this validation is conservative: the test-space error is `a·(Δ̂ − Δ)`, and `a` is 0.92
for height and 0.84 for agb, so the leaderboard score should be roughly `a²` ≈ 0.85/0.71 times my
validation number, other things being equal.

### 4.2 Results so far

| model | val score | vs persistence |
|---|---|---|
| persistence (Δ = 0) | 0.30402 | 1.0x |
| mean Δ | 0.29002 | 1.0x |
| age-conditional mean Δ | 0.23966 | 1.3x |
| ridge on 111 features | 0.12458 | 2.4x |
| **residual MLP, 111 feats, 30 epochs** | **0.01163** | **26x** |
| residual MLP, 301 feats + multi-horizon, 60 epochs | 0.01231 | 25x |

The MLP (4 residual pre-norm blocks, width 512, a learned 15-way seed-age embedding, multi-task
heads on all 7 variables with the 5 unscored ones down-weighted to 0.15) reaches **0.01163** — 26x
better than persistence and 10.7x better than ridge, so the mapping is strongly nonlinear.

### 4.3 An honest negative result

The "obvious" upgrade — v2, with 301 features instead of 111 (adding per-year min/max and four
decade anomalies) plus auxiliary supervision at years 10/20/30 — came out **worse**: 0.01231 against
0.01163. More features and more supervision made it worse, which is worth taking seriously rather
than tuning past. Running an ablation (`run_ablate.sh`) to separate the two changes:
A = v2 features alone, B = multi-horizon alone, C = neither, all at matched capacity and epochs.

## Entry 5: the ablation — simpler climate features win

Three runs at matched capacity (width 640, 5 blocks, 60 epochs), differing only in the feature
set and whether auxiliary year-10/20/30 supervision was used:

| run | climate features | multi-horizon | val score |
|---|---|---|---|
| A | v2 (301: + min/max + decade anomalies) | no | 0.01242 |
| **B** | **v1 (111: mean/std/trend)** | **yes** | **0.01137** |
| C | v1 (111: mean/std/trend) | no | 0.01157 |

Two conclusions:

**The richer climate features hurt, by 9%** (A vs C: 0.01242 vs 0.01157). This is the opposite of
the usual expectation and worth understanding rather than just accepting. The features that were
added — per-year min/max and four decade anomalies — describe *when in the 40-year record* things
happened. But every training site shares the same 1980–2020 weather realisation, so a "decade-2
anomaly" is substantially a global temporal pattern rather than a site property. The model can fit
it in-sample and it does not generalise to a held-out site. Worse, it would transfer especially
badly to the SSP scenarios, whose 40-year temporal structure is different by construction. So the
simpler aggregation is both more accurate here **and** the safer choice under the scenario shift —
the two criteria agree, which is a comfortable place to be.

**Multi-horizon supervision helps slightly**, 1.7% (B vs C: 0.01137 vs 0.01157). Asking the model to
also predict the year-10/20/30 deltas at weight 0.25 regularises the year-40 head a little. Kept.

Best configuration so far: **v1 climate block + multi-horizon, 0.01137**.

Next: the v4 feature set, which adds a **site-level initial-state profile**. The reasoning is that a
row for (site, age = 1) currently sees only that row's own initial state plus climate — yet the same
site's age-500 initial state is a direct readout of the site's carrying capacity, which is far more
informative about achievable biomass than any climate summary. `test_y0` ships all 15 seed ages for
every site, so this costs nothing at inference. v4 = v1 climate block + 30 profile features
(the height/agb-vs-age curve at 6 anchor ages, per-variable mean/max across ages, total span,
mid-curve slope) + 4 "headroom" features giving each row's distance from its own site's ceiling.

---

## Entry 6: v4 works, and where the error now lives

**v4 (site-level initial-state profile) scores 0.00987**, against 0.01137 for the same configuration
without it — a **13% improvement**, the largest single gain after the delta reformulation itself.

The stratified error analysis (`src/analyze.py`) confirms the mechanism rather than just the result.
Before the profile features, age 1 was expected to be the hardest case: it has the weakest
`corr(y0, y40)` of any seed age (0.758 for agb, Entry 2.4) because young stands actually grow.
After adding them, **age 1 is the easiest** — 0.00253, versus 0.01585 at age 500. Knowing the site's
age-500 initial state tells the model where a young stand is heading, which is exactly what the
feature was designed to supply.

*(An aside on process: the first run of this analysis reported a nonsensical 0.04898 with 107k rows
at age 500. The script was deriving the seed age from a fixed column offset that is only correct for
the v2 layout. Replaced with `row % 15`, which is exact by construction since rows are site-major
with the 15 ages in fixed order — and is now used in the trainer too.)*

Remaining error, after the fix:

| stratum | finding |
|---|---|
| seed age | rises monotonically with age: 0.0025 (age 1) → 0.0159 (age 500); ages ≥150 hold 64% of the error |
| climate | warm-temperate + tropical = **56%** of error; polar just 1.2% |
| Δ magnitude | top three deciles (\|Δ\|>0.535) = **71%** of error |

So the headroom is in mature stands in warm climates that move a lot — not in the young-stand
dynamics the benchmark literature emphasises.

---

## Entry 7: pinning down the affine slope with a sampling-invariant test

Entry 3 left `a` under-determined, and a sensitivity check showed this mattered more than I first
judged: predicting with `a = AFF_A` versus `a = 1` changes the submission by 0.018 (height) and
0.037 (agb) in scaled units — an MSE gap of **0.0038** if one is right and the other wrong, which is
~40% of the model's whole score. Worth resolving properly.

**The problem with marginals.** Quantile matching cannot distinguish "the scale was transformed"
from "the test sites are a different sample" — both reshape the marginal distribution. And the
evidence genuinely pulled both ways: the bulk quantiles favour `a ≈ 0.92`, but the observed test
maximum (29.77 m against a training maximum of 35.0 m) is what `a = 1` would predict.

**A test that sampling cannot fake.** The height–agb allometry `p(agb | height)` is a property of
forest physics, not of which grid cells you drew. Under the affine map the test conditional must
satisfy `median_test_agb(h) = a_agb · median_true_agb((h − b_h)/a_h) + b_agb`, so fitting that curve
identifies both slopes at once (`src/identify_a.py`, 691k training and 78k test rows, barren rows
excluded).

| (a_height, a_agb) | conditional-curve RMSE |
|---|---|
| (1.000, 1.000) — no transform | **1.2157** |
| (1.000, 0.840) | 1.8098 |
| (0.921, 1.000) | 0.5909 |
| (0.921, 0.840) — current constants | 0.5533 |
| (0.877, 0.840) — landmark height | 0.5267 |
| (0.860, 0.720) — grid optimum | 0.2849 |

**`a = 1` is decisively rejected** — it is the worst-fitting hypothesis on the axis that sampling
cannot contaminate. All three independent estimators (landmark 0.877, quantile 0.921, allometry
0.860) agree that `a_height` lies in roughly 0.86–0.92.

This test has its own caveat, and I would rather state it than hide it: it assumes the height–agb
relationship itself is the same in the test population. If those sites hold systematically different
vegetation types, the allometry could genuinely differ. So it is corroborating evidence, not proof.

**Decision: keep `a = (0.9206, 0.8401)`.** Within the surviving range the choice barely matters —
moving `a_height` from 0.921 to 0.89 shifts predictions by ~5e-5 in MSE, and `a_agb` from 0.840 to
0.78 by ~2e-4, both negligible against a score near 0.01. It was only the rejected `a = 1` that
carried real cost. The remaining uncertainty is documented rather than tuned away.

### 7.1 The inherited submission was indeed broken

Entry 0 flagged `submissions/meta_monthly_student_submission.csv` as implausible. Confirmed: it
predicts a year-40 scaled height of 0.1356, i.e. **1.48 m**, for sites whose *year-0* height is
already 8.34 m on average. It forecasts mature forests collapsing to knee height. The new pipeline
gives 0.857 scaled (9.35 m) against a year-0 mean of 8.34 m — modest growth, which is what the
training trajectories show.

---

## Entry 8: final ensemble, ablations closed out, submission written

**5-fold over sites**, v4 features, width 640 / 5 blocks / 70 epochs, multi-horizon at 0.25:

```
fold scores  0.00943  0.01009  0.01005  0.00998  0.01003
OOF OVERALL  0.00992   (height 0.01193, agb 0.00790)
```

A spread of 0.0007 across folds, so the estimate is stable. Against persistence at 0.30402 that is
**31x**. Note this measures a *single* model's generalisation; the submitted file averages all five,
which should be modestly better but cannot be measured here (every site is in four of the five
training sets, so the ensemble has no held-out data). The quoted figure is the conservative one.

### 8.1 CO₂ ablation — the exclusion is free

Entry 3.4 argued CO₂ should be dropped on distribution-shift grounds, with the ablation left open.
Run now (v4 + the three CO₂ aggregates, identical configuration):

| | score |
|---|---|
| CO₂ excluded (v4, shipped) | 0.00987 |
| CO₂ retained | 0.00983 |

0.4% — within seed noise. Exactly as predicted: a feature with no cross-site variance carries no
usable in-distribution signal, so removing a 100%-out-of-range input costs nothing. Decision
confirmed on evidence rather than argument alone.

### 8.2 Extrapolation risk — measured, and it is small

Test features do reach |z| = 28, but the training data itself reaches |z| = 21, and only **0.076%**
of test feature values exceed the model's built-in ±8σ input clip. A harder variant clamping every
feature to the training min/max moves the submission by an implied MSE of 0.00056 (5.6% of score),
written to `submissions/submission_clipped.csv`. **The unclipped file is the one to submit** — the
±8σ clip is the operative guard, and hard clamping would squash genuinely-sunnier test sites into
looking like the sunniest training site, which is its own bias.

### 8.3 Submission checks

`submissions/submission.csv` — 185,130 rows, id order byte-identical to `sample_submission.csv`,
no non-finite values, floor clamped exactly at `b / y_mean`.

| quantity | value |
|---|---|
| year-0 mean (given) | height 8.346 m, agb 2.600 |
| year-40 mean (predicted) | height 9.596 m, agb 2.489 |
| mean 40-year change | +1.250 m height, −0.111 agb |
| rows where height decreases | 34.5% |
| ssp585 − ssp126 | height +0.0069, agb −0.0057 (scaled) |

Direction and magnitude match the training trajectories (mean training Δ is +1.77 m height, −0.07
agb — canopy grows while biomass dips slightly). The high-emissions scenario gives marginally taller
but slightly less carbon-dense stands, consistent with warming raising productivity and respiration
losses together.

### 8.4 What I would do next, in priority order

1. **Attack the actual error concentration.** 64% of remaining error is in seed ages ≥150 and 56% is
   in warm-temperate/tropical sites. A model specialised on mature warm-climate stands, or a loss
   reweighted toward them, targets the error where it lives rather than where intuition says it is.
2. **A held-out-from-everything site set.** The current design cannot measure the ensemble's own
   generalisation. Reserving ~5% of sites from all folds would fix that cheaply.
3. **Resolve the affine slope `a` properly** by finding a third sampling-invariant landmark.
   Currently `a_height` is only bounded to 0.86–0.92 (the impact is small, but it is untested).
4. **Do not** invest in autoregressive rollout machinery for this metric. The evidence across this
   session is consistent: the level is given, the correction is what needs predicting, and every
   feature set that leaned harder on temporal ordering generalised worse.

---

## Entry 9: real leaderboard scores arrive — offline validation was systematically misleading

A later session built a sequence-Transformer variant (per-year climate encoder, self-attention over
the year axis, a delta head) and, following the "is horizon 40 special?" question, a variable-horizon
version trained with windowed `(start year, end year)` sampling instead of only ever year-0→year-40.
Three legitimate submissions plus the old broken rollout submission were scored on the real Kaggle
leaderboard:

| submission | offline validation | leaderboard | ratio (worse by) |
|---|---|---|---|
| `submission.csv` (v4 tabular MLP, 5-fold, full 54,152-site grid) | 0.00992 | **0.151** | 15.2x |
| `submission_transformer_global.csv` (sequence Transformer, full grid, fold-0 holdout) | 0.0135 | **0.159** | 11.8x |
| `submission_transformer_restrain4.csv` (variable-horizon Transformer, `res_train4_test8.npz` only, test8 holdout) | 0.0432 | **0.147** | 3.4x |
| `meta_monthly_student_submission.csv` (old broken rollout, ×2) | — | 0.372, 0.417 | — |

Two findings, and the second is the one worth remembering:

**1. Every offline validation number was wildly optimistic — by 3x to 15x.** REPORT.md §6 risk 2
flagged this as a possibility ("validation cannot simulate the scenario shift... the single largest
unquantified risk") but it was a hedge, not a measured effect, until now. Site-holdout validation —
whether the 852-site test8 split or the full-grid 20% fold holdout — only ever tests generalisation
to *unseen sites within the same shared 40-year training climate realisation*. It has never once
tested generalisation to a genuinely different climate trajectory, which is exactly what the SSP
scenarios are and what the real leaderboard scores.

**2. The ranking is inverted, and there's a plausible, specific reason why.** The model that
validated *worst* offline (the small-data variable-horizon Transformer, 0.0432, worse than a flat MLP
on the identical data at 0.0387) is the *best* real submission. The model that validated best offline
(the full-grid v4 MLP, 0.00992) comes second on the leaderboard. The full-grid sequence Transformer —
worst offline among the full-grid models too — is worst on the leaderboard as well.

Working hypothesis, consistent with Entry 5's original finding: every model here that had more direct
exposure to the single shared training climate trajectory — more sites drawn from it, or (for the
sequence Transformer) direct positional/temporal access to its exact shape via self-attention over
the year axis — had more opportunity to fit patterns specific to *that one trajectory*, not a
genuinely site- and climate-conditional response. Entry 5 already showed this happening once
(decade-anomaly features fit the shared record and transferred worse); the full-grid sequence
Transformer's self-attention over 40 absolute year positions is a stronger version of the same
exposure, and it fits the same failure signature. Because every site shares one weather realisation,
this kind of overfitting is invisible to *any* site-holdout split — it only shows up against a truly
different climate trajectory.

The variable-horizon Transformer breaks this pattern, plausibly because of the windowed training
introduced for an unrelated reason (arbitrary-horizon querying): sampling many `(t, e)` pairs instead
of only the fixed year-0→year-40 anchor forces the model to predict from many different points within
the training trajectory rather than memorising its one specific absolute shape. That's exactly the
kind of exposure that should generalise better to a *different* trajectory. Consistent with this, its
offline-to-leaderboard gap (3.4x) is far smaller than either full-grid model's (11.8-15.2x) — its
offline number, while nominally the "worst" of the three, was actually the most honest one.

**Caveat.** N=3 real submissions is not enough to be certain of a causal mechanism, and the public
leaderboard previews a random subset of sites (`README.md`: "your live score previews the final
one"), so some of the fine-grained ranking could be public-subset noise rather than a true
generalisation difference. The *size* of the offline-to-leaderboard gap (3-15x, consistently in the
same direction across all three) is too large to be noise; the precise ordering between the three
legitimate submissions should be treated as suggestive, not proven.

### 9.1 What this changes going forward

1. **Site-holdout validation cannot be trusted for this task, possibly not even in relative ranking
   between architectures.** Any further model comparison should be treated with this in mind, not
   just as a caveat in a risks section.
2. **A genuine climate-shift-aware validation split is now worth building**, rather than trusting
   site-holdout numbers at all — e.g. a synthetic perturbation of the training climate forcing, or a
   holdout stratified on climate-trajectory features rather than site identity.
3. **The natural next experiment**: combine the windowed/variable-horizon training technique with the
   *full* 54,152-site grid, on the hypothesis that "more data" and "climate-shift robustness from
   windowing" are separable benefits that could compound rather than trade off. Untested.
4. **Treat `submission_transformer_restrain4.csv` as the current best real candidate**, not
   `submission.csv`, until a more reliable validation methodology says otherwise.
