# CarbonGlobe / IEEE BigData Cup 2026 — Approach and Results

*Written for: engineers and researchers working on this challenge — assumes familiarity with the
CarbonGlobe dataset, but not with any of the choices made here.*

Companion documents: `agent/LOGBOOK.md` (chronological record of every experiment, including the
ones that failed) and `agent/DATASET_NOTES.md` (an earlier session's dataset survey — **note that
its channel-layout table is incorrect; see §2.1**).

---

## 1. Summary

The competition asks for the **year-40 height and above-ground biomass** of 6,171 sites × 15 seed
ages under two future-climate scenarios, scored as MSE on mean-scaled values. The dataset it is
built from — CarbonGlobe — was designed for a different task: emulating a 7-variable, 40-step,
12-month autoregressive simulation. Much of the value here came from noticing that gap and
exploiting it, and from establishing what the files actually contain before modelling anything.

Three decisions did most of the work:

1. **Predict the change, not the state.** `y40 ≈ y0` plus a correction; a two-parameter line through
   `y0` already beats the age-conditional mean by 10x. Modelling `Δ = y40 − y0` in one shot removes
   both the identity map and the error compounding that dominates autoregressive rollouts.
2. **Do the data forensics first.** Four properties of the files materially change the model, and
   three are undocumented: an interleaved channel layout, a fully redundant month axis, an affine
   scale shift between the test initial states and the training data, and CO₂ inputs that fall
   100% outside the training range.
3. **Give every row its site's whole initial-state profile.** A row for (site, age 1) becomes much
   easier once it can see that site's age-500 initial state, which reveals the site's carrying
   capacity. This alone improved the score 13%.

**Result: 0.00992 out-of-fold in the competition metric, against 0.30402 for persistence
(`ŷ40 = y0`) — 31x better.** Submission: `submissions/submission.csv`.

---

## 2. What the data actually contains

Everything here was established by measurement, not assumption. Scripts are in `src/`.

### 2.1 The channel layout in the inherited notes is wrong

`DATASET_NOTES.md` describes the 136 input channels as contiguous blocks (24 CO₂, then 24 humidity,
then 24 temperature, …). Per-channel statistics refute this: under that reading `ta_h` would have a
minimum of 0 and a maximum of 1143 — not a temperature. The real layout is

```
0       ta_m      air temperature (K)
1       pr        precipitation (mm)
2:8     tsl1..6   soil temperature at 6 depths (K)
8:32    co2       24 hours-of-day (ppm)
32      dst       disturbance — constant 0.012 everywhere, no information
33:129  24 interleaved repeats of (hus, ta_h, rsds, sfcWind)   <-- stride 4, NOT blocks
129:136 k_sat, s_theta, r_theta, L, n, m, sd   (static soil)
```

Confirmed by the diurnal cycles: channels `34+4k` trace a temperature curve peaking at hour 13, and
channels `35+4k` a solar curve floored at exactly 0 overnight. Under the block reading, any feature
engineering silently blends humidity, temperature, radiation and wind together.

### 2.2 The month axis of the targets is pure replication

`glob_Y{age}` is shaped `(54152, 41, 12, 7)`, but across 1,500 random sites × 15 ages × 41 years the
maximum deviation between months is **exactly 0.0**. The targets are annual and always were, and
11/12 of ~11 GB of target data is duplication.

This also settles a real ambiguity in the submission spec: since `test_y0` has no month axis, one
otherwise has to guess whether "annual" means the December value or the annual mean. It means
neither — they are the same number.

### 2.3 The test initial states are on a shifted scale

Training `y0` has an exact floor of **0.0** for height, agb, soil, lai and rh (barren land), hit by
~9% of rows. The test set has an exact floor too — **−0.9009 / −0.4014 / −0.5324 / +0.1179 /
+0.0847** — hit by ~9.5% of rows, and **identical at every seed age from 10 to 500**.

A negative biomass floor cannot arise from sampling different sites, and per-age-identical minima
cannot arise from noise. This is a deterministic per-variable affine map, `test = a·true + b`, with
`b` readable directly off the floor and quantile-fit R² of 0.998–0.9999.

The slope `a` is harder, because marginal distributions cannot separate "the scale was transformed"
from "the test sites are a different sample" — both reshape the marginal, and the evidence pulled
both ways (bulk quantiles favour `a ≈ 0.92`; the observed test maximum of 29.77 m against a training
maximum of 35.0 m is what `a = 1` predicts). It was resolved using the **height–agb allometry**, a
conditional relationship that sampling cannot fake. That test rejects `a = 1` decisively (curve RMSE
1.22 versus 0.53) and agrees with two other estimators that `a_height ≈ 0.86–0.92`. See LOGBOOK
Entry 7 for the full argument and the residual uncertainty.

This is also where the delta formulation pays off a second time: **the intercept cancels from a
difference**, `Δ_test = a·Δ_true`, so the distortion collapses to a single scale factor on the
correction term rather than contaminating every step of a rollout.

### 2.4 CO₂ is unusable, and excluding it is not optional

Training CO₂ runs 357 → 419 ppm over the 40 years (the historical record). `ssp126` runs
363.8 → 371.2 and `ssp585` runs 363.9 → 393.1 — far milder than real SSP forcing. Per-site 40-year
means put **100.00% of test sites outside the training range** in both scenarios.

Worse, CO₂ is near-uniform in space, so across training sites it carries almost no cross-sectional
variance — it is effectively a single global time series, perfectly collinear with the year index.
No causal CO₂ response can be learned from that, only the collinearity. A model reading CO₂ ≈ 368
would infer "year 9" and hold the stand there for the entire forecast.

**CO₂ channels are excluded**, and §5 reports the ablation confirming this costs nothing
in-distribution. Scenario discrimination survives through temperature (+0.28 K), radiation and VPD,
which do differ between the two scenarios.

### 2.5 Two things that were checked and dropped

- **Mapping SSP sites back onto the global grid.** `sites_ssp.csv` is only row indices 0…6170, so
  the scenarios ship no coordinates. Fingerprinting sites by their 7 static soil properties fails:
  zero exact matches, and a nearest/second-nearest distance ratio with median **0.843** (only 0.5%
  of sites below 0.1). The assignment is not identifiable, so pursuing it would have injected wrong
  labels into features.
- **An apparent noise signature in the test inputs.** The time-invariant soil channels appeared to
  vary in time (std up to 1.39), which would have implied deliberately noised test inputs. It is a
  float32 artifact: `k_sat` is of order 1e6, where float32 spacing is ~0.5. The training data shows
  the same artifact. Relative noise is 0.0000.

---

## 3. Approach

### 3.1 Why one-shot delta regression rather than a rollout

The benchmark literature for this dataset evaluates a 40-step autoregressive rollout and spends
considerable effort fighting the resulting drift (teacher forcing, random-walk noise injection).
That effort is aimed at a problem this competition does not pose. Measured on 8,000 random sites in
the competition metric:

| predictor | score |
|---|---|
| all zeros | 2.78402 |
| global mean | 1.68412 |
| age-conditional mean | 1.61426 |
| **persistence (`ŷ40 = y0`)** | **0.29690** |
| **per-age linear fit on `y0` alone** | **0.16533** |

Persistence beats the age-conditional mean by 5.4x, and two coefficients per age per variable beat
it by 9.8x. Of year-40 height variance, **91.1% is across sites and 0.3% across seed ages**; mean
trajectories are nearly flat (age-100 height drifts 11.78 → 11.90 m over the full 40 years).

So the task is a small correction on top of a known level. Predicting the level directly spends
model capacity relearning an identity map; rolling forward 40 times compounds error into it. A
direct `Δ` model has neither failure mode, and gets the affine cancellation of §2.3 for free.

The cost, stated plainly: the model cannot express year-to-year feedback, and cannot represent a
site whose trajectory depends on the *order* of climate events rather than their 40-year statistics.
That is the bet. The measured gap between it and persistence (30x) is what justifies it.

### 3.2 Pipeline

The 14 GB input cube is streamed once and reduced to 32 annual summaries per (site, year)
(`src/features.py`, 277 MB total). Alongside conventional means and variances these include
bioclimatic predictors borrowed from species-distribution modelling: growing degree days above 5 °C,
warmest/coldest-quarter temperature, wettest/driest-quarter precipitation, a VPD proxy via Tetens'
equation, an aridity index, and the temperature–precipitation phase correlation — which distinguishes
Mediterranean from monsoon climates at identical annual totals.

**Design matrix:** one row per (site, seed age) — **812,280 × 149**.

| block | count | content |
|---|---|---|
| climate | 93 | 40-year mean / std / linear trend of 31 annual features |
| static | 7 | soil hydraulic properties (`k_sat` log-scaled) |
| initial state | 7 | the row's own `y0`, all 7 variables |
| **site profile** | **30** | the site's height/agb-vs-age curve at 6 anchor ages; per-variable mean and max across ages; total span; mid-curve slope |
| engineered | 12 | seed age (linear and log), allometric ratios, carbon-use and light-use efficiency, and **headroom** — the row's distance from its own site's ceiling |

The **site profile** block is the main modelling idea beyond the delta reformulation. A row for
(site, age 1) otherwise sees only its own near-zero initial state plus climate. But the same site's
age-500 initial state is a direct readout of that site's carrying capacity — far more informative
about achievable biomass than any climate summary. `test_y0` ships all 15 seed ages for every site,
so this costs nothing at inference.

**Model.** Residual pre-norm MLP (width 640, 5 blocks, GELU) with a learned 15-way seed-age
embedding, predicting `Δ` for all 7 variables at 4 horizons (years 10/20/30/40). Only year-40 height
and agb are scored; the other five variables are weighted 0.15 and the other three horizons 0.25, as
auxiliary regularisation. Loss is MSE in exactly the competition's units (`Δ / y_mean`).

**Validation.** 5-fold **over sites** — every site is held out exactly once, so all 15 seed ages of a
held-out site are unseen together, and the 5-model ensemble collectively covers all 54,152 sites
while still yielding an honest out-of-fold estimate.

Training on all global sites is legitimate here: the competition's test set is a *separate* set of
SSP simulations, so the `train/val/test.csv` split — which exists for the research benchmark — does
not constrain this task.

**Inference.**

```
y0_true  = (test_y0 − b) / a        # undo the affine distortion of §2.3
Δ̂        = model(features, y0_true) # delta in training space
y40_test = test_y0 + a · Δ̂          # back to the space the solution is stored in
y40_test = max(y40_test, b)         # a true value of 0 maps to b
submit   = y40_test / y_mean        # 10.90645 for height, 3.39567 for agb
```

---

## 4. Results

### 4.1 Progression

All figures are the competition metric on **held-out sites**, lower is better.

| stage | score | vs persistence |
|---|---|---|
| persistence (`Δ = 0`) | 0.30402 | 1.0x |
| age-conditional mean `Δ` | 0.23966 | 1.3x |
| ridge regression, 111 features | 0.12458 | 2.4x |
| residual MLP, v1 features | 0.01157 | 26x |
| + multi-horizon auxiliary supervision | 0.01137 | 27x |
| + **site initial-state profile** (v4) | 0.00987 | 31x |
| **5-fold ensemble, out-of-fold** | **0.00992** | **31x** |

Fold scores: 0.00943, 0.01009, 0.01005, 0.00998, 0.01003 — a spread of 0.0007, so the estimate is
stable. Split by variable, out-of-fold: **height 0.01193, agb 0.00790**.

One caveat on what this number measures: the out-of-fold figure is the generalisation of a *single*
model. The submitted file averages all five, which should be modestly better on the test set, but
that gain cannot be measured with this design — every site is in four of the five training sets, so
no held-out data exists for the ensemble as a whole. The quoted 0.00992 is therefore the
conservative reading.

### 4.2 Where the remaining error is

| stratum | finding |
|---|---|
| seed age | rises monotonically: 0.0025 (age 1) → 0.0159 (age 500); ages ≥150 carry 64% of the error |
| climate | warm-temperate + tropical = **56%** of error; polar just 1.2% |
| Δ magnitude | top three deciles (\|Δ\|>0.535) = **71%** of error |

The age result is the clearest evidence that the site-profile block does what it was designed to.
Before it, age 1 was expected to be the hardest case — it has the weakest `corr(y0, y40)` of any
seed age (0.758 for agb) because young stands genuinely grow. After it, **age 1 is the easiest
stratum of all**. Knowing the site's age-500 initial state tells the model where a young stand is
headed.

Headroom therefore lies in mature stands in warm climates that move a lot — not in the young-stand
dynamics the benchmark literature emphasises.

### 4.3 Sanity of the submission

| quantity | value |
|---|---|
| year-0 mean (given) | height 8.346 m, agb 2.600 |
| year-40 mean (predicted) | height 9.596 m, agb 2.489 |
| mean 40-year change | **+1.250 m height, −0.111 agb** |
| rows where height decreases | 34.5% |
| ssp585 − ssp126 | height +0.0069, agb −0.0057 (scaled units) |

The direction and magnitude match the training trajectories, where mean `Δ` is +1.77 m for height
and −0.07 for agb — biomass dips slightly over 40 years while canopy height grows. The high-emissions
scenario gives marginally taller but slightly less carbon-dense stands, consistent with warming
raising both productivity and respiration losses.

For contrast, the inherited `meta_monthly_student_submission.csv` predicts a year-40 scaled height of
0.1356, i.e. **1.48 m**, for sites whose *year-0* height averages 8.35 m — mature forests collapsing
to knee height. That pipeline was optimising the research rollout metric, and something in it had
gone badly wrong.

---

## 5. What did not work

Recording these because none was obvious in advance, and each cost real time.

**Richer climate features made things worse, by 9%.** Adding per-year min/max and four decade
anomalies (301 features instead of 111) degraded the score from 0.01157 to 0.01242 at matched
capacity. The explanation matters more than the number: every training site shares the same
1980–2020 weather realisation, so a "decade-2 anomaly" is substantially a *global temporal pattern*
rather than a site property. The model fits it in-sample and it does not transfer to a held-out
site — and it would transfer worse still to the SSP scenarios, whose 40-year temporal structure is
different by construction. Accuracy and robustness pointed the same way, which is a comfortable
place to land.

**A sequence model over the 40 years was built and then deliberately not used.** A dilated temporal
CNN + GRU reading the raw `(40, 31)` climate sequence was written (`src/model3.py`) for ensemble
diversity. Given the result above — that giving a model more access to the *temporal ordering* of a
single shared weather realisation actively hurts generalisation — running it would have amplified
exactly the failure mode the ablation had just exposed. It is left in the repository, unused, with
this reasoning attached.

**Site matching failed.** Fingerprinting SSP sites against the global grid by their static soil
properties is not identifiable (nearest/second-nearest distance ratio median 0.843). Abandoned
rather than used at 84% ambiguity.

**Multi-horizon supervision helps, but only slightly** — 1.7% (0.01137 vs 0.01157). Kept; not where
the gains are.

**A false alarm worth recording.** The time-invariant soil channels appeared to vary in time in the
test set, which would have meant deliberately noised inputs and a very different modelling response.
It was a float32 artifact — `k_sat` is of order 1e6, where float32 spacing is ~0.5 — and the training
data shows it identically. Caught before it influenced anything.

**CO₂ ablation.** Retaining CO₂ (v4 + the three CO₂ aggregates, identical configuration) scores **0.00983**
against 0.00987 without — a 0.4% difference, within seed noise. Exactly what §2.4 predicts: a
feature with no cross-site variance carries no usable in-distribution signal. Excluding it is
therefore free in-sample and removes a 100%-out-of-range input at test time.

---

## 6. Risks and limitations

1. **The affine slope `a` is estimated, not known.** Three independent estimators put `a_height` in
   0.86–0.92 and all reject `a = 1`, but the allometry test assumes the height–agb relationship is
   the same in the test population. Within the surviving range the impact is small (moving `a_height`
   from 0.921 to 0.89 shifts predictions by ~5e-5 in MSE); the rejected `a = 1` was the expensive
   error, and it is rejected from the one direction sampling cannot contaminate.
2. **Validation cannot simulate the scenario shift.** Held-out sites test spatial generalisation, not
   generalisation to a different climate trajectory. All training sites share one 40-year weather
   realisation, so no honest internal estimate of scenario-transfer error exists. This is the single
   largest unquantified risk, and it is why the simpler climate aggregation was preferred even where
   the accuracy difference was modest.
3. **Some test features sit outside the training range.** `rsds_max` (19% of sites), `rsds_peak`
   (16%) and `tah_dtr` (11%) fall outside the training 0.5–99.5 percentile band — the test sites are
   sunnier with a wider diurnal swing. An MLP extrapolates linearly there, where a tree ensemble
   would clamp. Measured: only **0.076%** of test feature values exceed the model's built-in ±8σ input clip, and the
   training data itself reaches |z| = 21, so extrapolation is confined to a small tail that the clip
   already bounds. A more aggressive variant that clamps every feature to the training min/max
   (`src/submit.py --clip`, written to `submissions/submission_clipped.csv`) moves the submission by
   an implied MSE of 0.00056 — 5.6% of the score. The unclipped file is submitted, since the ±8σ clip
   is the operative guard and hard clamping would distort genuinely-sunnier sites into looking like
   the sunniest training site.
4. **Leaderboard score should be better than the validation number.** Test-space error is
   `a·(Δ̂ − Δ)` with `a` = 0.92 (height) and 0.84 (agb), so the score should scale by roughly `a²`
   ≈ 0.85 / 0.71, other things equal. The quoted out-of-fold figure is therefore conservative —
   though only with respect to scaling, not with respect to risk 2.
5. **Early stopping selects on the validation fold**, which mildly optimistically biases each fold's
   reported score. The out-of-fold aggregate inherits that bias. It is consistent across every model
   compared here, so the rankings hold.

---

## 7. Reproducing

Interpreter: `E:\miniconda3\envs\torch\python.exe` (torch 2.13 + ROCm; plain `python` is a Windows
Store stub and will not work).

```
python src/build_cache.py y      # collapse the redundant month axis   -> cache/Y_glob.npy
python src/build_cache.py x      # stream the 14 GB cube               -> cache/feat_glob.npy
python src/build_cache.py ssp    # both scenarios                      -> cache/feat_ssp*.npy
python src/dataset.py            # v1 design matrix + site-holdout mask
python src/dataset4.py           # v4 design matrix (the one used)
python src/train_folds.py --feat X4 --nfold 5 --epochs 70 --d 640 --nblk 5 --aux_hor 0.25 --tag f
python src/submit.py --pattern "f[0-9].pt" --out submission.csv
```

Supporting analyses: `src/explore0*.py` (array forensics), `src/y0_probe.py`, `src/affine_probe.py`,
`src/landmark_probe.py`, `src/identify_a.py` (the affine transform), `src/shift_check.py`
(train/test distribution shift), `src/analyze.py` (stratified error breakdown).
