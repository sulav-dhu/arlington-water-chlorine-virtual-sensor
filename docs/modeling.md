# Step 3 — models, selection and results

Code: `src/chlorine/modeling/` (design, spatial, rk, gbm, ensemble, run, report; backtest, sensitivity,
calibration, prospective for §7-8). Final model, intended use and limits: [model_card.md](model_card.md).

```bash
uv run python -m chlorine.modeling.run                                  # tuning + validation (development)
uv run python -m chlorine.modeling.run --split test --confirm-test      # once, after selection
```

Reports: `data/eval/step3/<split>/modeling_report.md` (+ results, comparisons, predictions, parameters).

## 1. The model: regression kriging with hierarchical site effects

```
y = city_week + X·β + s(place) + a(place, week) + α·own_recent + noise
```

| Term | What it is | How it is estimated |
|---|---|---|
| `city_week` | mean of the other monitored places this week | from data (leave-own-place-out) |
| `X·β` | public fixed effects: pressure plane, distance to each plant, hydrant features, Davis-area flag, plant-regime × plane/distance interactions, season (± utility region code) | recency-weighted ridge |
| `s(place)` | static site level | prior kriged from other places' levels (distance × same-plane covariance); places with history update it with their own recency-weighted mean residual (Bayesian, design effect 2 for autocorrelation) |
| `a(place, week)` | this week's local anomaly | simple kriging of same-week anomalies of other places; covariance fitted to the data |
| `own_recent` | the place's own anomalies in the last 8 weeks (decay ~4 weeks) | weight α fitted on training residuals; site-history models only |
| noise | within-week scatter | estimated; together with the kriging and site-level variances gives each prediction an SD |

Prediction intervals: the SD above × empirical quantiles of standardized training residuals, separately for
Kubala-outage periods.

**Leakage guard (found and fixed during development, before any test run).** A neighbor's anomaly is
measured against its city-wide weekly mean, which includes the target's own value; interpolating
neighbors' anomalies would feed the target's value back to itself. For targets in the pool, each
neighbor's baseline is rebuilt without the target, `(S − y_q − y_p)/(N − 2)`. An end-to-end test
(`tests/test_modeling.py`) confirms a target's own value, held-out places and the test period never
change a prediction.

**Estimator check.** Covariance fitting recovers known parameters from simulated Gaussian-process
data (variance 0.100, range 2.03 km, cross-plane factor 0.51 for truth 0.1, 2.0, 0.5). An earlier
version placed binned covariances at bin midpoints, biasing variance and range; it now uses each
bin's mean pair distance.

## 2. Candidates

| Code | Model | Uses site history | Uses utility region code |
|---|---|---|---|
| B6 | regression kriging | ✓ | |
| B6r | regression kriging + region | ✓ | ✓ |
| G6 | gradient boosting on neighbor contrasts, location features, regime, season, own recent deviations + site offsets | ✓ | ✓ |
| E6 | average of B6 and G6 | ✓ | ✓ (via G6) |
| B7 | regression kriging, no site history (any address) | | |
| B7r | B7 + region | | ✓ |
| G7 | gradient boosting, no site history | | (✓) one feature: same-week same-region neighbor contrast |
| E7 | average of B7 and G7 | | (✓) via G7; removing it changes nothing (§7) |

Round 1 = B6, B6r, G6, B7, B7r, G7. **Round 2 (E6, E7) was added after reading round-1 validation
results**, because B6 and G6 were within noise of each other and averaging dissimilar models is a
standard robustness step. To check that this was not tuning to one year, every candidate was also
scored on a second validation year (2022).

Ablations of B7/B6: without static kriging, without anomaly kriging, distance-only covariance (no
plane), without plant-regime terms, without own recent anomalies, without recency weighting.

## 3. Tuning and selection (fixed before the test run)

- Only tuned setting: the recency half-life (365 / 730 / 1461 days / none) of the regression-kriging
  models, by known-site RMSE on the inner validation year 2022 (trained on 2015-2021) → **365 days**.
  For the test split it is re-chosen on validation 2023.
- Selection rule: in each protocol, the candidate with the lowest validation-2023 RMSE.
- **Selected: E6 for known sites and outages; E7 for unseen sites.** Best public-only: B6 (known sites,
  outages), E7 (unseen sites).
- Test plan: references + all eight candidates, evaluated once; the headline is the selected model.

## 4. Validation results (2023; outages = leave-one-out over pre-2024 Kubala outages)

RMSE in mg/L, [95% cluster-bootstrap interval of the difference to the best Step 2 reference].

| | Best reference | Selected | Δ [95% CI] | Best public-only |
|---|---|---|---|---|
| Known sites | B5 0.304 | **E6 0.272** | −0.032 [−0.045, −0.020] | B6 0.277 |
| Unseen sites | B4 0.316 | **E7 0.308** | −0.008 [−0.024, +0.004] | E7 0.308 |
| Kubala outages | B5p 0.405 | **E6 0.356** | −0.049 [−0.068, −0.032] | B6 0.367 |

Second validation year (2022, trained on 2015-2021): the same ordering holds. Known sites: E6 0.272
vs. B5 0.315; unseen sites: E7 0.327 vs. B4 0.347.

Other validation findings:
- At the near-daily sites E6 reaches RMSE 0.285 against the 0.28 mg/L within-week noise floor.
- B6 intervals are well calibrated (81% coverage for a nominal 80%) and the sharpest (width 0.63).
- **What matters (ablations):** kriging same-week anomalies (+0.027 to +0.082 RMSE without it), the
  place's own recent anomalies (+0.015 known, +0.029 outages), recency weighting (+0.006 known).
  Static kriging, the plane factor in the covariance and the regime terms add ≤ 0.003 once the
  rest is in.
- **Fitted structure:** weekly anomalies share SD 0.26 mg/L over a ~2.3 km range; places in
  different pressure planes co-vary only 47% as much as places in the same plane (the public
  pipe-network structure is real, but planes are contiguous, so distance already captures most of
  it); within-week scatter SD 0.31.
- The flexible model (G6/G7) is not clearly better than the structured one (B6/B7) except during
  outages, where it learns how neighbor contrasts change when a plant is off.

## 5. Test results (2024 – Aug 2026, evaluated once)

Models refit on 2015-2023 (half-life re-chosen on 2023: 365 days). Logged in
`data/eval/test_access_log.jsonl`. RMSE in mg/L; Δ vs. the best Step 2 reference on validation, with
95% cluster-bootstrap intervals.

| | Step 2 references | Selected | Δ vs. reference [95% CI] |
|---|---|---|---|
| Known sites | B5 0.352 · B5p 0.343 | **E6 0.310** (B6 0.318, G6 0.315) | −0.042 [−0.055, −0.029] vs. B5 |
| Unseen sites | B4 0.368 · B4p 0.358 | **E7 0.352** (B7 0.364, G7 0.358) | −0.016 [−0.030, −0.001] vs. B4 |

| Test subset | Best reference | Selected |
|---|---|---|
| Known sites during the two Kubala outages (658 samples) | B5p 0.412 | **E6 0.363** |
| Known sites, places new since 2023 (817 samples, 24 places) | B5p 0.365 | **E6 0.309** |
| Near-daily sites (noise floor 0.28) | B5p 0.354 | **E6 0.309** |
| Unseen sites during the Kubala outages | B4p 0.431 | **E7 0.415** |

What the test run shows:

1. **The gains generalize and grow.** E6 cuts known-site error by 12% against B5 (10% against the best
   reference on test, B5p), more than on validation (10.5%), and is the best model in every test subset.
2. **Biggest gains where they matter:** during plant outages (−0.049 vs. B5p) and at new sites
   (−0.056).
3. **Public data alone is competitive with the utility's knowledge.** At known sites, E7, which uses no
   site history and the region code only through one neighbor feature that adds nothing (§7), beats B5,
   which uses both (0.332 vs. 0.352, Δ −0.020 [−0.028,
   −0.012]). At unseen sites E7 is the best model, and its edge over the region-code reference is now
   significant.
4. **Prediction intervals are too narrow in 2024-26**: 80% intervals cover 73-76% (E6 0.75, E7 0.73).
   They were calibrated on in-sample training residuals, which understate out-of-sample error, and
   2024-26 was more volatile (outages, Pierce-Burch construction). Recalibration on rolling out-of-sample
   residuals was developed on 2017-2023 only (§7); its independent check is the prospective test (§8).
5. Ranking samples by risk works: AUC for samples < 2.5 mg/L is 0.93 (E6) and 0.88 (E7).

## 6. Limitations and next steps

- Plant on/off flags are inferred from sampling gaps; the model would benefit most from an operations log.
- Intervals: out-of-sample adaptive calibration (§7) holds 75-85% coverage every development year, but the
  first weeks of a plant outage remain under-covered.
- The public-data model predicts the static level of an unmonitored place only from neighbors and
  public attributes; a pipe network (diameters, flow paths, water age) is the missing piece for
  unseen sites, where the gap to known sites is still ~0.04 mg/L.
- Only the recency half-life was tuned; the gradient-boosting settings are fixed defaults.

## 7. After the test run: backtest, final choice and interval calibration (development data only)

Nothing in this section uses 2024-26 for evaluation. Numbers: [reports/key_numbers.md](../reports/key_numbers.md).

```bash
uv run python -m chlorine.modeling.backtest              # E6/E7 + references, 2017-2023 -> data/eval/backtest/dev.parquet
uv run python -m chlorine.modeling.backtest --members    # B6, G6, B7, G7 alone -> dev_members.parquet
uv run python -m chlorine.modeling.backtest --horizon 2019:2022   # no retraining for 3 years
uv run python -m chlorine.modeling.sensitivity           # E7 without the region code -> dev_public_strict.parquet
uv run python -m chlorine.modeling.calibration           # interval study -> data/eval/calibration/
```

**Rolling backtest.** For each year 2017-2023, models are trained on all earlier years and predict that
year, under the known-site and unseen-site protocols. The 2022 and 2023 years reproduce the Step 3
validation runs exactly; this is a checked invariant of the numbers sheet.

| Pooled 2017-2023, RMSE difference [95% CI, places resampled] | Known sites | Unseen sites |
|---|---|---|
| Final − best baseline | E6 − B5p −0.040 [−0.048, −0.032]; better in 7/7 years | E7 − B4p −0.019 [−0.031, −0.008]; 7/7 |
| Final − kriging alone | E6 − B6 −0.004 [−0.007, −0.001]; 7/7 | E7 − B7 −0.011 [−0.021, −0.004]; 7/7 |
| Kriging alone − best baseline | B6 − B5p −0.036 [−0.042, −0.029] | B7 − B4p −0.008 [−0.018, +0.002] |
| Without the region code | E7s − E7 −0.000 [−0.001, +0.001] | E7s − E7 −0.000 [−0.001, +0.001] |

**Final choice: E6 (known sites) and E7 (unseen sites)**, not plain kriging.
- At unseen sites, kriging alone is not reliably better than the plane-neighbor baseline; the average is.
- At known sites, the average adds a small, consistent gain (better in all 7 years).
- B6/B7 remain the explainable components.
- G7's single region-code feature (a same-week neighbor contrast within the target's sampling region)
  contributes nothing. The strict-public variant E7s is equivalent, so the public-data conclusion stands.

**Interval calibration.** Methods compared year by year over 2019-2023, each year calibrated only on
earlier years' out-of-sample errors:
- **C0:** the models' own intervals.
- **C1:** split-conformal on SD-scaled out-of-sample residuals, by plant regime, recency-weighted.
- **C2:** C1 plus adaptive conformal inference, γ = 0.1.

The selection rule was fixed in advance: lowest interval score among methods with mean coverage in 75-85%.
- C0 covers 80.6% on average at known sites but 64% during the 2020 outage.
- C2 is within 75-85% in all 10 protocol-years and has the best outage coverage.
- **Selected: C0 for known sites (interval score 1.094 vs. 1.100 for C2) and C2 for unseen sites**
  ([fig9](../reports/figures/fig9_interval_calibration.png)).

Calibrating on 2023 alone was not possible: 2023 has no outage samples.

**Not staleness.** A model trained through 2019 and used without retraining for 2020-2022 loses at most
0.023 coverage and 0.004 mg/L RMSE against yearly retraining. The 2024-26 interval shortfall is explained
by volatility (outages, construction), not by model age.

## 8. A fresh test set: prospective registration

The test period cannot judge new choices any more. So the final models (E6/E7, intervals C0/C2) were frozen
on all data through **2026-08-02** and pre-registered: hypotheses, criteria, minimum sample, fingerprints,
and a one-shot evaluation command for the next export. Details:
[prospective_registration.md](prospective_registration.md). A dry run inside development data passes all
structural checks, including bit-identity of the frozen and evaluated models.
