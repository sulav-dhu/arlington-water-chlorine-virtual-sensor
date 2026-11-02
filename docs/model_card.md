# Model card: chlorine-residual virtual sensor (E6 / E7)

Status: **final** for the current project phase; frozen on data through 2026-08-02 and pre-registered for a
prospective test ([prospective_registration.md](prospective_registration.md)). Every number below is quoted
from [reports/key_numbers.md](../reports/key_numbers.md) (IDs in `code`), which is generated from the result
files and checked for consistency.

## 1. What it does

For a routine total-chlorine (chloramine) sample at a place in a given week, the model estimates the residual
in mg/L, with an 80% prediction interval, from:

1. what the **other monitored places measured that same week**;
2. **public location attributes**: pressure plane (from public hydrant data), distance to each plant,
   hydrant age and density, coordinates, Davis-area flag;
3. the **plant regime**: whether the Kubala or Pierce-Burch plant was offline (inferred from gaps in plant
   sampling);
4. **season**;
5. for monitored places only: **the place's own history**.

It is a spatial **nowcast** ("virtual sensor") that fills in places and weeks without a sample. It does not
forecast.

| | Known sites (monitored places) | Unseen sites (unmonitored addresses) |
|---|---|---|
| Model | **E6** = average of B6 (regression kriging with hierarchical site effects) and G6 (gradient boosting) | **E7** = average of B7 (regression kriging) and G7 (gradient boosting), no site history |
| 80% interval | **C0**: the model's own (in-sample calibrated) | **C2**: adaptive conformal on out-of-sample errors |
| Non-public inputs | site history; utility sampling-region code (in G6) | utility region code through one G7 feature; removing it changes nothing (`backtest.unseen_site.E7s_minus_E7` = −0.000 [−0.001, +0.001]) |
| Code | `modeling/backtest.py:e6` | `modeling/backtest.py:e7` |

Model internals (terms, estimation, leakage guards) are described in [modeling.md](modeling.md) §1.

## 2. Final choice: the averaged models, not plain kriging

The question was whether the averages (E6/E7) or plain regression kriging (B6/B7) should be the final
model. It was settled on development data only (rolling backtest 2017-2023: each year predicted by models
trained only on earlier years; [fig10](../reports/figures/fig10_backtest_by_year.png)):

| Pooled 2017-2023 RMSE difference [95% CI, places resampled] | Known sites | Unseen sites |
|---|---|---|
| Final − best baseline | E6 − B5p **−0.040** [−0.048, −0.032] | E7 − B4p **−0.019** [−0.031, −0.008] |
| Final − kriging alone | E6 − B6 **−0.004** [−0.007, −0.001] | E7 − B7 **−0.011** [−0.021, −0.004] |
| Kriging alone − best baseline | B6 − B5p −0.036 [−0.042, −0.029] | B7 − B4p −0.008 [−0.018, **+0.002**] |
| Years final beats kriging alone / best baseline | 7 of 7 / 7 of 7 | 7 of 7 / 7 of 7 |

Decision: **E6 for known sites, E7 for unseen sites.** This agrees with the Step 3 selection rule (lowest
validation RMSE), which picked the same models before the test run.

- At unseen sites, plain kriging is **not** reliably better than the simple plane-neighbor baseline (its CI
  includes 0). The average is what makes the gain robust.
- At known sites, kriging already carries almost all the gain; averaging adds a small but consistent ~1%.
- **Plain kriging stays in the system** as the explainable component. Its prediction decomposes into city
  level + public fixed effects + site level + local anomaly (`RegressionKriging._components`), which is how
  individual predictions should be explained to operators. It is also the fallback if the boosting member
  cannot be run.

What the average costs: less one-to-one interpretability; a second model to maintain; its interval is the
average of the members' quantiles (a pragmatic choice, not a proper mixture).

## 3. Performance

RMSE in mg/L. The test period was evaluated **once** for these models, before this phase (Step 3).

| | Validation 2023 | Test 2024-26 | Backtest 2017-2023 (yearly range) |
|---|---|---|---|
| Known sites, E6 | **0.272** (B5 0.304; Δ −0.032 [−0.045, −0.020]) | **0.310** (B5 0.352; Δ −0.042 [−0.055, −0.029]) | 0.272-0.350 |
| Unseen sites, E7 | **0.308** (B4 0.316; Δ −0.008 [−0.024, +0.004]) | **0.352** (B4 0.368; Δ −0.016 [−0.030, −0.001]) | 0.308-0.425 |
| Kubala outages, E6 (pre-2024, leave-one-out) | **0.356** (B5p 0.405; Δ −0.049 [−0.068, −0.032]) | 0.363 vs. B5p 0.412 (658 samples) | |

- **Floors:** within-week scatter at near-daily sites is 0.28 mg/L (`noise.within_week_sd`), and replicate
  measurements differ by 0.14 (`noise.measurement_sd`). A weekly model cannot predict a single sample much
  better than about 0.28.
- **Ranking low samples works; a fixed alarm rule is conservative.** AUC for samples < 2.5 mg/L is 0.93 (E6,
  test) and 0.88 (E7, test). The rule "lower 10% bound < 2.5" catches 47% of such samples on test while
  flagging 2% (base rate 1.8%; `final.test.known_site.E6_alarm_lt25`). During the pre-2024 Kubala outages
  (leave-one-out) it catches 95% while flagging 41% (base rate 23.5%). Use the risk ranking, and set the alarm threshold to the response capacity.
- **New places:** at places first sampled after 2023, E6 reaches 0.309 against B5p's 0.365 (817 test
  samples).

## 4. Prediction intervals

**Problem:** on test 2024-26, the original 80% intervals covered 73-76% (`final.test.coverage_range`).

**Study (development data only).** Three methods were compared in a rolling design over 2019-2023. Each year
was calibrated only on the out-of-sample errors of earlier years (from the backtest), and the selection rule
was fixed before the study ran: lowest interval score among methods whose mean coverage lies in 75-85%.

- **C0:** the model's own interval (in-sample residuals).
- **C1:** split-conformal: recency-weighted quantiles of out-of-sample residuals scaled by the model's SD,
  separately for Kubala-outage and normal weeks.
- **C2:** C1 plus adaptive conformal inference (Gibbs & Candès 2021, γ = 0.1). The tail levels move weekly
  with observed misses; at unseen sites the feedback is E7's misses at monitored sites.

Calibrating on 2023 alone, as first proposed, could not work: 2023 contains no Kubala-outage samples, so it
cannot teach outage behavior, and it gives a single coverage number. The rolling design keeps the same
discipline (out-of-sample errors only, nothing from 2024-26) with five years and four outages.

| Mean over 2019-2023 | C0 | C1 | C2 |
|---|---|---|---|
| Known sites: coverage (yearly range; years in 75-85%) | 80.6% (77-85%; 4/5) | 81.4% (78-86%; 4/5) | 80.4% (78-82%; **5/5**) |
| Known sites: coverage in outage weeks | 77.4% | 81.9% | **83.5%** |
| Known sites: interval score | **1.094** | 1.108 | 1.100 |
| Unseen sites: coverage (yearly range; years in 75-85%) | 78.7% (74-83%; 4/5) | 81.7% (77-87%; 4/5) | 77.9% (75-79%; **5/5**) |
| Unseen sites: coverage in outage weeks | 74.5% | 82.4% | 81.8% |
| Unseen sites: interval score | 1.299 | 1.305 | **1.292** |

Findings ([fig9](../reports/figures/fig9_interval_calibration.png)):

1. **The original intervals are calibrated on average but not in volatile periods.** C0 averages 80.6% at
   known sites but drops to 64% during the 2020 outage. The 2024-26 shortfall matches that pattern: those
   years had two Kubala outages and Pierce-Burch construction.
2. **Model staleness is not the cause.** One model trained through 2019 and used for 2020-2022 without
   retraining loses at most 0.023 coverage and 0.004 mg/L RMSE against yearly retraining
   (`cal.horizon_max_coverage_gap`, `cal.horizon_max_rmse_gap`).
3. **Adaptive intervals (C2) are the robust fix.** They stay within 75-85% in every year for both
   protocols, and raise outage coverage (2020 known sites: 64% → 74%). The worst outage is still
   under-covered.
4. **Selected by the pre-declared rule:** C2 for unseen sites and C0 for known sites. At known sites, C0's
   interval score is marginally better (1.094 vs. 1.100). C2 is registered as the alternative, and the
   prospective test decides (H3, H4).

The corrected intervals have **not** been checked on 2024-26, by design: that would be a third use of the
test set. Their first independent check is the prospective test.

## 5. Intended use

**Use it for:**
- Screening between samples: which places and weeks are probably lowest.
- Prioritizing extra sampling or flushing checks.
- Estimating residual at addresses without a sampling site, inside the footprint of the monitored network.
- Showing how the system responds when a plant goes offline.

**Do not use it for:**
- Regulatory compliance, or replacing any required sample: decisions about compliance need measured values.
- Real-time alarms without same-week network samples. With only the prior 7 days (the causal setting), the
  best reference is +0.04-0.05 mg/L worse ([evaluation.md](evaluation.md) §5).
- Forecasting future weeks.
- Locations outside the service area or far from every monitored place.
- Disinfection-byproduct or microbial risk.

## 6. Limits

1. **Needs the network.** Predictions rely on other places' samples from the same week. In weeks with few
   samples, the model falls back to plane and climatology terms and its intervals widen.
2. **Plant status is inferred, not logged.** Kubala outages come from gaps in plant sampling: 7 inferred,
   6 with a city-wide dip (`outage.kubala_with_dip`). Pierce-Burch flags are weaker. An unrecognized event
   degrades accuracy and coverage. An operations log is the single most valuable addition.
3. **Unseen sites lack hydraulics.** Public data has no pipe diameters, flow paths or water age. At unseen
   sites, E7 trails E6 at known sites by about 0.04 mg/L (test 0.352 vs. 0.310).
4. **Unseen-site errors may be optimistic for atypical addresses.** Held-out places are still official
   sampling sites, chosen to be representative. Dead ends and low-demand mains are probably
   under-represented. In the protocol, held-out places also keep their utility region code; at a new
   address it must be inferred from the nearest site, or E7s (no region code, equally accurate) used.
5. **Intervals:** see §4. They remain least reliable in the first weeks of a plant outage.
6. **Drift.** Site levels and plant operations change (Pierce-Burch construction 2021-2026). Static offsets
   learned on old data stopped helping in 2024-26 ([evaluation.md](evaluation.md) §5, finding 5). Retrain
   yearly; recency weighting (half-life 365 days) is already built in.
7. **Weekly resolution.** The model cannot resolve within-week variation (0.28 mg/L SD) or single-sample
   measurement noise (0.14).
8. **Evaluation reuse.** 2024-26 was evaluated twice (Step 2 references, Step 3 models). It was then used
   once more, as training and calibration data for the frozen model, with no metrics computed. All uses
   are logged in `data/eval/test_access_log.jsonl`. Any further model choice must be judged on data after
   2026-08-02.
9. **Tuning.** Only the recency half-life was tuned; the boosting settings are fixed defaults.

## 7. Data, privacy and security

- **Training data:**
  - 30,701 routine samples at 208 places (221 sampling locations), Jan 2015 - Jul 2026, from the
    utility's lab export.
  - Public City of Arlington GIS layers (address points, hydrants, road centerlines).
- **Personal data:** customer names and phone numbers appear in the raw export (237 rows contain a phone
  number) and are never written to any output. Analysts are pseudonymized. Geocoding runs locally; no
  address leaves the machine.
- **Sharing:** outputs at address level could reveal where the system is weakest, so share aggregates only.
  `data/` and the raw export are git-ignored.

## 8. Reproduce and maintain

```bash
uv run python -m chlorine.modeling.backtest                 # rolling backtest 2017-2023
uv run python -m chlorine.modeling.backtest --members       # B6/G6/B7/G7 alone (final-choice evidence)
uv run python -m chlorine.modeling.sensitivity              # E7 without the region code
uv run python -m chlorine.modeling.calibration              # interval study + selection
uv run python -m chlorine.reporting.figures                 # fig6-fig10
uv run python -m chlorine.reporting.numbers                 # key-numbers sheet + consistency checks
uv run python -m chlorine.modeling.prospective dry-run      # rehearsal of the prospective test
```

- **Yearly:** retrain on all data, and extend the calibration pool with that year's out-of-sample
  predictions.
- **Any change to model code:** the code fingerprint in the registration changes and the prospective test
  refuses to run. Changes belong in a new registration.
