# Evaluation (Step 2) — protocols, reference models and results

> Step 3 models evaluated with these protocols: see [modeling.md](modeling.md).

Code: `src/chlorine/evaluation/`. This document is the evaluation specification: every new model is
judged by these rules, against these reference numbers.

```bash
uv run python -m chlorine.evaluation.run --learning-curve          # validation (default)
uv run python -m chlorine.evaluation.run --split test --confirm-test   # test, logged
uv run python -m chlorine.evaluation.figures                       # reports/figures/*.png (+ CSV)
```

Outputs: `data/eval/<split>/` → `evaluation_report.md`, `results.csv` (metrics per protocol ×
subset × model), `predictions.parquet` (one row per model × sample × protocol, for error analysis),
`learning_curve.csv`, `run.json`.

## 1. The task

Estimate the total chlorine (mg/L) of a routine sample from **what the other monitored places
measured** and **where the sample was taken**. This is a spatial nowcast ("virtual sensor"):

| Information | Allowed? |
|---|---|
| Other places' samples in the same calendar week (`wk_*` features) | yes: the primary setting |
| Other places' samples in the 7 days strictly before (`t7_*`) | yes: the causal/real-time setting |
| Location attributes (pressure plane, region, coordinates, hydrant features) | yes |
| Plant on/off flags from plant sampling gaps (`jk_offline`, `pb_offline`) | yes |
| The place's own earlier samples (`own_prev_*`) | only for models marked *site history*, known sites only |
| The sample's own value, anything from its own place in the same week | **never** |
| `*_offline_confidence` (uses the outcome) | **never** |

Unit: one sample (30,701 in-scope samples; the 2 non-detects are not evaluation targets). Neighbor
unit: the `place_group`, so a renumbered site is one place, never its own neighbor, and always in a
single fold. Neighbor means average each place's samples first, so a near-daily site counts once.

## 2. Splits and protocols

Periods: **train** 2015 – 2022, **validation** 2023, **test** 2024 – Aug 2026. Validation models train
on the train period; test models are refit on train + validation.

| Protocol | Question | Design |
|---|---|---|
| `known_site` | How well can we fill in a monitored site's value? | Predict evaluation-period samples at all places. Places with no training history (5 in 2023, 24 in 2024-26) are reported as `new_sites` |
| `unseen_site` | How well can we estimate chlorine where nobody samples? | 5-fold cross-validation over place groups, stratified by pressure plane and balanced by size. Held-out places are removed from the neighbor pool, so they behave as unmonitored |
| `outage` | Does the model hold up when a plant goes offline? | Leave-one-outage-out over the 5 Kubala outages before 2024 (2023 has none) |

**Test-set policy.** Development uses validation only; with `--split val` the test period is deleted
before anything else runs. `--split test` requires `--confirm-test` and appends to
`data/eval/test_access_log.jsonl`. The reference models were run on test exactly once (2026-10-12),
after all their settings were fixed. **Do not use test results to choose between new models.**

**Leakage is tested, not assumed** (`tests/test_evaluation.py`): setting every value of a fold's
held-out places to 1000 leaves their predictions exactly unchanged; a sample's own value and the
whole test period never change a validation prediction; neighbor means match a brute-force recount.

## 3. Metrics

RMSE (primary), MAE, bias, R², RMSE with every place weighted equally; pinball loss, 80% interval
coverage and width; AUC for detecting samples < 2.5 and < 3.0 mg/L, recall and alarm rate of
"q10 < threshold"; Spearman correlation of observed vs. predicted site means. 95% intervals come
from a **cluster bootstrap over places** (1,000 draws), because samples from one place are
correlated; ΔRMSE intervals are paired against B1.

**Noise floor.** Separate same-moment readings by two programs differ with SD ≈ 0.14 mg/L (18
pairs). At the near-daily sites, samples scatter around their own weekly mean with SD ≈ 0.28 mg/L:
the error of a model that knew each place's weekly mean exactly. Weekly-resolution models cannot
get much below it.

## 4. Reference models

Transparent "neighbor estimate + learned offset" predictors with settings fixed in `EvalConfig`
(not tuned). Offsets are empirical-Bayes shrunk toward their parent (site → plane/region → all);
prediction intervals scale training residuals by the place's or plane's residual spread.

| Code | Model | Site history |
|---|---|---|
| B0 / B0c | overall mean / monthly climatology | |
| B0p | previous sample at the same place | ✓ |
| B1 | city-wide mean of other places this week | |
| B2 / B2p / B3 | B1 + region / pressure-plane / site offset | B3 ✓ |
| B4 | same-week mean of same-region places, blended toward B2 by neighbor count | |
| B4p | same-week distance-weighted mean of same-plane places (public data), blended toward B2p | |
| B5 / B5p | B4 / B4p + site offset | ✓ |
| B5r | B3 + site offsets specific to Kubala / Pierce-Burch outages | ✓ |
| B1c / B4pc | B1 / B4p with the causal 7-day setting | |

## 5. Results (RMSE, mg/L; 95% intervals in the reports)

| Model | Known 2023 | Unseen 2023 | Outages (pre-2024) | Known 2024-26 | Unseen 2024-26 |
|---|---|---|---|---|---|
| B0 overall mean | 0.455 | 0.456 | 0.852 | 0.487 | 0.487 |
| B1 city-wide weekly mean | 0.394 | 0.393 | 0.528 | 0.397 | 0.396 |
| B2 + region offset | 0.349 | 0.352 | 0.518 | 0.395 | 0.399 |
| B4 region neighbors | 0.312 | **0.316** | 0.456 | 0.362 | 0.368 |
| B4p plane neighbors (public) | 0.319 | 0.325 | 0.441 | 0.353 | **0.358** |
| B5 region neighbors + site | **0.304** | — | 0.408 | 0.352 | — |
| B5p plane neighbors + site (public) | 0.308 | — | **0.405** | **0.343** | — |
| B1c / B4pc causal 7-day | 0.415 / 0.359 | 0.416 / 0.366 | 0.575 / 0.550 | 0.423 / 0.406 | 0.423 / 0.409 |

Findings:

1. **Same-week neighbors carry most of the signal.** Going from the overall mean to the best
   neighbor model cuts RMSE by 30-33% (0.455 → 0.304 in 2023, 0.487 → 0.343 in 2024-26). The causal
   7-day setting is consistently worse (B4pc vs. B4p: +0.04 to +0.05, and +0.11 during outages), so the
   same week's network readings carry real information.
2. **Almost all of the gain survives at unmonitored places.** For the neighbor models, unseen-site RMSE
   is within 0.006 of known-site RMSE.
3. **Little room is left at weekly resolution.** At the near-daily sites the best known-site model
   (0.326 in 2023) sits only ~0.17 mg/L RMS above the 0.28 within-week floor.
4. **Public pressure-plane geography matches the utility's region codes.** No difference in calm
   2023 (B4p − B4 = +0.008, 95% CI −0.004 to +0.020); ahead in 2024-26 overall (−0.010, CI −0.021 to
   −0.00004: borderline). Larger gaps during outages (−0.018), at new sites (−0.043) and in the West
   plane (−0.038) are suggestive but not individually significant.
5. **Static offsets drift.** Region offsets learned on 2015-2022 help in 2023 (B2 0.349 vs. B1 0.394)
   but not in 2024-26 (0.395 vs. 0.397); site levels and plant operations changed.
6. **Outages are the hard case** (best 0.405 vs. 0.304 in calm times). Site-specific outage offsets
   (B5r) beat plain site offsets (B3; 0.439 vs. 0.477), but same-week neighbors do better still.
7. **More sampling sites help with diminishing returns for region models** (B4 unseen: 0.364 with 36
   monitored places → 0.316 with 143); the distance-weighted public model gains most from density.
8. **Intervals are miscalibrated by period**: too wide in calm 2023 (coverage 0.84-0.89 for a nominal
   0.80), too narrow during outages (0.71-0.75). Variability itself changes over time.

## 6. Adding a model

Subclass `evaluation.models.Model`: implement `_fit(train)` and `_point(df)` (return one value per
row, never NaN), set `uses_site_history` honestly, and add it to the model list passed to the
protocols. Training frames contain every feature column described in `evaluation/features.py`; the
shared calibrator turns point predictions into quantiles unless you override `predict`. Develop on
`--split val`; compare against the table above; run the test split once, at the end.

## 7. Limitations

- Plant on/off flags are inferred from sampling gaps (see `docs/cleaning_v1.md`).
- In `unseen_site`, held-out places keep their utility region code; a genuinely new address would
  need it inferred from location (public pressure plane does not have this problem).
- The outage protocol trains on data before and after each outage: it measures spatial response to
  an outage, not forecasting.
- 2023 is unusually calm; validation numbers are optimistic relative to 2024-26.
- The noise floor is measured at only 7 near-daily sites and 18 replicate pairs.
