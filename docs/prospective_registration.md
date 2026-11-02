# Prospective test: pre-registration

Machine-readable record: [reports/prospective_registration.json](../reports/prospective_registration.json)
(registered 2026-10-13 UTC, with two logged amendments made before any post-cutoff data exists).
Code: `src/chlorine/modeling/prospective.py`.

## Why

The 2024-26 test period has been evaluated twice (Step 2 references, Step 3 models). Any further model
choice judged on it would be tuned to it. New data cannot be created, but the next lab export will contain
samples taken after the current export. The honest substitute for a fresh test set is therefore:

1. Freeze the final models, their interval calibration and the analysis plan **now**.
2. Evaluate them **once**, on samples taken after the cutoff.

## What is frozen

| Item | Value |
|---|---|
| Cutoff | **2026-08-02** (a Sunday, so no week straddles it; the last v1 sample is 2026-07-31) |
| Training data | all 30,701 v1 modeling samples (2015-2026), 208 places |
| Known-site model | E6 = average(B6 regression kriging with site effects, G6 gradient boosting) |
| Unseen-site model | E7 = average(B7, G7), no site history; refit per fold with the frozen specification |
| Comparators | B5p (known), B4p (unseen): the strongest Step 2 references on 2024-26, a deliberately hard bar |
| 80% intervals | primary C0 (known) and C2 (unseen); the other method is reported as secondary |
| Interval calibration | out-of-sample residual pool 2017-2026 (hash registered); adaptive levels as of the cutoff |
| Model fingerprints | E6 `5e0d331c310c…`, E7 `bc54ce676ff4…` (probe predictions to 1e-6 mg/L) |
| Code fingerprint | hash of every module the frozen models import (syntax tree; docstrings and comments excluded) |

Freezing used the 2024-26 rows as training data and as calibration-pool rows (their out-of-sample
predictions from models trained on earlier years). No accuracy or coverage was computed. The use is logged
in `data/eval/test_access_log.jsonl`.

## Hypotheses and criteria

| | Hypothesis | Supported if |
|---|---|---|
| H1 | At known sites, E6 has lower RMSE than B5p | the 95% cluster-bootstrap CI of RMSE(E6) − RMSE(B5p) lies below 0 |
| H2 | At unseen sites, E7 has lower RMSE than B4p | same criterion |
| H3 | The primary 80% intervals are calibrated | coverage within 75-85% in each protocol |
| H4 | Adaptive intervals help during outages (only if post-cutoff Kubala outages occur) | C2 outage coverage closer to 80% than C0 |

**Also reported** (descriptive, no hypothesis): RMSE with CIs for every model; AUC, recall and alarm rate
for samples < 2.5 mg/L; Kubala-outage and new-site subsets; interval width and interval score.

## When and how

- **When:** once, on the first export with **at least 12 months** of samples after the cutoff (about 2,600
  samples).
- **Why 12 months:** the dry run on six months of development data gave CI half-widths of ±0.012 mg/L (H1)
  and ±0.022 (H2). The expected effects are about −0.040 and −0.019 (backtest). Six months would leave H2
  underpowered; twelve months gives roughly ±0.016, still only moderate power for H2. This is stated in
  advance.
- **Command:** `uv run python -m chlorine.modeling.prospective evaluate --raw NEW.xlsx --confirm`. It:
  1. cleans the new export with the v1 rules;
  2. refits the frozen specification on data up to the cutoff;
  3. predicts every later sample in both protocols;
  4. applies the frozen calibration;
  5. writes `data/eval/prospective/results/`.

## Integrity guards (built into the code)

- **Refuses to run** if:
  - the model-code fingerprint changed;
  - the calibration pool changed;
  - the export has fewer than 12 months after the cutoff;
  - results already exist (one shot).
- **Reports, but does not block:**
  - whether the refit on the new export's history reproduces the registered model fingerprints (a new
    export may revise historical rows);
  - whether the models fitted inside the protocol equal the frozen ones;
  - whether the cleaning code or reference tables changed (new sites may need overrides).
- **Ledger:** the run is logged to the test-access ledger before any metric is computed.

## Dry run

`uv run python -m chlorine.modeling.prospective dry-run` rehearses the whole path inside development data
(pseudo cutoff 2023-07-02, pseudo export ending 2023-12-31; no test-period row is touched). All eight
structural checks pass, including "models fitted inside the protocol are identical to the frozen models".

Two negative controls confirm the checks can fail:
- With a mid-week cutoff, the frozen and protocol-fitted models differ. This is why the cutoff must be a
  Sunday.
- Changing the recency half-life changes the fingerprint.

## Amendments (before any post-cutoff data)

1. H1/H2 wording: B5p/B4p are the strongest references on 2024-26, not the best on 2023 validation (B5/B4).
   Comparators unchanged.
2. E7's description was corrected: it is "no site history", not strictly public-data-only. G7 uses a
   same-region neighbor contrast that needs the utility's region code; a strict-public variant is
   equivalent on development data. The code fingerprint was also redefined as the frozen models' import
   closure, so unrelated new modules do not invalidate it. The models are unchanged: a refit reproduces
   both registered fingerprints.

## What would change the conclusions

- **H1 fails:** the known-site gain does not hold out of time, and the Step 3 test result was optimistic.
- **H2 fails:** the unseen-site gain over the plane-neighbor baseline is too small to matter. Expect this
  with moderate probability even if the effect is real (see power above).
- **H3 fails:** recalibrate with the post-cutoff errors. That would be a new registration, not a fix to
  this one.
