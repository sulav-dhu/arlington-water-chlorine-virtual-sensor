# Cleaning v1 — rules, decisions and rationale

This is the specification for `chlorine.cleaning.pipeline`. Numbers below are from the v1 build
(raw export sha256 `7d2377bb…`); `data/processed/v1/dq_report.md` has the full, regenerated figures.

## 1. Scope (what the modeling table contains)

| Decision | Choice | Why |
|---|---|---|
| Target | Total chlorine, field test (`Analysis = Chlres-field`), mg/L | The system is chloraminated; total chlorine is the compliance measure and the only analyte sampled everywhere |
| Programs | `Monthly Bact` + `Monthly Cl` | Routine distribution monitoring. Construction, complaints, CIP and special projects are biased or non-representative; NAP, tower and plant samples are kept in `measurements` for later use |
| Locations | Routine distribution sites only (codes like W045, plus address-named samples linked to them) | Plants, tanks and ad-hoc addresses are different populations |
| Non-detects | Kept, `censored = True`, value empty, `detection_limit = 0.08` | Treat as left-censored, never as zero (2 in-scope rows) |

Funnel: 37,250 total-chlorine rows → 6,414 outside scope projects → 83 not at a routine location →
31 location conflicts → 5 not reportable → 14 duplicate copies → **30,703 rows, 221 locations,
208 physical places**.

## 2. Location resolution

The site code in the export is not a reliable location key. Rules, in order:

1. **Normalize codes.** `W45` → `W045`; `NO39` → `N039` (letter O typed for zero).
2. **Overrides** (`config/site_overrides.csv`): W042 and W052 are the Pierce-Burch lab tap, N034 and
   SW031 are tanks.
3. **Cluster each code's alias addresses into physical places.** Two addresses are the same place
   when the street names match (after normalization, similarity ≥ 0.85) and house numbers are within
   30 (a few lots, ~50-150 m). Normalization handles formatting, suffix and ordinal variants
   (`2nd` ↔ `SECOND`, `PWY` ↔ `PKWY`), squashed names (`1234MAPLEGROVEDR`) and typos.
   **Written directionals are respected**: the same house number on `W Pioneer Pkwy` and
   `E Pioneer Pkwy` is miles apart, so they are never merged.
4. **Classify clusters.** The largest cluster is the code's anchor. The others are:
   - `conflict`: another code's anchor (or a plant address). The rows are kept but excluded from
     modeling (31 rows, 13 clusters). Example: 19 daily samples in March 2024 coded SW049 but
     addressed to W066's location, right after W066's daily sampling stopped. Resolve in
     `site_overrides.csv` if field records show which is right.
   - `secondary`: ≥ 5 rows elsewhere; a second real location for the code (none in v1).
   - `reuse`: a small cluster ≥ 180 days outside the code's active period; the code was re-used
     (SE057 in 2018 vs. 2023+; E073). These become `SE057.1`, `SE057.2`.
   - `variant`: anything else; a typo (e.g. "120 Main St" for 1200 Main St). Assigned to the code.
5. **Miscodes.** A code with < 5 rows whose anchor is another code's anchor is reassigned (W005 →
   W045; SW057's Red Ridge rows → SW060).
6. **No alias address** → the code's location active at that date.
7. **Address-named samples** (NAP sites, program samples logged by address) are linked to the
   closest matching routine/facility place; otherwise they become their own `nap_site` or `adhoc`
   location. 137 in-scope rows come from such links (e.g. 28 `Monthly Cl` rows logged
   under W045's street address instead of its code). Neighbor-house repeat samples carry `address_offset > 0` and
   `loc_confidence = medium`.
8. **Place groups.** Locations at the same physical place share a `place_group` (11 groups, e.g.
   W065 and W066, two house numbers apart; N031 → N039 two doors apart). Many such codes start
   in April 2016, when the Revised Total Coliform Rule took effect, which suggests a sampling-plan
   revision. **Use `place_group` for spatial cross-validation folds**, or the same place leaks across
   folds.

Every cluster and its role is listed in `location_review.csv`.

## 3. Duplicates

Rows sharing (location, sample time, analysis, analyte) are duplicates; in-plant sub-points
(`PLANT-*-OTHER`) are keyed by sample name. One copy is kept (scope project first, then `Reported`
status, then lowest row id). When copies disagree (`dup_conflict`), they are almost always two
programs measuring the same water at once (e.g. Monthly Bact 2.6 vs. Weekly NAP 2.59), so the
preferred copy is kept and `dup_value_range` records the spread.

**Measurement repeatability** (routine locations): of 473 same-moment pairs, 96% are identical (one
reading entered twice); the 19 separate readings differ by median 0.10, 90th percentile 0.28,
max 0.54 mg/L. Use this as the measurement-noise floor.

## 4. Values and time

- Results are parsed to `detected | nd | not_reportable | unparsed`; values outside plausible ranges
  are flagged, never silently dropped. No in-scope chlorine value is out of range; the 13 flagged
  chlorine values (6.2-8.8 mg/L) are all outside the modeling scope (construction, special-project,
  plant and one complaint sample).
- `decimals_observed` is a lower bound: Excel drops trailing zeros (3.80 → 3.8).
- The sample time is authoritative. 822 rows have an analysis time > 1 h before sampling and 539
  have time 00:00; both are flagged.

## 5. Geocoding and public-GIS features

All geocoding is local against public City of Arlington layers (`scripts/fetch_public_gis.sh`).
Order: exact address point → street-centerline range interpolation → nearest house number (±40)
→ the same after correcting the street name to the closest known one. If a written directional is
missing and matches are > 300 m apart, the result is `ambiguous_direction`; for routine sites it is
resolved to the candidate inside the region's envelope (E/W region codes agree with street
directionals in the data), otherwise left for `config/geocode_overrides.csv`.

Coverage: 99.9% of in-scope samples geocoded (SW023 is ambiguous between Eden Rd and S Eden Rd;
W067 has no address). `geocode_uncertainty_m` is ~10 m for address points and half the segment length
for interpolation; two rural sites (SW018, SW024) are > 300 m from any hydrant and have large
uncertainty.

Features from the public hydrant layer (an export of the utility's network GIS):
- `pressure_plane`: majority of the 7 nearest hydrants (median agreement 100%). **The region code
  (W/E/N/SE/SW/C) is not the pressure plane**: 35 of 42 W-coded sites are in the LOWER plane.
- `flush_program_share`, `hydrant_median_year`, `hydrants_in_radius` within 500 m.
- `dist_jk_km`, `dist_pb_km`: straight-line distance to the Kubala / Pierce-Burch plants.
- `in_davis_phase2_area`: inside the Davis Phase II upper-plane expansion (Cooper–Fielder–Arkansas–
  Sanford, completed ~Oct 2024); 89% of today's hydrants in that box are UPPER although the area lies
  in the otherwise LOWER north. The plane of these 7 sites may have changed around 2024.

## 6. Plant operating state

There is no operations log. A gap of ≥ 28 days with no finished-water sample (any analyte) at a plant
is treated as the plant being offline (`events.csv`). Raw-water samples do not count: Lake Arlington
raw water was sampled weekly straight through PB's 2022-2026 winter shutdowns, and counting it hid
every PB outage after mid-2022.
- **Kubala (JK):** 7 periods. Six coincide with city-wide dips of 0.3-0.8 mg/L; Feb 2017 matches a city
  notice. Nov 2021–Mar 2022 shows no dip (`ambiguous`).
- **Pierce-Burch (PB):** periods nearly every winter, consistent with its documented winter idling; no
  dip expected (JK carries the city).
- `jk_offline` / `pb_offline` use plant sampling dates only and are safe model features.
  `*_confidence` uses the chlorine dip, i.e. the outcome: reporting only, **never a feature**.
- 205 in-scope rows (0.67%) fall in overlapping JK and PB gaps, which is physically implausible: at
  least one is a sampling lapse.
- When PB is offline (JK carries the city) the city-wide residual is usually *higher*: in 13 of 14
  non-boundary PB periods, median +0.23 mg/L. That is the opposite of a JK outage, so plant regime
  moves chlorine in both directions.

## 7. Privacy

Raw `SampleAlias` is never written (in complaint projects it holds customer names/phones); phone
numbers are stripped before address parsing; complaint sample names that are not addresses are
redacted; analysts are pseudonymized (key in `private/`). The pipeline aborts if any output contains
a phone-number pattern. Outputs still contain addresses and coordinates: keep `data/` private.

## 8. Open items for the team

1. Resolve the 13 cross-code conflicts (`location_review.csv`, role `conflict`) from field records.
2. Add coordinates for SW023 and the 8 tanks without an address (`config/geocode_overrides.csv`).
3. Confirm plant outages with operations staff if possible; especially Nov 2021–Mar 2022.
4. Pressure-plane membership before 2024 for the Davis area, and the Phase I (2018) area.
5. Meaning of hydrant `FlushProgram` codes `M` and `A`.

## 9. Changing the rules

All parameters are in `src/chlorine/config.py`; reference data in `config/`. After a change:
rebuild, read `dq_report.md`, and update the golden numbers in `tests/test_outputs_v1.py`
deliberately. For a new export, bump `version` and `expected_raw_sha256`.
