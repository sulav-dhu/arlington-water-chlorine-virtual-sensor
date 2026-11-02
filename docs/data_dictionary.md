# Data dictionary — cleaning v1

All files are in `data/processed/v1/`. Join keys: `row_id` (one raw export row) and `location_id`.

## chlorine_v1.parquet — modeling table

One row per in-scope routine total-chlorine sample (unique on `location_id`, `sampled_at`).

| Column | Type | Meaning |
|---|---|---|
| `row_id` | int | Row of the raw export (0-based, file order) |
| `location_id` | str | Physical location (`W045`; `SE057.2` when a code was re-used at a new address) |
| `site_code` | str | Utility site code of the location |
| `place_group` | str | Locations at the same physical place share it. **Group CV folds by this** |
| `region` | str | Prefix of the site code (C, E, N, SE, SW, W): the utility's sampling region, **not** the pressure plane |
| `pressure_plane` | str | UPPER / LOWER / WEST from the 7 nearest public hydrants (today's boundaries) |
| `high_frequency_site` | bool | ≥ 4 in-scope samples/month (the near-daily sites: W045, SE035, W065, W066, SW049, SW050, SW051) |
| `sampled_at` | datetime | Sample time (authoritative) |
| `sample_date`, `week_start` | datetime | Day; Monday of the week |
| `value_mg_l` | float | Total chlorine, mg/L. Empty when censored |
| `censored` | bool | Non-detect (below `detection_limit`) |
| `detection_limit` | float | Reporting limit for censored rows (0.08 mg/L) |
| `decimals_observed` | int | Decimals in the reported value; a lower bound (Excel drops trailing zeros) |
| `project` | str | `Monthly Bact` or `Monthly Cl` |
| `analyst_id` | str | Pseudonymized analyst (A01…); useful as a measurement-bias random effect |
| `lims_status` | str | LIMS workflow status |
| `loc_method` | str | How the location was assigned: `alias_address`, `code_only`, `alias_variant`, `address_named_link`, `reassigned_by_address` |
| `loc_confidence` | str | `high` / `medium` (conflicts are excluded) |
| `address_offset` | float | For address-named links: house-number distance to the site (0 = same address) |
| `dup_conflict` | bool | Another program measured the same water at the same time with a different result |
| `dup_value_range` | float | Spread between those results (mg/L) |
| `flag_sampled_time_missing` | bool | Time recorded as 00:00 |
| `flag_analyzed_before_sampled` | bool | Analysis time > 1 h before sample time |
| `jk_offline`, `pb_offline` | bool | Inside an inferred plant-offline period. From plant sampling dates only; **safe as features** |
| `jk_offline_confidence`, `pb_offline_confidence` | str | Uses the chlorine dip (the outcome). **Never use as a feature** |
| `plant_regime` | str | `both_online`, `jk_offline`, `pb_offline`, `both_offline` (implausible; sampling lapse) |

## locations.csv — one row per location

| Column | Meaning |
|---|---|
| `location_id`, `location_type` | `routine`, `tank`, `plant`, `nap_site` (NAP-only address), `adhoc` (construction/complaint/etc. address) |
| `site_code`, `facility`, `point_type` | Code for routine sites; facility name and point type (finished/raw/lab_tap/process/elevated) for plants and tanks |
| `address` | Normalized street address (sensitive: keep private) |
| `resolution_role` | `anchor`, `reuse`, `code_only`, `facility`, `address_named` |
| `place_group` | Same-physical-place group |
| `lon`, `lat` | WGS84 coordinates (local geocode against public city layers) |
| `geocode_method` | `address_point`, `street_range`, `address_point_nearby`, `*_fuzzy_street`, `*_region_disambiguated`, `manual_override`, `ambiguous_direction`, `not_found`, `no_address` |
| `geocode_match` | Street core matched (or override reason) |
| `geocode_uncertainty_m` | ~10 m for address points; half the street segment for interpolation; 10 m per house number for nearby points |
| `region` | Site-code prefix |
| `pressure_plane`, `plane_vote_share` | Plane of the 7 nearest hydrants and the share agreeing |
| `nearest_hydrant_m` | Distance to the nearest public hydrant |
| `hydrants_in_radius` | Hydrants within 500 m (network density proxy) |
| `flush_program_share` | Share of hydrants within 500 m with FlushProgram M/A (meaning undocumented; probably flushing program) |
| `hydrant_median_year` | Median acceptance year of hydrants within 500 m (infrastructure age proxy) |
| `geocode_suspicious` | Nearest hydrant > 300 m away |
| `dist_jk_km`, `dist_pb_km` | Straight-line distance to the Kubala / Pierce-Burch plant |
| `in_davis_phase2_area` | Inside the Davis Phase II upper-plane expansion box (plane may have changed ~Oct 2024) |
| `n_rows_all`, `n_chlorine_in_scope` | Rows of any analyte; in-scope chlorine samples |
| `first_in_scope`, `last_in_scope`, `in_scope_samples_per_month`, `high_frequency_site` | Sampling history |

## events.csv

Inferred plant-offline periods (`source = inferred_from_plant_sampling_gaps`): `event_id`, `plant`,
`start`, `end` (inclusive), `days`, `boundary` (gap touches the start/end of the data),
`city_dip_mg_l` (city-wide residual inside minus ±56-day flanks), `confidence`
(`externally_confirmed`, `supported_by_dip`, `ambiguous`, `inferred`, `ambiguous_boundary`),
`external_event`. Documented events (`source = external_document`) add `event_type`, `date_precision`,
`description`, `source_url`.

## measurements.parquet — every export row

Raw fields renamed to snake_case (`wrk`, `sample_seq`, `client`, `project`, `analysis`, `analyte`,
`units`, `sampled_at`, `analyzed_at`, `received_at`, `status_date`, `mrl`, `lims_status`,
`result_text`), plus `sample_name` (normalized; non-address complaint names redacted),
`sample_class` (`routine`, `facility`, `address_named`, `other`, `unknown`), all location columns
above, parsed `value` / `result_status` (`detected`, `nd`, `not_reportable`, `unparsed`) / `censored`
/ `detection_limit`, duplicate columns (`dup_group`, `is_duplicate`, `dup_conflict`,
`dup_value_range`), flags (`flag_out_of_range`, `flag_analyzed_before_sampled`,
`flag_received_before_sampled`, `flag_sampled_time_missing`), `exclusion_reason` (total-chlorine
rows only: first failed scope rule) and `in_scope_v1`.

Not included anywhere: raw `SampleAlias` text and analyst initials.

## location_review.csv

One row per (site code, address cluster): `role` (`anchor`, `variant`, `reuse`, `secondary`,
`conflict`, `miscode`, `unresolved`), `address`, `n_rows`, `first`, `last`, resolved `location_id`,
`conflict_with`, number of `spellings`.
