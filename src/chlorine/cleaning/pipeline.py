"""Step 1 cleaning pipeline: raw LIMS export -> versioned analysis-ready tables.

    uv run python -m chlorine.cleaning.pipeline [--raw PATH] [--version v1]

Outputs (data/processed/<version>/, never committed):
  chlorine_v1.parquet     modeling table: in-scope routine total-chlorine samples, one row each
  measurements.parquet    every row of the export, cleaned, with flags and exclusion reasons
  locations.csv           one row per physical location, with geocode and public-GIS features
  events.csv              inferred plant offline periods + documented external events
  location_review.csv     per-code address clusters and how each was resolved (for team review)
  dq_report.md            data-quality report: funnel, checks, issues, open items
  manifest.json           input hash, parameters, row counts, library versions
  private/analyst_key.csv analyst initials <-> pseudonym (keep private)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CleaningConfig
from . import dedupe, events, geo, report
from .addresses import contains_phone, parse_address
from .load import load_raw, sha256
from .locations import LocationResolver, classify_sample, load_facilities, normalize_sample_name
from .privacy import COMPLAINT_PROJECTS, find_phone_numbers, normalize_analyst, pseudonymize_analysts
from .values import TOTAL_CHLORINE, parse_results

LOC_COLS = ["location_id", "loc_method", "loc_confidence", "loc_conflict_with", "address_offset"]


def resolve_locations(m: pd.DataFrame, cfg: CleaningConfig, resolver: LocationResolver) -> pd.DataFrame:
    overrides = pd.read_csv(cfg.reference_dir / "site_overrides.csv", dtype=str)
    override_map = dict(zip(overrides.site_code, overrides.target_location_id))
    facilities = list(resolver.facilities.values())

    keys = m[["sample_name", "Project", "SampleAlias"]].drop_duplicates()
    cls = {}
    for sn, proj, alias in keys.itertuples(index=False):
        cls[(sn, proj, alias)] = classify_sample(None if pd.isna(sn) else sn, proj, None if pd.isna(alias) else alias,
                                                 facilities, override_map)
    k = list(zip(m.sample_name, m.Project, m.SampleAlias))
    m["sample_class"] = [cls[x][0] for x in k]
    m["site_code"] = pd.Series([cls[x][1] for x in k], index=m.index, dtype="string")
    facility_id = pd.Series([cls[x][2] for x in k], index=m.index, dtype=object)
    named_address = pd.Series([cls[x][3] for x in k], index=m.index, dtype=object)

    for c in LOC_COLS:
        m[c] = pd.Series(None, index=m.index, dtype=object)

    routine = m.sample_class.eq("routine")
    usable_alias = routine & ~m.Project.isin(COMPLAINT_PROJECTS) & ~m.SampleAlias.map(contains_phone)
    alias_addr = pd.Series(None, index=m.index, dtype=object)
    alias_addr[usable_alias] = m.loc[usable_alias, "SampleAlias"].map(
        lambda a: None if pd.isna(a) else parse_address(a))
    r = resolver.resolve_routine(pd.DataFrame({"site_code": m.loc[routine, "site_code"],
                                               "sampled_at": m.loc[routine, "Sampled"],
                                               "alias_address": alias_addr[routine]}))
    m.loc[routine, r.columns] = r.values

    resolver.add_facilities()
    fac = m.sample_class.eq("facility")
    m.loc[fac, "location_id"] = facility_id[fac]
    m.loc[fac, "loc_method"] = np.where(m.loc[fac, "site_code"].notna(), "site_override", "facility_name")
    m.loc[fac, "loc_confidence"] = "high"

    named = m.sample_class.eq("address_named")
    a = resolver.resolve_address_named(pd.DataFrame({"address": named_address[named], "sampled_at": m.loc[named, "Sampled"],
                                                     "project": m.loc[named, "Project"]}))
    m.loc[named, a.columns] = a.values

    rest = m.location_id.isna()
    m.loc[rest, "loc_method"] = m.loc[rest, "loc_method"].fillna("unresolved")
    m.loc[rest, "loc_confidence"] = m.loc[rest, "loc_confidence"].fillna("none")
    types = {lid: rec["location_type"] for lid, rec in resolver.locations.items()}
    m["location_type"] = m.location_id.map(types)
    return m


def build_locations(resolver: LocationResolver, m: pd.DataFrame, chl: pd.DataFrame, cfg: CleaningConfig, gis: dict):
    groups = resolver.place_groups()
    recs = []
    for lid, rec in resolver.locations.items():
        addr = rec["address"]
        recs.append({"location_id": lid, "location_type": rec["location_type"], "site_code": rec["site_code"],
                     "facility": rec.get("facility"), "point_type": rec.get("point_type"),
                     "address": addr.display if addr else None, "resolution_role": rec["resolution_role"],
                     "place_group": groups[lid], "_geo": rec["geo_address"]})
    loc = pd.DataFrame(recs)
    gc = [gis["geocoder"].geocode(a) for a in loc._geo]
    for j, col in enumerate(["lon", "lat", "geocode_method", "geocode_match", "geocode_uncertainty_m"]):
        loc[col] = [g[j] for g in gc]
    loc = loc.drop(columns="_geo")
    loc["region"] = loc.site_code.str.extract(r"^([A-Z]+)")[0]
    _disambiguate_by_region(loc, [g[5] for g in gc], [g[6] for g in gc])
    _apply_geocode_overrides(loc, cfg.reference_dir / "geocode_overrides.csv")
    # Facility sub-points share their plant's coordinates.
    for plant in ("JK", "PB"):
        ref = loc[loc.location_id == f"PLANT-{plant}-FINISHED"].iloc[0]
        same = loc.location_id.str.startswith(f"PLANT-{plant}-")
        loc.loc[same, ["lon", "lat", "geocode_method", "geocode_uncertainty_m"]] = \
            [ref.lon, ref.lat, ref.geocode_method, ref.geocode_uncertainty_m]

    feats = geo.hydrant_features(loc.lon, loc.lat, gis["hydrants"], cfg.hydrant_k, cfg.neighborhood_radius_m)
    loc = pd.concat([loc.reset_index(drop=True), feats], axis=1)
    loc["geocode_suspicious"] = loc.nearest_hydrant_m > cfg.suspicious_hydrant_distance_m
    for plant in ("JK", "PB"):
        ref = loc[loc.location_id == f"PLANT-{plant}-FINISHED"].iloc[0]
        loc[f"dist_{plant.lower()}_km"] = geo.distance_km(loc.lon, loc.lat, ref.lon, ref.lat)
    loc["in_davis_phase2_area"] = geo.in_box(loc.lon, loc.lat, gis["davis_box"])

    counts = m.groupby("location_id").size().rename("n_rows_all")
    sc = chl.groupby("location_id").agg(n_chlorine_in_scope=("value", "size"), first_in_scope=("Sampled", "min"),
                                        last_in_scope=("Sampled", "max"))
    months = ((sc.last_in_scope - sc.first_in_scope).dt.days / 30.44).clip(lower=1)
    sc["in_scope_samples_per_month"] = (sc.n_chlorine_in_scope / months).round(2)
    loc = loc.join(counts, on="location_id").join(sc, on="location_id")
    loc["n_rows_all"] = loc.n_rows_all.fillna(0).astype(int)
    loc["n_chlorine_in_scope"] = loc.n_chlorine_in_scope.fillna(0).astype(int)
    loc["high_frequency_site"] = loc.in_scope_samples_per_month >= 4      # routine sites run ~2/month
    order = {"routine": 0, "tank": 1, "plant": 2, "nap_site": 3, "adhoc": 4}
    return loc.sort_values(["location_type", "location_id"], key=lambda s: s.map(order) if s.name == "location_type" else s)\
              .reset_index(drop=True)


def _disambiguate_by_region(loc: pd.DataFrame, methods, candidates, pad_deg: float = 0.005):
    """A routine address matching both W and E (or N and S) versions of a street is resolved to the
    candidate inside the bounding box of its region's unambiguously geocoded sites (padded ~500 m).
    E/W region codes and street directionals agree in the data (no E site on a W street)."""
    known = loc[loc.location_type.eq("routine") & loc.lat.notna()]
    env = known.groupby("region").agg(lon_min=("lon", "min"), lon_max=("lon", "max"),
                                      lat_min=("lat", "min"), lat_max=("lat", "max"))
    todo = loc.index[loc.geocode_method.eq("ambiguous_direction") & loc.location_type.eq("routine")]
    for i in todo:
        region = loc.at[i, "region"]
        if region not in env.index:
            continue
        e = env.loc[region]
        inside = [c for c in candidates[i] if e.lon_min - pad_deg <= c[0] <= e.lon_max + pad_deg
                  and e.lat_min - pad_deg <= c[1] <= e.lat_max + pad_deg]
        if inside and geo.spread_m([c[0] for c in inside], [c[1] for c in inside]) <= geo.AMBIGUOUS_SPREAD_M:
            loc.at[i, "lon"] = float(np.mean([c[0] for c in inside]))
            loc.at[i, "lat"] = float(np.mean([c[1] for c in inside]))
            loc.at[i, "geocode_uncertainty_m"] = max(c[2] for c in inside)
            loc.at[i, "geocode_method"] = f"{methods[i]}_region_disambiguated"


def _apply_geocode_overrides(loc: pd.DataFrame, path: Path):
    """Manual coordinates (config/geocode_overrides.csv) win over automatic geocoding."""
    if not path.exists():
        return
    ov = pd.read_csv(path)
    for r in ov.itertuples():
        m = loc.location_id.eq(r.location_id)
        if not m.any():
            raise ValueError(f"geocode override for unknown location {r.location_id}")
        loc.loc[m, ["lon", "lat", "geocode_uncertainty_m", "geocode_method", "geocode_match"]] = \
            [r.lon, r.lat, r.uncertainty_m, "manual_override", r.reason]


def exclusion_reason(m: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    """First failing criterion for each total-chlorine row; '' = in scope. Order = funnel order."""
    is_cl = m.Analyte.eq(TOTAL_CHLORINE) & m.Analysis.eq("Chlres-field")
    steps = [
        ("not_total_chlorine", ~is_cl),
        ("project_out_of_scope", ~m.Project.isin(cfg.scope_projects)),
        ("not_a_routine_location", ~m.location_type.eq("routine")),
        ("location_unresolved_or_conflict", ~m.loc_confidence.isin(["high", "medium"])),
        ("result_not_reportable", ~m.result_status.isin(["detected", "nd"])),
        ("duplicate_copy", m.is_duplicate),
        ("value_out_of_range", m.flag_out_of_range.fillna(False)),
    ]
    reason = pd.Series("", index=m.index, dtype=object)
    for name, mask in steps:
        reason = reason.mask(reason.eq("") & mask.fillna(False).astype(bool), name)
    return reason


def run(cfg: CleaningConfig) -> dict:
    out_dir = cfg.out_dir
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    stats: dict = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "version": cfg.version}

    print("Loading raw export ...")
    stats["raw_sha256"] = sha256(cfg.raw_xlsx)
    stats["raw_sha256_matches"] = stats["raw_sha256"] == cfg.expected_raw_sha256
    raw = load_raw(cfg.raw_xlsx)
    stats["raw_rows"] = len(raw)

    m = parse_results(raw, cfg.chlorine_max_mg_l)
    m["sample_name"] = normalize_sample_name(m.SampleName)
    m["analyst_norm"] = normalize_analyst(m.Analyst)
    m["analyst_id"], analyst_key = pseudonymize_analysts(m.analyst_norm)

    print("Resolving locations ...")
    resolver = LocationResolver(cfg, load_facilities(cfg.reference_dir / "facilities.csv"))
    m = resolve_locations(m, cfg, resolver)

    m = dedupe.mark_duplicates(m, cfg.scope_projects)
    m["flag_analyzed_before_sampled"] = m.Analyzed < m.Sampled - pd.Timedelta(hours=1)
    m["flag_received_before_sampled"] = m.Received < m.Sampled - pd.Timedelta(hours=1)
    m["flag_sampled_time_missing"] = m.Sampled.dt.hour.eq(0) & m.Sampled.dt.minute.eq(0)
    m["exclusion_reason"] = exclusion_reason(m, cfg)
    m["in_scope_v1"] = m.exclusion_reason.eq("")
    chl = m[m.in_scope_v1].copy()

    print("Loading public GIS layers and geocoding ...")
    roads = geo.load_layer(cfg.gis_dir, "road_centerlines")
    gis = {"geocoder": geo.Geocoder(geo.load_layer(cfg.gis_dir, "address_points"), roads, cfg.max_nearest_number_gap),
           "hydrants": geo.load_layer(cfg.gis_dir, "hydrants"), "davis_box": geo.davis_phase2_box(roads)}
    stats["gis_fetched_at"] = (cfg.gis_dir / "FETCHED_AT").read_text().strip()
    stats["gis_counts"] = json.loads((cfg.gis_dir / "counts.json").read_text())
    stats["davis_box"] = gis["davis_box"]
    h = gis["hydrants"]
    inside = geo.in_box(h.lon, h.lat, gis["davis_box"])
    stats["davis_box_hydrant_planes"] = h.loc[inside, "PressurePlane"].value_counts().to_dict()
    loc = build_locations(resolver, m, chl, cfg, gis)

    print("Inferring plant events ...")
    external = events.load_external(cfg.reference_dir / "external_events.csv")
    periods = events.score_periods(events.infer_offline_periods(m, cfg.plant_gap_min_days), chl, external,
                                   cfg.dip_flank_days, cfg.jk_dip_supported_mg_l)
    ev = pd.concat([periods.assign(source="inferred_from_plant_sampling_gaps"),
                    external.rename(columns={"scope": "plant"}).assign(source="external_document")], ignore_index=True)

    chl = events.annotate(chl, periods)
    static = loc.set_index("location_id")[["site_code", "place_group", "region", "pressure_plane", "high_frequency_site"]]
    chl = chl.drop(columns="site_code").join(static, on="location_id")
    chlorine = pd.DataFrame({
        "row_id": chl.row_id, "location_id": chl.location_id, "site_code": chl.site_code, "place_group": chl.place_group,
        "region": chl.region, "pressure_plane": chl.pressure_plane, "high_frequency_site": chl.high_frequency_site,
        "sampled_at": chl.Sampled, "sample_date": chl.Sampled.dt.normalize(),
        "week_start": chl.Sampled.dt.to_period("W-SUN").dt.start_time,
        "value_mg_l": chl.value, "censored": chl.censored, "detection_limit": chl.detection_limit,
        "decimals_observed": chl.decimals_observed, "project": chl.Project, "analyst_id": chl.analyst_id,
        "lims_status": chl.Status, "loc_method": chl.loc_method, "loc_confidence": chl.loc_confidence,
        "address_offset": pd.to_numeric(chl.address_offset),
        "dup_conflict": chl.dup_conflict, "dup_value_range": chl.dup_value_range,
        "flag_sampled_time_missing": chl.flag_sampled_time_missing,
        "flag_analyzed_before_sampled": chl.flag_analyzed_before_sampled,
        "jk_offline": chl.jk_offline, "jk_offline_confidence": chl.jk_offline_confidence,
        "pb_offline": chl.pb_offline, "pb_offline_confidence": chl.pb_offline_confidence,
        "plant_regime": chl.plant_regime,
    }).sort_values(["sampled_at", "location_id"]).reset_index(drop=True)

    # Measurements table: no raw alias text, no analyst initials; complaint sample names that are
    # not street addresses may be personal names and are redacted.
    redact = m.Project.isin(COMPLAINT_PROJECTS) & ~m.sample_name.map(lambda s: parse_address(s) is not None
                                                                    if pd.notna(s) else False)
    measurements = pd.DataFrame({
        "row_id": m.row_id, "wrk": m.Wrk, "sample_seq": m.Sample, "client": m.Client, "project": m.Project,
        "sample_name": m.sample_name.mask(redact, "<redacted>"), "sample_class": m.sample_class,
        "site_code": m.site_code, "location_id": m.location_id, "location_type": m.location_type,
        "loc_method": m.loc_method, "loc_confidence": m.loc_confidence, "loc_conflict_with": m.loc_conflict_with,
        "address_offset": pd.to_numeric(m.address_offset), "sampled_at": m.Sampled, "analyzed_at": m.Analyzed,
        "received_at": m.Received, "status_date": m.StatusDate, "analysis": m.Analysis, "analyte": m.Analyte,
        "units": m.FinalUnits, "result_text": m.result_text, "value": m.value, "result_status": m.result_status,
        "censored": m.censored, "detection_limit": m.detection_limit, "mrl": m.MRL,
        "decimals_observed": m.decimals_observed, "lims_status": m.Status, "analyst_id": m.analyst_id,
        "dup_group": m.dup_group, "is_duplicate": m.is_duplicate, "dup_conflict": m.dup_conflict,
        "dup_value_range": m.dup_value_range,
        "flag_out_of_range": m.flag_out_of_range.fillna(False),
        "flag_analyzed_before_sampled": m.flag_analyzed_before_sampled,
        "flag_received_before_sampled": m.flag_received_before_sampled,
        "flag_sampled_time_missing": m.flag_sampled_time_missing,
        "exclusion_reason": m.exclusion_reason.replace("not_total_chlorine", None).replace("", None),
        "in_scope_v1": m.in_scope_v1,
    })
    for c in ["client", "project", "sample_name", "sample_class", "site_code", "location_id", "location_type",
              "loc_method", "loc_confidence", "loc_conflict_with", "analysis", "analyte", "units", "result_text",
              "result_status", "lims_status", "analyst_id", "exclusion_reason"]:
        measurements[c] = measurements[c].astype("string")

    review = pd.DataFrame(resolver.review).sort_values(["site_code", "n_rows"], ascending=[True, False])

    print("Running checks and writing outputs ...")
    checks = report.run_checks(cfg, stats, measurements, chlorine, loc, periods)
    phone_hits = {name: find_phone_numbers(df) for name, df in
                  [("measurements", measurements), ("chlorine", chlorine), ("locations", loc), ("review", review)]}
    stats["phone_hits"] = phone_hits
    if any(phone_hits.values()):
        raise RuntimeError(f"Phone-number patterns found in outputs, refusing to write: {phone_hits}")

    measurements.to_parquet(out_dir / "measurements.parquet", index=False)
    chlorine.to_parquet(out_dir / f"chlorine_{cfg.version}.parquet", index=False)
    loc.to_csv(out_dir / "locations.csv", index=False)
    ev.to_csv(out_dir / "events.csv", index=False, date_format="%Y-%m-%d")
    review.to_csv(out_dir / "location_review.csv", index=False)
    analyst_key.to_csv(out_dir / "private" / "analyst_key.csv", index=False)

    report.write_report(out_dir / "dq_report.md", cfg, stats, m, measurements, chlorine, loc, periods, external,
                        review, checks)
    manifest = {
        "version": cfg.version, "generated_at": stats["generated_at"], "raw_file": cfg.raw_xlsx.name,
        "raw_sha256": stats["raw_sha256"], "raw_sha256_matches_expected": stats["raw_sha256_matches"],
        "gis_fetched_at": stats["gis_fetched_at"], "gis_counts": stats["gis_counts"],
        "rows": {"raw": len(raw), "measurements": len(measurements), f"chlorine_{cfg.version}": len(chlorine),
                 "locations": len(loc), "events": len(ev)},
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(cfg).items()},
        "checks": {c["name"]: c["status"] for c in checks},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "git_commit": _git_commit(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    failed = [c for c in checks if c["status"] == "FAIL"]
    print(f"Done: {len(chlorine)} in-scope chlorine samples at {chlorine.location_id.nunique()} locations -> {out_dir}")
    print(f"Checks: {sum(c['status'] == 'PASS' for c in checks)} pass, "
          f"{sum(c['status'] == 'WARN' for c in checks)} warn, {len(failed)} fail")
    if failed:
        raise RuntimeError("Failed checks: " + "; ".join(c["name"] for c in failed))
    return {"measurements": measurements, "chlorine": chlorine, "locations": loc, "events": ev, "checks": checks}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", type=Path, help="raw LIMS export (.xlsx)")
    p.add_argument("--version", default=None, help="output version label (default v1)")
    args = p.parse_args()
    cfg = CleaningConfig()
    if args.raw:
        cfg = dataclasses.replace(cfg, raw_xlsx=args.raw)
    if args.version:
        cfg = dataclasses.replace(cfg, version=args.version)
    run(cfg)


if __name__ == "__main__":
    main()
