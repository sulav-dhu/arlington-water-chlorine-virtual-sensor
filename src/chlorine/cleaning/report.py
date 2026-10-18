"""Validation checks and the Markdown data-quality report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CleaningConfig
from .geo import distance_km
from .values import TOTAL_CHLORINE

# Approximate Kubala coordinates from the EPA Risk Management Plan listing (2-decimal precision).
JK_REFERENCE = (-97.17, 32.63)


def _check(name, ok, detail, severity="FAIL"):
    return {"name": name, "status": "PASS" if ok else severity, "detail": detail}


def run_checks(cfg: CleaningConfig, stats, meas, chl, loc, periods) -> list[dict]:
    c = []
    c.append(_check("raw file matches the verified export (sha256)", stats["raw_sha256_matches"],
                    stats["raw_sha256"][:16] + "...", "WARN"))
    c.append(_check("every raw row is kept in measurements", len(meas) == stats["raw_rows"],
                    f"{len(meas)} of {stats['raw_rows']}"))
    c.append(_check("row_id is unique", meas.row_id.is_unique, ""))
    c.append(_check("modeling table has one row per (location, timestamp)",
                    not chl.duplicated(["location_id", "sampled_at"]).any(),
                    f"{int(chl.duplicated(['location_id', 'sampled_at']).sum())} duplicates"))
    v = chl.value_mg_l.dropna()
    c.append(_check("modeling values within [0, chlorine_max]", bool(((v >= 0) & (v <= cfg.chlorine_max_mg_l)).all()),
                    f"range {v.min():.2f}-{v.max():.2f} mg/L"))
    in_scope = meas[meas.in_scope_v1]
    c.append(_check("in-scope rows are routine locations with high/medium location confidence",
                    bool(in_scope.location_type.eq("routine").all() and in_scope.loc_confidence.isin(["high", "medium"]).all()), ""))
    c.append(_check("censored values have no numeric value", bool(chl.loc[chl.censored, "value_mg_l"].isna().all()),
                    f"{int(chl.censored.sum())} censored"))

    used = loc[loc.n_chlorine_in_scope > 0]
    w = used.n_chlorine_in_scope
    geo_cov = float(w[used.lat.notna()].sum() / w.sum())
    c.append(_check("in-scope samples at geocoded locations >= 95%", geo_cov >= 0.95, f"{geo_cov:.1%}", "WARN"))
    plane_cov = float(w[used.pressure_plane.notna() & used.pressure_plane.ne("UNKNOWN")].sum() / w.sum())
    c.append(_check("in-scope samples with a pressure plane >= 95%", plane_cov >= 0.95, f"{plane_cov:.1%}", "WARN"))
    sus = used[used.geocode_suspicious.fillna(False).astype(bool)]
    c.append(_check("no in-scope location is far (>300 m) from every hydrant", sus.empty,
                    ", ".join(sus.location_id) or "none", "WARN"))
    low_vote = used[used.plane_vote_share < 0.6]
    c.append(_check("pressure-plane vote is clear (>= 60%) at all in-scope locations", low_vote.empty,
                    ", ".join(low_vote.location_id) or "none", "WARN"))

    jk = loc[loc.location_id == "PLANT-JK-FINISHED"].iloc[0]
    d = float(distance_km([jk.lon], [jk.lat], *JK_REFERENCE)[0]) if pd.notna(jk.lon) else np.nan
    c.append(_check("Kubala geocode within 2 km of EPA reference coordinates", pd.notna(d) and d <= 2.0,
                    f"{d:.2f} km" if pd.notna(d) else "not geocoded", "WARN"))
    jk_p = periods[periods.plant == "JK"]
    feb17 = jk_p[(jk_p.start <= "2017-03-31") & (jk_p.end >= "2017-02-01")]
    c.append(_check("inferred Kubala outage overlaps the documented Feb 2017 outage",
                    len(feb17) == 1 and feb17.confidence.iloc[0] == "externally_confirmed",
                    ", ".join(feb17.event_id) or "none", "WARN"))
    both = float((chl.plant_regime == "both_offline").mean())
    c.append(_check("rows flagged with both plants offline < 1% (physically implausible)", both < 0.01,
                    f"{both:.2%}", "WARN"))
    return c


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            elif isinstance(v, pd.Timestamp):
                cells.append(f"{v:%Y-%m-%d}")
            else:
                cells.append("" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(path, cfg: CleaningConfig, stats, m, meas, chl, loc, periods, external, review, checks):
    is_cl = meas.analyte.eq(TOTAL_CHLORINE) & meas.analysis.eq("Chlres-field")
    cl = meas[is_cl]
    L = []
    w = L.append
    w(f"# Data-quality report — cleaning {cfg.version}\n")
    w(f"Generated {stats['generated_at']}. Input `{cfg.raw_xlsx.name}` (sha256 `{stats['raw_sha256'][:16]}…`, "
      f"{'matches' if stats['raw_sha256_matches'] else '**does not match**'} the verified export). "
      f"Public GIS layers fetched {stats['gis_fetched_at']}.\n")

    w("## Summary\n")
    span = f"{chl.sampled_at.min():%Y-%m-%d} to {chl.sampled_at.max():%Y-%m-%d}"
    w(f"- **{len(chl):,} in-scope total-chlorine samples** at **{chl.location_id.nunique()} locations** "
      f"({chl.place_group.nunique()} distinct physical places), {span}.\n")
    w(f"- Scope: projects {', '.join(cfg.scope_projects)}; routine distribution locations; reportable results; "
      f"duplicates and location conflicts removed.\n")
    w(f"- Checks: {sum(c['status'] == 'PASS' for c in checks)} pass, {sum(c['status'] == 'WARN' for c in checks)} warn, "
      f"{sum(c['status'] == 'FAIL' for c in checks)} fail (details at the end).\n")

    w("## 1. Funnel (total chlorine)\n")
    funnel = [("All rows in export", len(meas)), ("Total-chlorine rows (Chlres-field)", int(is_cl.sum()))]
    remaining = int(is_cl.sum())
    for reason in ["project_out_of_scope", "not_a_routine_location", "location_unresolved_or_conflict",
                   "result_not_reportable", "duplicate_copy", "value_out_of_range"]:
        n = int(cl.exclusion_reason.eq(reason).sum())
        remaining -= n
        funnel.append((f"− {reason.replace('_', ' ')}", -n))
    funnel.append(("**In scope (modeling table)**", remaining))
    w(_md_table(pd.DataFrame(funnel, columns=["step", "rows"])))
    w("\nOf the excluded total-chlorine rows outside scope projects, the largest groups are: "
      + ", ".join(f"{p} ({n})" for p, n in cl[cl.exclusion_reason.eq('project_out_of_scope')].project.value_counts().head(6).items())
      + ".\n")

    w("## 2. Sample classes and location resolution\n")
    sc = meas.groupby("sample_class").agg(rows=("row_id", "size"), locations=("location_id", "nunique")).reset_index()
    w(_md_table(sc))
    w("\nHow each row got its location (all analytes):\n\n")
    lm = meas.groupby(["loc_method", "loc_confidence"], dropna=False).size().rename("rows").reset_index()
    w(_md_table(lm))
    w("\nLocation types:\n\n")
    lt = loc.groupby("location_type").agg(locations=("location_id", "size"), rows=("n_rows_all", "sum"),
                                          in_scope_chlorine=("n_chlorine_in_scope", "sum")).reset_index()
    w(_md_table(lt))

    roles = review.role.value_counts()
    w(f"\n**Routine code address clusters** ({len(review)} clusters over {review.site_code.nunique()} codes): "
      + ", ".join(f"{k} {v}" for k, v in roles.items()) + ". Full list: `location_review.csv`.\n\n")
    multi = review[review.role.isin(["secondary", "reuse"])]
    w("Codes that map to more than one physical location (relocation or code re-use):\n\n")
    w(_md_table(review[review.site_code.isin(multi.site_code)][["site_code", "location_id", "role", "address", "n_rows",
                                                                 "first", "last"]]))
    w("\nCross-code conflicts (alias is another site's address; rows kept but excluded from modeling):\n\n")
    conf = review[review.role.eq("conflict")][["site_code", "address", "n_rows", "first", "last", "conflict_with"]]
    w(_md_table(conf))
    w("\nMiscoded site codes reassigned by address:\n\n")
    w(_md_table(review[review.role.eq("miscode")][["site_code", "address", "n_rows", "location_id"]]))
    shared = loc[loc.location_type.eq("routine")].groupby("place_group").location_id.agg(list)
    shared = shared[shared.map(len) > 1]
    w(f"\n**Place groups:** {len(shared)} physical places carry more than one routine location id "
      f"(e.g. a site renumbered or moved across the street). Use `place_group`, not `location_id`, for "
      f"spatial cross-validation folds:\n\n")
    w(_md_table(pd.DataFrame({"place_group": shared.index, "location_ids": shared.map(", ".join).values})))

    w("\n## 3. Duplicates\n")
    dups = meas[meas.dup_group.notna()]
    w(f"- {dups.dup_group.nunique():,} groups of rows share (location, sample time, analysis, analyte); one copy "
      f"is kept and {int(meas.is_duplicate.sum()):,} copies are dropped. In "
      f"{dups[dups.dup_conflict].dup_group.nunique()} groups the copies disagree; the preferred copy is kept and "
      f"flagged (`dup_conflict`, `dup_value_range`).\n")
    pairs = dups.groupby("dup_group").project.agg(lambda s: " + ".join(sorted(s))).value_counts().head(6)
    w("- Most common project pairs: " + "; ".join(f"{k} ({v})" for k, v in pairs.items()) + ".\n")
    rep = dups[dups.analyte.eq(TOTAL_CHLORINE) & dups.analysis.eq("Chlres-field") & dups.result_status.eq("detected")
               & dups.location_type.eq("routine")]
    rg = rep.groupby("dup_group").agg(n=("value", "size"), rng=("dup_value_range", "first"))
    rg = rg[rg.n >= 2]
    if len(rg):
        w(f"- **Measurement repeatability (total chlorine, routine locations):** {len(rg)} same-moment pairs; "
          f"{(rg.rng == 0).mean():.0%} identical (likely one reading entered twice). Among the "
          f"{int((rg.rng > 0).sum())} that differ (likely separate readings): median |Δ| "
          f"{rg.rng[rg.rng > 0].median():.2f}, 90th percentile {rg.rng[rg.rng > 0].quantile(0.9):.2f}, max "
          f"{rg.rng.max():.2f} mg/L.\n")

    w("\n## 4. Values and timestamps\n")
    rs = meas.groupby(["result_status"]).size().rename("rows").reset_index()
    w(_md_table(rs))
    w(f"\n- Total chlorine in scope: mean {chl.value_mg_l.mean():.2f}, median {chl.value_mg_l.median():.2f}, "
      f"min {chl.value_mg_l.min():.2f}, max {chl.value_mg_l.max():.2f} mg/L; {int(chl.censored.sum())} non-detects "
      f"(below the {', '.join(f'{x:g}' for x in sorted(chl.detection_limit.dropna().unique()))} mg/L reporting limit).\n")
    oor = meas[meas.flag_out_of_range]
    w(f"- Out-of-range values flagged (all analytes): {len(oor)} — "
      + ", ".join(f"{a} ({n})" for a, n in oor.analyte.value_counts().items()) + ".\n")
    w(f"- Analyzed > 1 h before sampled: {int(meas.flag_analyzed_before_sampled.sum())} rows; received before "
      f"sampled: {int(meas.flag_received_before_sampled.sum())}; sample time recorded as 00:00: "
      f"{int(meas.flag_sampled_time_missing.sum())}. The sample time is treated as authoritative.\n")
    dec = chl.decimals_observed.value_counts().sort_index()
    w("- Reported precision (decimals observed; lower bound because Excel drops trailing zeros): "
      + ", ".join(f"{int(k)} dp: {v:,}" for k, v in dec.items()) + ".\n")
    st = chl.lims_status.value_counts()
    allst = meas.lims_status.value_counts()
    w("- LIMS status of in-scope rows: " + ", ".join(f"{k} {v:,}" for k, v in st.items())
      + "; all rows: " + ", ".join(f"{k} {v:,}" for k, v in allst.items()) + ".\n")

    w("\n## 5. Geocoding and public-GIS features\n")
    used = loc[loc.n_chlorine_in_scope > 0]
    gm = used.groupby("geocode_method").agg(locations=("location_id", "size"),
                                            in_scope_samples=("n_chlorine_in_scope", "sum")).reset_index()
    w("In-scope locations by geocoding method (all matching is local against public city layers):\n\n")
    w(_md_table(gm))
    allm = loc.groupby(["location_type", "geocode_method"]).size().rename("locations").reset_index()
    w("\nAll locations:\n\n")
    w(_md_table(allm))
    pp = used.groupby("pressure_plane").agg(locations=("location_id", "size"),
                                            in_scope_samples=("n_chlorine_in_scope", "sum"),
                                            mean_vote_share=("plane_vote_share", "mean")).reset_index()
    w("\nPressure plane of in-scope locations (vote of the 7 nearest hydrants):\n\n")
    w(_md_table(pp))
    xt = pd.crosstab(used.region, used.pressure_plane)
    w("\nRegion prefix vs. pressure plane (the region code is not the plane):\n\n")
    w(_md_table(xt.reset_index()))
    w(f"\n- Nearest hydrant: median {used.nearest_hydrant_m.median():.0f} m, max {used.nearest_hydrant_m.max():.0f} m.\n")
    box = stats.get("davis_box")
    if box:
        inside = loc[loc.in_davis_phase2_area & loc.location_type.eq("routine")]
        w(f"- Davis Phase II area (approximate box from street centerlines: lon {box['lon_west']:.4f} to "
          f"{box['lon_east']:.4f}, lat {box['lat_south']:.4f} to {box['lat_north']:.4f}): "
          f"{len(inside)} routine locations inside ({', '.join(inside.location_id) or 'none'}). Their current "
          f"plane may differ from the plane before ~Oct 2024. Validation: hydrants inside the box today are "
          + ", ".join(f"{k} {v}" for k, v in stats["davis_box_hydrant_planes"].items())
          + " (the expansion moved this area into the upper plane).\n")
    for f in ["PLANT-JK-FINISHED", "PLANT-PB-FINISHED"]:
        r = loc[loc.location_id == f].iloc[0]
        w(f"- {f}: {r.lat:.5f}, {r.lon:.5f} ({r.geocode_method}); pressure plane at plant: {r.pressure_plane}.\n")
    tanks = loc[loc.location_type.eq("tank")]
    w(f"- Tanks with coordinates: {int(tanks.lat.notna().sum())} of {len(tanks)} "
      f"({', '.join(tanks[tanks.lat.notna()].location_id)}). Others need manual location (open item).\n")

    w("\n## 6. Plant operating-state events\n")
    w("Inferred from gaps of ≥ {} days in finished-water sampling at each plant (raw lake water is sampled even "
      "while a plant is idle, so it does not count). "
      "`city_dip_mg_l` = mean daily city-wide residual inside the period minus the ±{}-day flanks. "
      "Confidence uses the dip (the outcome) and must not be used as a model feature; "
      "`jk_offline`/`pb_offline` use plant sampling dates only.\n\n".format(cfg.plant_gap_min_days, cfg.dip_flank_days))
    w(_md_table(periods[["event_id", "plant", "start", "end", "days", "city_dip_mg_l", "confidence", "external_event"]],
                "{:+.2f}"))
    w("\nDocumented external events (config/external_events.csv):\n\n")
    w(_md_table(external[["event_id", "event_type", "scope", "start", "end", "date_precision"]]))
    reg = chl.plant_regime.value_counts()
    w("\nPlant regime of in-scope samples: " + ", ".join(f"{k} {v:,}" for k, v in reg.items()) + ".\n")

    w("\n## 7. Privacy\n")
    w("- Raw `SampleAlias` text is not written anywhere (in complaint projects it is a customer name/phone field).\n")
    w(f"- Complaint-project sample names that are not street addresses are redacted "
      f"({int((meas.sample_name == '<redacted>').sum())} rows).\n")
    w(f"- Analysts are pseudonymized ({meas.analyst_id.nunique()} ids); key in `private/analyst_key.csv`.\n")
    w(f"- Phone-number scan of all outputs: {sum(sum(v.values()) for v in stats['phone_hits'].values())} hits.\n")
    w("- Outputs still contain sampling-site and complaint addresses and coordinates: keep `data/` out of git and "
      "out of shared slides.\n")

    w("\n## 8. Checks\n")
    w(_md_table(pd.DataFrame(checks)[["status", "name", "detail"]]))

    w("\n## 9. Known limitations and open items\n")
    for item in [
        "Plant on/off periods are inferred from sampling gaps, not an operations log. Only Feb 2017 is externally "
        "documented. `ambiguous` periods may be sampling lapses.",
        "Pressure planes come from today's hydrant layer; boundaries moved in 2018 and 2024 (Davis expansions). "
        "Sites inside the Davis Phase II box are flagged; the Phase I area is unknown.",
        "Cross-code conflicts (Section 2) are excluded rather than resolved; the team can resolve them in "
        "`config/site_overrides.csv` if field records allow.",
        "Tank coordinates are missing for tanks without an address in the export (OpenStreetMap towers are unnamed).",
        "Reported precision is a lower bound (Excel drops trailing zeros).",
        "Hydrant `FlushProgram` codes (M, A) are undocumented; read `flush_program_share` as 'share of nearby "
        "hydrants in a flushing program (probably)'.",
        "Public GIS layers are live; re-fetching can change counts slightly. The fetch date is recorded.",
    ]:
        w(f"- {item}\n")
    path.write_text("".join(L))
