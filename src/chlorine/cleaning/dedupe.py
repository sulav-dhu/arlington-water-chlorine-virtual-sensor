"""Duplicate handling.

The same physical sample is often logged under two work orders (e.g. Monthly Bact and Weekly NAP
Sites). Duplicates share (location, sample time, analysis, analyte). The analysis is part of the key
because a field and a lab measurement of the same analyte (pH-field vs. pH) are different results.
Aggregated in-plant locations (PLANT-*-OTHER) are keyed by sample name so that different filters or
clearwells sampled at the same time are not mistaken for duplicates.
One copy per group is kept (in-scope project first, then 'Reported' status, then lowest row_id);
the others get is_duplicate = True. When the copies disagree (dup_conflict = True), they are almost
always two programs measuring the same water at the same moment (e.g. Monthly Bact 2.6 vs. Weekly
NAP 2.59), i.e. replicate measurements. The preferred copy is still kept, so a second program
sampling alongside does not remove the routine result; dup_value_range records the disagreement
for sensitivity analysis and doubles as a measurement-repeatability sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mark_duplicates(df: pd.DataFrame, scope_projects: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    loc_key = out["location_id"].fillna("SN:" + out["sample_name"].fillna("?"))
    subpoint = out["location_id"].fillna("").str.endswith("-OTHER")
    loc_key = loc_key.mask(subpoint, loc_key + "|" + out["sample_name"].fillna("?"))
    keys = [loc_key, out["Sampled"], out["Analysis"].fillna("?"), out["Analyte"].fillna("?")]
    result_key = out["result_status"] + ":" + out["value"].astype("string").fillna("")
    size = out.groupby(keys, dropna=False)["row_id"].transform("size")
    multi = size > 1
    n_results = result_key.groupby(keys, dropna=False).transform("nunique")

    out["dup_group"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[multi, "dup_group"] = out[multi].groupby([k[multi] for k in keys], dropna=False).ngroup().values
    out["dup_conflict"] = multi & (n_results > 1)
    rng = out["value"].groupby(keys, dropna=False).transform("max") - out["value"].groupby(keys, dropna=False).transform("min")
    out["dup_value_range"] = rng.where(multi).round(4)

    priority = np.where(out["Project"].isin(scope_projects), 0, 1) * 2 + np.where(out["Status"].eq("Reported"), 0, 1)
    order = pd.DataFrame({"g": out["dup_group"], "p": priority, "r": out["row_id"]})
    rank = order[multi].sort_values(["g", "p", "r"]).groupby("g").cumcount()
    out["is_duplicate"] = False
    out.loc[rank.index, "is_duplicate"] = (rank > 0).values
    return out
