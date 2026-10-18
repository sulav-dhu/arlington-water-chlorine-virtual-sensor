"""Result parsing: numeric values, non-detects (censored), unreportable results, range checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

TOTAL_CHLORINE = "Chlorine residual"
NOT_REPORTABLE = {"NOT ANALYZED", "INVALID RESULT", "LAB ERROR"}

# Physically plausible ranges by analyte; values outside are flagged, never silently dropped.
PLAUSIBLE_RANGE = {
    "Chlorine residual": (0.0, 6.0),
    "Chlorine residual, Free": (0.0, 6.0),
    "Monochloramine": (0.0, 6.0),
    "Ammonia": (0.0, 5.0),
    "Nitrite": (0.0, 3.0),
    "Nitrate": (0.0, 20.0),
    "pH": (4.0, 11.0),
    "pH-field meter": (4.0, 11.0),
    "Temp for pH in oC": (0.0, 40.0),
    "Temp for pH-field in oC": (0.0, 40.0),
    "Dissolved Oxygen": (0.0, 20.0),
    "Total Organic Carbon": (0.0, 30.0),
    "Dissolved Organic Carbon": (0.0, 30.0),
}


def parse_results(df: pd.DataFrame, chlorine_max: float) -> pd.DataFrame:
    """Adds value, result_status (detected | nd | not_reportable | unparsed), censored,
    detection_limit, decimals_observed and flag_out_of_range."""
    text = df["result_text"].astype("string")
    value = pd.to_numeric(text, errors="coerce")
    upper = text.str.upper()
    status = np.select(
        [value.notna(), upper.eq("ND").fillna(False), upper.isin(NOT_REPORTABLE).fillna(False)],
        ["detected", "nd", "not_reportable"], default="unparsed")
    out = df.copy()
    out["value"] = value.round(4)
    out["result_status"] = status
    out["censored"] = status == "nd"
    out["detection_limit"] = df["MRL"].where(out["censored"])
    # Lower bound on reported precision: Excel drops trailing zeros (3.80 -> 3.8).
    frac = text.str.extract(r"\.(\d+)$")[0]
    out["decimals_observed"] = frac.str.len().fillna(0).astype("Int64").where(status == "detected")

    lo = df["Analyte"].map(lambda a: PLAUSIBLE_RANGE.get(a, (np.nan, np.nan))[0])
    hi = df["Analyte"].map(lambda a: PLAUSIBLE_RANGE.get(a, (np.nan, np.nan))[1])
    hi = hi.where(df["Analyte"] != TOTAL_CHLORINE, chlorine_max)
    out["flag_out_of_range"] = ((out["value"] < lo) | (out["value"] > hi)).fillna(False).astype(bool)
    return out
