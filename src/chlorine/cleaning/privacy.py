"""Privacy controls.

* SampleAlias is free text. In complaint projects it holds the customer's name and phone number,
  so raw alias text is never written to any output; only parsed addresses and facility labels
  derived from it are kept.
* Analyst initials identify staff. Results are reported under stable pseudonyms (A01, A02, ...);
  the key is written to a separate private file.
"""
from __future__ import annotations

import pandas as pd

from .addresses import contains_phone

COMPLAINT_PROJECTS = {
    "Health Concern", "Health Request", "Customer Complaint - Other", "Taste and Odor Concern",
    "Discolored Water", "Particulates in Water", "Cloudy Water", "Seep or Tap Leak",
    "Customer Lead/Copper", "Water Hardness Issue",
}
_UNKNOWN_ANALYSTS = {"ARLINGTON"}


def normalize_analyst(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    return out.mask(out.isin(_UNKNOWN_ANALYSTS) | out.eq(""))


def pseudonymize_analysts(analyst: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Map normalized initials to A01.. in alphabetical order (stable for a given export)."""
    codes = sorted(analyst.dropna().unique())
    key = pd.DataFrame({"analyst_initials": codes, "analyst_id": [f"A{i + 1:02d}" for i in range(len(codes))]})
    mapping = dict(zip(key.analyst_initials, key.analyst_id))
    return analyst.map(mapping).astype("string"), key


def find_phone_numbers(df: pd.DataFrame) -> dict[str, int]:
    """Count cells containing a phone-number pattern, per text column."""
    hits = {}
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            n = int(df[col].dropna().astype(str).map(contains_phone).sum())
            if n:
                hits[col] = n
    return hits
