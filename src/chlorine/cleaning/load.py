"""Read the raw LIMS export into a typed DataFrame with stable row ids."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["Wrk", "Sample", "Client", "Project", "SampleName", "SampleAlias", "Sampled", "Analysis",
               "Analyte", "Analyst", "Analyzed", "tResult", "FinalUnits", "MRL", "Status", "StatusDate", "Received"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _result_text(v) -> str | None:
    """Excel cell -> text. Floats are printed without binary noise (3.3000000000000003 -> '3.3').
    Excel does not keep trailing zeros, so '3.80' is indistinguishable from '3.8'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, float):
        return format(v, ".10g")
    return str(v).strip()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Data", dtype={"tResult": object})
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw export is missing columns: {sorted(missing)}")
    df = df[RAW_COLUMNS].copy()
    df.insert(0, "row_id", range(len(df)))              # file order; stable across runs
    df["result_text"] = df.pop("tResult").map(_result_text)
    for col in ["Client", "Project", "SampleName", "SampleAlias", "Analysis", "Analyte", "Analyst", "FinalUnits", "Status"]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df["MRL"] = df["MRL"].round(4)                       # stored as float32 in the export (0.0799999...)
    return df
