"""Plant operating-state events.

No plant operations log is available, so plant on/off periods are inferred from gaps in plant
sampling: a run of >= plant_gap_min_days with no sample of any analyte at a plant's finished-water
tap is treated as the plant being offline. Only finished water counts: raw (lake) water keeps being
sampled weekly while a plant is idle (PBRAW through PB's 2022-2026 winter shutdowns), so raw
samples are not evidence that a plant is producing. Kubala (JK) outages are then checked against
the city-wide distribution residual (they cause dips); Pierce-Burch (PB) routinely idles in winter
without a measurable dip.

Columns for downstream use:
  jk_offline / pb_offline  - from plant sampling dates only. Safe as model features.
  *_confidence             - also uses the chlorine dip, i.e. the outcome. For reporting and
                             sensitivity analysis only; never use as a model feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLANT_POINTS = {
    "JK": ["PLANT-JK-FINISHED"],
    "PB": ["PLANT-PB-FINISHED", "PLANT-PB-SOUTH-FINISHED"],
}


def infer_offline_periods(meas: pd.DataFrame, min_gap_days: int) -> pd.DataFrame:
    data_start = meas["Sampled"].min().normalize()
    data_end = meas["Sampled"].max().normalize()
    rows = []
    for plant, points in PLANT_POINTS.items():
        days = np.sort(meas.loc[meas["location_id"].isin(points), "Sampled"].dt.normalize().unique())
        days = pd.DatetimeIndex(days)
        edges = [(data_start - pd.Timedelta(days=1), days[0], "start")] + \
                [(a, b, "") for a, b in zip(days[:-1], days[1:])] + \
                [(days[-1], data_end + pd.Timedelta(days=1), "end")]
        for a, b, boundary in edges:
            if (b - a).days - 1 >= min_gap_days:
                rows.append({"plant": plant, "start": a + pd.Timedelta(days=1), "end": b - pd.Timedelta(days=1),
                             "boundary": boundary})
    out = pd.DataFrame(rows).sort_values(["plant", "start"]).reset_index(drop=True)
    out["days"] = (out["end"] - out["start"]).dt.days + 1
    out.insert(0, "event_id", [f"INF-{p}-OFF-{s:%Y%m%d}" for p, s in zip(out.plant, out.start)])
    return out


def score_periods(periods: pd.DataFrame, chlorine: pd.DataFrame, external: pd.DataFrame, flank_days: int,
                  jk_supported: float) -> pd.DataFrame:
    """Add the city-wide residual dip (inside vs. flanking windows) and a confidence label."""
    daily = chlorine.groupby(chlorine["Sampled"].dt.normalize())["value"].mean()
    out = periods.copy()
    dips, conf, ext_ids = [], [], []
    for r in out.itertuples():
        same = out[out.plant == r.plant]
        inside = daily[r.start:r.end]
        flank = pd.concat([daily[r.start - pd.Timedelta(days=flank_days): r.start - pd.Timedelta(days=1)],
                           daily[r.end + pd.Timedelta(days=1): r.end + pd.Timedelta(days=flank_days)]])
        for s in same.itertuples():                         # flanks exclude other outages of the same plant
            flank = flank[(flank.index < s.start) | (flank.index > s.end)]
        dip = inside.mean() - flank.mean() if len(inside) and len(flank) else np.nan
        dips.append(round(float(dip), 3) if pd.notna(dip) else np.nan)
        ext = external[(external.event_type == "plant_offline") & (external.scope == r.plant)
                       & (external.start <= r.end) & (external.end >= r.start)]
        ext_ids.append(";".join(ext.event_id) or None)
        if r.boundary:
            conf.append("ambiguous_boundary")
        elif len(ext):
            conf.append("externally_confirmed")
        elif r.plant == "JK":
            conf.append("supported_by_dip" if pd.notna(dip) and dip <= jk_supported else "ambiguous")
        else:
            conf.append("inferred")
    out["city_dip_mg_l"] = dips
    out["external_event"] = ext_ids
    out["confidence"] = conf
    return out


def annotate(chlorine: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    out = chlorine.copy()
    t = out["Sampled"].dt.normalize()
    for plant in ("JK", "PB"):
        col, conf = f"{plant.lower()}_offline", f"{plant.lower()}_offline_confidence"
        out[col] = False
        out[conf] = pd.Series(pd.NA, index=out.index, dtype="string")
        for r in periods[periods.plant == plant].itertuples():
            m = (t >= r.start) & (t <= r.end)
            out.loc[m, col] = True
            out.loc[m, conf] = r.confidence
    out["plant_regime"] = np.select([out.jk_offline & out.pb_offline, out.jk_offline, out.pb_offline],
                                    ["both_offline", "jk_offline", "pb_offline"], default="both_online")
    return out


def load_external(path) -> pd.DataFrame:
    ext = pd.read_csv(path, dtype=str)
    for c in ("start", "end"):
        ext[c] = pd.to_datetime(ext[c])
    return ext
