"""Load the cleaned v1 tables into the analysis frame used by every protocol.

Unit of analysis: one in-scope routine total-chlorine sample. The 2 censored (non-detect) samples
are dropped from evaluation targets (they have no numeric value). The neighbor unit is the
place_group: locations at the same physical place (e.g. a renumbered site) count as one place, are
never each other's neighbors, and always fall in the same cross-validation fold.
"""
from __future__ import annotations

import pandas as pd

from ..config import EvalConfig

STATIC_FEATURES = ["lat", "lon", "dist_jk_km", "dist_pb_km", "flush_program_share", "hydrant_median_year",
                   "hydrants_in_radius", "in_davis_phase2_area"]


def load_frame(cfg: EvalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (samples, places). samples: one row per sample with target `y`; places: one row per
    place_group with coordinates and categorical attributes for neighbor weighting."""
    chl = pd.read_parquet(cfg.data_dir / f"chlorine_{cfg.data_version}.parquet")
    loc = pd.read_csv(cfg.data_dir / "locations.csv")
    df = chl[~chl.censored].copy()
    df["y"] = df.value_mg_l.astype(float)
    df["day"] = df.sample_date
    df["week"] = df.week_start
    df["month"] = df.sampled_at.dt.month
    df = df.merge(loc[["location_id"] + STATIC_FEATURES], on="location_id", how="left")
    for c in ["region", "pressure_plane"]:
        df[c] = df[c].astype(object).where(df[c].notna(), "UNKNOWN")
    df = df.sort_values(["sampled_at", "location_id"]).reset_index(drop=True)

    g = df.groupby("place_group")
    places = pd.DataFrame({
        "lat": g.lat.mean(), "lon": g.lon.mean(),
        "region": g.region.agg(lambda s: s.mode().iloc[0]),
        "pressure_plane": g.pressure_plane.agg(lambda s: s.mode().iloc[0]),
        "n_rows": g.size(),
    })
    # Every member of a place group shares the group's categorical attributes.
    df["region"] = df.place_group.map(places.region)
    df["pressure_plane"] = df.place_group.map(places.pressure_plane)
    return df, places
