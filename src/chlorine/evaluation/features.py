"""Leakage-safe features.

Neighbor features summarize what OTHER monitored places measured around the same time:
  1. each place's samples are averaged within a time bin, so a near-daily site counts once, like
     any other place, instead of seven times a week;
  2. those place means are averaged with a weighting scheme in which the target's own place group
     always has weight 0.
Only `pool` rows contribute: the pool is the monitored network. In the unseen-site protocol the
held-out places are not in the pool, so their values never enter any feature.

Information settings:
  wk  same calendar week (Mon-Sun), concurrent: a weekly "virtual sensor" estimate
  t7  the 7 days strictly before the sample's day: a real-time (causal) nowcast
Weighting schemes:
  city       all places                 region     same site-code region
  plane      same pressure plane        idw        exp(-distance / L) over all places
  plane_idw  idw within the same pressure plane
For every (setting, scheme) the features are <setting>_<scheme>_mean, the number of contributing
places <setting>_<scheme>_n and, for kernel schemes, the kernel weight sum <setting>_<scheme>_w.

Own-history features (own_prev_*) use only pool rows of the same place group strictly before the
sample; they are only meaningful when the place itself is monitored (known-site protocol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..cleaning.geo import to_xy

SCHEMES = ("city", "region", "plane", "idw", "plane_idw")
KERNEL_SCHEMES = ("idw", "plane_idw")


class NeighborPanel:
    """Place-by-place weight matrices for each scheme (diagonal and same-place weights are 0)."""

    def __init__(self, places: pd.DataFrame, idw_length_km: float):
        self.places = places
        self.ids = places.index.to_numpy()
        self.pos = pd.Series(np.arange(len(self.ids)), index=self.ids)
        xy = to_xy(places.lon, places.lat)
        dist_km = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000
        kernel = np.nan_to_num(np.exp(-dist_km / idw_length_km), nan=0.0)
        region, plane = places.region.to_numpy(), places.pressure_plane.to_numpy()
        same_region = (region[:, None] == region[None, :]) & (region[:, None] != "UNKNOWN")
        same_plane = (plane[:, None] == plane[None, :]) & (plane[:, None] != "UNKNOWN")
        self.weights = {"city": np.ones((len(self.ids),) * 2), "region": same_region.astype(float),
                        "plane": same_plane.astype(float), "idw": kernel, "plane_idw": kernel * same_plane}
        for w in self.weights.values():
            np.fill_diagonal(w, 0.0)
        self.dist_km = dist_km

    def aggregate(self, values: np.ndarray, mask: np.ndarray) -> dict[str, tuple[np.ndarray, ...]]:
        """values, mask: (bins x places) place means and availability -> per scheme (mean, n, wsum)."""
        filled = np.where(mask, values, 0.0)
        m = mask.astype(float)
        out = {}
        for name, w in self.weights.items():
            num, wsum, n = filled @ w.T, m @ w.T, m @ (w > 0).T
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(wsum > 0, num / wsum, np.nan)
            out[name] = (mean, n, wsum)
        return out


def _lookup(df, prefix, agg, bin_pos, place_pos, out):
    for name, (mean, n, wsum) in agg.items():
        out[f"{prefix}_{name}_mean"] = mean[bin_pos, place_pos]
        out[f"{prefix}_{name}_n"] = n[bin_pos, place_pos]
        if name in KERNEL_SCHEMES:
            out[f"{prefix}_{name}_w"] = wsum[bin_pos, place_pos]


def week_features(df: pd.DataFrame, pool: np.ndarray, panel: NeighborPanel, out: dict):
    weeks = np.sort(df.week.unique())
    wpos = pd.Series(np.arange(len(weeks)), index=weeks)
    pm = df[pool].groupby(["week", "place_group"]).y.mean()
    vals = np.full((len(weeks), len(panel.ids)), np.nan)
    vals[wpos[pm.index.get_level_values(0)].to_numpy(), panel.pos[pm.index.get_level_values(1)].to_numpy()] = pm.to_numpy()
    agg = panel.aggregate(vals, ~np.isnan(vals))
    _lookup(df, "wk", agg, wpos[df.week].to_numpy(), panel.pos[df.place_group].to_numpy(), out)


def trailing_features(df: pd.DataFrame, pool: np.ndarray, panel: NeighborPanel, out: dict, days: int = 7):
    """Each place's mean of its daily means over [day - days, day - 1], then neighbor aggregation."""
    start = df.day.min() - pd.Timedelta(days=days)
    all_days = pd.date_range(start, df.day.max(), freq="D")
    dpos = pd.Series(np.arange(len(all_days)), index=all_days)
    pm = df[pool].groupby(["day", "place_group"]).y.mean()
    s = np.zeros((len(all_days) + 1, len(panel.ids)))
    k = np.zeros_like(s)
    r, c = dpos[pm.index.get_level_values(0)].to_numpy() + 1, panel.pos[pm.index.get_level_values(1)].to_numpy()
    s[r, c], k[r, c] = pm.to_numpy(), 1.0
    cs, ck = np.cumsum(s, axis=0), np.cumsum(k, axis=0)
    # Row i of cs covers days [0, i-1]; window for day index d is days [d - days, d - 1].
    idx = np.arange(len(all_days))
    lo = np.clip(idx - days, 0, None)
    wsum_s, wsum_k = cs[idx] - cs[lo], ck[idx] - ck[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(wsum_k > 0, wsum_s / wsum_k, np.nan)
    agg = panel.aggregate(vals, wsum_k > 0)
    _lookup(df, "t7", agg, dpos[df.day].to_numpy(), panel.pos[df.place_group].to_numpy(), out)


def own_history(df: pd.DataFrame, pool: np.ndarray, out: dict, ref_col: str = "wk_city_mean", n_dev: int = 5):
    """Previous pool sample of the same place group strictly before each sample."""
    hist = df.loc[pool, ["place_group", "sampled_at", "y"]].copy()
    hist["dev"] = hist.y - out[ref_col][pool]
    hist = hist.sort_values("sampled_at")
    hist["dev_mean5"] = hist.groupby("place_group").dev.transform(lambda s: s.rolling(n_dev, min_periods=1).mean())
    hist = hist.rename(columns={"sampled_at": "prev_at", "y": "prev_y"})
    hist["sampled_at"] = hist.prev_at
    tgt = pd.DataFrame({"place_group": df.place_group.to_numpy(), "sampled_at": df.sampled_at.to_numpy(),
                        "i": np.arange(len(df))}).sort_values("sampled_at")
    m = pd.merge_asof(tgt, hist, on="sampled_at", by="place_group", allow_exact_matches=False,
                      direction="backward").sort_values("i")
    out["own_prev_y"] = m.prev_y.to_numpy()
    out["own_prev_dev"] = m.dev.to_numpy()
    out["own_dev_mean5"] = m.dev_mean5.to_numpy()
    out["own_prev_days"] = ((m.sampled_at - m.prev_at).dt.total_seconds() / 86400).to_numpy()


def build_features(df: pd.DataFrame, pool: np.ndarray, panel: NeighborPanel) -> pd.DataFrame:
    """Return df with neighbor and own-history features. `pool` marks rows of monitored places that
    may contribute values. A row's own value never enters its own features."""
    pool = np.asarray(pool, bool)
    out: dict[str, np.ndarray] = {}
    week_features(df, pool, panel, out)
    trailing_features(df, pool, panel, out)
    own_history(df, pool, out)
    feats = pd.DataFrame(out, index=df.index)
    return pd.concat([df, feats], axis=1)
