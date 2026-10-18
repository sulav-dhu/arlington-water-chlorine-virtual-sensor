"""Time periods, place-group folds and outage windows.

Periods: train <= train_end < val <= val_end < test. Models for the validation split are trained
on the train period; models for the test split are refit on train + val.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EvalConfig


def period(ts: pd.Series, cfg: EvalConfig) -> pd.Series:
    t = pd.to_datetime(ts)
    train_end = pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1)
    val_end = pd.Timestamp(cfg.val_end) + pd.Timedelta(days=1)
    return pd.Series(np.select([t < train_end, t < val_end], ["train", "val"], "test"), index=ts.index)


def fit_end(split: str, cfg: EvalConfig) -> pd.Timestamp:
    """Last timestamp (exclusive) a model may train on when evaluated on `split`."""
    end = cfg.train_end if split == "val" else cfg.val_end
    return pd.Timestamp(end) + pd.Timedelta(days=1)


def place_folds(places: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """Assign each place group to one of k folds, stratified by pressure plane and balanced by
    sample count (snake order over places sorted by size, random tie-breaks). Deterministic."""
    rng = np.random.default_rng(seed)
    fold = {}
    for _, g in places.groupby("pressure_plane", sort=True):
        order = g.assign(r=rng.random(len(g))).sort_values(["n_rows", "r"], ascending=[False, True]).index
        for i, pg in enumerate(order):
            rnd, pos = divmod(i, k)
            fold[pg] = pos if rnd % 2 == 0 else k - 1 - pos
    return pd.Series(fold, name="fold").reindex(places.index)


def outage_windows(events: pd.DataFrame, plant: str = "JK", before: str | None = None) -> pd.DataFrame:
    """Inferred offline periods of a plant (from sampling gaps only), optionally ending before a date."""
    ev = events[(events.plant == plant) & events.source.eq("inferred_from_plant_sampling_gaps")
                & events.boundary.isna()].copy()
    ev["start"], ev["end"] = pd.to_datetime(ev.start), pd.to_datetime(ev.end)
    if before is not None:
        ev = ev[ev.end < pd.Timestamp(before)]
    return ev[["event_id", "start", "end", "confidence"]].reset_index(drop=True)
