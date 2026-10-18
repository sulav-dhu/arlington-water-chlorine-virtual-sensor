"""Evaluation protocols.

known_site   Predict samples in the evaluation period at places of the monitored network, from
             same-week neighbors (+ the place's own history, for models that use it). Places with no
             training-period samples ("new sites") are flagged and reported separately.
unseen_site  k-fold cross-validation over place groups (stratified by pressure plane): models train
             on the other folds' places in the training period and predict the held-out places in the
             evaluation period. Held-out places are excluded from the neighbor pool, so they behave
             like unmonitored locations. Models that need site history are skipped.
outage       Leave-one-outage-out over Kubala outages before the test period: train on all pre-test
             data outside the outage window, predict every sample inside it.

Test-period rows are dropped before anything else when evaluating on the validation split, and the
outage protocol never uses test-period data, so development runs cannot touch the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EvalConfig
from .features import NeighborPanel, build_features
from .models import Model
from .splits import fit_end, period

KEEP = ["row_id", "place_group", "location_id", "region", "pressure_plane", "high_frequency_site", "jk_offline",
        "pb_offline", "sampled_at", "y"]


def _collect(model: Model, rows: pd.DataFrame, protocol: str, split: str, **extra) -> pd.DataFrame:
    p = model.predict(rows)
    out = rows[KEEP].copy()
    out[p.columns] = p.to_numpy()
    out.insert(0, "model", model.code)
    out.insert(1, "model_name", model.name)
    out.insert(2, "protocol", protocol)
    out.insert(3, "split", split)
    for k, v in extra.items():
        out[k] = v
    return out


def development_frame(df: pd.DataFrame, split: str, cfg: EvalConfig) -> pd.DataFrame:
    """For the validation split, remove the test period entirely."""
    if split == "test":
        return df
    return df[period(df.sampled_at, cfg).ne("test")].reset_index(drop=True)


def run_known_site(df, panel: NeighborPanel, models, split: str, cfg: EvalConfig) -> pd.DataFrame:
    df = development_frame(df, split, cfg)
    feats = build_features(df, np.ones(len(df), bool), panel)
    train = feats[feats.sampled_at < fit_end(split, cfg)]
    ev = feats[period(feats.sampled_at, cfg).eq(split)]
    new_site = ~ev.place_group.isin(set(train.place_group))
    out = []
    for m in models:
        m.bind(feats, panel)
        m.fit(train)
        out.append(_collect(m, ev, "known_site", split, new_site=new_site.to_numpy()))
    return pd.concat(out, ignore_index=True)


def run_unseen_site(df, panel: NeighborPanel, models, split: str, cfg: EvalConfig, folds: pd.Series,
                    train_places: set | None = None, only_folds: tuple[int, ...] | None = None) -> pd.DataFrame:
    """train_places optionally restricts which non-held-out places are monitored (learning curves);
    only_folds restricts which folds are run (tests)."""
    df = development_frame(df, split, cfg)
    models = [m for m in models if not m.uses_site_history]
    fold = df.place_group.map(folds).to_numpy()
    out = []
    for f in np.unique(fold[~np.isnan(fold)]).astype(int):
        if only_folds is not None and f not in only_folds:
            continue
        held = fold == f
        pool = ~held if train_places is None else (~held & df.place_group.isin(train_places).to_numpy())
        feats = build_features(df, pool, panel)
        train = feats[pool & (feats.sampled_at < fit_end(split, cfg)).to_numpy()]
        ev = feats[held & period(feats.sampled_at, cfg).eq(split).to_numpy()]
        if ev.empty:
            continue
        for m in models:
            m.bind(feats[pool], panel)
            m.fit(train)
            out.append(_collect(m, ev, "unseen_site", split, fold=f))
    return pd.concat(out, ignore_index=True)


def run_outage(df, panel: NeighborPanel, models, cfg: EvalConfig, windows: pd.DataFrame) -> pd.DataFrame:
    df = development_frame(df, "val", cfg)
    feats = build_features(df, np.ones(len(df), bool), panel)
    day = feats.sampled_at.dt.normalize()
    out = []
    for w in windows.itertuples():
        inside = ((day >= w.start) & (day <= w.end)).to_numpy()
        train, ev = feats[~inside], feats[inside]
        for m in models:
            m.bind(feats, panel)
            m.fit(train)
            out.append(_collect(m, ev, "outage", "pre_test", event_id=w.event_id, event_confidence=w.confidence))
    return pd.concat(out, ignore_index=True)
