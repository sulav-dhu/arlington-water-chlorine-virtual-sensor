"""Reference models: the baseline ladder every new model must beat.

All are transparent "neighbor estimate + learned offset" predictors with no tuned hyperparameters
(settings are fixed in EvalConfig). Every model predicts every row it is given: when a neighbor
mean is unavailable it falls back to the city mean, then to the training mean for that calendar
month. Each model outputs a point prediction (`pred`) and quantiles (q10, q50, q90) from a shared
heteroscedastic residual calibration: training residuals are scaled by the residual spread of the
row's place (when the model may use site history and the place has enough rows) or pressure plane,
and the empirical quantiles of the scaled residuals are added back.

Code   Name                     Uses site history  Idea
B0     global_mean              no    training mean
B0c    climatology              no    training mean for the calendar month
B0p    persistence              yes   previous sample at the same place
B1     city_week                no    mean of the other places this week
B2     city_week+region         no    B1 + region offset
B2p    city_week+plane          no    B1 + pressure-plane offset (public data)
B3     city_week+site           yes   B1 + site offset
B4     region_week              no    same-week mean of same-region places, blended toward B2
B4p    plane_idw_week           no    same-week distance-weighted mean of same-plane places, blended toward B2p
B5     region_week+site         yes   B4 + site offset
B5p    plane_idw_week+site      yes   B4p + site offset
B5r    city_week+site+regime    yes   B3 + site offsets specific to Kubala / Pierce-Burch outages
B1c    city_7d                  no    B1 with the causal setting (7 days before the sample)
B4pc   plane_idw_7d             no    B4p with the causal setting
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EvalConfig


class Offsets:
    """Hierarchical empirical-Bayes offsets of residuals: global -> levels[0] -> levels[0:2] ...
    A group's offset is (sum of residuals + k * parent offset) / (n + k). A row whose group was
    not seen in training gets the offset of its deepest seen ancestor."""

    def __init__(self, levels: list[str], k: float):
        self.levels, self.k = levels, k

    def fit(self, df: pd.DataFrame, resid: np.ndarray) -> "Offsets":
        d = df[self.levels].copy()
        d["r"] = resid
        self.global_ = float(np.mean(resid)) if len(resid) else 0.0
        self.tables, parent = [], None
        for depth in range(len(self.levels)):
            keys = self.levels[: depth + 1]
            g = d.groupby(keys).r.agg(["sum", "count"])
            if depth == 0:
                prior = pd.Series(self.global_, index=g.index)
            else:
                pkeys = g.index.droplevel(-1)
                prior = pd.Series(parent.reindex(pkeys).to_numpy(), index=g.index)
            off = (g["sum"] + self.k * prior) / (g["count"] + self.k)
            self.tables.append(off)
            parent = off
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        val = np.full(len(df), self.global_)
        for depth, table in enumerate(self.tables):
            keys = self.levels[: depth + 1]
            idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
            found = table.reindex(idx).to_numpy()
            val = np.where(np.isnan(found), val, found)
        return val


class Calibrator:
    """Heteroscedastic residual quantiles: q = pred + scale(row) * quantile(resid / scale)."""

    def __init__(self, quantiles, levels: list[str], min_rows: int):
        self.quantiles, self.levels, self.min_rows = quantiles, levels, min_rows

    def _scale(self, df):
        s = np.full(len(df), self.global_sd)
        for lvl, table in zip(self.levels, self.tables):
            found = table.reindex(df[lvl]).to_numpy()
            s = np.where(np.isnan(found), s, found)
        return s

    def fit(self, df, resid) -> "Calibrator":
        d = df[self.levels].copy()
        d["r"] = resid
        self.global_sd = float(np.std(resid))
        self.tables = []
        for lvl in self.levels:
            g = d.groupby(lvl).r.agg(["std", "count"])
            self.tables.append(g.loc[g["count"] >= self.min_rows, "std"].clip(lower=0.02))
        z = resid / self._scale(df)
        self.zq = {q: float(np.quantile(z, q)) for q in self.quantiles}
        return self

    def predict(self, df, point) -> dict[str, np.ndarray]:
        s = self._scale(df)
        return {f"q{int(round(q * 100))}": point + s * z for q, z in self.zq.items()}


class Model:
    code = name = ""
    uses_site_history = False
    info = "wk"               # neighbor information setting: wk (same week) or t7 (prior 7 days)

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    # -- to override -------------------------------------------------------------------------
    def bind(self, pool: pd.DataFrame, panel) -> None:
        """Receive the neighbor pool (rows of monitored places, with features) before fitting. The
        harness guarantees the pool never contains held-out places or out-of-split periods."""

    def _fit(self, train: pd.DataFrame) -> None:
        pass

    def _point(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # -- shared machinery -------------------------------------------------------------------------
    def _city(self, df) -> np.ndarray:
        """City-wide neighbor mean, falling back to monthly climatology."""
        c = df[f"{self.info}_city_mean"].to_numpy()
        return np.where(np.isnan(c), self._clim(df), c)

    def _clim(self, df) -> np.ndarray:
        return df.month.map(self.month_mean).fillna(self.mean_).to_numpy()

    def fit(self, train: pd.DataFrame) -> "Model":
        self.mean_ = float(train.y.mean())
        self.month_mean = train.groupby("month").y.mean()
        self._fit(train)
        resid = train.y.to_numpy() - self._point(train)
        levels = ["pressure_plane", "place_group"] if self.uses_site_history else ["pressure_plane"]
        self.cal = Calibrator(self.cfg.quantiles, levels, self.cfg.scale_min_rows).fit(train, resid)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        point = self._point(df)
        out = pd.DataFrame({"pred": point}, index=df.index)
        for k, v in self.cal.predict(df, point).items():
            out[k] = v
        return out


class GlobalMean(Model):
    code, name = "B0", "global_mean"

    def _point(self, df):
        return np.full(len(df), self.mean_)


class Climatology(Model):
    code, name = "B0c", "climatology"

    def _point(self, df):
        return self._clim(df)


class Persistence(Model):
    code, name, uses_site_history = "B0p", "persistence", True

    def _point(self, df):
        p = df.own_prev_y.to_numpy()
        return np.where(np.isnan(p), self._clim(df), p)


class CityOffset(Model):
    """City neighbor mean + hierarchical offsets (none, region, plane, or site)."""

    def __init__(self, cfg, code, name, levels=(), info="wk", regimes=()):
        super().__init__(cfg)
        self.code, self.name, self.levels, self.info, self.regimes = code, name, list(levels), info, regimes
        self.uses_site_history = "place_group" in self.levels

    def _fit(self, train):
        base = self._city(train)
        self.off = Offsets(self.levels, self.cfg.offset_shrink_rows).fit(train, train.y.to_numpy() - base) \
            if self.levels else None
        resid = train.y.to_numpy() - base - (self.off.transform(train) if self.off else 0)
        self.regime_off = {}
        for col in self.regimes:            # extra site offsets that apply only during an outage
            m = train[col].to_numpy(bool)
            if m.any():
                self.regime_off[col] = Offsets(self.levels, self.cfg.offset_shrink_rows).fit(train[m], resid[m])

    def _point(self, df):
        p = self._city(df) + (self.off.transform(df) if self.off else 0)
        for col, off in self.regime_off.items():
            p = p + np.where(df[col].to_numpy(bool), off.transform(df), 0.0)
        return p


class GroupBlend(Model):
    """Same-week mean of a neighbor group (region, or same-plane distance-weighted), blended with a
    prior (city mean + group offset) by how much neighbor information there is; optional site
    offsets on top."""

    def __init__(self, cfg, code, name, scheme, prior_level, site=False, info="wk"):
        super().__init__(cfg)
        self.code, self.name, self.scheme, self.info = code, name, scheme, info
        self.prior_level, self.site = prior_level, site
        self.uses_site_history = site

    def _base(self, df):
        prior = self._city(df) + self.prior_off.transform(df)
        mean = df[f"{self.info}_{self.scheme}_mean"].to_numpy()
        if self.scheme in ("idw", "plane_idw"):
            amount, k = df[f"{self.info}_{self.scheme}_w"].to_numpy(), self.cfg.blend_k_weight
        else:
            amount, k = df[f"{self.info}_{self.scheme}_n"].to_numpy(), self.cfg.blend_k_places
        lam = np.where(np.isnan(mean), 0.0, amount / (amount + k))
        return lam * np.nan_to_num(mean) + (1 - lam) * prior

    def _fit(self, train):
        self.prior_off = Offsets([self.prior_level], self.cfg.offset_shrink_rows).fit(
            train, train.y.to_numpy() - self._city(train))
        self.site_off = Offsets([self.prior_level, "place_group"], self.cfg.offset_shrink_rows).fit(
            train, train.y.to_numpy() - self._base(train)) if self.site else None

    def _point(self, df):
        return self._base(df) + (self.site_off.transform(df) if self.site_off else 0)


def reference_models(cfg: EvalConfig) -> list[Model]:
    plane_site = ["pressure_plane", "place_group"]
    return [
        GlobalMean(cfg), Climatology(cfg), Persistence(cfg),
        CityOffset(cfg, "B1", "city_week"),
        CityOffset(cfg, "B2", "city_week+region", ["region"]),
        CityOffset(cfg, "B2p", "city_week+plane", ["pressure_plane"]),
        CityOffset(cfg, "B3", "city_week+site", plane_site),
        GroupBlend(cfg, "B4", "region_week", "region", "region"),
        GroupBlend(cfg, "B4p", "plane_idw_week", "plane_idw", "pressure_plane"),
        GroupBlend(cfg, "B5", "region_week+site", "region", "region", site=True),
        GroupBlend(cfg, "B5p", "plane_idw_week+site", "plane_idw", "pressure_plane", site=True),
        CityOffset(cfg, "B5r", "city_week+site+regime", plane_site, regimes=("jk_offline", "pb_offline")),
        CityOffset(cfg, "B1c", "city_7d", info="t7"),
        GroupBlend(cfg, "B4pc", "plane_idw_7d", "plane_idw", "pressure_plane", info="t7"),
    ]
