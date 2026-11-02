"""Gradient-boosting comparator: does a flexible black box beat the structured model?

Learns y - city_week from same-week neighbor contrasts (region, plane, distance-weighted), public
location features (plant distances, hydrant features, coordinates, pressure plane), plant regime and
season; optionally the utility region code and, for site-history variants, the place's own recent
deviations plus empirical-Bayes site offsets on the residuals. Fixed, conservative settings (no
tuning on validation or test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..evaluation.models import Model, Offsets

PLANE_CODES = {"LOWER": 0, "UPPER": 1, "WEST": 2, "UNKNOWN": 3}
REGION_CODES = {"C": 0, "E": 1, "N": 2, "SE": 3, "SW": 4, "W": 5, "UNKNOWN": 6}
STATIC = ["dist_jk_km", "dist_pb_km", "flush_program_share", "hydrant_median_year", "hydrants_in_radius",
          "in_davis_phase2_area", "lat", "lon"]


class GBM(Model):
    def __init__(self, cfg, code, name, *, site_history: bool, region: bool):
        super().__init__(cfg)
        self.code, self.name, self.uses_site_history, self.region = code, name, site_history, region

    def _X(self, df) -> tuple[np.ndarray, list[bool]]:
        city = self._city(df)
        cols, cat = {}, []
        cols["city"] = city
        for scheme, amt in (("region", "n"), ("plane", "n"), ("idw", "w"), ("plane_idw", "w")):
            cols[f"d_{scheme}"] = df[f"wk_{scheme}_mean"].to_numpy() - city
            cols[f"amt_{scheme}"] = df[f"wk_{scheme}_{amt}"].to_numpy()
        for c in STATIC:
            cols[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        cols["jk_off"], cols["pb_off"] = df.jk_offline.to_numpy(float), df.pb_offline.to_numpy(float)
        doy = df.sampled_at.dt.dayofyear.to_numpy()
        cols["season_sin"], cols["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        cols["plane"] = df.pressure_plane.map(PLANE_CODES).to_numpy(float)
        if self.region:
            cols["region"] = df.region.map(REGION_CODES).to_numpy(float)
        if self.uses_site_history:
            cols["own_dev_mean5"], cols["own_prev_dev"] = df.own_dev_mean5.to_numpy(), df.own_prev_dev.to_numpy()
            cols["own_prev_days"] = df.own_prev_days.to_numpy()
        X = pd.DataFrame(cols)
        cat = [c in ("plane", "region") for c in X.columns]
        return X.to_numpy(float), cat

    def _fit(self, train):
        X, cat = self._X(train)
        r = train.y.to_numpy() - self._city(train)
        self.model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=1.0, categorical_features=cat, random_state=0)
        self.model.fit(X, r)
        self.site_off = None
        if self.uses_site_history:
            self.site_off = Offsets(["pressure_plane", "place_group"], self.cfg.offset_shrink_rows).fit(
                train, r - self.model.predict(X))

    def _point(self, df):
        X, _ = self._X(df)
        p = self._city(df) + self.model.predict(X)
        return p + (self.site_off.transform(df) if self.site_off else 0)
