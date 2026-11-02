"""Fixed-effects design matrix for the Step 3 models.

Columns (all available for any address, i.e. public data, unless noted):
  plane      pressure-plane indicators (LOWER is the base level)
  static     distance to each plant, hydrant-derived features, Davis expansion-area flag
  regime     Kubala / Pierce-Burch offline flags and their interactions with plane and plant distance
             (an outage changes which plant feeds which neighborhood)
  season     annual sine/cosine
  region     the utility's site-code region (optional; not public)
Numeric columns are imputed with the training median plus a missing indicator, then standardized
with training statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATIC = ["dist_jk_km", "dist_pb_km", "flush_program_share", "hydrant_median_year", "hydrants_in_radius",
          "in_davis_phase2_area"]
PLANES = ["UPPER", "WEST", "UNKNOWN"]
REGIONS = ["E", "N", "SE", "SW", "W", "UNKNOWN"]       # C is the base level


def raw_design(df: pd.DataFrame, region: bool, regime: bool, season: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for p in PLANES:
        X[f"plane_{p}"] = df.pressure_plane.eq(p).astype(float)
    for c in STATIC:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    X["hydrants_in_radius"] = np.log1p(X.hydrants_in_radius)
    if regime:
        jk, pb = df.jk_offline.astype(float), df.pb_offline.astype(float)
        X["jk_off"], X["pb_off"] = jk, pb
        X["jk_off_x_dist_pb"], X["pb_off_x_dist_jk"] = jk * X.dist_pb_km, pb * X.dist_jk_km
        X["jk_off_x_dist_jk"], X["pb_off_x_dist_pb"] = jk * X.dist_jk_km, pb * X.dist_pb_km
        for p in ("UPPER", "WEST"):
            X[f"jk_off_x_{p}"], X[f"pb_off_x_{p}"] = jk * X[f"plane_{p}"], pb * X[f"plane_{p}"]
    if season:
        doy = df.sampled_at.dt.dayofyear.to_numpy()
        X["season_sin"], X["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if region:
        for r in REGIONS:
            X[f"region_{r}"] = df.region.eq(r).astype(float)
    return X


class Standardizer:
    """Median imputation (+ missing indicators for columns with gaps in training) and z-scoring."""

    def fit(self, X: pd.DataFrame) -> "Standardizer":
        self.cols = list(X.columns)
        self.median = X.median()
        self.missing_cols = [c for c in self.cols if X[c].isna().any()]
        Z = self._impute(X)
        self.mean, self.sd = Z.mean(), Z.std().replace(0, 1.0).fillna(1.0)
        return self

    def _impute(self, X):
        Z = X[self.cols].fillna(self.median)
        for c in self.missing_cols:
            Z[f"{c}_missing"] = X[c].isna().astype(float)
        return Z

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._impute(X)
        return ((Z - self.mean.reindex(Z.columns).fillna(0)) / self.sd.reindex(Z.columns).fillna(1)).to_numpy()

    @property
    def names(self) -> list[str]:
        return self.cols + [f"{c}_missing" for c in self.missing_cols]


def weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Weighted ridge with an unpenalized intercept -> (intercept, coefficients)."""
    sw = w / w.sum()
    xm, ym = sw @ X, sw @ y
    Xc, yc = X - xm, y - ym
    A = (Xc * w[:, None]).T @ Xc + lam * np.eye(X.shape[1])
    beta = np.linalg.solve(A, (Xc * w[:, None]).T @ yc)
    return float(ym - xm @ beta), beta
