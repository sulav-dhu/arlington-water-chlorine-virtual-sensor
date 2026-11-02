"""Spatial covariance estimation and simple kriging over places.

Covariance model between two places at distance d (km):
    C(d, same_plane) = sigma2 * exp(-d / length_km) * (1 if same pressure plane else cross_plane)
sigma2 is the spatially shared variance, length_km its range, and cross_plane in [0, 1] how much
less two places co-vary when they are in different pressure planes (a public pipe-network fact).
Parameters are fitted by weighted least squares to binned empirical covariances (method of moments,
pairs within max_km). Observations carry their own noise variance on the diagonal (unshared
place-level variation + sampling noise of a mean of n samples).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Covariance:
    sigma2: float
    length_km: float
    cross_plane: float

    def __call__(self, dist_km: np.ndarray, same_plane: np.ndarray) -> np.ndarray:
        return self.sigma2 * np.exp(-dist_km / self.length_km) * np.where(same_plane, 1.0, self.cross_plane)


def binned_pairs(dist_km, same_plane, products, weights, max_km: float, edges=None):
    """Average pair products into (distance bin, same-plane) cells -> arrays for fitting. Each cell is
    placed at the mean distance of its pairs, not the bin midpoint (pairs are not spread evenly within
    a bin; using midpoints biases the fitted variance and range)."""
    edges = np.array([0, 1, 2, 3, 4, 5, 6.5, 8, 10, 12]) if edges is None else edges
    edges = edges[edges <= max_km + 1e-9]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for s in (True, False):
            m = (dist_km > lo) & (dist_km <= hi) & (same_plane == s)
            if weights[m].sum() > 0:
                out.append((np.average(dist_km[m], weights=weights[m]), s,
                            np.average(products[m], weights=weights[m]), weights[m].sum()))
    return np.array(out, dtype=float) if out else np.zeros((0, 4))


def fit_covariance(cells: np.ndarray, fix_cross_plane: float | None = None) -> Covariance:
    """cells: rows (mid_km, same_plane, mean_product, weight)."""
    d, s, c, w = cells.T
    sw = np.sqrt(w / w.sum())

    def resid(p):
        sig, ell, rho = p[0], p[1], (fix_cross_plane if fix_cross_plane is not None else p[2])
        return sw * (Covariance(sig, ell, rho)(d, s.astype(bool)) - c)

    c0 = min(max(float(np.max(c)), 1e-4), 9.0)          # starting guess must lie inside the bounds
    fit = least_squares(resid, x0=[c0, 2.0, 0.6], bounds=([1e-6, 0.2, 0.0], [10.0, 50.0, 1.0]))
    rho = fix_cross_plane if fix_cross_plane is not None else float(fit.x[2])
    return Covariance(float(fit.x[0]), float(fit.x[1]), rho)


class Kriger:
    """Simple (zero-mean) kriging over places.

    dist_km, same_plane: (P x P) matrices for all places; group: place -> exclusion group id (a place
    never uses observations from its own group, e.g. the same physical place under another code).
    """

    def __init__(self, cov: Covariance, dist_km: np.ndarray, same_plane: np.ndarray, group: np.ndarray):
        self.cov, self.dist, self.same, self.group = cov, dist_km, same_plane, group
        self.has_xy = ~np.isnan(dist_km).all(axis=1)

    def predict(self, obs_idx: np.ndarray, obs_val: np.ndarray, obs_noise: np.ndarray, target_idx: np.ndarray):
        """Kriged values and kriging variances at target places from noisy observations at obs places.
        Returns (mean, var); targets without coordinates or usable neighbors get (0, sigma2)."""
        ok = self.has_xy[obs_idx]
        obs_idx, obs_val, obs_noise = obs_idx[ok], obs_val[ok], obs_noise[ok]
        mean = np.zeros(len(target_idx))
        var = np.full(len(target_idx), self.cov.sigma2)
        if len(obs_idx) == 0:
            return mean, var
        K = self.cov(self.dist[np.ix_(obs_idx, obs_idx)], self.same[np.ix_(obs_idx, obs_idx)]) + np.diag(obs_noise)
        obs_groups = self.group[obs_idx]
        cache = {}
        for t, p in enumerate(target_idx):
            if not self.has_xy[p]:
                continue
            keep = obs_groups != self.group[p]
            if not keep.any():
                continue
            key = keep.tobytes()
            if key not in cache:
                cache[key] = np.linalg.cholesky(K[np.ix_(keep, keep)] + 1e-9 * np.eye(int(keep.sum())))
            L = cache[key]
            k = self.cov(self.dist[p, obs_idx[keep]], self.same[p, obs_idx[keep]])
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, k))
            mean[t] = alpha @ obs_val[keep]
            var[t] = max(self.cov.sigma2 - k @ alpha, 0.0)
        return mean, var
