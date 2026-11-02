"""Regression kriging with hierarchical site effects (models B6, B7 and ablations).

    y = city_week + X·beta + s(place) + a(place, week) + noise

city_week  mean of the other places this week (the network level)
X·beta     public fixed effects (pressure plane, plant distances, hydrant features, plant-regime
           interactions, season; optionally the utility region code), recency-weighted ridge
s(place)   static site level. Its prior is kriged from other places' levels (spatial covariance
           fitted to the data, distance x same-plane); places with their own history combine that
           prior with their recency-weighted mean residual (Bayesian update). Models without site
           history use the prior only, as for an unmonitored address.
a(place,w) this week's anomaly, kriged from other places' same-week anomalies
noise      within-week scatter
Site-history models also add alpha x the place's own recent anomalies (decayed over ~4 weeks).

beta and s are estimated by backfitting. Every variance component is estimated from training data,
so each prediction has a standard deviation; intervals are its empirical standardized-residual
quantiles, separately for Kubala-outage periods (more volatile).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.models import Model
from .design import Standardizer, raw_design, weighted_ridge
from .spatial import Covariance, Kriger, binned_pairs, fit_covariance

DESIGN_EFFECT = 2.0     # within-site residuals are autocorrelated (lag-1 week ~0.36): halve effective n
OWN_DECAY_DAYS = 28.0
OWN_WINDOW_DAYS = 56


def cap_to_budget(cov: Covariance, budget: float, max_share: float = 0.9) -> Covariance:
    """The shared spatial variance cannot exceed the measured variance left after sampling noise; keep
    at least 10% of it as unshared place-level variation so very close neighbors are not treated as
    exact copies of the target."""
    return Covariance(min(cov.sigma2, max(max_share * budget, 1e-6)), cov.length_km, cov.cross_plane)


class RegressionKriging(Model):
    def __init__(self, cfg, code, name, *, site_history: bool, region=False, regime=True, season=True,
                 static_kriging=True, anomaly_kriging=True, own_history=True, cross_plane: float | None = None,
                 half_life_days: float = 730.0, ridge_lambda: float = 1.0, backfit_iters: int = 3):
        super().__init__(cfg)
        self.code, self.name, self.uses_site_history = code, name, site_history
        self.region, self.regime, self.season = region, regime, season
        self.static_kriging, self.anomaly_kriging = static_kriging, anomaly_kriging
        self.own_history = own_history and site_history
        self.cross_plane, self.half_life_days = cross_plane, half_life_days
        self.ridge_lambda, self.backfit_iters = ridge_lambda, backfit_iters

    # -- context ----------------------------------------------------------------------------------
    def bind(self, pool: pd.DataFrame, panel) -> None:
        self.pool, self.panel = pool, panel
        plane = panel.places.pressure_plane.to_numpy()
        self.same = (plane[:, None] == plane[None, :]) & (plane[:, None] != "UNKNOWN")
        self.group = np.arange(len(panel.ids))

    def _pidx(self, df):
        return self.panel.pos[df.place_group].to_numpy()

    def _recency(self, t: pd.Series) -> np.ndarray:
        if not np.isfinite(self.half_life_days):
            return np.ones(len(t))
        age = (self.t_end - t).dt.total_seconds().to_numpy() / 86400
        return 0.5 ** (np.clip(age, 0, None) / self.half_life_days)

    def _fixed(self, df) -> np.ndarray:
        X = self.std.transform(raw_design(df, self.region, self.regime, self.season))
        return self.b0 + X @ self.beta

    # -- static site levels ----------------------------------------------------------------------------
    def _site_levels(self, pidx, e, w):
        P = len(self.panel.ids)
        sw, swe, sw2 = (np.bincount(pidx, x, minlength=P) for x in (w, w * e, w ** 2))
        has = sw > 0
        ebar = np.where(has, swe / np.where(has, sw, 1), 0.0)
        n_eff = np.where(has, sw ** 2 / np.where(has, sw2, 1), 0.0) / DESIGN_EFFECT
        self.sigma_e2 = float(np.sum(w * (e - ebar[pidx]) ** 2) / np.sum(w))
        noise = np.where(n_eff > 0, self.sigma_e2 / np.maximum(n_eff, 1e-9), np.inf)
        has_xy = ~np.isnan(self.panel.dist_km).all(axis=1)
        obs = has & (n_eff >= 5) & has_xy
        self.mu_s = float(np.average(ebar[obs], weights=1 / noise[obs])) if obs.any() else 0.0
        o = ebar[obs] - self.mu_s
        V_s = float(np.var(o)) if obs.sum() > 1 else 0.05
        if self.static_kriging and obs.sum() >= 10:
            idx = np.flatnonzero(obs)
            iu = np.triu_indices(len(idx), 1)
            D = self.panel.dist_km[np.ix_(idx, idx)][iu]
            S = self.same[np.ix_(idx, idx)][iu]
            cells = binned_pairs(D, S, np.outer(o, o)[iu], np.ones(len(D)), max_km=12)
            self.cov_s = cap_to_budget(fit_covariance(cells, self.cross_plane), V_s - float(np.mean(noise[obs])))
            self.tau_s2 = max(V_s - self.cov_s.sigma2 - float(np.mean(noise[obs])), 0.0)
            kr = Kriger(self.cov_s, self.panel.dist_km, self.same, self.group)
            m, v = kr.predict(idx, o, noise[obs] + self.tau_s2, np.arange(P))
            prior_mean, prior_var = self.mu_s + m, v + self.tau_s2
        else:
            self.cov_s = None
            prior_mean, prior_var = np.full(P, self.mu_s), np.full(P, max(V_s, 1e-4))
        prec = np.where(has, n_eff / self.sigma_e2, 0.0) + 1 / prior_var
        self.s_prior, self.s_prior_var = prior_mean, prior_var
        self.s_post = np.where(has, (ebar * n_eff / self.sigma_e2 + prior_mean / prior_var) / prec, prior_mean)
        self.s_post_var = 1 / prec
        self.has_data = has

    # -- weekly anomalies --------------------------------------------------------------------------------
    def _pool_anomalies(self):
        pool = self.pool
        a = pool.y.to_numpy() - self._city(pool) - self._fixed(pool) - self.s_post[self._pidx(pool)]
        self.pool_a = pd.DataFrame({"week": pool.week.to_numpy(), "p": self._pidx(pool), "t": pool.sampled_at.to_numpy(),
                                    "a": a})
        self.pool_a["y"] = pool.y.to_numpy()
        pw = self.pool_a.groupby(["week", "p"]).agg(a=("a", "mean"), n=("a", "size"), ym=("y", "mean")).reset_index()
        # Per week: places, mean anomaly, sample count, mean value (the city baseline is built from these).
        self.pw = {wk: (g.p.to_numpy(), g.a.to_numpy(), g.n.to_numpy(), g.ym.to_numpy()) for wk, g in pw.groupby("week")}
        # Variance components from the training period only.
        tr = self.pool_a[self.pool_a.t <= np.datetime64(self.t_end)]
        g = tr.groupby(["week", "p"]).a
        means, sizes = g.mean(), g.size()
        dev = tr.a - g.transform("mean")
        multi = g.transform("size").to_numpy() > 1
        dof = multi.sum() - (sizes > 1).sum()          # sum over place-weeks of (n - 1)
        self.sigma_w2 = float((dev[multi] ** 2).sum() / dof)
        V_a = float(means.var())
        if self.anomaly_kriging:
            prods, D, S = [], [], []
            for _, sub in means.groupby(level=0):
                idx = sub.index.get_level_values(1).to_numpy()
                if len(idx) < 2:
                    continue
                iu = np.triu_indices(len(idx), 1)
                d = self.panel.dist_km[np.ix_(idx, idx)][iu]
                keep = d <= 8
                prods.append(np.outer(sub.to_numpy(), sub.to_numpy())[iu][keep])
                D.append(d[keep])
                S.append(self.same[np.ix_(idx, idx)][iu][keep])
            D, S, prods = np.concatenate(D), np.concatenate(S), np.concatenate(prods)
            budget = V_a - float(np.mean(self.sigma_w2 / sizes))
            self.cov_a = cap_to_budget(fit_covariance(binned_pairs(D, S, prods, np.ones(len(D)), max_km=8),
                                                      self.cross_plane), budget)
            self.tau_a2 = max(budget - self.cov_a.sigma2, 0.0)
            self.kriger_a = Kriger(self.cov_a, self.panel.dist_km, self.same, self.group)
        else:
            self.cov_a, self.tau_a2 = None, max(V_a - float(np.mean(self.sigma_w2 / sizes)), 0.0)

    def _krige_anomalies(self, df):
        mean, var = np.zeros(len(df)), np.full(len(df), self.cov_a.sigma2 if self.cov_a else 0.0)
        if not self.anomaly_kriging:
            return mean, var
        pidx = self._pidx(df)
        for wk, rows in pd.Series(np.arange(len(df))).groupby(df.week.to_numpy()):
            if wk not in self.pw:
                continue
            obs_p, obs_a, obs_n, obs_y = self.pw[wk]
            noise = self.tau_a2 + self.sigma_w2 / obs_n
            targets = np.unique(pidx[rows.to_numpy()])
            in_pool = np.isin(targets, obs_p)
            lut_m, lut_v = {}, {}
            if (~in_pool).any():
                m, v = self.kriger_a.predict(obs_p, obs_a, noise, targets[~in_pool])
                lut_m.update(zip(targets[~in_pool], m))
                lut_v.update(zip(targets[~in_pool], v))
            # A target that is itself in the pool contributed to every neighbor's city baseline (the mean of
            # the other places that week). Rebuild those baselines without it, so the target's own value can
            # not flow back through its neighbors' anomalies: city_q = (S - y_q) / (N - 1) becomes
            # (S - y_q - y_p) / (N - 2).
            S, N = obs_y.sum(), len(obs_y)
            for p in targets[in_pool]:
                if N <= 2:
                    lut_m[p], lut_v[p] = 0.0, self.cov_a.sigma2
                    continue
                y_p = obs_y[obs_p == p][0]
                corrected = obs_a + (S - obs_y) / (N - 1) - (S - obs_y - y_p) / (N - 2)
                m, v = self.kriger_a.predict(obs_p, corrected, noise, np.array([p]))
                lut_m[p], lut_v[p] = m[0], v[0]
            r = rows.to_numpy()
            mean[r] = [lut_m[p] for p in pidx[r]]
            var[r] = [lut_v[p] for p in pidx[r]]
        return mean, var

    def _own_recent(self, df) -> np.ndarray:
        """Decayed mean of the place's own pool anomalies in the 8 weeks strictly before each sample."""
        out = np.full(len(df), np.nan)
        if not self.own_history:
            return out
        by_p = {p: (g.t.to_numpy(), g.a.to_numpy()) for p, g in self.pool_a.sort_values("t").groupby("p")}
        pidx, times = self._pidx(df), df.sampled_at.to_numpy()
        for p in np.unique(pidx):
            if p not in by_p:
                continue
            ts, av = by_p[p]
            rows = np.flatnonzero(pidx == p)
            hi = np.searchsorted(ts, times[rows], side="left")
            lo = np.searchsorted(ts, times[rows] - np.timedelta64(OWN_WINDOW_DAYS, "D"), side="left")
            for r, a0, a1 in zip(rows, lo, hi):
                if a1 > a0:
                    age = (times[r] - ts[a0:a1]) / np.timedelta64(1, "D")
                    w = np.exp(-age / OWN_DECAY_DAYS)
                    out[r] = np.sum(w * av[a0:a1]) / np.sum(w)
        return out

    # -- fit / predict -----------------------------------------------------------------------------------
    def _components(self, df):
        pidx = self._pidx(df)
        base = self._city(df) + self._fixed(df)
        if self.uses_site_history:
            s, s_var = self.s_post[pidx], self.s_post_var[pidx]
        else:
            s, s_var = self.s_prior[pidx], self.s_prior_var[pidx]
        a, a_var = self._krige_anomalies(df)
        own = self._own_recent(df)
        return base, s, s_var, a, a_var, own

    def fit(self, train: pd.DataFrame) -> "RegressionKriging":
        self.mean_ = float(train.y.mean())
        self.month_mean = train.groupby("month").y.mean()
        self.t_end = train.sampled_at.max()
        r = train.y.to_numpy() - self._city(train)
        Xr = raw_design(train, self.region, self.regime, self.season)
        self.std = Standardizer().fit(Xr)
        X = self.std.transform(Xr)
        w = self._recency(train.sampled_at)
        pidx = self._pidx(train)
        s_row = np.zeros(len(train))
        for _ in range(self.backfit_iters):
            self.b0, self.beta = weighted_ridge(X, r - s_row, w, self.ridge_lambda)
            self._site_levels(pidx, r - self.b0 - X @ self.beta, w)
            s_row = self.s_post[pidx]
        self._pool_anomalies()
        base, s, s_var, a, a_var, own = self._components(train)
        resid = train.y.to_numpy() - base - s - a
        ok = ~np.isnan(own)
        self.alpha = float(np.clip(np.sum(own[ok] * resid[ok]) / np.sum(own[ok] ** 2), 0, 1)) if ok.sum() > 50 else 0.0
        pred = base + s + a + self.alpha * np.nan_to_num(own)
        sd = np.sqrt(s_var + a_var + self.tau_a2 + self.sigma_w2)
        z = (train.y.to_numpy() - pred) / sd
        self.zq = {}
        for regime in (False, True):
            m = train.jk_offline.to_numpy(bool) == regime
            zz, ww = (z[m], w[m]) if m.sum() >= 200 else (z, w)
            order = np.argsort(zz)
            cw = np.cumsum(ww[order]) / ww.sum()
            self.zq[regime] = {q: float(zz[order][min(np.searchsorted(cw, q), len(zz) - 1)]) for q in self.cfg.quantiles}
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        base, s, s_var, a, a_var, own = self._components(df)
        pred = base + s + a + self.alpha * np.nan_to_num(own)
        sd = np.sqrt(s_var + a_var + self.tau_a2 + self.sigma_w2)
        out = pd.DataFrame({"pred": pred, "sd": sd}, index=df.index)
        jk = df.jk_offline.to_numpy(bool)
        for q in self.cfg.quantiles:
            zq = np.where(jk, self.zq[True][q], self.zq[False][q])
            out[f"q{int(round(q * 100))}"] = pred + sd * zq
        return out

    def describe(self) -> dict:
        d = {"alpha_own": self.alpha, "sigma_w": self.sigma_w2 ** 0.5, "tau_a": self.tau_a2 ** 0.5,
             "sigma_e": self.sigma_e2 ** 0.5}
        for tag, c in (("static", self.cov_s), ("anomaly", self.cov_a)):
            if c is not None:
                d.update({f"{tag}_sd": c.sigma2 ** 0.5, f"{tag}_range_km": c.length_km, f"{tag}_cross_plane": c.cross_plane})
        return d
