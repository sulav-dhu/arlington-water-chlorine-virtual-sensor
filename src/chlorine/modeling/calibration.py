"""Prediction-interval calibration on out-of-sample errors.

The Step 3 intervals were calibrated on in-sample training residuals; in-sample errors are smaller
than out-of-sample ones, and 2024-26 was volatile, so 80% intervals covered only 73-76% on test.
This module recalibrates on the out-of-sample predictions of the rolling backtest (backtest.py).

Methods (an 80% interval, q10 to q90)
  C0   in-sample   the model's own intervals (current method)
  C1u  conformal   weighted empirical quantiles of out-of-sample residuals y - pred, separately for
                   Kubala-outage weeks and normal weeks, recency-weighted (half-life 365 days)
  C1   conformal   the same on residuals scaled by the model's predictive SD, z = (y - pred) / sd,
                   so intervals widen where the model is less certain (sparse neighbors, no history)
  C2   adaptive    C1 plus adaptive conformal inference (Gibbs & Candès 2021): after each week, the
                   two tail levels move by gamma x (observed miss rate - 10%). For known sites the
                   feedback is the sites' own outcomes; for unmonitored addresses, which are never
                   observed, it is the coverage of the same model at monitored sites predicted as if
                   unmonitored (the known-site run of E7).

Study design: for each target year T in 2019-2023 the pool contains only out-of-sample predictions
from years before T; adaptive levels carry over from year to year as they would in operation.
Selection rule (fixed before the study was run): per protocol, the method with the lowest mean
interval score over 2019-2023 among those whose mean coverage lies in [0.75, 0.85]. The interval
score (Gneiting & Raftery 2007) is a proper scoring rule: width + (2/0.2) x distance outside.

    uv run python -m chlorine.modeling.calibration     # -> data/eval/calibration/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import EvalConfig

ALPHA = 0.2
P_LO, P_HI = 0.10, 0.90
GAMMAS = (0.01, 0.03, 0.1)
POOL_HALF_LIFE_DAYS = 365.0
MIN_REGIME_ROWS = 200
STUDY_YEARS = range(2019, 2024)
TARGETS = {"known_site": "E6", "unseen_site": "E7"}


def interval_score(y, lo, hi, alpha=ALPHA) -> np.ndarray:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


class ResidualPool:
    """Out-of-sample residuals (optionally SD-scaled), recency-weighted, split by plant regime."""

    def __init__(self, oos: pd.DataFrame, scaled: bool, ref_time, half_life_days: float = POOL_HALF_LIFE_DAYS):
        self.scaled = scaled
        z = (oos.y - oos.pred).to_numpy()
        if scaled:
            z = z / oos.sd.to_numpy()
        age = (pd.Timestamp(ref_time) - oos.sampled_at).dt.total_seconds().to_numpy() / 86400
        w = 0.5 ** (np.clip(age, 0, None) / half_life_days)
        regime = oos.jk_offline.to_numpy(bool)
        self.parts = {}
        for r in (False, True):
            m = regime == r if (regime == r).sum() >= MIN_REGIME_ROWS else np.ones(len(z), bool)
            order = np.argsort(z[m])
            zz, ww = z[m][order], w[m][order]
            cw = np.cumsum(ww)
            self.parts[r] = (zz, (cw - ww / 2) / cw[-1])          # midpoint plotting positions

    def quantile(self, p, regime: bool) -> float:
        zz, cdf = self.parts[regime]
        return float(np.interp(p, cdf, zz))

    def intervals(self, df: pd.DataFrame, p_lo=P_LO, p_hi=P_HI) -> tuple[np.ndarray, np.ndarray]:
        jk = df.jk_offline.to_numpy(bool)
        scale = df.sd.to_numpy() if self.scaled else 1.0
        zlo = np.where(jk, self.quantile(p_lo, True), self.quantile(p_lo, False))
        zhi = np.where(jk, self.quantile(p_hi, True), self.quantile(p_hi, False))
        return df.pred.to_numpy() + scale * zlo, df.pred.to_numpy() + scale * zhi


class AdaptiveLevels:
    """Adaptive conformal inference on the two tail levels, updated once per week."""

    def __init__(self, gamma: float):
        self.gamma, self.p_lo, self.p_hi = gamma, P_LO, P_HI

    def update(self, miss_lo: float, miss_hi: float):
        self.p_lo = float(np.clip(self.p_lo + self.gamma * (P_LO - miss_lo), 0.005, 0.45))
        self.p_hi = float(np.clip(self.p_hi + self.gamma * (miss_hi - (1 - P_HI)), 0.55, 0.995))


def adaptive_intervals(target: pd.DataFrame, signal: pd.DataFrame, pool: ResidualPool, levels: AdaptiveLevels):
    """Intervals for `target`; after each week, levels are updated from `signal`'s misses that week."""
    lo, hi = np.full(len(target), np.nan), np.full(len(target), np.nan)
    tw, sw = target.week.to_numpy(), signal.week.to_numpy()
    for wk in np.unique(np.concatenate([tw, sw])):
        t = tw == wk
        if t.any():
            lo[t], hi[t] = pool.intervals(target[t], levels.p_lo, levels.p_hi)
        s = sw == wk
        if s.any():
            slo, shi = pool.intervals(signal[s], levels.p_lo, levels.p_hi)
            ys = signal.y.to_numpy()[s]
            levels.update(float(np.mean(ys < slo)), float(np.mean(ys > shi)))
    return lo, hi


def _metrics(y, lo, hi, jk) -> dict:
    cover = (y >= lo) & (y <= hi)
    return {"n": len(y), "coverage": float(cover.mean()), "coverage_outage": float(cover[jk].mean()) if jk.any() else np.nan,
            "coverage_normal": float(cover[~jk].mean()), "width": float(np.mean(hi - lo)),
            "interval_score": float(np.mean(interval_score(y, lo, hi)))}


def study(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, code in TARGETS.items():
        preds = dev[(dev.protocol == protocol) & (dev.model == code)].sort_values("sampled_at")
        signal_all = dev[(dev.protocol == "known_site") & (dev.model == code)].sort_values("sampled_at")
        levels = {g: AdaptiveLevels(g) for g in GAMMAS}
        for T in STUDY_YEARS:
            pool_rows, target = preds[preds.year < T], preds[preds.year == T]
            signal = signal_all[signal_all.year == T]
            y, jk = target.y.to_numpy(), target.jk_offline.to_numpy(bool)
            ref = pd.Timestamp(f"{T}-01-01")
            out = {"C0": (target.q10.to_numpy(), target.q90.to_numpy())}
            out["C1u"] = ResidualPool(pool_rows, False, ref).intervals(target)
            pool = ResidualPool(pool_rows, True, ref)
            out["C1"] = pool.intervals(target)
            for g, lv in levels.items():
                out[f"C2 (gamma={g})"] = adaptive_intervals(target, signal, pool, lv)
            for method, (lo, hi) in out.items():
                rows.append({"protocol": protocol, "model": code, "year": T, "method": method, **_metrics(y, lo, hi, jk)})
    return pd.DataFrame(rows)


def select(res: pd.DataFrame) -> dict:
    summary = res.groupby(["protocol", "method"]).agg(
        coverage=("coverage", "mean"), coverage_outage=("coverage_outage", "mean"), width=("width", "mean"),
        interval_score=("interval_score", "mean"), worst_year_coverage=("coverage", "min")).reset_index()
    chosen = {}
    for protocol, g in summary.groupby("protocol"):
        ok = g[g.coverage.between(0.75, 0.85)]
        best = (ok if len(ok) else g).sort_values("interval_score").iloc[0]
        chosen[protocol] = {"method": best.method, **{k: float(best[k]) for k in
                                                       ("coverage", "coverage_outage", "width", "interval_score")}}
    return {"summary": summary, "chosen": chosen}


def main():
    cfg = EvalConfig()
    dev = pd.read_parquet(cfg.out_root / "backtest" / "dev.parquet")
    dev["week"] = dev.sampled_at.dt.to_period("W-SUN").dt.start_time
    out_dir = cfg.out_root / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = study(dev)
    sel = select(res)
    res.to_csv(out_dir / "study.csv", index=False)
    sel["summary"].to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "selection.json").write_text(json.dumps(sel["chosen"], indent=2))
    print(sel["summary"].round(3).to_string(index=False))
    print(json.dumps(sel["chosen"], indent=2))


if __name__ == "__main__":
    main()
