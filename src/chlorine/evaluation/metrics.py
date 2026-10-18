"""Metrics.

Point:        rmse, mae, bias (mean prediction - observed), r2 (vs. the evaluation-set mean),
              rmse_site_equal (each place weighted equally, so near-daily sites do not dominate)
Distribution: pinball loss averaged over the predicted quantiles, 80% interval coverage and width
Low tail:     for "low residual" events (y < threshold): AUC of ranking samples by predicted value,
              recall and alarm rate of the rule "q10 < threshold"
Site level:   for places with enough samples, Spearman correlation and RMSE between observed and
              predicted mean residual ("can we rank and map the sites?")
Uncertainty:  cluster bootstrap over place groups (samples from one place are correlated), giving
              95% intervals for RMSE and for the RMSE difference against a reference model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pinball(y, q, tau) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def auc_low(y, pred, threshold) -> tuple[float, int]:
    """AUC for detecting y < threshold using -pred as the score (lower prediction = more alarming)."""
    pos = np.asarray(y) < threshold
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan, n_pos
    ranks = stats.rankdata(-np.asarray(pred))
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), n_pos


def summarize(f: pd.DataFrame, quantiles, thresholds, min_site_samples: int) -> dict:
    """f: columns y, pred, place_group and q<pct> columns (e.g. q10, q50, q90)."""
    y, p = f.y.to_numpy(), f.pred.to_numpy()
    err = p - y
    out = {"n": len(f), "places": f.place_group.nunique(),
           "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
           "bias": float(np.mean(err)),
           "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)) if len(f) > 1 else np.nan}
    w = 1.0 / f.groupby("place_group").y.transform("size").to_numpy()
    out["rmse_site_equal"] = float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))
    qcols = [f"q{int(round(q * 100))}" for q in quantiles]
    if all(c in f for c in qcols) and f[qcols].notna().all().all():
        out["pinball"] = float(np.mean([pinball(y, f[c], q) for c, q in zip(qcols, quantiles)]))
        lo, hi = f[qcols[0]].to_numpy(), f[qcols[-1]].to_numpy()
        out["coverage80"] = float(np.mean((y >= lo) & (y <= hi)))
        out["width80"] = float(np.mean(hi - lo))
    for t in thresholds:
        tag = str(t).replace(".", "")
        out[f"auc_lt{tag}"], out[f"n_lt{tag}"] = auc_low(y, p, t)
        if "q10" in f:
            alarm = f.q10.to_numpy() < t
            out[f"recall_lt{tag}"] = float(alarm[y < t].mean()) if (y < t).any() else np.nan
            out[f"alarm_rate_lt{tag}"] = float(alarm.mean())
    site = f.groupby("place_group").agg(obs=("y", "mean"), pred=("pred", "mean"), n=("y", "size"))
    site = site[site.n >= min_site_samples]
    if len(site) >= 5:
        constant = site.obs.nunique() < 2 or site.pred.nunique() < 2      # e.g. the global-mean model
        out["site_spearman"] = np.nan if constant else float(stats.spearmanr(site.obs, site.pred).statistic)
        out["site_rmse"] = float(np.sqrt(np.mean((site.obs - site.pred) ** 2)))
        out["n_sites_ranked"] = len(site)
    return out


def bootstrap_rmse(preds: pd.DataFrame, reference: str, reps: int, seed: int) -> pd.DataFrame:
    """Cluster bootstrap over place groups. preds: model, place_group, y, pred (one evaluation set).
    Returns per model: rmse_lo, rmse_hi, d_rmse (vs. reference), d_rmse_lo, d_rmse_hi, p_better."""
    sse = (preds.assign(se=(preds.pred - preds.y) ** 2)
           .groupby(["model", "place_group"]).agg(sse=("se", "sum"), n=("se", "size")).reset_index())
    places = np.sort(sse.place_group.unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(places), size=(reps, len(places)))
    counts = np.stack([np.bincount(d, minlength=len(places)) for d in draws])   # reps x places
    res = {}
    for model, g in sse.groupby("model"):
        g = g.set_index("place_group").reindex(places).fillna(0.0)
        tot_sse, tot_n = counts @ g.sse.to_numpy(), counts @ g.n.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            res[model] = np.sqrt(tot_sse / tot_n)
    out = []
    ref = res.get(reference)
    for model, r in res.items():
        row = {"model": model, "rmse_lo": np.nanpercentile(r, 2.5), "rmse_hi": np.nanpercentile(r, 97.5)}
        if ref is not None:            # every model predicts every row of an evaluation set
            d = r - ref
            row.update(d_rmse_lo=np.nanpercentile(d, 2.5), d_rmse_hi=np.nanpercentile(d, 97.5),
                       p_better=float(np.nanmean(d < 0)))
        out.append(row)
    return pd.DataFrame(out)
