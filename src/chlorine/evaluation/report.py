"""Evaluation summaries, noise floor and the Markdown report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EvalConfig
from .metrics import bootstrap_rmse, summarize

REFERENCE = "B1"
SHOWN = ["B1", "B2", "B2p", "B3", "B4", "B4p", "B5", "B5p", "B5r"]


def subsets(p: pd.DataFrame) -> dict[str, pd.Series]:
    s = {"all": pd.Series(True, index=p.index)}
    for plane in ("LOWER", "UPPER", "WEST"):
        s[f"plane={plane}"] = p.pressure_plane.eq(plane)
    s["routine_frequency_sites"] = ~p.high_frequency_site.astype(bool)
    s["high_frequency_sites"] = p.high_frequency_site.astype(bool)
    if "new_site" in p and p.new_site.any():
        s["new_sites"] = p.new_site.astype(bool)
        s["sites_with_history"] = ~p.new_site.astype(bool)
    if p.jk_offline.any():
        s["jk_offline"] = p.jk_offline.astype(bool)
    if "event_id" in p:
        for e in p.event_id.dropna().unique():
            s[f"event={e}"] = p.event_id.eq(e)
    return s


def summarize_predictions(preds: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    rows = []
    for (protocol, split), p in preds.groupby(["protocol", "split"], sort=False):
        boot = bootstrap_rmse(p, REFERENCE, cfg.bootstrap_reps, cfg.seed).set_index("model")
        for sub, mask in subsets(p).items():
            for (code, name), g in p[mask].groupby(["model", "model_name"], sort=False):
                if g.empty:
                    continue
                r = {"protocol": protocol, "split": split, "subset": sub, "model": code, "model_name": name,
                     **summarize(g, cfg.quantiles, cfg.low_thresholds, cfg.min_site_eval_samples)}
                if sub == "all" and code in boot.index:
                    r.update(boot.loc[code].to_dict())
                rows.append(r)
    return pd.DataFrame(rows)


def noise_floor(df: pd.DataFrame, cfg: EvalConfig) -> dict:
    """Two floors no model can beat. Measurement: separate same-moment readings by two programs
    (routine locations). Within-week: spread of samples around their own place-week mean at the
    near-daily sites, i.e. the error of a model that knew every place's weekly mean exactly."""
    out = {}
    meas = pd.read_parquet(cfg.data_dir / "measurements.parquet")
    rep = meas[(meas.analyte == "Chlorine residual") & (meas.analysis == "Chlres-field")
               & meas.location_type.eq("routine") & meas.dup_group.notna() & meas.result_status.eq("detected")]
    g = rep.groupby("dup_group").value.agg(["min", "max", "size"])
    g = g[g["size"] == 2]
    diff = (g["max"] - g["min"])[lambda d: d > 0]
    out["replicate_pairs"] = int(len(diff))
    out["measurement_sd"] = float(np.sqrt(np.mean(diff ** 2) / 2)) if len(diff) else np.nan
    hf = df[df.high_frequency_site.astype(bool)]
    resid = hf.y - hf.groupby(["place_group", "week"]).y.transform("mean")
    n = hf.groupby(["place_group", "week"]).y.transform("size")
    dof = (n - 1).clip(lower=0)
    out["within_week_sd"] = float(np.sqrt((resid[n > 1] ** 2).sum() / dof[n > 1].groupby(
        [hf.place_group[n > 1], hf.week[n > 1]]).first().sum()))
    return out


def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ladder_table(res: pd.DataFrame, protocol: str, split: str) -> str:
    r = res[(res.protocol == protocol) & (res.split == split) & (res.subset == "all")]
    lines = ["| Code | Model | RMSE [95% CI] | ΔRMSE vs B1 [95% CI] | MAE | Bias | R² | RMSE site-equal | "
             "Pinball | Cover80 | Width80 | AUC <2.5 | Site ρ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in r.itertuples():
        d = "" if x.model == REFERENCE else f"{x.rmse - r[r.model == REFERENCE].rmse.iloc[0]:+.3f} " \
            f"[{x.d_rmse_lo:+.3f}, {x.d_rmse_hi:+.3f}]"
        lines.append(f"| {x.model} | {x.model_name} | {_fmt(x.rmse)} [{_fmt(x.rmse_lo)}, {_fmt(x.rmse_hi)}] | {d} | "
                     f"{_fmt(x.mae)} | {x.bias:+.3f} | {_fmt(x.r2)} | {_fmt(x.rmse_site_equal)} | {_fmt(x.pinball)} | "
                     f"{_fmt(x.coverage80, 2)} | {_fmt(x.width80, 2)} | {_fmt(x.auc_lt25, 2)} | "
                     f"{_fmt(getattr(x, 'site_spearman', np.nan), 2)} |")
    n = r.iloc[0]
    return "\n".join(lines) + f"\n\n{int(n.n):,} samples at {int(n.places)} places; " \
                              f"{int(n.n_lt25)} samples below 2.5 mg/L.\n"


def breakdown_table(res: pd.DataFrame, protocol: str, split: str, prefixes: tuple[str, ...]) -> str:
    r = res[(res.protocol == protocol) & (res.split == split) & res.subset.str.startswith(prefixes)
            & res.model.isin(SHOWN)]
    if r.empty:
        return "_none_\n"
    t = r.pivot_table(index="subset", columns="model", values="rmse", sort=False)[[m for m in SHOWN if m in set(r.model)]]
    n = r.groupby("subset", sort=False).n.first()
    lines = ["| Subset | n | " + " | ".join(t.columns) + " |", "|---|---|" + "---|" * len(t.columns)]
    for sub, row in t.iterrows():
        best = row.min()
        cells = [f"**{v:.3f}**" if np.isclose(v, best) else f"{v:.3f}" for v in row]
        lines.append(f"| {sub} | {int(n[sub]):,} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(path, cfg: EvalConfig, res: pd.DataFrame, floor: dict, meta: dict, curve: pd.DataFrame | None):
    split = meta["split"]
    L = [f"# Evaluation report — reference models, {split} split\n\n",
         f"Generated {meta['generated_at']}. Data `{cfg.data_version}`; periods: train ≤ {cfg.train_end}, "
         f"validation ≤ {cfg.val_end}, test after. "
         + ("Models are trained on the train period and evaluated on validation (2023)."
            if split == "val" else "**Test split**: models refit on train + validation and evaluated on the test "
                                   "period. This run is logged in `data/eval/test_access_log.jsonl`.") + "\n\n"]
    w = L.append
    w("## Headline\n\n")
    for protocol, sp, label in [("known_site", split, "Known sites"), ("unseen_site", split, "Unseen sites"),
                                ("outage", "pre_test", "Kubala outages")]:
        r = res[(res.protocol == protocol) & (res.split == sp) & (res.subset == "all")].set_index("model")
        if r.empty:
            continue
        best = r.rmse.idxmin()
        w(f"- **{label}:** best reference model {best} ({r.loc[best, 'model_name']}) RMSE "
          f"{r.loc[best, 'rmse']:.3f} mg/L (R² {r.loc[best, 'r2']:.2f}) vs. {r.loc['B1', 'rmse']:.3f} for the "
          f"city-wide weekly mean (B1) and {r.loc['B0', 'rmse']:.3f} for the overall mean (B0); "
          f"AUC for samples < 2.5 mg/L {r.loc[best, 'auc_lt25']:.2f}.\n")
    hf = res[(res.protocol == "known_site") & (res.split == split) & (res.subset == "high_frequency_sites")]
    if not hf.empty:   # like-for-like: the within-week floor can only be measured at the near-daily sites
        kb = hf.rmse.min()
        gap = np.sqrt(max(kb ** 2 - floor["within_week_sd"] ** 2, 0))
        w(f"- **Noise floor:** at the near-daily sites, where it can be measured, samples scatter around their own "
          f"weekly mean with SD {floor['within_week_sd']:.2f} mg/L; the best known-site RMSE there is {kb:.3f}, so "
          f"only ~{gap:.2f} mg/L (RMS) is left to explain by any model working at weekly resolution.\n\n")
    w("## How to read this\n\n")
    w("- **Task:** estimate a sample's total chlorine (mg/L) from what other monitored places measured the same "
      "week, plus location attributes; models marked *site history* also use the place's own past samples.\n")
    w("- **Protocols:** `known_site` = places of the monitored network; `unseen_site` = 5-fold cross-validation "
      "over place groups, held-out places act as unmonitored locations (their values never enter any feature); "
      "`outage` = leave-one-Kubala-outage-out over pre-test outages.\n")
    w("- **Uncertainty:** 95% intervals from a cluster bootstrap over places. ΔRMSE < 0 means better than B1 "
      "(city-wide weekly mean).\n")
    w(f"- **Noise floor:** separate same-moment readings differ with SD ≈ {floor['measurement_sd']:.2f} mg/L "
      f"({floor['replicate_pairs']} pairs), and samples at the near-daily sites scatter around their own weekly "
      f"mean with SD ≈ {floor['within_week_sd']:.2f} mg/L. A weekly-resolution model cannot get RMSE much below "
      f"the latter.\n\n")

    for protocol, title in [("known_site", "Known sites"), ("unseen_site", "Unseen sites"),
                            ("outage", "Kubala outages (leave-one-outage-out, pre-test)")]:
        sp = "pre_test" if protocol == "outage" else split
        if not ((res.protocol == protocol) & (res.split == sp)).any():
            continue
        w(f"## {title}\n\n")
        w(ladder_table(res, protocol, sp))
        w("\nRMSE by subset (bold = best shown model):\n\n")
        w(breakdown_table(res, protocol, sp, ("plane=", "routine", "high", "new", "sites_with", "jk_", "event=")))
        w("\n")

    if curve is not None and not curve.empty:
        w("## Value of more monitored places (unseen sites, learning curve)\n\n")
        w("Each fold's held-out places are predicted while only a random fraction of the remaining places is "
          "monitored (5 random draws per fraction). Error falls as the network grows; the slope is the value of "
          "adding sampling sites.\n\n")
        t = curve.groupby(["model", "fraction"]).agg(places=("monitored_places", "mean"), rmse=("rmse", "mean"),
                                                     rmse_sd=("rmse", "std")).reset_index()
        w("| Model | Fraction monitored | Avg monitored places | RMSE (mean ± sd over draws) |\n|---|---|---|---|\n")
        for x in t.itertuples():
            w(f"| {x.model} | {x.fraction:.2f} | {x.places:.0f} | {x.rmse:.3f} ± {x.rmse_sd:.3f} |\n")
        w("\n")

    w("## Notes\n\n")
    for note in [
        "Reference-model settings are fixed in `EvalConfig` and were not tuned on validation or test.",
        "Prediction intervals are calibrated on training residuals; coverage far from 0.80 signals a change in "
        "variability between periods (2023 was calmer than 2015-2022, outages are more volatile).",
        "Site ρ for models that use the place's own recent samples (B0p) is nearly trivial in the known-site "
        "protocol: the previous sample is from the same evaluation period.",
        "In `unseen_site`, held-out places keep their utility region code; for a truly new address it would have "
        "to be inferred from location.",
        "The outage protocol trains on data before and after each held-out outage; it measures the spatial "
        "response to a plant outage, not forecasting ability.",
    ]:
        w(f"- {note}\n")
    path.write_text("".join(L))
