"""Step 3 experiments: tuning, candidates, ablations, selection, and the single test run.

    uv run python -m chlorine.modeling.run                                  # inner tuning + validation 2023
    uv run python -m chlorine.modeling.run --split test --confirm-test      # after validation, logged

Protocol (fixed before any test run)
  * Tuning: the only tuned setting is the recency half-life of the regression-kriging models, chosen
    by known-site RMSE on an inner validation year (2022, models trained on 2015-2021). For the test
    split it is chosen on the official validation year (2023).
  * Selection: in each protocol, the candidate (not ablation) with the lowest validation RMSE.
  * Candidates were added in two rounds: round 1 (B6, B6r, G6, B7, B7r, G7) and, after reading round-1
    validation results, round 2: equal-weight averages E6 = B6+G6 and E7 = B7+G7 (B6 and G6 were within
    noise of each other). To guard against tuning to one year, every candidate is also reported on a
    second validation year (2022, trained on 2015-2021); the selection rule itself is unchanged.
  * Test: references + all candidates are evaluated once; the headline is the selected model.
Outputs: data/eval/step3/<split>/ (report, results, predictions, tuning, parameters, selection).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import EvalConfig
from ..evaluation.data import load_frame
from ..evaluation.features import NeighborPanel
from ..evaluation.harness import run_known_site, run_outage, run_unseen_site
from ..evaluation.metrics import bootstrap_rmse
from ..evaluation.models import reference_models
from ..evaluation.report import noise_floor, summarize_predictions
from ..evaluation.run import _git_commit
from ..evaluation.splits import outage_windows, place_folds
from .ensemble import Ensemble
from .gbm import GBM
from .rk import RegressionKriging
from .report import ABLATION_PARENT, write_report

HALF_LIVES = (365.0, 730.0, 1461.0, float("inf"))
REFERENCES = ("B0", "B1", "B2", "B4", "B4p", "B5", "B5p", "B5r")
COMPARATOR = {"known_site": "B5", "unseen_site": "B4", "outage": "B5p"}   # best reference on validation
CANDIDATES = ("B6", "B6r", "G6", "B7", "B7r", "G7", "E6", "E7")
PUBLIC_ONLY = ("B6", "B7", "G7", "E7")    # no utility region code (site history is the utility's own data)


def candidates(cfg, h) -> list:
    return [
        RegressionKriging(cfg, "B6", "rk_site", site_history=True, half_life_days=h),
        RegressionKriging(cfg, "B6r", "rk_site+region", site_history=True, region=True, half_life_days=h),
        GBM(cfg, "G6", "gbm_site+region", site_history=True, region=True),
        RegressionKriging(cfg, "B7", "rk_public", site_history=False, half_life_days=h),
        RegressionKriging(cfg, "B7r", "rk_public+region", site_history=False, region=True, half_life_days=h),
        GBM(cfg, "G7", "gbm_public", site_history=False, region=False),
        Ensemble(cfg, "E6", "avg(B6,G6)", [RegressionKriging(cfg, "B6", "rk_site", site_history=True, half_life_days=h),
                                           GBM(cfg, "G6", "gbm_site+region", site_history=True, region=True)]),
        Ensemble(cfg, "E7", "avg(B7,G7)", [RegressionKriging(cfg, "B7", "rk_public", site_history=False, half_life_days=h),
                                           GBM(cfg, "G7", "gbm_public", site_history=False, region=False)]),
    ]


def ablations(cfg, h) -> list:
    return [
        RegressionKriging(cfg, "A1", "B7 without static kriging", site_history=False, static_kriging=False, half_life_days=h),
        RegressionKriging(cfg, "A2", "B7 without anomaly kriging", site_history=False, anomaly_kriging=False, half_life_days=h),
        RegressionKriging(cfg, "A3", "B7, distance-only covariance", site_history=False, cross_plane=1.0, half_life_days=h),
        RegressionKriging(cfg, "A4", "B7 without plant-regime terms", site_history=False, regime=False, half_life_days=h),
        RegressionKriging(cfg, "A5", "B6 without own recent anomalies", site_history=True, own_history=False, half_life_days=h),
        RegressionKriging(cfg, "A6", "B6 without recency weighting", site_history=True, half_life_days=float("inf")),
    ]


def refs(cfg) -> list:
    return [m for m in reference_models(cfg) if m.code in REFERENCES]


def tune(df, panel, cfg, split) -> tuple[float, pd.DataFrame]:
    tcfg = dataclasses.replace(cfg, train_end="2021-12-31", val_end="2022-12-31") if split == "val" else cfg
    rows = []
    for h in HALF_LIVES:
        ms = [RegressionKriging(tcfg, "B6", "rk_site", site_history=True, half_life_days=h),
              RegressionKriging(tcfg, "B7", "rk_public", site_history=False, half_life_days=h)]
        p = run_known_site(df, panel, ms, "val", tcfg)
        for code, g in p.groupby("model"):
            rows.append({"half_life_days": h, "model": code, "rmse": float(np.sqrt(np.mean((g.pred - g.y) ** 2))),
                         "tuning_year": tcfg.train_end[:4] if split == "val" else "2023"})
    t = pd.DataFrame(rows)
    best = t[t.model == "B6"].sort_values("rmse").iloc[0].half_life_days
    return float(best), t


def comparisons(preds: pd.DataFrame, cfg) -> pd.DataFrame:
    """Paired cluster-bootstrap RMSE difference of every model against the protocol's comparator."""
    out = []
    for (protocol, split), p in preds.groupby(["protocol", "split"]):
        ref = COMPARATOR[protocol]
        b = bootstrap_rmse(p, ref, cfg.bootstrap_reps, cfg.seed)
        b["protocol"], b["split"], b["comparator"] = protocol, split, ref
        out.append(b)
    return pd.concat(out, ignore_index=True)


def second_year(df, panel, cfg, folds, h) -> pd.DataFrame:
    """References and candidates on 2022 (trained on 2015-2021): is the 2023 ranking stable?"""
    icfg = dataclasses.replace(cfg, train_end="2021-12-31", val_end="2022-12-31")
    ms = lambda: [m for m in refs(icfg) if m.code in ("B1", "B4", "B5", "B5p")] + candidates(icfg, h)  # noqa: E731
    p = pd.concat([run_known_site(df, panel, ms(), "val", icfg), run_unseen_site(df, panel, ms(), "val", icfg, folds)])
    rows = []
    for (protocol, code), g in p.groupby(["protocol", "model"]):
        rows.append({"protocol": protocol, "model": code, "rmse_2022": float(np.sqrt(np.mean((g.pred - g.y) ** 2))),
                     "n_2022": len(g)})
    return pd.DataFrame(rows)


def ablation_comparisons(preds: pd.DataFrame, cfg) -> pd.DataFrame:
    """Paired bootstrap of each ablation against its full parent model, per protocol."""
    out = []
    for (protocol, split), p in preds.groupby(["protocol", "split"]):
        for abl, parent in ABLATION_PARENT.items():
            q = p[p.model.isin([abl, parent])]
            if q.model.nunique() < 2:
                continue
            b = bootstrap_rmse(q, parent, cfg.bootstrap_reps, cfg.seed).set_index("model").loc[abl]
            out.append({"protocol": protocol, "split": split, "model": abl, "parent": parent,
                        "d_rmse_lo": b.d_rmse_lo, "d_rmse_hi": b.d_rmse_hi, "p_better": b.p_better})
    return pd.DataFrame(out)


def rebuild_report(split: str):
    """Regenerate modeling_report.md from saved outputs; never fits or evaluates a model."""
    cfg = EvalConfig()
    out_dir = cfg.out_root / "step3" / split
    res, comp = pd.read_csv(out_dir / "results.csv"), pd.read_csv(out_dir / "comparisons.csv")
    tuning, params = pd.read_csv(out_dir / "tuning.csv"), json.loads((out_dir / "params.json").read_text())
    sel_all = json.loads((cfg.out_root / "step3" / "val" / "selection.json").read_text())
    selection = {k: v for k, v in sel_all.items() if isinstance(v, dict)}
    abl = None
    rob = pd.read_csv(out_dir / "second_year.csv") if (out_dir / "second_year.csv").exists() else None
    if split == "val":
        preds = pd.read_parquet(out_dir / "predictions.parquet")
        abl = ablation_comparisons(preds, cfg)
    df, _ = load_frame(cfg)
    write_report(out_dir / "modeling_report.md", cfg, split, res, comp, tuning, params, selection,
                 noise_floor(df, cfg), sel_all["half_life_days"], abl, rob)
    print(f"-> {out_dir / 'modeling_report.md'}")


def main():
    ap = argparse.ArgumentParser(description="Step 3 model experiments.")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from saved outputs (no models run)")
    args = ap.parse_args()
    if args.report_only:
        return rebuild_report(args.split)
    if args.split == "test" and not args.confirm_test:
        ap.error("--split test touches the held-out test period; add --confirm-test to proceed")
    cfg = EvalConfig()
    out_dir = cfg.out_root / "step3" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "test" and not (cfg.out_root / "step3" / "val" / "selection.json").exists():
        ap.error("run the validation experiments first: the selection must exist before the test run")

    df, places = load_frame(cfg)
    events = pd.read_csv(cfg.data_dir / "events.csv")
    panel = NeighborPanel(places, cfg.idw_length_km)
    folds = place_folds(places, cfg.n_folds, cfg.seed)
    meta = {"split": args.split, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "git_commit": _git_commit()}

    h, tuning = tune(df, panel, cfg, args.split)
    meta["half_life_days"] = h
    print(f"half-life selected: {h} days")
    models = lambda: refs(cfg) + candidates(cfg, h) + (ablations(cfg, h) if args.split == "val" else [])  # noqa: E731
    known_models = models()
    preds = [run_known_site(df, panel, known_models, args.split, cfg),
             run_unseen_site(df, panel, models(), args.split, cfg, folds)]
    if args.split == "val":
        preds.append(run_outage(df, panel, models(), cfg, outage_windows(events, "JK", before=cfg.val_end)))
    preds = pd.concat(preds, ignore_index=True)
    res = summarize_predictions(preds, cfg)
    comp = comparisons(preds, cfg)
    abl = ablation_comparisons(preds, cfg) if args.split == "val" else None
    params = {m.code: m.describe() for m in known_models if hasattr(m, "describe")}
    robustness = second_year(df, panel, cfg, folds, h) if args.split == "val" else None

    selection = {}
    if args.split == "val":
        allr = res[res.subset.eq("all") & res.model.isin(CANDIDATES)]
        for protocol, g in allr.groupby("protocol"):
            best = g.sort_values("rmse").iloc[0]
            pub = g[g.model.isin(PUBLIC_ONLY)].sort_values("rmse").iloc[0]
            selection[protocol] = {"selected": best.model, "rmse": best.rmse,
                                   "best_public_only": pub.model, "public_only_rmse": pub.rmse}
        (out_dir / "selection.json").write_text(json.dumps({"half_life_days": h, **selection}, indent=2))
    else:
        selection = {k: v for k, v in json.loads((cfg.out_root / "step3" / "val" / "selection.json").read_text()).items()
                     if isinstance(v, dict)}
        with open(cfg.out_root / "test_access_log.jsonl", "a") as f:
            f.write(json.dumps({**meta, "step": 3, "models": sorted(preds.model.unique())}) + "\n")

    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    res.to_csv(out_dir / "results.csv", index=False)
    comp.to_csv(out_dir / "comparisons.csv", index=False)
    tuning.to_csv(out_dir / "tuning.csv", index=False)
    if robustness is not None:
        robustness.to_csv(out_dir / "second_year.csv", index=False)
    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=float))
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    write_report(out_dir / "modeling_report.md", cfg, args.split, res, comp, tuning, params, selection,
                 noise_floor(df, cfg), h, abl, robustness)
    for protocol, sel in selection.items():
        r = res[(res.protocol == protocol) & res.subset.eq("all")].set_index("model")
        if r.empty:
            continue
        ref = COMPARATOR[protocol]
        print(f"{protocol:12s} selected {sel['selected']} RMSE {r.loc[sel['selected'], 'rmse']:.3f} "
              f"vs {ref} {r.loc[ref, 'rmse']:.3f}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
