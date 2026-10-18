"""Run the evaluation protocols on the reference models.

    uv run python -m chlorine.evaluation.run                      # validation split (default)
    uv run python -m chlorine.evaluation.run --learning-curve     # + value-of-more-sites curve
    uv run python -m chlorine.evaluation.run --split test --confirm-test

Outputs go to data/eval/<split>/: predictions.parquet (one row per model x sample x protocol),
results.csv (metrics per protocol x subset x model), evaluation_report.md and run.json.
Test-split runs are appended to data/eval/test_access_log.jsonl so test-set use is auditable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EvalConfig
from .data import load_frame
from .features import NeighborPanel
from .harness import run_known_site, run_outage, run_unseen_site
from .models import reference_models
from .report import noise_floor, summarize_predictions, write_report
from .splits import outage_windows, place_folds

CURVE_MODELS = ("B2", "B4", "B4p")


def learning_curve(df, places, panel, cfg, folds, fractions=(0.25, 0.5, 0.75, 1.0), draws=5) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    models = [m for m in reference_models(cfg) if m.code in CURVE_MODELS]
    # Monitored places that actually sampled during the training period, averaged over folds.
    train_places = set(df.place_group[df.sampled_at <= cfg.train_end])
    rows = []
    for frac in fractions:
        for d in range(draws if frac < 1 else 1):
            keep = set(places.index[rng.random(len(places)) < frac]) if frac < 1 else set(places.index)
            monitored = np.mean([len((keep & train_places) - set(folds.index[folds == f])) for f in range(cfg.n_folds)])
            p = run_unseen_site(df, panel, models, "val", cfg, folds, train_places=keep)
            for code, g in p.groupby("model"):
                rows.append({"model": code, "fraction": frac, "draw": d, "monitored_places": monitored,
                             "rmse": float(np.sqrt(np.mean((g.pred - g.y) ** 2)))})
    return pd.DataFrame(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Evaluate the reference models.")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-test", action="store_true", help="required for --split test")
    ap.add_argument("--learning-curve", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.confirm_test:
        ap.error("--split test touches the held-out test period; add --confirm-test to proceed")

    cfg = EvalConfig()
    out_dir = cfg.out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"split": args.split, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "data_version": cfg.data_version, "git_commit": _git_commit(),
            "data_manifest_sha256": json.loads((cfg.data_dir / "manifest.json").read_text())["raw_sha256"]}

    df, places = load_frame(cfg)
    events = pd.read_csv(cfg.data_dir / "events.csv")
    panel = NeighborPanel(places, cfg.idw_length_km)
    folds = place_folds(places, cfg.n_folds, cfg.seed)

    preds = [run_known_site(df, panel, reference_models(cfg), args.split, cfg),
             run_unseen_site(df, panel, reference_models(cfg), args.split, cfg, folds)]
    if args.split == "val":
        preds.append(run_outage(df, panel, reference_models(cfg), cfg, outage_windows(events, "JK", before=cfg.val_end)))
    preds = pd.concat(preds, ignore_index=True)
    res = summarize_predictions(preds, cfg)
    curve = learning_curve(df, places, panel, cfg, folds) if args.learning_curve and args.split == "val" else None

    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    res.to_csv(out_dir / "results.csv", index=False)
    if curve is not None:
        curve.to_csv(out_dir / "learning_curve.csv", index=False)
    floor = noise_floor(df, cfg)
    meta["noise_floor"] = floor
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    write_report(out_dir / "evaluation_report.md", cfg, res, floor, meta, curve)
    if args.split == "test":
        with open(cfg.out_root / "test_access_log.jsonl", "a") as f:
            f.write(json.dumps({**meta, "models": sorted(preds.model.unique())}) + "\n")
    best = res[(res.subset == "all")].sort_values("rmse").groupby(["protocol", "split"]).head(1)
    for r in best.itertuples():
        print(f"{r.protocol:12s} {r.split:9s} best {r.model} {r.model_name}: RMSE {r.rmse:.3f}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
