"""Rolling-origin backtest: out-of-sample predictions year by year.

For each year Y, models train on everything before Y and predict Y, under the known-site protocol
(all places) and the unseen-site protocol (5-fold over places). Every row after Y is dropped first
(the harness' development frame), so a backtest year never sees later data.

Uses:
  * calibrating prediction intervals on genuinely out-of-sample errors (calibration.py);
  * checking that the selected models beat the baselines year after year, not only in 2023.
Development uses years up to 2023 only. Years 2024-2026 (the test period) are used only to build
the calibration pool of the frozen prospective model (prospective.py), never to evaluate.

    uv run python -m chlorine.modeling.backtest            # 2017-2023 -> data/eval/backtest/dev.parquet
    uv run python -m chlorine.modeling.backtest --members  # B6/G6/B7/G7 alone -> dev_members.parquet
"""
from __future__ import annotations

import argparse
import dataclasses

import pandas as pd

from ..config import EvalConfig
from ..evaluation.data import load_frame
from ..evaluation.features import NeighborPanel
from ..evaluation.harness import run_known_site, run_unseen_site
from ..evaluation.models import reference_models
from ..evaluation.splits import place_folds
from .ensemble import Ensemble
from .gbm import GBM
from .rk import RegressionKriging

HALF_LIFE_DAYS = 365.0            # selected on the inner validation year (docs/modeling.md)
REFERENCE_CODES = ("B1", "B4", "B4p", "B5", "B5p")


def e6(cfg):
    return Ensemble(cfg, "E6", "avg(B6,G6)", [
        RegressionKriging(cfg, "B6", "rk_site", site_history=True, half_life_days=HALF_LIFE_DAYS),
        GBM(cfg, "G6", "gbm_site+region", site_history=True, region=True)])


def e7(cfg):
    return Ensemble(cfg, "E7", "avg(B7,G7)", [
        RegressionKriging(cfg, "B7", "rk_public", site_history=False, half_life_days=HALF_LIFE_DAYS),
        GBM(cfg, "G7", "gbm_public", site_history=False, region=False)])


def members(cfg):
    """The ensembles' members on their own: is averaging better than plain kriging (B6/B7) every year?"""
    return [RegressionKriging(cfg, "B6", "rk_site", site_history=True, half_life_days=HALF_LIFE_DAYS),
            GBM(cfg, "G6", "gbm_site+region", site_history=True, region=True),
            RegressionKriging(cfg, "B7", "rk_public", site_history=False, half_life_days=HALF_LIFE_DAYS),
            GBM(cfg, "G7", "gbm_public", site_history=False, region=False)]


def year_cfg(cfg: EvalConfig, year: int) -> EvalConfig:
    return dataclasses.replace(cfg, train_end=f"{year - 1}-12-31", val_end=f"{year}-12-31")


def backtest(years, cfg: EvalConfig | None = None, with_references: bool = True, only_members: bool = False) -> pd.DataFrame:
    cfg = cfg or EvalConfig()
    df, places = load_frame(cfg)
    panel = NeighborPanel(places, cfg.idw_length_km)
    folds = place_folds(places, cfg.n_folds, cfg.seed)
    out = []
    for y in years:
        ycfg = year_cfg(cfg, y)
        refs = lambda: [m for m in reference_models(ycfg) if m.code in REFERENCE_CODES] if with_references else []  # noqa: E731, B023 (called in the same iteration)
        if only_members:      # run_unseen_site skips the members that need site history
            known = run_known_site(df, panel, members(ycfg), "val", ycfg)
            unseen = run_unseen_site(df, panel, members(ycfg), "val", ycfg, folds)
        else:
            known = run_known_site(df, panel, refs() + [e6(ycfg), e7(ycfg)], "val", ycfg)
            unseen = run_unseen_site(df, panel, refs() + [e7(ycfg)], "val", ycfg, folds)
        out.append(pd.concat([known, unseen], ignore_index=True).assign(year=y))
        print(f"backtest {y}: {len(known):,} known-site and {len(unseen):,} unseen-site predictions", flush=True)
    return pd.concat(out, ignore_index=True)


def horizon(train_end_year: int, last_year: int, cfg: EvalConfig | None = None) -> pd.DataFrame:
    """One model trained through train_end_year predicts every later year up to last_year without being
    retrained: does accuracy and interval coverage decay with years since training?"""
    cfg = dataclasses.replace(cfg or EvalConfig(), train_end=f"{train_end_year}-12-31", val_end=f"{last_year}-12-31")
    df, places = load_frame(cfg)
    panel = NeighborPanel(places, cfg.idw_length_km)
    folds = place_folds(places, cfg.n_folds, cfg.seed)
    known = run_known_site(df, panel, [e6(cfg), e7(cfg)], "val", cfg)
    unseen = run_unseen_site(df, panel, [e7(cfg)], "val", cfg, folds)
    out = pd.concat([known, unseen], ignore_index=True)
    out["years_since_training"] = out.sampled_at.dt.year - train_end_year
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", default="2017-2023")
    ap.add_argument("--horizon", help="TRAIN_END:LAST, e.g. 2019:2022 (development years only)")
    ap.add_argument("--members", action="store_true", help="backtest the ensemble members -> dev_members.parquet")
    args = ap.parse_args()
    if args.horizon:
        a, b = (int(x) for x in args.horizon.split(":"))
        if b > 2023:
            ap.error("development only: the last year must be <= 2023")
        cfg = EvalConfig()
        out = horizon(a, b, cfg)
        (cfg.out_root / "backtest").mkdir(parents=True, exist_ok=True)
        out.to_parquet(cfg.out_root / "backtest" / f"horizon_{a}_{b}.parquet", index=False)
        print(f"-> horizon_{a}_{b}.parquet")
        return
    lo, hi = (int(x) for x in args.years.split("-"))
    if hi > 2023:
        ap.error("development backtests stop at 2023; later years are only used by prospective.py")
    cfg = EvalConfig()
    out_dir = cfg.out_root / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "dev_members.parquet" if args.members else "dev.parquet"
    preds = backtest(range(lo, hi + 1), cfg, only_members=args.members)
    preds.to_parquet(out_dir / name, index=False)
    print(f"-> {out_dir / name}")


if __name__ == "__main__":
    main()
