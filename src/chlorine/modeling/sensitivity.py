"""Sensitivity: E7 without any use of the utility's sampling-region code.

G7, and so E7, uses same-week neighbor contrasts within the target's sampling region (`wk_region_*`). Those
need the utility's region code (site-code prefix) of the target place, so E7 uses no site history but is not
strictly public-data-only; at a genuinely new address the region would have to be inferred from location.
E7s replaces G7 by G7s, identical except that the two region-contrast features are made constant (trees
cannot split on them), to measure what the region code adds. Development years only; the frozen E7 is
unchanged.

    uv run python -m chlorine.modeling.sensitivity     # -> data/eval/backtest/dev_public_strict.parquet
"""
from __future__ import annotations

import pandas as pd

from ..config import EvalConfig
from ..evaluation.data import load_frame
from ..evaluation.features import NeighborPanel
from ..evaluation.harness import run_known_site, run_unseen_site
from ..evaluation.splits import place_folds
from .backtest import HALF_LIFE_DAYS, year_cfg
from .ensemble import Ensemble
from .gbm import GBM
from .rk import RegressionKriging


class PublicGBM(GBM):
    """GBM whose same-region neighbor contrast is neutralized: d_region = 0 and amt_region = 0 for every row."""

    def _X(self, df):
        return super()._X(df.assign(wk_region_mean=self._city(df), wk_region_n=0.0))


def e7s(cfg):
    return Ensemble(cfg, "E7s", "avg(B7,G7s)", [
        RegressionKriging(cfg, "B7", "rk_public", site_history=False, half_life_days=HALF_LIFE_DAYS),
        PublicGBM(cfg, "G7s", "gbm_public_strict", site_history=False, region=False)])


def strict_models(cfg):
    return [e7s(cfg), PublicGBM(cfg, "G7s", "gbm_public_strict", site_history=False, region=False)]


def main(years=(2017, 2018, 2019, 2020, 2021, 2022, 2023)):
    cfg = EvalConfig()
    df, places = load_frame(cfg)
    panel = NeighborPanel(places, cfg.idw_length_km)
    folds = place_folds(places, cfg.n_folds, cfg.seed)
    out = []
    for y in years:
        ycfg = year_cfg(cfg, y)
        known = run_known_site(df, panel, strict_models(ycfg), "val", ycfg)
        unseen = run_unseen_site(df, panel, strict_models(ycfg), "val", ycfg, folds)
        out.append(pd.concat([known, unseen], ignore_index=True).assign(year=y))
        print(f"sensitivity {y}: done", flush=True)
    path = cfg.out_root / "backtest" / "dev_public_strict.parquet"
    pd.concat(out, ignore_index=True).to_parquet(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
