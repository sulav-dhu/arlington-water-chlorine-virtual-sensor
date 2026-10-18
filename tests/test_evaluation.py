import numpy as np
import pandas as pd
import pytest

from chlorine.config import EvalConfig
from chlorine.evaluation.features import NeighborPanel, build_features
from chlorine.evaluation.harness import development_frame, run_known_site, run_unseen_site
from chlorine.evaluation.metrics import auc_low, bootstrap_rmse, pinball, summarize
from chlorine.evaluation.models import Offsets, reference_models
from chlorine.evaluation.splits import place_folds

CFG = EvalConfig()


def synthetic(n_places=8, weeks=30, seed=0):
    """Small panel: places on a line 1 km apart, alternating regions/planes, 1-3 samples a week."""
    rng = np.random.default_rng(seed)
    places = pd.DataFrame({"lat": 32.7 + np.arange(n_places) * 0.009, "lon": -97.1,
                           "region": ["E", "W"] * (n_places // 2), "pressure_plane": ["UPPER", "LOWER"] * (n_places // 2),
                           "n_rows": 1}, index=[f"P{i}" for i in range(n_places)])
    rows = []
    start = pd.Timestamp("2022-11-07")
    for w in range(weeks):
        for p in places.index:
            for _ in range(rng.integers(1, 4)):
                t = start + pd.Timedelta(weeks=w, days=int(rng.integers(0, 7)), hours=int(rng.integers(8, 15)))
                rows.append({"place_group": p, "location_id": p, "sampled_at": t, "y": 3.5 + rng.normal(0, 0.4)})
    df = pd.DataFrame(rows).sort_values("sampled_at").reset_index(drop=True)
    df["day"] = df.sampled_at.dt.normalize()
    df["week"] = df.sampled_at.dt.to_period("W-SUN").dt.start_time
    df["month"] = df.sampled_at.dt.month
    df["region"] = df.place_group.map(places.region)
    df["pressure_plane"] = df.place_group.map(places.pressure_plane)
    df["row_id"] = np.arange(len(df))
    df["high_frequency_site"] = False
    df["jk_offline"] = df["pb_offline"] = False
    return df, places


def test_week_features_match_brute_force():
    df, places = synthetic()
    f = build_features(df, np.ones(len(df), bool), NeighborPanel(places, 3.0))
    pm = df.groupby(["week", "place_group"]).y.mean()
    for i in range(0, len(df), 7):
        others = pm.loc[df.week[i]].drop(df.place_group[i])
        assert np.isclose(f.wk_city_mean[i], others.mean())
        same_region = others[places.region.reindex(others.index).eq(df.region[i]).to_numpy()]
        assert np.isclose(f.wk_region_mean[i], same_region.mean())


def test_features_never_use_own_or_out_of_pool_values():
    df, places = synthetic()
    panel = NeighborPanel(places, 3.0)
    pool = ~df.place_group.isin(["P2", "P5"]).to_numpy()
    base = build_features(df, pool, panel)
    feature_cols = [c for c in base.columns if c.startswith(("wk_", "t7_", "own_"))]
    # 1) values of places outside the pool never matter
    pert = df.copy()
    pert.loc[~pool, "y"] = 1000.0
    pd.testing.assert_frame_equal(base[feature_cols], build_features(pert, pool, panel)[feature_cols])
    # 2) a sample's own value never enters its own features
    full = build_features(df, np.ones(len(df), bool), panel)
    for i in [3, 50, 120]:
        pert = df.copy()
        pert.loc[i, "y"] = 1000.0
        np.testing.assert_allclose(full.loc[i, feature_cols].astype(float),
                                   build_features(pert, np.ones(len(df), bool), panel).loc[i, feature_cols].astype(float))


def test_trailing_window_is_strictly_before():
    df, places = synthetic(n_places=2, weeks=1)
    df = df.iloc[:0]
    t = pd.Timestamp("2023-03-10 10:00")
    rows = [("P0", t), ("P1", t)] + [("P0", t + pd.Timedelta(days=d)) for d in (1, 7, 8)]
    df = pd.DataFrame([{"place_group": p, "location_id": p, "sampled_at": s, "y": 3.0 if p == "P1" else 1.0}
                       for p, s in rows])
    df["day"], df["week"] = df.sampled_at.dt.normalize(), df.sampled_at.dt.to_period("W-SUN").dt.start_time
    df["region"], df["pressure_plane"] = "E", "UPPER"
    f = build_features(df, np.ones(len(df), bool), NeighborPanel(places.iloc[:2], 3.0))
    p0 = f[f.place_group == "P0"].set_index("sampled_at").t7_city_mean
    assert np.isnan(p0[t])                                       # same-day neighbor sample excluded
    assert p0[t + pd.Timedelta(days=1)] == 3.0 and p0[t + pd.Timedelta(days=7)] == 3.0
    assert np.isnan(p0[t + pd.Timedelta(days=8)])                # outside the 7-day window


def test_place_folds_deterministic_and_stratified():
    _, places = synthetic(n_places=40)
    places["n_rows"] = np.arange(40)
    a, b = place_folds(places, 5, 1), place_folds(places, 5, 1)
    assert a.equals(b) and a.notna().all()
    per_plane = pd.crosstab(places.pressure_plane, a)
    assert (per_plane.max(axis=1) - per_plane.min(axis=1)).max() <= 1


def test_offsets_shrink_toward_parent():
    df = pd.DataFrame({"g": ["a"] * 2 + ["b"] * 40, "s": ["a1", "a2"] + ["b1"] * 40})
    resid = np.r_[1.0, 1.0, np.zeros(40)]
    off = Offsets(["g", "s"], k=20).fit(df, resid)
    glob = 2 / 42
    assert np.isclose(off.tables[0]["a"], (2 + 20 * glob) / 22)
    assert np.isclose(off.transform(pd.DataFrame({"g": ["a"], "s": ["new"]}))[0], off.tables[0]["a"])


def test_models_always_predict():
    df, places = synthetic()
    df["lat"] = df["lon"] = 0.0
    f = build_features(df, np.ones(len(df), bool), NeighborPanel(places, 3.0))
    train, ev = f[f.sampled_at < "2023-03-01"], f[f.sampled_at >= "2023-03-01"].copy()
    ev[[c for c in ev.columns if c.startswith(("wk_", "t7_")) and c.endswith("_mean")]] = np.nan
    for m in reference_models(CFG):
        p = m.fit(train).predict(ev)
        assert p.notna().all().all(), m.code
        assert (p.q10 <= p.q90).all()


def test_metrics():
    assert pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert pinball([0.0], [1.0], 0.9) == pytest.approx(0.1)
    auc, n = auc_low(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 1.0, 3.5, 4.5]), 2.5)
    assert n == 2 and auc == 1.0
    f = pd.DataFrame({"y": [3.0, 3.2, 2.8, 3.1] * 5, "pred": [3.0] * 20, "place_group": list("abcde") * 4,
                      "q10": 2.5, "q50": 3.0, "q90": 3.5})
    s = summarize(f, (0.1, 0.5, 0.9), (2.5,), 1)
    assert s["coverage80"] == 1.0 and s["n"] == 20
    boot = bootstrap_rmse(f.assign(model="B1"), "B1", 50, 0)
    assert boot.rmse_lo.iloc[0] <= np.sqrt(np.mean((f.pred - f.y) ** 2)) <= boot.rmse_hi.iloc[0]


# -- end-to-end leakage checks on the real v1 data ---------------------------------------------------
real = pytest.mark.skipif(not (CFG.data_dir / "chlorine_v1.parquet").exists(), reason="v1 data not built")


@pytest.fixture(scope="module")
def real_data():
    from chlorine.evaluation.data import load_frame
    df, places = load_frame(CFG)
    return df, places, NeighborPanel(places, CFG.idw_length_km), place_folds(places, CFG.n_folds, CFG.seed)


@real
def test_unseen_site_predictions_ignore_held_out_values(real_data):
    df, places, panel, folds = real_data
    models = [m for m in reference_models(CFG) if m.code in ("B2", "B4", "B4p")]
    base = run_unseen_site(df, panel, models, "val", CFG, folds)
    for f in (0, 3):
        # Corrupt every value (all periods) of the places held out in fold f. Their predictions in that
        # fold must not move: they are unmonitored, so no feature or fitted parameter may use them.
        pert = df.copy()
        pert.loc[df.place_group.map(folds).eq(f), "y"] = 1000.0
        again = run_unseen_site(pert, panel, models, "val", CFG, folds)
        a = base[base.fold == f].set_index(["model", "row_id"]).pred.sort_index()
        b = again[again.fold == f].set_index(["model", "row_id"]).pred.sort_index()
        assert len(a) > 0 and a.index.equals(b.index)
        np.testing.assert_allclose(a, b)


@real
def test_known_site_prediction_ignores_own_value_and_test_period(real_data):
    df, places, panel, _ = real_data
    models = [m for m in reference_models(CFG) if m.code in ("B0p", "B3", "B5", "B5p", "B5r")]
    base = run_known_site(df, panel, models, "val", CFG).set_index(["model", "row_id"]).pred
    dev = development_frame(df, "val", CFG)
    target = dev[dev.sampled_at >= "2023-06-01"].row_id.iloc[0]
    pert = df.copy()
    pert.loc[pert.row_id == target, "y"] = 1000.0
    pert.loc[pert.sampled_at >= "2024-01-01", "y"] = -1000.0                 # test period must be invisible
    again = run_known_site(pert, panel, models, "val", CFG).set_index(["model", "row_id"]).pred
    np.testing.assert_allclose(base.xs(target, level="row_id"), again.xs(target, level="row_id"))
    other = base.index.get_level_values("row_id") != target
    assert np.allclose(base[other].groupby(level="model").size(), again[other].groupby(level="model").size())
