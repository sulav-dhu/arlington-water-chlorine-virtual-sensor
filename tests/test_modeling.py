import numpy as np
import pandas as pd
import pytest

from chlorine.config import EvalConfig
from chlorine.evaluation.features import NeighborPanel
from chlorine.evaluation.harness import run_known_site, run_unseen_site
from chlorine.evaluation.splits import place_folds
from chlorine.modeling.design import Standardizer, weighted_ridge
from chlorine.modeling.ensemble import Ensemble
from chlorine.modeling.gbm import GBM
from chlorine.modeling.rk import RegressionKriging
from chlorine.modeling.spatial import Covariance, Kriger, binned_pairs, fit_covariance

CFG = EvalConfig()


TRUTH = Covariance(0.1, 2.0, 0.5)


def gp_field(n_side=12, weeks=400, cov=TRUTH, seed=0):
    """Places on a 1-km grid, two pressure planes split down the middle; weekly GP draws."""
    rng = np.random.default_rng(seed)
    xy = np.array([(i, j) for i in range(n_side) for j in range(n_side)], float)
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    plane = (xy[:, 0] >= n_side / 2)
    same = plane[:, None] == plane[None, :]
    K = cov(D, same) + 1e-9 * np.eye(len(xy))
    L = np.linalg.cholesky(K)
    return D, same, (L @ rng.standard_normal((len(xy), weeks))).T     # weeks x places


def test_covariance_fit_recovers_parameters():
    truth = Covariance(0.1, 2.0, 0.5)
    D, same, Z = gp_field(cov=truth, weeks=1600)
    iu = np.triu_indices(D.shape[0], 1)
    prods = (Z[:, :, None] * Z[:, None, :]).mean(0)[iu]
    est = fit_covariance(binned_pairs(D[iu], same[iu], prods, np.ones(len(prods)), max_km=8))
    assert est.sigma2 == pytest.approx(0.1, rel=0.1)
    assert est.length_km == pytest.approx(2.0, rel=0.1)
    assert est.cross_plane == pytest.approx(0.5, abs=0.1)


def test_kriging_is_calibrated_and_excludes_own_place():
    truth = Covariance(0.1, 2.0, 0.5)
    D, same, Z = gp_field(cov=truth, weeks=300, seed=1)
    kr = Kriger(truth, D, same, np.arange(D.shape[0]))
    obs = np.arange(D.shape[0])
    noise = np.full(len(obs), 0.01)
    rng = np.random.default_rng(2)
    z = []
    for w in range(300):
        y = Z[w] + rng.normal(0, 0.1, len(obs))
        m, v = kr.predict(obs, y, noise, np.array([40]))          # place 40 is observed: must be excluded
        z.append((Z[w, 40] - m[0]) / np.sqrt(v[0]))
        assert not np.isclose(m[0], y[40])
    z = np.array(z)
    assert abs(z.mean()) < 0.2 and 0.8 < z.std() < 1.2         # standardized errors ~ N(0, 1)


def test_weighted_ridge_and_standardizer():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    X.loc[::10, "b"] = np.nan
    y = 1.0 + 2.0 * X.a.to_numpy() + rng.normal(0, 0.1, 500)
    st = Standardizer().fit(X)
    assert "b_missing" in st.names
    b0, beta = weighted_ridge(st.transform(X), y, np.ones(500), lam=1e-6)
    assert beta[0] * X.a.std() == pytest.approx(2.0 * X.a.std(), rel=0.05)


# -- end-to-end leakage checks on the real v1 data ---------------------------------------------------
real = pytest.mark.skipif(not (CFG.data_dir / "chlorine_v1.parquet").exists(), reason="v1 data not built")


@pytest.fixture(scope="module")
def real_data():
    from chlorine.evaluation.data import load_frame
    df, places = load_frame(CFG)
    return df, places, NeighborPanel(places, CFG.idw_length_km), place_folds(places, CFG.n_folds, CFG.seed)


def _public_models():
    return [RegressionKriging(CFG, "B7", "rk_public", site_history=False),
            Ensemble(CFG, "E7", "avg", [RegressionKriging(CFG, "B7", "rk_public", site_history=False),
                                        GBM(CFG, "G7", "gbm_public", site_history=False, region=False)])]


@real
def test_unseen_site_models_ignore_held_out_values(real_data):
    df, places, panel, folds = real_data
    f = 2
    base = run_unseen_site(df, panel, _public_models(), "val", CFG, folds, only_folds=(f,))
    pert = df.copy()
    pert.loc[df.place_group.map(folds).eq(f), "y"] = 1000.0
    again = run_unseen_site(pert, panel, _public_models(), "val", CFG, folds, only_folds=(f,))
    a = base[base.fold == f].set_index(["model", "row_id"])[["pred", "q10", "q90"]].sort_index()
    b = again[again.fold == f].set_index(["model", "row_id"])[["pred", "q10", "q90"]].sort_index()
    assert len(a) > 0
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


@real
def test_known_site_models_ignore_own_value_and_test_period(real_data):
    df, places, panel, _ = real_data
    models = lambda: [RegressionKriging(CFG, "B6", "rk_site", site_history=True),  # noqa: E731
                      GBM(CFG, "G6", "gbm_site", site_history=True, region=True)]
    base = run_known_site(df, panel, models(), "val", CFG).set_index(["model", "row_id"]).pred
    target = df[(df.sampled_at >= "2023-07-01") & (df.sampled_at < "2024-01-01")].row_id.iloc[5]
    pert = df.copy()
    pert.loc[pert.row_id == target, "y"] = 1000.0
    pert.loc[pert.sampled_at >= "2024-01-01", "y"] = -1000.0
    again = run_known_site(pert, panel, models(), "val", CFG).set_index(["model", "row_id"]).pred
    np.testing.assert_allclose(base.xs(target, level="row_id"), again.xs(target, level="row_id"))
