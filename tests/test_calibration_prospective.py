import json

import numpy as np
import pandas as pd
import pytest

from chlorine.config import ROOT, EvalConfig
from chlorine.modeling import prospective
from chlorine.modeling.calibration import (P_HI, P_LO, AdaptiveLevels, ResidualPool, adaptive_intervals,
                                           interval_score, select)

CFG = EvalConfig()


def frame(n, seed=0, start="2020-01-06", jk=False, scale=1.0):
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) % 364, unit="D")
    pred = rng.uniform(2, 4, n)
    return pd.DataFrame({"sampled_at": t, "pred": pred, "sd": np.full(n, scale), "jk_offline": np.full(n, jk),
                         "y": pred + scale * rng.standard_normal(n)}).assign(
        week=lambda d: d.sampled_at.dt.to_period("W-SUN").dt.start_time)


def test_interval_score():
    y, lo, hi = np.array([1.0, 0.0, 3.0]), np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])
    # inside: width; below: width + 2/0.2 x 0.5; above: width + 2/0.2 x 1.5
    assert interval_score(y, lo, hi, 0.2) == pytest.approx([1.0, 6.0, 16.0])


def test_residual_pool_quantiles_regimes_and_recency():
    normal, outage = frame(4000, 1), frame(4000, 2, jk=True, scale=3.0)
    pool = ResidualPool(pd.concat([normal, outage]), scaled=False, ref_time="2021-01-01")
    z90 = 1.2816
    assert pool.quantile(0.9, False) == pytest.approx(z90, abs=0.1)
    assert pool.quantile(0.9, True) == pytest.approx(3 * z90, abs=0.3)          # separate outage regime
    lo, hi = pool.intervals(pd.concat([normal, outage]))
    inside = (pd.concat([normal, outage]).y.to_numpy() >= lo) & (pd.concat([normal, outage]).y.to_numpy() <= hi)
    assert inside.mean() == pytest.approx(0.8, abs=0.02)
    # SD-scaled residuals: same z-quantiles whatever the scale
    assert ResidualPool(outage, True, "2021-01-01").quantile(0.9, True) == pytest.approx(z90, abs=0.1)
    # recency: an old wide regime is down-weighted against a recent narrow one
    old, new = frame(3000, 3, start="2010-01-04", scale=3.0), frame(3000, 4, start="2020-01-06")
    q = ResidualPool(pd.concat([old, new]), False, "2021-01-01").quantile(0.9, False)
    assert q < 1.6


def test_adaptive_levels_move_the_right_way():
    lv = AdaptiveLevels(0.1)
    lv.update(miss_lo=0.5, miss_hi=0.5)        # too many misses -> widen
    assert lv.p_lo < P_LO and lv.p_hi > P_HI
    lv = AdaptiveLevels(0.1)
    lv.update(0.0, 0.0)                        # no misses -> narrow
    assert lv.p_lo > P_LO and lv.p_hi < P_HI
    lv = AdaptiveLevels(10.0)
    lv.update(1.0, 1.0)
    assert 0.005 <= lv.p_lo and lv.p_hi <= 0.995  # clipped


def test_adaptive_intervals_use_only_past_weeks():
    pool = ResidualPool(frame(3000, 5), False, "2021-01-01")
    target = frame(700, 6, start="2021-01-04")
    lo, hi = adaptive_intervals(target, target, pool, AdaptiveLevels(0.1))
    changed = target.copy()
    last = changed.week == changed.week.max()
    changed.loc[last, "y"] += 50.0                 # outcomes of the last week cannot change any interval
    lo2, hi2 = adaptive_intervals(changed, changed, pool, AdaptiveLevels(0.1))
    assert np.allclose(lo, lo2) and np.allclose(hi, hi2)
    first = (target.week == target.week.min()).to_numpy()
    base_lo, base_hi = pool.intervals(target[first])
    assert np.allclose(lo[first], base_lo) and np.allclose(hi[first], base_hi)   # week 1 uses the initial levels


def test_selection_rule():
    rows = []
    for method, cov, score in (("A", 0.70, 0.9), ("B", 0.80, 1.1), ("C", 0.84, 1.0)):
        rows += [{"protocol": "known_site", "method": method, "year": y, "coverage": cov, "coverage_outage": cov,
                  "width": 1.0, "interval_score": score} for y in (2019, 2020)]
    assert select(pd.DataFrame(rows))["chosen"]["known_site"]["method"] == "C"   # A is sharper but out of band


def test_cutoff_must_be_a_sunday():
    assert prospective.eval_cfg("2026-08-02").train_end == "2026-08-02"
    with pytest.raises(ValueError):
        prospective.eval_cfg("2026-08-11")


def test_code_hash_ignores_docs_but_not_code(tmp_path, monkeypatch):
    monkeypatch.setattr(prospective, "ROOT", tmp_path)
    f = tmp_path / "m.py"
    f.write_text('"""Doc."""\nX = 1  # note\n\ndef g():\n    """Doc."""\n    return X\n')
    h = prospective._hash_files([f])
    f.write_text('"""Other doc."""\nX = 1\n\ndef g():\n    return X\n')
    assert prospective._hash_files([f]) == h
    f.write_text('"""Doc."""\nX = 2\n\ndef g():\n    return X\n')
    assert prospective._hash_files([f]) != h


def test_fingerprint_ignores_row_order():
    class Fixed:
        def predict(self, df):
            return pd.DataFrame({"pred": df.v, "q10": df.v - 1, "q90": df.v + 1}, index=df.index)
    probe = pd.DataFrame({"row_id": [3, 1, 2], "v": [0.1, 0.2, 0.3]})
    a = prospective.fingerprint(Fixed(), probe)
    assert prospective.fingerprint(Fixed(), probe.iloc[::-1].assign(row_id=[7, 8, 9])) == a
    assert prospective.fingerprint(Fixed(), probe.assign(v=[0.1, 0.2, 0.3000011])) != a


registered = pytest.mark.skipif(not prospective.REGISTRATION.exists(), reason="no registration")


@registered
def test_registration_is_complete_and_matches_code():
    reg = json.loads(prospective.REGISTRATION.read_text())
    for key in ("cutoff", "code_fingerprints", "model_fingerprints", "intervals", "calibration", "hypotheses",
                "calibration_pool_sha256", "when"):
        assert key in reg
    assert pd.Timestamp(reg["cutoff"]).dayofweek == 6
    assert reg["code_fingerprints"]["model"] == prospective.code_fingerprints()["model"], \
        "model code changed after registration: the prospective test would not test the registered models"
    pool = pd.read_parquet(CFG.out_root / "prospective" / "calibration_pool.parquet")
    assert prospective._pool_hash(pool) == reg["calibration_pool_sha256"]
    assert pool.sampled_at.max() <= pd.Timestamp(reg["cutoff"]) + pd.Timedelta(days=1)


@pytest.mark.skipif(not (ROOT / "reports" / "key_numbers.json").exists(), reason="numbers sheet not built")
def test_numbers_sheet_checks_pass_and_ids_unique():
    sheet = json.loads((ROOT / "reports" / "key_numbers.json").read_text())
    assert all(c["ok"] for c in sheet["checks"]), [c["check"] for c in sheet["checks"] if not c["ok"]]
    ids = [r["id"] for r in sheet["numbers"]]
    assert len(ids) == len(set(ids))


def test_model_fingerprint_covers_exactly_the_frozen_models_imports():
    files = {f.name for f in prospective.import_closure(ROOT / "src/chlorine/modeling/prospective.py")
             if "cleaning" not in f.parts}
    assert {"rk.py", "gbm.py", "ensemble.py", "calibration.py", "features.py", "harness.py", "config.py"} <= files
    assert not files & {"sensitivity.py", "figures.py", "report.py", "run.py", "numbers.py"}
