"""A fresh test set: freeze and pre-register the final models now, evaluate them once on future data.

The 2024-26 test period has been used twice (Step 2 references, Step 3 models), so it can no longer
judge new choices. The honest substitute is a prospective test: the final models are trained on all
data up to the cutoff, their specification, interval calibration and fingerprints are frozen in
reports/prospective_registration.json, and they are evaluated exactly once, on the next lab export's
samples after the cutoff (docs/prospective_registration.md).

    uv run python -m chlorine.modeling.prospective freeze      # writes the registration (done once, now)
    uv run python -m chlorine.modeling.prospective dry-run     # rehearses the whole path on 2023 data
    uv run python -m chlorine.modeling.prospective evaluate --raw NEW.xlsx --confirm   # once, on new data

The cutoff is a Sunday, the end of the last complete week of the v1 modeling table, so no week straddles
it: the models fitted inside the evaluation protocol are then identical to the frozen ones (checked by
fingerprint). `freeze` uses 2024-26 rows as training and calibration data for the prospective model;
that use is logged in the test-access ledger and no accuracy is computed.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, CleaningConfig, EvalConfig
from ..evaluation.data import load_frame
from ..evaluation.features import NeighborPanel, build_features
from ..evaluation.harness import run_known_site, run_unseen_site
from ..evaluation.metrics import bootstrap_rmse, summarize
from ..evaluation.models import reference_models
from ..evaluation.splits import place_folds
from .backtest import HALF_LIFE_DAYS, REFERENCE_CODES, backtest, e6, e7
from .calibration import AdaptiveLevels, ResidualPool, adaptive_intervals, interval_score

CUTOFF = "2026-08-02"                 # Sunday; the last modeling sample of v1 is 2026-07-31
DRY_RUN_CUTOFF, DRY_RUN_END = "2023-07-02", "2023-12-31"     # rehearsal inside development data only
REGISTRATION = ROOT / "reports" / "prospective_registration.json"
POOL_FIRST_YEAR = 2017
GAMMA = 0.1
PROBE_DAYS = 28
# Pre-registered 80% interval methods per protocol (primary chosen by the calibration selection rule).
INTERVALS = {"known_site": {"model": "E6", "primary": "C0", "secondary": "C2"},
             "unseen_site": {"model": "E7", "primary": "C2", "secondary": "C0"}}
COMPARATORS = {"known_site": "B5p", "unseen_site": "B4p"}
UNSEEN_DESCRIPTION = ("E7 = average(B7 regression kriging, G7 gradient boosting); no site history. G7 uses same-week "
                      "same-region neighbor contrasts, which need the target's utility sampling-region code (removing "
                      "them changes development RMSE by 0.000: modeling/sensitivity.py). Refit per fold with the frozen "
                      "specification in the unseen-site protocol.")
MIN_MONTHS = 12          # dry-run CI half-widths at 6 months: +-0.012 (H1), +-0.022 (H2, expected effect ~0.02)
HYPOTHESES = [
    "H1 (known sites): E6 has lower RMSE than B5p, the strongest Step 2 reference on the most recent data (2024-26; "
    "a deliberately hard bar); supported if the 95% cluster-bootstrap CI of the RMSE difference lies below 0.",
    "H2 (unseen sites): E7 has lower RMSE than B4p, likewise the strongest reference on 2024-26; same criterion.",
    "H3 (calibration): the primary 80% intervals (known: C0, unseen: C2) cover between 75% and 85% of post-cutoff "
    "samples in each protocol.",
    "H4 (outages, only if Kubala-outage samples exist after the cutoff): C2 coverage during outages is closer to 80% "
    "than C0 coverage.",
]


def _out(*parts) -> Path:
    p = EvalConfig().out_root.joinpath("prospective", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_files(files) -> str:
    """Python files are hashed by their syntax tree without docstrings (comments and docs may change; code may not);
    other files byte for byte."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        if f.suffix == ".py":
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                body = getattr(node, "body", None)
                if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                        and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    node.body = body[1:] or [ast.Pass()]
            h.update(ast.dump(tree, include_attributes=False).encode())
        else:
            h.update(f.read_bytes())
    return h.hexdigest()


def import_closure(start: Path) -> list[Path]:
    """Every package module `start` imports, directly or indirectly (relative imports, including function-level)."""
    seen, todo = set(), [start]
    while todo:
        f = todo.pop()
        for node in ast.walk(ast.parse(f.read_text())):
            if not (isinstance(node, ast.ImportFrom) and node.level > 0):
                continue
            base = f.parent
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                target = base.joinpath(*node.module.split("."))
                cands = [target.with_suffix(".py"), target / "__init__.py"]
            else:
                cands = [base / f"{a.name}.py" for a in node.names]
            for c in cands:
                if c.exists() and c not in seen:
                    seen.add(c)
                    todo.append(c)
    return sorted(seen)


def code_fingerprints() -> dict:
    """model: every module the frozen models, features and calibration import (must not change before
    evaluation; unrelated new modules do not count). cleaning: cleaning code and reference tables (may
    legitimately gain entries for new sites; reported). runner: this file (bug fixes allowed; reported)."""
    src = ROOT / "src" / "chlorine"
    runner = src / "modeling" / "prospective.py"
    model = [f for f in import_closure(runner) if "cleaning" not in f.relative_to(src).parts and f != runner]
    cleaning = sorted((src / "cleaning").glob("*.py")) + sorted((ROOT / "config").glob("*.csv"))
    return {"model": _hash_files(model), "cleaning": _hash_files(cleaning),
            "runner": _hash_files([runner])}


def eval_cfg(cutoff: str) -> EvalConfig:
    if pd.Timestamp(cutoff).dayofweek != 6:
        raise ValueError("the cutoff must be a Sunday (end of a W-SUN week) so no week straddles it")
    return dataclasses.replace(EvalConfig(), train_end=cutoff, val_end="2100-12-31")


def _probe(df: pd.DataFrame, panel: NeighborPanel, cutoff: str) -> pd.DataFrame:
    """Fixed probe rows: the last PROBE_DAYS of training data, with features built from pre-cutoff rows only."""
    pre = df[df.sampled_at < pd.Timestamp(cutoff) + pd.Timedelta(days=1)].reset_index(drop=True)
    feats = build_features(pre, np.ones(len(pre), bool), panel)
    return feats[feats.sampled_at >= pd.Timestamp(cutoff) - pd.Timedelta(days=PROBE_DAYS)]


def fingerprint(model, probe: pd.DataFrame) -> str:
    """Hash of the rounded probe predictions and bounds, as a sorted set of rows (independent of row ids and order,
    which a new export may renumber): equal iff the fitted models agree to 1e-6 mg/L on every probe row."""
    p = np.round(model.predict(probe)[["pred", "q10", "q90"]].to_numpy(), 6) + 0.0      # + 0.0 folds -0.0 into 0.0
    p = p[np.lexsort(p.T[::-1])]
    return hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()


def fit_final(df: pd.DataFrame, places: pd.DataFrame, cutoff: str) -> dict:
    """The frozen models: E6 and E7 fitted on every row up to the cutoff (no other rows exist for them)."""
    cfg = eval_cfg(cutoff)
    panel = NeighborPanel(places, cfg.idw_length_km)
    pre = df[df.sampled_at < pd.Timestamp(cutoff) + pd.Timedelta(days=1)].reset_index(drop=True)
    feats = build_features(pre, np.ones(len(pre), bool), panel)
    probe = _probe(df, panel, cutoff)
    out = {}
    for m in (e6(cfg), e7(cfg)):
        m.bind(feats, panel)
        m.fit(feats)
        out[m.code] = fingerprint(m, probe)
    return out


def run_after_cutoff(df: pd.DataFrame, places: pd.DataFrame, cutoff: str):
    """Known-site and unseen-site predictions for every sample after the cutoff, plus fingerprints of the
    known-site E6/E7 fitted inside the protocol (to compare with the frozen ones)."""
    cfg = eval_cfg(cutoff)
    panel = NeighborPanel(places, cfg.idw_length_km)
    folds = place_folds(places, cfg.n_folds, cfg.seed)
    refs = lambda: [m for m in reference_models(cfg) if m.code in REFERENCE_CODES]  # noqa: E731
    final = [e6(cfg), e7(cfg)]
    known = run_known_site(df, panel, refs() + final, "val", cfg)
    unseen = run_unseen_site(df, panel, refs() + [e7(cfg)], "val", cfg, folds)
    probe = _probe(df, panel, cutoff)
    prints = {m.code: fingerprint(m, probe) for m in final}
    preds = pd.concat([known, unseen], ignore_index=True)
    preds["week"] = preds.sampled_at.dt.to_period("W-SUN").dt.start_time
    return preds, prints


def calibration_state(pool: pd.DataFrame, cutoff: str) -> dict:
    """Frozen interval calibration: conformal quantiles at the cutoff, and the adaptive tail levels reached by
    replaying the out-of-sample pool year by year (each year calibrated only on earlier years, as in the study)."""
    pool = pool.assign(week=pool.sampled_at.dt.to_period("W-SUN").dt.start_time)
    state = {}
    for protocol, spec in INTERVALS.items():
        code = spec["model"]
        p = pool[(pool.protocol == protocol) & (pool.model == code)].sort_values("sampled_at")
        signal = pool[(pool.protocol == "known_site") & (pool.model == code)].sort_values("sampled_at")
        levels = AdaptiveLevels(GAMMA)
        for y in sorted(p.year.unique())[2:]:                 # the first two years only seed the pool
            past = ResidualPool(p[p.year < y], True, pd.Timestamp(f"{y}-01-01"))
            adaptive_intervals(p[p.year == y], signal[signal.year == y], past, levels)
        rp = ResidualPool(p, True, pd.Timestamp(cutoff))
        state[protocol] = {
            "model": code, "pool_rows": len(p), "pool_years": [int(p.year.min()), int(p.year.max())],
            "z_quantiles": {("outage" if r else "normal"): {"q10": rp.quantile(0.1, r), "q90": rp.quantile(0.9, r)}
                            for r in (False, True)},
            "adaptive_levels_at_cutoff": {"p_lo": levels.p_lo, "p_hi": levels.p_hi, "gamma": GAMMA},
        }
    return state


def apply_intervals(preds: pd.DataFrame, pool: pd.DataFrame, state: dict, cutoff: str) -> pd.DataFrame:
    """E6/E7 rows with C0 (the model's own) and C2 (adaptive conformal from the frozen state) intervals."""
    out = []
    for protocol, spec in INTERVALS.items():
        code = spec["model"]
        p = preds[(preds.protocol == protocol) & (preds.model == code)].sort_values("sampled_at").copy()
        signal = preds[(preds.protocol == "known_site") & (preds.model == code)].sort_values("sampled_at")
        rp = ResidualPool(pool[(pool.protocol == protocol) & (pool.model == code)], True, pd.Timestamp(cutoff))
        st = state[protocol]["adaptive_levels_at_cutoff"]
        levels = AdaptiveLevels(st["gamma"])
        levels.p_lo, levels.p_hi = st["p_lo"], st["p_hi"]
        p["c2_lo"], p["c2_hi"] = adaptive_intervals(p, signal, rp, levels)
        out.append(p)
    return pd.concat(out, ignore_index=True)


def score(preds: pd.DataFrame, ints: pd.DataFrame, reps: int = 1000, seed: int = 20261012) -> dict:
    """Every pre-registered metric: accuracy by model and subset, paired bootstrap vs. comparators, coverage."""
    cfg = EvalConfig()
    rows = []
    for (protocol, code), g in preds.groupby(["protocol", "model"]):
        subsets = {"all": g, "kubala_outage": g[g.jk_offline.astype(bool)]}
        if protocol == "known_site":
            subsets["new_site"] = g[g.new_site.eq(True)]
        for name, s in subsets.items():
            if len(s):
                rows.append({"protocol": protocol, "model": code, "subset": name,
                             **summarize(s, cfg.quantiles, (2.5,), cfg.min_site_eval_samples)})
    comp = pd.concat([bootstrap_rmse(preds[preds.protocol == pr], ref, reps, seed).assign(protocol=pr, comparator=ref)
                      for pr, ref in COMPARATORS.items()], ignore_index=True)
    cover = []
    for protocol, g in ints.groupby("protocol"):
        jk = g.jk_offline.to_numpy(bool)
        for method, lo, hi in (("C0", g.q10.to_numpy(), g.q90.to_numpy()), ("C2", g.c2_lo.to_numpy(), g.c2_hi.to_numpy())):
            inside = (g.y.to_numpy() >= lo) & (g.y.to_numpy() <= hi)
            cover.append({"protocol": protocol, "model": g.model.iloc[0], "method": method,
                          "primary": method == INTERVALS[protocol]["primary"], "n": len(g),
                          "coverage": float(inside.mean()), "n_outage": int(jk.sum()),
                          "coverage_outage": float(inside[jk].mean()) if jk.any() else np.nan,
                          "width": float(np.mean(hi - lo)),
                          "interval_score": float(np.mean(interval_score(g.y, lo, hi)))})
    return {"results": pd.DataFrame(rows), "comparisons": comp, "coverage": pd.DataFrame(cover)}


def _ledger(entry: dict):
    with open(EvalConfig().out_root / "test_access_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _pool_hash(pool: pd.DataFrame) -> str:
    cols = pool.sort_values(["protocol", "model", "row_id"])[["row_id", "pred", "sd", "y"]]
    return hashlib.sha256(pd.util.hash_pandas_object(cols, index=False).to_numpy().tobytes()).hexdigest()


POOL_COLS = ["protocol", "model", "row_id", "place_group", "sampled_at", "y", "pred", "sd", "q10", "q90",
             "jk_offline", "year"]


def freeze():
    if REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION} exists; the registration is written once")
    cfg = EvalConfig()
    dev = pd.read_parquet(cfg.out_root / "backtest" / "dev.parquet")
    _ledger({"split": "test", "step": "prospective-freeze", "generated_at": _now(),
             "purpose": "2024-26 used as training data and as out-of-sample calibration-pool rows for the frozen "
                        "prospective model; no accuracy or coverage computed or viewed"})
    later = backtest(range(2024, 2027), cfg, with_references=False)
    pool = pd.concat([dev[POOL_COLS], later[POOL_COLS]], ignore_index=True)
    pool = pool[pool.model.isin(["E6", "E7"]) & (pool.year >= POOL_FIRST_YEAR)]
    pool = pool[pool.sampled_at < pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)]
    pool.to_parquet(_out() / "calibration_pool.parquet", index=False)
    df, places = load_frame(cfg)
    n_train = int((df.sampled_at < pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)).sum())
    if n_train != len(df):
        raise SystemExit("the cutoff must include every v1 modeling sample")
    reg = {
        "registered_at": _now(), "cutoff": CUTOFF,
        "evaluation_rows": "in-scope routine total-chlorine samples (v1 cleaning rules) with sampled_at after the cutoff",
        "data_version": cfg.data_version,
        "raw_sha256": json.loads((cfg.data_dir / "manifest.json").read_text())["raw_sha256"],
        "training_rows": n_train, "training_places": int(df.place_group.nunique()),
        "code_fingerprints": code_fingerprints(),
        "models": {"known_site": "E6 = average(B6 regression kriging with site effects, G6 gradient boosting)",
                   "unseen_site": UNSEEN_DESCRIPTION,
                   "comparators": COMPARATORS, "other_references": [c for c in REFERENCE_CODES if c not in
                                                                    COMPARATORS.values()],
                   "half_life_days": HALF_LIFE_DAYS},
        "model_fingerprints": fit_final(df, places, CUTOFF),
        "intervals": INTERVALS,
        "calibration": calibration_state(pool, CUTOFF),
        "calibration_pool_sha256": _pool_hash(pool),
        "metrics": ["RMSE (95% cluster-bootstrap CI over places) and paired RMSE difference vs. the comparator",
                    "coverage, width and interval score of 80% intervals (C0 and C2)",
                    "AUC and recall for samples < 2.5 mg/L", "subsets: Kubala-outage samples, new sites"],
        "hypotheses": HYPOTHESES,
        "when": f"once, on the first export with at least {MIN_MONTHS} months of samples after the cutoff",
        "command": "uv run python -m chlorine.modeling.prospective evaluate --raw NEW.xlsx --confirm",
    }
    REGISTRATION.write_text(json.dumps(reg, indent=2))
    print(f"registered -> {REGISTRATION}")
    print(json.dumps({k: reg[k] for k in ("cutoff", "training_rows", "model_fingerprints")}, indent=2))


def dry_run():
    """Rehearse the whole evaluation path inside development data: a pseudo export ending DRY_RUN_END, frozen at
    DRY_RUN_CUTOFF. Only structural checks are printed; the metrics are written to disk to prove the path runs."""
    cfg = EvalConfig()
    df, places = load_frame(cfg)
    df = df[df.sampled_at < pd.Timestamp(DRY_RUN_END) + pd.Timedelta(days=1)].reset_index(drop=True)
    frozen_a = fit_final(df, places, DRY_RUN_CUTOFF)
    frozen_b = fit_final(df, places, DRY_RUN_CUTOFF)
    preds, inside = run_after_cutoff(df, places, DRY_RUN_CUTOFF)
    dev = pd.read_parquet(cfg.out_root / "backtest" / "dev.parquet")
    pool = dev[dev.model.isin(["E6", "E7"]) & (dev.sampled_at < pd.Timestamp(DRY_RUN_CUTOFF) + pd.Timedelta(days=1))]
    state = calibration_state(pool[POOL_COLS], DRY_RUN_CUTOFF)
    ints = apply_intervals(preds, pool, state, DRY_RUN_CUTOFF)
    res = score(preds, ints, reps=200)
    out = _out("dry_run")
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    after = df[df.sampled_at >= pd.Timestamp(DRY_RUN_CUTOFF) + pd.Timedelta(days=1)]
    known = preds[preds.protocol == "known_site"]
    unseen = preds[preds.protocol == "unseen_site"]
    checks = {
        "frozen models refit deterministically": frozen_a == frozen_b,
        "models fitted inside the protocol are identical to the frozen models": inside == frozen_a,
        "every post-cutoff sample predicted by every known-site model":
            all(set(g.row_id) == set(after.row_id) for _, g in known.groupby("model")),
        "every post-cutoff sample predicted by every unseen-site model":
            all(set(g.row_id) == set(after.row_id) for _, g in unseen.groupby("model")),
        "no prediction for a sample at or before the cutoff":
            bool((preds.sampled_at >= pd.Timestamp(DRY_RUN_CUTOFF) + pd.Timedelta(days=1)).all()),
        "C2 intervals for every E6/E7 row, lower < upper":
            bool(ints[["c2_lo", "c2_hi"]].notna().all().all() and (ints.c2_lo < ints.c2_hi).all()),
        "every registered metric computed": bool(len(res["results"]) and res["comparisons"].d_rmse_lo.notna().any()
                                                 and len(res["coverage"]) == 4),
        "calibration pool contains no row after the cutoff": bool(pool.sampled_at.max() <= pd.Timestamp(DRY_RUN_CUTOFF)
                                                                  + pd.Timedelta(days=1)),
    }
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    print(f"dry run: frozen {DRY_RUN_CUTOFF}, {len(after):,} later samples (2023, development data); "
          f"metrics written to {out}, not shown")
    return checks


def evaluate(raw: Path, confirm: bool):
    if not confirm:
        raise SystemExit("evaluate runs once, on new data; add --confirm")
    out = EvalConfig().out_root / "prospective" / "results"
    if (out / "results.csv").exists():
        raise SystemExit(f"already evaluated ({out}); the prospective test is used once")
    reg = json.loads(REGISTRATION.read_text())
    now = code_fingerprints()
    if now["model"] != reg["code_fingerprints"]["model"]:
        raise SystemExit("model code changed since registration: this would not test the registered models")
    pool = pd.read_parquet(_out() / "calibration_pool.parquet")
    if _pool_hash(pool) != reg["calibration_pool_sha256"]:
        raise SystemExit("the calibration pool changed since registration")
    from ..cleaning.pipeline import run as clean
    clean(dataclasses.replace(CleaningConfig(), raw_xlsx=raw, version="prospective"))
    ecfg = dataclasses.replace(EvalConfig(), data_version="prospective")
    df, places = load_frame(ecfg)
    cutoff = reg["cutoff"]
    n_after = int((df.sampled_at >= pd.Timestamp(cutoff) + pd.Timedelta(days=1)).sum())
    months = (df.sampled_at.max() - pd.Timestamp(cutoff)).days / 30.4
    if months < MIN_MONTHS:
        raise SystemExit(f"only {months:.1f} months after the cutoff ({n_after} samples); registered minimum is "
                         f"{MIN_MONTHS}")
    _ledger({"split": "prospective", "generated_at": _now(), "rows_after_cutoff": n_after,
             "code_fingerprints_match": {k: now[k] == reg["code_fingerprints"][k] for k in now}})
    frozen = fit_final(df, places, cutoff)
    preds, inside = run_after_cutoff(df, places, cutoff)
    ints = apply_intervals(preds, pool, reg["calibration"], cutoff)
    res = score(preds, ints)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    preds.drop(columns="week").to_parquet(out / "predictions.parquet", index=False)
    integrity = {"history_reproduces_frozen_models": {k: frozen[k] == reg["model_fingerprints"][k] for k in frozen},
                 "protocol_models_equal_frozen": inside == frozen,
                 "cleaning_code_changed": now["cleaning"] != reg["code_fingerprints"]["cleaning"]}
    (out / "integrity.json").write_text(json.dumps(integrity, indent=2))
    print(res["results"][res["results"].subset == "all"].to_string(index=False))
    print(res["comparisons"].to_string(index=False))
    print(res["coverage"].to_string(index=False))
    print(json.dumps(integrity, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    sub.add_parser("dry-run")
    e = sub.add_parser("evaluate")
    e.add_argument("--raw", type=Path, required=True)
    e.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    if args.cmd == "freeze":
        freeze()
    elif args.cmd == "dry-run":
        dry_run()
    else:
        evaluate(args.raw, args.confirm)


if __name__ == "__main__":
    main()
