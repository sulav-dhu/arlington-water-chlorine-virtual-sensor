"""Key-numbers sheet: every number the slides quote, computed from the result files, each with its source.

    uv run python -m chlorine.reporting.figures     # first: some numbers are read from the figure CSVs
    uv run python -m chlorine.reporting.numbers     # -> reports/key_numbers.md + reports/key_numbers.json

Each number has a stable ID (e.g. `clean.samples`), a display string, the raw value and the file it came from.
Slides and documents cite the display string of an ID, so text and figures cannot drift apart. Consistency
checks assert that numbers available from two sources agree (e.g. the 2023 backtest year reproduces the
Step 3 validation run exactly); the script fails if any check fails.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, CleaningConfig, EvalConfig
from ..evaluation.data import load_frame
from ..evaluation.figures import OUT as FIG
from .figures import funnel_numbers

REPORTS = ROOT / "reports"


def mg(x) -> str:
    return f"{x:.3f}"


def pct(x, nd=0) -> str:
    return f"{100 * x:.{nd}f}%"


def ci(lo, hi) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


@dataclass
class Sheet:
    rows: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    section: str = ""

    def add(self, key: str, value, display: str, label: str, source):
        assert key not in {r["id"] for r in self.rows}, key
        src = [source] if isinstance(source, (str, Path)) else list(source)
        self.rows.append({"section": self.section, "id": key, "value": _plain(value), "display": display, "label": label,
                          "source": ", ".join(Path(s).relative_to(ROOT).as_posix() if Path(s).is_absolute() else str(s)
                                              for s in src)})

    def check(self, name: str, ok: bool, detail: str = ""):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})


def _results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index(["protocol", "subset", "model"])


def build() -> Sheet:
    ccfg, cfg = CleaningConfig(), EvalConfig()
    ev = cfg.out_root
    s = Sheet()

    # -- data and cleaning ----------------------------------------------------------------------------------
    s.section = "Data and cleaning"
    man_p = ccfg.out_dir / "manifest.json"
    man = json.loads(man_p.read_text())
    fn = funnel_numbers()
    df, places = load_frame(cfg)
    steps = {label: n for label, n, _ in fn["steps"]}
    s.add("clean.raw_rows", man["rows"]["raw"], f"{man['rows']['raw']:,}", "rows in the lab export (20 tests)", man_p)
    s.add("clean.chlorine_results", steps["Total-chlorine results"], f"{steps['Total-chlorine results']:,}",
          "total-chlorine field results", FIG / "fig6_cleaning_funnel.csv")
    prev = None
    for label, n, why in fn["steps"][1:]:
        if prev is not None:
            key = "clean.removed." + label.lower().replace(" ", "_").replace("-", "_")
            s.add(key, prev - n, f"{prev - n:,}", f"removed at step '{label}': {why}", FIG / "fig6_cleaning_funnel.csv")
        prev = n
    s.add("clean.modeling_rows", man["rows"]["chlorine_v1"], f"{man['rows']['chlorine_v1']:,}",
          "rows in the modeling table chlorine_v1", man_p)
    s.add("clean.samples", len(df), f"{len(df):,}", "modeling samples with a numeric value (2 non-detects excluded)",
          cfg.data_dir / "chlorine_v1.parquet")
    s.add("clean.locations", df.location_id.nunique(), f"{df.location_id.nunique()}",
          "sampling locations (site codes resolved to physical locations)", cfg.data_dir / "chlorine_v1.parquet")
    s.add("clean.places", df.place_group.nunique(), f"{df.place_group.nunique()}", "places (place groups) sampled",
          cfg.data_dir / "chlorine_v1.parquet")
    first, last = df.sampled_at.min(), df.sampled_at.max()
    s.add("clean.period", [str(first.date()), str(last.date())], f"{first:%b %Y} - {last:%b %Y}",
          "first and last modeling sample", cfg.data_dir / "chlorine_v1.parquet")
    years = (last - first).days / 365.25
    s.add("clean.samples_per_year", len(df) / years, f"{len(df) / years:,.0f}", "modeling samples per year",
          cfg.data_dir / "chlorine_v1.parquet")
    fix_keys = ["address_linked", "multi_spelling_codes", "grouped_places", "duplicates_removed", "phone_rows",
                "geocoded_share"]
    assert len(fix_keys) == len(fn["fixes"])
    for key, (k, v) in zip(fix_keys, fn["fixes"].items()):
        src = ccfg.raw_xlsx.name + " (local, never copied)" if key == "phone_rows" else ccfg.out_dir / "measurements.parquet"
        s.add("clean.fix." + key, v, pct(v, 1) if isinstance(v, float) else f"{v:,}", k,
              ["src/chlorine/reporting/figures.py:funnel_numbers", src])
    s.add("clean.gis_layers", man["gis_counts"], ", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in
                                                         man["gis_counts"].items()),
          "public city GIS features used for geocoding and covariates", man_p)
    s.check("funnel ends at the modeling table", fn["steps"][-1][1] == man["rows"]["chlorine_v1"])
    s.check("modeling samples = table rows - non-detects",
            len(df) == man["rows"]["chlorine_v1"] - int(pd.read_parquet(cfg.data_dir / "chlorine_v1.parquet").censored.sum()))
    s.check("funnel place count = analysis place count", fn["places"] == df.place_group.nunique())
    s.check("all cleaning checks PASS or WARN, none FAIL", "FAIL" not in man["checks"].values())

    # -- noise floor and variance ----------------------------------------------------------------------------
    s.section = "Noise floor and sources of variation"
    run_p = ev / "val" / "run.json"
    floor = json.loads(run_p.read_text())["noise_floor"]
    s.add("noise.measurement_sd", floor["measurement_sd"], f"{floor['measurement_sd']:.2f} mg/L",
          f"measurement noise: SD of same-moment replicate readings ({floor['replicate_pairs']} pairs)", run_p)
    s.add("noise.within_week_sd", floor["within_week_sd"], f"{floor['within_week_sd']:.2f} mg/L",
          "within-week scatter at the near-daily sites: the error of knowing each place's weekly mean exactly", run_p)
    var_p = FIG / "fig8_variance_breakdown.csv"
    var = pd.read_csv(var_p)
    keys = ["city_week", "site_level", "local_week", "within_week"]
    for key, r in zip(keys, var.itertuples()):
        s.add(f"var.{key}", r.share, pct(r.share), f"share of variance: {r.component} (SD {r.sd_mg_l:.2f} mg/L)", var_p)
    s.add("var.total_sd", float(df.y.std()), f"{df.y.std():.2f} mg/L", "SD of all modeling samples",
          cfg.data_dir / "chlorine_v1.parquet")
    s.check("variance shares sum to 1", abs(var.share.sum() - 1) < 1e-9)
    s.check("fig8 noise floor = evaluation noise floor", abs(var.sd_mg_l.iloc[3] - floor["within_week_sd"]) < 1e-9)

    # -- plant outages ---------------------------------------------------------------------------------------
    s.section = "Plant outages"
    ev_p = cfg.data_dir / "events.csv"
    events = pd.read_csv(ev_p)
    jk = events[(events.plant == "JK") & (events.source == "inferred_from_plant_sampling_gaps")]
    supported = jk[jk.confidence.isin(["externally_confirmed", "supported_by_dip"])]
    s.add("outage.kubala_inferred", len(jk), f"{len(jk)}", "Kubala outages inferred from gaps in plant sampling", ev_p)
    s.add("outage.kubala_with_dip", len(supported), f"{len(supported)} of {len(jk)}",
          "inferred Kubala outages that coincide with a city-wide dip (one also documented by a city notice)", ev_p)
    s.add("outage.dip_range", [-supported.city_dip_mg_l.max(), -supported.city_dip_mg_l.min()],
          f"{-supported.city_dip_mg_l.max():.1f}-{-supported.city_dip_mg_l.min():.1f} mg/L",
          "city-wide drop in residual during those outages", ev_p)
    s.add("outage.share_of_samples", float(df.jk_offline.mean()), pct(df.jk_offline.mean()),
          "share of modeling samples taken while Kubala was offline", cfg.data_dir / "chlorine_v1.parquet")
    plane_p = FIG / "fig7_kubala_outages_by_plane.csv"
    for r in pd.read_csv(plane_p).itertuples():
        s.add(f"outage.drop.{r.pressure_plane.lower()}", r.mean_drop, f"{r.mean_drop:+.2f} mg/L ({r.places} places)",
              f"mean change at sites in the {r.pressure_plane.title()} plane during outages (same calendar months)",
              plane_p)

    # -- Step 2 baselines -------------------------------------------------------------------------------------
    s.section = "Baselines (Step 2)"
    val_p, test_p = ev / "val" / "results.csv", ev / "test" / "results.csv"
    val, test = _results(val_p), _results(test_p)
    for split, res, path in (("val", val, val_p), ("test", test, test_p)):
        for protocol in ("known_site", "unseen_site"):
            for code in ("B0", "B1", "B4", "B4p", "B5", "B5p"):
                if (protocol, "all", code) not in res.index:
                    continue
                r = res.loc[(protocol, "all", code)]
                s.add(f"base.{split}.{protocol}.{code}", r.rmse, f"{mg(r.rmse)} [{mg(r.rmse_lo)}-{mg(r.rmse_hi)}]",
                      f"RMSE mg/L [95% CI], {code}, {protocol.replace('_', ' ')}, "
                      f"{'validation 2023' if split == 'val' else 'test 2024-26'}", path)
    b0, b5 = val.loc[("known_site", "all", "B0")].rmse, val.loc[("known_site", "all", "B5")].rmse
    s.add("base.val.gain_neighbors", 1 - b5 / b0, pct(1 - b5 / b0),
          "error cut from overall mean (B0) to best neighbor baseline (B5), known sites 2023", val_p)
    lc_p = ev / "val" / "learning_curve.csv"
    lc = pd.read_csv(lc_p).groupby(["model", "fraction"]).agg(places=("monitored_places", "mean"), rmse=("rmse", "mean"))
    lo, hi = lc.loc[("B4", 0.25)], lc.loc[("B4", 1.0)]
    s.add("base.learning_curve.B4", [lo.rmse, hi.rmse], f"{mg(lo.rmse)} with {lo.places:.0f} sites -> {mg(hi.rmse)} with "
          f"{hi.places:.0f}", "unseen-site RMSE of B4 as more places are monitored (validation 2023)", lc_p)
    cov = val.xs("all", level="subset")
    cov = cov[cov.index.get_level_values("model").isin(["B4", "B4p", "B5", "B5p"])
              & cov.index.get_level_values("protocol").isin(["known_site", "unseen_site"])]
    s.add("base.val.coverage_range", [cov.coverage80.min(), cov.coverage80.max()],
          f"{pct(cov.coverage80.min())}-{pct(cov.coverage80.max())}",
          "80%-interval coverage of the neighbor baselines in calm 2023 (too wide)", val_p)

    # -- Step 3 final models ------------------------------------------------------------------------------------
    s.section = "Final models (Step 3: E6 known sites, E7 unseen sites)"
    s3v_p, s3t_p = ev / "step3" / "val" / "results.csv", ev / "step3" / "test" / "results.csv"
    s3v, s3t = _results(s3v_p), _results(s3t_p)
    cmp_p = {"val": ev / "step3" / "val" / "comparisons.csv", "test": ev / "step3" / "test" / "comparisons.csv"}
    for split, res, path in (("val", s3v, s3v_p), ("test", s3t, s3t_p)):
        comp = pd.read_csv(cmp_p[split]).set_index(["protocol", "model"])
        for protocol, code, headline in (("known_site", "E6", True), ("unseen_site", "E7", True), ("outage", "E6", True),
                                         ("known_site", "B6", False), ("unseen_site", "B7", False),
                                         ("known_site", "E7", False)):
            if (protocol, "all", code) not in res.index:           # the outage protocol is validation-only
                continue
            r = res.loc[(protocol, "all", code)]
            period = "pre-2024 Kubala outages, leave-one-out" if protocol == "outage" else (
                "validation 2023" if split == "val" else "test 2024-26")
            name = f"{period}, {protocol.replace('_', ' ')}"
            s.add(f"final.{split}.{protocol}.{code}", r.rmse, mg(r.rmse), f"RMSE mg/L, {code}, {name}", path)
            if headline and (protocol, code) in comp.index:
                c = comp.loc[(protocol, code)]
                ref = res.loc[(protocol, "all", c.comparator)].rmse
                s.add(f"final.{split}.{protocol}.{code}_vs_{c.comparator}", r.rmse - ref,
                      f"{r.rmse - ref:+.3f} {ci(c.d_rmse_lo, c.d_rmse_hi)}",
                      f"RMSE difference {code} - {c.comparator} [95% cluster-bootstrap CI], {name}", cmp_p[split])
                s.add(f"final.{split}.{protocol}.{code}_gain_vs_{c.comparator}", 1 - r.rmse / ref, pct(1 - r.rmse / ref),
                      f"relative error cut {code} vs. {c.comparator}, {name}", path)
            if code in ("E6", "E7"):
                s.add(f"final.{split}.{protocol}.{code}_coverage80", r.coverage80, pct(r.coverage80),
                      f"80%-interval coverage, {code}, {name} (original in-sample calibration)", path)
                s.add(f"final.{split}.{protocol}.{code}_auc_lt25", r.auc_lt25, f"{r.auc_lt25:.2f}",
                      f"AUC for flagging samples < 2.5 mg/L, {code}, {name}", path)
                s.add(f"final.{split}.{protocol}.{code}_alarm_lt25", [r.recall_lt25, r.alarm_rate_lt25, r.n_lt25 / r.n],
                      f"catches {pct(r.recall_lt25)} of samples < 2.5 while flagging {pct(r.alarm_rate_lt25)} "
                      f"(base rate {pct(r.n_lt25 / r.n, 1)})",
                      f"alarm rule 'lower 10% bound < 2.5 mg/L', {code}, {name}", path)
    for subset, label in (("jk_offline", "during Kubala outages"), ("new_sites", "at places new since 2023")):
        r, b = s3t.loc[("known_site", subset, "E6")], s3t.loc[("known_site", subset, "B5p")]
        s.add(f"final.test.known_site.E6.{subset}", [r.rmse, b.rmse], f"{mg(r.rmse)} vs. B5p {mg(b.rmse)} (n={r.n:,.0f})",
              f"test 2024-26 RMSE {label}, known sites", s3t_p)
    cov = [s3t.loc[k].coverage80 for k in (("known_site", "all", "E6"), ("known_site", "all", "E7"),
                                           ("unseen_site", "all", "E7"))]
    s.add("final.test.coverage_range", [min(cov), max(cov)], f"{pct(min(cov))}-{pct(max(cov))}",
          "80%-interval coverage on test 2024-26 of E6 and E7 in both protocols (original in-sample calibration): "
          "too narrow, the problem addressed below", s3t_p)

    # -- rolling backtest -------------------------------------------------------------------------------------------
    s.section = "Rolling backtest 2017-2023 (development data only)"
    bt_p = FIG / "fig10_backtest_by_year.csv"
    bt = pd.read_csv(bt_p, dtype={"year": str})
    yearly = bt[bt.year != "2017-2023"]
    pooled = bt[bt.year == "2017-2023"].set_index(["protocol", "model"])
    for protocol, final, base, krig in (("known_site", "E6", "B5p", "B6"), ("unseen_site", "E7", "B4p", "B7")):
        piv = yearly[yearly.protocol == protocol].pivot(index="year", columns="model", values="rmse")
        wins_b, wins_k = int((piv[final] < piv[base]).sum()), int((piv[final] < piv[krig]).sum())
        s.add(f"backtest.{protocol}.{final}_beats_{base}_years", wins_b, f"{wins_b} of {len(piv)}",
              f"years in which {final} has lower RMSE than {base}", bt_p)
        s.add(f"backtest.{protocol}.{final}_beats_{krig}_years", wins_k, f"{wins_k} of {len(piv)}",
              f"years in which {final} has lower RMSE than kriging alone ({krig})", bt_p)
        for other in (base, krig):
            r = pooled.loc[(protocol, other)]
            s.add(f"backtest.{protocol}.{final}_minus_{other}", r.final_minus_this,
                  f"{r.final_minus_this:+.3f} {ci(r.diff_lo, r.diff_hi)}",
                  f"pooled 2017-2023 RMSE difference {final} - {other} [95% CI, places resampled]", bt_p)
    dev = pd.concat([pd.read_parquet(ev / "backtest" / "dev.parquet"),
                     pd.read_parquet(ev / "backtest" / "dev_members.parquet")], ignore_index=True)
    from ..evaluation.metrics import bootstrap_rmse
    for protocol, krig, base in (("known_site", "B6", "B5p"), ("unseen_site", "B7", "B4p")):
        u = dev[dev.protocol == protocol]
        b = bootstrap_rmse(u[u.model.isin([krig, base])], base, 1000, cfg.seed).set_index("model").loc[krig]
        d = float(np.sqrt(np.mean((u[u.model == krig].pred - u[u.model == krig].y) ** 2))
                  - np.sqrt(np.mean((u[u.model == base].pred - u[u.model == base].y) ** 2)))
        s.add(f"backtest.{protocol}.{krig}_minus_{base}", d, f"{d:+.3f} {ci(b.d_rmse_lo, b.d_rmse_hi)}",
              f"pooled 2017-2023: kriging alone ({krig}) minus the best baseline ({base}), {protocol.replace('_', ' ')}",
              ev / "backtest" / "dev_members.parquet")
    ps_p = ev / "backtest" / "dev_public_strict.parquet"
    ps = pd.read_parquet(ps_p)
    for protocol in ("unseen_site", "known_site"):
        g = pd.concat([dev[(dev.protocol == protocol) & (dev.model == "E7")], ps[(ps.protocol == protocol) & (ps.model == "E7s")]])
        b = bootstrap_rmse(g, "E7", 1000, cfg.seed).set_index("model").loc["E7s"]
        d = float(np.sqrt(np.mean((g[g.model == "E7s"].pred - g[g.model == "E7s"].y) ** 2))
                  - np.sqrt(np.mean((g[g.model == "E7"].pred - g[g.model == "E7"].y) ** 2)))
        s.add(f"backtest.{protocol}.E7s_minus_E7", d, f"{d:+.3f} {ci(b.d_rmse_lo, b.d_rmse_hi)}",
              f"pooled 2017-2023: strictly public E7s (no utility region code) minus E7, {protocol.replace('_', ' ')}",
              ps_p)
    # The 2023 and 2022 backtest years use exactly the Step 3 validation setups: they must reproduce them.
    for protocol, code in (("known_site", "E6"), ("unseen_site", "E7"), ("known_site", "B6"), ("unseen_site", "B7")):
        got = yearly[(yearly.protocol == protocol) & (yearly.model == code) & (yearly.year == "2023")].rmse.iloc[0]
        s.check(f"backtest 2023 {code} {protocol} = Step 3 validation", abs(got - s3v.loc[(protocol, "all", code)].rmse) < 1e-9,
                f"{got:.6f}")
    second = pd.read_csv(ev / "step3" / "val" / "second_year.csv").set_index(["protocol", "model"])
    for protocol, code in (("known_site", "E6"), ("unseen_site", "E7")):
        got = yearly[(yearly.protocol == protocol) & (yearly.model == code) & (yearly.year == "2022")].rmse.iloc[0]
        s.check(f"backtest 2022 {code} {protocol} = Step 3 second validation year",
                abs(got - second.loc[(protocol, code)].rmse_2022) < 1e-9, f"{got:.6f}")
    for protocol, code in (("known_site", "B5p"), ("unseen_site", "B4p")):
        got = yearly[(yearly.protocol == protocol) & (yearly.model == code) & (yearly.year == "2023")].rmse.iloc[0]
        s.check(f"backtest 2023 {code} {protocol} = Step 2 validation", abs(got - val.loc[(protocol, "all", code)].rmse) < 1e-9)

    # -- prediction intervals ---------------------------------------------------------------------------------------
    s.section = "Prediction intervals (rolling calibration study 2019-2023)"
    st_p, sel_p = ev / "calibration" / "study.csv", ev / "calibration" / "selection.json"
    st = pd.read_csv(st_p)
    sel = json.loads(sel_p.read_text())
    names = {"C0": "C0", "C1": "C1", "C2 (gamma=0.1)": "C2"}
    for protocol in ("known_site", "unseen_site"):
        for method, short in names.items():
            g = st[(st.protocol == protocol) & (st.method == method)]
            inband = int(g.coverage.between(0.75, 0.85).sum())
            s.add(f"cal.{protocol}.{short}.coverage", g.coverage.mean(),
                  f"{pct(g.coverage.mean(), 1)} (years {pct(g.coverage.min())}-{pct(g.coverage.max())}; "
                  f"{inband} of {len(g)} in 75-85%)", f"80%-interval coverage, {short}, {protocol.replace('_', ' ')}", st_p)
            s.add(f"cal.{protocol}.{short}.coverage_outage", g.coverage_outage.mean(), pct(g.coverage_outage.mean(), 1),
                  f"mean coverage during Kubala outages (2019-2022), {short}, {protocol.replace('_', ' ')}", st_p)
            s.add(f"cal.{protocol}.{short}.interval_score", g.interval_score.mean(), f"{g.interval_score.mean():.3f}",
                  f"mean interval score (lower is better), {short}, {protocol.replace('_', ' ')}", st_p)
        s.add(f"cal.{protocol}.selected", sel[protocol]["method"], names[sel[protocol]["method"]],
              f"interval method chosen by the pre-declared rule, {protocol.replace('_', ' ')}", sel_p)
    o = st[(st.year == 2020) & (st.protocol == "known_site")].set_index("method").coverage_outage
    s.add("cal.known_site.outage_2020", [o["C0"], o["C2 (gamma=0.1)"]],
          f"{pct(o['C0'])} -> {pct(o['C2 (gamma=0.1)'])}", "coverage during the 2020 Kubala outage, C0 -> C2 (known sites)",
          st_p)
    # Staleness test: one model trained through 2019 predicts 2020-2022 without retraining, vs. yearly retraining.
    hz_p = ev / "backtest" / "horizon_2019_2022.parquet"
    hz = pd.read_parquet(hz_p)
    gaps = {}
    for protocol, code in (("known_site", "E6"), ("unseen_site", "E7"), ("known_site", "E7")):
        for y in (2020, 2021, 2022):
            a_ = hz[(hz.protocol == protocol) & (hz.model == code) & (hz.sampled_at.dt.year == y)]
            b_ = dev[(dev.protocol == protocol) & (dev.model == code) & (dev.year == y)]
            cov_ = lambda g: float(((g.y >= g.q10) & (g.y <= g.q90)).mean())  # noqa: E731
            gaps[(protocol, code, y)] = (cov_(b_) - cov_(a_), float(np.sqrt(np.mean((a_.pred - a_.y) ** 2))
                                                                    - np.sqrt(np.mean((b_.pred - b_.y) ** 2))))
    prim = [v for k, v in gaps.items() if k[:2] != ("known_site", "E7")]
    s.add("cal.horizon_max_coverage_gap", max(abs(c) for c, _ in prim), f"{max(abs(c) for c, _ in prim):.3f}",
          "largest coverage difference, yearly retraining vs. none, 1-3 years after training (E6 known, E7 unseen; "
          f"E7 at known sites: {max(abs(gaps[('known_site', 'E7', y)][0]) for y in (2020, 2021, 2022)):.3f})",
          [hz_p, ev / "backtest" / "dev.parquet"])
    s.add("cal.horizon_max_rmse_gap", max(abs(r) for _, r in prim), f"{max(abs(r) for _, r in prim):.3f} mg/L",
          "largest RMSE difference, no retraining vs. yearly retraining (E6 known, E7 unseen)",
          [hz_p, ev / "backtest" / "dev.parquet"])
    c0_2023 = st[(st.protocol == "known_site") & (st.method == "C0") & (st.year == 2023)].coverage.iloc[0]
    s.check("C0 coverage 2023 (study) = Step 3 validation E6 coverage",
            abs(c0_2023 - s3v.loc[("known_site", "all", "E6")].coverage80) < 1e-9)
    s.check("fig9 CSV matches the calibration study",
            np.allclose(pd.read_csv(FIG / "fig9_interval_calibration.csv").coverage.to_numpy(),
                        st[st.method.isin(names)].coverage.to_numpy()))

    # -- prospective test -------------------------------------------------------------------------------------------
    s.section = "Prospective test (fresh test set)"
    reg_p = REPORTS / "prospective_registration.json"
    reg = json.loads(reg_p.read_text())
    s.add("prosp.cutoff", reg["cutoff"], reg["cutoff"], "frozen models trained on all samples up to this date", reg_p)
    s.add("prosp.training_rows", reg["training_rows"], f"{reg['training_rows']:,}", "training samples of the frozen models",
          reg_p)
    s.add("prosp.when", reg["when"], reg["when"], "when the single prospective evaluation runs", reg_p)
    s.add("prosp.hypotheses", len(reg["hypotheses"]), f"{len(reg['hypotheses'])}", "pre-registered hypotheses (H1-H4)",
          reg_p)
    s.add("prosp.fingerprints", reg["model_fingerprints"], ", ".join(f"{k} {v[:12]}" for k, v in
                                                                     reg["model_fingerprints"].items()),
          "fingerprints of the frozen models (first 12 hex digits)", reg_p)
    log_p = ev / "test_access_log.jsonl"
    log = [json.loads(line) for line in log_p.read_text().splitlines() if line.strip()]
    s.add("prosp.test_uses", sum(e.get("split") == "test" for e in log), f"{sum(e.get('split') == 'test' for e in log)}",
          "logged uses of the 2024-26 test period (Step 2, Step 3, prospective freeze)", log_p)
    s.check("registration cutoff covers every modeling sample", reg["training_rows"] == len(df))
    return s


def write(s: Sheet):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "key_numbers.json").write_text(json.dumps({"numbers": s.rows, "checks": s.checks}, indent=2, default=str))
    lines = ["# Key numbers", "",
             "Generated by `uv run python -m chlorine.reporting.numbers` from the result files; do not edit by hand.",
             "Slides and documents quote the **Display** column; cite the ID in speaker notes.", ""]
    for section in dict.fromkeys(r["section"] for r in s.rows):
        lines += [f"## {section}", "", "| ID | Display | What it is | Source |", "|---|---|---|---|"]
        for r in (r for r in s.rows if r["section"] == section):
            lines.append(f"| `{r['id']}` | {r['display']} | {r['label']} | `{r['source']}` |")
        lines.append("")
    lines += ["## Consistency checks", "", "| Check | Result |", "|---|---|"]
    lines += [f"| {c['check']} | {'PASS' if c['ok'] else 'FAIL'} {c['detail']} |" for c in s.checks]
    (REPORTS / "key_numbers.md").write_text("\n".join(lines) + "\n")


def main():
    s = build()
    write(s)
    failed = [c["check"] for c in s.checks if not c["ok"]]
    print(f"{len(s.rows)} numbers, {len(s.checks)} checks -> {REPORTS / 'key_numbers.md'}")
    if failed:
        raise SystemExit(f"FAILED consistency checks: {failed}")


if __name__ == "__main__":
    main()
