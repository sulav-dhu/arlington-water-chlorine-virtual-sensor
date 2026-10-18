"""Key-numbers sheet: every number the slides quote, computed from the result files, each with its source.

    uv run python -m chlorine.reporting.figures     # first: some numbers are read from the figure CSVs
    uv run python -m chlorine.reporting.numbers     # -> reports/key_numbers.md + reports/key_numbers.json

Each number has a stable ID (e.g. `clean.samples`), a display string, the raw value and the file it came from.
This version covers data cleaning, exploration and the baseline models.
Slides and documents cite the display string of an ID, so text and figures cannot drift apart. Consistency
checks assert that numbers available from two sources agree (e.g. the cleaning funnel ends exactly at the
modeling table); the script fails if any check fails.
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
