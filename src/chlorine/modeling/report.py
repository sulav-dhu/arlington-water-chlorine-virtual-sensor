"""Markdown report for the Step 3 experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

ABLATION_PARENT = {"A1": "B7", "A2": "B7", "A3": "B7", "A4": "B7", "A5": "B6", "A6": "B6"}
SHOW_REFS = ("B1", "B4", "B4p", "B5", "B5p", "B5r")
PROTOCOLS = [("known_site", "Known sites"), ("unseen_site", "Unseen sites (5-fold by place)"),
             ("outage", "Kubala outages (leave-one-outage-out, pre-test)")]


def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def model_table(res, comp, protocol, split, codes) -> str:
    r = res[(res.protocol == protocol) & (res.split == split) & res.subset.eq("all")].set_index("model")
    c = comp[(comp.protocol == protocol) & (comp.split == split)].set_index("model")
    ref = c.comparator.iloc[0]
    lines = [f"| Code | Model | RMSE [95% CI] | Δ vs {ref} [95% CI] | P(better than {ref}) | R² | Pinball | "
             "Cover80 | Width80 | AUC <2.5 | Site ρ |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for code in [m for m in codes if m in r.index]:
        x, k = r.loc[code], c.loc[code]
        d = "" if code == ref else f"{x.rmse - r.loc[ref, 'rmse']:+.3f} [{k.d_rmse_lo:+.3f}, {k.d_rmse_hi:+.3f}]"
        pb = "" if code == ref else _f(k.p_better, 2)
        bold = "**" if code in ("B6", "B6r", "G6", "E6", "B7", "B7r", "G7", "E7") else ""
        lines.append(f"| {bold}{code}{bold} | {x.model_name} | {_f(x.rmse)} [{_f(k.rmse_lo)}, {_f(k.rmse_hi)}] | {d} | "
                     f"{pb} | {_f(x.r2, 2)} | {_f(x.pinball)} | {_f(x.coverage80, 2)} | {_f(x.width80, 2)} | "
                     f"{_f(x.auc_lt25, 2)} | {_f(x.get('site_spearman', np.nan), 2)} |")
    return "\n".join(lines) + "\n"


def ablation_table(abl: pd.DataFrame, res, split) -> str:
    if abl.empty:
        return "_none_\n"
    lines = ["| Protocol | Ablation | What is removed | Parent RMSE | Ablation RMSE | Δ [95% CI] | P(ablation worse) |",
             "|---|---|---|---|---|---|---|"]
    for x in abl.itertuples():
        r = res[(res.protocol == x.protocol) & res.subset.eq("all")].set_index("model")
        lines.append(f"| {x.protocol} | {x.model} | {r.loc[x.model, 'model_name']} | {_f(r.loc[x.parent, 'rmse'])} | "
                     f"{_f(r.loc[x.model, 'rmse'])} | {r.loc[x.model, 'rmse'] - r.loc[x.parent, 'rmse']:+.3f} "
                     f"[{x.d_rmse_lo:+.3f}, {x.d_rmse_hi:+.3f}] | {1 - x.p_better:.2f} |")
    return "\n".join(lines) + "\n"


def subset_table(res, protocol, split, codes) -> str:
    r = res[(res.protocol == protocol) & (res.split == split) & ~res.subset.eq("all") & res.model.isin(codes)]
    if r.empty:
        return "_none_\n"
    t = r.pivot_table(index="subset", columns="model", values="rmse", sort=False)[[c for c in codes if c in set(r.model)]]
    n = r.groupby("subset", sort=False).n.first()
    lines = ["| Subset | n | " + " | ".join(t.columns) + " |", "|---|---|" + "---|" * len(t.columns)]
    for sub, row in t.iterrows():
        best = row.min()
        lines.append(f"| {sub} | {int(n[sub]):,} | " + " | ".join(f"**{v:.3f}**" if np.isclose(v, best) else f"{v:.3f}"
                                                                  for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_report(path, cfg, split, res, comp, tuning, params, selection, floor, h, abl=None, robustness=None):
    abl = pd.DataFrame() if abl is None else abl
    L = []
    w = L.append
    w(f"# Step 3 modeling report — {split}\n\n")
    w(("Validation: models trained on 2015-2022, evaluated on 2023 (outages: leave-one-out over pre-2024 Kubala "
       "outages). " if split == "val" else
       "**Test (2024 – Aug 2026), evaluated once with the selection fixed on validation**; models refit on 2015-2023. ")
      + f"Recency half-life of the regression-kriging models: {h:g} days (chosen on "
      + ("the inner validation year 2022" if split == "val" else "validation 2023") + ").\n\n")

    w("## Headline\n\n")
    for protocol, label in PROTOCOLS:
        if protocol not in selection or not (res.protocol == protocol).any():   # no outage protocol on test
            continue
        sel = selection[protocol]
        r = res[(res.protocol == protocol) & res.subset.eq("all")].set_index("model")
        c = comp[comp.protocol == protocol].set_index("model")
        ref = c.comparator.iloc[0]
        for tag, code in (("selected", sel["selected"]), ("best public-only", sel["best_public_only"])):
            if tag == "best public-only" and code == sel["selected"]:
                continue
            x, k = r.loc[code], c.loc[code]
            w(f"- **{label}, {tag}: {code}** ({x.model_name}) RMSE {x.rmse:.3f} vs. {ref} {r.loc[ref, 'rmse']:.3f}: "
              f"Δ {x.rmse - r.loc[ref, 'rmse']:+.3f} [{k.d_rmse_lo:+.3f}, {k.d_rmse_hi:+.3f}], "
              f"P(better) {k.p_better:.2f}; R² {x.r2:.2f}; 80% interval coverage {x.coverage80:.2f} "
              f"(width {x.width80:.2f}); AUC <2.5 mg/L {x.auc_lt25:.2f}.\n")
    hf = res[(res.protocol == "known_site") & res.subset.eq("high_frequency_sites")]
    if not hf.empty:
        best = hf.sort_values("rmse").iloc[0]
        w(f"- **Noise floor:** at the near-daily sites the best model ({best.model}) reaches RMSE {best.rmse:.3f} against "
          f"a within-week floor of {floor['within_week_sd']:.2f} mg/L.\n")
    w("\n")

    w("## Tuning (recency half-life, known-site RMSE)\n\n")
    t = tuning.pivot_table(index="half_life_days", columns="model", values="rmse")
    w("| Half-life (days) | " + " | ".join(t.columns) + " |\n|---|" + "---|" * len(t.columns) + "\n")
    for hl, row in t.iterrows():
        w(f"| {hl:g} | " + " | ".join(f"{v:.4f}" for v in row) + " |\n")
    w("\n")

    codes = list(SHOW_REFS) + ["B6", "B6r", "G6", "E6", "B7", "B7r", "G7", "E7", "A1", "A2", "A3", "A4", "A5", "A6"]
    for protocol, label in PROTOCOLS:
        sp = "pre_test" if protocol == "outage" else split
        if not ((res.protocol == protocol) & (res.split == sp)).any():
            continue
        w(f"## {label}\n\n")
        w(model_table(res, comp, protocol, sp, codes))
        w("\nRMSE by subset:\n\n")
        w(subset_table(res, protocol, sp, [c for c in ("B4", "B5", "B5p", "B6", "G6", "E6", "B7", "G7", "E7")]))
        w("\n")

    if robustness is not None and not robustness.empty:
        w("## Second validation year (robustness)\n\n")
        w("Same models trained on 2015-2021 and evaluated on 2022 (known sites; unseen sites). A ranking that holds in "
          "both years is less likely to be noise.\n\n")
        r23 = res[res.subset.eq("all") & res.protocol.isin(["known_site", "unseen_site"])][["protocol", "model", "rmse"]]
        t = robustness.merge(r23, on=["protocol", "model"], how="left")
        w("| Protocol | Model | RMSE 2022 | RMSE 2023 |\n|---|---|---|---|\n")
        for x in t.sort_values(["protocol", "rmse_2022"]).itertuples():
            w(f"| {x.protocol} | {x.model} | {x.rmse_2022:.3f} | {_f(x.rmse)} |\n")
        w("\n")

    if not abl.empty:
        w("## Ablations (what each component is worth)\n\n")
        w(ablation_table(abl, res, split))
        w("\n")

    w("## Fitted structure (known-site fit)\n\n")
    if "B6" in params:
        p = params["B6"]
        w(f"- Weekly anomalies: shared SD {p['anomaly_sd']:.3f} mg/L, correlation range {p['anomaly_range_km']:.1f} km, "
          f"cross-plane factor {p['anomaly_cross_plane']:.2f} (places in different pressure planes co-vary "
          f"{p['anomaly_cross_plane']:.0%} as much as places in the same plane); unshared place-week SD {p['tau_a']:.3f}; "
          f"within-week scatter SD {p['sigma_w']:.3f}.\n")
        if "static_sd" in p:
            w(f"- Static site levels: shared SD {p['static_sd']:.3f}, range {p['static_range_km']:.1f} km, cross-plane "
              f"factor {p['static_cross_plane']:.2f}.\n")
        w(f"- Weight on the place's own recent anomalies: {p['alpha_own']:.2f}.\n")
    w("\n## Notes\n\n")
    for note in ["Comparators are the best reference models on validation in Step 2 (B5 known, B4 unseen, B5p outages).",
                 "P(better) = share of cluster-bootstrap draws (over places) in which the model has lower RMSE.",
                 "B7/B7r/G7 never use a place's own samples; in the known-site protocol they predict each place as if it "
                 "were unmonitored.",
                 "Region code (B6r, B7r, G6) is utility information; B6, B7 and G7 use public data plus, for B6, the "
                 "place's own history."]:
        w(f"- {note}\n")
    path.write_text("".join(L))
