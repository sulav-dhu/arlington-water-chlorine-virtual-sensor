"""Step 3 presentation figures (PNG + CSV twin each, aggregates only).

    uv run python -m chlorine.modeling.figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import EvalConfig  # noqa: E402
from ..evaluation.figures import AXIS, INK2, MUTED, OUT, SLOT, SURFACE, _dot, _setup, _style, _title  # noqa: E402

PROGRESSION = {
    "known_site": [("B0", "Overall mean"), ("B1", "City-wide weekly mean"), ("B5", "Best Step 2 reference (B5)"),
                   ("B6", "Regression kriging (B6)"), ("G6", "Gradient boosting (G6)"), ("E6", "Average of both (E6), selected")],
    "unseen_site": [("B0", "Overall mean"), ("B1", "City-wide weekly mean"), ("B4", "Best Step 2 reference (B4)"),
                    ("B7", "Regression kriging, public (B7)"), ("G7", "Gradient boosting, public (G7)"),
                    ("E7", "Average of both (E7), selected")],
}
ABLATIONS = [("A2", "Same-week anomaly interpolation"), ("A5", "Place's own recent anomalies"),
             ("A6", "Recency weighting"), ("A1", "Static site-level interpolation"),
             ("A4", "Plant-regime terms"), ("A3", "Pressure plane in the covariance")]


def progression(cfg: EvalConfig, floor: float):
    rows = []
    for split, label in (("val", "validation 2023"), ("test", "test 2024-26")):
        res = pd.read_csv(cfg.out_root / "step3" / split / "results.csv")
        comp = pd.read_csv(cfg.out_root / "step3" / split / "comparisons.csv")
        for protocol, models in PROGRESSION.items():
            r = res[(res.protocol == protocol) & res.subset.eq("all")].set_index("model")
            c = comp[comp.protocol == protocol].set_index("model")
            for code, name in models:
                rows.append({"protocol": protocol, "period": label, "model": code, "label": name,
                             "rmse": r.loc[code, "rmse"], "rmse_lo": c.loc[code, "rmse_lo"], "rmse_hi": c.loc[code, "rmse_hi"]})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.74, bottom=0.12, wspace=0.62)
    for ax, (protocol, models), title in zip(axes, PROGRESSION.items(), ["Known sites", "Unseen sites (public data only)"]):
        _style(ax)
        ax.axvline(floor, color=MUTED, linewidth=1)
        ypos = {code: len(models) - 1 - i for i, (code, _) in enumerate(models)}
        for j, (period, color) in enumerate(zip(["validation 2023", "test 2024-26"], SLOT)):
            for x in t[(t.protocol == protocol) & (t.period == period)].itertuples():
                y = ypos[x.model] + (0.15 if j == 0 else -0.15)
                ax.plot([x.rmse_lo, x.rmse_hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
                _dot(ax, x.rmse, y, color)
        ax.set_yticks(list(ypos.values()), [n for _, n in models])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("RMSE (mg/L), lower is better")
        ax.set_xlim(0.25, 0.52)
        ax.set_ylim(-0.95, len(models) - 0.45)
    axes[0].text(floor + 0.003, -0.9, f"noise floor {floor:.2f}", color=INK2, fontsize=8.5, va="bottom")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", mec=SURFACE, mew=2, ms=7) for c in SLOT[:2]]
    fig.legend(handles, ["Validation 2023", "Test 2024-26"], loc="upper right", bbox_to_anchor=(0.98, 0.86), ncol=2,
               frameon=False, labelcolor=INK2)
    _title(fig, "Selected models: 10-12% lower error at known sites, 3-4% at unseen sites",
           "Dot = RMSE, line = 95% cluster-bootstrap interval. Models were selected on validation; test was evaluated once.")
    fig.savefig(OUT / "fig4_model_progression.png", dpi=200)
    t.to_csv(OUT / "fig4_model_progression.csv", index=False)
    plt.close(fig)


def ablations(cfg: EvalConfig):
    from .run import ablation_comparisons
    preds = pd.read_parquet(cfg.out_root / "step3" / "val" / "predictions.parquet")
    res = pd.read_csv(cfg.out_root / "step3" / "val" / "results.csv")
    abl = ablation_comparisons(preds, cfg)
    r = res[res.subset.eq("all")].set_index(["protocol", "model"]).rmse
    abl["delta"] = [r[(p, m)] - r[(p, par)] for p, m, par in zip(abl.protocol, abl.model, abl.parent)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.29, right=0.97, top=0.72, bottom=0.13)
    _style(ax)
    ax.axvline(0, color=AXIS, linewidth=1)
    protocols = [("known_site", "Known sites"), ("unseen_site", "Unseen sites"), ("outage", "Kubala outages")]
    for i, (code, _) in enumerate(ABLATIONS):
        y = len(ABLATIONS) - 1 - i
        for j, ((protocol, _), color) in enumerate(zip(protocols, SLOT)):
            x = abl[(abl.model == code) & (abl.protocol == protocol)]
            if x.empty:
                continue
            x = x.iloc[0]
            yy = y + (1 - j) * 0.22
            ax.plot([x.d_rmse_lo, x.d_rmse_hi], [yy, yy], color=color, linewidth=2, solid_capstyle="round")
            _dot(ax, x.delta, yy, color, 6.5)
    ax.set_yticks(range(len(ABLATIONS)), [n for _, n in ABLATIONS][::-1])
    ax.set_xlabel("Increase in RMSE when the component is removed (mg/L)")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", mec=SURFACE, mew=2, ms=7) for c in SLOT]
    fig.legend(handles, [n for _, n in protocols], loc="upper left", bbox_to_anchor=(0.29, 0.86), ncol=3, frameon=False,
               labelcolor=INK2)
    _title(fig, "Interpolating this week's neighbors is what matters",
           "Validation 2023 (outages: pre-2024, leave-one-out). Dot = change in RMSE, line = 95% interval; right of 0 = the "
           "component helps.")
    fig.savefig(OUT / "fig5_ablations.png", dpi=200)
    abl.to_csv(OUT / "fig5_ablations.csv", index=False)
    plt.close(fig)


def main():
    cfg = EvalConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    _setup()
    floor = json.loads((cfg.out_root / "val" / "run.json").read_text())["noise_floor"]["within_week_sd"]
    progression(cfg, floor)
    ablations(cfg)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
