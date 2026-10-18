"""Presentation figures from the evaluation outputs.

    uv run python -m chlorine.evaluation.figures

Reads data/eval/{val,test}/ and writes static PNGs to reports/figures/, each with a CSV twin holding
exactly the plotted numbers (the table view). Figures contain aggregate metrics only.
Style: light chart surface, categorical slots 1-3 (validated with the dataviz palette checker),
thin marks, solid hairline grid, legend for 2+ series, selective direct labels.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import ROOT, EvalConfig  # noqa: E402
from .metrics import bootstrap_rmse  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
OUT = ROOT / "reports" / "figures"

LABELS = {"B0": "Overall mean", "B1": "City-wide weekly mean", "B2": "+ region offset",
          "B2p": "+ pressure-plane offset", "B3": "+ site offset", "B4": "Same-week region neighbors",
          "B4p": "Same-week plane neighbors (distance)", "B5": "Region neighbors + site offset",
          "B5p": "Plane neighbors + site offset"}


def _setup():
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _style(ax, grid_axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _dot(ax, x, y, color, size=7):
    ax.plot(x, y, "o", ms=size, color=color, mec=SURFACE, mew=2, zorder=3)


def _title(fig, title, subtitle):
    fig.text(0.012, 0.975, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=10, color=INK2, va="top")


def ladder(val: pd.DataFrame, test: pd.DataFrame, floor: float):
    codes = ["B0", "B1", "B2", "B2p", "B3", "B4", "B4p", "B5", "B5p"]
    rows = []
    for split, res in (("validation 2023", val), ("test 2024-26", test)):
        r = res[res.subset.eq("all") & res.model.isin(codes) & res.protocol.isin(["known_site", "unseen_site"])]
        rows.append(r.assign(period=split)[["protocol", "period", "model", "rmse", "rmse_lo", "rmse_hi"]])
    t = pd.concat(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.31, right=0.98, top=0.76, bottom=0.12, wspace=0.08)
    ypos = {c: len(codes) - 1 - i for i, c in enumerate(codes)}
    for ax, protocol, name in zip(axes, ["known_site", "unseen_site"], ["Known sites", "Unseen sites (5-fold by place)"]):
        _style(ax)
        ax.axvline(floor, color=MUTED, linewidth=1)
        for j, (period, color) in enumerate(zip(["validation 2023", "test 2024-26"], SLOT)):
            d = t[(t.protocol == protocol) & (t.period == period)]
            for x in d.itertuples():
                y = ypos[x.model] + (0.17 if j == 0 else -0.17)
                ax.plot([x.rmse_lo, x.rmse_hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
                _dot(ax, x.rmse, y, color)
        missing = [c for c in codes if c not in set(t[t.protocol == protocol].model)]
        for c in missing:
            ax.text(0.37, ypos[c], "n/a: needs site history", color=MUTED, fontsize=8.5, va="center", ha="center")
        ax.set_title(name, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("RMSE (mg/L), lower is better")
    axes[0].set_yticks(list(ypos.values()), [f"{c}  {LABELS[c]}" for c in ypos])
    axes[0].set_ylim(-0.7, len(codes) - 0.3)
    axes[0].set_xlim(0.26, 0.52)
    axes[1].text(floor + 0.003, len(codes) - 0.55, f"noise floor {floor:.2f}\n(within-week scatter)", color=INK2,
                 fontsize=8.5, va="top")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", mec=SURFACE, mew=2, ms=7) for c in SLOT[:2]]
    fig.legend(handles, ["Validation 2023", "Test 2024-26"], loc="upper right", bbox_to_anchor=(0.98, 0.9), ncol=2,
               frameon=False, labelcolor=INK2)
    _title(fig, "Same-week neighbors carry most of the signal",
           "Dot = RMSE, line = 95% cluster-bootstrap interval. Vertical line: within-week scatter at the near-daily "
           "sites,\na floor for any model working at weekly resolution.")
    fig.savefig(OUT / "fig1_baseline_ladder.png", dpi=200)
    t.to_csv(OUT / "fig1_baseline_ladder.csv", index=False)
    plt.close(fig)


def learning_curve(curve: pd.DataFrame):
    names = {"B2": "Region offset (B2)", "B4": "Region neighbors (B4)", "B4p": "Plane neighbors, distance (B4p)"}
    t = curve.groupby(["model", "fraction"]).agg(monitored_places=("monitored_places", "mean"), rmse=("rmse", "mean"),
                                                 rmse_sd=("rmse", "std")).reset_index().fillna({"rmse_sd": 0})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.subplots_adjust(left=0.1, right=0.7, top=0.8, bottom=0.13)
    _style(ax, "y")
    for color, code in zip(SLOT, ["B2", "B4", "B4p"]):
        d = t[t.model == code].sort_values("monitored_places")
        ax.fill_between(d.monitored_places, d.rmse - d.rmse_sd, d.rmse + d.rmse_sd, color=color, alpha=0.1, linewidth=0)
        ax.plot(d.monitored_places, d.rmse, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        for x, y in zip(d.monitored_places, d.rmse):
            _dot(ax, x, y, color, 6.5)
        last = d.iloc[-1]
        ax.text(last.monitored_places + 5, last.rmse, f"{names[code]}  {last.rmse:.3f}", color=INK2, fontsize=9,
                va="center")
    ax.set_xlim(25, 150)
    ax.set_xlabel("Monitored places (training period)")
    ax.set_ylabel("RMSE at unmonitored places (mg/L)")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", mec=SURFACE, mew=2, ms=7) for c in SLOT]
    fig.legend(handles, list(names.values()), loc="upper left", bbox_to_anchor=(0.1, 0.89), ncol=3, frameon=False,
               labelcolor=INK2, fontsize=8.5)
    _title(fig, "More sampling sites help; distance-weighted models gain most from density",
           "Validation 2023, unseen-site protocol; band = ±1 SD over 5 random subsets of monitored places.")
    fig.savefig(OUT / "fig2_learning_curve.png", dpi=200)
    t.to_csv(OUT / "fig2_learning_curve.csv", index=False)
    plt.close(fig)


def plane_vs_region(val_preds: pd.DataFrame, test_preds: pd.DataFrame, cfg: EvalConfig):
    """Paired comparison B4p (public plane + distance) vs B4 (utility region code), known-site protocol,
    with a cluster-bootstrap 95% interval for the RMSE difference in every row."""
    spec = [(val_preds, None, "2023, calm year (validation)"), (test_preds, None, "2024-26, all samples (test)"),
            (test_preds, "jk_offline", "2024-26, during Kubala outages"), (test_preds, "new_site", "2024-26, new sites"),
            (test_preds, "west", "2024-26, West pressure plane")]
    rows = []
    for preds, subset, label in spec:
        q = preds[(preds.protocol == "known_site") & preds.model.isin(["B4", "B4p"])]
        if subset == "west":
            q = q[q.pressure_plane == "WEST"]
        elif subset:
            q = q[q[subset].astype(bool)]
        b = bootstrap_rmse(q, "B4", cfg.bootstrap_reps, cfg.seed).set_index("model").loc["B4p"]
        rmse = q.assign(se=(q.pred - q.y) ** 2).groupby("model").se.mean() ** 0.5
        rows.append({"row": label, "n": int((q.model == "B4").sum()), "places": q.place_group.nunique(),
                     "region_B4": rmse["B4"], "plane_B4p": rmse["B4p"], "difference": rmse["B4p"] - rmse["B4"],
                     "ci_lo": b.d_rmse_lo, "ci_hi": b.d_rmse_hi, "p_plane_better": b.p_better})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    fig.subplots_adjust(left=0.25, right=0.76, top=0.72, bottom=0.13)
    _style(ax)
    for i, x in enumerate(t.itertuples()):
        y = len(t) - 1 - i
        ax.plot([x.region_B4, x.plane_B4p], [y, y], color=AXIS, linewidth=2, zorder=1)
        _dot(ax, x.region_B4, y, SLOT[1])
        _dot(ax, x.plane_B4p, y, SLOT[0])
        clear = x.ci_hi < 0
        hi = "<0" if -0.0005 < x.ci_hi < 0 else f"{x.ci_hi:+.3f}"      # e.g. -0.00004 would print as "-0.000"
        ax.text(1.03, y, f"{x.difference:+.3f}  [{x.ci_lo:+.3f}, {hi}]", transform=ax.get_yaxis_transform(),
                color=INK if clear else INK2, fontsize=9.5, va="center", fontweight="bold" if clear else "normal")
    ax.text(1.03, len(t) - 0.4, "plane − region  [95% CI]", transform=ax.get_yaxis_transform(), color=MUTED, fontsize=8.5)
    ax.set_yticks(range(len(t)), [f"{r}\n{n:,} samples, {p} places" for r, n, p in zip(t.row[::-1], t.n[::-1], t.places[::-1])],
                  linespacing=1.5)
    ax.set_xlim(0.29, 0.455)
    ax.set_xlabel("RMSE (mg/L), lower is better (known-site protocol)")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", mec=SURFACE, mew=2, ms=7) for c in SLOT[:2]]
    fig.legend(handles, ["Public pressure plane + distance (B4p)", "Utility region code (B4)"], loc="upper left",
               bbox_to_anchor=(0.004, 0.85), ncol=2, frameon=False, labelcolor=INK2)
    _title(fig, "Public pressure-plane geography matches the utility's region codes",
           "No difference in calm 2023; ahead overall in 2024-26 (borderline: the 95% interval just excludes zero). Larger "
           "gaps during\noutages, at new sites and in the West plane are suggestive only. Bold: interval excludes zero "
           "(cluster bootstrap over places).")
    fig.savefig(OUT / "fig3_plane_vs_region.png", dpi=200)
    t.to_csv(OUT / "fig3_plane_vs_region.csv", index=False)
    plt.close(fig)


def main():
    cfg = EvalConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    _setup()
    val = pd.read_csv(cfg.out_root / "val" / "results.csv")
    test = pd.read_csv(cfg.out_root / "test" / "results.csv")
    floor = json.loads((cfg.out_root / "val" / "run.json").read_text())["noise_floor"]["within_week_sd"]
    ladder(val, test, floor)
    curve_path = cfg.out_root / "val" / "learning_curve.csv"
    if curve_path.exists():
        learning_curve(pd.read_csv(curve_path))
    plane_vs_region(pd.read_parquet(cfg.out_root / "val" / "predictions.parquet"),
                    pd.read_parquet(cfg.out_root / "test" / "predictions.parquet"), cfg)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
