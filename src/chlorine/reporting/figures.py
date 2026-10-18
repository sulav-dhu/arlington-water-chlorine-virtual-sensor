"""Presentation figures for the data and baseline story (PNG + CSV twin each; aggregates only).

    uv run python -m chlorine.reporting.figures

fig6  cleaning funnel: from the raw export to the modeling table, and what cleaning fixed
fig7  Kubala outages: city-wide residual over time, and the drop at each site by pressure plane
fig8  where the variation in chlorine comes from (variance decomposition, noise floor)
fig11 sampling coverage: places sampled per week and per year
fig12 distribution of chlorine with the regulatory minimum, screening threshold and yearly-average maximum
fig13 season vs. plant outages
fig14 evaluation design (time split, hidden places, no-peeking rules)
fig15 where the best public-data baselines struggle (outages, new sites)
fig16 project roadmap: completed vs. proposed steps
"""
from __future__ import annotations

import json
import textwrap

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import CleaningConfig, EvalConfig  # noqa: E402
from ..evaluation.data import load_frame  # noqa: E402
from ..evaluation.figures import AXIS, INK, INK2, MUTED, OUT, SLOT, SURFACE, _dot, _setup, _style, _title  # noqa: E402
from ..evaluation.report import noise_floor  # noqa: E402

SLOT4 = SLOT + ["#eda100"]          # categorical slots 1-4, validated for adjacent use (stacked bars)
BAND = "#898781"                    # neutral wash for shaded periods (not a data series)


def phone_rows() -> int:
    """Raw-export rows whose free text holds a phone number (counted locally; the text itself is never written).
    Also counts numbers written as ten bare digits, which the cleaning guard's pattern does not match."""
    import re
    from ..cleaning.addresses import contains_phone
    from ..cleaning.load import load_raw
    raw = load_raw(CleaningConfig().raw_xlsx)
    bare = re.compile(r"(?<!\d)\d{10}(?!\d)")
    text = raw.select_dtypes(include=["object", "string"]).astype("string").fillna("")
    hit = text.apply(lambda col: col.map(contains_phone).astype(bool) | col.str.contains(bare)).any(axis=1)
    return int(hit.sum())


def funnel_numbers() -> dict:
    d = CleaningConfig().out_dir
    meas = pd.read_parquet(d / "measurements.parquet")
    chl = pd.read_parquet(d / "chlorine_v1.parquet")
    cl = meas[(meas.analyte == "Chlorine residual") & (meas.analysis == "Chlres-field")]
    reasons = cl.exclusion_reason.value_counts()
    steps = [("All rows in the lab export (20 tests)", len(meas), None),
             ("Total-chlorine results", len(cl), "other tests (pH, ammonia, nitrite, ...)")]
    remaining = len(cl)
    for reason, label, why in [
            ("project_out_of_scope", "In routine sampling programs", "construction, special projects, NAP, plant and tank programs"),
            ("not_a_routine_location", "At distribution sampling sites", "plant taps, ad-hoc addresses"),
            ("location_unresolved_or_conflict", "Location confirmed", "site code and address disagree"),
            ("result_not_reportable", "Reportable result", "'Not Analyzed', 'Invalid', 'Lab Error'"),
            ("duplicate_copy", "One copy per sample", "same sample logged twice")]:
        remaining -= int(reasons.get(reason, 0))
        steps.append((label, remaining, why))
    assert remaining == len(chl), "funnel does not reproduce the modeling table"
    loc = pd.read_csv(d / "locations.csv")
    rev = pd.read_csv(d / "location_review.csv")
    used = loc[loc.n_chlorine_in_scope > 0]
    fixes = {
        "samples linked to their site by address instead of code": int((chl.loc_method == "address_named_link").sum()),
        "site codes with several address spellings, reconciled": int(
            rev.groupby("site_code").spellings.sum().gt(1).sum()),
        "renumbered or moved sites grouped as one place": int(
            (loc[loc.location_type == "routine"].groupby("place_group").size() > 1).sum()),
        "duplicate copies removed (all tests)": int(meas.is_duplicate.sum()),
        "rows with a customer phone number, never written out": phone_rows(),
        "share of samples geocoded locally": float(used[used.lat.notna()].n_chlorine_in_scope.sum()
                                                   / used.n_chlorine_in_scope.sum()),
    }
    return {"steps": steps, "fixes": fixes, "places": int(chl.place_group.nunique()),
            "locations": int(chl.location_id.nunique())}


def funnel():
    nb = funnel_numbers()
    steps = nb["steps"]
    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_axes([0.27, 0.12, 0.42, 0.66])
    _style(ax)
    ys = np.arange(len(steps))[::-1]
    for y, (_label, n, why), prev in zip(ys, steps, [None] + [s[1] for s in steps[:-1]]):
        ax.barh(y, n, height=0.5, color=SLOT[0], zorder=2)
        removed = "" if prev is None else f"   −{prev - n:,}"
        ax.text(n + 600, y, f"{n:,}", color=INK, fontsize=10, va="center", fontweight="bold")
        if prev is not None and why:
            ax.text(n + 600, y - 0.32, f"{removed.strip()}  {why}", color=MUTED, fontsize=8, va="center")
    ax.set_yticks(ys, [s[0] for s in steps])
    ax.set_xlim(0, 75000)
    ax.set_xticks([0, 20000, 40000, 60000], ["0", "20,000", "40,000", "60,000"])
    ax.set_xlabel("Rows")
    # What cleaning fixed rather than removed.
    fx = fig.add_axes([0.74, 0.12, 0.25, 0.66])
    fx.axis("off")
    fx.text(0, 1.0, "Fixed, not removed", fontsize=11, fontweight="bold", color=INK, va="top")
    lines = []
    for k, v in nb["fixes"].items():
        lines.append((f"{v:.1%}" if isinstance(v, float) else f"{v:,}", k))
    for i, (v, k) in enumerate(lines):
        fx.text(0, 0.88 - i * 0.15, v, fontsize=13, fontweight="bold", color=INK, va="top")
        fx.text(0, 0.88 - i * 0.15 - 0.065, k, fontsize=8.5, color=INK2, va="top")
    _title(fig, f"Cleaning: 37,250 chlorine results → {steps[-1][1]:,} modeling samples at {nb['places']} places",
           "Most rows leave by scope (other sampling programs), not by error; the data problems were fixed rather than "
           "dropped.")
    fig.savefig(OUT / "fig6_cleaning_funnel.png", dpi=200)
    pd.DataFrame([{"step": s, "rows": n, "removed_because": w} for s, n, w in steps]).to_csv(
        OUT / "fig6_cleaning_funnel.csv", index=False)
    plt.close(fig)


def outage_numbers(df: pd.DataFrame, events: pd.DataFrame):
    pw = df.groupby(["week", "place_group"]).y.mean()
    city = pw.groupby(level=0).mean().rename("city_mean").reset_index()
    jk = events[(events.plant == "JK") & events.source.eq("inferred_from_plant_sampling_gaps")].copy()
    jk["start"], jk["end"] = pd.to_datetime(jk.start), pd.to_datetime(jk.end)
    off = df.jk_offline.astype(bool)
    month = df.sampled_at.dt.month
    rows = []
    for pg, g in df.groupby("place_group"):
        o, n = g[off[g.index]], g[~off[g.index]]
        n = n[month[n.index].isin(month[o.index].unique())]       # same calendar months, controls season
        if len(o) >= 5 and len(n) >= 10:
            rows.append({"place_group": pg, "pressure_plane": g.pressure_plane.iloc[0],
                         "drop_mg_l": o.y.mean() - n.y.mean(), "n_outage": len(o)})
    return city, jk, pd.DataFrame(rows)


def outages():
    cfg = EvalConfig()
    df, _ = load_frame(cfg)
    events = pd.read_csv(cfg.data_dir / "events.csv")
    city, jk, drops = outage_numbers(df, events)
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_axes([0.06, 0.13, 0.6, 0.62])
    _style(ax, "y")
    for r in jk.itertuples():
        ax.axvspan(r.start, r.end, color=BAND, alpha=0.18, linewidth=0, zorder=0)
    ax.plot(city.week, city.city_mean, color=SLOT[0], linewidth=1.2, zorder=2)
    ax.set_ylabel("City-wide weekly mean (mg/L)")
    ax.set_ylim(1.5, 4.6)
    conf = jk[jk.confidence == "externally_confirmed"].iloc[0]
    ax.text(conf.end + pd.Timedelta(days=25), 4.45, "documented by a city notice", fontsize=8, color=INK2, va="center")
    amb = jk[jk.confidence == "ambiguous"]
    for r in amb.itertuples():
        ax.text(r.start + (r.end - r.start) / 2, 4.45, "no dip", fontsize=8, color=MUTED, ha="center")
    handles = [plt.Line2D([], [], color=SLOT[0], linewidth=2), plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.25)]
    ax.legend(handles, ["City-wide weekly mean residual", "Kubala plant offline (inferred from sampling gaps)"],
              loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
    # Right: drop at each site during outages, by pressure plane.
    bx = fig.add_axes([0.74, 0.13, 0.24, 0.62])
    _style(bx, "x")
    bx.axvline(0, color=AXIS, linewidth=1)
    order = ["UPPER", "LOWER", "WEST"]
    rng = np.random.default_rng(0)
    summary = []
    for i, plane in enumerate(order):
        d = drops[drops.pressure_plane == plane]
        y = len(order) - 1 - i
        bx.scatter(d.drop_mg_l, y + rng.uniform(-0.18, 0.18, len(d)), s=14, color=SLOT[0], alpha=0.55, linewidths=0)
        bx.plot([d.drop_mg_l.mean()] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2)
        summary.append({"pressure_plane": plane, "places": len(d), "mean_drop": d.drop_mg_l.mean(),
                        "median_drop": d.drop_mg_l.median()})
        bx.text(0.22, y, f"{d.drop_mg_l.mean():+.2f}\n({len(d)} places)", fontsize=8.5, color=INK2, va="center")
    bx.set_yticks(range(len(order)), [p.title() + " plane" for p in order][::-1])
    bx.set_xlim(-1.8, 0.75)
    bx.set_xlabel("Change during outages (mg/L)")
    bx.set_title("Drop at each site, by pressure plane", loc="left", fontsize=10, fontweight="bold", color=INK)
    dips = jk[jk.confidence.isin(["externally_confirmed", "supported_by_dip"])].city_dip_mg_l
    _title(fig, f"When the Kubala plant goes offline, the city-wide residual drops {-dips.max():.1f}-{-dips.min():.1f} mg/L",
           "Six of seven inferred outages coincide with a dip; sites in the Upper plane, which Kubala feeds, drop most. "
           "Line = mean; dots = sites (same calendar months compared).")
    fig.savefig(OUT / "fig7_kubala_outages.png", dpi=200)
    city.to_csv(OUT / "fig7_kubala_outages_weekly.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "fig7_kubala_outages_by_plane.csv", index=False)
    plt.close(fig)


def variance_numbers(df: pd.DataFrame, within_sd: float) -> pd.DataFrame:
    """Method-of-moments split of sample variance into city-week level, site level, local site-week deviation
    and within-week scatter (the noise floor estimated at the near-daily sites)."""
    pw = df.groupby(["week", "place_group"]).y.agg(["mean", "size"]).reset_index()
    pw["city"] = pw.groupby("week")["mean"].transform("mean")
    pw["dev"] = pw["mean"] - pw.city
    pw["off"] = pw.groupby("place_group").dev.transform("mean")
    pw["local"] = pw.dev - pw.off
    rows = df[["week", "place_group"]].merge(pw, on=["week", "place_group"])
    s2 = within_sd ** 2
    comps = {"City-wide weekly level": rows.city.var(),
             "Site's long-run level": rows.off.var(),
             "Local site-week deviation": rows.local.var() - float(np.mean(s2 / rows["size"])),
             "Within-week scatter (noise floor)": s2}
    total = float(df.y.var())
    out = pd.DataFrame({"component": list(comps), "variance": list(comps.values())})
    out["share"] = out.variance / out.variance.sum()
    out["sd_mg_l"] = np.sqrt(out.variance)
    out.attrs["total_variance"] = total
    return out


def variance():
    cfg = EvalConfig()
    df, _ = load_frame(cfg)
    floor = noise_floor(df, cfg)
    v = variance_numbers(df, floor["within_week_sd"])
    captured = ["the network: other sites' readings this week", "site history (known sites) or public features (unseen)",
                "same-week neighbors and the site's recent readings", "nothing: irreducible at weekly resolution"]
    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.66])
    left = 0.0
    for (r, color, cap) in zip(v.itertuples(), SLOT4, captured):
        ax.barh(1.0, r.share, left=left, height=0.42, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(left + r.share / 2, 1.0, f"{r.share:.0%}", color="white" if color == SLOT4[0] else INK, fontsize=13,
                fontweight="bold", ha="center", va="center")
        width = max(int(r.share * 95), 14)
        ax.text(left + 0.006, 0.66, textwrap.fill(r.component, width), color=INK, fontsize=9.5, fontweight="bold", va="top")
        ax.text(left + 0.006, 0.36, f"SD {r.sd_mg_l:.2f} mg/L", color=INK2, fontsize=8.5, va="top")
        ax.text(left + 0.006, 0.2, textwrap.fill("Captured by " + cap, width), color=MUTED, fontsize=8, va="top")
        left += r.share
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 1.3)
    ax.axis("off")
    _title(fig, "Where the variation in chlorine residual comes from",
           f"All {len(df):,} modeling samples, 2015-2026 (total SD {df.y.std():.2f} mg/L). Within-week scatter is measured "
           f"at the near-daily sites ({floor['within_week_sd']:.2f} mg/L): no weekly model can remove it.")
    fig.savefig(OUT / "fig8_variance_breakdown.png", dpi=200)
    v.to_csv(OUT / "fig8_variance_breakdown.csv", index=False)
    plt.close(fig)


# -- Proposal figures (data and baseline only; no main-model results) --------------------------------------------


def sampling_numbers(df: pd.DataFrame) -> dict:
    wk = df.groupby("week").agg(places=("place_group", "nunique"), samples=("y", "size")).reset_index()
    yearly = df.groupby(df.sampled_at.dt.year).place_group.nunique()
    per = []
    for _, g in df.groupby("place_group"):          # per place, while it is active (first to last sample)
        weeks = g.week.drop_duplicates()
        span = (weeks.max() - weeks.min()).days / 7 + 1
        gaps = g.sampled_at.sort_values().diff().dt.total_seconds().dropna() / 86400
        per.append((len(weeks) / span, gaps.median() if len(gaps) else np.nan))
    per = pd.DataFrame(per, columns=["share_weeks", "median_gap"])
    return {"weekly": wk, "yearly_places": yearly, "median_places_week": float(wk.places.median()),
            "median_gap_days": float(per.median_gap.median()), "share_weeks": float(per.share_weeks.median()),
            "places_under_half": int((per.share_weeks < 0.5).sum()), "places": len(per)}


def sampling_coverage():
    df, _ = load_frame(EvalConfig())
    nb = sampling_numbers(df)
    wk, yearly = nb["weekly"], nb["yearly_places"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.71, bottom=0.12)
    _style(ax, "y")
    ax.plot(wk.week, wk.places, color=SLOT[0], linewidth=0.6, alpha=0.35, zorder=1)
    roll = wk.set_index("week").places.rolling(13, center=True, min_periods=7).mean()
    ax.plot(roll.index, roll.to_numpy(), color=SLOT[0], linewidth=1.8, zorder=2,
            label="Places sampled per week (thin: each week; thick: 13-week average)")
    edges = [pd.Timestamp(f"{y}-01-01") for y in yearly.index] + [df.sampled_at.max()]
    ax.step(edges, list(yearly.to_numpy()) + [yearly.to_numpy()[-1]], where="post", color=SLOT[1], linewidth=1.6,
            zorder=2, label="Places sampled at least once that year")
    ax.set_ylim(0, 180)
    ax.set_ylabel("Places")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    _title(fig, f"About {nb['median_places_week']:.0f} places are sampled in a typical week, of {yearly.min()}-"
                f"{yearly.max()} active each year",
           f"{len(df):,} routine samples, 2015 to July 2026. A typical place is sampled about every two weeks (median "
           f"{nb['median_gap_days']:.1f} days) and has a sample in only {nb['share_weeks']:.0%} of its active weeks;\n{nb['places_under_half']} of "
           f"{nb['places']} places are unsampled in most weeks, and most of the city's addresses are never sampled.")
    fig.savefig(OUT / "fig11_sampling_coverage.png", dpi=200)
    wk.to_csv(OUT / "fig11_sampling_coverage_weekly.csv", index=False)
    yearly.rename("places_sampled").rename_axis("year").reset_index().to_csv(
        OUT / "fig11_sampling_coverage_yearly.csv", index=False)
    plt.close(fig)


def chlorine_distribution():
    df, _ = load_frame(EvalConfig())
    y = df.y
    bins = np.round(np.arange(-0.05, 6.06, 0.1), 2)     # centered on the 0.1 mg/L values the lab records
    counts, _ = np.histogram(y, bins=bins)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.75, bottom=0.12)
    _style(ax, "y")
    ax.bar(bins[:-1] + 0.05, counts, width=0.1, align="center", color=SLOT[0], edgecolor=SURFACE, linewidth=0.5, zorder=2)
    top = counts.max()
    marks = [(0.5, f"0.5  Texas minimum,\neverywhere in the pipes\n({int((y < 0.5).sum())} of {len(y):,} samples below)"),
             (2.5, f"2.5  our screening threshold\nfor 'low' (not a legal limit)\n{(y < 2.5).mean():.0%} of samples below"),
             (4.0, f"4.0  maximum for the yearly\naverage (single samples may\nexceed it: {(y > 4.0).mean():.0%} do)")]
    for x, label in marks:
        ax.axvline(x, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
        ax.text(x + 0.06, top * 0.97, label, fontsize=8.5, color=INK2, va="top")
    ax.set_xlim(0, 6)
    ax.set_xlabel("Total chlorine (mg/L)")
    ax.set_ylabel("Samples")
    _title(fig, f"Residual is usually {y.quantile(.1):.1f}-{y.quantile(.9):.1f} mg/L, well above the 0.5 mg/L minimum",
           f"All {len(y):,} routine samples, 2015-2026 (average {y.mean():.2f} mg/L). Limits: Texas 30 TAC 290.110 "
           "(chloramine, measured as total chlorine).")
    fig.savefig(OUT / "fig12_chlorine_distribution.png", dpi=200)
    pd.DataFrame({"value_mg_l": np.round(bins[:-1] + 0.05, 2), "samples": counts}).to_csv(OUT / "fig12_chlorine_distribution.csv",
                                                                       index=False)
    plt.close(fig)


def season_plant_numbers(df: pd.DataFrame) -> dict:
    monthly = df.groupby(df.sampled_at.dt.month).y.mean()
    yearly = df.groupby(df.sampled_at.dt.year).agg(mean=("y", "mean"), jk_share=("jk_offline", "mean"),
                                                   n=("y", "size"))
    w = df.groupby("week").agg(y=("y", "mean"), jk=("jk_offline", "mean"))
    return {"monthly": monthly, "yearly": yearly, "r": float(np.corrcoef(yearly["mean"], yearly.jk_share)[0, 1]),
            "week_on": float(w[w.jk < 0.5].y.mean()), "week_off": float(w[w.jk >= 0.5].y.mean())}


def season_vs_plant():
    df, _ = load_frame(EvalConfig())
    nb = season_plant_numbers(df)
    monthly, yearly = nb["monthly"], nb["yearly"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.72, bottom=0.13, wspace=0.08)
    for ax in (a, b):
        _style(ax, "y")
    a.plot(monthly.index, monthly.to_numpy(), color=SLOT[0], linewidth=1.6, zorder=2)
    for m, v in monthly.items():
        _dot(a, m, v, SLOT[0], size=6)
    a.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    a.set_title("By calendar month (season)", loc="left", fontsize=10, fontweight="bold", color=INK)
    a.set_ylabel("Average total chlorine (mg/L)")
    a.set_ylim(3.0, 3.8)
    for yr_, r in yearly.iterrows():
        _dot(b, 100 * r.jk_share, r["mean"], SLOT[1], size=7)
        label = f"{yr_}" + (" (Jan-Jul)" if yr_ == yearly.index.max() else "")
        b.text(100 * r.jk_share + 0.8, r["mean"] + 0.012, label, fontsize=8, color=INK2)
    b.set_xlim(-2, 42)
    b.set_xlabel("Share of the year's samples taken while Kubala was offline (%)")
    b.set_title("By year, against Kubala outage time", loc="left", fontsize=10, fontweight="bold", color=INK)
    _title(fig, "Plant outages, not season, move Arlington's chlorine levels",
           f"Monthly averages span only {monthly.min():.2f}-{monthly.max():.2f} mg/L. Years with more Kubala outage time "
           f"average lower (correlation {nb['r']:.2f}).\nWeek by week, the city-wide average is {nb['week_on']:.2f} mg/L "
           f"with Kubala running and {nb['week_off']:.2f} with it offline.")
    fig.savefig(OUT / "fig13_season_vs_plant.png", dpi=200)
    monthly.rename("mean_mg_l").rename_axis("month").reset_index().to_csv(OUT / "fig13_season_monthly.csv", index=False)
    yearly.rename_axis("year").reset_index().to_csv(OUT / "fig13_plant_yearly.csv", index=False)
    plt.close(fig)


def evaluation_design():
    from matplotlib.patches import Rectangle
    df, _ = load_frame(EvalConfig())
    n_train = int((df.sampled_at < "2023-01-01").sum())
    n_val = int(((df.sampled_at >= "2023-01-01") & (df.sampled_at < "2024-01-01")).sum())
    n_test = int((df.sampled_at >= "2024-01-01").sum())
    end = df.sampled_at.max()
    fig = plt.figure(figsize=(11, 5.2))
    t = fig.add_axes([0.04, 0.52, 0.92, 0.24])
    t.axis("off")
    x0 = pd.Timestamp("2015-01-01")
    span = lambda d: (pd.Timestamp(d) - x0).days / 365.25  # noqa: E731
    segs = [(x0, "2023-01-01", SLOT[0], f"TRAIN 2015-2022\nmodels learn patterns\n{n_train:,} samples"),
            ("2023-01-01", "2024-01-01", SLOT[1], f"VALID.\n2023\nchoose\nmodels\n{n_val:,}"),
            ("2024-01-01", end, SLOT[2], f"TEST 2024-Jul 2026\nfinal check, used once\n{n_test:,} samples")]
    for s, e, c, label in segs:
        t.add_patch(Rectangle((span(s), 0.35), span(e) - span(s), 0.5, color=c, linewidth=0))
        t.text(span(s) + (span(e) - span(s)) / 2, 0.6, label, ha="center", va="center",
               fontsize=7.2 if label.startswith("VALID") else 8.5,
               color="white" if c == SLOT[0] else INK, fontweight="bold")
    for yr_ in range(2015, 2027):
        t.text(span(f"{yr_}-01-01"), 0.18, str(yr_), fontsize=8, color=MUTED, ha="left")
    t.set_xlim(0, span(end) + 0.05)
    t.set_ylim(0, 1)
    t.text(0, 1.0, "1. Time: learn from the past, judge on later years", fontsize=10.5, fontweight="bold", color=INK)
    # Unseen-site schematic: 5 rounds, one fifth of the places hidden each time (abstract, not real locations).
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, (40, 2))
    fold = np.arange(40) % 5
    for k in range(5):
        ax = fig.add_axes([0.04 + k * 0.105, 0.07, 0.095, 0.25])
        ax.axis("off")
        ax.scatter(pts[fold != k, 0], pts[fold != k, 1], s=14, color=SLOT[0], linewidths=0)
        ax.scatter(pts[fold == k, 0], pts[fold == k, 1], s=22, facecolor=SURFACE, edgecolor=SLOT[1], linewidths=1.4)
        ax.set_title(f"round {k + 1}", fontsize=8, color=MUTED, pad=2)
    fig.text(0.04, 0.38, "2. Places: hide whole places to test unsampled locations", fontsize=10.5, fontweight="bold",
             color=INK)
    fig.text(0.04, 0.345, "blue = monitored places the model may use; open orange = hidden places it must estimate "
                          "(schematic)", fontsize=8.5, color=INK2)
    fig.text(0.60, 0.38, "3. No peeking", fontsize=10.5, fontweight="bold", color=INK)
    rules = ["A prediction never uses its own sample,", "or anything from its own place that week.",
             "The test years are removed while developing.", "Automated checks: setting hidden values to 1000",
             "leaves every prediction unchanged.", "Scores come with 95% ranges from resampling", "whole places."]
    for i, r in enumerate(rules):
        fig.text(0.60, 0.31 - i * 0.04, r, fontsize=9, color=INK2)
    _title(fig, "How every model is tested: the same rules for the baseline and any future model",
           "Two scenarios: known sites (fill in weeks without a sample) and unseen places (never sampled). A third test "
           "hides each plant outage.")
    fig.savefig(OUT / "fig14_evaluation_design.png", dpi=200)
    pd.DataFrame([{"period": "train", "start": "2015-01-01", "end": "2022-12-31", "samples": n_train},
                  {"period": "validation", "start": "2023-01-01", "end": "2023-12-31", "samples": n_val},
                  {"period": "test", "start": "2024-01-01", "end": str(end.date()), "samples": n_test}]).to_csv(
        OUT / "fig14_evaluation_design.csv", index=False)
    plt.close(fig)


HARD_CASES = [("val", "known_site", "B5p", None, "Known sites, calm 2023"),
              ("val", "unseen_site", "B4p", None, "Unseen places, 2023"),
              ("test", "known_site", "B5p", None, "Known sites, 2024-26"),
              ("test", "unseen_site", "B4p", None, "Unseen places, 2024-26"),
              ("test", "known_site", "B5p", "new_site", "New sampling sites, 2024-26"),
              ("val", "outage", "B5p", None, "Kubala outages before 2024"),
              ("test", "known_site", "B5p", "jk_offline", "Kubala outages, 2024-26")]


def hard_case_numbers() -> pd.DataFrame:
    from ..evaluation.metrics import bootstrap_rmse
    cfg = EvalConfig()
    preds = {s: pd.read_parquet(cfg.out_root / s / "predictions.parquet") for s in ("val", "test")}
    rows = []
    for split, protocol, model, subset, label in HARD_CASES:
        g = preds[split]
        g = g[(g.protocol == protocol) & (g.model == model)]
        if subset:
            g = g[g[subset].fillna(False).astype(bool)]
        b = bootstrap_rmse(g, model, cfg.bootstrap_reps, cfg.seed).iloc[0]
        rows.append({"situation": label, "split": split, "protocol": protocol, "model": model, "n": len(g),
                     "places": g.place_group.nunique(), "rmse": float(np.sqrt(np.mean((g.pred - g.y) ** 2))),
                     "rmse_lo": b.rmse_lo, "rmse_hi": b.rmse_hi})
    return pd.DataFrame(rows)


def baseline_hard_cases():
    cfg = EvalConfig()
    t = hard_case_numbers()
    floor = json.loads((cfg.out_root / "val" / "run.json").read_text())["noise_floor"]["within_week_sd"]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.72, bottom=0.12)
    _style(ax)
    ax.axvline(floor, color=MUTED, linewidth=1)
    ax.text(floor + 0.002, len(t) - 0.45, f"noise floor {floor:.2f}", fontsize=8.5, color=MUTED)
    for i, r in enumerate(t.itertuples()):
        y = len(t) - 1 - i
        c = SLOT[1] if r.protocol == "unseen_site" else SLOT[0]
        ax.plot([r.rmse_lo, r.rmse_hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
        _dot(ax, r.rmse, y, c)
        ax.text(r.rmse_hi + 0.004, y, f"{r.rmse:.3f}", fontsize=9, color=INK2, va="center")
    ax.set_yticks(range(len(t)), [f"{s}  ({n:,} samples)" for s, n in zip(t.situation[::-1], t.n[::-1])])
    ax.set_xlim(0.25, 0.52)
    ax.set_xlabel("Typical miss, RMSE (mg/L), lower is better")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", mec=SURFACE, mew=2, ms=7) for c in SLOT[:2]]
    fig.legend(handles, ["Known sites: plane neighbors + site offset (B5p)", "Unseen places: plane neighbors (B4p)"],
               loc="upper left", bbox_to_anchor=(0.004, 0.86), ncol=2, frameon=False, labelcolor=INK2, fontsize=9)
    _title(fig, "The baseline nears the noise floor in calm weeks, but not during outages or at new sites",
           "Best public-data baselines. Dot = typical miss; line = 95% range (resampling places; new sites are only 24 "
           "places, hence wide).\nThese hard cases are where a better model must improve.")
    fig.savefig(OUT / "fig15_baseline_hard_cases.png", dpi=200)
    t.to_csv(OUT / "fig15_baseline_hard_cases.csv", index=False)
    plt.close(fig)


def project_roadmap():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    man = json.loads((CleaningConfig().out_dir / "manifest.json").read_text())
    done = [("1  Data", f"Lab export ({man['rows']['raw']:,} rows)\n+ public city GIS"),
            ("2  Cleaning", "Scope, locations,\nduplicates, privacy"),
            ("3  Features", "Pressure plane, plant\ndistance, hydrants"),
            ("4  Plant status", "Outages inferred\nfrom sampling gaps"),
            ("5  Exploration", "Outages, sources of\nvariation, noise floor"),
            ("6  Fair testing", "Time split, hidden\nplaces, leakage tests"),
            ("7  Baselines", "Neighbor-averaging\nladder + results")]
    nxt = [("8  Main model", "Regression kriging\n(exploring)"), ("9  Uncertainty", "Calibrated\nconfidence ranges"),
           ("10  Final evaluation", "Held-out and new data;\nreport, deliverables")]
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.78])
    ax.axis("off")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2.25)
    w, h = 0.86, 0.72

    def box(x, y, title, body, filled):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=SLOT[0] if filled else SURFACE, edgecolor=SLOT[0], linewidth=1.4,
                                    linestyle="-" if filled else (0, (4, 3))))
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="center", fontsize=9, fontweight="bold",
                color="white" if filled else INK)
        ax.text(x + w / 2, y + 0.27, body, ha="center", va="center", fontsize=7.6,
                color="white" if filled else INK2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))

    xs = [0.05 + i * 0.99 for i in range(7)]
    for i, (tt, bb) in enumerate(done):
        box(xs[i], 1.35, tt, bb, True)
        if i:
            arrow(xs[i - 1] + w, 1.35 + h / 2, xs[i], 1.35 + h / 2)
    ax.text(xs[0], 2.15, "COMPLETED: presented today", fontsize=9.5, fontweight="bold", color=SLOT[0], va="center")
    for i, (tt, bb) in enumerate(nxt):
        box(xs[4 + i], 0.1, tt, bb, False)
        if i:
            arrow(xs[3 + i] + w, 0.1 + h / 2, xs[4 + i], 0.1 + h / 2)
    # elbow connector: down from step 7, left above the proposed row, down into step 8
    ax.plot([xs[6] + w / 2, xs[6] + w / 2, xs[4] + w / 2], [1.35, 1.05, 1.05], color=MUTED, lw=1)
    arrow(xs[4] + w / 2, 1.05, xs[4] + w / 2, 0.1 + h + 0.01)
    ax.text(xs[3] + w, 0.46, "PROPOSED:\nnext phase", fontsize=9.5, fontweight="bold", color=INK2, ha="right",
            va="center")
    _title(fig, "Project roadmap: from raw lab records to a city-wide virtual sensor",
           "Steps 1-7 are complete. Steps 8-10 are proposed: we are exploring regression kriging as the main model; "
           "further research is needed.")
    fig.savefig(OUT / "fig16_project_roadmap.png", dpi=200)
    pd.DataFrame([{"step": tt, "detail": bb.replace("\n", " "), "status": s} for s, items in
                  (("completed", done), ("proposed", nxt)) for tt, bb in items]).to_csv(
        OUT / "fig16_project_roadmap.csv", index=False)
    plt.close(fig)


def proposal():
    """The proposal figures (data and baseline only)."""
    sampling_coverage()
    chlorine_distribution()
    season_vs_plant()
    evaluation_design()
    baseline_hard_cases()
    project_roadmap()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _setup()
    funnel()
    outages()
    variance()
    proposal()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
