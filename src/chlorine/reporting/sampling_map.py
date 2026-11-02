"""Interactive map of model results at every sampling place.

    uv run python -m chlorine.reporting.sampling_map

Writes reports/map/chlorine_results_map.html, one self-contained file with no outside requests.
Addresses are left out.

To add a model, fill in a row of config/map_models.csv with its code (the `model` value in the file),
a label, a group, and the predictions.parquet paths for validation and test, then rerun. A row with no
paths appears in the model menu as an empty slot. The file needs the columns of the evaluation
harness: model, protocol, split, row_id, y, pred, q10, q90.

Test-period views only display saved predictions. They are not for choosing models or settings.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..cleaning.addresses import contains_phone
from ..config import ROOT, EvalConfig

MODELS_CSV = ROOT / "config" / "map_models.csv"
TEMPLATE = Path(__file__).with_name("sampling_map_template.html")
OUT = ROOT / "reports" / "map" / "chlorine_results_map.html"
GIS_PAGES = ROOT / "data" / "external" / "arlington_gis" / "pages"
ROAD_LEVEL = {"Freeway": 2, "Highway": 2, "Major Arterial": 1, "Collector": 1}
RMSE_TOLERANCE = 0.001  # predictions are stored to 0.001 mg/L

SCENARIOS = [
    {"id": "val_known", "file": "val", "protocol": "known_site", "split": "val", "is_test": False,
     "label": "2023 validation: monitored places"},
    {"id": "val_unseen", "file": "val", "protocol": "unseen_site", "split": "val", "is_test": False,
     "label": "2023 validation: places hidden from the model"},
    {"id": "outage", "file": "val", "protocol": "outage", "split": "pre_test", "is_test": False,
     "label": "Kubala outages before 2024: each outage hidden"},
    {"id": "test_known", "file": "test", "protocol": "known_site", "split": "test", "is_test": True,
     "label": "2024 to July 2026 test: monitored places"},
    {"id": "test_unseen", "file": "test", "protocol": "unseen_site", "split": "test", "is_test": True,
     "label": "2024 to July 2026 test: places hidden from the model"},
]


def _milli(s: pd.Series) -> list:
    return [None if pd.isna(v) else int(round(v * 1000)) for v in s]


def load_registry(path: Path = MODELS_CSV) -> pd.DataFrame:
    reg = pd.read_csv(path, dtype=str, keep_default_na=False)
    if reg.code.duplicated().any():
        raise ValueError(f"duplicate model codes in {path}")
    return reg


def _read_predictions(reg: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
    cols = ["model", "protocol", "split", "row_id", "place_group", "sampled_at", "jk_offline", "new_site",
            "y", "pred", "q10", "q90"]
    frames = {}
    for path, group in reg[reg[column] != ""].groupby(column):
        p = pd.read_parquet(ROOT / path, columns=cols)
        frames[path] = p[p.model.isin(set(group.code))]
    return frames


def _scenario(sc: dict, reg: pd.DataFrame, frames: dict[str, pd.DataFrame], place_index: dict) -> dict:
    column = f"{sc['file']}_predictions"
    parts = {}
    for r in reg[reg[column] != ""].itertuples():
        f = frames[getattr(r, column)]
        g = f[(f.model == r.code) & (f.protocol == sc["protocol"]) & (f.split == sc["split"])]
        if len(g):
            parts[r.code] = g
    if not parts:
        raise ValueError(f"no predictions for scenario {sc['id']}")
    base = max(parts.values(), key=len).drop_duplicates("row_id").sort_values(["sampled_at", "row_id"])
    base = base[["row_id", "place_group", "sampled_at", "jk_offline", "new_site", "y"]].set_index("row_id")
    preds = {}
    for code, g in parts.items():
        if g.row_id.duplicated().any() or not set(g.row_id) <= set(base.index):
            raise ValueError(f"{code} rows in {sc['id']} do not match the scenario's samples")
        g = g.set_index("row_id")
        if not np.allclose(g.y, base.y.loc[g.index]):
            raise ValueError(f"{code} has different measured values in {sc['id']}")
        g = g.reindex(base.index)
        preds[code] = {"p": _milli(g.pred), "lo": _milli(g.q10), "hi": _milli(g.q90)}
    days = (base.sampled_at - pd.Timestamp("1970-01-01")).dt.days
    return {
        **{k: sc[k] for k in ("id", "label", "is_test")},
        "place": [place_index[p] for p in base.place_group],
        "day": [int(d) for d in days],
        "y": _milli(base.y),
        "jk": [int(bool(v)) for v in base.jk_offline.fillna(False)],
        "new": [int(bool(v)) for v in base.new_site.fillna(False)],
        "preds": preds,
    }


def _places(cfg: EvalConfig, wanted: set[str]) -> list[dict]:
    loc = pd.read_csv(cfg.data_dir / "locations.csv", dtype={"site_code": str})
    tab = pd.read_parquet(cfg.data_dir / "chlorine_v1.parquet", columns=["location_id", "place_group"])
    counts = tab.groupby(["place_group", "location_id"]).size().rename("n").reset_index()
    meta = loc.set_index("location_id")
    out = []
    for place in sorted(wanted):
        rows = counts[counts.place_group == place].sort_values("n", ascending=False)
        m = meta.loc[rows.location_id]
        placed = m.dropna(subset=["lon", "lat"])
        top = placed.iloc[0] if len(placed) else m.iloc[0]  # the most-sampled location that has coordinates
        out.append({
            "id": place,
            "lon": None if pd.isna(top.lon) else round(float(top.lon), 5),
            "lat": None if pd.isna(top.lat) else round(float(top.lat), 5),
            "codes": sorted({c for c in m.site_code.dropna()}),
            "plane": None if pd.isna(top.pressure_plane) else str(top.pressure_plane).title(),
            "region": None if pd.isna(top.region) else str(top.region),
            "total": int(rows.n.sum()),
        })
    return out


def _plants(cfg: EvalConfig) -> list[dict]:
    loc = pd.read_csv(cfg.data_dir / "locations.csv")
    p = loc[loc.location_type == "plant"].dropna(subset=["lon", "lat"]).drop_duplicates("facility")
    names = {"JK": "Kubala plant", "PB": "Pierce-Burch plant"}
    return [{"name": names[r.facility], "lon": round(r.lon, 5), "lat": round(r.lat, 5)} for r in p.itertuples()]


def _roads() -> list[list]:
    """Public road centerlines as [level, lon0, lat0, dlon, dlat, ...] in units of 0.0001 degree."""
    roads = []
    for f in sorted(GIS_PAGES.glob("road_centerlines_*.json")):
        for feat in json.loads(f.read_text())["features"]:
            level = ROAD_LEVEL.get(feat["attributes"].get("roadclass"), 0)
            for path in (feat.get("geometry") or {}).get("paths", []):
                pts = [(round(x * 1e4), round(y * 1e4)) for x, y in path]
                kept = [pts[0]] + [b for a, b in zip(pts, pts[1:]) if a != b]
                if len(kept) < 2:
                    continue
                flat = [level, *kept[0]]
                for a, b in zip(kept, kept[1:]):
                    flat += [b[0] - a[0], b[1] - a[1]]
                roads.append(flat)
    return roads


def build_payload(registry: Path = MODELS_CSV, cfg: EvalConfig | None = None) -> dict:
    cfg = cfg or EvalConfig()
    reg = load_registry(registry)
    frames = {c: _read_predictions(reg, c) for c in ("val_predictions", "test_predictions")}
    wanted = set(pd.read_parquet(cfg.data_dir / "chlorine_v1.parquet", columns=["place_group"]).place_group)
    places = _places(cfg, wanted)  # every sampled place, including those with no results in any view
    index = {p["id"]: i for i, p in enumerate(places)}
    scenarios = [_scenario(sc, reg, frames[f"{sc['file']}_predictions"], index) for sc in SCENARIOS]
    models = [{"code": r.code, "label": r.label, "group": r.group,
               "ready": bool(r.val_predictions or r.test_predictions)} for r in reg.itertuples()]
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "noise_floor": 0.28,
        "models": models, "places": places, "plants": _plants(cfg), "roads": _roads(), "scenarios": scenarios,
    }


def scenario_rmse(sc: dict, code: str) -> float:
    pred = np.array(sc["preds"][code]["p"], dtype=float)
    y = np.array(sc["y"], dtype=float)
    return float(np.sqrt(np.nanmean((pred - y) ** 2)) / 1000)


def check_against_results(payload: dict, cfg: EvalConfig | None = None) -> int:
    """Every baseline's RMSE on the map must equal the evaluation's saved result. Returns checks made."""
    cfg = cfg or EvalConfig()
    specs = {s["id"]: s for s in SCENARIOS}
    checked = 0
    for sc in payload["scenarios"]:
        spec = specs[sc["id"]]
        res = pd.read_csv(ROOT / "data" / "eval" / spec["file"] / "results.csv")
        res = res[(res.protocol == spec["protocol"]) & (res.split == spec["split"]) & (res.subset == "all")]
        for r in res.itertuples():
            if r.model in sc["preds"]:
                if abs(scenario_rmse(sc, r.model) - r.rmse) > RMSE_TOLERANCE:
                    raise ValueError(f"{sc['id']} {r.model}: map {scenario_rmse(sc, r.model)} vs results {r.rmse}")
                checked += 1
    return checked


def write_map(payload: dict, out: Path = OUT) -> Path:
    data = json.dumps(payload, separators=(",", ":"))
    if contains_phone(json.dumps({"places": payload["places"], "models": payload["models"]})):
        raise ValueError("phone-number pattern found in map data; refusing to write")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.read_text().replace("__MAP_DATA__", data.replace("</", "<\\/")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the sampling-place results map.")
    ap.add_argument("--models", type=Path, default=MODELS_CSV, help="model list (default config/map_models.csv)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    payload = build_payload(args.models)
    checked = check_against_results(payload)
    path = write_map(payload, args.out)
    on_map = sum(p["lon"] is not None for p in payload["places"])
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB): {on_map} of {len(payload['places'])} places on the map; "
          f"{checked} RMSE values match the saved results")


if __name__ == "__main__":
    main()
