"""Local geocoding and public-GIS location features.

Everything runs offline against layers downloaded by scripts/fetch_public_gis.sh; no address
leaves this machine.

Geocoding order (first hit wins, tried for every candidate street core):
  1. address_point         exact house number + street in the public service-line address points
  2. street_range          interpolation along the matching road-centerline address range
  3. address_point_nearby  nearest house number on the same street (within max_nearest_number_gap)
  4. *_fuzzy_street        1-3 again after correcting the street name to the closest known street
Written directionals are respected (see Geocoder). geocode_uncertainty_m is ~10 m for an address
point, half the matched segment length for street_range, and 10 m per house number of offset for
address_point_nearby.
Features from the public fire-hydrant layer (an export of the utility's network GIS):
  pressure plane (vote of the k nearest hydrants), distance to the nearest hydrant, hydrant count,
  flushing-program share and median hydrant acceptance year within a radius.
"""
from __future__ import annotations

import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .addresses import Address, parse_address, street_core

LAT0, LON0 = 32.70, -97.12


def to_xy(lon, lat) -> np.ndarray:
    """Local equirectangular projection in meters (accurate to <0.1% across the city)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return np.c_[(lon - LON0) * 111_320 * np.cos(np.radians(LAT0)), (lat - LAT0) * 110_540]


def load_layer(gis_dir: Path, name: str) -> pd.DataFrame:
    pages = sorted((gis_dir / "pages").glob(f"{name}_*.json"))
    if not pages:
        raise FileNotFoundError(f"No pages for layer '{name}' in {gis_dir}; run scripts/fetch_public_gis.sh")
    rows = []
    for p in pages:
        j = json.loads(p.read_text())
        if "error" in j:
            raise ValueError(f"{p.name}: server error {j['error']}")
        for f in j["features"]:
            rec = dict(f["attributes"])
            g = f.get("geometry") or {}
            if "x" in g:
                rec["lon"], rec["lat"] = g["x"], g["y"]
            elif "paths" in g:
                rec["paths"] = g["paths"]
            rows.append(rec)
    df = pd.DataFrame(rows).drop_duplicates("OBJECTID")
    expected = json.loads((gis_dir / "counts.json").read_text()).get(name)
    if expected is not None and len(df) != expected:
        raise ValueError(f"Layer '{name}': {len(df)} records, expected {expected}")
    return df


AMBIGUOUS_SPREAD_M = 300.0


class Geocoder:
    """Direction-aware local geocoder. Candidates whose written directional conflicts with the
    query are dropped; if the remaining candidates are more than AMBIGUOUS_SPREAD_M apart (e.g.
    "1500 EXAMPLE PKWY" exists as both W and E), the result is 'ambiguous_direction' rather
    than an average of two far-apart places."""

    def __init__(self, address_points: pd.DataFrame, roads: pd.DataFrame, max_number_gap: int):
        self.max_gap = max_number_gap
        self.exact = defaultdict(list)       # (number, core) -> [(direction, lon, lat)]
        self.street = defaultdict(list)      # core -> [(number, direction, lon, lat)]
        for text, lon, lat in zip(address_points.FullAddress, address_points.lon, address_points.lat):
            a = parse_address(text)
            if a is None or pd.isna(lon):
                continue
            for core in a.cores:
                self.exact[(a.number, core)].append((a.direction, lon, lat))
                self.street[core].append((a.number, a.direction, lon, lat))
        self.segments = defaultdict(list)    # core -> [(lo, hi, from, to, path, direction, length_m)]
        for r in roads.itertuples():
            if not isinstance(r.fullname, str) or not isinstance(r.paths, list) or not r.paths:
                continue
            core, direction = street_core(r.fullname)
            path = np.array(r.paths[0])
            xy = to_xy(path[:, 0], path[:, 1])
            cum = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            for lo, hi in ((r.fromleft, r.toleft), (r.fromright, r.toright)):
                if pd.notna(lo) and pd.notna(hi) and lo > 0 and hi > 0:
                    self.segments[core].append((min(lo, hi), max(lo, hi), lo, hi, (path, cum), direction, cum[-1]))
        self.names = sorted(set(self.street) | set(self.segments))

    @staticmethod
    def _pick(cands, direction):
        """cands: [(direction, lon, lat, uncertainty_m)] ->
        (lon, lat, uncertainty_m) | ("ambiguous", [(lon, lat), ...]) | None."""
        if direction:
            cands = [c for c in cands if c[0] in (None, direction)]
        if not cands:
            return None
        if spread_m([c[1] for c in cands], [c[2] for c in cands]) > AMBIGUOUS_SPREAD_M:
            return "ambiguous", [(c[1], c[2], c[3]) for c in cands]
        return float(np.mean([c[1] for c in cands])), float(np.mean([c[2] for c in cands])), max(c[3] for c in cands)

    def _exact(self, number, core, direction):
        return self._pick([(d, lon, lat, 10.0) for d, lon, lat in self.exact.get((number, core), [])], direction)

    def _interpolate(self, number, core, direction):
        cands = []
        for lo, hi, a, b, (path, cum), d, length in self.segments.get(core, []):
            if lo <= number <= hi:
                frac = 0.5 if a == b else (number - a) / (b - a)
                t = np.clip(frac, 0, 1) * cum[-1]
                cands.append((d, float(np.interp(t, cum, path[:, 0])), float(np.interp(t, cum, path[:, 1])),
                              round(max(length / 2, 10.0), 1)))
        return self._pick(cands, direction)

    def _nearby(self, number, core, direction):
        pts = [p for p in self.street.get(core, []) if direction is None or p[1] in (None, direction)]
        if not pts:
            return None
        gap = min(abs(p[0] - number) for p in pts)
        if gap > self.max_gap:
            return None
        return self._pick([(d, lon, lat, 10.0 * gap) for n, d, lon, lat in pts if abs(n - number) == gap], direction)

    def _try(self, addr: Address, cores, suffix: str):
        for method, fn in (("address_point", self._exact), ("street_range", self._interpolate),
                           ("address_point_nearby", self._nearby)):
            for core in cores:
                hit = fn(addr.number, core, addr.direction)
                if hit and hit[0] == "ambiguous":
                    return np.nan, np.nan, "ambiguous_direction", core, np.nan, method + suffix, hit[1]
                if hit:
                    return hit[0], hit[1], method + suffix, core, hit[2], None, None
        return None

    def geocode(self, addr: Address | None):
        """-> (lon, lat, method, matched_street_core, uncertainty_m, ambiguous_method, candidates).
        For 'ambiguous_direction', candidates lists the (lon, lat, uncertainty_m) options to resolve."""
        if addr is None:
            return np.nan, np.nan, "no_address", None, np.nan, None, None
        hit = self._try(addr, addr.cores, "")
        if hit:
            return hit
        fuzzy = []
        for core in addr.cores:
            fuzzy += difflib.get_close_matches(core, self.names, n=2, cutoff=0.88)
        hit = self._try(addr, list(dict.fromkeys(fuzzy)), "_fuzzy_street")
        return hit if hit else (np.nan, np.nan, "not_found", None, np.nan, None, None)


def spread_m(lon, lat) -> float:
    """Diameter-like spread of a point set: twice the largest distance from the centroid."""
    xy = to_xy(lon, lat)
    return float(np.hypot(*(xy - xy.mean(axis=0)).T).max() * 2)


def hydrant_features(lon, lat, hydrants: pd.DataFrame, k: int, radius_m: float) -> pd.DataFrame:
    """Per-point features from the public hydrant layer. NaN coordinates give NaN features."""
    h = hydrants.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    tree = cKDTree(to_xy(h.lon, h.lat))
    plane = h["PressurePlane"].fillna("UNKNOWN").to_numpy()
    flush = h["FlushProgram"].isin(["M", "A"]).to_numpy(float)
    year = pd.to_datetime(h["AcceptanceDate"], unit="ms", errors="coerce").dt.year.to_numpy(float)
    pts = to_xy(lon, lat)
    ok = ~np.isnan(pts).any(axis=1)
    out = pd.DataFrame(index=range(len(pts)), columns=["pressure_plane", "plane_vote_share", "nearest_hydrant_m",
                                                       "hydrants_in_radius", "flush_program_share",
                                                       "hydrant_median_year"], dtype=object)
    if ok.any():
        dist, idx = tree.query(pts[ok], k=k)
        votes = [Counter(plane[i]).most_common(1)[0] for i in idx]
        nb = tree.query_ball_point(pts[ok], r=radius_m)
        out.loc[ok, "pressure_plane"] = [v[0] for v in votes]
        out.loc[ok, "plane_vote_share"] = [round(v[1] / k, 3) for v in votes]
        out.loc[ok, "nearest_hydrant_m"] = np.round(dist[:, 0], 1)
        out.loc[ok, "hydrants_in_radius"] = [len(i) for i in nb]
        out.loc[ok, "flush_program_share"] = [round(float(flush[i].mean()), 3) if i else np.nan for i in nb]
        out.loc[ok, "hydrant_median_year"] = [float(np.nanmedian(year[i])) if i and np.isfinite(year[i]).any() else np.nan
                                              for i in nb]
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c])
    return out


# Phase II of the Davis Upper Pressure Plane expansion (completed ~Oct 2024) moved the area bounded
# by these streets from the lower to the upper plane. Hydrant planes are current, so sites inside
# the box may have been in a different plane before then.
DAVIS_PHASE2_STREETS = {"east": {"N COOPER ST", "S COOPER ST"}, "west": {"N FIELDER RD", "S FIELDER RD"},
                        "south": {"W ARKANSAS LN", "E ARKANSAS LN"}, "north": {"W SANFORD ST", "E SANFORD ST"}}


def davis_phase2_box(roads: pd.DataFrame) -> dict | None:
    def verts(names):
        segs = roads[roads.fullname.isin(names)].paths
        return np.vstack([np.array(p[0]) for p in segs if isinstance(p, list) and p]) if len(segs) else None

    v = {side: verts(n) for side, n in DAVIS_PHASE2_STREETS.items()}
    if any(x is None for x in v.values()):
        return None
    east, west = np.median(v["east"][:, 0]), np.median(v["west"][:, 0])
    south, north = np.median(v["south"][:, 1]), np.median(v["north"][:, 1])
    for _ in range(2):    # refine each side using only the stretch between the other two sides
        east = np.median(v["east"][(v["east"][:, 1] >= south) & (v["east"][:, 1] <= north), 0])
        west = np.median(v["west"][(v["west"][:, 1] >= south) & (v["west"][:, 1] <= north), 0])
        south = np.median(v["south"][(v["south"][:, 0] >= west) & (v["south"][:, 0] <= east), 1])
        north = np.median(v["north"][(v["north"][:, 0] >= west) & (v["north"][:, 0] <= east), 1])
    return {"lon_west": float(west), "lon_east": float(east), "lat_south": float(south), "lat_north": float(north)}


def in_box(lon, lat, box: dict | None) -> np.ndarray:
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if box is None:
        return np.zeros(len(lon), bool)
    return (lon >= box["lon_west"]) & (lon <= box["lon_east"]) & (lat >= box["lat_south"]) & (lat <= box["lat_north"])


def distance_km(lon, lat, lon0: float, lat0: float) -> np.ndarray:
    a, b = to_xy(lon, lat), to_xy([lon0], [lat0])
    return np.round(np.hypot(*(a - b).T) / 1000, 3)
