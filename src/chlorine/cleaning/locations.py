"""Location resolution: assign every measurement a physical location_id.

Sample identity in the export is messy: routine site codes have typos ("NO39"), formatting
variants ("W45"), alias addresses that disagree with the code, codes re-used years later at a
new address, plant taps miscoded as routine sites, and NAP/program samples logged under
address-style names ("1234MAPLEGROVECT") instead of codes.

Approach for routine codes (W/SE/SW/E/N/C + number):
  1. Cluster each code's alias addresses into physical places (same street, house numbers
     within a few lots).
  2. The largest cluster is the code's anchor. Every other cluster is classified:
       conflict   - it is another code's anchor (or a plant/tank address): the code and the
                    alias disagree, so the row is kept but excluded from modeling;
       secondary  - a substantial cluster elsewhere: a real second location of the code;
       reuse      - a small cluster far outside the code's active period: code re-used;
       variant    - otherwise a typo / alias error: assigned to the code's location.
  3. A code with only a handful of rows whose anchor is another code's anchor is a miscode
     (W005 = W045) and its rows are reassigned to that location.
  4. Rows without an alias address inherit the code's location active at that date.
Address-named rows are linked to a routine/facility place when they match one; otherwise they
become their own location (NAP-only site or ad-hoc address). Each location also gets a
place_group: locations that are the same physical place (e.g. W065 and W066, two house
numbers apart) share it, so later spatial cross-validation can keep them in the same fold.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd

from ..config import CleaningConfig
from .addresses import Address, contains_phone, parse_address, same_place, street_similarity
from .privacy import COMPLAINT_PROJECTS

ROUTINE_RE = re.compile(r"^(SE|SW|NE|NW|N|E|W|C)O?(\d{1,3})$")   # "NO39" is N039 typed with a letter O
EST_RE = re.compile(r"^EST\d{3}$")
TANK_KEYWORD_RE = re.compile(r"EST|TOWER|TANK")
NAP_PROJECTS = {"Weekly NAP Sites", "Quarterly NAP Sites", "Semi-Annual NAP Sites", "Monthly NAP - Do Not Use"}


def normalize_sample_name(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"\s+", "", regex=True).replace("", pd.NA)


def routine_code(sample_name: str | None) -> str | None:
    if sample_name is None or pd.isna(sample_name):
        return None
    m = ROUTINE_RE.match(sample_name)
    return f"{m.group(1)}{int(m.group(2)):03d}" if m else None


@dataclass
class Facility:
    location_id: str
    location_type: str
    facility: str
    point_type: str
    pattern: re.Pattern
    est_code: str | None
    address: Address | None
    notes: str


def load_facilities(path) -> list[Facility]:
    t = pd.read_csv(path, dtype=str).fillna("")
    return [Facility(r.location_id, r.location_type, r.facility, r.point_type, re.compile(r.name_pattern),
                     r.est_code or None, parse_address(r.address) if r.address else None, r.notes)
            for r in t.itertuples()]


def classify_sample(sn, project, alias, facilities: list[Facility], overrides: dict[str, str]):
    """-> (sample_class, site_code, facility_id, address). sample_class is one of
    routine | facility | address_named | other | unknown."""
    if sn is None or pd.isna(sn):
        return "unknown", None, None, None
    code = routine_code(sn)
    if code:
        if code in overrides:
            return "facility", code, overrides[code], None
        return "routine", code, None, None
    if EST_RE.match(sn):
        fac = next((f for f in facilities if f.est_code == sn), None)
        return ("facility", None, fac.location_id, None) if fac else ("other", None, None, None)
    for f in facilities:
        if f.location_type == "plant" and f.pattern.search(sn):
            return "facility", None, f.location_id, None
    for f in facilities:
        if f.location_type == "tank" and f.pattern.search(sn) and (TANK_KEYWORD_RE.search(sn) or project == "Towers"
                                                                    or f.pattern.fullmatch(sn)):
            return "facility", None, f.location_id, None
    addr = parse_address(sn)
    if addr:
        return "address_named", None, None, addr
    # In complaint projects the alias is a customer contact field, never an address; elsewhere an
    # alias carrying a phone number is a contact field too.
    if project not in COMPLAINT_PROJECTS and alias is not None and not pd.isna(alias) and not contains_phone(alias):
        addr = parse_address(alias)
        if addr:
            return "address_named", None, None, addr
    return "other", None, None, None


@dataclass
class Cluster:
    rep: Address
    rows: list = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    role: str = ""
    location_id: str | None = None
    conflict_with: str | None = None

    @property
    def n(self) -> int:
        return len(self.rows)

    def geo_rep(self) -> Address:
        """Spelling used for geocoding: the most common one that carries a directional, if any."""
        directed = [(n, a) for a, n in self.counts.items() if a.direction]
        return max(directed, key=lambda x: x[0])[1] if directed else self.rep

    def add(self, idx, addr: Address, t: pd.Timestamp):
        self.rows.append(idx)
        self.counts[addr] += 1
        self.first = t if self.first is None or t < self.first else self.first
        self.last = t if self.last is None or t > self.last else self.last


def _time_distance(t: pd.Timestamp, first: pd.Timestamp, last: pd.Timestamp) -> float:
    if first is None:
        return float("inf")
    if first <= t <= last:
        return 0.0
    return min(abs((t - first).days), abs((t - last).days))


def _interval_gap_days(a_first, a_last, b_first, b_last) -> int:
    return max((b_first - a_last).days, (a_first - b_last).days, 0)


def _stable_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha1(text.encode()).hexdigest()[:8].upper()}"


class LocationResolver:
    def __init__(self, cfg: CleaningConfig, facilities: list[Facility]):
        self.cfg = cfg
        self.facilities = {f.location_id: f for f in facilities}
        self.places: list[tuple[str, Address]] = [(f.location_id, f.address) for f in facilities if f.address]
        self.locations: dict[str, dict] = {}
        self.review: list[dict] = []

    def _same(self, a: Address, b: Address) -> bool:
        return same_place(a, b, self.cfg.same_place_max_number_gap, self.cfg.same_street_min_similarity)

    def _best_place(self, addr: Address, anchors: dict, exclude: str) -> str | None:
        """Closest established place (another code's anchor, or a facility) at the same physical
        place as addr: smallest house-number gap, then most similar street."""
        cands = [(code, a.rep) for code, a in anchors.items() if code != exclude] + self.places
        best = None
        for key, rep in cands:
            if self._same(addr, rep):
                score = (abs(addr.number - rep.number), -street_similarity(addr, rep))
                if best is None or score < best[0]:
                    best = (score, key)
        return best[1] if best else None

    # -- routine codes -------------------------------------------------------------------------
    def resolve_routine(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: index = measurement index; columns site_code, sampled_at, alias_address."""
        cfg = self.cfg
        clusters: dict[str, list[Cluster]] = defaultdict(list)
        n_rows = rows.groupby("site_code").size().to_dict()
        keyed = rows[rows.alias_address.notna()]
        for code, g in keyed.groupby("site_code"):
            order = g.alias_address.value_counts().index        # most frequent spelling first
            by_addr = {a: sub for a, sub in g.groupby("alias_address", sort=False)}
            for addr in order:
                target = next((c for c in clusters[code] if self._same(c.rep, addr)), None)
                if target is None:
                    target = Cluster(rep=addr)
                    clusters[code].append(target)
                for idx, t in zip(by_addr[addr].index, by_addr[addr].sampled_at):
                    target.add(idx, addr, t)
        for code in clusters:
            clusters[code].sort(key=lambda c: (-c.n, c.first))
            for c in clusters[code]:
                c.rep = c.counts.most_common(1)[0][0]
            clusters[code][0].role = "anchor"
        anchors = {code: cl[0] for code, cl in clusters.items()}

        # Miscodes: a rarely used code whose anchor is an established code's anchor or a facility.
        established_by_rows = {c: a for c, a in anchors.items() if n_rows[c] >= cfg.location_min_rows}
        miscode_target: dict[str, str] = {}
        for code, a in anchors.items():
            if n_rows[code] < cfg.location_min_rows:
                tgt = self._best_place(a.rep, established_by_rows, exclude=code)
                if tgt:
                    miscode_target[code] = tgt

        # Classify clusters.
        established = {c: a for c, a in anchors.items() if c not in miscode_target}
        for code, cl in clusters.items():
            if code in miscode_target:
                tgt = miscode_target[code]
                tgt_rep = anchors[tgt].rep if tgt in anchors else self.facilities[tgt].address
                for c in cl:            # only clusters at the target's place are reassigned
                    if self._same(c.rep, tgt_rep):
                        c.role, c.conflict_with = "miscode", tgt
                    else:
                        other = self._best_place(c.rep, established, exclude=code)
                        c.role, c.conflict_with = ("conflict", other) if other else ("unresolved", None)
                continue
            for c in cl[1:]:
                other = self._best_place(c.rep, established, exclude=code)
                kept = [k for k in cl if k.role in ("anchor", "secondary", "reuse")]
                if other:
                    c.role, c.conflict_with = "conflict", other
                elif c.n >= cfg.location_min_rows:
                    c.role = "secondary"
                elif all(_interval_gap_days(c.first, c.last, k.first, k.last) >= cfg.code_reuse_min_gap_days
                         for k in kept):
                    c.role = "reuse"
                else:
                    c.role = "variant"
            own = sorted([c for c in cl if c.role in ("anchor", "secondary", "reuse")], key=lambda c: c.first)
            for i, c in enumerate(own):
                c.location_id = code if len(own) == 1 else f"{code}.{i + 1}"
                self._add_location(c.location_id, "routine", site_code=code, address=c.rep, n_rows_keyed=c.n,
                                   role=c.role, first=c.first, last=c.last, geo_address=c.geo_rep())
        for code in set(n_rows) - set(clusters):                  # codes never given an address
            self._add_location(code, "routine", site_code=code, address=None, n_rows_keyed=0, role="code_only")
        # Conflicts and miscodes point at the other code's anchor location id.
        for cl in clusters.values():
            for c in cl:
                if c.conflict_with in anchors:
                    c.conflict_with = anchors[c.conflict_with].location_id

        # Row assignment.
        out = pd.DataFrame(index=rows.index, columns=["location_id", "loc_method", "loc_confidence",
                                                      "loc_conflict_with"], dtype=object)
        own_locs = defaultdict(list)                              # code -> [(location_id, first, last, n)]
        for code, cl in clusters.items():
            for c in cl:
                if c.location_id:
                    own_locs[code].append((c.location_id, c.first, c.last, c.n))

        def nearest(code, t):
            cands = own_locs.get(code)
            if not cands:
                return code if code in self.locations else None
            return min(cands, key=lambda x: (_time_distance(t, x[1], x[2]), -x[3]))[0]

        for code, cl in clusters.items():
            for c in cl:
                for idx in c.rows:
                    t = rows.at[idx, "sampled_at"]
                    if c.location_id:
                        out.loc[idx] = [c.location_id, "alias_address", "high", None]
                    elif c.role == "variant":
                        out.loc[idx] = [nearest(code, t), "alias_variant", "medium", None]
                    elif c.role == "miscode":
                        out.loc[idx] = [c.conflict_with, "reassigned_by_address", "medium", None]
                    elif c.role == "conflict":
                        out.loc[idx] = [nearest(code, t), "alias_conflict", "conflict", c.conflict_with]
                    else:
                        out.loc[idx] = [None, "unresolved", "none", None]
                self.review.append({"site_code": code, "address": c.rep.display, "role": c.role, "n_rows": c.n,
                                    "first": c.first.date(), "last": c.last.date(),
                                    "location_id": c.conflict_with if c.role == "miscode" else c.location_id,
                                    "conflict_with": c.conflict_with if c.role == "conflict" else None,
                                    "spellings": len(c.counts)})

        unkeyed = rows[rows.alias_address.isna()]
        for idx, code, t in zip(unkeyed.index, unkeyed.site_code, unkeyed.sampled_at):
            if code in miscode_target:
                out.loc[idx] = [None, "unresolved", "none", None]
                continue
            n_own = len(own_locs.get(code, [])) or 1
            out.loc[idx] = [nearest(code, t), "code_only", "high" if n_own == 1 else "medium", None]
        return out

    # -- address-named rows -------------------------------------------------------------------
    def resolve_address_named(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: columns address (Address), sampled_at, project. Call after resolve_routine and
        add_facilities so routine and facility places are registered."""
        cfg = self.cfg
        by_number = defaultdict(list)
        for lid, rec in self.locations.items():
            if rec["address"] is not None:
                by_number[rec["address"].number].append(lid)

        def link(addr: Address, t):
            best = None
            for num in range(addr.number - cfg.same_place_max_number_gap, addr.number + cfg.same_place_max_number_gap + 1):
                for lid in by_number.get(num, []):
                    rec = self.locations[lid]
                    sim = street_similarity(addr, rec["address"])
                    if sim < cfg.same_street_min_similarity:
                        continue
                    key = (abs(num - addr.number), -sim, _time_distance(t, rec.get("first"), rec.get("last")))
                    if best is None or key < best[0]:
                        best = (key, lid)
            return best

        out = pd.DataFrame(index=rows.index, columns=["location_id", "loc_method", "loc_confidence",
                                                      "loc_conflict_with", "address_offset"], dtype=object)
        unmatched = []
        for idx, addr, t in zip(rows.index, rows.address, rows.sampled_at):
            hit = link(addr, t)
            if hit:
                (gap, sim, _), lid = hit
                out.loc[idx] = [lid, "address_named_link", "high" if gap == 0 and -sim >= 0.95 else "medium", None, gap]
            else:
                unmatched.append(idx)

        # Unmatched addresses become their own locations: same house number + similar street.
        groups: list[tuple[Address, list]] = []
        for addr, sub in rows.loc[unmatched].groupby("address", sort=False):
            g = next((g for g in groups if g[0].number == addr.number
                      and street_similarity(g[0], addr) >= cfg.same_street_min_similarity), None)
            if g is None:
                groups.append((addr, list(sub.index)))
            else:
                g[1].extend(sub.index)
        for _, idxs in groups:
            projects = set(rows.loc[idxs, "project"])
            ltype = "nap_site" if projects & NAP_PROJECTS else "adhoc"
            spellings = Counter(rows.loc[idxs, "address"])
            rep = spellings.most_common(1)[0][0]
            directed = [(n, a) for a, n in spellings.items() if a.direction]
            geo_rep = max(directed, key=lambda x: x[0])[1] if directed else rep
            lid = _stable_id("NAP" if ltype == "nap_site" else "ADH", f"{rep.number}|{sorted(rep.cores)[0]}")
            times = rows.loc[idxs, "sampled_at"]
            self._add_location(lid, ltype, site_code=None, address=rep, n_rows_keyed=len(idxs), role="address_named",
                               first=times.min(), last=times.max(), geo_address=geo_rep)
            out.loc[idxs, ["location_id", "loc_method", "loc_confidence", "loc_conflict_with", "address_offset"]] = \
                [lid, "address_named_own", "high", None, 0]
        return out

    # -- bookkeeping ----------------------------------------------------------------------------
    def _add_location(self, lid, ltype, site_code, address, n_rows_keyed, role, first=None, last=None,
                      geo_address=None):
        self.locations[lid] = {"location_id": lid, "location_type": ltype, "site_code": site_code,
                               "address": address, "geo_address": geo_address or address,
                               "n_rows_keyed": n_rows_keyed, "resolution_role": role, "first": first, "last": last}

    def add_facilities(self):
        for f in self.facilities.values():
            self._add_location(f.location_id, f.location_type, site_code=None, address=f.address,
                               n_rows_keyed=0, role="facility")
            self.locations[f.location_id].update(facility=f.facility, point_type=f.point_type, notes=f.notes)

    def place_groups(self) -> dict[str, str]:
        """Union locations that are the same physical place. Ad-hoc addresses are attached to a
        non-ad-hoc place but never used as links, so chains of nearby complaints cannot merge
        two distinct sites."""
        core_ids = [lid for lid, r in self.locations.items() if r["address"] is not None and r["location_type"] != "adhoc"]
        parent = {lid: lid for lid in self.locations}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        by_number = defaultdict(list)
        for lid in core_ids:
            by_number[self.locations[lid]["address"].number].append(lid)
        gap = self.cfg.same_place_max_number_gap
        for lid in core_ids:
            a = self.locations[lid]["address"]
            for num in range(a.number - gap, a.number + gap + 1):
                for other in by_number.get(num, []):
                    if other != lid and self._same(a, self.locations[other]["address"]):
                        parent[find(lid)] = find(other)
        groups = defaultdict(list)
        for lid in self.locations:
            groups[find(lid)].append(lid)
        label = {lid: min(members) for members in groups.values() for lid in members}
        for lid, r in self.locations.items():
            if r["location_type"] == "adhoc" and r["address"] is not None:
                a = r["address"]
                near = [o for num in range(a.number - gap, a.number + gap + 1) for o in by_number.get(num, [])
                        if self._same(a, self.locations[o]["address"])]
                if near:
                    label[lid] = label[min(near, key=lambda o: abs(self.locations[o]["address"].number - a.number))]
        return label
