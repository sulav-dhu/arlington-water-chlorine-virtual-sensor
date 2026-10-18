"""Street-address normalization and fuzzy matching.

Addresses appear in three shapes:
  * spaced LIMS aliases:      "1234 Maple Grove Dr."
  * squashed sample names:    "1234MAPLEGROVEDR", "4321OLDMILLCEMETARY"
  * public GIS addresses:     "1234 MAPLE GROVE DR"
Each is reduced to an Address(number, cores, direction): the house number, candidate "street
cores" (street name without directionals, suffix, spaces or punctuation) and the written
directional, if any. A squashed name has no token boundaries, so it yields several candidates
(with/without a trailing suffix or leading directional); matching takes the best-scoring pair.

Directionals are not part of the core because aliases often omit them, but they are not ignored:
e.g. "1500 W EXAMPLE PWY" and "1500 E EXAMPLE PWY" can be miles apart, so two addresses
with different written directionals are never the same place.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from functools import lru_cache

SUFFIXES = {
    "DRIVE": "DR", "DR": "DR", "STREET": "ST", "ST": "ST", "ROAD": "RD", "RD": "RD",
    "COURT": "CT", "CT": "CT", "LANE": "LN", "LN": "LN", "TRAIL": "TRL", "TRL": "TRL", "TR": "TRL",
    "BOULEVARD": "BLVD", "BLVD": "BLVD", "BLV": "BLVD", "AVENUE": "AVE", "AVE": "AVE",
    "PARKWAY": "PKWY", "PKWY": "PKWY", "CIRCLE": "CIR", "CIR": "CIR", "PLACE": "PL", "PL": "PL",
    "PLC": "PL", "HIGHWAY": "HWY", "HWY": "HWY", "FREEWAY": "FWY", "FWY": "FWY", "TERRACE": "TER",
    "TER": "TER", "WAY": "WAY", "LOOP": "LOOP", "COVE": "CV", "CV": "CV", "POINT": "PT", "PT": "PT",
    "CROSSING": "XING", "XING": "XING",
    # Abbreviations used by the city GIS ("GRANTS PWY") and typos seen in the export ("BLDV").
    "PWY": "PKWY", "PKY": "PKWY", "BLDV": "BLVD", "BVLD": "BLVD", "DRV": "DR", "CRT": "CT",
}
# The city GIS spells ordinals out ("W SECOND ST"); the export uses numerals ("W 2nd ST").
ORDINALS = {"1ST": "FIRST", "2ND": "SECOND", "3RD": "THIRD", "4TH": "FOURTH", "5TH": "FIFTH",
            "6TH": "SIXTH", "7TH": "SEVENTH", "8TH": "EIGHTH", "9TH": "NINTH", "10TH": "TENTH"}
DIRECTIONS = {"N": "N", "NORTH": "N", "S": "S", "SOUTH": "S", "E": "E", "EAST": "E",
              "W": "W", "WEST": "W", "NE": "NE", "NW": "NW", "SE": "SE", "SW": "SW"}
_HIGHWAY_WORDS = {"US", "HWY", "HIGHWAY", "FWY", "FREEWAY", "SH", "IH", "I", "FM", "STATE", "BUS", "FRONTAGE", "RD"}
# Longest first so "DRIVE" is stripped before "DR" from squashed names.
_SQUASHED_SUFFIXES = sorted(SUFFIXES, key=len, reverse=True)
# Only abbreviations: full words like NORTH/WEST are usually part of a name (NORTHCREST).
_SQUASHED_DIRECTIONS = ["NE", "NW", "SE", "SW", "N", "S", "E", "W"]
_MIN_STEM = 4

PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}")
# Optional unit letter ("1901-B", "7701A"); never a directional ("401W MAIN ST").
_NUMBER_RE = re.compile(r"^\s*(\d{1,6})(?:\s*-?\s*([A-DF-MO-RT-VX-Z])(?![A-Z]))?\s*(.*)$")


@dataclass(frozen=True)
class Address:
    number: int
    cores: tuple[str, ...]        # candidate street cores
    display: str                  # normalized human-readable form, e.g. "1234 MAPLE GROVE DR"
    direction: str | None = None  # written pre/post directional ("W EXAMPLE PWY" -> "W")


def _clean(text: str) -> str:
    text = re.sub(r"[.,#;:'\"()]", " ", str(text).upper())
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def parse_address(text: str | None) -> Address | None:
    """Parse a spaced or squashed street address; None if it is not address-like."""
    if text is None:
        return None
    m = _NUMBER_RE.match(_clean(PHONE_RE.sub(" ", str(text))))   # never let a phone number become part of an address
    if not m or not m.group(3):
        return None
    number, tokens = int(m.group(1)), m.group(3).split()
    if tokens and tokens[0] == m.group(1):        # "1111 1111 EXAMPLE LN"
        tokens = tokens[1:]
    if not tokens:
        return None
    direction = None
    if len(tokens) == 1:
        cores = _squashed_cores(tokens[0])
        display = f"{number} {tokens[0]}"
    else:
        std = standardize_tokens(tokens)
        core = street_core_from_tokens(std)
        glued = std[-1] not in SUFFIXES.values() and std[-1] not in DIRECTIONS.values()
        cores = (core, *_strip_suffixes(core)) if glued else (core,)   # "W MAINST" -> MAIN
        display = f"{number} {' '.join(std)}"
        direction = street_direction(std)
    cores = tuple(c for c in dict.fromkeys(cores) if c and sum(ch.isalpha() for ch in c) >= 2)
    return Address(number, cores, display, direction) if cores else None


def standardize_tokens(tokens: list[str]) -> list[str]:
    return [ORDINALS.get(t, SUFFIXES.get(t, DIRECTIONS.get(t, t))) for t in tokens]


def street_direction(std_tokens: list[str]) -> str | None:
    """Written directional: leading ("W EXAMPLE PWY") or trailing ("BRAMBLE CIR N")."""
    if len(std_tokens) > 1 and std_tokens[0] in DIRECTIONS.values():
        return std_tokens[0]
    if len(std_tokens) > 1 and std_tokens[-1] in DIRECTIONS.values():
        return std_tokens[-1]
    return None


def street_core_from_tokens(tokens: list[str]) -> str:
    """Street core from standardized tokens: drop trailing suffix/directional, leading directional."""
    routes = [t for t in tokens if t.isdigit()]
    if routes and all(t.isdigit() or t in _HIGHWAY_WORDS or t in DIRECTIONS.values() for t in tokens):
        # "US HWY 287", "US 287 FWY", "HWY 287" -> HWY287; the business route is a different road.
        return "HWY" + routes[0] + ("BUS" if "BUS" in tokens else "")
    core = list(tokens)
    while len(core) > 1 and (core[-1] in SUFFIXES.values() or core[-1] in DIRECTIONS.values()):
        core.pop()
    while len(core) > 1 and core[0] in DIRECTIONS.values():
        core.pop(0)
    return re.sub(r"[^A-Z0-9]", "", "".join(core))


def street_core(name: str) -> tuple[str, str | None]:
    """(core, direction) of a bare street name such as a road-centerline 'fullname'."""
    tokens = standardize_tokens(_clean(name).split())
    return (street_core_from_tokens(tokens), street_direction(tokens)) if tokens else ("", None)


def _squashed_cores(word: str) -> tuple[str, ...]:
    """Candidate cores for a squashed street string like 'NEGREENOAKSBLVD'."""
    word = re.sub(r"[^A-Z0-9]", "", word)
    out = []
    for s in (word, *_strip_suffixes(word)):
        out.append(s)
        for d in _SQUASHED_DIRECTIONS:
            if s.startswith(d) and len(s) - len(d) >= _MIN_STEM:
                out.append(s[len(d):])
                break
    return tuple(dict.fromkeys(out))


def _strip_suffixes(word: str) -> list[str]:
    """Up to two trailing suffix strips: LESTERWAYDR -> LESTERWAY -> LESTER.

    The first strip may leave a short name (OAKST -> OAK); a second strip must leave at least
    _MIN_STEM characters so names are not over-stripped (LESTER -/-> LES)."""
    stems = []
    for i in range(2):
        for suf in _SQUASHED_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= (3 if i == 0 else _MIN_STEM):
                word = word[: -len(suf)]
                stems.append(word)
                break
        else:
            break
    return stems


def street_similarity(a: Address, b: Address) -> float:
    return max(_ratio(x, y) for x in a.cores for y in b.cores)


@lru_cache(maxsize=None)
def _ratio(x: str, y: str) -> float:
    return 1.0 if x == y else difflib.SequenceMatcher(None, x, y).ratio()


def directions_compatible(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a == b


def same_place(a: Address, b: Address, max_number_gap: int, min_similarity: float) -> bool:
    return (abs(a.number - b.number) <= max_number_gap and directions_compatible(a.direction, b.direction)
            and street_similarity(a, b) >= min_similarity)


def contains_phone(text: object) -> bool:
    return text is not None and bool(PHONE_RE.search(str(text)))
