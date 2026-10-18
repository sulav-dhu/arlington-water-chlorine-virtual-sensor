import pandas as pd
import pytest

from chlorine.cleaning.addresses import parse_address
from chlorine.cleaning.locations import LocationResolver, classify_sample, load_facilities, routine_code
from chlorine.config import CleaningConfig

CFG = CleaningConfig()
FACILITIES = load_facilities(CFG.reference_dir / "facilities.csv")


@pytest.mark.parametrize("name,code", [("W45", "W045"), ("W045", "W045"), ("NO39", "N039"), ("SE035", "SE035"),
                                       ("EST001", None), ("EP001", None), ("JKTAP", None), ("1200BALLPARKWAY", None)])
def test_routine_code(name, code):
    assert routine_code(name) == code


@pytest.mark.parametrize("name,project,cls,fid", [
    ("JKTAP", "Weekly Tap", "facility", "PLANT-JK-FINISHED"),
    ("EP001/PBTAP", "Quarterly DBP Sites", "facility", "PLANT-PB-FINISHED"),
    ("JKF15", "JK Special Projects", "facility", "PLANT-JK-OTHER"),
    ("EST009", "Towers", "facility", "TANK-MATLOCK"),
    ("SETOWERHYDRANT", "Special", "facility", "TANK-SOUTHEAST"),
    ("TIERREVERDETOWER", "Special", "facility", "TANK-TIERRA-VERDE"),
    ("100MATLOCKMEADOWDR", "Monthly Bact", "address_named", None),       # starts like a tank name, is an address
    ("W052", "Monthly Cl", "facility", "PLANT-PB-LAB"),                 # site override
    ("PILOTPLANT", "Pilot Plant Project", "other", None),
])
def test_classify(name, project, cls, fid):
    got = classify_sample(name, project, None, FACILITIES, {"W052": "PLANT-PB-LAB"})
    assert got[0] == cls and got[2] == fid


def test_complaint_alias_is_never_used_as_address():
    got = classify_sample("JOHNDOE", "Health Concern", "100 Main St", FACILITIES, {})
    assert got[0] == "other"


def _routine_rows(spec):
    """spec: list of (code, alias, date, count)."""
    rows = []
    for code, alias, date, n in spec:
        for i in range(n):
            rows.append({"site_code": code, "sampled_at": pd.Timestamp(date) + pd.Timedelta(days=i),
                         "alias_address": parse_address(alias) if alias else None})
    return pd.DataFrame(rows)


def test_resolve_routine_rules():
    rows = _routine_rows([
        ("A001", "100 Main St", "2016-01-01", 20),
        ("A001", "100 Main", "2016-03-01", 1),           # spelling variant, same place
        ("A001", "500 Oak St", "2016-04-01", 1),         # another code's address -> conflict
        ("A001", None, "2016-05-01", 2),                 # no alias -> inherits code location
        ("B002", "500 Oak St", "2016-01-01", 10),
        ("C003", "100 Main St", "2016-02-01", 1),        # rare code at A001's address -> miscode
        ("D004", "900 Elm Dr", "2015-01-01", 10),
        ("D004", "300 Pine Ln", "2020-01-01", 2),        # years later, elsewhere -> code re-use
        ("D004", "910 Elm Dr", "2015-06-01", 2),         # same place as anchor
    ])
    res = LocationResolver(CFG, [])
    out = res.resolve_routine(rows).join(rows)
    by = lambda code: out[out.site_code == code]
    assert set(by("A001").location_id) == {"A001"}
    assert by("A001").loc_confidence.value_counts().to_dict() == {"high": 23, "conflict": 1}
    assert by("A001").loc[by("A001").loc_confidence == "conflict", "loc_conflict_with"].iloc[0] == "B002"
    assert by("C003").location_id.tolist() == ["A001"]
    assert by("C003").loc_method.tolist() == ["reassigned_by_address"]
    d = by("D004")
    assert set(d.location_id) == {"D004.1", "D004.2"}
    assert d[d.alias_address.map(lambda a: a.number == 300)].location_id.unique().tolist() == ["D004.2"]


def test_place_groups_link_neighbouring_codes():
    rows = _routine_rows([("W065", "2017 Cedar Park BLVD", "2016-01-01", 10),
                          ("W066", "2019 Cedar Park BLVD", "2021-01-01", 10),
                          ("W070", "2017 Other Park BLVD", "2021-01-01", 10)])
    res = LocationResolver(CFG, [])
    res.resolve_routine(rows)
    g = res.place_groups()
    assert g["W065"] == g["W066"] == "W065"
    assert g["W070"] == "W070"
