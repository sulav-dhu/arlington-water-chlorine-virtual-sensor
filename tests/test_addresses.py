import pytest

from chlorine.cleaning.addresses import parse_address, same_place, street_core


def same(a, b):
    return same_place(parse_address(a), parse_address(b), 30, 0.85)


@pytest.mark.parametrize("a,b", [
    ("1234 Maple Grove Dr.", "1234MAPLEGROVEDR"),                 # spaced vs squashed sample name
    ("4321 Old Mill Cemetery RD", "4321OLDMILLCEMETARY"),             # misspelling, missing suffix
    ("2468 Bramble Cir N", "2468 N Bramble CIR"),            # post- vs pre-directional
    ("410 W Main ST", "410W.MAINST"),                              # glued suffix, punctuation
    ("1490 S. Oak ST", "1490S.OAKST"),                             # short street name
    ("3579 Lester Way DR", "3579LESTERWAY"),                       # two suffixes
    ("1357 Walter Lowe RD", "1357WALTERLOWERD"),
    ("2017 Cedar Park BLVD", "2019 Cedar Park BLVD"),        # neighbouring lots
    ("1003 Kensington CT", "1003KENSINTONCT"),
])
def test_same_place_variants(a, b):
    assert same(a, b)


@pytest.mark.parametrize("a,b", [
    ("1500 W Example PWY", "1500 E Example PWY"),                  # opposite sides of the city
    ("2400 NE Green Oaks BLVD", "1600 NE Green Oaks BLVD"),        # same street, far apart
    ("5900 Sample DR", "5700 Sample DR"),                          # 200 house numbers apart
    ("1234 Maple Grove DR", "1234 Sublett RD"),
])
def test_different_places(a, b):
    assert not same(a, b)


def test_missing_directional_is_compatible():
    assert same("1500 Example PKWY", "1500 E Example PWY")


def test_normalizations():
    assert parse_address("1500 W 2nd St").cores == ("SECOND",)          # city GIS spells ordinals out
    assert parse_address("1500 W 2nd St").direction == "W"
    assert parse_address("4000 SW Green Oaks BLDV").display == "4000 SW GREEN OAKS BLVD"
    assert parse_address("1111 1111 Example LN").display == "1111 EXAMPLE LN"
    a = parse_address("1901-B Lakewood DR")
    assert (a.number, a.cores) == (1901, ("LAKEWOOD",))
    assert parse_address("7001 US HWY 287").cores == ("HWY287",)
    assert street_core("US 287 FWY") == ("HWY287", None)
    assert street_core("US 287 BUS HWY") == ("HWY287BUS", None)          # business route is another road
    assert street_core("W EXAMPLE PWY") == ("EXAMPLE", "W")


@pytest.mark.parametrize("text", ["landfill", "SE035", "PB Lab", "", None, "Harwell EST"])
def test_not_addresses(text):
    assert parse_address(text) is None


def test_phone_numbers_never_enter_an_address():
    a = parse_address("1234 Elm Ct, (817) 555-0100")
    assert a is not None and "555" not in a.display and "0100" not in a.display
