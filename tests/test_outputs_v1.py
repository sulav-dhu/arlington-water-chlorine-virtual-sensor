"""Integration checks on the built v1 outputs. Run `uv run python -m chlorine.cleaning.pipeline` first.

Golden numbers were verified against the raw export (sha256 7d2377bb...) by an independent
recomputation; if the export or a cleaning rule changes, update them deliberately."""
import json

import pandas as pd
import pytest

from chlorine.cleaning.addresses import contains_phone
from chlorine.config import CleaningConfig

OUT = CleaningConfig().out_dir
pytestmark = pytest.mark.skipif(not (OUT / "chlorine_v1.parquet").exists(), reason="v1 outputs not built")


@pytest.fixture(scope="module")
def data():
    return {"chl": pd.read_parquet(OUT / "chlorine_v1.parquet"), "meas": pd.read_parquet(OUT / "measurements.parquet"),
            "loc": pd.read_csv(OUT / "locations.csv"), "ev": pd.read_csv(OUT / "events.csv"),
            "manifest": json.loads((OUT / "manifest.json").read_text())}


def test_golden_counts(data):
    assert data["manifest"]["raw_sha256_matches_expected"]
    assert len(data["meas"]) == 58_492
    assert len(data["chl"]) == 30_703
    assert data["chl"].location_id.nunique() == 221


def test_modeling_table_invariants(data):
    chl = data["chl"]
    assert not chl.duplicated(["location_id", "sampled_at"]).any()
    assert chl.value_mg_l.dropna().between(0, 6).all()
    assert chl.project.isin(["Monthly Bact", "Monthly Cl"]).all()
    assert chl.loc_confidence.isin(["high", "medium"]).all()
    assert chl.site_code.notna().all()
    in_scope = data["meas"][data["meas"].in_scope_v1]
    assert set(in_scope.row_id) == set(chl.row_id)


def test_no_personal_data_in_outputs(data):
    for name in ("chl", "meas", "loc"):
        df = data[name]
        assert not {"SampleAlias", "Analyst", "analyst_norm"} & set(df.columns)
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                assert not df[col].dropna().astype(str).map(contains_phone).any(), f"{name}.{col}"


def test_known_resolutions(data):
    m = data["meas"]
    cl = m[(m.analyte == "Chlorine residual") & (m.analysis == "Chlres-field")]
    assert set(cl[cl.site_code == "W005"].location_id) == {"W045"}                 # miscode
    assert set(cl[cl.site_code == "W052"].location_id) == {"PLANT-PB-LAB"}         # override
    assert set(cl[cl.sample_name == "NO39"].site_code) == {"N039"}                 # letter O typo
    sw049 = cl[(cl.site_code == "SW049") & (cl.loc_confidence == "conflict")]
    assert len(sw049) == 19 and set(sw049.loc_conflict_with) == {"W066"}           # March 2024 conflict
    linked = cl[(cl.location_id == "W045") & (cl.loc_method == "address_named_link") & cl.in_scope_v1]
    assert len(linked) >= 28                                                       # logged by address, not code


def test_locations_and_features(data):
    loc = data["loc"]
    rt = loc[(loc.location_type == "routine") & (loc.n_chlorine_in_scope > 0)]
    assert rt.pressure_plane.notna().mean() > 0.98
    e046 = loc[loc.location_id == "E046"].iloc[0]                                   # written E, not W, directional
    assert e046.lon > -97.08 and e046.geocode_method.endswith("region_disambiguated")
    w = loc.set_index("location_id")
    assert w.loc["W065", "place_group"] == w.loc["W066", "place_group"]
    assert set(loc[loc.high_frequency_site == True].location_id) == {"W045", "SE035", "W065", "W066", "SW049", "SW050", "SW051"}


def test_events(data):
    ev = data["ev"]
    jk = ev[(ev.plant == "JK") & (ev.source == "inferred_from_plant_sampling_gaps")]
    assert len(jk) == 7
    assert (jk.confidence == "externally_confirmed").sum() == 1
    assert jk[jk.start == "2017-02-02"].confidence.iloc[0] == "externally_confirmed"
