"""The results map: its numbers match the saved evaluation, and it holds no customer data."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from chlorine.cleaning.addresses import contains_phone
from chlorine.config import ROOT, EvalConfig
from chlorine.reporting.sampling_map import MODELS_CSV, OUT, build_payload, check_against_results, load_registry

DATA_READY = (ROOT / "data" / "eval" / "test" / "predictions.parquet").exists()


def test_registry_lists_baselines_and_an_empty_slot():
    reg = load_registry(MODELS_CSV).set_index("code")
    assert reg.loc["B5p", "val_predictions"]
    assert reg.loc["NEXT", "val_predictions"] == "" and reg.loc["NEXT", "test_predictions"] == ""


def test_output_is_in_reports():
    assert OUT.relative_to(ROOT).parts[0] == "reports"


@pytest.mark.skipif(not DATA_READY, reason="needs the private data")
def test_payload_matches_results_and_has_no_private_fields():
    payload = build_payload()
    assert check_against_results(payload) > 0
    assert {m["code"]: m["ready"] for m in payload["models"]}["NEXT"] is False
    n_places = pd.read_parquet(EvalConfig().data_dir / "chlorine_v1.parquet", columns=["place_group"]).place_group.nunique()
    assert len(payload["places"]) == n_places
    assert set(payload["places"][0]) == {"id", "lon", "lat", "codes", "plane", "region", "total"}
    text = json.dumps({"places": payload["places"], "models": payload["models"]})
    assert not contains_phone(text)
