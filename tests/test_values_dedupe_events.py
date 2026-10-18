import pandas as pd

from chlorine.cleaning import dedupe, events
from chlorine.cleaning.values import parse_results


def test_parse_results():
    df = pd.DataFrame({
        "result_text": ["3.8", "ND", "Not Analyzed", "7.2", "4", "abc"],
        "MRL": [0.08] * 6,
        "Analyte": ["Chlorine residual"] * 6,
    })
    out = parse_results(df, chlorine_max=6.0)
    assert out.result_status.tolist() == ["detected", "nd", "not_reportable", "detected", "detected", "unparsed"]
    assert out.censored.tolist() == [False, True, False, False, False, False]
    assert pd.isna(out.value[1]) and out.detection_limit[1] == 0.08
    assert out.flag_out_of_range.tolist()[:4] == [False, False, False, True]
    assert out.decimals_observed[0] == 1 and out.decimals_observed[4] == 0


def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = {"row_id": range(n), "Analysis": ["Chlres-field"] * n, "Analyte": ["Chlorine residual"] * n,
            "result_status": ["detected"] * n, "Status": ["Reported"] * n, "sample_name": ["X"] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_duplicates_keep_preferred_copy_and_flag_disagreement():
    t = pd.Timestamp("2020-01-01 09:00")
    df = _rows(location_id=["A", "A", "B", "B"], Sampled=[t] * 4, value=[3.5, 3.5, 2.6, 2.59],
               Project=["Weekly NAP Sites", "Monthly Bact", "Weekly NAP Sites", "Monthly Bact"])
    out = dedupe.mark_duplicates(df, ("Monthly Bact", "Monthly Cl"))
    assert out.is_duplicate.tolist() == [True, False, True, False]      # Monthly Bact copy kept
    assert out.dup_conflict.tolist() == [False, False, True, True]
    assert round(out.dup_value_range[3], 2) == 0.01


def test_plant_subpoints_are_not_duplicates():
    t = pd.Timestamp("2020-01-01 09:00")
    df = _rows(location_id=["PLANT-JK-OTHER"] * 2, sample_name=["JKF3", "JKF5"], Sampled=[t] * 2,
               value=[1.0, 2.0], Project=["JK Special Projects"] * 2)
    assert not dedupe.mark_duplicates(df, ("Monthly Bact",)).is_duplicate.any()


def test_offline_period_inference():
    days = pd.to_datetime(["2020-01-01", "2020-01-08", "2020-03-01", "2020-03-08"])
    m = pd.DataFrame({"location_id": ["PLANT-JK-FINISHED"] * 4 + ["PLANT-PB-FINISHED"] * 4,
                      "Sampled": list(days) * 2})
    m.loc[4:, "Sampled"] = pd.to_datetime(["2020-01-01", "2020-01-20", "2020-02-10", "2020-03-08"])
    p = events.infer_offline_periods(m, min_gap_days=28)
    jk = p[p.plant == "JK"]
    assert len(jk) == 1
    assert jk.start.iloc[0] == pd.Timestamp("2020-01-09") and jk.end.iloc[0] == pd.Timestamp("2020-02-29")
    assert p[p.plant == "PB"].empty                                     # no PB gap reaches 28 days
