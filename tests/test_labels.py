from pathlib import Path

import pandas as pd
import pytest

from pitstop.contracts import KEY, RAW_LAPS_DTYPES
from pitstop.labels import build_labels

FIXTURES = Path(__file__).parent / "fixtures" / "raw"
RACE = "2022_01_R"
DRIVER = "44"


def laps_at(*numbers):
    rows = [
        {
            "race_id": RACE,
            "DriverNumber": DRIVER,
            "LapNumber": lap,
            "Time": lap * 100.0,
            "LapTime": 100.0,
            "PitInTime": float("nan"),
            "PitOutTime": float("nan"),
            "Stint": 1.0,
            "Compound": "MEDIUM",
            "TyreLife": float(lap),
            "Position": 1.0,
            "TrackStatus": "1",
            "IsAccurate": True,
            "FastF1Generated": False,
        }
        for lap in numbers
    ]
    return pd.DataFrame(rows).astype(RAW_LAPS_DTYPES).set_index("LapNumber", drop=False)


def stop(*, compound="HARD", age=1.0, stint=2.0):
    laps = laps_at(18, 19, 20, 21, 22)
    laps.loc[20, "PitInTime"] = 1999.0
    laps.loc[21, ["Compound", "TyreLife", "Stint", "PitOutTime"]] = [
        compound,
        age,
        stint,
        2019.0,
    ]
    return laps


def test_lap_20_stop_labels_only_t19_and_ignores_accuracy_and_lap_time():
    laps = stop()
    laps.loc[19:21, "IsAccurate"] = False
    laps.loc[19:21, "LapTime"] = float("nan")
    labels, audit = build_labels(laps)
    assert list(labels.columns) == [*KEY, "y"]
    assert labels.y.dtype == "int8"
    assert labels.loc[labels.y.eq(1), "lap_number"].tolist() == [19]
    evidence = audit.set_index("lap_number")
    assert evidence.loc[19, "target_lap"] == 20
    assert evidence.loc[19, "reason_code"] == "compound_change"
    assert evidence.loc[20, "y"] == 0
    assert pd.isna(evidence.loc[22, "y"])
    assert evidence.loc[22, "reason_code"] == "missing_target_lap"


@pytest.mark.parametrize("missing,reason", [(20, "missing_target_lap"), (21, "missing_exit_lap")])
def test_exact_successor_gap_does_not_shift_onto_another_lap_or_driver(missing, reason):
    laps = stop()
    different_driver = laps.loc[[missing]].assign(DriverNumber="16")
    different_race = laps.loc[[missing]].assign(race_id="2023_01_R")
    laps = pd.concat([laps.drop(index=missing), different_driver, different_race])
    labels, audit = build_labels(laps)
    candidate = audit.query("race_id == @RACE and driver_number == @DRIVER and lap_number == 19")
    assert candidate.reason_code.item() == reason
    assert candidate.y.isna().all()
    assert not labels.y.eq(1).any()


@pytest.mark.parametrize("generated,reason", [(20, "missing_target_lap"), (21, "missing_exit_lap")])
def test_generated_lap_is_neither_a_sample_nor_successor_evidence(generated, reason):
    laps = stop()
    laps.loc[generated, "FastF1Generated"] = True
    labels, audit = build_labels(laps)
    assert generated not in audit.lap_number.tolist()
    assert not labels.y.eq(1).any()
    assert audit.loc[audit.lap_number.eq(19), "reason_code"].item() == reason


def test_pit_lane_retirement_drops_pre_entry_sample():
    laps = stop().loc[:20]
    labels, audit = build_labels(laps)
    assert labels.lap_number.tolist() == [18]
    evidence = audit.set_index("lap_number")
    assert evidence.loc[19, "reason_code"] == "missing_exit_lap"
    assert evidence.loc[20, "reason_code"] == "missing_target_lap"


@pytest.mark.parametrize("age", [1.0, 7.0, 24.0])
def test_same_compound_change_includes_fresh_and_previously_used_sets(age):
    labels, audit = build_labels(stop(compound="MEDIUM", age=age))
    assert labels.loc[labels.y.eq(1), "lap_number"].tolist() == [19]
    assert audit.loc[audit.lap_number.eq(19), "reason_code"].item() == (
        "same_compound_age_discontinuity"
    )


def test_continuous_penalty_visit_needs_exact_confirmation_despite_stint_increase():
    laps = stop(compound="MEDIUM", age=21.0)
    wrong_lap = {(RACE, DRIVER, 19): "fixtures/control:17"}
    labels, audit = build_labels(laps, confirmed_transits=wrong_lap)
    assert 19 not in labels.lap_number.tolist()
    assert (
        audit.loc[audit.lap_number.eq(19), "reason_code"].item() == "unconfirmed_continuous_tyres"
    )

    source = "fixtures/control:18"
    labels, audit = build_labels(laps, confirmed_transits={(RACE, DRIVER, 20): source})
    assert labels.loc[labels.lap_number.eq(19), "y"].item() == 0
    visit = audit.loc[audit.lap_number.eq(19)].iloc[0]
    assert visit.reason_code == "confirmed_no_change_transit"
    assert visit.transit_source == source


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("TyreLife", float("nan"), "missing_or_unknown_tyre_evidence"),
        ("TyreLife", 0.0, "missing_or_unknown_tyre_evidence"),
        ("Compound", None, "missing_or_unknown_tyre_evidence"),
        ("Compound", "UNKNOWN", "missing_or_unknown_tyre_evidence"),
        ("Stint", 1.0, "unconfirmed_stint_transition"),
        ("Stint", float("nan"), "unconfirmed_stint_transition"),
        ("PitOutTime", float("nan"), "missing_or_invalid_pit_exit"),
        ("PitOutTime", 1997.0, "missing_or_invalid_pit_exit"),
    ],
)
def test_incomplete_or_ambiguous_pit_evidence_is_unlabelled(field, value, reason):
    laps = stop()
    laps.loc[21, field] = value
    labels, audit = build_labels(laps)
    assert audit.loc[audit.lap_number.eq(19), "reason_code"].item() == reason
    assert 19 not in labels.lap_number.tolist()


def test_conflicting_no_change_confirmation_leaves_visit_unlabelled():
    labels, audit = build_labels(
        stop(), confirmed_transits={(RACE, DRIVER, 20): "fixtures/control"}
    )
    assert 19 not in labels.lap_number.tolist()
    assert (
        audit.loc[audit.lap_number.eq(19), "reason_code"].item() == "conflicting_transit_evidence"
    )


def test_confirmation_requires_a_source():
    with pytest.raises(ValueError, match="nonblank source"):
        build_labels(stop(), confirmed_transits={(RACE, DRIVER, 20): " "})


def test_v1_fallback_is_identical_across_split_boundaries():
    races = {"2024_24_R": "train", "2025_12_R": "val", "2025_13_R": "test"}
    laps = pd.concat(
        [
            stop(compound="MEDIUM", age=21.0).assign(race_id=race, split=split)
            for race, split in races.items()
        ]
    )
    labels, audit = build_labels(laps)
    assert len(labels.loc[labels.lap_number.eq(19)]) == 0
    visits = audit.loc[audit.lap_number.eq(19)]
    assert set(visits.race_id) == set(races)
    assert set(visits.reason_code) == {"unconfirmed_continuous_tyres"}


def test_fixture_labels_use_raw_keys_and_input_order_is_irrelevant():
    laps = pd.read_parquet(FIXTURES / "laps.parquet")
    before = laps.copy(deep=True)
    labels, audit = build_labels(laps)
    bahrain = labels.query("race_id == @RACE and driver_number == @DRIVER")
    assert bahrain.loc[bahrain.lap_number.eq(10), "y"].item() == 1
    generated = laps.loc[laps.FastF1Generated, ["race_id", "DriverNumber", "LapNumber"]]
    assert set(generated.itertuples(index=False, name=None)).isdisjoint(
        audit[list(KEY)].itertuples(index=False, name=None)
    )
    shuffled = build_labels(laps.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(labels, shuffled[0])
    pd.testing.assert_frame_equal(audit, shuffled[1])
    pd.testing.assert_frame_equal(laps, before)
