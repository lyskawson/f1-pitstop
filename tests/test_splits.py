from pathlib import Path

import pandas as pd
import pytest

from pitstop.contracts import RAW_LAPS_DTYPES
from pitstop.ingest import D3_CLEANING, coverage, read_json
from pitstop.splits import apply_exclusions, build_split_manifest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "raw"


def event(season, round_number, *, special=False):
    return {
        "race_id": f"{season}_{round_number:02d}_R",
        "season": season,
        "round_number": round_number,
        "session": "R",
        "special_tyre_rule": special,
    }


def test_fixed_split_bounds_and_order():
    events = [event(2025, n) for n in (24, 13, 12, 1)] + [event(y, 7) for y in (2024, 2022, 2023)]
    manifest = build_split_manifest(events)
    assert list(manifest["races"].items()) == [
        ("2022_07_R", "train"),
        ("2023_07_R", "train"),
        ("2024_07_R", "train"),
        ("2025_01_R", "val"),
        ("2025_12_R", "val"),
        ("2025_13_R", "test"),
        ("2025_24_R", "test"),
    ]
    assert list(build_split_manifest(reversed(events))["races"]) == list(manifest["races"])
    assert build_split_manifest(reversed(events)) == manifest


@pytest.mark.parametrize("season,round_number", [(2021, 1), (2025, 25), (2026, 1)])
def test_event_outside_v1_is_rejected(season, round_number):
    with pytest.raises(ValueError, match="outside fixed splits"):
        build_split_manifest([event(season, round_number)])


def test_duplicate_and_mismatched_event_id_are_rejected():
    first = event(2022, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        build_split_manifest([first, first])
    with pytest.raises(ValueError, match="race identity"):
        build_split_manifest([{**first, "race_id": "2022_02_R"}])


def test_committed_split_covers_the_accepted_schedule():
    manifest = read_json(ROOT / "manifests" / "splits.json")
    raw = read_json(ROOT / "manifests" / "data.json")
    scheduled = [
        event(int(race[:4]), int(race[5:7])) for race in raw["coverage"]["requested_races"]
    ]
    assert manifest == build_split_manifest(scheduled)
    assert list(manifest["races"]) == sorted(manifest["races"])


def raw_race(race_id, *, wet=False, red=False):
    rows = []
    drivers = {
        "1": range(1, 11),
        "2": [1, 2, 7, 8],
        "3": range(1, 11),
        "4": [1, 5, 6],
    }
    for driver, numbers in drivers.items():
        for lap in numbers:
            rows.append(
                {
                    "race_id": race_id,
                    "DriverNumber": driver,
                    "LapNumber": lap,
                    "Time": 100.0 * lap,
                    "LapTime": 100.0,
                    "PitInTime": float("nan"),
                    "PitOutTime": float("nan"),
                    "Stint": 1.0,
                    "Compound": (
                        "INTERMEDIATE"
                        if wet and driver == "1" and lap == 1
                        else None
                        if driver == "3" and lap <= 2
                        else "MEDIUM"
                    ),
                    "TyreLife": (
                        float("nan") if driver in {"1", "3"} and lap <= int(driver) else float(lap)
                    ),
                    "Position": float(driver),
                    "TrackStatus": "145" if red and driver == "1" and lap == 1 else "4",
                    "IsAccurate": False,
                    "FastF1Generated": False,
                }
            )
    return pd.DataFrame(rows).astype(RAW_LAPS_DTYPES | {"Compound": "string"})


def test_d3_exclusions_are_identical_across_splits_and_leave_assignments_fixed():
    events = [
        event(2023, 18, special=True),
        event(2025, 8, special=True),
        event(2025, 23, special=True),
    ]
    events += [event(2022, 1), event(2025, 12), event(2025, 13)]
    laps = pd.concat(
        [
            raw_race(e["race_id"], wet=not e["special_tyre_rule"], red=not e["special_tyre_rule"])
            for e in events
        ],
        ignore_index=True,
    )
    before = laps.copy(deep=True)
    manifest = build_split_manifest(events)
    retained, audit = apply_exclusions(laps, events, manifest)
    assert set(manifest["races"]) == {e["race_id"] for e in events}
    assert manifest == build_split_manifest(events)
    assert set(retained.race_id) == {e["race_id"] for e in events if e["special_tyre_rule"]}
    assert set(retained.DriverNumber) == {"1", "4"}
    assert retained.special_tyre_rule.all()
    assert set(retained.split) == {"train", "val", "test"}
    assert retained.TrackStatus.eq("4").all()
    for split in ("train", "val", "test"):
        assert {e["reason_code"] for e in audit["exclusions"] if e["split"] == split} == {
            "wet",
            "red_flag",
            "large_gap",
            "missing_Compound",
            "missing_TyreLife",
        }
    _, exclusions = coverage(laps, sorted(events, key=lambda e: e["race_id"]), D3_CLEANING)
    assert [
        {k: v for k, v in item.items() if k not in {"split", "special_tyre_rule"}}
        for item in audit["exclusions"]
    ] == exclusions
    _, reordered = apply_exclusions(laps.iloc[::-1], events[::-1], manifest)
    assert reordered == audit
    pd.testing.assert_frame_equal(laps, before)


def test_verified_fixture_can_use_a_split_manifest_with_other_races():
    events = read_json(FIXTURES / "events.json")
    manifest = build_split_manifest([*events, event(2025, 13)])
    laps = pd.read_parquet(FIXTURES / "laps.parquet")
    retained, audit = apply_exclusions(laps, events, manifest)
    assert set(retained.split) == {"train"}
    assert {e["race_id"] for e in audit["races"]} == set(laps.race_id)
    assert manifest["races"]["2025_13_R"] == "test"
    with pytest.raises(ValueError, match="coverage or fixed split"):
        apply_exclusions(laps, events[:1], manifest)
