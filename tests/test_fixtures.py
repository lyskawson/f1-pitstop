"""Offline checks for T0 acquisition, evidence and the two real race fixtures."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from pitstop.contracts import RAW_LAPS_DTYPES, validate_raw_laps

FIXTURES = Path(__file__).resolve().parent / "fixtures/raw"


def test_real_fixtures_follow_raw_contract_and_exclusion_rules():
    laps = pd.read_parquet(FIXTURES / "laps.parquet")
    events = json.loads((FIXTURES / "events.json").read_text())
    validate_raw_laps(laps)
    assert set(laps.columns) == set(RAW_LAPS_DTYPES)
    assert set(laps.race_id) == {event["race_id"] for event in events} == {"2022_01_R", "2022_02_R"}
    assert len(events) == 2
    assert not laps.Compound.isin(["INTERMEDIATE", "WET"]).any()
    assert not laps.TrackStatus.str.contains("5", na=False).any()
    for event in events:
        assert event["session"] == "R"
        assert event["season"] == 2022
        assert event["round_number"] == int(event["race_id"][5:7])
        assert isinstance(event["total_laps"], int) and event["total_laps"] > 0
        assert event["total_laps_source"]
        assert isinstance(event["special_tyre_rule"], bool)
        race = laps.loc[laps.race_id.eq(event["race_id"])]
        assert race.LapNumber.max() == event["total_laps"]
        assert race.PitInTime.notna().any() and race.PitOutTime.notna().any()


@pytest.fixture
def spike(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "notebooks/t0_fastf1_spike.py"
    spec = importlib.util.spec_from_file_location("t0_spike", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "data/t0")
    monkeypatch.setattr(module.fastf1.Cache, "enable_cache", Mock())
    monkeypatch.setattr(module.fastf1.Cache, "offline_mode", Mock())
    monkeypatch.setattr(module.time, "sleep", Mock())
    return module


@pytest.mark.parametrize("failure", [ValueError("Failed to load any schedule data."), None])
def test_schedule_retries_failure_or_empty_result(spike, monkeypatch, failure):
    schedule = pd.DataFrame({"RoundNumber": [1]})
    get_schedule = Mock(side_effect=[failure if failure else schedule.iloc[:0], schedule])
    monkeypatch.setattr(spike.fastf1, "get_event_schedule", get_schedule)
    assert spike.load_schedule(2024, online=True) is schedule
    assert get_schedule.call_count == 2
    spike.time.sleep.assert_called_once_with(60)


def test_offline_schedule_failure_does_not_wait(spike, monkeypatch):
    get_schedule = Mock(side_effect=ValueError("Missing cached schedule"))
    monkeypatch.setattr(spike.fastf1, "get_event_schedule", get_schedule)
    with pytest.raises(ValueError, match="Missing cached schedule"):
        spike.load_schedule(2024, online=False)
    assert get_schedule.call_count == 1
    spike.time.sleep.assert_not_called()


@pytest.mark.parametrize("recovers", [True, False])
def test_download_retries_2024_and_reports_missing_season(spike, monkeypatch, capsys, recovers):
    attempts = {}

    def get_schedule(year, **kwargs):
        attempts[year] = attempts.get(year, 0) + 1
        if year == 2024 and (not recovers or attempts[year] == 1):
            raise ValueError("Failed to load any schedule data.")
        return pd.DataFrame({"RoundNumber": [1], "EventName": ["Test race"]})

    monkeypatch.setattr(spike.fastf1, "get_event_schedule", get_schedule)
    monkeypatch.setattr(spike, "load_race", Mock(return_value=(None,) * 5))
    assert spike.run(online=True) == (0 if recovers else 1)
    log = json.loads((spike.OUTPUT / "download_log.json").read_text())
    assert log["seasons_requested"] == [2022, 2023, 2024, 2025]
    assert log["missing_seasons"] == ([] if recovers else [2024])
    assert log["complete"] is recovers
    assert attempts[2024] == (2 if recovers else 3)
    assert attempts[2025] == 1  # A missing season must not prevent later seasons.
    if not recovers:
        assert "ERROR: missing season schedules: [2024]" in capsys.readouterr().out
        assert spike.time.sleep.call_args_list[-2:] == [((60,),), ((120,),)]


@pytest.mark.parametrize("has_weather", [True, False])
def test_weather_is_requested_and_required(spike, monkeypatch, has_weather):
    session = SimpleNamespace(
        load=Mock(),
        laps=pd.DataFrame({"lap": [1]}),
        race_control_messages=pd.DataFrame({"Message": ["GREEN LIGHT"]}),
        weather_data=pd.DataFrame({"AirTemp": [20.0] if has_weather else []}),
        api_path="/test/",
    )
    event = SimpleNamespace(get_session=Mock(return_value=session), RoundNumber=1)
    monkeypatch.setattr(spike._api, "lap_count", Mock(return_value={"TotalLaps": [50]}))
    monkeypatch.setattr(spike, "pit_stops", Mock(return_value=pd.DataFrame()))
    monkeypatch.setattr(
        spike,
        "Ergast",
        Mock(
            return_value=SimpleNamespace(
                get_driver_info=Mock(return_value=SimpleNamespace(is_complete=True))
            )
        ),
    )
    if has_weather:
        assert spike.load_race(event, 2024)[0] is session
    else:
        with pytest.raises(ValueError, match="No weather data loaded"):
            spike.load_race(event, 2024)
    session.load.assert_called_once_with(laps=True, messages=True, telemetry=False, weather=True)


def test_audit_uses_historical_numbers_and_finds_stop_go_messages(spike):
    laps = pd.DataFrame(
        {
            "race_id": ["2022_01_R"] * 3,
            "DriverNumber": ["1"] * 3,
            "LapNumber": [1, 2, 3],
            "Time": [100.0, 200.0, 300.0],
            "LapTime": [100.0] * 3,
            "PitInTime": [None, 190.0, None],
            "PitOutTime": [None, None, 210.0],
            "Stint": [1.0, 1.0, 2.0],
            "Compound": ["HARD"] * 3,
            "TyreLife": [1.0, 2.0, 3.0],
            "Position": [1.0] * 3,
            "TrackStatus": ["1"] * 3,
            "IsAccurate": [True, False, False],
            "FastF1Generated": [False] * 3,
        }
    ).astype(RAW_LAPS_DTYPES)
    for name in ("Time", "LapTime", "PitInTime", "PitOutTime"):
        laps[name] = pd.to_timedelta(laps[name], unit="s")
    laps["Driver"] = "VER"
    laps["LapStartTime"] = pd.to_timedelta([0, 100, 200], unit="s")
    session = SimpleNamespace(
        laps=laps,
        total_laps=3,
        event=SimpleNamespace(RoundNumber=1, EventName="Test race"),
        weather_data=pd.DataFrame({"AirTemp": [20.0]}),
    )
    messages = pd.DataFrame(
        {"Message": ["10 SECOND STOP/GO PENALTY FOR CAR 1"], "Lap": [2], "RacingNumber": [None]}
    )
    counts = pd.DataFrame({"Time": pd.to_timedelta([0], unit="s"), "TotalLaps": [3]})
    stops = pd.DataFrame({"driverId": ["max_verstappen"], "lap": [2]})
    drivers = pd.DataFrame(
        {"driverId": ["max_verstappen"], "driverCode": ["VER"], "driverNumber": [3.0]}
    )
    evidence, _ = spike.audit(session, messages, counts, stops, drivers, "2022_01_R")
    assert evidence["e_pit_matches"]["matched"] == 1
    assert evidence["e_pit_matches"]["unmapped_driver_ids"] == []
    penalty = evidence["b_penalties"][0]
    assert penalty["driver_number"] == "1"
    assert penalty["context"] and penalty["all_pit_passes"]
