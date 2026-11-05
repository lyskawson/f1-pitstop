"""Offline checks for the T0 acquisition script; race fixture checks follow export."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest


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
