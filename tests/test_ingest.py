"""Offline ingestion, raw preservation and portable snapshot verification."""

import json
import logging
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fastf1
import pandas as pd
import pytest
import yaml

from pitstop import ingest
from pitstop.cli import main

FIXTURES = Path(__file__).parent / "fixtures/raw"


@pytest.fixture
def raw():
    frame = pd.read_parquet(FIXTURES / "laps.parquet")
    frame = frame.loc[frame.DriverNumber.eq("1")].head(5).reset_index(drop=True)
    frame.loc[0, "Compound"] = pd.NA
    frame.loc[1, "TrackStatus"] = pd.NA
    frame.loc[2, "FastF1Generated"] = True
    frame.loc[3, "IsAccurate"] = False
    frame.loc[3, "PitInTime"] = 123.0  # Keep markers even outside a lap interval.
    frame.loc[4, "PitOutTime"] = 125.0
    return frame


@pytest.mark.parametrize("field", ["Compound", "TrackStatus"])
@pytest.mark.parametrize("missing", ["nan", "None", None, float("nan"), pd.NA])
def test_raw_adapter_preserves_rows_nulls_and_markers(raw, field, missing):
    source = raw.copy()
    source[field] = pd.Series(
        [missing, "HARD" if field == "Compound" else "145", missing, missing, missing]
    )
    expected = source.copy()
    expected[field] = pd.Series([pd.NA, source.loc[1, field], pd.NA, pd.NA, pd.NA], dtype="string")
    for name in ingest.TIME_COLUMNS:
        source[name] = pd.to_timedelta(source[name], unit="s")
    before = source.copy(deep=True)
    actual = ingest.raw_laps(SimpleNamespace(laps=source), "2022_01_R")
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("field,value", [("LapNumber", 1.5), ("IsAccurate", None)])
def test_adapter_rejects_lossy_key_or_boolean_coercion(raw, field, value):
    source = raw.copy()
    for name in ingest.TIME_COLUMNS:
        source[name] = pd.to_timedelta(source[name], unit="s")
    source[field] = source[field].astype(object)
    source.loc[0, field] = value
    with pytest.raises(ValueError, match="integral|known booleans"):
        ingest.raw_laps(SimpleNamespace(laps=source), "2022_01_R")


@pytest.fixture
def builder(tmp_path, monkeypatch, raw):
    config = yaml.safe_load((ingest.ROOT / "configs/data.yaml").read_text())
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config))
    (tmp_path / "data/cache").mkdir(parents=True)
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    monkeypatch.setattr(ingest.subprocess, "check_output", Mock(side_effect=["a" * 40, ""]))
    monkeypatch.setattr(fastf1.Cache, "configure", Mock())
    monkeypatch.setattr(fastf1.Cache, "offline_mode", Mock())
    monkeypatch.setattr(
        ingest,
        "load_schedule",
        lambda year: pd.DataFrame({"RoundNumber": [1], "EventName": ["Test race"]}),
    )

    def load(event, year):
        race_id = f"{year}_01_R"
        laps = raw.copy()
        laps["race_id"] = pd.Series([race_id] * len(laps), dtype="string")
        metadata = {
            "race_id": race_id,
            "season": year,
            "round_number": 1,
            "session": "R",
            "event_name": "Test race",
            "total_laps": 9,
            "total_laps_source": "cached LapCount",
            "special_tyre_rule": False,
        }
        status = pd.DataFrame(
            {"race_id": [race_id], "Time": [0.0], "Status": ["1"], "Message": ["Green"]}
        )
        control = pd.DataFrame(
            {"race_id": [race_id], "Time": ["2022-01-01T12:00:00"], "Message": ["GREEN LIGHT"]}
        )
        counts = {
            "race_id": race_id,
            "source_url": "https://example.invalid/LapCount.jsonStream",
            "time_basis": "session seconds",
            "parser": "fastf1._api.lap_count",
            "rows": [
                {"Time": 0.0, "TotalLaps": 10, "CurrentLap": None},
                {"Time": 50.0, "TotalLaps": 9, "CurrentLap": 1},
            ],
        }
        source = {
            "race_id": race_id,
            "service": "synthetic cache",
            "url": "https://example.invalid/",
            "season": year,
            "round": 1,
            "session": "R",
            "acquired_at_utc": None,
            "acquisition_timestamp_note": "Unknown",
        }
        return laps, metadata, status, control, counts, source

    monkeypatch.setattr(ingest, "load_race", load)
    return config_path


@pytest.fixture
def snapshot(builder):
    return ingest.build_snapshot(builder, "v1")


def rewrite_manifest(snapshot, change):
    path = Path(snapshot["manifest"])
    manifest = json.loads(path.read_text())
    change(manifest)
    ingest.write_json(path, manifest)
    return path


def rehash(snapshot, name):
    path = Path(snapshot["manifest"])
    record = ingest.file_record(path.parent / name)
    return rewrite_manifest(
        snapshot, lambda m: m.update(files=[record if f["path"] == name else f for f in m["files"]])
    )


def test_manifest_archive_roundtrip_and_cache_independence(snapshot, raw, tmp_path, monkeypatch):
    path = Path(snapshot["manifest"])
    manifest = json.loads(path.read_text())
    assert snapshot["complete"] is True
    assert snapshot["requested_races"] == snapshot["loaded_races"] == 4
    assert manifest["fastf1_version"] == fastf1.__version__
    assert manifest["git_commit"] == "a" * 40 and manifest["git_dirty"] is False
    assert all(s["acquired_at_utc"] is None for s in manifest["sources"])
    assert manifest["coverage"]["races"][0]["drivers"][0]["missing"]["Compound"] == 1
    actual = pd.read_parquet(path.parent / "laps.parquet")
    pd.testing.assert_frame_equal(
        actual.loc[actual.race_id.eq("2022_01_R")].reset_index(drop=True), raw
    )
    assert json.loads((path.parent / "events.json").read_text())[0]["total_laps"] == 9
    assert actual.LapNumber.max() == 5
    assert (
        json.loads((path.parent / "lap_count_evidence.json").read_text())[0]["rows"][0]["TotalLaps"]
        == 10
    )
    extracted = tmp_path / "extracted"
    with tarfile.open(snapshot["archive"]) as archive:
        assert set(archive.getnames()) == {
            f"v1/{name}" for name in ["manifest.json", *(f["path"] for f in manifest["files"])]
        }
        archive.extractall(extracted, filter="data")
    # A new interpreter cannot import FastF1 or open the cache or credential store.
    script = """
import sys
sys.modules["fastf1"] = None
def audit(event, args):
    blocked = ("data/cache", "f1auth", "download_log")
    if event == "open" and any(word in str(args[0]) for word in blocked):
        raise AssertionError("verification touched acquisition state")
sys.addaudithook(audit)
from pitstop.cli import main
raise SystemExit(main(["verify-data", "--manifest", sys.argv[1]]))
"""
    # Undo the builder-only revision mock so this is a real subprocess.
    monkeypatch.undo()
    result = subprocess.run(
        [sys.executable, "-c", script, str(extracted / "v1/manifest.json")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rows"] == 20


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra"])
def test_rejects_changed_missing_or_undeclared_payload(snapshot, mutation):
    root = Path(snapshot["manifest"]).parent
    if mutation == "tamper":
        (root / "events.json").write_text("[]")
    elif mutation == "missing":
        (root / "laps.parquet").unlink()
    else:
        (root / "credentials.txt").write_text("not part of snapshot")
    with pytest.raises(ValueError, match="hash/size|Missing payload|Undeclared"):
        ingest.verify_data(snapshot["manifest"])


@pytest.mark.parametrize(
    "name",
    ["../outside", "/etc/passwd", "nested/../../outside", "nested\\outside", "./events.json"],
)
def test_rejects_unsafe_manifest_paths(snapshot, name):
    path = rewrite_manifest(snapshot, lambda m: m["files"][1].update(path=name))
    with pytest.raises(ValueError, match="payload path"):
        ingest.verify_data(path)


def test_rejects_symlink_payload(snapshot, tmp_path):
    path = Path(snapshot["manifest"]).parent / "events.json"
    outside = tmp_path / "outside.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes|Symlink"):
        ingest.verify_data(snapshot["manifest"])


@pytest.mark.parametrize("mutation", ["dtype", "duplicate", "text_null", "infinity"])
def test_rejects_invalid_raw_data_even_with_matching_hash(snapshot, mutation):
    path = Path(snapshot["manifest"]).parent / "laps.parquet"
    laps = pd.read_parquet(path)
    if mutation == "dtype":
        laps["LapNumber"] = laps.LapNumber.astype(float)
    elif mutation == "duplicate":
        laps = pd.concat([laps, laps.iloc[:1]], ignore_index=True)
    elif mutation == "text_null":
        laps.loc[0, "Compound"] = "None"
    else:
        laps.loc[0, "LapTime"] = float("inf")
    laps.to_parquet(path, index=False)
    rehash(snapshot, "laps.parquet")
    with pytest.raises(ValueError, match="dtype|duplicate|textual null|infinity"):
        ingest.verify_data(snapshot["manifest"])


def test_coverage_and_exclusions_are_checked_against_payload(snapshot):
    path = rewrite_manifest(
        snapshot, lambda m: m["coverage"]["races"][0]["drivers"][0]["missing"].update(Compound=0)
    )
    with pytest.raises(ValueError, match="coverage/exclusions"):
        ingest.verify_data(path)


def test_incomplete_coverage_is_reported_and_cli_fails(snapshot, capsys):
    def incomplete(m):
        m["coverage"]["requested_races"].append("2022_02_R")
        m["coverage"]["complete"] = False
        m["errors"].append(
            {
                "race_id": "2022_02_R",
                "stage": "cache_export",
                "error_type": "ValueError",
                "message": "Missing cached response",
                "ingestion_succeeded": False,
            }
        )

    path = rewrite_manifest(snapshot, incomplete)
    assert ingest.verify_data(path)["complete"] is False
    capsys.readouterr()
    assert main(["verify-data", "--manifest", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["loaded_races"] == 4
    rewrite_manifest(snapshot, lambda m: m["coverage"].update(complete=True))
    with pytest.raises(ValueError, match="False completeness"):
        ingest.verify_data(path)


@pytest.mark.parametrize("stage", ["schedule", "race", "suppressed_miss"])
def test_missing_cache_is_explicit_and_no_snapshot_or_archive_is_published(
    builder, monkeypatch, stage
):
    if stage == "schedule":
        monkeypatch.setattr(
            ingest, "load_schedule", Mock(side_effect=ValueError("Missing schedule cache"))
        )
    elif stage == "race":
        monkeypatch.setattr(ingest, "load_race", Mock(side_effect=ValueError("Missing race cache")))
    else:
        load = ingest.load_race

        def partial(*args):
            logging.getLogger("fastf1").warning(
                "cache miss", extra={"offline_cache_miss_url": "https://example.invalid/missing"}
            )
            return load(*args)

        monkeypatch.setattr(ingest, "load_race", partial)
    with pytest.raises(ValueError, match="Incomplete cache coverage"):
        ingest.build_snapshot(builder, "v1")
    raw = builder.parent / "data/raw"
    assert not (raw / "v1").exists() and not (raw / "v1.tar.gz").exists()
    manifest = json.loads(next(raw.glob("*-failure.json")).read_text())
    assert not manifest["coverage"]["complete"] and manifest["errors"]
    assert manifest["coverage"]["loaded_races"] == []
    assert all(error["ingestion_succeeded"] is False for error in manifest["errors"])


@pytest.mark.parametrize("existing", ["directory", "archive"])
def test_existing_destinations_are_immutable(builder, existing):
    root = builder.parent / "data/raw"
    root.mkdir(parents=True)
    if existing == "directory":
        (root / "v1").mkdir()
        marker = root / "v1/original"
    else:
        marker = root / "v1.tar.gz"
    marker.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="already exists"):
        ingest.build_snapshot(builder, "v1")
    assert marker.read_bytes() == b"unchanged"


def test_d3_metadata_keeps_excluded_and_uncertain_raw_evidence(raw):
    laps = raw.copy()
    laps["LapNumber"] = [1, 2, 3, 8, 9]
    laps.loc[0, "Compound"] = "WET"
    laps.loc[1, "TrackStatus"] = "145"
    laps.loc[2, "TyreLife"] = float("nan")
    before = laps.copy(deep=True)
    events = [{"race_id": "2022_01_R", "special_tyre_rule": False}]
    policy = yaml.safe_load((Path(__file__).parents[1] / "configs/data.yaml").read_text())[
        "cleaning"
    ]
    races, exclusions = ingest.coverage(laps, events, policy)
    assert {e["reason_code"] for e in exclusions} == {
        "wet",
        "red_flag",
        "large_gap",
        "missing_TyreLife",
    }
    assert races[0]["drivers"][0]["gaps"] == [{"after_lap": 3, "before_lap": 8, "missing_laps": 4}]
    assert races[0]["rows"] == 5
    pd.testing.assert_frame_equal(laps, before)


def test_offline_guard_blocks_transport_before_credentials(tmp_path, monkeypatch):
    from fastf1.internals.f1account import LivetimingAuthAdapter, token_manager
    from requests import Request
    from requests.adapters import HTTPAdapter

    monkeypatch.setattr(fastf1.Cache, "configure", Mock())
    monkeypatch.setattr(fastf1.Cache, "offline_mode", Mock())
    token = Mock(side_effect=AssertionError("credential access"))
    monkeypatch.setattr(token_manager, "get", token)
    with ingest.offline_cache(tmp_path):
        for adapter in (HTTPAdapter(), LivetimingAuthAdapter(token_manager)):
            with pytest.raises(RuntimeError, match="forbids network and authentication"):
                adapter.send(Request("GET", "https://example.invalid").prepare())
    token.assert_not_called()
    fastf1.Cache.offline_mode.assert_called_once_with(True)


def test_download_requires_explicit_cache_mode(builder, capsys):
    assert main(["download", "--config", str(builder), "--snapshot", "v1"]) == 1
    assert "requires --config, --from-cache and --snapshot" in capsys.readouterr().err


@pytest.mark.parametrize(
    "payload", ["track_status.parquet", "race_control.parquet", "lap_count_evidence.json"]
)
def test_rejects_invalid_timestamped_evidence_with_matching_hash(snapshot, payload):
    path = Path(snapshot["manifest"]).parent / payload
    if payload.endswith("parquet"):
        frame = pd.read_parquet(path)
        frame.loc[0, "Time"] = float("inf") if payload.startswith("track") else "not-a-time"
        frame.to_parquet(path, index=False)
    else:
        evidence = json.loads(path.read_text())
        evidence[0]["rows"][0]["Time"] = None
        ingest.write_json(path, evidence)
    rehash(snapshot, payload)
    with pytest.raises(ValueError, match="timestamps|seconds"):
        ingest.verify_data(snapshot["manifest"])


def test_missing_acquisition_provenance_is_rejected(snapshot):
    path = rewrite_manifest(snapshot, lambda m: m["sources"][0].pop("acquisition_timestamp_note"))
    with pytest.raises(KeyError, match="acquisition_timestamp_note"):
        ingest.verify_data(path)


def test_unreported_missing_race_is_rejected(snapshot):
    path = rewrite_manifest(
        snapshot, lambda m: m["coverage"]["requested_races"].append("2022_02_R")
    )
    with pytest.raises(ValueError, match="explicit ingestion errors"):
        ingest.verify_data(path)


def test_real_loader_preserves_planned_announcements_and_control_timestamps(raw, monkeypatch):
    from fastf1 import _api

    laps = raw.copy()
    for name in ingest.TIME_COLUMNS:
        laps[name] = pd.to_timedelta(laps[name], unit="s")
    counts = {
        "Time": pd.to_timedelta([0, 30, 40], unit="s"),
        "TotalLaps": [12, None, 9],
        "CurrentLap": [None, 1, None],
    }
    session = SimpleNamespace(
        laps=laps,
        total_laps=9,
        api_path="/static/test/Race/",
        load=Mock(),
        track_status=pd.DataFrame(
            {"Time": pd.to_timedelta([0], unit="s"), "Status": ["1"], "Message": ["Green"]}
        ),
        race_control_messages=pd.DataFrame(
            {"Time": pd.to_datetime(["2022-01-01T12:00:00"]), "Message": ["PENALTY"], "Lap": [3]}
        ),
    )
    event = SimpleNamespace(
        get_session=Mock(return_value=session), RoundNumber=1, EventName="Test race"
    )
    monkeypatch.setattr(_api, "lap_count", Mock(return_value=counts))
    actual, metadata, status, control, evidence, source = ingest.load_race(event, 2022)
    pd.testing.assert_frame_equal(actual, raw)
    session.load.assert_called_once_with(laps=True, messages=True, telemetry=False, weather=False)
    assert metadata["total_laps"] == 9 and actual.LapNumber.max() == 5
    assert [row["TotalLaps"] for row in evidence["rows"]] == [12, None, 9]
    assert [row["Time"] for row in evidence["rows"]] == [0.0, 30.0, 40.0]
    assert evidence["source_url"].endswith("/static/test/Race/LapCount.jsonStream")
    assert source["acquired_at_utc"] is None
    assert status.Time.tolist() == [0.0]
    assert control.Time.tolist() == ["2022-01-01T12:00:00"] and control.Message.tolist() == [
        "PENALTY"
    ]
