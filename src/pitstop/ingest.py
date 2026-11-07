"""Build immutable raw snapshots from a human-populated FastF1 cache, offline."""

import hashlib
import json
import logging
import re
import socket
import subprocess
import tarfile
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from pitstop.contracts import RAW_LAPS_DTYPES, make_race_id, validate_race_id, validate_raw_laps

ROOT = Path(__file__).resolve().parents[2]
SPECIAL = {"2023_18_R", "2025_08_R", "2025_23_R"}
TIME_COLUMNS = ("Time", "LapTime", "PitInTime", "PitOutTime")
D3_CLEANING = {
    "max_feature_gap": 3,
    "max_missing_compound_or_tyre_life_fraction": 0.10,
    "exclude_compounds": ["INTERMEDIATE", "WET"],
    "exclude_track_status": "5",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=invalid)


def file_record(path, name=None):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": name or path.name, "bytes": path.stat().st_size, "sha256": digest}


def raw_laps(session, race_id):
    """Normalize types without dropping, filling or reassigning any raw row."""
    laps = pd.DataFrame(session.laps).copy()
    laps["race_id"] = race_id
    for name, dtype in RAW_LAPS_DTYPES.items():
        if name in TIME_COLUMNS:
            laps[name] = laps[name].dt.total_seconds().astype("float64")
        elif dtype == "str":
            laps[name] = laps[name].astype("string")
            if name in ("Compound", "TrackStatus"):
                laps[name] = laps[name].mask(laps[name].isin(["nan", "None"]), pd.NA)
        else:
            # Reject fractional lap indices and unknown booleans instead of coercing them.
            if name == "LapNumber" and not laps[name].mod(1).eq(0).all():
                raise ValueError("LapNumber must be integral")
            if dtype == "bool" and (
                laps[name].isna().any() or not laps[name].isin([True, False]).all()
            ):
                raise ValueError(f"{name} must contain known booleans")
            laps[name] = laps[name].astype(dtype)
    laps = laps[list(RAW_LAPS_DTYPES)]
    validate_laps(laps)
    return laps


def validate_laps(laps):
    validate_raw_laps(laps)
    if laps.empty:
        raise ValueError("No raw laps")
    for field in ("Compound", "TrackStatus"):
        if laps[field].isin(["nan", "None"]).any():
            raise ValueError(f"{field} contains textual nulls")


def coverage(laps, events, cleaning):
    """Describe D3 exclusions; keep every raw row, including excluded evidence."""
    races, exclusions = [], []
    for event in events:
        race_id = event["race_id"]
        race = laps.loc[laps.race_id.eq(race_id)]
        drivers = []

        def exclude(scope, code, reason, race_id=race_id, **keys):
            exclusions.append(
                {"scope": scope, "race_id": race_id, **keys, "reason_code": code, "reason": reason}
            )

        if race.Compound.isin(cleaning["exclude_compounds"]).any():
            exclude("race", "wet", "At least one INTERMEDIATE/WET lap")
        if race.TrackStatus.str.contains(
            cleaning["exclude_track_status"], regex=False, na=False
        ).any():
            exclude("race", "red_flag", "At least one lap contains red-flag status 5")
        for driver, group in race.groupby("DriverNumber", sort=True):
            numbers = sorted(group.LapNumber.tolist())
            gaps = [
                {"after_lap": a, "before_lap": b, "missing_laps": b - a - 1}
                for a, b in zip(numbers, numbers[1:], strict=False)
                if b - a > 1
            ]
            missing = {name: int(group[name].isna().sum()) for name in RAW_LAPS_DTYPES}
            drivers.append(
                {
                    "driver_number": driver,
                    "rows": len(group),
                    "first_lap": numbers[0],
                    "last_lap": numbers[-1],
                    "gaps": gaps,
                    "missing": missing,
                }
            )
            if any(gap["missing_laps"] > cleaning["max_feature_gap"] for gap in gaps):
                exclude(
                    "race_driver",
                    "large_gap",
                    "More than three consecutive missing laps",
                    driver_number=driver,
                )
            for field in ("Compound", "TyreLife"):
                if (
                    missing[field] / len(group)
                    > cleaning["max_missing_compound_or_tyre_life_fraction"]
                ):
                    exclude(
                        "race_driver",
                        f"missing_{field}",
                        f"More than 10% missing {field}",
                        driver_number=driver,
                    )
        races.append(
            {
                "race_id": race_id,
                "rows": len(race),
                "special_tyre_rule": event["special_tyre_rule"],
                "drivers": drivers,
            }
        )
    return races, exclusions


@contextmanager
def offline_cache(cache):
    """Cache-only HTTP plus transport/authentication guards, even on cache misses."""
    import fastf1
    from fastf1.internals.f1account import LivetimingAuthAdapter
    from requests.adapters import HTTPAdapter

    if not cache.is_dir():
        raise ValueError(f"Missing human-populated cache: {cache}")
    fastf1.Cache.configure(cache_dir=str(cache))
    fastf1.Cache.offline_mode(True)
    with ExitStack() as stack:
        for target, name in (
            (HTTPAdapter, "send"),
            (LivetimingAuthAdapter, "send"),
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
        ):
            stack.enter_context(
                patch.object(
                    target, name, side_effect=RuntimeError("T2 forbids network and authentication")
                )
            )
        yield


def cached_call(operation, *args):
    """FastF1 can suppress cache misses during load; do not call that success."""
    misses = []

    class CacheMisses(logging.Handler):
        def emit(self, record):
            if url := getattr(record, "offline_cache_miss_url", None):
                misses.append(url)

    handler = CacheMisses()
    logger = logging.getLogger("fastf1")
    logger.addHandler(handler)
    try:
        result = operation(*args)
        if misses:
            raise ValueError(f"Missing cached responses: {misses}")
        return result
    finally:
        logger.removeHandler(handler)


def load_schedule(year):
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False, backend="fastf1")
    if (
        schedule.empty
        or not schedule.RoundNumber.gt(0).all()
        or schedule.RoundNumber.duplicated().any()
    ):
        raise ValueError(f"Missing or invalid cached schedule for {year}")
    return schedule


def load_race(event, year):
    from fastf1 import _api

    session = event.get_session("R")
    session.load(laps=True, messages=True, telemetry=False, weather=False)
    race_id = make_race_id(year, int(event.RoundNumber))
    laps = raw_laps(session, race_id)
    status = pd.DataFrame(session.track_status).copy()
    control = pd.DataFrame(session.race_control_messages).copy()
    counts = pd.DataFrame(_api.lap_count(session.api_path))
    if status.empty or control.empty or counts.empty or counts.TotalLaps.dropna().empty:
        raise ValueError(f"Missing cached status/control/lap-count stream: {race_id}")
    status["Time"] = status.Time.dt.total_seconds().astype("float64")
    # Race-control Time is UTC wall time, not session elapsed time. Retain it as ISO text.
    control["Time"] = control.Time.map(lambda value: value.isoformat() if pd.notna(value) else None)
    for frame in (status, control):
        frame.insert(0, "race_id", race_id)
    counts["Time"] = counts.Time.dt.total_seconds().astype("float64")
    total = session.total_laps
    if total is None or total != counts.TotalLaps.dropna().iloc[-1]:
        raise ValueError(f"Session.total_laps does not match last announcement: {race_id}")
    metadata = {
        "race_id": race_id,
        "season": year,
        "round_number": int(event.RoundNumber),
        "session": "R",
        "event_name": str(event.EventName),
        "total_laps": int(total),
        "total_laps_source": "FastF1 Session.total_laps; last non-null LapCount.TotalLaps",
        "special_tyre_rule": race_id in SPECIAL,
    }
    source = {
        "race_id": race_id,
        "service": "F1 live timing via human-populated FastF1 cache",
        "url": "https://livetiming.formula1.com" + session.api_path,
        "season": year,
        "round": int(event.RoundNumber),
        "session": "R",
        "acquired_at_utc": None,
        "acquisition_timestamp_note": (
            "Unknown: parsed cache and T0 log do not establish acquisition time; "
            "build time is separate"
        ),
    }
    announcements = {
        "race_id": race_id,
        "source_url": source["url"] + "LapCount.jsonStream",
        "time_basis": "session seconds",
        "parser": "fastf1._api.lap_count",
        "rows": json.loads(counts.to_json(orient="records", double_precision=15)),
    }
    return laps, metadata, status, control, announcements, source


def validate_events(events, laps):
    if not isinstance(events, list) or not events:
        raise ValueError("events must be a nonempty list")
    ids = []
    for event in events:
        expected = make_race_id(event["season"], event["round_number"])
        if (
            event["race_id"] != expected
            or event["session"] != "R"
            or event["season"] not in range(2022, 2026)
        ):
            raise ValueError("Invalid event race identity")
        if type(event["total_laps"]) is not int or event["total_laps"] <= 0:
            raise ValueError("Invalid event total_laps")
        if not event["event_name"] or not event["total_laps_source"]:
            raise ValueError("Missing event provenance")
        if type(event["special_tyre_rule"]) is not bool or event["special_tyre_rule"] != (
            expected in SPECIAL
        ):
            raise ValueError("Invalid special tyre rule flag")
        ids.append(expected)
    if len(ids) != len(set(ids)) or set(ids) != set(laps.race_id):
        raise ValueError("Event/lap race coverage differs")
    return ids


def safe_payload(root, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("Invalid payload path")
    parts = PurePosixPath(name)
    if parts.is_absolute() or not parts.parts or any(p in (".", "..") for p in name.split("/")):
        raise ValueError(f"Unsafe payload path: {name}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Payload escapes snapshot: {name}")
    if any((root / Path(*parts.parts[:i])).is_symlink() for i in range(1, len(parts.parts) + 1)):
        raise ValueError(f"Symlink payload: {name}")
    return path


def verify_data(manifest_path):
    """Verify portable snapshot bytes and recompute coverage without importing FastF1."""
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    for field in (
        "schema_version",
        "snapshot",
        "created_at_utc",
        "fastf1_version",
        "git_commit",
        "sources",
        "files",
        "coverage",
        "exclusions",
        "errors",
        "cleaning",
    ):
        if field not in manifest:
            raise ValueError(f"Manifest missing {field}")
    if manifest["schema_version"] != 1 or root.name != manifest["snapshot"]:
        raise ValueError("Manifest schema/snapshot directory mismatch")
    if manifest_path.name != "manifest.json" or manifest_path.is_symlink() or root.is_symlink():
        raise ValueError("Verify the manifest.json inside the declared snapshot")
    datetime.fromisoformat(manifest["created_at_utc"].replace("Z", "+00:00"))
    if manifest["cleaning"] != D3_CLEANING:
        raise ValueError("Cleaning policy differs from approved D3")
    if re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"]) is None:
        raise ValueError("Invalid producing git_commit")
    if not manifest["fastf1_version"].startswith("3.9."):
        raise ValueError("FastF1 version must follow approved D10 3.9 pin")
    files = manifest["files"]
    names = [item["path"] for item in files]
    if len(names) != len(set(names)) or "manifest.json" in names:
        raise ValueError("Duplicate payload or self-hashed manifest")
    if not {"laps.parquet", "events.json"}.issubset(names):
        raise ValueError("Missing required payload declaration")
    for item in files:
        path = safe_payload(root, item["path"])
        if not path.is_file():
            raise ValueError(f"Missing payload: {item['path']}")
        if file_record(path, item["path"]) != item:
            raise ValueError(f"Payload hash/size mismatch: {item['path']}")
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != {*names, "manifest.json"}:
        raise ValueError("Undeclared or missing snapshot files")
    laps = pd.read_parquet(root / "laps.parquet")
    validate_laps(laps)
    events = read_json(root / "events.json")
    loaded = validate_events(events, laps)
    races, exclusions = coverage(laps, events, manifest["cleaning"])
    cov = manifest["coverage"]
    requested = cov["requested_races"]
    for race_id in requested:
        validate_race_id(race_id)
        if int(race_id[:4]) not in range(2022, 2026):
            raise ValueError("Requested race outside T2 seasons")
    missing_seasons = cov["missing_seasons"]
    if cov["requested_seasons"] != [2022, 2023, 2024, 2025] or set(missing_seasons) - set(
        cov["requested_seasons"]
    ):
        raise ValueError("Invalid season coverage")
    if {int(r[:4]) for r in requested} | set(missing_seasons) != set(cov["requested_seasons"]):
        raise ValueError("Unreported missing season")
    if len(requested) != len(set(requested)) or not set(loaded).issubset(requested):
        raise ValueError("Invalid requested race coverage")
    if (
        loaded != cov["loaded_races"]
        or races != cov["races"]
        or exclusions != manifest["exclusions"]
    ):
        raise ValueError("Manifest coverage/exclusions differ from raw payloads")
    if [s["race_id"] for s in manifest["sources"]] != loaded:
        raise ValueError("Missing source provenance")
    for source, event in zip(manifest["sources"], events, strict=True):
        if (source["season"], source["round"], source["session"]) != (
            event["season"],
            event["round_number"],
            "R",
        ):
            raise ValueError("Source/event identity differs")
        if not source["url"] or not source["service"]:
            raise ValueError("Missing source provenance")
        if source["acquired_at_utc"] is None:
            if not source["acquisition_timestamp_note"]:
                raise ValueError("Unknown acquisition timestamp needs provenance note")
        else:
            datetime.fromisoformat(source["acquired_at_utc"].replace("Z", "+00:00"))
    for name, fields in (
        ("track_status.parquet", ["Time", "Status", "Message"]),
        ("race_control.parquet", ["Time", "Message"]),
    ):
        if name in names:
            frame = pd.read_parquet(root / name)
            if not {"race_id", *fields}.issubset(frame.columns) or set(frame.race_id) != set(
                loaded
            ):
                raise ValueError(f"Invalid raw stream: {name}")
            if frame.Time.isna().any():
                raise ValueError(f"Missing stream timestamps: {name}")
            if name == "track_status.parquet":
                if frame.Time.dtype != np.dtype("float64") or not np.isfinite(frame.Time).all():
                    raise ValueError("Invalid track-status seconds")
            elif (
                pd.to_datetime(frame.Time, errors="coerce", utc=True, format="ISO8601").isna().any()
            ):
                raise ValueError("Invalid race-control UTC timestamps")
    # This optional evidence has no approved dataset-reader or feature semantics.
    if "lap_count_evidence.json" in names:
        evidence = read_json(root / "lap_count_evidence.json")
        if [entry["race_id"] for entry in evidence] != loaded:
            raise ValueError("Lap-count evidence coverage differs")
        for entry, event in zip(evidence, events, strict=True):
            counts = pd.DataFrame(entry["rows"])
            if counts.empty or counts.Time.isna().any() or not np.isfinite(counts.Time).all():
                raise ValueError("Invalid lap-count timestamps")
            values = counts.TotalLaps.dropna()
            if (
                values.empty
                or not np.isfinite(values).all()
                or not values.gt(0).all()
                or not values.mod(1).eq(0).all()
                or values.iloc[-1] != event["total_laps"]
            ):
                raise ValueError("Lap-count evidence differs from event")
            if (
                not entry["source_url"]
                or entry["time_basis"] != "session seconds"
                or not entry["parser"]
            ):
                raise ValueError("Missing lap-count provenance")
    missing = set(requested) - set(loaded)
    failures = {e.get("race_id") for e in manifest["errors"]}
    if missing - failures:
        raise ValueError("Missing races lack explicit ingestion errors")
    for error in manifest["errors"]:
        if (
            not all(error[field] for field in ("stage", "error_type", "message"))
            or error["ingestion_succeeded"] is not False
        ):
            raise ValueError("Invalid ingestion failure record")
    complete = (
        set(requested) == set(loaded) and not cov["missing_seasons"] and not manifest["errors"]
    )
    if cov["complete"] is not complete:
        raise ValueError("False completeness claim")
    return {
        "snapshot": manifest["snapshot"],
        "complete": complete,
        "requested_races": len(requested),
        "loaded_races": len(loaded),
        "rows": len(laps),
        "errors": manifest["errors"],
        "missing_seasons": cov["missing_seasons"],
    }


def build_snapshot(config_path, snapshot):
    """Stage, verify and archive; publish a directory only after every race succeeds."""
    import fastf1

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", snapshot) is None:
        raise ValueError("snapshot must be a simple immutable version name")
    config = yaml.safe_load(Path(config_path).read_text())
    if config["seasons"] != [2022, 2023, 2024, 2025] or config["session"] != "R":
        raise ValueError("T2 requires 2022–2025 Race sessions")
    if config["cleaning"] != D3_CLEANING:
        raise ValueError("Cleaning policy differs from approved D3")
    raw = ROOT / config["paths"]["raw"]
    destination = raw / snapshot
    archive = raw / f"{snapshot}.tar.gz"
    if destination.exists() or destination.is_symlink() or archive.exists() or archive.is_symlink():
        raise ValueError(f"Immutable destination already exists: {destination} or {archive}")
    raw.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    manifest = {
        "schema_version": 1,
        "snapshot": snapshot,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fastf1_version": fastf1.__version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "sources": [],
        "files": [],
        "coverage": {},
        "exclusions": [],
        "errors": [],
        "cleaning": config["cleaning"],
        "acceptance": "pending human acceptance",
        "raw_streams": {
            "track_status.parquet": "FastF1 Session.track_status; Time in session seconds",
            "race_control.parquet": "FastF1 Session.race_control_messages; Time is UTC ISO text",
            "lap_count_evidence.json": (
                "Archival evidence only; reader schema and feature policy remain unapproved"
            ),
        },
    }
    requested, missing_seasons = [], []
    frames, events, statuses, controls, evidence = [], [], [], [], []
    with tempfile.TemporaryDirectory(prefix=f".{snapshot}-", dir=raw) as temporary:
        staging = Path(temporary) / snapshot
        staging.mkdir()
        with offline_cache(ROOT / config["paths"]["cache"]):
            for year in config["seasons"]:
                try:
                    schedule = cached_call(load_schedule, year)
                except Exception as exc:
                    missing_seasons.append(year)
                    manifest["errors"].append(
                        {
                            "season": year,
                            "stage": "schedule",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "ingestion_succeeded": False,
                        }
                    )
                    continue
                for _, event in schedule.iterrows():
                    race_id = make_race_id(year, int(event.RoundNumber))
                    requested.append(race_id)
                    print(f"Exporting cached {race_id} {event.EventName}", flush=True)
                    try:
                        laps, metadata, status, control, counts, source = cached_call(
                            load_race, event, year
                        )
                        frames.append(laps)
                        events.append(metadata)
                        statuses.append(status)
                        controls.append(control)
                        evidence.append(counts)
                        manifest["sources"].append(source)
                    except Exception as exc:
                        manifest["errors"].append(
                            {
                                "race_id": race_id,
                                "stage": "cache_export",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                                "ingestion_succeeded": False,
                            }
                        )
        laps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        races, exclusions = coverage(laps, events, config["cleaning"]) if frames else ([], [])
        manifest["coverage"] = {
            "requested_seasons": config["seasons"],
            "requested_races": requested,
            "loaded_races": [e["race_id"] for e in events],
            "missing_seasons": missing_seasons,
            "races": races,
            "complete": not manifest["errors"] and bool(events),
        }
        manifest["exclusions"] = exclusions
        if not manifest["coverage"]["complete"]:
            failure = Path(temporary + "-failure.json")
            write_json(failure, manifest)
            raise ValueError(
                f"Incomplete cache coverage; no snapshot published. Diagnostics: {failure}"
            )
        laps.to_parquet(staging / "laps.parquet", index=False)
        write_json(staging / "events.json", events)
        pd.concat(statuses, ignore_index=True).to_parquet(
            staging / "track_status.parquet", index=False
        )
        pd.concat(controls, ignore_index=True).to_parquet(
            staging / "race_control.parquet", index=False
        )
        write_json(staging / "lap_count_evidence.json", evidence)
        manifest["files"] = [file_record(path) for path in sorted(staging.iterdir())]
        write_json(staging / "manifest.json", manifest)
        summary = verify_data(staging / "manifest.json")
        packed = Path(temporary) / archive.name
        with tarfile.open(packed, "w:gz") as tar:
            for name in ["manifest.json", *(item["path"] for item in manifest["files"])]:
                tar.add(staging / name, arcname=f"{snapshot}/{name}", recursive=False)
        with tempfile.TemporaryDirectory(prefix="pitstop-unpack-") as extracted:
            with tarfile.open(packed) as tar:
                tar.extractall(extracted, filter="data")
            verify_data(Path(extracted) / snapshot / "manifest.json")
        # A separate lock serializes publication without reserving a partial snapshot.
        with (raw / f".{snapshot}.lock").open("x"):
            try:
                if destination.exists() or archive.exists():
                    raise ValueError("Immutable destination appeared during build")
                staging.rename(destination)
                packed.rename(archive)
            finally:
                (raw / f".{snapshot}.lock").unlink()
    return {
        **summary,
        "manifest": str(destination / "manifest.json"),
        "archive": str(archive),
        "archive_sha256": file_record(archive)["sha256"],
    }
