"""T0 exploration: human download, then cache-only evidence and real fixtures."""

import argparse
import json
import logging
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import fastf1
import pandas as pd
from fastf1 import _api
from fastf1.ergast import Ergast

from pitstop.contracts import RAW_LAPS_DTYPES, make_race_id, validate_raw_laps

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/t0"
SPECIAL = {"2023_18_R", "2025_08_R", "2025_23_R"}
RED_EXAMPLES = {"2022_10_R", "2023_03_R", "2024_08_R"}
RATE_LIMIT_WORDS = ("rate limit", "ratelimit", "429", "too many requests")


def write_json(path, value):
    """Replace completed JSON atomically, including after an interrupted rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


class RateLimitLog(logging.Handler):
    """FastF1 sometimes logs and suppresses exceptions inside Session.load."""

    def __init__(self):
        super().__init__()
        self.hit = False

    def emit(self, record):
        message = record.getMessage()
        if record.exc_info:
            message += str(record.exc_info[1])
        if any(word in message.lower() for word in RATE_LIMIT_WORDS):
            self.hit = True


def retry(operation, online):
    """Wait out FastF1's hourly limits; ordinary per-race errors go to coverage."""
    while True:
        observer = RateLimitLog()
        logger = logging.getLogger("fastf1")
        logger.addHandler(observer)
        error = None
        try:
            value = operation()
        except Exception as exc:
            error = exc
            observer.hit |= any(
                word in f"{type(exc).__name__}: {exc}".lower() for word in RATE_LIMIT_WORDS
            )
        finally:
            logger.removeHandler(observer)
        if observer.hit and online:
            print("Rate limit reached; waiting one hour before retrying.", flush=True)
            time.sleep(3600)
            continue
        if error:
            raise error
        return value


def pit_stops(year, round_number):
    """Fetch every page, including a legitimate empty pit-stop response."""
    response = Ergast().get_pit_stops(year, round_number, limit=100)
    frames = list(response.content)
    count = sum(len(frame) for frame in frames)
    while count < response.total_results:
        response = response.get_next_result_page()
        frames.extend(response.content)
        added = sum(len(frame) for frame in response.content)
        if not added:
            raise ValueError("Jolpica pagination made no progress")
        count += added
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_race(event, year):
    session = event.get_session("R")
    session.load(laps=True, messages=True, telemetry=False, weather=False)
    if session.laps.empty:
        raise ValueError("No laps loaded")
    # Access each required source: Session.load can suppress individual failures.
    messages = session.race_control_messages
    if messages.empty:
        raise ValueError("No race-control messages loaded")
    counts = pd.DataFrame(_api.lap_count(session.api_path))
    if counts.empty or counts.TotalLaps.dropna().empty:
        raise ValueError("No total-lap announcements loaded")
    stops = pit_stops(year, int(event.RoundNumber))
    # Driver IDs are needed for Jolpica matching; do not use session.results.
    drivers = Ergast().get_driver_info(year, int(event.RoundNumber), limit=100)
    if not drivers.is_complete:
        raise ValueError("Incomplete Jolpica driver mapping")
    return session, messages, counts, stops, drivers


def raw_laps(session, race_id):
    laps = pd.DataFrame(session.laps).copy()
    laps["race_id"] = race_id
    for name, dtype in RAW_LAPS_DTYPES.items():
        if name in ("Time", "LapTime", "PitInTime", "PitOutTime"):
            laps[name] = laps[name].dt.total_seconds()
        elif dtype == "str":
            laps[name] = laps[name].astype("string")
        else:
            laps[name] = laps[name].astype(dtype)
    laps = laps[list(RAW_LAPS_DTYPES)]
    validate_raw_laps(laps)
    return laps


def lap_context(laps, driver, lap, radius=2):
    selected = laps.loc[
        laps.DriverNumber.eq(str(driver)) & laps.LapNumber.between(lap - radius, lap + radius)
    ]
    return records(selected)


def audit(session, messages, counts, stops, drivers, race_id):
    laps = raw_laps(session, race_id)
    original = pd.DataFrame(session.laps).sort_values(["DriverNumber", "LapNumber"])
    reasons = []
    if laps.Compound.isin(["INTERMEDIATE", "WET"]).any():
        reasons.append("wet")
    if laps.TrackStatus.str.contains("5", na=False).any():
        reasons.append("red_flag")
    event = {
        "race_id": race_id,
        "season": int(race_id[:4]),
        "round_number": int(session.event.RoundNumber),
        "session": "R",
        "event_name": session.event.EventName,
        "total_laps": int(session.total_laps),
        "total_laps_source": "FastF1 Session.total_laps; last non-null LapCount.TotalLaps",
        "special_tyre_rule": race_id in SPECIAL,
    }
    announcements = counts.copy()
    announcements["Time"] = pd.to_timedelta(announcements.Time).dt.total_seconds()
    announced = counts.TotalLaps.dropna().astype(int)
    # Validate pit timestamps against the lap interval, without filtering any rows.
    boundaries = {}
    for column in ("PitInTime", "PitOutTime"):
        marked = original[column].notna()
        known = marked & original.LapStartTime.notna() & original.Time.notna()
        outside = known & (
            (original[column] < original.LapStartTime) | (original[column] > original.Time)
        )
        boundaries[column] = {
            "marked": int(marked.sum()),
            "unverifiable": int((marked & ~known).sum()),
            "outside_lap": records(original.loc[outside, ["DriverNumber", "LapNumber", column]]),
            "examples": records(
                original.loc[known].head(3)[
                    ["DriverNumber", "LapNumber", "LapStartTime", "Time", column]
                ]
            ),
        }
    penalty_messages = messages.loc[
        messages.Message.str.contains(
            r"DRIVE[ -]?THROUGH|STOP[ -]?AND[ -]?GO", case=False, na=False
        )
    ]
    penalties = []
    for message in records(penalty_messages):
        driver = message.get("RacingNumber")
        lap = message.get("Lap")
        # Message times are UTC while raw lap times are session seconds.
        # Use the explicit message lap; do not guess which stop served a penalty.
        penalties.append(
            {
                "message": message,
                "context": lap_context(laps, driver, lap, radius=5) if driver and lap else [],
                "interpretation": "Review issuance/served wording and surrounding stops manually",
            }
        )
    accuracy = {}
    previous_status = original.groupby("DriverNumber").TrackStatus.shift(1)
    for name, mask in {
        "in_laps": original.PitInTime.notna(),
        "out_laps": original.PitOutTime.notna(),
        "generated": original.FastF1Generated,
        "non_green_yellow": ~original.TrackStatus.isin(["1", "2", "12", "21"]),
        "after_any_sc": previous_status.str.contains("4", na=False),
        "after_exact_sc": previous_status.eq("4"),
    }.items():
        accuracy[name] = {
            "observations": int(mask.sum()),
            "accurate": int((mask & original.IsAccurate).sum()),
            "counterexamples": records(
                original.loc[
                    mask & original.IsAccurate,
                    ["DriverNumber", "LapNumber", "TrackStatus", "IsAccurate"],
                ]
            ),
        }
    changes = []
    clean = laps.loc[~laps.FastF1Generated]
    for driver, group in clean.groupby("DriverNumber"):
        by_lap = group.set_index("LapNumber")
        for row in group.loc[group.PitInTime.notna()].itertuples():
            following = by_lap.loc[row.LapNumber + 1] if row.LapNumber + 1 in by_lap.index else None
            if following is not None and following.Stint > row.Stint:
                changes.append({"DriverNumber": driver, "lap": row.LapNumber})
    mapping = dict(zip(drivers.driverId, drivers.driverNumber.astype(str), strict=True))
    unmatched = list(range(len(changes)))
    matched = []
    missing = []
    for stop in records(stops):
        driver = mapping.get(stop["driverId"])
        candidates = [
            i
            for i in unmatched
            if changes[i]["DriverNumber"] == driver and abs(changes[i]["lap"] - stop["lap"]) <= 1
        ]
        if candidates:
            index = min(candidates, key=lambda i: abs(changes[i]["lap"] - stop["lap"]))
            unmatched.remove(index)
            matched.append({"fastf1": changes[index], "jolpica": stop})
        else:
            missing.append({"DriverNumber": driver, "jolpica": stop})
    red_context = []
    if race_id in RED_EXAMPLES:
        for driver, group in laps.groupby("DriverNumber"):
            red = group.loc[group.TrackStatus.str.contains("5", na=False)]
            for lap in red.LapNumber:
                red_context.extend(lap_context(laps, driver, lap))
    evidence = {
        "event": event,
        "exclusion_reasons": reasons,
        "lap_rows": len(laps),
        "a_pit_boundaries": boundaries,
        "b_penalties": penalties,
        "c_total_laps": {
            "loader": event["total_laps"],
            "changes": announced.nunique() > 1,
            "announcements": records(announcements),
        },
        "d_red_flags": {
            "messages": records(
                messages.loc[messages.Message.str.contains("RED FLAG", case=False, na=False)]
            ),
            "lap_context": red_context,
        },
        "e_pit_matches": {
            "fastf1_changes": len(changes),
            "jolpica_stops": len(stops),
            "matched": len(matched),
            "match_rate": len(matched) / len(stops) if len(stops) else None,
            "pairs": matched,
            "unmatched_jolpica": missing,
            "unmatched_fastf1": [changes[i] for i in unmatched],
        },
        "g_accuracy": accuracy,
        "FastF1Generated_exists": "FastF1Generated" in original,
    }
    return evidence, laps


def run(online):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    fastf1.Cache.offline_mode(not online)
    log = {"fastf1_version": fastf1.__version__, "races": [], "schedule_errors": []}
    summary = {
        "fastf1_version": fastf1.__version__,
        "date": datetime.now(UTC).isoformat(),
        "races": [],
        "errors": [],
        "schedule_errors": [],
        "fixtures": [],
        "special_tyre_rule_race_ids": sorted(SPECIAL),
    }
    fixture_laps = []
    for year in range(2022, 2026):
        try:
            schedule = retry(
                lambda y=year: fastf1.get_event_schedule(
                    y, include_testing=False, backend="fastf1"
                ),
                online,
            )
        except Exception as exc:
            error = {"season": year, "error": f"{type(exc).__name__}: {exc}"}
            log["schedule_errors"].append(error)
            summary["schedule_errors"].append(error)
            write_json(
                OUTPUT / ("download_log.json" if online else "summary.json"),
                log if online else summary,
            )
            continue
        for _, event in schedule.iterrows():
            race_id = make_race_id(year, int(event.RoundNumber))
            print(
                f"{'Downloading' if online else 'Analyzing'} {race_id} {event.EventName}",
                flush=True,
            )
            entry = {"race_id": race_id, "status": "in_progress"}
            log["races"].append(entry)
            if online:
                write_json(OUTPUT / "download_log.json", log)
            try:
                session, messages, counts, stops, drivers = retry(
                    lambda e=event, y=year: load_race(e, y), online
                )
                entry["status"] = "ok"
                if not online:
                    evidence, laps = audit(session, messages, counts, stops, drivers, race_id)
                    summary["races"].append(evidence)
                    if not evidence["exclusion_reasons"] and len(fixture_laps) < 2:
                        fixture_laps.append(laps)
                        summary["fixtures"].append(evidence["event"])
                print(f"{race_id}: ok", flush=True)
            except Exception as exc:
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                summary["errors"].append(entry.copy())
                print(f"{race_id}: {entry['error']}", flush=True)
            write_json(
                OUTPUT / ("download_log.json" if online else "summary.json"),
                log if online else summary,
            )
    if online:
        return int(bool(log["schedule_errors"] or any(r["status"] != "ok" for r in log["races"])))
    summary["coverage"] = {
        "loaded": len(summary["races"]),
        "failed": len(summary["errors"]),
        "excluded": sum(bool(r["exclusion_reasons"]) for r in summary["races"]),
    }
    write_json(OUTPUT / "summary.json", summary)
    if len(fixture_laps) != 2:
        print("Need two dry races without red flags; fixtures were not replaced.")
        return 1
    fixture_dir = ROOT / "tests/fixtures/raw"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(fixture_laps, ignore_index=True)
    validate_raw_laps(combined)
    combined.to_parquet(fixture_dir / "laps.parquet", index=False)
    write_json(fixture_dir / "events.json", summary["fixtures"])
    return int(bool(summary["errors"] or summary["schedule_errors"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "analyze"])
    args = parser.parse_args()
    if args.command == "download":
        return run(online=True)
    # Offline mode plus a socket guard: even unexpected FastF1 fallback requests fail closed.
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("T0 analyze is offline")):
        return run(online=False)


if __name__ == "__main__":
    raise SystemExit(main())
