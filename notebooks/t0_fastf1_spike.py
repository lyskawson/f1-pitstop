"""T0 exploration: human download, then cache-only evidence and real fixtures."""

import argparse
import json
import logging
import re
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
SEASONS = (2022, 2023, 2024, 2025)


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


def load_schedule(year, online):
    """Retry transient schedule failures; offline analysis never waits or downloads."""
    attempts = 3 if online else 1
    for attempt in range(1, attempts + 1):
        try:
            schedule = retry(
                lambda: fastf1.get_event_schedule(year, include_testing=False, backend="fastf1"),
                online,
            )
            if schedule.empty or not schedule.RoundNumber.gt(0).all():
                raise ValueError("Empty or invalid race schedule")
            return schedule
        except Exception as exc:
            print(f"Season {year}: schedule attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt == attempts:
                raise
            delay = 60 * attempt
            print(f"Retrying season {year} schedule in {delay} seconds.", flush=True)
            time.sleep(delay)


def load_race(event, year):
    session = event.get_session("R")
    session.load(laps=True, messages=True, telemetry=False, weather=True)
    if session.laps.empty:
        raise ValueError("No laps loaded")
    # Access each required source: Session.load can suppress individual failures.
    messages = session.race_control_messages
    if messages.empty:
        raise ValueError("No race-control messages loaded")
    if session.weather_data.empty:
        raise ValueError("No weather data loaded")
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
            if name in ("Compound", "TrackStatus"):
                # FastF1 can already encode a missing measurement as literal text.
                laps[name] = laps[name].mask(laps[name].isin(["nan", "None"]), pd.NA)
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
    timed = original[
        [
            "DriverNumber",
            "LapNumber",
            "LapStartTime",
            "Time",
            "PitInTime",
            "PitOutTime",
            "Stint",
            "Compound",
            "FastF1Generated",
        ]
    ].copy()
    timed["LapNumber"] = timed.LapNumber.astype("int64")
    for name in ("LapStartTime", "Time", "PitInTime", "PitOutTime"):
        timed[name] = timed[name].dt.total_seconds()
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
            "outside_lap": records(timed.loc[outside]),
            "examples": records(timed.loc[known].head(3)),
        }
    penalty_messages = messages.loc[
        messages.Message.str.contains(
            r"DRIVE[ -]?THROUGH|STOP[ /-]*(?:AND[ /-]*)?GO", case=False, na=False
        )
    ]
    penalties = []
    for message in records(penalty_messages):
        driver = message.get("RacingNumber")
        if not driver:
            car = re.search(r"\bCAR\s+(\d+)\b", message["Message"])
            driver = car.group(1) if car else None
        lap = message.get("Lap")
        # Message times are UTC while raw lap times are session seconds.
        # Use the explicit message lap; do not guess which stop served a penalty.
        penalties.append(
            {
                "message": message,
                "driver_number": driver,
                "context": lap_context(laps, driver, lap, radius=5) if driver and lap else [],
                # Served messages may arrive much later than the actual pass.
                "all_pit_passes": [
                    lap_context(laps, driver, int(number), radius=1)
                    for number in laps.loc[
                        laps.DriverNumber.eq(driver) & laps.PitInTime.notna(), "LapNumber"
                    ]
                ],
                "interpretation": "Review issuance/served wording and surrounding stops manually",
            }
        )
    accuracy = {}
    previous_status = original.groupby("DriverNumber").TrackStatus.shift(1)
    accuracy_laps = original.assign(PreviousTrackStatus=previous_status)
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
                accuracy_laps.loc[
                    mask & original.IsAccurate,
                    [
                        "DriverNumber",
                        "LapNumber",
                        "TrackStatus",
                        "PreviousTrackStatus",
                        "IsAccurate",
                    ],
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
    # Jolpica permanent numbers can reflect a later season. Use the historical
    # number from this race's laps, joined by driver code, never session.results.
    numbers = dict(zip(original.Driver, original.DriverNumber, strict=True))
    mapping = {row.driverId: numbers.get(row.driverCode) for row in drivers.itertuples()}
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
        "weather_rows": len(session.weather_data),
        "driver_coverage": [
            {
                "driver_number": driver,
                "rows": len(group),
                "first_lap": int(group.LapNumber.min()),
                "last_lap": int(group.LapNumber.max()),
                "missing": {name: int(value) for name, value in group.isna().sum().items()},
            }
            for driver, group in laps.groupby("DriverNumber")
        ],
        "a_pit_boundaries": boundaries,
        "b_penalties": penalties,
        "c_total_laps": {
            "loader": event["total_laps"],
            "maximum_recorded_lap": int(laps.LapNumber.max()),
            "maximum_non_generated_lap": int(clean.LapNumber.max()) if not clean.empty else None,
            "earliest_recorded_lap_end": (
                float(laps.Time.min()) if laps.Time.notna().any() else None
            ),
            "loader_matches_last_nonnull": event["total_laps"] == int(announced.iloc[-1]),
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
            "unmapped_driver_ids": sorted(
                {s["driverId"] for s in records(stops) if mapping.get(s["driverId"]) is None}
            ),
            "per_driver": [
                {
                    "driver_number": driver,
                    "fastf1_changes": sum(c["DriverNumber"] == driver for c in changes),
                    "jolpica_stops": sum(
                        mapping.get(s["driverId"]) == driver for s in records(stops)
                    ),
                    "matched": sum(p["fastf1"]["DriverNumber"] == driver for p in matched),
                }
                for driver in sorted(laps.DriverNumber.unique())
            ],
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
    for year in SEASONS:
        try:
            schedule = load_schedule(year, online)
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
        log["seasons_requested"] = list(SEASONS)
        log["missing_seasons"] = [error["season"] for error in log["schedule_errors"]]
        failed = [r["race_id"] for r in log["races"] if r["status"] != "ok"]
        log["complete"] = not log["missing_seasons"] and not failed
        write_json(OUTPUT / "download_log.json", log)
        if log["missing_seasons"]:
            print(f"ERROR: missing season schedules: {log['missing_seasons']}", flush=True)
        if failed:
            print(f"ERROR: incomplete races: {', '.join(failed)}", flush=True)
        print(
            "Download complete." if log["complete"] else "Download INCOMPLETE; rerun download.",
            flush=True,
        )
        return int(not log["complete"])
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
