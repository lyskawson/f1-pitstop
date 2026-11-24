"""Assign the fixed season-and-round splits before exclusions and keep test locked."""

import pandas as pd

from pitstop.contracts import make_race_id
from pitstop.ingest import D3_CLEANING, coverage, validate_laps


def build_split_manifest(events):
    """Assign raw events by season and round, including events D3 will exclude."""
    definition = {
        "train": {"seasons": [2022, 2023, 2024]},
        "val": {"season": 2025, "rounds": [1, 12]},
        "test": {"season": 2025, "rounds": [13, 24]},
    }
    assignments = {}
    for event in events:
        season, round_number = event["season"], event["round_number"]
        race_id = make_race_id(season, round_number)
        if event["race_id"] != race_id or event["session"] != "R":
            raise ValueError(f"Invalid event race identity: {event['race_id']}")
        if race_id in assignments:
            raise ValueError(f"Duplicate event: {race_id}")
        if season in definition["train"]["seasons"]:
            assignments[race_id] = "train"
            continue
        for split in ("val", "test"):
            first, last = definition[split]["rounds"]
            if season == definition[split]["season"] and first <= round_number <= last:
                assignments[race_id] = split
                break
        else:
            raise ValueError(f"Event outside fixed splits: {race_id}")
    return {
        "schema_version": 1,
        "definition": definition,
        "races": dict(sorted(assignments.items())),
    }


def apply_exclusions(laps, events, split_manifest):
    """Return retained raw rows with split metadata and a separate D3 audit.

    Supply all raw rows for each selected event. A caller can select a split
    by event before reading laps and use the same full split manifest.
    """
    validate_laps(laps)
    events = sorted(events, key=lambda event: event["race_id"])
    expected = build_split_manifest(events)
    splits = split_manifest["races"]
    if (
        split_manifest["schema_version"] != expected["schema_version"]
        or split_manifest["definition"] != expected["definition"]
        or any(splits.get(race) != split for race, split in expected["races"].items())
        or set(expected["races"]) != set(laps["race_id"])
    ):
        raise ValueError("Raw lap/event coverage or fixed split assignment differs")

    races, exclusions = coverage(laps, events, D3_CLEANING)
    special = {event["race_id"]: event["special_tyre_rule"] for event in events}
    for record in [*races, *exclusions]:
        record["split"] = splits[record["race_id"]]
        record["special_tyre_rule"] = special[record["race_id"]]
    excluded_races = {item["race_id"] for item in exclusions if item["scope"] == "race"}
    excluded_drivers = {
        (item["race_id"], item["driver_number"])
        for item in exclusions
        if item["scope"] == "race_driver"
    }
    driver_keys = pd.MultiIndex.from_frame(laps[["race_id", "DriverNumber"]])
    retained = laps.loc[
        ~laps["race_id"].isin(excluded_races) & ~driver_keys.isin(excluded_drivers)
    ].copy()
    retained["split"] = retained["race_id"].map(splits)
    retained["special_tyre_rule"] = retained["race_id"].map(special)
    audit = {"schema_version": 1, "races": races, "exclusions": exclusions}
    return retained, audit
