"""Build next-lap tyre-change labels by raw-lap self-joins before feature filtering."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pitstop.contracts import KEY, RAW_KEY, validate_raw_laps

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
LABELS = {
    "no_pit_entry": 0,
    "confirmed_no_change_transit": 0,
    "compound_change": 1,
    "same_compound_age_discontinuity": 1,
}


def build_labels(
    laps: pd.DataFrame,
    *,
    confirmed_transits: Mapping[tuple[str, str, int], str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return accepted KEY/y rows and separate evidence for every real lap t.

    Pass unfilled raw laps. Confirmations map an exact (race, driver, entry
    lap) to a nonblank source reference for a verified no-change transit.
    The caller must verify that source and preserve its provenance. Supply
    none if the available payloads cannot substantiate an exact confirmation.
    """
    validate_raw_laps(laps)
    real = laps.loc[~laps["FastF1Generated"]].reset_index(drop=True)
    evidence = real[list(RAW_KEY)].copy()
    for offset, suffix, pit_field in ((1, "in", "PitInTime"), (2, "out", "PitOutTime")):
        fields = [pit_field, "Compound", "TyreLife", "Stint"]
        successor = real[[*RAW_KEY, *fields]].rename(
            columns={field: f"{field}_{suffix}" for field in fields}
        )
        successor["LapNumber"] -= offset
        present = f"has_{suffix}_lap"
        successor[present] = True
        evidence = evidence.merge(successor, on=list(RAW_KEY), how="left", validate="one_to_one")
        evidence[present] = evidence[present].eq(True)

    evidence = evidence.rename(columns=dict(zip(RAW_KEY, KEY, strict=True)))
    evidence["target_lap"] = evidence["lap_number"] + 1
    confirmed_transits = confirmed_transits or {}
    if any(not isinstance(ref, str) or not ref.strip() for ref in confirmed_transits.values()):
        raise ValueError("Confirmed transits require a nonblank source reference")
    entries = evidence[["race_id", "driver_number", "target_lap"]]
    evidence["transit_source"] = pd.array(
        [confirmed_transits.get(key) for key in entries.itertuples(index=False, name=None)],
        dtype="string",
    )
    evidence["reason_code"] = _label_reasons(evidence)
    evidence["y"] = evidence["reason_code"].map(LABELS).astype("Int8")
    evidence = evidence.sort_values(list(KEY), ignore_index=True)
    labels = evidence.loc[evidence["y"].notna(), [*KEY, "y"]].copy()
    labels["y"] = labels["y"].astype("int8")
    return labels.reset_index(drop=True), evidence


def _label_reasons(evidence):
    age_in, age_out = evidence["TyreLife_in"], evidence["TyreLife_out"]
    known_tyres = (
        evidence["Compound_in"].isin(COMPOUNDS)
        & evidence["Compound_out"].isin(COMPOUNDS)
        & age_in.ge(1)
        & age_out.ge(1)
        & age_in.mod(1).eq(0)
        & age_out.mod(1).eq(0)
    )
    changed_compound = evidence["Compound_in"].ne(evidence["Compound_out"]).fillna(False)
    continuous = ~changed_compound & age_out.eq(age_in + 1)
    stint_in = evidence["Stint_in"]
    next_stint = stint_in.ge(1) & stint_in.mod(1).eq(0) & evidence["Stint_out"].eq(stint_in + 1)
    confirmed = evidence["transit_source"].notna()

    return np.select(
        [
            ~evidence["has_in_lap"],
            evidence["PitInTime_in"].isna(),
            ~evidence["has_out_lap"],
            ~evidence["PitOutTime_out"].gt(evidence["PitInTime_in"]),
            ~known_tyres,
            continuous & confirmed,
            confirmed,
            continuous,
            ~next_stint,
            changed_compound,
        ],
        [
            "missing_target_lap",
            "no_pit_entry",
            "missing_exit_lap",
            "missing_or_invalid_pit_exit",
            "missing_or_unknown_tyre_evidence",
            "confirmed_no_change_transit",
            "conflicting_transit_evidence",
            "unconfirmed_continuous_tyres",
            "unconfirmed_stint_transition",
            "compound_change",
        ],
        default="same_compound_age_discontinuity",
    )
