"""Compute the ordered causal feature vector using only observations available at lap end."""

import numpy as np
import pandas as pd

from pitstop.contracts import FEATURES, KEY, known_total_laps, validate_raw_laps

MAX_GAP = 3  # D3: forward-fill raw inputs for at most three laps
GAP_SECONDS = 60.0
DRY = ("SOFT", "MEDIUM", "HARD")
FILLED = ["Stint", "Compound", "TyreLife", "Position", "TrackStatus"]


class UnresolvedInputError(ValueError):
    """Required inputs are still missing after the allowed causal fill; `cases` lists them."""

    def __init__(self, cases):
        self.cases = cases
        first = cases.iloc[0]
        counts = cases.feature.value_counts().to_dict()
        super().__init__(
            f"Unresolved required inputs in {len(cases)} cells {counts}; "
            f"first: race {first.race_id}, driver {first.driver_number}, "
            f"lap {first.lap_number}, feature {first.feature}"
        )


def build_features(laps, lap_counts, retained_pairs=None):
    """Return KEY and FEATURES (float64) for the real laps of the retained pairs.

    `laps` are raw laps of all drivers (field statistics use all of them); `lap_counts` is the
    lap_count_evidence list; `retained_pairs` is a set of (race_id, driver_number), default all.
    Raises UnresolvedInputError for retained laps instead of dropping rows or inventing values.
    """
    validate_raw_laps(laps)
    entries = {entry["race_id"]: entry for entry in lap_counts}
    real = laps.loc[~laps["FastF1Generated"]]
    features = pd.concat(
        [
            _race_features(race, entries.get(rid))
            for rid, race in real.groupby("race_id", sort=True)
        ],
        ignore_index=True,
    )
    if retained_pairs is not None:
        pairs = pd.MultiIndex.from_frame(features[list(KEY[:2])])
        features = features.loc[pairs.isin(retained_pairs)].reset_index(drop=True)
    missing = features[list(FEATURES)].isna().to_numpy()
    if missing.any():
        rows, columns = np.nonzero(missing)
        cases = features.iloc[rows][list(KEY)].assign(feature=np.array(FEATURES)[columns])
        raise UnresolvedInputError(cases)
    return features


def _fill_gaps(race):
    """Forward-fill raw inputs within each driver for at most MAX_GAP laps; never backward."""
    parts = []
    for _, driver in race.groupby("DriverNumber", sort=True):
        driver = driver.sort_values("LapNumber").set_index("LapNumber", drop=False)
        driver.index.name = None
        continuous = range(driver.LapNumber.min(), driver.LapNumber.max() + 1)
        filled = driver.reindex(continuous)[FILLED].ffill(limit=MAX_GAP)
        driver[FILLED] = filled.loc[driver.index]
        parts.append(driver)
    return pd.concat(parts, ignore_index=True)


def _driver_history(driver):
    """stint_deg and mandatory_done after each lap, from laps up to and including it."""
    deg, done, seen, stint, times = [], [], set(), None, []
    for lap in driver.itertuples():
        if lap.Stint != stint:
            stint, times = lap.Stint, []
        if lap.IsAccurate and not np.isnan(lap.LapTime):
            times.append(lap.LapTime)
        deg.append(np.mean(times[-3:]) - min(times) if times else 0.0)
        seen.add(lap.Compound)
        done.append(float(len(seen & set(DRY)) >= 2))
    return deg, done


def _flag(found):
    return found.to_numpy(dtype="float64", na_value=np.nan)


def _race_features(race, entry):
    race = _fill_gaps(race)  # sorted by driver, then lap
    drivers, laps, times = (race[c].to_numpy() for c in ("DriverNumber", "LapNumber", "Time"))
    lap_time = race.LapTime.to_numpy()

    # Field median over accurate laps finished by T, forward-filled within each driver.
    accurate = (race.IsAccurate & race.LapTime.notna()).to_numpy()
    order = np.argsort(times[accurate])
    done_time, done_lap = times[accurate][order], lap_time[accurate][order]
    known = np.searchsorted(done_time, times, side="right")
    median = np.array([np.median(done_lap[:n]) if n else np.nan for n in known])
    relative = pd.Series(np.where(accurate, lap_time / median - 1, np.nan))
    relative = (
        relative.groupby(drivers).ffill().fillna(0.0).to_numpy()
    )  # 0 until the first accurate lap

    history = [_driver_history(d) for _, d in race.groupby("DriverNumber", sort=True)]
    stint_deg = np.concatenate([h[0] for h in history])
    mandatory_done = np.concatenate([h[1] for h in history])

    # Field features use session time; lap numbers only name the lap of a crossing.
    time_of = dict(zip(zip(drivers, laps, strict=True), times, strict=True))
    crossings = {lap: (times[laps == lap], drivers[laps == lap]) for lap in np.unique(laps)}
    # A pit entry counts once its in-lap row is complete (lap Time <= T), as in a truncated stream.
    pit = race.PitInTime.to_numpy()
    entered = ~np.isnan(pit)
    pit_time, pit_end, pit_driver = pit[entered], times[entered], drivers[entered]
    ahead, behind, rival_pits, remaining = [], [], [], []
    for driver, lap, cutoff in zip(drivers, laps, times, strict=True):
        crossed, who = crossings[lap]
        before = crossed[(who != driver) & (crossed <= cutoff)]
        ahead.append(np.clip(cutoff - before.max(), 0, GAP_SECONDS) if before.size else GAP_SECONDS)
        own_previous = time_of.get((driver, lap - 1))
        behind.append(GAP_SECONDS)
        if own_previous is not None:
            crossed, who = crossings[lap - 1]
            after = crossed[(who != driver) & (crossed > own_previous) & (crossed <= cutoff)]
            if after.size:
                behind[-1] = np.clip(after.min() - own_previous, 0, GAP_SECONDS)
        low = time_of.get((driver, lap - 2), np.nan) if lap > 2 else -np.inf
        hit = pit_driver[(pit_driver != driver) & (pit_time > low) & (pit_end <= cutoff)]
        rival_pits.append(np.nan if np.isnan(low) else len(set(hit)))
        try:
            remaining.append(known_total_laps(entry, cutoff, f"{driver} lap {lap}") - lap)
        except (ValueError, TypeError):  # no announcement known at T, or no evidence at all
            remaining.append(np.nan)

    status = race.TrackStatus.str
    compound_known = race.Compound.isin(DRY).to_numpy()
    columns = {
        "tyre_age": race.TyreLife.to_numpy(),
        "stops_done": race.Stint.to_numpy() - 1,
        "laps_remaining": remaining,
        "mandatory_done": mandatory_done,
        "lap_time_rel": relative,
        "stint_deg": stint_deg,
        "position": race.Position.to_numpy(),
        "gap_ahead": ahead,
        "gap_behind": behind,
        "sc": _flag(status.contains("4", regex=False)),
        "vsc": _flag(status.contains("[67]")),
        "rival_pits_2": rival_pits,
    }
    for compound in DRY:
        columns[f"compound_{compound.lower()}"] = np.where(
            compound_known, race.Compound.isin([compound]).to_numpy(), np.nan
        )
    features = pd.DataFrame(columns)[list(FEATURES)].astype("float64")
    keys = race[["race_id", "DriverNumber", "LapNumber"]]
    keys.columns = list(KEY)
    return pd.concat([keys.reset_index(drop=True), features], axis=1)
