import numpy as np
import pandas as pd
import pytest

from pitstop.contracts import FEATURES, KEY, RAW_LAPS_DTYPES
from pitstop.features import UnresolvedInputError, build_features

RACE = "2022_01_R"
DRY_ONLY = RAW_LAPS_DTYPES | {"Compound": "string"}


def lap_counts(race_id=RACE, rows=((0.0, 38.0), (1500.0, None), (2000.0, 40.0))):
    return {
        "race_id": race_id,
        "source_url": "https://example.invalid/LapCount.jsonStream",
        "time_basis": "session seconds",
        "parser": "fastf1._api.lap_count",
        "rows": [{"Time": t, "TotalLaps": n, "CurrentLap": 1} for t, n in rows],
    }


def synthetic_race(race_id=RACE, n_laps=40, n_drivers=6, seed=0):
    """Random but valid raw laps: one stop per driver, some inaccurate laps and small gaps."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_drivers):
        pit, time = int(rng.integers(12, 28)), 1.5 * i
        for lap in range(1, n_laps + 1):
            lap_time = float(rng.uniform(88, 96))
            time += lap_time
            stint = 1 + (lap > pit)
            rows.append(
                {
                    "race_id": race_id,
                    "DriverNumber": str(10 + i),
                    "LapNumber": lap,
                    "Time": time,
                    "LapTime": lap_time,
                    "PitInTime": time - 20 if lap == pit else np.nan,
                    "PitOutTime": time - lap_time + 3 if lap == pit + 1 else np.nan,
                    "Stint": float(stint),
                    "Compound": ("SOFT", "MEDIUM")[i % 2] if stint == 1 else "HARD",
                    "TyreLife": float(lap if stint == 1 else lap - pit),
                    "Position": float(1 + (i + lap) % n_drivers),
                    "TrackStatus": "4" if lap in (12, 13) else "6" if lap == 20 else "1",
                    "IsAccurate": lap > 1 and rng.random() < 0.8 and lap not in (pit, pit + 1),
                    "FastF1Generated": False,
                }
            )
    laps = pd.DataFrame(rows)
    laps.loc[(laps.DriverNumber == "10") & (laps.LapNumber == 5), "TyreLife"] = np.nan
    laps.loc[(laps.DriverNumber == "10") & laps.LapNumber.isin([6, 7]), "Compound"] = None
    return laps.astype(DRY_ONLY)


def truncate(laps, counts, cutoff):
    """Keep every stream's observations with Time <= cutoff, for all drivers."""
    rows = [row for row in counts["rows"] if row["Time"] <= cutoff]
    return laps[laps.Time <= cutoff], {**counts, "rows": rows}


def test_known_values_on_a_handmade_race():
    rows = []
    for driver, offset in (("A", 0.0), ("B", 5.0), ("C", 30.0)):
        for lap in (1, 2, 3, 4):
            late_a = driver == "A" and lap >= 3
            rows.append(
                {
                    "race_id": RACE,
                    "DriverNumber": driver,
                    "LapNumber": lap,
                    "Time": 100.0 * lap + offset,
                    "LapTime": 110.0 if (driver, lap) == ("C", 3) else 100.0,
                    "PitInTime": 302.0 if (driver, lap) == ("B", 3) else np.nan,
                    "PitOutTime": np.nan,
                    "Stint": 2.0 if late_a else 1.0,
                    "Compound": "HARD" if late_a else "SOFT" if driver == "A" else "MEDIUM",
                    "TyreLife": float(lap),
                    "Position": 1.0 + "ABC".index(driver),
                    "TrackStatus": "41" if driver == "C" and lap == 3 else "1",
                    "IsAccurate": True,
                    "FastF1Generated": False,
                }
            )
    laps = pd.DataFrame(rows).astype(DRY_ONLY)
    features = build_features(laps, [lap_counts(rows=((0.0, 4.0),))]).set_index(
        ["driver_number", "lap_number"]
    )
    a3, b3, c3 = (features.loc[(d, 3)] for d in "ABC")
    assert (a3.gap_ahead, a3.gap_behind) == (60, 5)  # nobody ahead; B crossed lap 2 at 205
    assert (b3.gap_ahead, b3.gap_behind) == (5, 25)
    assert (c3.gap_ahead, c3.gap_behind) == (25, 60)  # nobody crossed lap 2 after 230 by 330
    assert (a3.rival_pits_2, b3.rival_pits_2, c3.rival_pits_2) == (0, 0, 1)  # B enters at 302
    assert c3.lap_time_rel == pytest.approx(0.1)  # median of nine accurate laps is 100
    assert (c3.sc, c3.vsc, a3.sc) == (1, 0, 0)
    assert (a3.stops_done, a3.mandatory_done, a3.laps_remaining) == (1, 1, 1)
    assert features.loc[("A", 2), "mandatory_done"] == 0
    assert (b3.compound_soft, b3.compound_medium, b3.compound_hard) == (0, 1, 0)


def test_truncated_run_reproduces_full_run_features():
    laps, counts = synthetic_race(), lap_counts()
    full = build_features(laps, [counts]).set_index(["driver_number", "lap_number"])
    rng = np.random.default_rng(1)
    for index in rng.choice(len(laps), size=200, replace=False):
        row = laps.iloc[index]
        cut_laps, cut_counts = truncate(laps, counts, row.Time)
        cut = build_features(cut_laps, [cut_counts]).set_index(["driver_number", "lap_number"])
        key = (row.DriverNumber, row.LapNumber)
        np.testing.assert_allclose(
            cut.loc[key, list(FEATURES)], full.loc[key, list(FEATURES)], atol=1e-6
        )


def test_rival_pit_after_cutoff_does_not_change_the_row():
    laps = synthetic_race()
    base = build_features(laps, [lap_counts()]).set_index(["driver_number", "lap_number"])
    own = laps[laps.DriverNumber == "10"].set_index("LapNumber").Time
    cutoff, window_start = own[20], own[18]
    rival = laps[laps.DriverNumber.eq("11") & laps.PitInTime.isna()]
    later = rival[rival.Time > cutoff].index[0]
    inside = rival[(rival.Time > window_start + 1) & (rival.Time <= cutoff)].index[0]
    after = laps.copy()
    after.loc[later, "PitInTime"] = cutoff + 5  # a rival still on track at T
    changed = build_features(after, [lap_counts()]).set_index(["driver_number", "lap_number"])
    pd.testing.assert_series_equal(changed.loc[("10", 20)], base.loc[("10", 20)])
    # Control: an entry inside (T(t-2), T] is counted.
    counted = laps.copy()
    counted.loc[inside, "PitInTime"] = laps.Time[inside] - 1
    counted = build_features(counted, [lap_counts()]).set_index(["driver_number", "lap_number"])
    assert counted.loc[("10", 20), "rival_pits_2"] == base.loc[("10", 20), "rival_pits_2"] + 1


def test_is_accurate_masks_time_features_but_never_removes_rows():
    laps = synthetic_race()
    inaccurate = laps.assign(IsAccurate=laps.LapNumber.eq(1))
    features = build_features(inaccurate, [lap_counts()])
    assert list(features.columns) == [*KEY, *FEATURES]
    assert len(features) == len(laps)
    assert set(zip(features.driver_number, features.lap_number, strict=True)) == set(
        zip(laps.DriverNumber, laps.LapNumber, strict=True)
    )
    one = features[features.driver_number.eq("11")]
    # Only lap 1 is accurate: lap_time_rel keeps its lap-1 value and stint_deg stays 0.
    assert one.lap_time_rel.nunique() == 1
    assert one.stint_deg.eq(0).all()


def test_stint_deg_restarts_at_a_new_stint():
    laps = synthetic_race(seed=3)
    features = build_features(laps, [lap_counts()])
    first_laps = laps[laps.Stint.eq(2) & laps.PitOutTime.notna()]
    for _, lap in first_laps.iterrows():
        row = features[
            features.driver_number.eq(lap.DriverNumber) & features.lap_number.eq(lap.LapNumber)
        ]
        assert row.stint_deg.iloc[0] == 0  # out-lap is inaccurate and starts the stint


def test_laps_remaining_uses_the_count_known_at_the_cutoff():
    laps = synthetic_race()
    features = build_features(laps, [lap_counts()]).merge(
        laps[["DriverNumber", "LapNumber", "Time"]],
        left_on=["driver_number", "lap_number"],
        right_on=["DriverNumber", "LapNumber"],
    )
    known = np.where(features.Time < 2000, 38, 40)
    assert (features.laps_remaining == known - features.lap_number).all()
    assert (features.laps_remaining < 40 - features.lap_number).any()  # earlier laps used 38


def test_unresolved_inputs_raise_with_context():
    laps = synthetic_race()
    with pytest.raises(
        UnresolvedInputError, match=r"race 2022_01_R, driver \d+, lap \d+, feature laps_remaining"
    ):
        build_features(laps, [lap_counts(rows=((5000.0, 40.0),))])
    missing_age = laps.copy()
    missing_age.loc[
        (missing_age.DriverNumber == "12") & (missing_age.LapNumber == 1), "TyreLife"
    ] = np.nan
    with pytest.raises(UnresolvedInputError, match="driver 12, lap 1, feature tyre_age"):
        build_features(missing_age, [lap_counts()])
    too_long = laps.copy()
    too_long.loc[
        (too_long.DriverNumber == "12") & too_long.LapNumber.between(10, 14), "Position"
    ] = np.nan
    with pytest.raises(UnresolvedInputError, match="driver 12, lap 13, feature position"):
        build_features(too_long, [lap_counts()])  # forward-fill stops after three laps
    with pytest.raises(UnresolvedInputError, match="laps_remaining"):
        build_features(laps, [])


def test_lap_time_rel_is_zero_until_the_first_accurate_lap_then_carried_forward():
    laps = synthetic_race()
    mine = (laps.DriverNumber == "10") & (laps.LapNumber <= 6)
    laps.loc[mine, "IsAccurate"] = laps.loc[mine, "LapNumber"].isin([5])  # first accurate: lap 5
    features = build_features(laps, [lap_counts()]).set_index(["driver_number", "lap_number"])
    rel = features.loc["10", "lap_time_rel"]
    assert (rel.loc[1:4] == 0).all()
    assert rel.loc[5] != 0 and rel.loc[6] == rel.loc[5]  # lap 6 is inaccurate: carried value


def test_pairs_outside_the_sample_set_feed_the_field_but_never_raise():
    laps = synthetic_race()
    base = build_features(laps, [lap_counts()])
    broken = laps.copy()
    broken.loc[(broken.DriverNumber == "13") & (broken.LapNumber <= 10), "TyreLife"] = np.nan
    with pytest.raises(UnresolvedInputError, match="driver 13"):
        build_features(broken, [lap_counts()])
    kept = {(RACE, str(d)) for d in range(10, 16) if d != 13}
    features = build_features(broken, [lap_counts()], kept)
    assert "13" not in set(features.driver_number)
    expected = base[base.driver_number != "13"].reset_index(drop=True)
    pd.testing.assert_frame_equal(features, expected)  # driver 13 still shapes the field
