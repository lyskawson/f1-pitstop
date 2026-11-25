import numpy as np
import pandas as pd

from pitstop.contracts import (
    FEATURES,
    KEY,
    SEQ_CHANNELS,
    WINDOW,
    validate_samples,
    validate_sequences,
)
from pitstop.windows import build_windows

LAP, DRIVER, RACE = (FEATURES.index(name) for name in ("tyre_age", "position", "laps_remaining"))


def encoded_features(skip=()):
    """Feature table that stores lap, driver and race codes in three channels."""
    rows = [
        (race, driver, lap)
        for race in ("2022_01_R", "2022_02_R")
        for driver in ("1", "2")
        for lap in range(1, 21)
        if (race, driver, lap) not in skip
    ]
    table = pd.DataFrame(rows, columns=list(KEY))
    table[list(FEATURES)] = 0.0
    table["tyre_age"] = table.lap_number.astype(float)
    table["position"] = table.driver_number.astype(float)
    table["laps_remaining"] = table.race_id.str[6:7].astype(float)
    return table


def samples_for(table):
    samples = table[list(KEY)].copy()
    samples.insert(0, "row_id", np.arange(len(samples)))
    return samples.assign(y=np.int8(0), split="train", sc=False, vsc=False, special_tyre_rule=False)


def test_windows_end_at_t_and_stay_in_one_race_and_driver():
    table = encoded_features()
    samples = samples_for(table)
    windows = build_windows(table, samples)
    validate_samples(samples)
    validate_sequences({"row_id": samples.row_id.to_numpy(), "X": windows}, samples)
    assert windows.shape == (len(samples), WINDOW, len(SEQ_CHANNELS))
    assert (windows[:, -1, LAP] == samples.lap_number).all()  # last step is lap t
    real = windows[:, :, -1] == 0
    assert (real.sum(axis=1) == np.minimum(samples.lap_number, WINDOW)).all()
    laps = np.diff(windows[:, :, LAP], axis=1)
    assert (laps[real[:, 1:]] == 1).all()  # consecutive laps (padding is not compared)
    assert (
        windows[:, :, DRIVER][real]
        == np.repeat(samples.driver_number.astype(float), real.sum(axis=1))
    ).all()
    assert (
        windows[:, :, RACE][real]
        == np.repeat(samples.race_id.str[6:7].astype(float), real.sum(axis=1))
    ).all()
    assert (windows[:, :, :-1][~real] == 0).all()


def test_history_that_is_not_a_sample_row_gives_context_and_gaps_stop_the_window():
    table = encoded_features(skip={("2022_01_R", "1", 10)})
    samples = samples_for(table)
    chosen = samples[(samples.race_id == "2022_01_R") & (samples.driver_number == "1")]
    # Lap 10 has no feature row, so the windows of laps 12 and 20 stop before it.
    for lap, expected in ((12, [11, 12]), (20, list(range(11, 21)))):
        window = build_windows(table, chosen[chosen.lap_number == lap])[0]
        assert list(window[window[:, -1] == 0][:, LAP]) == expected


def test_changing_data_after_t_does_not_change_the_window():
    table = encoded_features()
    samples = samples_for(table)
    changed = table.copy()
    future = changed.lap_number > 10
    changed.loc[future, list(FEATURES)] = 99.0
    early = samples[samples.lap_number <= 10]
    np.testing.assert_array_equal(build_windows(table, early), build_windows(changed, early))
