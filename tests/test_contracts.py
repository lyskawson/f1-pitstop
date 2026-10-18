"""Offline contract tests using only in-memory toy data, never FastF1 downloads."""

import argparse
import io
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pytest_socket import SocketBlockedError

from pitstop.cli import build_parser, main
from pitstop.contracts import (
    FEATURES,
    KEY,
    PREDICTIONS_DTYPES,
    RAW_KEY,
    RAW_LAPS_DTYPES,
    SAMPLES_DTYPES,
    SEQ_CHANNELS,
    SPLITS,
    WINDOW,
    make_race_id,
    validate_predictions,
    validate_race_id,
    validate_raw_laps,
    validate_samples,
    validate_sequences,
    validate_tabular,
)

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    "download",
    "verify-data",
    "build-dataset",
    "train",
    "predict",
    "evaluate",
    "export-demo",
)
RUN_ID = "20261019T120000Z_mlp_s42_abcdef12"


def make_raw_laps():
    return pd.DataFrame(
        {
            "race_id": ["2022_10_R"] * 4,
            "DriverNumber": ["44"] * 4,
            "LapNumber": [18, 19, 20, 21],
            "Time": [1800, 1890, 1980, 2070],
            "LapTime": [90, 90, 90, 90],
            "PitInTime": [np.nan, np.nan, 1979, np.nan],
            "PitOutTime": [np.nan, np.nan, np.nan, 1988],
            "Stint": [1, 1, 1, 2],
            "Compound": ["SOFT", "SOFT", "SOFT", "MEDIUM"],
            "TyreLife": [10, 11, 12, 1],
            "Position": [2, 2, 2, 3],
            "TrackStatus": ["1", "1", "4", "1"],
            "IsAccurate": [True, True, False, False],
            "FastF1Generated": [False] * 4,
        }
    ).astype(RAW_LAPS_DTYPES)


def make_samples():
    return pd.DataFrame(
        {
            "row_id": [7, 12, 21],
            "race_id": ["2022_10_R", "2025_01_R", "2025_13_R"],
            "driver_number": ["44"] * 3,
            "lap_number": [18, 19, 20],
            "y": [0, 1, 0],
            "split": ["train", "val", "test"],
            "sc": [False, False, True],
            "vsc": [False] * 3,
            "special_tyre_rule": [False] * 3,
        }
    ).astype(SAMPLES_DTYPES)


def make_predictions():
    frame = make_samples()[[*KEY, "split", "y"]].copy()
    frame["run_id"] = RUN_ID
    frame["score"] = [0.0, 0.5, 1.0]
    frame["target_lap"] = frame["lap_number"] + 1
    return frame.astype(PREDICTIONS_DTYPES)


FRAME_CASES = (
    (validate_raw_laps, make_raw_laps, RAW_LAPS_DTYPES, RAW_KEY),
    (validate_samples, make_samples, SAMPLES_DTYPES, KEY),
    (validate_predictions, make_predictions, PREDICTIONS_DTYPES, KEY),
)
ARRAY_CASES = ((validate_tabular, (3, 15)), (validate_sequences, (3, 15, 16)))


def make_arrays(shape):
    return {"row_id": make_samples()["row_id"].to_numpy(), "X": np.zeros(shape, dtype="float32")}


def test_feature_order_and_constants():
    assert KEY == ("race_id", "driver_number", "lap_number")
    assert FEATURES == (
        "tyre_age",
        "compound_soft",
        "compound_medium",
        "compound_hard",
        "stops_done",
        "laps_remaining",
        "mandatory_done",
        "lap_time_rel",
        "stint_deg",
        "position",
        "gap_ahead",
        "gap_behind",
        "sc",
        "vsc",
        "rival_pits_2",
    )
    assert len(FEATURES) == len(set(FEATURES)) == 15
    assert SEQ_CHANNELS == (*FEATURES, "is_pad")
    assert len(SEQ_CHANNELS) == 16
    assert WINDOW == 15
    assert SPLITS == ("train", "val", "test")


def test_race_id_helpers():
    assert make_race_id(2022, 10) == "2022_10_R"
    assert make_race_id(2025, 1) == "2025_01_R"
    validate_race_id("2025_24_R")


@pytest.mark.parametrize("race_id", ["2025_1_R", "2025_00_R", "2025_01_S", "2025_01_R\n", 1])
def test_invalid_race_id_helper(race_id):
    with pytest.raises(ValueError, match="race_id"):
        validate_race_id(race_id)


@pytest.mark.parametrize(
    "year,round_number", [(2025, 0), (2025, 100), (202, 1), (True, 1), (2025, "1")]
)
def test_invalid_race_id_builder(year, round_number):
    with pytest.raises(ValueError):
        make_race_id(year, round_number)


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_valid_frame_and_parquet_round_trip(validator, factory, dtypes, key):
    frame = factory()
    validator(frame)
    payload = io.BytesIO()
    frame.to_parquet(payload, index=False)
    payload.seek(0)
    validator(pd.read_parquet(payload))


@pytest.mark.parametrize(
    "validator,factory,column",
    [
        (validator, factory, column)
        for validator, factory, dtypes, _ in FRAME_CASES
        for column in dtypes
    ],
)
def test_missing_frame_columns(validator, factory, column):
    with pytest.raises(ValueError, match="missing required columns"):
        validator(factory().drop(columns=column))


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_wrong_frame_dtype(validator, factory, dtypes, key):
    frame = factory()
    frame[key[2]] = frame[key[2]].astype("float64")
    with pytest.raises(ValueError, match="dtype int64"):
        validator(frame)


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_driver_numbers_are_strings(validator, factory, dtypes, key):
    frame = factory()
    frame[key[1]] = frame[key[1]].astype(object)
    frame.loc[0, key[1]] = 44
    with pytest.raises(ValueError, match="dtype str"):
        validator(frame)


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_duplicate_frame_key(validator, factory, dtypes, key):
    frame = factory()
    with pytest.raises(ValueError, match="duplicate key"):
        validator(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_bad_frame_race_id(validator, factory, dtypes, key):
    frame = factory()
    frame.loc[0, "race_id"] = "2025_1_R"
    with pytest.raises(ValueError, match="race_id"):
        validator(frame)


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_null_required_column(validator, factory, dtypes, key):
    frame = factory()
    frame["race_id"] = frame["race_id"].astype("string")
    frame.loc[0, "race_id"] = pd.NA
    with pytest.raises(ValueError, match="must not contain nulls"):
        validator(frame)


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_positive_lap_numbers(validator, factory, dtypes, key):
    frame = factory()
    frame.loc[0, key[2]] = 0
    with pytest.raises(ValueError, match="must be positive"):
        validator(frame)


@pytest.mark.parametrize("validator,factory,dtypes,key", FRAME_CASES)
def test_wrong_frame_container(validator, factory, dtypes, key):
    with pytest.raises(ValueError, match="DataFrame"):
        validator(factory().to_dict())


def test_raw_missing_measurements_remain_nullable():
    frame = make_raw_laps()
    frame.loc[0, ["Compound", "TrackStatus"]] = None
    frame.loc[0, ["Time", "LapTime", "TyreLife", "Stint", "Position"]] = np.nan
    validate_raw_laps(frame)


def test_raw_times_must_be_float_seconds():
    frame = make_raw_laps()
    frame["Time"] = pd.to_timedelta(frame["Time"], unit="s")
    with pytest.raises(ValueError, match="Time must have dtype float64"):
        validate_raw_laps(frame)


@pytest.mark.parametrize(
    "validator,factory",
    [(validate_samples, make_samples), (validate_predictions, make_predictions)],
)
@pytest.mark.parametrize(
    "column,value,message", [("y", 2, "y must be in"), ("split", "validation", "split must be in")]
)
def test_bad_labels_and_splits(validator, factory, column, value, message):
    frame = factory()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validator(frame)


@pytest.mark.parametrize("row_id,message", [(7, "unique"), (-1, "nonnegative")])
def test_bad_sample_row_ids(row_id, message):
    frame = make_samples()
    frame.loc[1, "row_id"] = row_id
    with pytest.raises(ValueError, match=message):
        validate_samples(frame)


@pytest.mark.parametrize("score", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_invalid_probabilities(score):
    frame = make_predictions()
    frame.loc[0, "score"] = score
    with pytest.raises(ValueError, match="score"):
        validate_predictions(frame)


def test_next_lap_target():
    frame = make_predictions()
    frame.loc[0, "target_lap"] = frame.loc[0, "lap_number"]
    with pytest.raises(ValueError, match="target_lap must equal lap_number"):
        validate_predictions(frame)


def test_prediction_key_includes_run_id():
    frame = make_predictions()
    other_run = frame.copy()
    other_run["run_id"] = "20261019T120001Z_gru_s43_abcdef12"
    validate_predictions(pd.concat([frame, other_run], ignore_index=True))


def test_bad_run_id():
    frame = make_predictions()
    frame.loc[0, "run_id"] = "temporary-run"
    with pytest.raises(ValueError, match="run_id must match"):
        validate_predictions(frame)


@pytest.mark.parametrize("validator,shape", ARRAY_CASES)
def test_valid_arrays_and_npz_round_trip(validator, shape):
    arrays = make_arrays(shape)
    validator(arrays, make_samples())
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    payload.seek(0)
    with np.load(payload, allow_pickle=False) as loaded:
        validator(loaded, make_samples())


@pytest.mark.parametrize("validator,shape", ARRAY_CASES)
@pytest.mark.parametrize("column", ["row_id", "X"])
def test_missing_array(validator, shape, column):
    arrays = make_arrays(shape)
    del arrays[column]
    with pytest.raises(ValueError, match="missing required arrays"):
        validator(arrays, make_samples())


@pytest.mark.parametrize("validator,shape", ARRAY_CASES)
@pytest.mark.parametrize("column,dtype", [("row_id", "int32"), ("X", "float64")])
def test_wrong_array_dtype(validator, shape, column, dtype):
    arrays = make_arrays(shape)
    arrays[column] = arrays[column].astype(dtype)
    with pytest.raises(ValueError, match="dtype"):
        validator(arrays, make_samples())


@pytest.mark.parametrize("validator,shape", ARRAY_CASES)
@pytest.mark.parametrize("problem", ["rows", "channels", "dimensions", "row_id"])
def test_wrong_array_shape(validator, shape, problem):
    arrays = make_arrays(shape)
    if problem == "rows":
        arrays["X"] = arrays["X"][:2]
    elif problem == "channels":
        arrays["X"] = arrays["X"][..., :-1]
    elif problem == "dimensions":
        arrays["X"] = arrays["X"].reshape(-1)
    else:
        arrays["row_id"] = arrays["row_id"][:, None]
    with pytest.raises(ValueError, match="shape"):
        validator(arrays, make_samples())


def test_wrong_sequence_window_length():
    arrays = make_arrays((3, 14, 16))
    with pytest.raises(ValueError, match="shape"):
        validate_sequences(arrays, make_samples())


@pytest.mark.parametrize("validator,shape", ARRAY_CASES)
@pytest.mark.parametrize("problem", ["duplicate", "reordered", "unknown"])
def test_array_row_identity(validator, shape, problem):
    arrays = make_arrays(shape)
    if problem == "duplicate":
        arrays["row_id"][1] = arrays["row_id"][0]
    elif problem == "reordered":
        arrays["row_id"] = arrays["row_id"][::-1]
    else:
        arrays["row_id"] += 100
    with pytest.raises(ValueError, match="unique|align"):
        validator(arrays, make_samples())


@pytest.mark.parametrize("validator,shape", ARRAY_CASES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_array_inputs_are_finite(validator, shape, value):
    arrays = make_arrays(shape)
    arrays["X"].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        validator(arrays, make_samples())


def test_valid_left_padding():
    arrays = make_arrays((3, 15, 16))
    arrays["X"][0, :14, -1] = 1
    validate_sequences(arrays, make_samples())


@pytest.mark.parametrize("problem", ["nonbinary", "interior", "last_step", "nonzero_features"])
def test_invalid_sequence_padding(problem):
    arrays = make_arrays((3, 15, 16))
    if problem == "nonbinary":
        arrays["X"][0, 0, -1] = 0.5
    elif problem == "interior":
        arrays["X"][0, 1, -1] = 1
    elif problem == "last_step":
        arrays["X"][0, :, -1] = 1
    else:
        arrays["X"][0, 0, -1] = 1
        arrays["X"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="binary|left-padding|real lap|must be zero"):
        validate_sequences(arrays, make_samples())


def test_cli_exposes_exactly_seven_commands():
    actions = [
        action
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(actions) == 1
    assert tuple(actions[0].choices) == COMMANDS


@pytest.mark.parametrize("command", COMMANDS)
def test_cli_stub_message(command, capsys):
    assert main([command]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"pitstop {command}: not implemented yet\n"


def test_cli_process_exit_status():
    result = subprocess.run(
        [sys.executable, "-m", "pitstop.cli", "train"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "pitstop train: not implemented yet\n"


def test_console_entry_point():
    result = subprocess.run(["pitstop", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "export-demo" in result.stdout


def test_all_configs_load():
    files = sorted((ROOT / "configs").glob("*.yaml"))
    assert {file.stem for file in files} == {
        "data",
        "constant",
        "logreg",
        "lightgbm",
        "mlp",
        "gru",
        "tcn",
    }
    for file in files:
        config = yaml.safe_load(file.read_text())
        assert isinstance(config, dict)
        assert config["seed"] == 42
        if file.stem == "data":
            assert config["window"] == WINDOW
        else:
            assert config["model"] == file.stem
            assert set(config["data"]) == {"samples", "tabular", "sequences", "preprocess"}
            assert config["selection"] == {"split": "val", "metric": "average_precision"}
            assert config["threshold"] == {"split": "val", "metric": "f1", "tie_break": "largest"}


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
def test_internet_socket_is_blocked(family):
    with pytest.raises(SocketBlockedError):
        socket.socket(family, socket.SOCK_STREAM)


def test_unix_socket_is_allowed():
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as local:
        assert local.family == socket.AF_UNIX
