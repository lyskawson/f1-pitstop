"""Shared artifact schemas and validators; see docs/contracts.md for stage interfaces."""

import re
from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

KEY = ("race_id", "driver_number", "lap_number")
RAW_KEY = ("race_id", "DriverNumber", "LapNumber")
FEATURES = (
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
SEQ_CHANNELS = (*FEATURES, "is_pad")
WINDOW = 15
SPLITS = ("train", "val", "test")
RACE_ID_PATTERN = r"[1-9][0-9]{3}_(?:0[1-9]|[1-9][0-9])_R"
RUN_ID_PATTERN = r"[0-9]{8}T[0-9]{6}Z_(?:constant|logreg|lightgbm|mlp|gru|tcn)_s[0-9]+_[0-9a-f]{8}"

# "str" accepts pandas string columns or object columns containing only strings.
# Other dtypes are exact NumPy dtypes; nullable raw values are listed explicitly.
RAW_LAPS_DTYPES = {
    "race_id": "str",
    "DriverNumber": "str",
    "LapNumber": "int64",
    "Time": "float64",
    "LapTime": "float64",
    "PitInTime": "float64",
    "PitOutTime": "float64",
    "Stint": "float64",
    "Compound": "str",
    "TyreLife": "float64",
    "Position": "float64",
    "TrackStatus": "str",
    "IsAccurate": "bool",
    "FastF1Generated": "bool",
}
RAW_NULLABLE = (
    "Time",
    "LapTime",
    "PitInTime",
    "PitOutTime",
    "Stint",
    "Compound",
    "TyreLife",
    "Position",
    "TrackStatus",
)
SAMPLES_DTYPES = {
    "row_id": "int64",
    "race_id": "str",
    "driver_number": "str",
    "lap_number": "int64",
    "y": "int8",
    "split": "str",
    "sc": "bool",
    "vsc": "bool",
    "special_tyre_rule": "bool",
}
PREDICTIONS_DTYPES = {
    "race_id": "str",
    "driver_number": "str",
    "lap_number": "int64",
    "run_id": "str",
    "split": "str",
    "y": "int8",
    "score": "float64",
    "target_lap": "int64",
}
TABULAR_SPEC = {"row_id": ("int64", (None,)), "X": ("float32", (None, len(FEATURES)))}
SEQUENCES_SPEC = {
    "row_id": ("int64", (None,)),
    "X": ("float32", (None, WINDOW, len(SEQ_CHANNELS))),
}


def validate_race_id(race_id: str) -> None:
    """Require a four-digit year, round 01..99, and Race session suffix."""
    if not isinstance(race_id, str) or re.fullmatch(RACE_ID_PATTERN, race_id) is None:
        raise ValueError(f"race_id must match YYYY_RR_R (round 01..99), got {race_id!r}")


def make_race_id(year: int, round_number: int) -> str:
    """Build a Race identifier, for example make_race_id(2022, 10)."""
    for name, value in (("year", year), ("round_number", round_number)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    race_id = f"{year:04d}_{round_number:02d}_R"
    validate_race_id(race_id)
    return race_id


def _validate_frame(
    frame: pd.DataFrame,
    dtypes: Mapping[str, str],
    name: str,
    key: tuple[str, ...],
    nullable: tuple[str, ...] = (),
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")
    missing = sorted(set(dtypes) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    for column, expected in dtypes.items():
        values = frame[column]
        if expected == "str":
            string_dtype = is_string_dtype(values.dtype) or is_object_dtype(values.dtype)
            if not string_dtype or not values.dropna().map(lambda x: isinstance(x, str)).all():
                raise ValueError(f"{name}.{column} must have dtype str with string values")
        elif values.dtype != np.dtype(expected):
            raise ValueError(f"{name}.{column} must have dtype {expected}, got {values.dtype}")
        if column not in nullable and values.isna().any():
            raise ValueError(f"{name}.{column} must not contain nulls")
        if expected.startswith("float") and np.isinf(values.to_numpy()).any():
            raise ValueError(f"{name}.{column} must not contain infinity")
    if frame.duplicated(list(key)).any():
        raise ValueError(f"{name} has duplicate key {key}")
    if not frame["race_id"].str.fullmatch(RACE_ID_PATTERN).all():
        raise ValueError(f"{name}.race_id must match YYYY_RR_R (round 01..99)")
    if frame[key[1]].str.len().eq(0).any():
        raise ValueError(f"{name}.{key[1]} must not be empty")
    if frame[key[2]].lt(1).any():
        raise ValueError(f"{name}.{key[2]} must be positive")


def _validate_targets(frame: pd.DataFrame, name: str) -> None:
    if not frame["y"].isin((0, 1)).all():
        raise ValueError(f"{name}.y must be in {{0, 1}}")
    if not frame["split"].isin(SPLITS).all():
        raise ValueError(f"{name}.split must be in {SPLITS}")


def validate_raw_laps(laps: pd.DataFrame) -> None:
    """Validate normalized raw FastF1 columns, retaining missing measurements."""
    _validate_frame(laps, RAW_LAPS_DTYPES, "raw_laps", RAW_KEY, RAW_NULLABLE)


def validate_samples(samples: pd.DataFrame) -> None:
    """Validate sample keys, stable row identifiers, binary labels, and split names."""
    _validate_frame(samples, SAMPLES_DTYPES, "samples", KEY)
    _validate_targets(samples, "samples")
    if samples["row_id"].duplicated().any():
        raise ValueError("samples.row_id must be unique")
    if samples["row_id"].lt(0).any():
        raise ValueError("samples.row_id must be nonnegative")


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Validate per-run probabilities and next-lap targets."""
    _validate_frame(predictions, PREDICTIONS_DTYPES, "predictions", (*KEY, "run_id"))
    _validate_targets(predictions, "predictions")
    if not predictions["run_id"].str.fullmatch(RUN_ID_PATTERN).all():
        raise ValueError("predictions.run_id must match YYYYMMDDTHHMMSSZ_<model>_s<seed>_<sha8>")
    if not predictions["score"].between(0, 1).all():
        raise ValueError("predictions.score must be a probability in [0, 1]")
    if not predictions["target_lap"].eq(predictions["lap_number"] + 1).all():
        raise ValueError("predictions.target_lap must equal lap_number + 1")


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    samples: pd.DataFrame,
    spec: Mapping[str, tuple[str, tuple[int | None, ...]]],
    name: str,
) -> None:
    validate_samples(samples)
    if not isinstance(arrays, Mapping):
        raise ValueError(f"{name} must be a mapping of NumPy arrays")
    missing = sorted(set(spec) - set(arrays))
    if missing:
        raise ValueError(f"{name} missing required arrays: {missing}")
    for field, (dtype, dimensions) in spec.items():
        array = arrays[field]
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype):
            raise ValueError(f"{name}.{field} must be a NumPy array with dtype {dtype}")
        shape = tuple(len(samples) if size is None else size for size in dimensions)
        if array.shape != shape:
            raise ValueError(f"{name}.{field} must have shape {shape}, got {array.shape}")
    if np.unique(arrays["row_id"]).size != len(arrays["row_id"]):
        raise ValueError(f"{name}.row_id must be unique")
    if not np.array_equal(arrays["row_id"], samples["row_id"].to_numpy()):
        raise ValueError(f"{name}.row_id must align with samples.row_id in the same order")
    if not np.isfinite(arrays["X"]).all():
        raise ValueError(f"{name}.X must contain finite values only")


def validate_tabular(arrays: Mapping[str, np.ndarray], samples: pd.DataFrame) -> None:
    """Require one finite float32 feature vector per sample, in sample order."""
    _validate_arrays(arrays, samples, TABULAR_SPEC, "tabular")


def validate_sequences(arrays: Mapping[str, np.ndarray], samples: pd.DataFrame) -> None:
    """Require aligned finite windows with binary left-padding and a real final step."""
    _validate_arrays(arrays, samples, SEQUENCES_SPEC, "sequences")
    inputs = arrays["X"]
    is_pad = inputs[:, :, -1]
    if not np.isin(is_pad, (0, 1)).all():
        raise ValueError("sequences.is_pad must be binary")
    if (np.diff(is_pad, axis=1) > 0).any():
        raise ValueError("sequences.is_pad must mark left-padding only")
    if is_pad[:, -1].any():
        raise ValueError("sequences final step must be a real lap")
    if np.any(inputs[:, :, :-1][is_pad == 1] != 0):
        raise ValueError("sequences padded feature values must be zero")
