"""Write aligned samples and input arrays with preprocessing fitted on training rows only."""

import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pitstop.contracts import (
    FEATURES,
    KEY,
    SAMPLES_DTYPES,
    SEQ_CHANNELS,
    WINDOW,
    validate_samples,
    validate_sequences,
    validate_tabular,
)
from pitstop.features import GAP_SECONDS, MAX_GAP, build_features
from pitstop.ingest import ROOT, file_record, read_json, verify_data, write_json
from pitstop.labels import build_labels
from pitstop.splits import apply_exclusions
from pitstop.windows import build_windows

UNSCALED = ("compound_soft", "compound_medium", "compound_hard", "mandatory_done", "sc", "vsc")
SCALED = [name for name in FEATURES if name not in UNSCALED]


def fit_scaling(train_features):
    """Mean and scale (ddof 0; 1 for a constant column) of the numeric features."""
    std = train_features[SCALED].std(ddof=0)
    return {
        "columns": SCALED,
        "mean": train_features[SCALED].mean().tolist(),
        "scale": std.where(std > 0, 1.0).tolist(),
        "ddof": 0,
    }


def apply_scaling(features, scaling):
    """Standardize with frozen statistics; one-hot and boolean channels stay unscaled."""
    scaled = features.copy()
    columns = scaling["columns"]
    scaled[columns] = (features[columns] - np.array(scaling["mean"])) / np.array(scaling["scale"])
    return scaled


def make_dataset(laps, events, evidence, split_manifest):
    """Return validated samples, tabular and sequence arrays, and the train-only scaling."""
    retained, _ = apply_exclusions(laps, events, split_manifest)
    labels, _ = build_labels(retained)  # unfilled raw laps, before any feature work
    # Field statistics use every driver of a kept race; pair exclusions only choose sample rows.
    field = laps.loc[laps["race_id"].isin(retained["race_id"].unique())]
    pairs = set(zip(retained["race_id"], retained["DriverNumber"], strict=True))
    features = build_features(field, evidence, pairs)
    race_info = retained[["race_id", "split", "special_tyre_rule"]].drop_duplicates("race_id")
    samples = labels.merge(features[[*KEY, "sc", "vsc"]], on=list(KEY), validate="one_to_one")
    samples = samples.merge(race_info, on="race_id", validate="many_to_one")
    samples = samples.sort_values(list(KEY), ignore_index=True)
    samples.insert(0, "row_id", np.arange(len(samples)))
    flags = {"row_id": "int64", "sc": bool, "vsc": bool, "special_tyre_rule": bool}
    samples = samples[list(SAMPLES_DTYPES)].astype(flags)
    validate_samples(samples)

    train = samples["split"].eq("train").to_numpy()
    on_samples = samples[list(KEY)].merge(features, on=list(KEY), validate="many_to_one")
    scaling = fit_scaling(on_samples[train])
    scaled = apply_scaling(features, scaling)
    tabular = {
        "row_id": samples["row_id"].to_numpy(),
        "X": samples[list(KEY)].merge(scaled, on=list(KEY))[list(FEATURES)].to_numpy("float32"),
    }
    sequences = {"row_id": tabular["row_id"], "X": build_windows(scaled, samples)}
    validate_tabular(tabular, samples)
    validate_sequences(sequences, samples)
    return samples, tabular, sequences, scaling


def build_dataset(config_path, snapshot):
    """Build the processed snapshot from a verified raw snapshot and write it."""
    config = yaml.safe_load(Path(config_path).read_text())
    raw = ROOT / config["paths"]["raw"] / snapshot
    output = ROOT / config["paths"]["processed"]
    splits_path = ROOT / config["paths"]["splits"]
    if output.exists():
        raise ValueError(f"Processed snapshot already exists: {output}")
    verify_data(raw / "manifest.json")  # check hashes before reading any payload
    samples, tabular, sequences, scaling = make_dataset(
        pd.read_parquet(raw / "laps.parquet"),
        read_json(raw / "events.json"),
        read_json(raw / "lap_count_evidence.json"),
        read_json(splits_path),
    )
    train = samples["split"].eq("train").to_numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        samples.to_parquet(staging / "samples.parquet", index=False)
        np.savez_compressed(staging / "tabular.npz", **tabular)
        np.savez_compressed(staging / "sequences.npz", **sequences)
        git = ["git", "-C", str(ROOT)]
        preprocess = {
            "schema_version": 1,
            "data_sha256": file_record(raw / "manifest.json")["sha256"],
            "splits_sha256": file_record(splits_path)["sha256"],
            "feature_order": list(FEATURES),
            "sequence_channels": list(SEQ_CHANNELS),
            "window": WINDOW,
            "fit_split": "train",
            "fit_row_ids_sha256": hashlib.sha256(
                samples["row_id"][train].to_numpy("<i8").tobytes()
            ).hexdigest(),
            "scaling": scaling,
            "missing_values": {
                "feature_forward_fill_limit": MAX_GAP,
                "gap_seconds": int(GAP_SECONDS),
                "unresolved": "error",
                "leading_lap_time_rel": 0,
            },
            "payload_sha256": {
                name: file_record(staging / name)["sha256"]
                for name in ("samples.parquet", "tabular.npz", "sequences.npz")
            },
            "git_commit": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output([*git, "status", "--porcelain"], text=True)),
        }
        write_json(staging / "preprocess.json", preprocess)
        staging.rename(output)
    per_split = samples.groupby("split")["y"]
    return {
        "complete": True,
        "output_dir": str(output),
        "rows": per_split.size().to_dict(),
        "positives": per_split.sum().astype(int).to_dict(),
    }
