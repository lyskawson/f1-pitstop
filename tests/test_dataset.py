import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from test_features import lap_counts, synthetic_race

import pitstop.dataset
from pitstop.cli import main
from pitstop.contracts import FEATURES, KEY, SEQ_CHANNELS, validate_samples
from pitstop.dataset import SCALED, fit_scaling, make_dataset
from pitstop.features import UnresolvedInputError
from pitstop.splits import build_split_manifest

# (race, laps, seed): train, train, val, test
RACES = (("2022_01_R", 40, 0), ("2024_05_R", 40, 1), ("2025_01_R", 30, 2), ("2025_13_R", 40, 3))


def raw_inputs():
    laps = pd.concat([synthetic_race(r, n, seed=s) for r, n, s in RACES], ignore_index=True)
    events = [
        {
            "race_id": r,
            "season": int(r[:4]),
            "round_number": int(r[5:7]),
            "session": "R",
            "special_tyre_rule": False,
        }
        for r, _, _ in RACES
    ]
    evidence = [lap_counts(r, ((0.0, float(n)),)) for r, n, _ in RACES]
    return laps, events, evidence, build_split_manifest(events)


def test_artifacts_are_aligned_and_scaled_on_train_only():
    samples, tabular, sequences, scaling = make_dataset(*raw_inputs())
    validate_samples(samples)
    assert samples.split.drop_duplicates().tolist() == ["train", "val", "test"]
    assert samples.sc.any() and samples.vsc.any() and samples.y.sum() > 0
    # Same row_id order in all three tables; the last window step is the tabular vector.
    assert np.array_equal(tabular["row_id"], samples.row_id)
    assert np.array_equal(sequences["row_id"], samples.row_id)
    assert np.array_equal(sequences["X"][:, -1, :-1], tabular["X"])
    assert scaling["columns"] == SCALED and scaling["ddof"] == 0
    columns = [FEATURES.index(name) for name in SCALED]
    train, other = samples.split.eq("train").to_numpy(), ~samples.split.eq("train").to_numpy()
    scaled = tabular["X"][:, columns]
    np.testing.assert_allclose(scaled[train].mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(scaled[train].std(axis=0), 1, atol=1e-4)
    assert abs(scaled[other].mean(axis=0)).max() > 0.01  # val/test do not re-fit the statistics
    assert tabular["X"][:, FEATURES.index("sc")].max() == 1  # boolean channels unscaled


def test_constant_column_gets_scale_one():
    train = pd.DataFrame({name: [1.0, 3.0] for name in SCALED}).assign(position=[2.0, 2.0])
    scaling = fit_scaling(train)
    assert scaling["scale"][SCALED.index("position")] == 1.0
    assert scaling["mean"][SCALED.index("tyre_age")] == 2.0
    assert scaling["scale"][SCALED.index("tyre_age")] == 1.0


def test_unresolved_input_stops_the_build():
    laps, events, evidence, splits = raw_inputs()
    laps.loc[(laps.race_id == "2025_13_R") & (laps.LapNumber == 1), "Position"] = np.nan
    with pytest.raises(
        UnresolvedInputError, match="race 2025_13_R, driver 10, lap 1, feature position"
    ):
        make_dataset(laps, events, evidence, splits)


def test_build_dataset_writes_validated_artifacts_with_provenance(tmp_path, monkeypatch):
    laps, events, evidence, splits = raw_inputs()
    raw = tmp_path / "raw" / "v1"
    raw.mkdir(parents=True)
    laps.to_parquet(raw / "laps.parquet", index=False)
    (raw / "manifest.json").write_text("{}")
    (raw / "events.json").write_text(json.dumps(events))
    (raw / "lap_count_evidence.json").write_text(json.dumps(evidence))
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    paths = {
        "raw": str(tmp_path / "raw"),
        "processed": str(tmp_path / "out"),
        "splits": str(tmp_path / "splits.json"),
    }
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"paths": paths}))
    monkeypatch.setattr(
        pitstop.dataset, "verify_data", lambda path: None
    )  # raw bytes are synthetic

    assert main(["build-dataset", "--config", str(tmp_path / "data.yaml"), "--snapshot", "v1"]) == 0
    out = tmp_path / "out"
    samples = pd.read_parquet(out / "samples.parquet")
    with np.load(out / "tabular.npz", allow_pickle=False) as tabular:
        assert np.array_equal(tabular["row_id"], samples.row_id)
    preprocess = json.loads((out / "preprocess.json").read_text())
    train_ids = samples.row_id[samples.split == "train"].to_numpy("<i8")
    assert preprocess["fit_row_ids_sha256"] == hashlib.sha256(train_ids.tobytes()).hexdigest()
    assert (
        preprocess["splits_sha256"]
        == hashlib.sha256((tmp_path / "splits.json").read_bytes()).hexdigest()
    )
    assert preprocess["feature_order"] == list(FEATURES) and preprocess[
        "sequence_channels"
    ] == list(SEQ_CHANNELS)
    assert set(preprocess["payload_sha256"]) == {"samples.parquet", "tabular.npz", "sequences.npz"}
    assert len(preprocess["git_commit"]) == 40
    assert main(["build-dataset", "--config", str(tmp_path / "data.yaml"), "--snapshot", "v1"]) == 1


def test_key_columns_stay_unique_across_splits():
    samples = make_dataset(*raw_inputs())[0]
    assert not samples.duplicated(list(KEY)).any()
