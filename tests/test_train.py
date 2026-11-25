import json
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import yaml

from pitstop import train as train_module
from pitstop.contracts import FEATURES, RUN_ID_PATTERN, validate_predictions
from pitstop.predict import predict_run
from pitstop.train import train

PARAMETERS = {
    "constant": {"strategy": "train_prevalence"},
    "logreg": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 200,
        "fit_intercept": True,
        "class_weight": None,
    },
    "lightgbm": {
        "objective": "binary",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "n_estimators": 60,
        "min_child_samples": 10,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "class_weight": None,
        "n_jobs": 1,
        "verbosity": -1,
    },
}


def write_dataset(path, x_shift=0.0, flip_val_test=False):
    """Tiny processed snapshot; `x_shift` and `flip_val_test` alter val/test rows only."""
    rng = np.random.default_rng(0)
    n = 300
    split = np.repeat(["train", "val", "test"], n // 3)
    x = rng.normal(size=(n, len(FEATURES))).astype("float32")
    y = (x[:, 0] + rng.normal(size=n) > 1.2).astype("int8")
    held_out = split != "train"
    x[held_out] += x_shift
    if flip_val_test:
        y[held_out] = 1 - y[held_out]
    samples = pd.DataFrame(
        {
            "row_id": np.arange(n),
            "race_id": "2022_01_R",
            "driver_number": [str(i) for i in range(n)],
            "lap_number": 1 + np.arange(n),
            "y": y,
            "split": split,
            "sc": False,
            "vsc": False,
            "special_tyre_rule": False,
        }
    )
    path.mkdir(exist_ok=True)
    samples.to_parquet(path / "samples.parquet", index=False)
    np.savez(path / "tabular.npz", row_id=samples["row_id"].to_numpy(), X=x)
    (path / "preprocess.json").write_text(json.dumps({"data_sha256": "ab" * 32}))


def make_config(tmp_path, model, **dataset_options):
    data = tmp_path / "data"
    write_dataset(data, **dataset_options)
    config = {
        "model": model,
        "seed": 42,
        "input": "tabular",
        "data": {
            "samples": str(data / "samples.parquet"),
            "tabular": str(data / "tabular.npz"),
            "preprocess": str(data / "preprocess.json"),
        },
        "output_dir": str(tmp_path / "runs"),
        "parameters": PARAMETERS[model],
        "early_stopping": {"rounds": 20, "split": "val", "metric": "average_precision"},
    }
    path = tmp_path / f"{model}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def run(tmp_path, model, **dataset_options):
    result = train(make_config(tmp_path, model, **dataset_options))
    return tmp_path / "runs" / result["run_id"]


@pytest.mark.parametrize("model", PARAMETERS)
def test_run_directory_and_predictions(tmp_path, model):
    run_dir = run(tmp_path, model)
    assert re.fullmatch(RUN_ID_PATTERN, run_dir.name)
    assert run_dir.name.split("_")[1:3] == [model, "s42"]
    for name in (
        "config.yaml",
        "git_commit.txt",
        "data_hash.txt",
        "seed.txt",
        "environment.json",
        "preprocess.json",
        "history.json",
        "predictions.parquet",
    ):
        assert (run_dir / name).exists()
    assert (run_dir / "data_hash.txt").read_text().strip() == "ab" * 32
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    validate_predictions(predictions)
    assert np.isfinite(predictions["score"]).all()
    assert set(predictions["split"]) == {"train", "val"}  # the test split is never predicted
    assert predictions["run_id"].eq(run_dir.name).all()


@pytest.mark.parametrize("model", PARAMETERS)
def test_save_load_and_seed_reproducibility(tmp_path, model):
    (tmp_path / "a").mkdir()
    first = run(tmp_path / "a", model)
    second = run(tmp_path, model)
    before = pd.read_parquet(first / "predictions.parquet")["score"]
    assert before.equals(pd.read_parquet(second / "predictions.parquet")["score"])
    predict_run(second)  # reloads the saved model from disk
    assert before.equals(pd.read_parquet(second / "predictions.parquet")["score"])


@pytest.mark.parametrize("model", ["constant", "logreg"])
def test_only_train_rows_are_used(tmp_path, model):
    (tmp_path / "b").mkdir()
    base = pd.read_parquet(run(tmp_path, model) / "predictions.parquet")
    other = pd.read_parquet(
        run(tmp_path / "b", model, x_shift=5.0, flip_val_test=True) / "predictions.parquet"
    )
    train_rows = base["split"] == "train"
    assert base.loc[train_rows, "score"].equals(other.loc[train_rows, "score"])


def test_lightgbm_early_stopping_uses_val(tmp_path):
    run_dir = run(tmp_path, "lightgbm")
    history = json.loads((run_dir / "history.json").read_text())
    curve = [h["val_average_precision"] for h in history[:-1]]
    assert all(np.isfinite(h["train_loss"]) for h in history[:-1])
    assert history[-1]["best_iteration"] == int(np.argmax(curve)) + 1
    assert len(curve) < PARAMETERS["lightgbm"]["n_estimators"]


def test_run_directory_is_never_overwritten(tmp_path, monkeypatch):
    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(train_module, "datetime", FixedClock)
    config = make_config(tmp_path, "constant")
    result = train(config)
    marker = tmp_path / "runs" / result["run_id"] / "seed.txt"
    marker.write_text("changed\n")
    with pytest.raises(FileExistsError):
        train(config)
    assert marker.read_text() == "changed\n"
