"""Test the MLP and the shared PyTorch loop on tiny synthetic data."""

import json
import math

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from test_train import make_config

from pitstop.models.mlp import MLP
from pitstop.predict import predict_run
from pitstop.train import fit_torch, train

ARCHITECTURE = {
    "input_size": 15,
    "hidden_sizes": [64, 32],
    "output_size": 1,
    "dropout": 0.2,
}
TRAINING = {
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "pos_weight": 1.0,
    "batch_size": 32,
    "max_epochs": 40,
    "patience": 5,
    "gradient_clip_norm": None,
}


def mlp_run(tmp_path, seed=42, **dataset_options):
    """Train a tiny MLP in its own tmp dir and return the run directory."""
    tmp_path.mkdir(exist_ok=True)
    path = make_config(tmp_path, "constant", **dataset_options)
    config = yaml.safe_load(path.read_text())
    config |= {"model": "mlp", "seed": seed, "architecture": ARCHITECTURE, "training": TRAINING}
    del config["parameters"], config["early_stopping"]
    path.write_text(yaml.safe_dump(config))
    return tmp_path / "runs" / train(path)["run_id"]


def scores(run_dir):
    return pd.read_parquet(run_dir / "predictions.parquet")


def test_logits_shape_and_bias_init():
    model = MLP(ARCHITECTURE, 0.2)
    assert model(torch.zeros(7, 15)).shape == (7,)
    assert math.isclose(model.out.bias.item(), math.log(0.25), rel_tol=1e-6)


def test_overfits_small_separable_sample():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 32)
    x = (rng.normal(size=(64, 15)) + 3 * y[:, None]).astype("float32")
    config = {"model": "mlp", "seed": 1, "architecture": ARCHITECTURE, "training": TRAINING}
    state, _ = fit_torch(config, x, y, x, y)
    model = MLP(ARCHITECTURE, 0.5)
    model.load_state_dict(state)
    assert average_precision_score(y, model.eval()(torch.from_numpy(x)).detach()) > 0.99


def test_reload_seed_and_early_stopping(tmp_path):
    first = mlp_run(tmp_path / "a")
    before = scores(first)
    assert set(before["split"]) == {"train", "val"}  # test rows never scored
    predict_run(first)  # rebuilds from config.yaml and model.pt
    assert before["score"].equals(scores(first)["score"])
    assert before["score"].equals(scores(mlp_run(tmp_path / "b"))["score"])
    assert not before["score"].equals(scores(mlp_run(tmp_path / "c", seed=7))["score"])
    history = json.loads((first / "history.json").read_text())
    curve = [h["val_average_precision"] for h in history[:-1]]
    assert history[-1]["best_epoch"] == int(np.argmax(curve)) + 1


def noisy_split(seed):
    """Weak signal plus noisy labels: val AP peaks early, then the net memorises train noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(64, 15)).astype("float32")
    y = ((x[:, 0] + rng.normal(size=64)) > 1).astype("int64")
    return x, y


def test_returned_weights_are_best_epoch_not_last():
    (x, y), (x_val, y_val) = noisy_split(0), noisy_split(1)
    config = {"model": "mlp", "seed": 1, "architecture": ARCHITECTURE}
    config["training"] = TRAINING | {"learning_rate": 0.05, "max_epochs": 30, "patience": 30}
    state, history = fit_torch(config, x, y, x_val, y_val)
    curve = [h["val_average_precision"] for h in history[:-1]]
    best = history[-1]["best_epoch"]
    assert best < len(curve)  # the peak is before the last epoch
    model = MLP(ARCHITECTURE, 0.5)
    model.load_state_dict(state)
    ap = average_precision_score(y_val, model.eval()(torch.from_numpy(x_val)).detach())
    assert math.isclose(ap, max(curve), rel_tol=1e-6)


def test_val_data_does_not_change_training():
    x, y = noisy_split(0)
    config = {"model": "mlp", "seed": 1, "architecture": ARCHITECTURE}
    config["training"] = TRAINING | {"max_epochs": 3, "patience": 30}
    losses = []
    for seed in (1, 2):  # only val differs; val may pick the epoch but never feeds the loss
        x_val, y_val = noisy_split(seed)
        _, history = fit_torch(config, x, y, x_val, y_val)
        losses.append([h["train_loss"] for h in history[:-1]])
    assert losses[0] == losses[1]
