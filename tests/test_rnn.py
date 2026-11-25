"""Test the GRU/LSTM model on tiny synthetic windows."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import average_precision_score
from test_train import make_config

from pitstop.models.rnn import RNN
from pitstop.predict import predict_run
from pitstop.train import fit_torch, train

ARCHITECTURE = {
    "input_size": 16,
    "cell": "gru",
    "hidden_size": 8,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": False,
    "output_size": 1,
}
TRAINING = {
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "pos_weight": 1.0,
    "batch_size": 32,
    "max_epochs": 40,
    "patience": 5,
    "gradient_clip_norm": 1.0,
}


def windows(n, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 15, 16)).astype("float32")
    x[:, :, -1] = 0  # is_pad: all real laps
    return x + shift


def test_logits_shape_and_bias_init():
    for cell in ("gru", "lstm"):
        model = RNN(ARCHITECTURE | {"cell": cell}, 0.2)
        assert model(torch.from_numpy(windows(7))).shape == (7,)
        assert math.isclose(model.out.bias.item(), math.log(0.25), rel_tol=1e-6)


def test_rows_do_not_mix():
    model = RNN(ARCHITECTURE, 0.2).eval()
    x = torch.from_numpy(windows(5))
    x[1, :5] = 0  # row 1 is a short history: left zero padding marked by is_pad
    x[1, :5, -1] = 1
    with torch.no_grad():
        batch = model(x)
        alone = torch.cat([model(x[i : i + 1]) for i in range(5)])
        x[0, 0] += 1  # an earlier step of row 0 may change row 0 only
        changed = model(x)
    assert torch.allclose(batch, alone, atol=1e-6)
    assert not torch.allclose(changed[0], batch[0])
    assert torch.equal(changed[1:], batch[1:])


def test_bidirectional_raises():
    with pytest.raises(ValueError):
        RNN(ARCHITECTURE | {"bidirectional": True}, 0.2)


def test_overfits_small_separable_sample():
    y = np.repeat([0, 1], 32)
    x = windows(64) + 2 * y[:, None, None].astype("float32")
    x[:, :, -1] = 0
    config = {"model": "gru", "seed": 1, "architecture": ARCHITECTURE, "training": TRAINING}
    state, _ = fit_torch(config, x, y, x, y)
    model = RNN(ARCHITECTURE, 0.5)
    model.load_state_dict(state)
    assert average_precision_score(y, model.eval()(torch.from_numpy(x)).detach()) > 0.99


def gru_run(tmp_path, seed=42):
    """Train through train() on tiny windows and return the run directory."""
    tmp_path.mkdir(exist_ok=True)
    path = make_config(tmp_path, "constant")
    config = yaml.safe_load(path.read_text())
    sequences = tmp_path / "data" / "sequences.npz"
    row_id = pd.read_parquet(tmp_path / "data" / "samples.parquet")["row_id"].to_numpy()
    np.savez(sequences, row_id=row_id, X=windows(len(row_id)))
    config |= {"model": "gru", "seed": seed, "input": "sequences"}
    config |= {"architecture": ARCHITECTURE, "training": TRAINING | {"max_epochs": 2}}
    config["data"]["sequences"] = str(sequences)
    del config["parameters"], config["early_stopping"]
    path.write_text(yaml.safe_dump(config))
    return tmp_path / "runs" / train(path)["run_id"]


def test_reload_and_seed(tmp_path):
    first = gru_run(tmp_path / "a")
    before = pd.read_parquet(first / "predictions.parquet")["score"]
    predict_run(first)  # rebuilds from config.yaml and model.pt
    assert before.equals(pd.read_parquet(first / "predictions.parquet")["score"])
    second = pd.read_parquet(gru_run(tmp_path / "b") / "predictions.parquet")["score"]
    assert before.equals(second)
