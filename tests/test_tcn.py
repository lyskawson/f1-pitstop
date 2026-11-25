"""Test the causal TCN on tiny synthetic windows."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import average_precision_score
from test_rnn import TRAINING, windows
from test_train import make_config

from pitstop.models.tcn import TCN
from pitstop.predict import predict_run
from pitstop.train import fit_torch, train

ARCHITECTURE = {
    "input_size": 16,
    "window": 15,
    "blocks": 3,
    "convolutions_per_block": 2,
    "channels": 8,
    "kernel_size": 3,
    "dilations": [1, 2, 4],
    "causal": True,
    "residual": True,
    "dropout": 0.1,
    "head": "last_step",
    "output_size": 1,
}


def test_conv_stack_is_causal():
    model = TCN(ARCHITECTURE, 0.2).eval()
    x = torch.from_numpy(windows(3))
    with torch.no_grad():
        before = model.steps(x)
        for s in (0, 4, 9, 14):  # includes the last step
            changed = x.clone()
            changed[:, s] += 1
            after = model.steps(changed)
            assert torch.equal(after[:, :, :s], before[:, :, :s])
            assert not torch.equal(after[:, :, s], before[:, :, s])


# Padding sub-check skipped: left-padded rows carry is_pad = 1 as an input feature, so
# a different amount of padding is a different input, not the same suffix.


def test_receptive_field_covers_window():
    model = TCN(ARCHITECTURE, 0.2).eval()
    x = torch.from_numpy(windows(2))
    changed = x.clone()
    changed[:, 0] += 1
    with torch.no_grad():
        assert not torch.equal(model(changed), model(x))


def test_logits_shape_and_bias_init():
    model = TCN(ARCHITECTURE, 0.2)
    assert model(torch.from_numpy(windows(7))).shape == (7,)
    assert math.isclose(model.out.bias.item(), math.log(0.25), rel_tol=1e-6)


def test_non_causal_raises():
    with pytest.raises(ValueError):
        TCN(ARCHITECTURE | {"causal": False}, 0.2)


def test_rows_do_not_mix():
    model = TCN(ARCHITECTURE, 0.2).eval()
    x = torch.from_numpy(windows(5))
    with torch.no_grad():
        alone = torch.cat([model(x[i : i + 1]) for i in range(5)])
        assert torch.allclose(model(x), alone, atol=1e-6)


def test_overfits_small_separable_sample():
    y = np.repeat([0, 1], 32)
    x = windows(64) + 2 * y[:, None, None].astype("float32")
    x[:, :, -1] = 0
    config = {"model": "tcn", "seed": 1, "architecture": ARCHITECTURE, "training": TRAINING}
    state, _ = fit_torch(config, x, y, x, y)
    model = TCN(ARCHITECTURE, 0.5)
    model.load_state_dict(state)
    assert average_precision_score(y, model.eval()(torch.from_numpy(x)).detach()) > 0.99


def tcn_run(tmp_path, seed=42):
    """Train through train() on tiny windows and return the run directory."""
    tmp_path.mkdir(exist_ok=True)
    path = make_config(tmp_path, "constant")
    config = yaml.safe_load(path.read_text())
    sequences = tmp_path / "data" / "sequences.npz"
    row_id = pd.read_parquet(tmp_path / "data" / "samples.parquet")["row_id"].to_numpy()
    np.savez(sequences, row_id=row_id, X=windows(len(row_id)))
    config |= {"model": "tcn", "seed": seed, "input": "sequences"}
    config |= {"architecture": ARCHITECTURE, "training": TRAINING | {"max_epochs": 2}}
    config["data"]["sequences"] = str(sequences)
    del config["parameters"], config["early_stopping"]
    path.write_text(yaml.safe_dump(config))
    return tmp_path / "runs" / train(path)["run_id"]


def test_reload_and_seed(tmp_path):
    first = tcn_run(tmp_path / "a")
    before = pd.read_parquet(first / "predictions.parquet")["score"]
    predict_run(first)  # rebuilds from config.yaml and model.pt
    assert before.equals(pd.read_parquet(first / "predictions.parquet")["score"])
    second = pd.read_parquet(tcn_run(tmp_path / "b") / "predictions.parquet")["score"]
    assert before.equals(second)
