"""Load frozen models and preprocessing to produce keyed next-lap probability predictions."""

import lightgbm  # noqa: F401  (import before torch)
import numpy as np
import pandas as pd
import torch
import yaml

from pitstop.contracts import (
    KEY,
    validate_predictions,
    validate_samples,
    validate_sequences,
    validate_tabular,
)
from pitstop.ingest import ROOT
from pitstop.models import baselines, mlp  # baselines imports lightgbm before torch

MODELS = {"mlp": mlp.MLP}  # T9 and T10 add "gru" and "tcn"


def build_model(config, prevalence=0.5):
    """Rebuild a neural architecture from config; load_state_dict overwrites the bias init."""
    return MODELS[config["model"]](config["architecture"], prevalence)


def load_inputs(config):
    """Load and validate the samples and the input array chosen by config `input`."""
    samples = pd.read_parquet(ROOT / config["data"]["samples"])
    kind = config["input"]  # "tabular" or "sequences"
    with np.load(ROOT / config["data"][kind], allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    validate_samples(samples)
    (validate_tabular if kind == "tabular" else validate_sequences)(arrays, samples)
    return samples, arrays["X"]


def predict_run(run_dir):
    """Write predictions.parquet for train and val rows only; the test split is never read."""
    run_dir = ROOT / run_dir
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    samples, x = load_inputs(config)
    rows = samples["split"].isin(("train", "val")).to_numpy()
    samples, x = samples[rows], x[rows]  # test rows never reach the model
    if config["model"] in baselines.FILES:
        score = baselines.score(baselines.load(config["model"], run_dir), x)
    else:
        model = build_model(config)
        model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
        with torch.no_grad():
            score = torch.sigmoid(model.eval()(torch.from_numpy(x))).numpy()
    predictions = samples[[*KEY, "split", "y"]].assign(
        run_id=run_dir.name, score=np.asarray(score, dtype="float64")
    )
    predictions["target_lap"] = predictions["lap_number"] + 1
    validate_predictions(predictions)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    return {"complete": True, "run_id": run_dir.name, "rows": int(rows.sum())}
