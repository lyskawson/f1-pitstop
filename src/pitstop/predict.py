"""Load frozen models and preprocessing to produce keyed next-lap probability predictions."""

import numpy as np
import pandas as pd
import yaml

from pitstop.contracts import KEY, validate_predictions, validate_samples, validate_tabular
from pitstop.ingest import ROOT
from pitstop.models import baselines


def load_inputs(config):
    """Load and validate the processed samples and tabular features."""
    samples = pd.read_parquet(ROOT / config["data"]["samples"])
    with np.load(ROOT / config["data"]["tabular"], allow_pickle=False) as arrays:
        tabular = {name: arrays[name] for name in arrays.files}
    validate_samples(samples)
    validate_tabular(tabular, samples)
    return samples, tabular["X"]


def predict_run(run_dir):
    """Write predictions.parquet for train and val rows only; the test split is never read."""
    run_dir = ROOT / run_dir
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    samples, x = load_inputs(config)
    rows = samples["split"].isin(("train", "val")).to_numpy()
    samples, x = samples[rows], x[rows]  # test rows never reach the model
    model = baselines.load(config["model"], run_dir)
    predictions = samples[[*KEY, "split", "y"]].assign(
        run_id=run_dir.name,
        score=np.asarray(baselines.score(model, x), dtype="float64"),
    )
    predictions["target_lap"] = predictions["lap_number"] + 1
    validate_predictions(predictions)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    return {"complete": True, "run_id": run_dir.name, "rows": int(rows.sum())}
