"""Train configured models with one shared PyTorch loop and validation-AP model selection."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import yaml

from pitstop.ingest import ROOT, write_json
from pitstop.models import baselines
from pitstop.predict import load_inputs, predict_run

PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "pyyaml")


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _environment(lock):
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=False)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "uv": uv.stdout.strip() or None,
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def train(config_path):
    """Fit a baseline on train rows, write a complete run directory and its predictions."""
    config = yaml.safe_load(Path(config_path).read_text())
    kind, seed = config["model"], config["seed"]
    samples, x = load_inputs(config)
    split = samples["split"].to_numpy()
    y = samples["y"].to_numpy()
    train_rows, val_rows = split == "train", split == "val"

    commit = _git("rev-parse", "HEAD")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / config["output_dir"] / f"{stamp}_{kind}_s{seed}_{commit[:8]}"
    run_dir.mkdir(parents=True)  # raises if the run already exists

    model, history = baselines.fit(config, x[train_rows], y[train_rows], x[val_rows], y[val_rows])
    baselines.save(model, kind, run_dir)
    preprocess = ROOT / config["data"]["preprocess"]
    (run_dir / "preprocess.json").write_bytes(preprocess.read_bytes())
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "git_commit.txt").write_text(commit + "\n")
    (run_dir / "data_hash.txt").write_text(json.loads(preprocess.read_text())["data_sha256"] + "\n")
    (run_dir / "seed.txt").write_text(f"{seed}\n")
    write_json(run_dir / "environment.json", _environment(ROOT / "uv.lock"))
    write_json(run_dir / "history.json", history)
    return predict_run(run_dir)
