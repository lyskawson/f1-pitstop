"""Train configured models with one shared PyTorch loop and validation-AP model selection."""

import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import lightgbm  # noqa: F401  (import before torch)
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from pitstop.ingest import ROOT, write_json
from pitstop.models import baselines
from pitstop.predict import build_model, load_inputs, predict_run

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


def _scores(model, x):
    with torch.no_grad():
        return torch.sigmoid(model.eval()(x)).numpy()


def fit_torch(config, x_train, y_train, x_val, y_val):
    """Train a neural model on train rows; val AP picks the epoch. Returns (state_dict, history)."""
    seed, cfg = config["seed"], config["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    prevalence = float(y_train.mean())
    x_train, y_train, x_val = map(torch.from_numpy, (x_train, y_train.astype("float32"), x_val))
    model = build_model(config, prevalence)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["pos_weight"]))
    shuffler = torch.Generator().manual_seed(seed)
    history, best_ap, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total = 0.0
        for rows in torch.randperm(len(x_train), generator=shuffler).split(cfg["batch_size"]):
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[rows]), y_train[rows])
            loss.backward()
            if cfg["gradient_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            total += loss.item() * len(rows)
        ap = float(average_precision_score(y_val, _scores(model, x_val)))
        history.append(
            {"epoch": epoch, "train_loss": total / len(x_train), "val_average_precision": ap}
        )
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg["patience"]:
            break
    history.append({"best_epoch": best_epoch})
    return best_state, history


def train(config_path):
    """Fit a model on train rows, write a complete run directory and its predictions."""
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

    started = time.perf_counter()
    data = (x[train_rows], y[train_rows], x[val_rows], y[val_rows])
    if kind in baselines.FILES:
        model, history = baselines.fit(config, *data)
        baselines.save(model, kind, run_dir)
    else:
        state, history = fit_torch(config, *data)
        torch.save(state, run_dir / "model.pt")
    preprocess = ROOT / config["data"]["preprocess"]
    (run_dir / "preprocess.json").write_bytes(preprocess.read_bytes())
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "git_commit.txt").write_text(commit + "\n")
    (run_dir / "data_hash.txt").write_text(json.loads(preprocess.read_text())["data_sha256"] + "\n")
    (run_dir / "seed.txt").write_text(f"{seed}\n")
    environment = _environment(ROOT / "uv.lock") | {"train_seconds": time.perf_counter() - started}
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "history.json", history)
    return predict_run(run_dir)
