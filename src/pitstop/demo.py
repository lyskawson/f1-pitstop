"""Export the D11 demo pack from saved runs. Reads files only: no training, no evaluation."""

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from pitstop.contracts import KEY, validate_predictions

RULE = {"name": "seed42_one_run_per_model", "version": 1}
N_RACES = 3
PAYLOAD = ("predictions.parquet", "samples.parquet", "metrics.json", "thresholds.json")


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def export_demo(
    runs_dir="runs",
    samples="data/processed/v1/samples.parquet",
    events="data/raw/v1/events.json",
    report="reports/test-v1.json",
    output_dir="demo",
    check_clean=True,
):
    """Write the five-file demo pack and return the metadata record."""
    runs_dir, out = Path(runs_dir), Path(output_dir)
    if check_clean:
        out_rel = out.resolve().relative_to(Path.cwd().resolve())
        if _git("status", "--porcelain", "--", ".", f":(exclude){out_rel}"):
            raise ValueError("export requires a clean committed revision")
    # The last commit touching code, so committing demo/ does not change the pack.
    code_commit = _git("log", "-1", "--format=%H", "--", "src", "pyproject.toml", "uv.lock")

    run_ids = [r["run_id"] for r in json.loads(Path(report).read_text())["runs"] if r["seed"] == 42]
    run_ids = sorted(run_ids)
    metrics, thresholds = {}, {}
    for run_id in run_ids:
        run = runs_dir / run_id
        metrics[run_id] = json.loads((run / "metrics.json").read_text())
        thresholds[run_id] = json.loads((run / "threshold.json").read_text())
        test = metrics[run_id].get("split_provenance", {}).get("test", {})
        if "test" not in metrics[run_id]["splits"] or not test.get("test_approval"):
            raise ValueError(f"{run_id}: no approved test metrics; refusing to export")

    samples = pd.read_parquet(samples)
    test_rows = samples[samples["split"] == "test"]
    races = sorted(test_rows["race_id"].unique())[:N_RACES]
    if len(races) < N_RACES:
        raise ValueError(f"need {N_RACES} test races, found {len(races)}")
    pack_samples = test_rows[test_rows["race_id"].isin(races)]
    pack_samples = pack_samples[[*KEY, "sc", "vsc", "special_tyre_rule", "y"]]
    pack_samples = pack_samples.sort_values(list(KEY)).reset_index(drop=True)

    frames = []
    for run_id in run_ids:
        run_predictions = pd.read_parquet(runs_dir / run_id / "predictions.parquet")
        frames.append(run_predictions[run_predictions["race_id"].isin(races)])
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions[predictions["split"] == "test"]
    predictions = predictions.sort_values(["run_id", *KEY]).reset_index(drop=True)
    validate_predictions(predictions)
    if len(predictions) != len(pack_samples) * len(run_ids):
        raise ValueError("predictions do not cover every test sample of every chosen run")

    out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    pack_samples.to_parquet(out / "samples.parquet", index=False)
    _dump(out / "metrics.json", metrics)
    _dump(out / "thresholds.json", thresholds)

    names = {e["race_id"]: e["event_name"] for e in json.loads(Path(events).read_text())}
    meta = {
        "race_ids": races,
        "event_names": {r: names[r] for r in races},
        "rule": RULE,
        "run_ids": run_ids,
        "run_git_commits": sorted({t["git_commit"] for t in thresholds.values()}),
        "export_git_commit": code_commit,
        "data_sha256": sorted({t["data_sha256"] for t in thresholds.values()}),
        "packages": {p: version(p) for p in ("numpy", "pandas", "pyarrow")},
        "payload_sha256": {name: _sha256(out / name) for name in PAYLOAD},
    }
    _dump(out / "metadata.json", meta)
    return meta
