"""Evaluate one run against declared samples and publish frozen artifacts."""

import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pitstop.contracts import (
    KEY,
    PREDICTIONS_DTYPES,
    RUN_ID_PATTERN,
    SAMPLES_DTYPES,
    SPLITS,
    validate_predictions,
    validate_samples,
)
from pitstop.metrics import group_metrics, select_threshold


def align_predictions(predictions, samples, *, split="val", run_id=None):
    """Check a run's predictions and return complete, key-ordered rows for a split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    validate_predictions(predictions)
    validate_samples(samples)
    runs = predictions["run_id"].unique()
    if run_id is None:
        if len(runs) != 1:
            raise ValueError("Supply run_id when predictions do not contain exactly one run")
        run_id = str(runs[0])
    if re.fullmatch(RUN_ID_PATTERN, run_id) is None or run_id not in runs:
        raise ValueError("run_id must identify a run in predictions")

    if samples.groupby("race_id", observed=True)["split"].nunique().gt(1).any():
        raise ValueError("Declared samples split a race across multiple splits")
    selected = predictions.loc[predictions["run_id"].eq(run_id), list(PREDICTIONS_DTYPES)]
    declared = samples[list(SAMPLES_DTYPES)]
    joined = selected.merge(
        declared,
        on=list(KEY),
        how="left",
        validate="one_to_one",
        suffixes=("_pred", ""),
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("Predictions contain keys missing from declared samples")
    if joined["y_pred"].ne(joined["y"]).any():
        raise ValueError("Prediction labels disagree with declared samples")
    if joined["split_pred"].ne(joined["split"]).any():
        raise ValueError("Prediction splits disagree with declared samples")
    rows = joined.loc[joined["split"].eq(split), [*declared.columns, "score"]]
    if len(rows) != declared["split"].eq(split).sum():
        raise ValueError(f"Predictions are missing declared {split} samples for {run_id}")
    return run_id, rows.sort_values(list(KEY), ignore_index=True)


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _git_revision():
    root = Path(__file__).resolve().parent

    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                "Evaluation requires a Git checkout to identify its producing revision"
            )
        return result.stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


def _load_json(payload):
    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    document = json.loads(payload, parse_constant=invalid)
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON object")
    return document


def _json_bytes(document):
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _frozen_threshold(path, rows, split, identity, provenance):
    if path.is_symlink():
        raise ValueError("threshold.json must be a regular file")
    if path.exists():
        payload = path.read_bytes()
        frozen = _load_json(payload)
        if any(frozen.get(key) != value for key, value in identity.items()):
            raise ValueError("Frozen threshold belongs to another run or data manifest")
        for field in ("threshold", "validation_f1"):
            value = frozen.get(field)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"Invalid frozen {field}")
        commit = frozen.get("git_commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("Invalid frozen threshold producing revision")
        if "inputs" in frozen and (
            frozen["inputs"]["samples_sha256"] != provenance["inputs"]["samples_sha256"]
            or frozen["inputs"]["source"] != provenance["inputs"]["source"]
        ):
            raise ValueError("Frozen threshold belongs to different samples or input source")
        return frozen["threshold"], payload, False

    if split != "val":
        raise ValueError("Test evaluation requires a threshold already frozen on validation")
    if rows.empty:
        raise ValueError("Cannot freeze a threshold without validation samples")
    threshold, f1 = select_threshold(rows["y"], rows["score"])
    payload = _json_bytes(
        {
            **identity,
            **provenance,
            "threshold": threshold,
            "validation_f1": f1,
        }
    )
    return threshold, payload, True


def _evaluate_groups(rows, threshold, bootstrap_samples, seed):
    neutral = rows["sc"] | rows["vsc"]
    groups = {
        "overall": rows,
        "green": rows.loc[~neutral],
        "sc_vsc": rows.loc[neutral],
        "special_tyre_rule": rows.loc[rows["special_tyre_rule"]],
    }
    results, missing = {}, {}
    for name, group in groups.items():
        results[name], missing[name] = group_metrics(
            group,
            threshold,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    return results, missing


def _audit(groups, synthetic):
    triggered = [
        f"{group}.{metric}"
        for group, values in groups.items()
        for metric in ("average_precision", "f1")
        if values[metric] is not None and values[metric] > 0.6
    ]
    return {
        "required": bool(triggered) and not synthetic,
        "status": "synthetic_fixture"
        if synthetic
        else ("pending" if triggered else "not_triggered"),
        "evidence": {"cutoff": 0.6, "triggered_metrics": triggered},
    }


def _publish(directory, threshold_payload, new_threshold, metrics, previous):
    """Stage every payload before publication and keep each file complete on disk."""
    threshold_path = directory / "threshold.json"
    metrics_path = directory / "metrics.json"
    with TemporaryDirectory(prefix=".evaluate-", dir=directory) as temp:
        staged_threshold = Path(temp) / threshold_path.name
        staged_metrics = Path(temp) / metrics_path.name
        if new_threshold:
            staged_threshold.write_bytes(threshold_payload)
        staged_metrics.write_bytes(_json_bytes(metrics))
        linked = False
        try:
            if new_threshold:
                os.link(staged_threshold, threshold_path)
                linked = True
            if previous:
                os.replace(staged_metrics, metrics_path)
            else:
                os.link(staged_metrics, metrics_path)
        except BaseException:
            if linked and not metrics_path.exists():
                threshold_path.unlink()
            raise


def _save_evaluation(directory, rows, split, identity, provenance, bootstrap, synthetic):
    threshold_path = directory / "threshold.json"
    metrics_path = directory / "metrics.json"
    if metrics_path.is_symlink():
        raise ValueError("metrics.json must be a regular file")
    previous = _load_json(metrics_path.read_bytes()) if metrics_path.exists() else None
    if previous is not None and not threshold_path.exists():
        raise ValueError("Existing metrics require their original frozen threshold")
    threshold, payload, new_threshold = _frozen_threshold(
        threshold_path,
        rows,
        split,
        identity,
        provenance,
    )
    common = {
        **{key: identity[key] for key in ("schema_version", "run_id", "data_sha256")},
        "threshold_file_sha256": _digest(payload),
    }
    if previous is not None:
        if any(previous.get(key) != value for key, value in common.items()):
            raise ValueError("Existing metrics belong to another run, dataset or threshold")
        if any(previous["bootstrap"].get(key) != value for key, value in bootstrap.items()):
            raise ValueError("Existing metrics use different bootstrap settings")
        if any(
            value["inputs"]["source"] != provenance["inputs"]["source"]
            or value["inputs"]["samples_sha256"] != provenance["inputs"]["samples_sha256"]
            for value in previous["split_provenance"].values()
        ):
            raise ValueError("Existing metrics use different samples or input source")
        if split in previous["splits"]:
            if previous["split_provenance"][split] != provenance:
                raise ValueError(f"Existing {split} metrics have different producing inputs")
            return previous

    groups, missing = _evaluate_groups(rows, threshold, bootstrap["samples"], bootstrap["seed"])
    metrics = previous or {
        **common,
        "bootstrap": {**bootstrap, "undefined_replicates": {}},
        "splits": {},
        "split_provenance": {},
        "leakage_audit": {"by_split": {}},
    }
    metrics["git_commit"] = provenance["git_commit"]
    metrics["splits"][split] = groups
    metrics["split_provenance"][split] = provenance
    metrics["bootstrap"]["undefined_replicates"][split] = missing
    audits = metrics["leakage_audit"]["by_split"]
    audits[split] = _audit(groups, synthetic)
    required = any(audit["required"] for audit in audits.values())
    metrics["leakage_audit"].update(
        required=required,
        status="pending" if required else audits[split]["status"],
    )
    _publish(directory, payload, new_threshold, metrics, previous is not None)
    return metrics


def evaluate_files(
    predictions_path,
    samples_path,
    data_manifest_path,
    output_dir,
    *,
    run_id=None,
    split="val",
    bootstrap_samples=1000,
    seed=42,
    synthetic=False,
    test_approval=None,
):
    """Evaluate a run offline and return its metrics.json document."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if split == "test" and (not isinstance(test_approval, str) or not test_approval.strip()):
        raise ValueError(
            "Test evaluation requires explicit human approval after configs are frozen"
        )
    if split == "val" and test_approval is not None:
        raise ValueError("Test approval is only applicable to the test split")

    predictions = Path(predictions_path).read_bytes()
    samples = Path(samples_path).read_bytes()
    manifest = Path(data_manifest_path).read_bytes()
    manifest_document = _load_json(manifest)
    if synthetic and manifest_document.get("synthetic") is not True:
        raise ValueError("Synthetic evaluation requires a fixture manifest with synthetic: true")
    run_id, rows = align_predictions(
        pd.read_parquet(io.BytesIO(predictions)),
        pd.read_parquet(io.BytesIO(samples)),
        split=split,
        run_id=run_id,
    )
    commit, dirty = _git_revision()
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "data_sha256": _digest(manifest),
        "chosen_on_split": "val",
        "selection_metric": "f1",
        "tie_break": "largest",
    }
    provenance = {
        "git_commit": commit,
        "git_dirty": dirty,
        "inputs": {
            "samples_sha256": _digest(samples),
            "predictions_sha256": _digest(predictions),
            "data_manifest_sha256": identity["data_sha256"],
            "source": "synthetic_fixture" if synthetic else "project",
        },
    }
    if test_approval is not None:
        provenance["test_approval"] = test_approval.strip()
    bootstrap = {
        "unit": "race",
        "samples": bootstrap_samples,
        "confidence_level": 0.95,
        "seed": seed,
        "method": "percentile",
        "undefined_handling": "omit_per_metric",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    directory = os.open(output, os.O_RDONLY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        return _save_evaluation(output, rows, split, identity, provenance, bootstrap, synthetic)
    finally:
        os.close(directory)
