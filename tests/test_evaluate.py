import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pitstop.cli import main
from pitstop.contracts import KEY, PREDICTIONS_DTYPES, SAMPLES_DTYPES
from pitstop.evaluate import align_predictions, evaluate_files

RUN = "20261124T120000Z_logreg_s42_abcdef12"
OTHER_RUN = "20261124T120000Z_mlp_s42_abcdef12"
COMMIT = "a" * 40


def synthetic_frames():
    samples = pd.DataFrame(
        [
            (0, "2025_01_R", "44", 1, 0, "val", False, False, True),
            (1, "2025_01_R", "44", 2, 1, "val", False, False, True),
            (2, "2025_01_R", "44", 3, 0, "val", True, False, True),
            (3, "2025_02_R", "16", 1, 0, "val", False, True, False),
            (4, "2025_13_R", "44", 1, 1, "test", False, False, False),
            (5, "2025_13_R", "44", 2, 0, "test", False, False, False),
        ],
        columns=list(SAMPLES_DTYPES),
    ).astype(SAMPLES_DTYPES)
    predictions = (
        samples[[*KEY, "split", "y"]]
        .assign(
            run_id=RUN,
            score=[0.4, 1, 0, 0.2, 0.5, 0.3],
            target_lap=samples["lap_number"] + 1,
        )
        .astype(PREDICTIONS_DTYPES)
    )
    return predictions, samples


def synthetic_files(directory):
    directory.mkdir(parents=True, exist_ok=True)
    predictions, samples = synthetic_frames()
    val, test = directory / "validation.parquet", directory / "test-fixture.parquet"
    samples_path, manifest = directory / "samples.parquet", directory / "manifest.json"
    predictions.loc[predictions["split"].eq("val")].to_parquet(val, index=False)
    predictions.loc[predictions["split"].eq("test")].to_parquet(test, index=False)
    samples.to_parquet(samples_path, index=False)
    manifest.write_text('{"synthetic": true}\n')
    return val, samples_path, manifest, directory / "artifacts", test


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr("pitstop.evaluate._git_revision", lambda: (COMMIT, False))
    return synthetic_files(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda p: p.iloc[1:], "missing declared val samples"),
        (lambda p: p.assign(driver_number="99"), "keys missing"),
        (lambda p: p.assign(y=p["y"].rsub(1).astype("int8")), "labels disagree"),
        (lambda p: p.assign(split="train"), "splits disagree"),
        (lambda p: pd.concat([p, p.iloc[:1]]), "duplicate key"),
        (lambda p: p.assign(score=1.5), "probability"),
        (lambda p: p.assign(target_lap=p["lap_number"]), "target_lap"),
    ],
)
def test_predictions_must_align(mutate, error):
    predictions, samples = synthetic_frames()
    with pytest.raises(ValueError, match=error):
        align_predictions(mutate(predictions), samples)


def test_sample_keys_and_explicit_run_selection():
    predictions, samples = synthetic_frames()
    with pytest.raises(ValueError, match="duplicate key"):
        align_predictions(predictions, pd.concat([samples, samples.iloc[:1]]))
    split_race = samples.copy()
    split_race.loc[0, "split"] = "train"
    with pytest.raises(ValueError, match="split a race"):
        align_predictions(predictions, split_race)
    multi = pd.concat([predictions, predictions.assign(run_id=OTHER_RUN, score=0.3)])
    with pytest.raises(ValueError, match="Supply run_id"):
        align_predictions(multi, samples)
    run, rows = align_predictions(multi, samples, run_id=OTHER_RUN)
    assert run == OTHER_RUN
    assert list(rows["score"]) == [0.3] * 4


def check_validation_artifacts(val, samples, manifest, output, *, commit, bootstrap_samples):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    threshold_path, metrics_path = output / "threshold.json", output / "metrics.json"
    threshold, metrics = (json.loads(path.read_bytes()) for path in (threshold_path, metrics_path))
    inputs = {
        "samples_sha256": sha(samples),
        "predictions_sha256": sha(val),
        "data_manifest_sha256": sha(manifest),
        "source": "synthetic_fixture",
    }
    common = {
        "schema_version": 1,
        "run_id": RUN,
        "data_sha256": sha(manifest),
        "git_commit": commit,
    }
    assert common.items() <= metrics.items()
    assert threshold == {
        **common,
        "threshold": 1,
        "validation_f1": 1,
        "chosen_on_split": "val",
        "selection_metric": "f1",
        "tie_break": "largest",
        "inputs": inputs,
        "git_dirty": False,
    }
    assert metrics["threshold_file_sha256"] == sha(threshold_path)
    assert metrics["split_provenance"]["val"]["inputs"] == inputs
    assert metrics["bootstrap"] | {"undefined_replicates": {}} == {
        "unit": "race",
        "samples": bootstrap_samples,
        "confidence_level": 0.95,
        "seed": 42,
        "method": "percentile",
        "undefined_handling": "omit_per_metric",
        "undefined_replicates": {},
    }
    assert set(metrics["splits"]) == {"val"}
    groups = metrics["splits"]["val"]
    assert {group: values["n_samples"] for group, values in groups.items()} == {
        "overall": 4,
        "green": 2,
        "sc_vsc": 2,
        "special_tyre_rule": 3,
    }
    overall = groups["overall"]
    assert (overall["n_positives"], overall["n_races"], overall["prevalence"]) == (1, 2, 0.25)
    assert overall["average_precision"] == overall["precision"] == overall["recall"] == 1
    assert overall["f1"] == overall["threshold"] == 1
    for k in ("1", "2"):
        assert overall["events"][k] == {
            "precision": 1,
            "recall": 1,
            "f1": 1,
            "false_alarms_per_race": 0,
        }
        assert overall["confidence_intervals"][f"events.{k}.recall"] == {"lower": 1, "upper": 1}
    assert groups["sc_vsc"]["average_precision"] is None
    assert groups["sc_vsc"]["undefined"]["average_precision"] == "no_positives"
    assert metrics["bootstrap"]["undefined_replicates"]["val"]["sc_vsc"]["f1"] == bootstrap_samples
    assert metrics["leakage_audit"]["status"] == "synthetic_fixture"
    assert metrics["leakage_audit"]["required"] is False
    return metrics


def test_cli_exports_valid_artifacts_and_exact_retry(files, capsys):
    val, samples, manifest, output, _ = files
    argv = [
        "evaluate",
        "--predictions",
        str(val),
        "--samples",
        str(samples),
        "--data-manifest",
        str(manifest),
        "--output-dir",
        str(output),
        "--synthetic",
        "--bootstrap-samples",
        "20",
    ]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["complete"] is True
    check_validation_artifacts(val, samples, manifest, output, commit=COMMIT, bootstrap_samples=20)
    before = {file.name: file.stat().st_ino for file in output.iterdir()}
    assert main(argv) == 0
    assert {file.name: file.stat().st_ino for file in output.iterdir()} == before


def test_frozen_threshold_and_validation_metrics_preserved_on_test_fixture(files):
    val, samples, manifest, output, test = files
    options = {"synthetic": True, "bootstrap_samples": 8}
    with pytest.raises(ValueError, match="threshold already frozen"):
        evaluate_files(
            test, samples, manifest, output, split="test", test_approval="fixture", **options
        )
    val_metrics = evaluate_files(val, samples, manifest, output, **options)
    threshold_path = output / "threshold.json"
    threshold_before = threshold_path.read_bytes(), threshold_path.stat().st_ino
    combined = evaluate_files(
        test,
        samples,
        manifest,
        output,
        split="test",
        test_approval="test suite synthetic fixture",
        **options,
    )
    assert (threshold_path.read_bytes(), threshold_path.stat().st_ino) == threshold_before
    assert combined["splits"]["val"] == val_metrics["splits"]["val"]
    assert combined["splits"]["test"]["overall"]["threshold"] == 1
    assert combined["splits"]["test"]["overall"]["f1"] == 0
    assert combined["split_provenance"]["test"]["test_approval"] == "test suite synthetic fixture"

    # A conflicting retry must leave both files intact.
    metrics_before = (output / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="different bootstrap settings"):
        evaluate_files(val, samples, manifest, output, synthetic=True, bootstrap_samples=9)
    pred = pd.read_parquet(val)
    pred.assign(score=0.5).to_parquet(val, index=False)
    with pytest.raises(ValueError, match="different producing inputs"):
        evaluate_files(val, samples, manifest, output, **options)
    assert (output / "metrics.json").read_bytes() == metrics_before


def test_cli_blocks_test_before_reading_inputs(tmp_path, capsys):
    absent = str(tmp_path / "absent")
    assert (
        main(
            [
                "evaluate",
                "--split",
                "test",
                "--predictions",
                absent,
                "--samples",
                absent,
                "--data-manifest",
                absent,
                "--output-dir",
                absent,
            ]
        )
        == 1
    )
    assert "explicit human approval" in capsys.readouterr().err
    assert not Path(absent).exists()


def test_project_mode_flags_high_fixture_scores(files, capsys):
    val, samples, manifest, output, _ = files
    manifest.write_text('{"fixture_for_project_mode_test": true}\n')
    with pytest.raises(ValueError, match="requires a fixture manifest"):
        evaluate_files(val, samples, manifest, output, synthetic=True)
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(val),
                "--samples",
                str(samples),
                "--data-manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--bootstrap-samples",
                "8",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is False
    audit = result["leakage_audit"]
    assert audit["required"] is True
    assert audit["status"] == "pending"
    assert "overall.f1" in audit["by_split"]["val"]["evidence"]["triggered_metrics"]


def test_publish_failure_rolls_back_new_threshold(files, monkeypatch):
    val, samples, manifest, output, _ = files
    import os

    link = os.link

    def fail_metrics(src, dst):
        if dst.name == "metrics.json":
            raise OSError("simulated publication failure")
        return link(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr("pitstop.evaluate.os.link", fail_metrics)
        with pytest.raises(OSError, match="publication failure"):
            evaluate_files(val, samples, manifest, output, synthetic=True, bootstrap_samples=4)
    assert list(output.iterdir()) == []
    evaluate_files(val, samples, manifest, output, synthetic=True, bootstrap_samples=4)
    assert {path.name for path in output.iterdir()} == {"threshold.json", "metrics.json"}
