import json

import pytest
import yaml

from pitstop import experiments

SPLIT_AND_EVENT_METRICS = (
    "f1",
    "event_k1_f1",
    "event_k1_false_alarms_per_race",
    "green_average_precision",
    "green_f1",
    "sc_vsc_average_precision",
    "sc_vsc_f1",
)


def fake_run(root, run_id, ap):
    (root / run_id).mkdir()
    overall = {"average_precision": ap}
    metrics = {"splits": {"val": {"overall": overall}}}
    (root / run_id / "metrics.json").write_text(json.dumps(metrics))


def test_grid_has_six_complete_configs_per_network(tmp_path):
    jobs = experiments.grid_configs(tmp_path)
    assert len(jobs) == 18
    assert len({path for path, _ in jobs}) == 18
    mlp = [meta for _, meta in jobs if meta["model"] == "mlp"]
    assert {(m["lr"], m["hidden"]) for m in mlp} == {
        (lr, h) for lr in experiments.LRS for h in experiments.HIDDEN
    }
    config = yaml.safe_load(jobs[0][0].read_text())
    assert config["architecture"]["hidden_sizes"] == [32, 16]
    assert config["data"]["samples"] and config["training"]["learning_rate"] == 5e-4


def test_select_prefers_val_ap_then_smaller_config(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RUNS", tmp_path)
    for run_id, ap in (("a", 0.3), ("b", 0.4), ("c", 0.4), ("d", 0.1)):
        fake_run(tmp_path, run_id, ap)
    tuning = [
        {"model": "gru", "lr": 1e-3, "hidden": 64, "run_id": "a"},
        {"model": "gru", "lr": 2e-3, "hidden": 64, "run_id": "b"},
        {"model": "gru", "lr": 2e-3, "hidden": 32, "run_id": "c"},
        {"model": "tcn", "lr": 1e-3, "hidden": 32, "run_id": "d"},
    ]
    chosen = experiments.select(tuning)
    assert chosen["gru"]["run_id"] == "c"  # tie on AP, smaller hidden wins
    assert chosen["tcn"]["run_id"] == "d"


def test_aggregate_and_markdown_keep_undefined_as_none():
    rows = [
        {"run_id": f"r{s}", "model": model, "seed": s, "average_precision": 0.2 + s / 100}
        | dict.fromkeys(SPLIT_AND_EVENT_METRICS, 0.5)
        | {"sc_vsc_f1": None}
        for s in (1, 2, 3)
        for model in ("mlp", "tcn")
    ]
    models = experiments.aggregate(rows)
    assert models["mlp"]["average_precision"]["mean"] == pytest.approx(0.22)
    assert models["mlp"]["average_precision"]["sd"] == pytest.approx(0.01)
    assert models["mlp"]["sc_vsc_f1"] == {"mean": None, "sd": None}
    doc = {
        "git_commit": "x",
        "data_sha256": "y",
        "val_groups": {
            g: {"n_samples": 100, "n_positives": 4, "n_races": 2} for g in ("overall", "sc_vsc")
        },
        "tuning": [
            {"model": "tcn", "lr": 0.001, "hidden": 64, "selected": True, "average_precision": 0.3}
        ],
        "models": models,
        "paired_bootstrap_vs_lightgbm": [
            {"model": "mlp", "seed": 1, "ap_difference": -0.1, "ci_lower": -0.2, "ci_upper": 0.0}
            | {"tie": True}
        ],
    }
    text = experiments.summary_markdown(doc)
    assert "0.220 ± 0.010" in text and "n/a" in text and "tie" in text
    assert "1 of 1 comparisons" in text and "0.3000" in text and "0.040" in text


def test_summarize_refuses_runs_from_two_commits(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RUNS", tmp_path)
    group = {"average_precision": 0.1, "f1": 0.1, "events": {"1": {"f1": 0.1}}}
    group["events"]["1"]["false_alarms_per_race"] = 1.0
    metrics = {"splits": {"val": {"overall": group, "green": group, "sc_vsc": group}}}
    for run_id, commit in (("a", "c1"), ("b", "c2")):
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "metrics.json").write_text(json.dumps(metrics))
        (tmp_path / run_id / "config.yaml").write_text("model: mlp\nseed: 42\n")
        (tmp_path / run_id / "git_commit.txt").write_text(commit)
        (tmp_path / run_id / "environment.json").write_text('{"git_dirty": false}')
    tuning = [{"model": "mlp", "lr": 1e-3, "hidden": 32, "run_id": "a"}]
    (tmp_path / "tuning.json").write_text(json.dumps(tuning))
    (tmp_path / "final.json").write_text('["b"]')
    with pytest.raises(ValueError, match="one clean revision"):
        experiments.summarize()


def test_test_summary_is_built_from_metrics_files(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    monkeypatch.setattr(experiments, "compare", lambda *args: [])
    (tmp_path / "reports").mkdir()
    (tmp_path / "runs").mkdir()
    val = {m: {"mean": 0.3, "sd": 0.1} for m in SPLIT_AND_EVENT_METRICS + ("average_precision",)}
    (tmp_path / "reports" / "validation-v1.json").write_text(json.dumps({"models": {"mlp": val}}))
    group = {"average_precision": 0.2, "f1": 0.25, "n_samples": 10, "n_positives": 2, "n_races": 1}
    group["events"] = {"1": {"f1": 0.2, "false_alarms_per_race": 3.0}}
    metrics = {
        "splits": {"test": dict.fromkeys(("overall", "green", "sc_vsc"), group)},
        "split_provenance": {
            "test": {"git_commit": "c", "git_dirty": False, "test_approval": "ok"}
        },
        "leakage_audit": {"by_split": {"test": {"required": False}}},
    }
    ids = ["a", "b"]
    for run_id in ids:
        (tmp_path / "runs" / run_id).mkdir()
        (tmp_path / "runs" / run_id / "metrics.json").write_text(json.dumps(metrics))
        (tmp_path / "runs" / run_id / "config.yaml").write_text("model: mlp\nseed: 42\n")
        (tmp_path / "runs" / run_id / "data_hash.txt").write_text("hash")
    (tmp_path / "runs" / "final.json").write_text(json.dumps(ids))
    doc = experiments.evaluate_test("ok")
    assert doc["models"]["mlp"]["average_precision"]["mean"] == pytest.approx(0.2)
    text = (tmp_path / "reports" / "test-v1.md").read_text()
    assert "0.300" in text and "0.200 ± 0.000" in text and "Approval: ok" in text
    metrics["leakage_audit"]["by_split"]["test"]["required"] = True
    (tmp_path / "runs" / "a" / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="Leak alarm"):
        experiments.evaluate_test("ok")
