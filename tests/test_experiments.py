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
        {"run_id": f"r{s}", "model": "mlp", "seed": s, "average_precision": 0.2 + s / 100}
        | dict.fromkeys(SPLIT_AND_EVENT_METRICS, 0.5)
        | {"sc_vsc_f1": None}
        for s in (1, 2, 3)
    ]
    models = experiments.aggregate(rows)
    assert models["mlp"]["average_precision"]["mean"] == pytest.approx(0.22)
    assert models["mlp"]["average_precision"]["sd"] == pytest.approx(0.01)
    assert models["mlp"]["sc_vsc_f1"] == {"mean": None, "sd": None}
    doc = {
        "git_commit": "x",
        "data_sha256": "y",
        "tuning": [],
        "models": models,
        "paired_bootstrap_vs_lightgbm": [
            {"model": "mlp", "seed": 1, "ap_difference": -0.1, "ci_lower": -0.2, "ci_upper": 0.0}
            | {"tie": True}
        ],
    }
    text = experiments.summary_markdown(doc)
    assert "0.220 ± 0.010" in text and "n/a" in text and "tie" in text
