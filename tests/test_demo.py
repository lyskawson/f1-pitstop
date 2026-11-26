import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pitstop.demo import PAYLOAD, _sha256, export_demo

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
RACES = ["2025_14_R", "2025_15_R", "2025_16_R", "2025_17_R"]
RUNS = {
    ("gru", 42): "20260101T000000Z_gru_s42_abcd1234",
    ("gru", 43): "20260101T000000Z_gru_s43_abcd1234",
    ("mlp", 42): "20260101T000000Z_mlp_s42_abcd1234",
}


def make_inputs(tmp_path, approved=True):
    rows = [(r, d, lap) for r in ["2025_01_R", *RACES] for d in ["1", "44"] for lap in range(1, 8)]
    samples = pd.DataFrame(rows, columns=["race_id", "driver_number", "lap_number"])
    samples["row_id"] = range(len(samples))
    samples["y"] = (samples.lap_number == 5).astype("int8")
    samples["split"] = ["val" if r == "2025_01_R" else "test" for r in samples.race_id]
    samples["sc"] = samples.lap_number == 3
    samples["vsc"] = False
    samples["special_tyre_rule"] = False
    samples.to_parquet(tmp_path / "samples.parquet")
    for (_, _), run_id in RUNS.items():
        run = tmp_path / "runs" / run_id
        run.mkdir(parents=True)
        preds = samples.assign(run_id=run_id, score=0.3, target_lap=samples.lap_number + 1)
        preds.drop(columns=["row_id", "sc", "vsc", "special_tyre_rule"]).to_parquet(
            run / "predictions.parquet"
        )
        provenance = {"test": {"test_approval": "ok"}} if approved else {"val": {}}
        metrics = {"splits": {"test": {}} if approved else {"val": {}}}
        (run / "metrics.json").write_text(
            json.dumps({**metrics, "run_id": run_id, "split_provenance": provenance})
        )
        threshold = {"run_id": run_id, "threshold": 0.5, "git_commit": "a", "data_sha256": "b"}
        (run / "threshold.json").write_text(json.dumps(threshold))
    report = {"runs": [{"run_id": i, "seed": s} for (_, s), i in RUNS.items()]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    events = [{"race_id": r, "event_name": f"GP {r}"} for r in RACES]
    (tmp_path / "events.json").write_text(json.dumps(events))


def export(tmp_path, out="demo"):
    return export_demo(
        tmp_path / "runs",
        tmp_path / "samples.parquet",
        tmp_path / "events.json",
        tmp_path / "report.json",
        tmp_path / out,
        check_clean=False,
    )


def test_export_pack(tmp_path):
    make_inputs(tmp_path)
    meta = export(tmp_path)
    out = tmp_path / "demo"
    assert sorted(p.name for p in out.iterdir()) == sorted([*PAYLOAD, "metadata.json"])
    assert meta["race_ids"] == RACES[:3]
    assert meta["run_ids"] == [RUNS["gru", 42], RUNS["mlp", 42]]
    assert meta["payload_sha256"] == {name: _sha256(out / name) for name in PAYLOAD}
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert set(predictions.race_id) == set(RACES[:3]) and set(predictions.split) == {"test"}
    assert len(predictions) == 2 * len(pd.read_parquet(out / "samples.parquet"))
    assert set(json.loads((out / "thresholds.json").read_text())) == set(meta["run_ids"])


def test_unapproved_run_refused(tmp_path):
    make_inputs(tmp_path, approved=False)
    with pytest.raises(ValueError, match="no approved test metrics"):
        export(tmp_path)


def test_export_is_byte_identical(tmp_path):
    make_inputs(tmp_path)
    export(tmp_path, "a")
    export(tmp_path, "b")
    for name in [*PAYLOAD, "metadata.json"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_app_loads_exported_pack(tmp_path, monkeypatch):
    make_inputs(tmp_path)
    export(tmp_path)
    monkeypatch.setenv("PITSTOP_DEMO_DIR", str(tmp_path / "demo"))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert len(at.selectbox) == 3
