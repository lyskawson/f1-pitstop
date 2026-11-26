import ast
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
RUNS = ["gru-1", "mlp-1"]


def make_pack(path):
    rows = [
        (race, drv, lap)
        for race in ["2023_01_Race", "2023_02_Race"]
        for drv in ["1", "44"]
        for lap in range(1, 21)
    ]
    samples = pd.DataFrame(rows, columns=["race_id", "driver_number", "lap_number"])
    samples["row_id"] = range(len(samples))
    samples["y"] = (samples.lap_number % 9 == 0).astype("int8")
    samples["split"] = "val"
    samples["sc"] = samples.lap_number.between(5, 6)
    samples["vsc"] = samples.lap_number == 12
    samples["special_tyre_rule"] = False
    samples.to_parquet(path / "samples.parquet")
    preds = pd.concat([samples.assign(run_id=r) for r in RUNS], ignore_index=True)
    preds["score"] = (preds.lap_number % 10) / 10
    preds["target_lap"] = preds.lap_number + 1
    preds[
        ["race_id", "driver_number", "lap_number", "run_id", "split", "y", "score", "target_lap"]
    ].to_parquet(path / "predictions.parquet")
    (path / "thresholds.json").write_text(json.dumps({r: {"threshold": 0.5} for r in RUNS}))


def run_app(monkeypatch, pack):
    monkeypatch.setenv("PITSTOP_DEMO_DIR", str(pack))
    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_default_run(tmp_path, monkeypatch):
    make_pack(tmp_path)
    at = run_app(monkeypatch, tmp_path)
    assert not at.exception
    assert len(at.selectbox) == 3


def test_switch_race_driver_model(tmp_path, monkeypatch):
    make_pack(tmp_path)
    at = run_app(monkeypatch, tmp_path)
    at.selectbox[0].select("2023_02_Race").run()
    assert not at.exception
    at.selectbox[1].select("44").run()
    assert not at.exception
    at.selectbox[2].select("mlp-1").run()
    assert not at.exception


def test_missing_pack_shows_info(tmp_path, monkeypatch):
    at = run_app(monkeypatch, tmp_path / "nope")
    assert not at.exception
    assert len(at.info) == 1


def test_app_has_no_heavy_imports():
    banned = {"torch", "lightgbm", "fastf1", "pitstop"}
    imported = set()
    for node in ast.walk(ast.parse(APP.read_text())):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & banned
