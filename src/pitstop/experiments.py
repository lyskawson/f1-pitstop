"""Bounded validation study: tune networks on val, run final configs with 3 seeds, compare."""

import argparse
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import pandas as pd
import yaml

from pitstop.ingest import ROOT, write_json

LRS = (5e-4, 1e-3, 2e-3)
HIDDEN = (32, 64)  # MLP uses [h, h // 2]; GRU hidden_size; TCN channels
SETTERS = {
    "mlp": lambda a, h: a.update(hidden_sizes=[h, h // 2]),
    "gru": lambda a, h: a.update(hidden_size=h),
    "tcn": lambda a, h: a.update(channels=h),
}
MODELS = ("constant", "logreg", "lightgbm", "mlp", "gru", "tcn")
SEEDS = (42, 43, 44)
RUNS = ROOT / "runs"
FINAL = ROOT / "configs" / "final"
SAMPLES, MANIFEST = "data/processed/v1/samples.parquet", "manifests/data.json"


def grid_configs(out_dir):
    """Write the D6 grid (3 lr x 2 sizes per network) as complete YAMLs; return (path, meta)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model, setter in SETTERS.items():
        for lr, hidden in product(LRS, HIDDEN):
            config = yaml.safe_load((ROOT / "configs" / f"{model}.yaml").read_text())
            config["training"]["learning_rate"] = lr
            setter(config["architecture"], hidden)
            path = out_dir / f"{model}_lr{lr:g}_h{hidden}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            jobs.append((path, {"model": model, "lr": lr, "hidden": hidden}))
    return jobs


def run_one(config_path, threads):
    """Train, predict and evaluate (val) one config; return its run id. Runs in a worker."""
    import torch

    from pitstop.evaluate import evaluate_files
    from pitstop.train import train

    torch.set_num_threads(threads)
    while True:
        try:
            run_id = train(str(config_path))["run_id"]
            break
        except FileExistsError:  # run ids have 1 s resolution; parallel starts can collide
            time.sleep(1)
    run_dir = f"runs/{run_id}"
    metrics = evaluate_files(
        f"{run_dir}/predictions.parquet", SAMPLES, MANIFEST, run_dir, run_id=run_id
    )
    if metrics["leakage_audit"]["required"]:
        raise RuntimeError(f"Leak alarm (D8): val AP or F1 > 0.6 in {run_id}")
    return run_id


def run_all(paths, workers=3, threads=4):
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        return list(pool.map(run_one, paths, [threads] * len(paths)))


def val_metrics(run_id):
    return json.loads((RUNS / run_id / "metrics.json").read_text())["splits"]["val"]


def select(tuning):
    """Best val AP per network; ties go to the smaller hidden size, then the lower lr."""
    best = {}
    for row in tuning:
        ap = val_metrics(row["run_id"])["overall"]["average_precision"]
        key = (-ap, row["hidden"], row["lr"])
        if row["model"] not in best or key < best[row["model"]][0]:
            best[row["model"]] = (key, row)
    return {model: row for model, (_, row) in best.items()}


def tune(workers=3, threads=4):
    """Run the grid, then write configs/final/<model>.yaml for every model."""
    jobs = grid_configs(RUNS / "grid_configs")
    run_ids = run_all([path for path, _ in jobs], workers, threads)
    tuning = [meta | {"run_id": run_id} for (_, meta), run_id in zip(jobs, run_ids, strict=True)]
    write_json(RUNS / "tuning.json", tuning)
    FINAL.mkdir(parents=True, exist_ok=True)
    for model in ("constant", "logreg", "lightgbm"):  # baselines are not tuned (D6)
        shutil.copy(ROOT / "configs" / f"{model}.yaml", FINAL / f"{model}.yaml")
    for model, row in select(tuning).items():
        shutil.copy(RUNS / row["run_id"] / "config.yaml", FINAL / f"{model}.yaml")
    return tuning


def final(workers=3, threads=4):
    """Run every final config with seeds 42, 43, 44."""
    out = RUNS / "final_configs"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in reversed(MODELS):  # slow networks first
        for seed in SEEDS:
            config = yaml.safe_load((FINAL / f"{model}.yaml").read_text()) | {"seed": seed}
            paths.append(out / f"{model}_s{seed}.yaml")
            paths[-1].write_text(yaml.safe_dump(config, sort_keys=False))
    run_ids = run_all(paths, workers, threads)
    write_json(RUNS / "final.json", run_ids)
    return run_ids


def _row(run_id):
    config = yaml.safe_load((RUNS / run_id / "config.yaml").read_text())
    val = val_metrics(run_id)
    overall = val["overall"]
    return {
        "run_id": run_id,
        "model": config["model"],
        "seed": config["seed"],
        "average_precision": overall["average_precision"],
        "f1": overall["f1"],
        "event_k1_f1": overall["events"]["1"]["f1"],
        "event_k1_false_alarms_per_race": overall["events"]["1"]["false_alarms_per_race"],
        "green_average_precision": val["green"]["average_precision"],
        "green_f1": val["green"]["f1"],
        "sc_vsc_average_precision": val["sc_vsc"]["average_precision"],
        "sc_vsc_f1": val["sc_vsc"]["f1"],
    }


def aggregate(rows):
    """Mean and sample sd (ddof=1) over seeds for every metric of every model."""
    frame = pd.DataFrame(rows).drop(columns=["run_id", "seed"]).groupby("model")
    mean, sd = frame.mean(), frame.std()

    def number(value):  # undefined metrics (e.g. no SC/VSC positives) stay None, not NaN
        return None if pd.isna(value) else float(value)

    return {
        model: {
            m: {"mean": number(mean.loc[model, m]), "sd": number(sd.loc[model, m])} for m in mean
        }
        for model in mean.index
    }


def _scores(run_id):
    from pitstop.evaluate import align_predictions

    predictions = pd.read_parquet(RUNS / run_id / "predictions.parquet")
    samples = pd.read_parquet(ROOT / SAMPLES)
    rows = align_predictions(predictions, samples, split="val", run_id=run_id)[1]
    threshold = json.loads((RUNS / run_id / "threshold.json").read_text())["threshold"]
    return rows, threshold


def compare(rows, bootstrap_samples=1000):
    """Paired race-level bootstrap of every model's AP minus LightGBM's, per seed."""
    from pitstop.metrics import paired_bootstrap

    ids = {(r["model"], r["seed"]): r["run_id"] for r in rows}
    out = []
    for (model, seed), run_id in ids.items():
        if model == "lightgbm":
            continue
        (left, left_t), (right, right_t) = _scores(run_id), _scores(ids[("lightgbm", seed)])
        result = paired_bootstrap(left, right, left_t, right_t, bootstrap_samples=bootstrap_samples)
        ci = result["confidence_intervals"]["average_precision"]
        out.append(
            {
                "model": model,
                "seed": seed,
                "ap_difference": result["differences"]["average_precision"],
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
                "tie": result["ties"]["average_precision"],
            }
        )
    return out


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def summary_markdown(doc):
    """Render the JSON summary as short Markdown tables."""
    tuning = [
        [t["model"], f"{t['lr']:g}", str(t["hidden"]), f"{t['average_precision']:.4f}"]
        + ["yes" if t["selected"] else ""]
        for t in doc["tuning"]
    ]
    names = ("average_precision", "f1", "event_k1_f1", "event_k1_false_alarms_per_race")
    models = [
        [model] + [f"{m[n]['mean']:.3f} ± {m[n]['sd']:.3f}" for n in names]
        for model, m in doc["models"].items()
    ]
    groups = ("green_average_precision", "green_f1", "sc_vsc_average_precision", "sc_vsc_f1")
    split = [
        [model] + ["n/a" if m[g]["mean"] is None else f"{m[g]['mean']:.3f}" for g in groups]
        for model, m in doc["models"].items()
    ]
    paired = [
        [c["model"], str(c["seed"]), f"{c['ap_difference']:+.4f}"]
        + [f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}]", "tie" if c["tie"] else "differs"]
        for c in doc["paired_bootstrap_vs_lightgbm"]
    ]
    return (
        "\n\n".join(
            [
                "# Validation study v1 (val split only)",
                f"Code revision `{doc['git_commit']}`, data `{doc['data_sha256']}`.",
                "## Tuning (seed 42, val AP)",
                _table(["model", "learning rate", "hidden", "val AP", "selected"], tuning),
                "## Models, mean ± sd over seeds 42, 43, 44",
                _table(
                    ["model", "val AP", "val F1", "event k=1 F1", "false alarms / race"], models
                ),
                "## Green vs SC/VSC laps, mean over seeds",
                _table(["model", "green AP", "green F1", "SC/VSC AP", "SC/VSC F1"], split),
                "## AP difference vs LightGBM (paired race bootstrap, 95% CI)",
                _table(["model", "seed", "AP difference", "95% CI", "result"], paired),
            ]
        )
        + "\n"
    )


def summarize(bootstrap_samples=1000):
    """Build reports/validation-v1.json and .md from runs/ (tuning.json and final.json)."""
    tuning = json.loads((RUNS / "tuning.json").read_text())
    chosen = {row["run_id"] for row in select(tuning).values()}
    for t in tuning:
        t["average_precision"] = val_metrics(t["run_id"])["overall"]["average_precision"]
        t["selected"] = t["run_id"] in chosen
    rows = [_row(run_id) for run_id in json.loads((RUNS / "final.json").read_text())]
    ids = [t["run_id"] for t in tuning] + [r["run_id"] for r in rows]
    commits = {(RUNS / i / "git_commit.txt").read_text().strip() for i in ids}
    dirty = [i for i in ids if json.loads((RUNS / i / "environment.json").read_text())["git_dirty"]]
    if len(commits) != 1 or dirty:
        raise ValueError(f"Runs must share one clean revision: {commits}, dirty {dirty}")
    doc = {
        "split": "val",
        "git_commit": commits.pop(),
        "data_sha256": (RUNS / rows[0]["run_id"] / "data_hash.txt").read_text().strip(),
        "bootstrap_samples": bootstrap_samples,
        "tuning": tuning,
        "runs": rows,
        "models": aggregate(rows),
        "paired_bootstrap_vs_lightgbm": compare(rows, bootstrap_samples),
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "validation-v1.json", doc)
    (reports / "validation-v1.md").write_text(summary_markdown(doc))
    return doc


if __name__ == "__main__":  # uv run python -m pitstop.experiments tune|final|summary
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("tune", "final", "summary"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4, help="Torch threads per worker")
    args = parser.parse_args()
    summarize() if args.step == "summary" else globals()[args.step](args.workers, args.threads)
