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
NETWORKS = ("mlp", "gru", "tcn")
MODELS = ("constant", "logreg", "lightgbm", "mlp", "gru", "tcn")
SEEDS = (42, 43, 44)
RUNS = ROOT / "runs"
FINAL = ROOT / "configs" / "final"
SAMPLES, MANIFEST = "data/processed/v1/samples.parquet", "manifests/data.json"
GROUP_METRICS = ("green_average_precision", "green_f1", "sc_vsc_average_precision", "sc_vsc_f1")


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
    for attempt in range(10):
        try:
            run_id = train(str(config_path))["run_id"]
            break
        except FileExistsError:  # run ids have 1 s resolution; parallel starts can collide
            if attempt == 9:
                raise
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


def val_metrics(run_id, split="val"):
    return json.loads((RUNS / run_id / "metrics.json").read_text())["splits"][split]


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


def _row(run_id, split="val"):
    config = yaml.safe_load((RUNS / run_id / "config.yaml").read_text())
    val = val_metrics(run_id, split)
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


def _scores(run_id, split="val"):
    from pitstop.evaluate import align_predictions

    predictions = pd.read_parquet(RUNS / run_id / "predictions.parquet")
    samples = pd.read_parquet(ROOT / SAMPLES)
    rows = align_predictions(predictions, samples, split=split, run_id=run_id)[1]
    threshold = json.loads((RUNS / run_id / "threshold.json").read_text())["threshold"]
    return rows, threshold


def compare(rows, bootstrap_samples=1000, split="val"):
    """Paired race-level bootstrap of every model's AP minus LightGBM's, per seed."""
    from pitstop.metrics import paired_bootstrap

    ids = {(r["model"], r["seed"]): r["run_id"] for r in rows}
    out = []
    for (model, seed), run_id in ids.items():
        if model == "lightgbm":
            continue
        (left, left_t), (right, right_t) = (
            _scores(i, split) for i in (run_id, ids[("lightgbm", seed)])
        )
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
    groups = GROUP_METRICS
    split = [
        [model] + ["n/a" if m[g]["mean"] is None else f"{m[g]['mean']:.3f}" for g in groups]
        for model, m in doc["models"].items()
    ]
    paired = [
        [c["model"], str(c["seed"]), f"{c['ap_difference']:+.4f}"]
        + [f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}]", "tie" if c["tie"] else "differs"]
        for c in doc["paired_bootstrap_vs_lightgbm"]
    ]
    counts = [
        [
            name,
            str(g["n_samples"]),
            str(g["n_positives"]),
            str(g["n_races"]),
            f"{g['n_positives'] / g['n_samples']:.3f}",
        ]
        for name, g in doc["val_groups"].items()
    ]
    overall = doc["val_groups"]["overall"]
    tuned_ap = next(
        t["average_precision"] for t in doc["tuning"] if t["model"] == "tcn" and t["selected"]
    )
    tied = sum(c["tie"] for c in doc["paired_bootstrap_vs_lightgbm"] if c["model"] in NETWORKS)
    n_networks = sum(c["model"] in NETWORKS for c in doc["paired_bootstrap_vs_lightgbm"])
    return (
        "\n\n".join(
            [
                "# Validation study v1 (val split only)",
                f"Runs produced by code revision `{doc['git_commit']}`, "
                f"data `{doc['data_sha256']}`.",
                "Caveats: configs were selected on val AP at seed 42 only, and the seed-42 final "
                "runs are the tuning runs, so val metrics of the selected configs are "
                f"optimistically biased (e.g. TCN tuning AP {tuned_ap:.4f} vs its 3-seed mean "
                f"{doc['models']['tcn']['average_precision']['mean']:.3f}). "
                f"All val numbers come from one split of {overall['n_races']} races. "
                "Unbiased numbers come only from the locked test split (T12).",
                "## Tuning (seed 42, val AP)",
                _table(["model", "learning rate", "hidden", "val AP", "selected"], tuning),
                "## Models, mean ± sd over seeds 42, 43, 44",
                _table(
                    ["model", "val AP", "val F1", "event k=1 F1", "false alarms / race"], models
                ),
                "## Green vs SC/VSC laps, mean over seeds",
                _table(["group", "laps", "positives", "races", "prevalence"], counts),
                _table(["model", "green AP", "green F1", "SC/VSC AP", "SC/VSC F1"], split),
                "SC/VSC has very few positives, so its AP is very noisy and no performance claim "
                "is made for it. An F1 of 0 means the frozen threshold (selected on the "
                "green-dominated val split) produces no SC/VSC hits.",
                "## AP difference vs LightGBM (paired race bootstrap, 95% CI)",
                _table(["model", "seed", "AP difference", "95% CI", "result"], paired),
                f"The networks are statistically tied with LightGBM in {tied} of {n_networks} "
                "comparisons (a tie means the CI contains 0); logreg and constant are clearly "
                "worse. No network is claimed to beat LightGBM.",
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
        "val_groups": {  # same val rows for every run
            name: {k: g[k] for k in ("n_samples", "n_positives", "n_races")}
            for name, g in val_metrics(rows[0]["run_id"]).items()
        },
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


def markdown_test(doc):
    """Render the test summary: test means next to the val means, groups, paired bootstrap."""
    names = ("average_precision", "f1", "event_k1_f1", "event_k1_false_alarms_per_race")
    header = ["model"] + [f"{n} {s}" for n in names for s in ("val", "test")]
    models = [
        [model]
        + [
            cell
            for n in names
            for cell in (
                f"{doc['val_models'][model][n]['mean']:.3f}",
                f"{m[n]['mean']:.3f} ± {m[n]['sd']:.3f}",
            )
        ]
        for model, m in doc["models"].items()
    ]
    groups = [
        [name, str(g["n_samples"]), str(g["n_positives"]), str(g["n_races"])]
        for name, g in doc["test_groups"].items()
    ]
    split = [
        [model] + ["n/a" if m[g]["mean"] is None else f"{m[g]['mean']:.3f}" for g in GROUP_METRICS]
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
                "# Locked test evaluation v1 (test split, evaluated once)",
                f"Runs produced by code revision `{doc['git_commit']}`, data "
                f"`{doc['data_sha256']}`. Approval: {doc['approval']}",
                "Frozen models and val thresholds; nothing was retrained or retuned. "
                "Mean over seeds 42, 43, 44 (± sample sd for test); val means from "
                "reports/validation-v1.json.",
                "## Models, val vs test",
                _table(header, models),
                "## Test groups",
                _table(["group", "laps", "positives", "races"], groups),
                _table(["model", "green AP", "green F1", "SC/VSC AP", "SC/VSC F1"], split),
                "Group counts are on the test split. SC/VSC and special-tyre-rule groups have "
                "few positives or races, so their metrics are noisy; no claim is made for them.",
                "## AP difference vs LightGBM (paired race bootstrap, 95% CI)",
                _table(["model", "seed", "AP difference", "95% CI", "result"], paired),
                f"A CI containing 0 is a tie. The test split has only "
                f"{doc['test_groups']['overall']['n_races']} races, so CIs are wide; a "
                "`differs` row at a single seed does not establish a ranking, and no model "
                "ranking is claimed.",
            ]
        )
        + "\n"
    )


def evaluate_test(approval, bootstrap_samples=1000):
    """Evaluate the 18 final runs on test once (after `pitstop predict --test-approval`).

    Writes reports/test-v1.{json,md}; raises on the D8 leak alarm before writing anything.
    """
    from pitstop.evaluate import evaluate_files

    ids = json.loads((RUNS / "final.json").read_text())
    for run_id in ids:
        run_dir = f"runs/{run_id}"
        metrics = json.loads((RUNS / run_id / "metrics.json").read_text())
        if "test" not in metrics["splits"]:  # a second call only rebuilds the summary
            metrics = evaluate_files(
                f"{run_dir}/predictions.parquet",
                SAMPLES,
                MANIFEST,
                run_dir,
                run_id=run_id,
                split="test",
                test_approval=approval,
            )
        if metrics["leakage_audit"]["by_split"]["test"]["required"]:
            raise RuntimeError(f"Leak alarm (D8): test AP or F1 > 0.6 in {run_id}")
    rows = [_row(i, "test") for i in ids]
    provenance = [
        json.loads((RUNS / i / "metrics.json").read_text())["split_provenance"]["test"] for i in ids
    ]
    commits = {p["git_commit"] for p in provenance}
    approvals = {p["test_approval"] for p in provenance}
    if len(commits) != 1 or len(approvals) != 1 or any(p["git_dirty"] for p in provenance):
        raise ValueError(f"Test runs must share one clean revision and approval: {commits}")
    validation = json.loads((ROOT / "reports" / "validation-v1.json").read_text())
    doc = {
        "split": "test",
        "git_commit": commits.pop(),
        "approval": approvals.pop(),
        "data_sha256": (RUNS / ids[0] / "data_hash.txt").read_text().strip(),
        "bootstrap_samples": bootstrap_samples,
        "test_groups": {
            name: {k: g[k] for k in ("n_samples", "n_positives", "n_races")}
            for name, g in val_metrics(ids[0], "test").items()
        },
        "runs": rows,
        "models": aggregate(rows),
        "val_models": validation["models"],
        "paired_bootstrap_vs_lightgbm": compare(rows, bootstrap_samples, "test"),
    }
    write_json(ROOT / "reports" / "test-v1.json", doc)
    (ROOT / "reports" / "test-v1.md").write_text(markdown_test(doc))
    return doc


if __name__ == "__main__":  # uv run python -m pitstop.experiments tune|final|summary
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("tune", "final", "summary", "test"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--test-approval", help="Required for the test step")
    parser.add_argument("--threads", type=int, default=4, help="Torch threads per worker")
    args = parser.parse_args()
    if args.step == "test":
        evaluate_test(args.test_approval)
    elif args.step == "summary":
        summarize()
    elif args.step == "tune":
        tune(args.workers, args.threads)
    else:
        final(args.workers, args.threads)
