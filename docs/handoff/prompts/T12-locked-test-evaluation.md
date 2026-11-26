TASK T12 — single locked test evaluation of the frozen T11 models     Owner: B     Risk: 3

PROJECT_ROOT = <application checkout>. NOTES_ROOT = <sibling notes directory>.
Report: NOTES_ROOT/reports/T12.md (written so it can serve as the PR body).
Branch: codex/t12-test-evaluation from up-to-date main (T11 merged as 228f320).

HUMAN APPROVAL = <paste the exact approval text, who gave it and the date>.
If this line is still a placeholder, STOP before step 2: do not read, score or
evaluate any test row.

Outcome: the 18 final T11 runs (6 models × seeds 42, 43, 44, listed under
"runs" in reports/validation-v1.json) are scored on the locked test split
exactly once with their frozen models and frozen val thresholds, evaluated
with T5 `pitstop evaluate --split test --test-approval "<HUMAN APPROVAL>"`, and
summarised in reports/test-v1.{json,md}. Nothing is retrained, retuned or
re-thresholded. The demo pack is NOT part of this task (T12b).

Read first: AGENTS.md; docs/decisions.md D4 (locked test), D7, D8 (leak alarm,
ties inside the CI), leakage rules; docs/contracts.md "train → predict",
"predict → evaluate", "evaluate → dashboard"; docs/runbook.md (T5, T11);
src/pitstop/{predict,evaluate,experiments,cli}.py; reports/validation-v1.*.

Steps:
1. Freeze check (no test access): for every run in validation-v1.json "runs",
   confirm the run directory exists under runs/, git_commit.txt is
   f91914766761844937bd1e83e77740daa5af8a17, environment.json git_dirty is
   false, threshold.json has chosen_on_split "val", and metrics.json has no
   "test" split. Write reports/test-v1-freeze.json with the run ids and the
   SHA-256 of each run's config.yaml, model file(s), threshold.json and
   metrics.json, plus the approval text. Commit it BEFORE step 2. If any run
   is missing or differs, STOP and report (the team decides whether to
   retrain; do not retrain yourself).
2. Code (smallest change): `predict_run` keeps scoring only train/val by
   default; an explicit non-empty test approval (function argument and a
   `--test-approval` flag on the existing `predict` subcommand, no new
   command) also scores test rows. Train/val scores must stay bit-identical.
   Extend experiments.py with a `test` step that evaluates and summarises
   (reuse `_row`, `aggregate`, `compare`, `summary_markdown`; parametrise the
   split rather than copying code). If a contract change seems necessary,
   STOP and report: it needs a dedicated contract PR.
3. Offline tests on tiny synthetic runs: without approval no test row is
   scored; with approval test rows appear and train/val scores are unchanged;
   the test summary is built from metrics.json files. (evaluate already
   returns stored metrics for a repeated split with identical inputs and
   raises for different inputs; do not change that.) No real data in tests.
4. Commit the code, then run once from that clean commit: predict with
   approval, `pitstop evaluate --split test --test-approval ...` per run (the
   val entry and threshold.json must stay unchanged; check the hashes from
   step 1), then the summary. If any test AP or F1 > 0.6, STOP: leak alarm
   (D8), report without publishing numbers.
5. After test access, never change a model, config, threshold or seed and
   never repeat the test evaluation. Regenerating the summary from saved
   files is allowed. If scoring or evaluation fails midway, STOP and report.

Summary content (reports/test-v1.json and .md; every md number from the json,
the json from runs/): per model mean ± sd over seeds of test AP, F1, event
k=1 F1 and false alarms per race, side by side with the val means from
validation-v1.json; group breakdown (green, SC/VSC, special tyre rule) with
sample/positive/race counts; paired race bootstrap (B = 1000) of each model vs
LightGBM per seed with the AP difference, 95% CI and tie flag; code revision,
data hash and approval text. Describe results neutrally: a CI containing 0 is
a tie; make no claims about the SC/VSC group beyond counts if positives are few.

Allowed tracked files: src/pitstop/predict.py, src/pitstop/cli.py (predict flag
only), src/pitstop/experiments.py, tests/test_experiments.py, tests/test_train.py
or a new tests/test_test_split.py, reports/test-v1-freeze.json,
reports/test-v1.json, reports/test-v1.md, docs/runbook.md (T12 section).
Do NOT edit contracts.py, docs/contracts.md, docs/decisions.md, configs, model
code, train.py, evaluate.py or validation-v1.*. No new dependencies. Never
commit runs/ or data/.

Acceptance:
uv run --offline --no-sync ruff check . &&
uv run --offline --no-sync ruff format --check . &&
uv run --offline --no-sync pytest
plus step 4 from one clean committed revision, with the step-1 hashes of
config, model and threshold files unchanged afterwards.

Deliver: commits on the branch, the report with full head SHA, scope, checks,
the freeze table, the test comparison table with CIs and ties, val vs test
side by side, the approval text as recorded in metrics.json, and known gaps.
Do not push unless the orchestrator's launch message says so.
