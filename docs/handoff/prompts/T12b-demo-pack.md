TASK T12b — demo pack export for the dashboard     Owner: C     Risk: 2

PROJECT_ROOT = <application checkout>. NOTES_ROOT = <sibling notes directory>.
Report: NOTES_ROOT/reports/T12b.md (written so it can serve as the PR body).
Branch: codex/t12b-demo-pack from up-to-date main (after T12 #21 and the
runs-v1 handoff PR).

Outcome: `pitstop export-demo` (the existing CLI stub; no new command) writes
the D11 demo pack to `demo/` from the saved T12 runs, exactly as specified in
docs/contracts.md "evaluate → dashboard": `predictions.parquet`,
`samples.parquet`, `metrics.json`, `thresholds.json`, `metadata.json`, for the
first three retained test races in ascending race_id. The pack is committed
(D11: Streamlit Community Cloud reads it from the repository). The T7 app
(`app/streamlit_app.py`) shows the real pack without code changes.

Inputs (all exist locally; never in Git): `runs/` from the runs-v1 release
(see docs/runbook.md "Frozen runs (runs-v1)"), `data/processed/v1` built from
data-v1, event names from `data/raw/v1/events.json`. No training, no
prediction, no evaluation: the test split was evaluated once in T12 and must
NEVER be evaluated again. Only read saved files.

Read first: AGENTS.md; docs/decisions.md D4, D11; docs/contracts.md
"evaluate → dashboard" and "predict → evaluate"; docs/runbook.md (T7, T12);
reports/test-v1.json, reports/test-v1-freeze.json; app/streamlit_app.py and
tests/test_app.py (what the app reads); src/pitstop/cli.py (export-demo stub),
src/pitstop/experiments.py.

Design (state any interpretation in the report):
- Runs: the six final models at seed 42 (one run per model, ids from
  reports/test-v1.json). Record this chosen rule and its version in
  metadata.json. Refuse to export a run whose metrics.json has no test split
  with a recorded test_approval (contract: export only after authorized test
  evaluation).
- Races: the first three retained test race_ids in ascending order from
  samples.parquet (split == "test"); currently 2025_14_R, 2025_15_R,
  2025_16_R. Do not hard-code them.
- predictions.parquet: only those races' test rows of the chosen runs, same
  schema as the run predictions (validate with contracts.validate_predictions).
  samples.parquet: KEY plus the status fields the dashboard needs (sc, vsc,
  special_tyre_rule, y), test rows of the three races only.
- thresholds.json: threshold.json records keyed by run_id, copied unchanged.
  metrics.json: the saved test metrics per run_id, copied, not recomputed.
- metadata.json: race IDs, event names, chosen rule/version, run IDs, code
  revision of the export and of the runs, data hash, package versions, and the
  SHA-256 of every payload file.
- Deterministic: running the export twice gives byte-identical files.
- Keep it short: one small module (e.g. src/pitstop/demo.py) and the CLI
  wiring. Do not import torch or lightgbm in the export path.

Tests (offline, synthetic, no real data): export from tiny synthetic run
directories and samples selects the first three test races and the seed-42
runs, writes the five files, metadata hashes match the payloads, a run
without approved test metrics is refused, a second export is byte-identical,
and the app's AppTest loads the exported synthetic pack without exceptions.

Real export (local): run `pitstop export-demo` from a clean committed revision,
commit `demo/`, then start the app headless or via AppTest on `demo/` and
report that it renders race, driver and model selectors for the real pack.
Report the pack size (keep it well under 5 MB) and the per-race row counts.

Allowed tracked files: src/pitstop/demo.py (new), src/pitstop/cli.py
(export-demo wiring only), tests/test_demo.py (new), demo/* (the five pack
files), docs/runbook.md (T12b section; replace the T7 line "demo/ is never
committed"). Do NOT edit contracts.py, docs/contracts.md, docs/decisions.md,
app/, configs, reports/ or model/training/evaluation code. If the contract
blocks you, STOP and report. No new dependencies.

Acceptance:
uv run --offline --no-sync ruff check . &&
uv run --offline --no-sync ruff format --check . &&
uv run --offline --no-sync pytest
plus the real export above.

Deliver: commits on the branch, the report with full head SHA, scope, checks,
the pack contents table (file, rows, SHA-256), the chosen races and runs, and
known gaps.
