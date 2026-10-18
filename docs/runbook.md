# Draft runbook

## Setup and local checks

Use uv and Python 3.12. Dependency installation needs internet; subsequent tests
use toy data and block internet sockets. No FastF1 data is needed for T1.

```sh
uv sync --locked --extra train
uv run ruff check .
uv run ruff format --check .
uv run --offline pytest
uv run pitstop --help
```

CI uses uv 0.11.7, matching the T1 local version. The train extra uses the PyTorch
CPU index. `grep -ci nvidia uv.lock` must print 0 (grep exits 1 when no matches).
Do not regenerate the lockfile during ordinary setup.

## Data and pipeline order

A human downloads the data once on a local machine, per D12, then shares an
immutable archive plus its SHA-256 manifest through a Release or shared drive.
Agents and tests work offline against that snapshot. Verify hashes before use.
Do not put snapshots, caches, credentials, or runs into Git.

The intended stage order is below. These commands are currently stubs: each
prints `pitstop <command>: not implemented yet` to stderr and exits 1. Future
tasks will add config, snapshot, run, and split arguments.

```sh
uv run pitstop download
uv run pitstop verify-data
uv run pitstop build-dataset
uv run pitstop train
uv run pitstop predict
uv run pitstop evaluate
uv run pitstop export-demo
```

Only a human runs download. Training/model selection and threshold selection
use train/val only. Test evaluation remains locked until explicit human approval
after configurations are frozen. Export the test demo only after that approval.

## Later tasks

- T0: check FastF1 stint behavior on drive-throughs and total_laps on shortened
  races; audit causal handling of missing observations before feature implementation.
- Implement ingestion/manifest verification, raw self-join labels, fixed split
  manifest, causal features, train-only preprocessing, and aligned windows.
- Add small race fixtures, leakage tests, model implementations and shared training,
  prediction, threshold selection, metrics, and a short offline CI training check.
- Add approved test evaluation, bootstrap comparisons, demo pack, dashboard,
  reports, notebooks, and the full README in their assigned tasks.
- After T1, protect main; use one short branch and one reviewed PR per task.
  Keep all artifacts consistent with [contracts.md](contracts.md) and
  [decisions.md](decisions.md).
