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

### FastF1 authentication

Only the human downloading data needs their own F1 TV Access (or higher) account
and logs in locally. See the [FastF1 authentication documentation](https://docs.fastf1.dev/versions/dev/api_reference/accounts_auth.html).
The human runs these commands locally (agents never run authentication):

```sh
uv run python -m fastf1 auth f1tv --authenticate
uv run python -m fastf1 auth f1tv --status
uv run python notebooks/t0_fastf1_spike.py download
```

To remove the local login when it is no longer needed:

```sh
uv run python -m fastf1 auth f1tv --clear
```

The download can be interrupted and rerun; successful requests are cached. It
loads Race sessions only, including race-control messages, weather and Jolpica pit
stops, without telemetry. Weather is cached for later exploration only and is not
used as a feature or added to the raw-laps fixtures. Rerun download to add weather
to races cached before this option was enabled.

Rate limits cause a wait and retry. Other schedule failures are retried up to
three times, with waits of 60 and 120 seconds. Per-race successes and errors are
recorded in `data/t0/download_log.json`. The final log includes `missing_seasons`
and `complete`; any missing season, empty schedule or failed race produces a
clear error and a nonzero exit status. Rerunning retries all requested seasons,
including previously failed schedules, while reusing cached data.

After the human confirms completion, run the cache-only analysis:

```sh
uv run python notebooks/t0_fastf1_spike.py analyze
```

This writes evidence to `data/t0/summary.json` and exports two dry races without
red flags to `tests/fixtures/raw/`. It blocks network connections and returns a
nonzero status for missing sources or incomplete coverage. Review penalty
message context before concluding whether a pit-lane pass changed Stint.

Tokens stay in the user's FastF1 data
directory. Never commit or paste tokens or credentials, or put them in CI, configs
or agent prompts. Tests and CI never need them.

Keep `data/cache/` after the download so later runs work offline.

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
