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

The stage order is below. T2 implements the configured cache-only `download`
and snapshot `verify-data` commands described next. Later stages remain stubs.
Bare commands without inputs retain the original stub response and exit 1.

```sh
uv run pitstop download
uv run pitstop verify-data
uv run pitstop build-dataset
uv run pitstop train
uv run pitstop predict
uv run pitstop evaluate
uv run pitstop export-demo
```

Only a human acquires data from the network. The T2 `download --from-cache`
command packages existing data offline and needs no login. Training/model
selection and threshold selection
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

Read [the T0 findings](fastf1-findings.md) before implementing labels or features.
The audit contradicts the assumption that Stint increases identify tyre changes
without exception. It also finds changing planned lap counts and a narrower
post-SC accuracy mask than the decisions describe. Human decisions and a
dedicated contract change are needed before implementing the affected rules.

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


## T2: build and share an offline raw snapshot

After the human download has populated `data/cache`, run from the repository root:

```sh
uv run --offline --no-sync pitstop download --config configs/data.yaml --from-cache --snapshot v1
uv run --offline --no-sync pitstop verify-data --manifest data/raw/v1/manifest.json
```

This exports Race sessions for 2022–2025. FastF1 cache-only mode, blocked HTTP
transports and blocked authentication prevent acquisition, including on a cache
miss. Suppressed FastF1 cache-miss logs also become explicit ingestion failures.
No telemetry, weather, labels, imputation, features or model evaluation is run.
A missing schedule or race fails the build; diagnostics in a uniquely named
`data/raw/.v1-*-failure.json` record requested/loaded coverage and errors.
The target `v1/` and archive are not published for an incomplete build.
Existing snapshot directories, archives and publication locks are never replaced.

The builder stages and verifies every payload, creates `data/raw/v1.tar.gz`,
extracts it into a temporary directory and verifies it again before publication.
Its final JSON prints the archive SHA-256. Keep that exact archive and hash for
Alek's GitHub Release; only Alek publishes. Build timestamps and archive metadata
mean a later build under another name need not have identical archive bytes.
Neither snapshots, archives, caches nor credentials belong in Git.

`laps.parquet` retains every source lap row, including generated/inaccurate rows,
pit markers and missing measurements. Literal `"nan"` and `"None"` become genuine
nulls only in Compound/TrackStatus. D3 exclusions are manifest metadata, computed
on exported raw rows: wet/red races, interior gaps exceeding three missing laps,
and race/driver pairs with more than 10% missing Compound or TyreLife. Counts,
first/last lap, interior gaps and per-field missingness describe observed drivers;
they do not prove the completeness of an unobserved driver's source data. No
leading/trailing laps or tyre-change labels are inferred. Special rules flag
Qatar 2023, Monaco 2025 and Qatar 2025 without deleting them.

The manifest records the actual FastF1 version and producing commit, dirty-tree
status, requested seasons/races, loaded races, failures, source URLs, payload sizes
and SHA-256 hashes. Acquisition timestamps are explicitly null when unknown:
parsed cache files and the T0 log do not establish when each source was acquired.
The snapshot build timestamp is separate. D10's approved 3.9 pin takes precedence
over the stale 3.8.x manifest prose in the protected `docs/contracts.md`.

`track_status.parquet` retains the timestamped status stream with Time in session
seconds. `race_control.parquet` retains all message fields with Time as UTC ISO
text. `lap_count_evidence.json` archives all parsed LapCount announcements in
source order, including null updates, source URL, parser and session seconds.
It is optional archival evidence, not a new required downstream schema. The
reader contract and handling before the first announcement require dedicated
contract work before T4. `events.json.total_laps` is the loader's last non-null
announced value, never an observed lap maximum; it is not approved as an earlier
sample's feature. D2 remains unresolved; uncertain tyre data remains unchanged.

`verify-data` reads only the unpacked snapshot. It rejects unsafe paths, symlink
payloads, undeclared/missing/changed files, invalid raw schemas, textual nulls,
invalid event/stream data and coverage/exclusions that disagree with the payloads.
An internally consistent incomplete snapshot reports `complete: false` and exits
2; malformed or corrupted data exits 1. Success exits 0. A fresh extracted copy
can be checked independently of the cache, credentials, Git and FastF1 imports:

```sh
tmp_snapshot=$(mktemp -d)
tar -xzf data/raw/v1.tar.gz -C "$tmp_snapshot"
uv run --offline --no-sync pitstop verify-data --manifest "$tmp_snapshot/v1/manifest.json"
```

The manifest stays inside the snapshot. Prepare a byte-identical copy at
`manifests/data.json` for review, and record human acceptance separately; copying
it does not grant acceptance. Verify the manifest **inside the unpacked version
directory**, not the canonical copy on its own. Payload paths are relative to
that directory. Store the archive hash and local acceptance evidence in the T2
report. Remote PR/CI, independent review and human snapshot acceptance are separate
steps and remain pending until actually performed.
