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
and snapshot `verify-data` commands. T3 provides the Python interfaces for
fixed splits, exclusions and labels. T5 implements file-based evaluation.
T4 implements `build-dataset`; T12b implements `export-demo`.

```sh
uv run pitstop download
uv run pitstop verify-data
uv run pitstop build-dataset
uv run pitstop train
uv run pitstop predict
uv run pitstop evaluate --help
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
It is optional for archives, validated by the shared `validate_lap_count_evidence`,
and is the input for T4 laps_remaining: see docs/contracts.md for the count known
at T. `events.json.total_laps` is the loader's last non-null
announced value, never an observed lap maximum; it is not approved as an earlier
sample's feature.

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

## Fixed splits and D3 exclusions

`manifests/splits.json` records all Race events in the verified snapshot, including
events excluded by D3. Assignments depend only on the season and round. T4 can
select the requested split by race ID before reading its raw laps.

`pitstop.splits.build_split_manifest(events)` reproduces the tracked manifest from
`events.json`. Pass all raw laps for the selected events and the full manifest to
`pitstop.splits.apply_exclusions(laps, events, split_manifest)`. It returns retained
raw rows with `split` and `special_tyre_rule`, plus a derived audit with per-race
coverage and every D3 reason keyed at race or race/driver scope. Run this on raw
rows, including generated rows, to match the verifier's coverage and thresholds.
Store the derived audit under `data/processed/` or in private notes.

## Next-lap labels

Use `pitstop.labels.build_labels(raw_laps)` on the verified split's unfilled raw
laps. It drops generated rows and joins the exact t+1 and t+2 laps by race and
driver. It returns a table of accepted `KEY`/`y` rows and a separate audit of
all real lap keys with observations, nullable labels and reason codes. Join the
accepted table back to D3-retained rows by `KEY` before constructing samples.
Keep the audit's successor observations out of features.

Optional `confirmed_transits` maps exact (race ID, driver number, entry lap)
keys to nonblank source references. Supply verified records only, with an
input hash and a locator that reproduces the confirmation. The v1 audit uses
the default empty mapping. Its archived control messages use UTC and field
lap numbers, while car laps use elapsed session seconds, so the audit lacks
an exact visit-level confirmation from snapshot payloads. The supplementary
training cache probe supports the D2 decision and stays outside the v1
labelling inputs. Continuous visits remain unresolved by default in every split.

For the T3 audit, verify the snapshot, read training events only for the
diagnostics, and record accepted and omitted counts by reason. Keep keyed
audit outputs private. Test evaluation is still gated by D4.

## T4: build the processed dataset

```sh
uv run --offline --no-sync pitstop build-dataset --config configs/data.yaml --snapshot v1
```

The command verifies the raw snapshot, applies the fixed splits and D3
exclusions, builds D2 labels from the unfilled raw laps, then computes the D5
features for every real (non-generated) lap of the retained pairs; field statistics use all drivers of a kept race. It writes
`samples.parquet`, `tabular.npz`, `sequences.npz` and `preprocess.json` to
`paths.processed` (it refuses to overwrite an existing directory). Statistics are fitted on
train sample rows only; `preprocess.json` records the data and split hashes, the
fit row-id hash, payload hashes and the producing commit.

Feature rules chosen where D5 is silent:

- Every aggregate uses only rows with `Time <= T` (laps, lap-count evidence). A rival's
  pit entry counts once its in-lap row has finished, so a truncated stream gives the
  same row.
- Raw `Stint`, `Compound`, `TyreLife`, `Position` and `TrackStatus` are forward-filled
  for at most 3 laps within a driver; anything still missing raises
  `UnresolvedInputError` naming race, driver, lap and feature. No row is dropped.
- `lap_time_rel` and `stint_deg` are masked by `IsAccurate` only. `lap_time_rel` is
  forward-filled without a lap limit; it is 0 until the first accurate lap of the driver (contract rule 1).
  `stint_deg` is 0 until the stint has an accurate lap.
- Window history may use laps that are not sample rows, as long as the lap is real,
  consecutive and at most `t`; a missing lap ends the window and the rest is padding.

## T5: evaluate one run

Use the declared samples table and the accepted data manifest used by the run.
Run this from the evaluation code's Git checkout. A clean checkout makes the
recorded revision reproducible.

```sh
uv run --offline --no-sync pitstop evaluate \
  --predictions "runs/$run_id/predictions.parquet" \
  --samples data/processed/v1/samples.parquet \
  --data-manifest manifests/data.json \
  --output-dir "runs/$run_id"
```

The four paths are required. `--split val` is the default. Supply `--run-id`
when the predictions file holds several runs. The chosen run must predict
every declared sample in the evaluated split exactly once. Other splits may
appear in either input. `--seed` and `--bootstrap-samples` default to 42 and
1000. Pass `--synthetic` only for generated fixtures whose manifest includes
`"synthetic": true`.

Evaluation freezes `threshold.json` on validation and records the result in
`metrics.json`. Both record the SHA-256 of inputs and the producing revision.
An exact retry preserves both files. A subsequent split can add records to
`metrics.json` while preserving the validation metrics and the threshold bytes.
Use the same declared sample file, manifest and bootstrap settings for both
splits. Conflicting retries fail. Empty validation cannot establish a threshold.
With all-negative validation, thresholds that produce alarms have F1 of zero
and the largest observed score wins. A group with neither a positive nor an
alarm has undefined F1. A group with either one and zero matches has F1 of zero.
Precision requires alarms, and recall and AP require positives.

Status groups use the flags on each sample at lap t. Event matching stays
within each group, race and driver and uses actual lap numbers. Bootstrap
resamples separate copies of complete races from each group. Metric intervals
use the defined replicates and record the number omitted. Metric and interval
names such as `events.1.recall` identify entries in the nested event records.

Test evaluation requires prior explicit human approval after the configs are
frozen. Use `--split test --test-approval "<approval reference>"` with the
test predictions and the existing validation run directory only after obtaining
approval. The command records the reference and requires the frozen threshold.
If a project's evaluated AP or lap F1 exceeds 0.6 in any group, the artifact
records a pending leakage audit and the command exits 2. Review that audit
before reporting results. Exit 0 indicates completion and exit 1 indicates
an input or publication error for a configured evaluation.

Training can use `pitstop.metrics.select_threshold(y, scores)` on validation
scores. For tabular evaluation, `pitstop.evaluate.align_predictions` validates
and aligns predictions to samples, and `pitstop.metrics.group_metrics` evaluates
the resulting rows at a supplied threshold. `pitstop.metrics.paired_bootstrap`
accepts aligned predictions from two runs for the same keys, labels and split,
and their respective frozen thresholds. It reports left-minus-right intervals
and marks an interval containing zero as a tie.

## T6: train the baselines

```sh
uv run --offline --no-sync pitstop train --config configs/constant.yaml
uv run --offline --no-sync pitstop train --config configs/logreg.yaml
uv run --offline --no-sync pitstop train --config configs/lightgbm.yaml
```

Each command fits on `split == "train"` rows only (LightGBM stops early on val AP
and saves the best iteration), writes `runs/<run_id>/` with the artifacts listed
in the contract, and then writes `predictions.parquet` for train and val rows.
The test split is never read for prediction. `pitstop predict --run-dir runs/<run_id>`
reloads the saved model and rewrites the predictions. Then run T5 evaluation on
the validation split for each run (see above).

LightGBM needs the system OpenMP library (`libgomp1` on Debian). If importing
it fails with `libgomp.so.1: cannot open shared object file`, install it.

## T8: MLP and the shared neural loop

```sh
uv run --offline --no-sync pitstop train --config configs/mlp.yaml
```

`train.fit_torch` is the loop for every neural model: AdamW, BCE with logits,
seeded per-epoch shuffling, early stopping on val AP, best-epoch weights. It
writes `model.pt` (CPU `state_dict`) and `history.json` (one entry per epoch plus
`{"best_epoch": n}`). `predict` rebuilds the model from `config.yaml` via
`predict.MODELS`. T10 adds its model class there. Config `input`
(`tabular` or `sequences`) selects the array. Then evaluate on val as above.

`fit_torch` calls `torch.use_deterministic_algorithms(True)` for the whole process; T9 and T10 models must use deterministic CPU ops.

## T9: GRU

```sh
uv run --offline --no-sync pitstop train --config configs/gru.yaml
```

`models/rnn.py` is a two-layer GRU (`cell: lstm` swaps in an LSTM) on the
15-lap windows from `sequences.npz`. The logit comes from the last time step,
which is always a real lap. `bidirectional: true` raises. It trains through
`fit_torch` like the MLP, with gradient clipping at 1.0. Evaluate on val as above.

## T10: TCN

```sh
uv run --offline --no-sync pitstop train --config configs/tcn.yaml
```

`models/tcn.py` is a causal TCN on the 15-lap windows: three residual blocks of
two convolutions (32 channels, kernel 3, dilations 1, 2, 4), padded on the left
only. The logit comes from the last time step. `causal: false` raises. It trains
through `fit_torch` like the GRU. Evaluate on val as above.

## T11: validation experiments

```sh
uv run --offline --no-sync python -m pitstop.experiments tune     # D6 grid, writes configs/final/
uv run --offline --no-sync python -m pitstop.experiments final    # 6 models x seeds 42, 43, 44
uv run --offline --no-sync python -m pitstop.experiments summary  # reports/validation-v1.{json,md}
```

Val split only. `--workers 3 --threads 4` are the defaults (parallel processes,
torch threads each). Run from one clean commit: `summary` refuses runs from several
revisions or a dirty tree. A val AP or F1 above 0.6 raises the D8 leak alarm and stops.

## T7: Streamlit race view

```sh
PITSTOP_DEMO_DIR=demo uv run --group app streamlit run app/streamlit_app.py
```

The app reads a precomputed pack (`predictions.parquet`, `samples.parquet`,
`thresholds.json`) from `demo/`, or from the directory in `PITSTOP_DEMO_DIR`. Without
a pack it shows an info message. The real pack is exported in T12b and committed to
`demo/`; tests use a synthetic pack.

## T12b: demo pack

```sh
uv run --offline --no-sync pitstop export-demo --output-dir demo   # needs runs/, data/processed/v1, data/raw/v1
```

Run it from a clean committed revision (a bare `export-demo` stays a stub, like the other stages, so pass `--output-dir`). It only reads saved files (no training,
prediction or evaluation) and writes `demo/{predictions,samples}.parquet`,
`metrics.json`, `thresholds.json` and `metadata.json`. Rule `seed42_one_run_per_model`
v1: the seed-42 run of each of the six models listed in `reports/test-v1.json`; a run
without test metrics and a recorded `test_approval` is refused. Races: the first three
test races of `samples.parquet` in ascending `race_id`. Metrics and thresholds are
copied unchanged. A second export gives byte-identical files; `metadata.json` records
the last commit that changed code (`src`, `pyproject.toml`, `uv.lock`), so committing
`demo/` does not change it.

## T12: locked test evaluation

Only with explicit human approval, once, from one clean commit, after `reports/test-v1-freeze.json`
is committed. Nothing is retrained or re-thresholded.

```sh
for run in $(uv run --offline --no-sync python -c "import json; print(*json.load(open('runs/final.json')))"); do
  uv run --offline --no-sync pitstop predict --run-dir "runs/$run" --test-approval "<approval>"
done
uv run --offline --no-sync python -m pitstop.experiments test --test-approval "<approval>"
```

`predict` adds test rows only with `--test-approval`. Run it once per run and never
repeat it: a rewrite of `predictions.parquet` no longer matches the test
`predictions_sha256` recorded in `metrics.json`. The `test` step runs
`evaluate --split test` for the 18 runs (frozen val thresholds), stops on the D8 leak
alarm, and writes `reports/test-v1.{json,md}`. Running the `test` step again only
rebuilds the summary from the saved metrics.

## Frozen runs (runs-v1)

The 36 T11 run directories, including the single T12 test evaluation of the 18
final runs, are published as the
[runs-v1 release](https://github.com/lyskawson/f1-pitstop/releases/tag/runs-v1).
From the repository root:

```sh
gh release download runs-v1 --repo lyskawson/f1-pitstop --pattern 'runs-v1.tar.gz*'
sha256sum -c runs-v1.tar.gz.sha256 && tar -xzf runs-v1.tar.gz
```

Archive SHA-256: `122b60c80561a9f32d361b2431c44b854ba16691b44d41e2acbb954bc94986f6`.
Extraction creates `runs/<run_id>/`, `runs/tuning.json` and `runs/final.json`.
The runs are read-only evidence: do not retrain them, and never repeat the test
evaluation. `runs/` stays out of Git.

## T13a: dashboard views

The app now has three tabs on the demo pack: Race view (unchanged), Field heatmap
(drivers x target lap, score colour, red outline = real tyre change, black dot =
score >= frozen threshold) and Model comparison (table and AP chart from the "test"
/ "overall" group of `metrics.json`; nothing is recomputed). Race and model
selectors are shared above the tabs. A missing or inconsistent `metrics.json` only
affects the comparison tab; `metadata.json` supplies event names (race_id fallback).
