# Pipeline contracts

The source of truth is [decisions.md](decisions.md). Executable column names,
dtypes, constants, and validators live in `src/pitstop/contracts.py`. T1 defines
interfaces; the pipeline stages are stubs. Contract changes after T1 require a
dedicated PR. No real race data or model results are included here.

## Keys and types

`KEY = (race_id, driver_number, lap_number)` identifies a sample at the end of lap t.
`race_id` is a string `YYYY_RR_R`, with a four-digit year 1000..9999 and a
zero-padded round 01..99, for example `2022_10_R`. Only Race sessions are allowed.
`make_race_id(year, round_number)` builds it; `validate_race_id` rejects other forms.
`driver_number` is a nonempty string, never an integer. `lap_number` is positive int64.

In the tables, `str` means a pandas string column or an object column containing
only strings. Numeric and boolean dtypes are exact NumPy dtypes, including after
Parquet reads. Nullable extension integers, numeric strings, and float lap numbers
must be normalized by their producer. Required columns may appear in any order;
extra columns are permitted. Duplicate column names are rejected. Array feature
order is fixed. Validators raise `ValueError` with the failed field or constraint.

## ingest → dataset: raw snapshot

```text
data/raw/<snapshot>/
  laps.parquet
  events.json
  manifest.json
  track_status.parquet       # when needed, timestamped raw stream
  race_control.parquet       # when needed, timestamped raw stream
```

`<snapshot>` is an immutable version name recorded in the manifest. Raw laps keep
the following FastF1 names, with `race_id` added. The unique raw key is
`(race_id, DriverNumber, LapNumber)`; downstream producers rename it to `KEY`.

| Column | Dtype | Null allowed | Purpose |
| --- | --- | --- | --- |
| race_id | str | no | Race identifier |
| DriverNumber | str | no | Driver identifier |
| LapNumber | int64 | no | Lap index |
| Time | float64 | yes | Session seconds at lap end, cutoff T |
| LapTime | float64 | yes | Lap duration in seconds |
| PitInTime | float64 | yes | Session seconds of pit entry; in-lap marker |
| PitOutTime | float64 | yes | Session seconds of pit exit; out-lap marker |
| Stint | float64 | yes | Tyre-change label and stops so far |
| Compound | str | yes | Dry one-hot features and whole-race wet exclusion |
| TyreLife | float64 | yes | Tyre age and missingness checks |
| Position | float64 | yes | Lap-end position |
| TrackStatus | str | yes | Status codes, including multi-code strings |
| IsAccurate | bool | no | Time-feature mask only |
| FastF1Generated | bool | no | Remove generated laps before labelling |

`events.json` contains one record per race: `race_id`, `season`, `round_number`,
`session` (`R`), `event_name`, `total_laps`, `total_laps_source`, and
`special_tyre_rule` (bool). The event name and total lap count live here, rather
than being repeated in the raw lap table. `total_laps` is the FastF1-reported
count. T0 must check shortened races and the loader's last-non-null behavior
before this count is used for earlier-lap features. If the count changed, preserve
the timestamped announcements and use only the count known at T; a final count
must not leak into earlier samples. Record findings and provenance.

All raw times are floating-point seconds, not timedelta, datetime, or integer
nanoseconds. Missing float measurements are NaN. A null `PitInTime` means no
recorded pit entry on that lap; it is never filled. Infinity is invalid.
Missing Compound/TrackStatus stay null, never the strings `"nan"` or `"None"`.
`validate_raw_laps` checks these types, nullability, race IDs, and unique raw keys.

`manifest.json` is copied to `manifests/data.json` when the snapshot is accepted.
Its required fields are:

- `schema_version`, `snapshot`, `created_at_utc`, `fastf1_version` (3.8.x),
  and the ingest `git_commit`.
- `sources`: source service/URL, season, round, session, and acquisition timestamp.
- `files`: snapshot-relative path, byte size, and SHA-256 for every payload file.
  The manifest itself is excluded from its own file-hash list.
- `coverage`: all requested races, per-race/per-driver lap counts and lap ranges,
  missing-field counts, and special tyre rule flags.
- `exclusions`: scope (`race`, `race_driver`, or `sample`), identifying keys,
  reason code, and readable reason. Include wet races, red flags, gaps exceeding
  three laps, and Compound/TyreLife missing on more than 10% of a driver's laps.
- `errors`: affected race/source, stage, error type, message, and whether ingestion
  succeeded. Failed downloads must not disappear from coverage.

Verify SHA-256 before dataset construction. Exclude an entire race for any
INTERMEDIATE/WET lap or TrackStatus containing `5`. Keep SC/VSC and pre-DNF laps.
Flag Qatar 2023, Monaco 2025, and Qatar 2025 as special tyre rule races. Build D2
labels by t+1/t+2 self-joins on raw laps after removing FastF1Generated rows and
before any feature filtering; never use `shift()`.

## Fixed splits

`manifests/splits.json` has `schema_version`, `definition` (season and inclusive
round bounds), and `races` (a mapping from every scheduled race ID to its split).
Create it before filtering, including races that will later be excluded:

| Split | Races |
| --- | --- |
| train | 2022–2024 Race sessions |
| val | 2025 rounds 1–12 |
| test | 2025 rounds 13–24 |

Use the same exclusion rules throughout. No random row split, cross-validation,
or train+val refit. Test evaluation requires explicit human approval after configs
are frozen, once before M3 (2026-12-20). The optional 2026 season is outside v1.

## dataset → train

The processed snapshot (initial config path `data/processed/v1/`) contains
`samples.parquet`, `tabular.npz`, `sequences.npz`, and `preprocess.json`.

| samples.parquet column | Dtype | Constraint |
| --- | --- | --- |
| row_id | int64 | Unique, nonnegative, stable within a processed snapshot |
| race_id | str | Part of KEY |
| driver_number | str | Part of KEY |
| lap_number | int64 | Part of KEY, current lap t |
| y | int8 | 0 or 1, next-lap tyre-change label from D2 |
| split | str | Exactly train, val, or test |
| sc, vsc, special_tyre_rule | bool | Current-lap status and race metadata |

No sample field may be null. KEY is unique even across splits.
`validate_samples` checks the table, row IDs, labels, and splits. `row_id` need
not be contiguous and is not the DataFrame index.

The feature vector is exactly this order (`FEATURES`, 15 dimensions):

```text
tyre_age, compound_soft, compound_medium, compound_hard, stops_done,
laps_remaining, mandatory_done, lap_time_rel, stint_deg, position,
gap_ahead, gap_behind, sc, vsc, rival_pits_2
```

Use D5 definitions verbatim. Every field aggregate is bounded by session time
T = Time(t), across all drivers and all streams. Gap ahead uses lap t; gap behind
uses lap t−1. Rival pit entries must be in `(T(t−2), T]`. Gaps are clipped to
0..60 seconds and missing gaps are 60 seconds. IsAccurate masks only lap_time_rel
and stint_deg; it must never select or remove label rows. Forward-fill these time
features from earlier accurate observations within the same race and driver;
stint_deg is 0 at stint start. Apply D3's maximum three-lap feature-gap fill,
never fill labels, never backward-fill. Remaining unresolved required inputs
cause a reported dataset-build error; no silent row removal or arbitrary fill.
Leading missing observations and shortened-race metadata need the T0 audit before
the later feature implementation can claim to build a complete dataset.

| Archive | Required arrays | Dtype and shape |
| --- | --- | --- |
| tabular.npz | row_id; X | int64 [N]; float32 [N, 15] |
| sequences.npz | row_id; X | int64 [N]; float32 [N, 15, 16] |

The archives' row IDs must be unique and exactly equal to samples.row_id in the
same order. N is the number of sample rows. Load with `np.load(..., allow_pickle=False)`.
X contains only finite values. `validate_tabular(arrays, samples)` and
`validate_sequences(arrays, samples)` check required arrays, dtype, shape, and alignment.

`SEQ_CHANNELS = FEATURES + (is_pad,)`. Each window covers t−14..t within one
race/driver and may cross stints. Laps are consecutive; the last step is lap t.
Left-pad short histories with zero feature vectors and is_pad=1; real steps have
is_pad=0. Padding is added after preprocessing. The final step is always real.
The sequence validator checks binary, contiguous left padding and zero padded
features. The later windows/leakage tests must check lap provenance and causality,
which cannot be recovered from X alone.

`preprocess.json` contains `schema_version`, `data_sha256`, `splits_sha256`,
`feature_order` (FEATURES), `sequence_channels` (SEQ_CHANNELS), `window` (15),
`fit_split` (`train`), `fit_row_ids_sha256`, and `scaling`:
`columns` (ordered numeric feature names), `mean`, `scale`, and `ddof` (0).
Fit on training sample feature rows only, excluding padding. Leave one-hot and
boolean channels unscaled; use scale=1 for a constant numeric column. Save
`missing_values` with `feature_forward_fill_limit` (3), `gap_seconds` (60), and
`unresolved` (`error`). Freeze and reuse these values on val/test and inference;
do not learn imputation or scaling from val/test. Record the processed payload
SHA-256 values and producing code revision with dataset provenance.

## train → predict: run artifacts

`run_id` has the structural format
`YYYYMMDDTHHMMSSZ_<model>_s<seed>_<sha8>`: UTC start time, model identifier
(`constant`, `logreg`, `lightgbm`, `mlp`, `gru`, or `tcn`), nonnegative seed,
and eight lowercase hexadecimal characters of the code commit. Never overwrite
an existing run directory, including on timestamp collisions.

Each `runs/<run_id>/` contains:

| File | Content |
| --- | --- |
| config.yaml | Complete resolved experiment config; no inheritance |
| git_commit.txt | Full producing commit SHA; record a dirty tree in environment metadata |
| data_hash.txt | SHA-256 of the exact accepted manifest.json bytes |
| seed.txt | Integer seed used by all random generators |
| environment.json | Python, platform, uv and installed package versions, uv.lock SHA-256 |
| preprocess.json | Exact frozen preprocessing copied from the dataset |
| model.json | Constant baseline's training prevalence |
| model.pkl | Fitted sklearn logistic regression; load trusted project artifacts only |
| model.txt | LightGBM booster with selected best iteration |
| model.pt | Neural CPU state_dict; rebuild architecture from config |
| history.json | Epoch/iteration train loss and val AP, selected step; empty list for noniterative fits |
| predictions.parquet | Predictions from the frozen selected model |
| threshold.json | Frozen threshold selected on validation |
| metrics.json | Evaluation result with provenance, intervals, and group breakdowns |

Only the applicable model file is required. Evaluation adds the last artifacts
after training. No test metrics are produced until authorized. Neural models
return one logit per sample; prediction applies sigmoid. Baselines emit
probabilities. AdamW, CPU, plain BCEWithLogitsLoss with pos_weight=1, and output
bias log(train prevalence / (1 − train prevalence)) are the network defaults.
Keep natural class ratios; no SMOTE, undersampling, or focal loss. Select models
and stop on val AP. Never refit on train+val. Final selected configs run with
seeds 42, 43, 44; at most six tuning configs per model as specified in D6.

## predict → evaluate

| predictions.parquet column | Dtype | Constraint |
| --- | --- | --- |
| race_id, driver_number | str | KEY identifiers |
| lap_number | int64 | Current lap t |
| run_id | str | Run identifier defined above |
| split | str | train, val, or test |
| y | int8 | 0 or 1, copied from the matching sample |
| score | float64 | Finite probability in [0, 1], never a logit |
| target_lap | int64 | Exactly lap_number + 1 |

No nulls. `(KEY, run_id)` is unique; multiple runs may predict the same KEY.
`validate_predictions` checks the schema, race/run ID formats, unique key,
labels, splits, probability range, and target-lap relation. Evaluation also joins
KEY back to the declared dataset to verify labels and splits, with no inner-join
loss of samples. Status breakdowns come from samples, not future lap status.

## evaluate → dashboard

`threshold.json` requires `schema_version`, `run_id`, `threshold` (0..1),
`chosen_on_split` (exactly `val`), `selection_metric` (`f1`), `tie_break`
(`largest`), `validation_f1`, `data_sha256`, and `git_commit`. Maximize F1 on
validation; ties choose the larger threshold. Alarm means score >= threshold.
Freeze once and reuse unchanged on test. Always record validation provenance.

`metrics.json` requires `schema_version`, `run_id`, `data_sha256`, `git_commit`,
`threshold_file_sha256`, `bootstrap`, `splits`, and `leakage_audit`.
`splits` maps evaluated split names to `overall`, `green`, `sc_vsc`, and
`special_tyre_rule` group records. Green means neither sc nor vsc at lap t;
special tyre rule races may overlap the status groups. Every group reports:

- `n_samples`, `n_positives`, `n_races`, `prevalence` and `average_precision`
  (`sklearn.metrics.average_precision_score`), always side by side.
- `threshold`, `precision`, `recall`, and `f1` at the frozen threshold.
- `events`, keyed by `"1"` (main) and `"2"` (auxiliary), each containing
  `recall`, `precision`, `f1`, and `false_alarms_per_race`. A positive label at t*
  is hit by a same-race/driver alarm in [t*−k, t*]; an alarm at t is correct if
  a positive lies in [t, t+k]. Each pit and alarm is counted once in its own
  denominator; late alarms cannot match backward. Divide false alarm count by
  the number of races in the evaluated group.
- `confidence_intervals`: per-metric `lower`/`upper` bounds at 95%, including
  event metrics. Undefined metrics are JSON null with a reason in `undefined`;
  never emit JSON NaN or invent a result for an empty group.

`bootstrap` records `unit: race`, `samples: 1000`, `confidence_level: 0.95`,
`seed`, interval method (`percentile`), and the handling/count of undefined
replicates. Resample races with replacement, including all their rows. Comparisons
use paired resampled races and report difference intervals; an interval containing
zero is a tie. ROC-AUC is optional in an `appendix` field only. If val/test AP or
F1 exceeds 0.6, `leakage_audit` must require an audit before results are reported.
Record audit status and evidence; never manufacture an approval.

The later `demo/` pack contains `predictions.parquet`, `samples.parquet`
(keys/status fields for the dashboard), `metrics.json`, `thresholds.json`
(threshold.json records by run_id), and `metadata.json`. It contains three
deterministically chosen retained test races (ascending race_id, first three);
export only after authorized test evaluation. `metadata.json` records race IDs,
event names, chosen-rule/version, run IDs, code/data hashes, package versions,
and SHA-256 of payloads. Dashboard runtime reads precomputed outputs without
torch or FastF1. Demo files and aggregate result numbers are produced in later tasks.

## Config keys

Every YAML is a complete config; no inheritance or includes. Seeds start at 42.
Paths are repository-relative. `configs/data.yaml` describes acquisition/building;
model configs repeat all paths they need and do not depend on loading data.yaml.

| data.yaml key | Meaning |
| --- | --- |
| seed, seasons, session, window | Seed; 2022–2025; Race only; 15-lap windows |
| paths.raw, processed, cache | Snapshot root, processed output directory, human download cache |
| paths.data_manifest, splits | Versioned manifest and fixed split file destinations |
| splits.train.seasons | Training seasons |
| splits.val/test.season, rounds | Held-out season and inclusive [first, last] round bounds |
| cleaning.max_feature_gap | Maximum causal feature forward-fill gap, 3 laps |
| cleaning.max_missing_compound_or_tyre_life_fraction | Drop race/driver when either field exceeds 0.10 missing |
| cleaning.exclude_compounds, exclude_track_status | Whole-race wet and red-flag exclusions |
| special_tyre_rules | Records of season, event_name, and human-readable rule |

| Model config key | Meaning |
| --- | --- |
| model, seed, input, output_dir | Model ID, random seed, tabular/sequences input, run-directory root |
| data.samples, tabular, sequences, preprocess | Four processed artifact paths |
| parameters | Native baseline parameters; shared seed is passed as random_state where supported |
| parameters.strategy | constant: train_prevalence; no fitting on val/test |
| parameters.C, penalty, solver, max_iter, fit_intercept, class_weight | Logistic regression regularization, optimizer, limit and unweighted fit |
| parameters.objective, num_leaves, learning_rate, n_estimators | LightGBM binary objective and D6 capacity/rate/tree budget |
| parameters.min_child_samples, subsample, subsample_freq, colsample_bytree | LightGBM child count and row/column sampling; frequency 1 enables row sampling |
| parameters.class_weight, n_jobs, verbosity | Unweighted LightGBM, worker count and logging; no scale_pos_weight |
| early_stopping.rounds, split, metric | LightGBM: 100 rounds without val AP improvement; use sklearn AP for monitoring |
| architecture.input_size, output_size, window | 15 tabular or 16 sequence channels, one logit, 15 steps |
| architecture.hidden_sizes, activation | MLP: [64, 32], ReLU |
| architecture.cell, hidden_size, num_layers, bidirectional | GRU/LSTM option, 64 units, two layers, false for v1; bidirectional requires later optional-protocol approval |
| architecture.head | last_step for sequence models |
| architecture.blocks, convolutions_per_block, channels | TCN: three residual blocks, two convolutions each, 32 channels |
| architecture.kernel_size, dilations, causal, residual | TCN: kernel 3, [1, 2, 4], true, true; receptive field 29 >= 15 |
| architecture.dropout, output_bias | MLP/GRU 0.2, TCN 0.1; train_log_odds bias initialization |
| training.optimizer, learning_rate, weight_decay | AdamW; MLP/GRU 0.001, TCN 0.002; weight decay 0.0001 |
| training.loss, pos_weight | bce_with_logits, 1.0 |
| training.batch_size, max_epochs, patience | MLP 512/50/8; GRU and TCN 256/40/8 |
| training.gradient_clip_norm, device | GRU 1.0; null disables clipping; cpu reference |
| selection.split, metric | val, average_precision; maximize it |
| threshold.split, metric, tie_break | val, f1, largest |

The logistic defaults and GRU/TCN weight decay fill settings not fixed in D6;
they are starting configuration choices, not changes to accepted decisions.
PyTorch uses the explicit CPU index as described in the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
CI pins the local uv version and uses
[setup-uv](https://github.com/astral-sh/setup-uv) for installation and caching.
