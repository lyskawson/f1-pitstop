# Accepted-work evidence transferred on 2026-11-24

This is a compact transcription of the completed authors' T3/T5 reports and
the orchestrator's remote checks. Detailed historical local audit files stay
in the original notes workspace. The PRs, source and CI are public.

## T3 labels, splits and exclusions

Author commit `e005c56537addd1df7dfba396b898f4c82926d03`, merged as
`515e72edff169a05abc152d340c54a3f1d3a83e2` in
[PR #5](https://github.com/lyskawson/f1-pitstop/pull/5).
The author reported passing Ruff and 254 offline tests. The orchestrator
verified exact PR head, seven-file scope, green checks/GitGuardian and
[successful main CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45383833653).
An independent review was waived by the human for 2026-11-24.

The author verified the raw archive as 92 races and 101,290 lap rows.
The fixed split covers 68 training, 12 validation and 12 test races before
exclusions. Training D3 retained 53 races and 57,032 raw rows.
The saved training-label audit reported 1,706 positives, 54,091 negatives,
1,167 omitted real samples and 68 removed generated rows after D3.

Omissions comprised 1,025 absent exact t+1 rows, 61 absent exact t+2 exit
rows, 78 continuous tyre histories without confirmations, two missing or
unrecognised tyre records and one missing/invalid exit marker. Positive
evidence comprised 1,337 compound changes and 369 same-compound age
discontinuities. The author reported no validation/test outcome use.
All v1 runtime confirmation mappings are empty.

The source was the original author report `T3.md` and its saved clean-commit
training-label-summary.json. The latter names the exact source commit and
input hashes. This summary preserves the reported aggregates for continuation.

## T5 metrics and evaluation

Author commit `383e3e82f6609a2ddcbf21e8eaba1ce22b9adbca`, merged as
`4927786152af1b6343f98720754d266e9b3a4433` in
[PR #6](https://github.com/lyskawson/f1-pitstop/pull/6).
The author reported passing Ruff, 278 offline tests and a successful public
CLI smoke run with 1,000 race bootstrap draws on synthetic inputs.
The orchestrator verified the exact PR head, six-file scope, green
checks/GitGuardian and
[successful main CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45399220816).
The same dated review waiver applied.

Tests covered hand-computed asymmetric event metrics, threshold ties and
endpoints, undefined groups, repeated bootstrap races, paired differences,
artifact provenance and frozen thresholds. A synthetic test request without
authorization was rejected. Real model training, validation experiments and
locked-test evaluation were outside this work.

The source was the original author report `T5.md` and its recorded smoke/check
outputs. Use the tracked tests to inspect the numerical cases. The synthetic
scores are software-verification evidence.

# Accepted work added on 2026-11-26

Transcribed from the author reports of T4c–T11 and the orchestrator's checks.
For every stage the orchestrator verified the exact PR head, file scope,
green checks/GitGuardian and main CI (links in HANDOFF.md). Each PR except
T5b had an independent reviewer pass, and fixes were mapped to findings in
the PR body. Real-data numbers below come from local `runs/` directories
(not in Git) or from committed `reports/`; val only, the test split is unused.

## T4c and T4d contracts

T4c (PR #9, head dfa8227) documents `lap_count_evidence.json` and adds the
shared validator and the causal `known_total_laps` lookup; 301 tests,
`verify-data` complete on v1 with the archive hash unchanged. T4d (PR #10,
head d37290a, contract files only) records the human decision of 2026-11-25:
`lap_time_rel = 0` until a causal accurate value exists (2,389 cells in the
first build; lap 1 is never IsAccurate in v1) and the 11 D5 interpretations
as contract rules.

## T4 features, windows and dataset

PR #11, head d4faf93; 320 tests. The real build from a clean commit passes
all schema validators with finite X: train 55,797 samples (1,706 positives,
53 races), val 10,415 (296, 10), test 12,354 (301, 11). Leakage tests cover
truncation at T across streams, rival data after T, window provenance,
train-only scaling and row alignment. The review found a rival PitInTime
leak; it was fixed before merge.

## T6, T8, T9, T10 models

T6 (PR #12, head e79807e): shared trainer, predict and constant/logreg/
LightGBM baselines; 330 tests; predict never scores test rows. T8 (PR #13,
head 9b4d626): MLP and the shared `fit_torch` loop (AdamW, val-AP early
stopping, best-epoch weights, deterministic CPU); 335 tests. T5b (PR #14)
made evaluate's `git_dirty` ignore untracked files. T9 (PR #15, head
508e92a): GRU, 341 tests. T10 (PR #16, head 7b7d908): causal TCN, receptive
field 29 ≥ 15, per-step causality tests; 348 tests; the reviewer re-checked
causality for every step. Each task trained one seed-42 default run and
evaluated it on val only; no leak alarm (all val AP and F1 far below 0.6).

## T11 validation study

PR #17, head c6e5409; 352 tests. All 36 runs (18 grid + 18 final) come from
one clean revision f919147 within the 2 h CPU budget. Selected configs are in
`configs/final/`; results, caveats and the paired bootstrap vs LightGBM are in
`reports/validation-v1.{json,md}`. The orchestrator recomputed val AP from
every referenced run's predictions.parquet and found no test rows.

## T7 Streamlit race view

PR #19, head ac9f4d0, merged as 38a17d3; 357 tests in CI. `app/streamlit_app.py`
reads a precomputed pack from `demo/` or `PITSTOP_DEMO_DIR` and shows the D11 race
view: race, driver and model selectors, score per target lap, real tyre changes,
SC/VSC bands and the frozen threshold. It imports no torch, lightgbm, fastf1 or
pitstop. Streamlit is in the `app` dependency group (locked 1.67.0), and
`app/requirements.txt` is exported from the lock. The offline AppTest uses
synthetic packs only. No real data, runs or test rows were used. An independent
reviewer found no blockers, and four fixes are mapped in the PR body.

## T12 locked test evaluation

PR #21, head e9889c4, merged as 19cfcb0; 359 tests. The approval given by
michalklamecki on behalf of the team on 2026-11-26 is recorded in
`reports/test-v1-freeze.json` and every final run's `metrics.json`. The freeze
file (hashes of config, model, threshold and metrics for the 18 final runs) was
committed before any test row was scored. The test evaluation ran once from
clean revision 75b39bc. Afterwards the orchestrator confirmed that the hashes
of config, model and threshold were unchanged, that the test AP recomputed
from predictions matched `metrics.json`, and that the val entries were
unchanged. No leak alarm was triggered. An independent reviewer approved
after recomputing all 15 paired-bootstrap rows. Results are in
`reports/test-v1.{json,md}`; the runs are in the runs-v1 release
(SHA-256 122b60c8…86f6).

## T12b demo pack

PR #23, head 85aad68, merged as 1c76890; 363 tests in CI. `pitstop
export-demo --output-dir demo` copies, without recomputation, the seed-42
run of each model for the first three retained test races (2025_14_R,
2025_15_R, 2025_16_R) into the committed `demo/` pack (329 kB). The pack
holds predictions, samples, metrics, thresholds and metadata with payload
hashes. The orchestrator confirmed that the scores and thresholds equal the
runs, that the hashes match, and that the pack contains only test rows of
those races. An independent reviewer re-exported the pack byte-identically
and approved.

## Report update

The report repository's PR #1 (merged as ee55629) rewrote the Polish report
through application 1c76890. The orchestrator cross-checked the chapter
numbers against `reports/` and the handoff files and corrected two values
before publishing. The PDF is compiled in Overleaf.
