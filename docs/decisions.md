# Decisions v1 — F1 pit stop prediction

Status: accepted 2026-10-19. Source: project plan + two independent reviews (ML and architecture), consolidated and checked against the FastF1 3.8.3 source code.
This file is the single source of truth. Any change needs approval from a human team member and an entry in the change log at the bottom.

## Context

- Goal: predict whether a driver will pit for new tyres on lap t+1, using only information available at the end of lap t. Results are shown in a Streamlit dashboard.
- Team: 3 students, 2026-10-19 → 2027-01-17. Roles: A — data (+ MLP), B — training and evaluation (+ GRU), C — dashboard and presentation (+ TCN).
- Data: FastF1 3.8.x, Race sessions 2022–2025. The Kaggle Playground S6E5 dataset (repo m-michalski/f1-pitstop-ml) is synthetic and is not used.

## D1 — Framework

- PyTorch 2.x without Lightning. One explicit, shared training loop (about 100 lines).
- Every network is an `nn.Module` returning one logit per sample. Optimizer: AdamW.
- CPU is the reference device; CUDA and MPS are optional.
- Code must stay short and readable: each student has to explain their model, `forward`, `backward`, `eval()` and `no_grad()`.

## D2 — Label

- Sample = (race_id, driver_number, lap t).
- y = 1 if lap t+1 is an in-lap (`PitInTime` not null) and tyres were changed: lap t+2 exists and `Stint(t+2) > Stint(t+1)`. Otherwise y = 0.
- Build labels with a self-join on (race_id, driver_number, lap_number = t+1 and t+2) on raw laps, before any filtering. Never use `shift()`.
- Drop rows with `FastF1Generated == True` before labelling.
- Drop sample t if lap t+1 does not exist, or if lap t+1 is an in-lap without lap t+2 (retirement in the pit lane).
- Pit-lane passes without a tyre change (drive-through, stop-and-go) give y = 0.
- Horizon: t+1 only.
- Verified in FastF1 source: `PitInTime` marks the in-lap and `PitOutTime` the out-lap.
- Not yet verified (Task T0): whether `Stint` stays unchanged on a drive-through. Fallback if `Stint` is unreliable: y = `PitInTime(t+1)` not null, excluding retirements; record the change below.

## D3 — Data scope

- Seasons 2022–2025, Race sessions only (no sprints).
- 2026 is an optional bonus test only, never used for training or tuning.
- Exclude a whole race if any driver completed at least one lap on INTERMEDIATE or WET.
- Exclude a whole race if any lap has `TrackStatus` containing '5' (red flag). Label masking instead of exclusion is a possible v2 improvement.
- Keep SC/VSC laps. Track status is a feature; results are also reported separately.
- DNF: laps before the retirement stay; only rows with an undefined label are dropped (D2).
- Gaps: make lap numbers continuous per (race, driver). Forward-fill feature gaps of at most 3 laps (features only, never labels). Drop the (race, driver) pair for larger gaps or when Compound/TyreLife is missing on more than 10% of laps.
- Flag races with special tyre rules (`special_tyre_rule = True`) and report them separately: Qatar 2023 (max 18 laps per set), Monaco 2025 (at least three sets of tyres), Qatar 2025 (max 25 laps per set).
- Every exclusion is listed with its reason in the data manifest.

## D4 — Split

- Train: 2022–2024. Validation: 2025 rounds 1–12. Test: 2025 rounds 13–24.
- The split is defined by round number in `manifests/splits.json`, fixed before filtering. Exclusions are identical in all splits.
- No cross-validation and no refit on train + validation.
- The test split is locked. It is evaluated once, after configs are frozen, before M3 (2026-12-20), and only with explicit human approval.

## D5 — Features and sequences

Only information available at T — the `Time` (end) of lap t of driver d — may be used.

Feature vector at t (15 dimensions):

1. `tyre_age` = TyreLife
2. `compound_soft`, `compound_medium`, `compound_hard` (one-hot, 3 dimensions)
3. `stops_done` = Stint − 1
4. `laps_remaining` = total_laps − t
5. `mandatory_done` = at least 2 different dry compounds used so far
6. `lap_time_rel` = LapTime / median LapTime of the field over IsAccurate laps with Time ≤ T, minus 1
7. `stint_deg` = mean of the last 3 IsAccurate laps in the current stint minus the best IsAccurate lap of this stint so far (0 at stint start)
8. `position`
9. `gap_ahead` = T − Time of the car that crossed the line just before, on lap t (clip 0–60 s; missing = 60)
10. `gap_behind` = the same for the car behind, using lap t−1 (the car behind finishes lap t after T)
11. `sc` = '4' in TrackStatus of lap t
12. `vsc` = '6' or '7' in TrackStatus of lap t
13. `rival_pits_2` = number of other drivers with PitInTime in (T(t−2), T]

Rules:

- On laps that are not IsAccurate, `lap_time_rel` and `stint_deg` are forward-filled within (race, driver).
- Standardization statistics are fitted on train only and saved in `preprocess.json`.
- Tabular models (logistic regression, LightGBM, MLP) get the vector at t.
- Sequence models (GRU, TCN) get a window of W = 15 laps, t−14..t, of the same (race, driver), aligned to t. A window never crosses race or driver boundaries; it may cross stint boundaries. Left zero-padding plus an extra `is_pad` channel → tensor `[N, 15, 16]` float32.
- `total_laps`: FastF1 documents it as the originally scheduled lap count, but the loader keeps the last non-null value of the stream. Check it on a shortened race in T0.
- Not in v1 (candidates for v2): weather, circuit identity, high-frequency telemetry.

## D6 — Models

- Baselines: constant score (train prevalence), logistic regression, LightGBM.
- Networks: MLP (owner A), GRU with a `cell: gru | lstm` flag (owner B), causal TCN (owner C).
- Bi-LSTM exists only as a `bidirectional` flag, used in the optional replication of the paper's protocol.
- Starting configs:
  - LightGBM: num_leaves 31, learning_rate 0.05, up to 2000 trees, early stopping after 100 rounds without val AP improvement, min_child_samples 50, subsample 0.8, colsample_bytree 0.8.
  - MLP: 15 → 64 → 32 → 1, ReLU, dropout 0.2; AdamW lr 1e-3, weight_decay 1e-4; batch 512; up to 50 epochs, patience 8.
  - GRU: 2 layers × 64, dropout 0.2, head on the last step; AdamW lr 1e-3; batch 256; up to 40 epochs, patience 8; gradient clipping 1.0.
  - TCN: 3 residual blocks × 2 causal convolutions, 32 channels, kernel 3, dilations 1/2/4 (receptive field 29 ≥ W), dropout 0.1; AdamW lr 2e-3; batch 256; up to 40 epochs, patience 8.
- Tuning: at most 6 configs per model (lr {5e-4, 1e-3, 2e-3} × hidden {32, 64}). The final config runs with seeds 42, 43 and 44.
- Success criterion: a reproducible pipeline and an honest comparison. Networks beating LightGBM is not required.

## D7 — Class imbalance (~3% positives)

- Plain `BCEWithLogitsLoss` (pos_weight = 1). Output bias initialized to log(π / (1 − π)), π = train prevalence (about −3.5).
- LightGBM without `scale_pos_weight`.
- One optional ablation: pos_weight = sqrt(N_neg / N_pos).
- No SMOTE, no undersampling, no focal loss. Validation and test keep the natural class ratio.
- Reason: the primary metric is ranking-based, and an unweighted loss keeps the dashboard's probabilities interpretable.

## D8 — Metrics and threshold

- Primary metric: Average Precision (`sklearn.metrics.average_precision_score`), always reported next to the prevalence (the AP of a no-skill model).
- Model selection and early stopping use validation AP.
- Threshold τ = argmax F1 on validation (ties → larger τ), frozen in `threshold.json` and reused on test. Report precision, recall and F1 at τ.
- Event metric (k = 1 main, k = 2 auxiliary):
  - A pit with label row t* is hit if there is an alarm (ŷ = 1) for the same (race, driver) in [t* − k, t*].
  - An alarm at t is correct if there is a pit with t* in [t, t + k].
  - Report event recall, event precision, event F1 and false alarms per race.
  - The window is asymmetric on purpose: an alarm after t* comes when the car is already in the pit lane.
- Report results separately for green flag and SC/VSC (at lap t). ROC-AUC only in an appendix.
- Uncertainty: 95% CI from a race-level bootstrap (B = 1000). Model comparisons use a paired bootstrap (the same resampled races for both models). Differences inside the CI count as a tie.
- Leak alarm: if validation or test AP or F1 exceeds 0.6, audit for leakage before reporting anything.

## D9 — Repository architecture

```
f1-pitstop/
├── pyproject.toml, uv.lock, .python-version, README.md, AGENTS.md
├── configs/            # complete experiment YAMLs, no inheritance
├── src/pitstop/
│   ├── cli.py          # single CLI: pitstop <command>
│   ├── contracts.py    # schemas, column names, dtypes, constants
│   ├── ingest.py       # FastF1 download/import → raw snapshot + manifest
│   ├── labels.py
│   ├── splits.py
│   ├── features.py
│   ├── windows.py
│   ├── dataset.py      # samples.parquet, sequences.npz, preprocess.json
│   ├── models/         # baselines.py, mlp.py, rnn.py, tcn.py
│   ├── train.py
│   ├── predict.py
│   ├── metrics.py
│   └── evaluate.py
├── app/                # Streamlit dashboard + its own requirements.txt
├── tests/              # unit, contract and leakage tests; tests/fixtures/ (2 small races)
├── notebooks/          # EDA and Colab launcher only, no pipeline logic
├── docs/               # decisions.md, plan.md, contracts.md, runbook.md, fastf1-findings.md
├── manifests/          # data manifest (SHA-256, coverage, exclusions), splits.json
├── demo/               # small versioned demo pack (3 test races)
├── reports/            # selected tables and figures, results.csv
├── data/               # gitignored: raw/, processed/, cache/
├── runs/               # gitignored: one folder per run
├── .orchestrator/      # gitignored: orchestrator state (state.md)
├── _input/             # gitignored: input files for the orchestrator
└── .github/workflows/ci.yml
```

- CLI commands: `download`, `verify-data`, `build-dataset`, `train`, `predict`, `evaluate`, `export-demo`.
- Every `runs/<run_id>/` holds: config, git commit, data hash, seed, environment, `preprocess.json`, model weights, training history, predictions and metrics.

### Interfaces between stages

Key K = (race_id: str `YYYY_RR_R`, driver_number: str, lap_number: int). Times are float seconds; null handling is documented in `docs/contracts.md`.

| Producer → consumer | Contract |
| --- | --- |
| ingest → dataset | `data/raw/<snapshot>/laps.parquet` (+ track status and race control if needed); manifest: FastF1 version, sources, SHA-256, coverage, exclusions, errors |
| dataset → train | `samples.parquet` (row_id, K, y, split, sc, vsc, special_tyre_rule); `tabular.npz` (row_id, X float32 [N, 15]); `sequences.npz` (row_id, X float32 [N, 15, 16]); `preprocess.json` (feature order, scaling fitted on train) |
| train → predict | `model.pt` (state_dict) or `model.txt` (LightGBM) + config + `preprocess.json` |
| predict → evaluate | `predictions.parquet`: K, run_id, split, y, score, target_lap = t+1; (K, run_id) unique |
| evaluate → dashboard | `metrics.json`, frozen `threshold.json`, predictions and version metadata; demo pack in `demo/` |

## D10 — Environment

- Python 3.12, uv, `pyproject.toml` and a committed `uv.lock`. Install: `uv sync --locked` (extra `train` for torch and lightgbm).
- `fastf1>=3.9.0a0,<3.10` (pre-release; the first version that supports the archive's F1 TV authentication); PyTorch CPU build as reference.
- ruff (lint and format) and pytest.
- GitHub Actions on every PR: `ruff check`, `ruff format --check`, pytest on fixtures and a short training run on fixtures. No network in CI.
- Experiment tracking: JSON per run plus `reports/results.csv`. No MLflow, W&B or DVC.
- Colab: a notebook installs uv, creates a Python 3.12 environment, unpacks the data snapshot and calls the same CLI.

## D11 — Dashboard

- Streamlit reads precomputed predictions only (no torch and no FastF1 at runtime).
- Views: race (selectors for race, driver and model; probability per lap vs. real pits; SC/VSC bands; threshold), field heatmap (driver × lap), model comparison.
- `demo/` holds a small pack with 3 deterministically chosen test races.
- Hosting: Streamlit Community Cloud with `app/requirements.txt` exported from the lock; local run as a fallback.
- Test: Streamlit `AppTest` switches race and driver without exceptions, with the network blocked.

## D12 — Working with AI agents

- Contracts and acceptance tests come first (`docs/contracts.md`, `tests/test_contracts.py`); parallel work starts after that.
- One task = one module or contract, with an explicit list of allowed files, inputs, outputs and an acceptance command.
- Tasks run one at a time. The orchestrator (Claude Code) writes a prompt per task; a human pastes it into Codex, opened in the same project folder.
- One short branch per task; PR to a protected `main`; squash merge; green CI and review by a human team member. The only exception is T1, which creates the repository with an initial commit on `main`.
- The reviewer re-runs the acceptance command. Agent reports are not evidence.
- FastF1 data is downloaded by a human on a local machine (`pitstop download`) and shared as a versioned archive (GitHub Release or shared drive). The manifest with SHA-256 is committed.
- Downloading FastF1 data needs the downloader's own F1 TV Access (or higher) account; the downloader logs in locally. The token stays in the user's FastF1 data directory and never goes into the repository, CI, configs or agent prompts; tests and CI never need it.
- The test split stays locked until a human approves its evaluation (D4).

## Leakage rules (enforced by tests)

1. **Label shift.** A fixture where a pit at the end of lap 20 gives the only positive at t = 19. Fewer than 1% of positives may have an in-lap or out-lap at t.
2. **Future aggregates.** No `groupby().transform` over a race or stint, no `rolling(center=True)`, no `pick_quicklaps()` on a whole session (its threshold is 107% of the best lap in the given set), no `session.results` fields except GridPosition, scaler fitted on train only. Truncation test: for about 200 random (race, driver, t), recompute features from data truncated to Time ≤ T (all streams, all drivers); they must equal the full-run features (atol 1e-6).
3. **Field features by time, not lap number.** Rival pits, gaps and any `merge_asof` use session time, never `direction='nearest'`. Unit test: a rival with PitInTime > T does not change the row's features.
4. **Windows.** `lap[-1] == t`, one (race, driver) per window, consecutive laps. Changing data after t does not change the inputs or the predictions for t.
5. **No row selection by future-dependent filters.** Never filter rows with `IsAccurate`, `pick_accurate()`, `pick_wo_box()` or `dropna(LapTime)`. In FastF1 3.8.3, IsAccurate is False for in-laps, out-laps, FastF1-generated laps, laps with any status other than green/yellow (so SC, VSC and red flag) and the first lap after an SC lap. IsAccurate may only mask time-based features. Test: positives per race equal tyre changes in the raw laps (and match Jolpica pit stops within ±1 lap); the positive rate before and after cleaning differs by less than 0.1 percentage point.

## Optional — only after M3 and with human approval

- Replication of the Sasikumar et al. (2025) protocol (random split of overlapping windows + SMOTE before the split) on the best model, compared with the causal protocol.
- 2026 season as an out-of-distribution test.
- Race replay on a track map.

## Change log

- 2026-10-19: v1 accepted.
- 2026-10-19: D10 — FastF1 pin changed from >=3.8.3,<3.9 to >=3.9.0a0,<3.10 because the F1 live-timing archive requires F1 TV authentication since about 2026-10-15 (theOehrly/Fast-F1#978); 3.9.0a0 is the first version that supports it. D12 — the data downloader needs an F1 TV Access account. Statements verified against FastF1 3.8.3 (status line, D2, Leakage rule 5) are re-checked on 3.9 in T0. Approved by a human team member.
