# F1 Pitstop orchestrator handoff

Updated on 2026-11-26 by michalklamecki's orchestrator, continuing the
2026-11-24 handoff. Application checkpoint:
`1c76890244470588e5778fe38ad63579aca4f9d4` (T12b merged, main CI green).
The T11 runs were produced at revision `f919147` (merged in `228f320`); their
single test evaluation ran at `75b39bc` (merged in `19cfcb0`).
Later documentation commits can contain this handoff. Inspect the live main
and open PRs before starting work.

## Start with T13a, the remaining dashboard views

The locked test split has been evaluated once (T12, approval recorded in
`reports/test-v1-freeze.json` and every final run's `metrics.json`). **Never
evaluate it again**, and never change the frozen configs, models or
thresholds.

The demo pack is committed in `demo/` (T12b: three test races, the seed-42
run of each model). The next task is `prompts/T13a-dashboard-views.md`,
relative to this file: field heatmap and model comparison views on that
pack. If T13a is already merged, continue with T13 (deployment) and T14.

All 36 T11 run directories, including the T12 test predictions and metrics,
are published as the
[runs-v1 release](https://github.com/lyskawson/f1-pitstop/releases/tag/runs-v1)
(SHA-256 `122b60c80561a9f32d361b2431c44b854ba16691b44d41e2acbb954bc94986f6`).
On any computer, download and extract it into the repository root (see the
runbook section "Frozen runs (runs-v1)"), next to `data/processed/v1` built
from data-v1. Run `uv sync --locked --extra train --group app` after pulling.

## Purpose and status

Predict whether a driver changes tyres on lap t+1 using information available
at the end of lap t. Compare baseline models, MLP, GRU and a causal TCN, then
present predictions in Streamlit. Neural Networks course; presentation on
2027-01-17, demo before the winter break (docs/plan.md).

The data pipeline, all six models, the validation study, the single test
evaluation, the Streamlit race view and the demo pack are merged. Remaining:
dashboard views and deployment, final documentation and the report update.

| Stage | Main commit | PR | Verification |
| --- | --- | --- | --- |
| T4c lap-count contract | b2f2db0c10ffa22e42ce2cf3fb819815c99ae0ed | [#9](https://github.com/lyskawson/f1-pitstop/pull/9) | 301 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45703896780) |
| T4d feature rules (contract) | f1b36767b51e314852e9a8c66fe08032016b8d03 | [#10](https://github.com/lyskawson/f1-pitstop/pull/10) | 301 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45707319534) |
| T4 features, windows, dataset | c4bde16bdf21d0db1e47863c728462a42c287f6d | [#11](https://github.com/lyskawson/f1-pitstop/pull/11) | 320 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45709235984) |
| T6 trainer and baselines | 9859030f656d877f9fa3e059f5fa22faa6579f95 | [#12](https://github.com/lyskawson/f1-pitstop/pull/12) | 330 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45711757824) |
| T8 MLP, shared torch loop | 2f97102b7e3b132506031748b9d2a0fa0175032f | [#13](https://github.com/lyskawson/f1-pitstop/pull/13) | 335 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45713176270) |
| T5b evaluate git_dirty fix | ac26cf46702f46b3e33aedbda3c4fa9749035c7c | [#14](https://github.com/lyskawson/f1-pitstop/pull/14) | [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45713565148) |
| T9 GRU | ad4ce2ab57d8b1d5958b139bdac996ced140b7ee | [#15](https://github.com/lyskawson/f1-pitstop/pull/15) | 341 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45715245934) |
| T10 causal TCN | 67460cd6d6dbc9579fc101d175ebb2f6dd7f3833 | [#16](https://github.com/lyskawson/f1-pitstop/pull/16) | 348 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45716834011) |
| T11 validation study | 228f3206e1c06712d6637e6f990c78803c2ea5ad | [#17](https://github.com/lyskawson/f1-pitstop/pull/17) | 352 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45778750918) |
| T7 Streamlit race view | 38a17d3a134da87a7facb3f4abec600360966692 | [#19](https://github.com/lyskawson/f1-pitstop/pull/19) | 357 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45805458778) |
| T12 locked test evaluation | 19cfcb01341da7d938cab5484e28415694778f39 | [#21](https://github.com/lyskawson/f1-pitstop/pull/21) | 359 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45825996850) |
| T12b demo pack | 1c76890244470588e5778fe38ad63579aca4f9d4 | [#23](https://github.com/lyskawson/f1-pitstop/pull/23) | 363 tests, [CI](https://github.com/lyskawson/f1-pitstop/actions/runs/45828093141) |

Earlier stages (T1, T1b, T0, T2, T2c, T3, T5, the first handoff #7 and the
report submodule #8) are listed in git history and `reports/accepted-work.md`.
That file also summarises the evidence for every stage above.

Validation result (T11, val split only; numbers from
`reports/validation-v1.md`): MLP, GRU and TCN reach val AP 0.141–0.150 at a
prevalence of 0.028, LightGBM 0.141, logreg 0.087. The networks are tied with
LightGBM in 8 of 9 paired race-bootstrap comparisons. These val numbers are
optimistically biased by selection.

Test result (T12, numbers from `reports/test-v1.md`; 11 races, prevalence
0.024): test AP MLP 0.115, GRU 0.113, LightGBM 0.109, TCN 0.105, logreg
0.095, constant 0.024 (means over seeds 42–44). The networks are tied with
LightGBM in 8 of 9 comparisons, logreg is tied with LightGBM at all three
seeds, and only the constant baseline is clearly worse. No model ranking is
claimed.

## Shared data

Unchanged from the first handoff: the
[data-v1 release](https://github.com/lyskawson/f1-pitstop/releases/tag/data-v1)
holds 92 Race sessions (2022–2025, 101,290 raw lap rows), archive SHA-256
`f30bff5fdaa6c280203483e878ca1e4c9189761e7471b6e8a83d74d93e09c410`.

```sh
uv sync --locked --extra train
mkdir -p data/raw
gh release download data-v1 --repo lyskawson/f1-pitstop --pattern 'v1.tar.gz*' --dir data/raw
(cd data/raw && sha256sum -c v1.tar.gz.sha256) && test ! -e data/raw/v1 && tar -xzf data/raw/v1.tar.gz -C data/raw
uv run --offline --no-sync pitstop verify-data --manifest data/raw/v1/manifest.json
uv run --offline --no-sync pitstop build-dataset --config configs/data.yaml --snapshot v1
```

The processed dataset has train 55,797 samples (1,706 positives, 53 races),
val 10,415 (296, 10 races) and test 12,354 (301, 11 races). T11 runs used
processed-data hash
`36ee960d81516ad4c3a4b514686d13da667f393868d567b39c99557ea376f0f5`.

## Decisions added since the first handoff

Read current AGENTS.md, docs/decisions.md and docs/contracts.md before
editing; the first handoff's D2/D4/leakage notes still hold.

- 2026-11-25 (T4d, approved by michalklamecki): `lap_time_rel = 0` while no
  accurate lap allows a causal value (lap 1 is never IsAccurate in v1), and
  the 11 D5 feature interpretations are contract rules in docs/contracts.md.
- 2026-11-25: T11 compute budget at most 2 h CPU; CPU only, no GPU or CUDA
  dependency. T11 selected configs by val AP at seed 42 (grid lr × hidden,
  D6) and ran seeds 42–44; `configs/final/*.yaml` are the frozen configs.
- Review policy chosen by michalklamecki for T4c–T11: every PR except the
  small T5b fix received an independent reviewer pass before merge, and
  fixes were mapped to findings in the PR body. Future process choices belong to the receiving human.
- The CLI is fixed at seven commands by `tests/test_contracts.py`; T11 runs
  as `python -m pitstop.experiments tune|final|summary` instead.
- 2026-11-26 (michalklamecki): the team approved the single T12 test
  evaluation; it ran once on michalklamecki's computer. Trained runs are
  shared as GitHub release assets (runs-v1, like data-v1), not committed to
  Git. The demo export is a separate task (T12b), and its small pack is
  committed under `demo/` because D11 hosting reads it from the repository.

Open questions for the team: Streamlit Community Cloud deployment (T13, needs
human authorization); the milestone dates in docs/plan.md.

## Workflow and acceptance

Follow AGENTS.md and the first handoff's workflow: one task, one branch
(`codex/` prefix), one PR; contract edits in a dedicated PR; English in code
and docs; offline tests; no data, runs or credentials in Git. The
orchestrator checks the full head SHA, file scope and CI against the report;
a human merges (squash), then local main is updated with `--ff-only`.

```sh
uv run --offline --no-sync ruff check .
uv run --offline --no-sync ruff format --check .
uv run --offline --no-sync pytest
```

Environment notes: CPU is the reference device and training is seeded and
deterministic. Known per-run times at 12 threads: MLP about 7 s, TCN about
80 s, GRU about 115 s. On Debian/WSL LightGBM needs `libgomp1`
(`sudo apt install libgomp1`). On macOS (Apple Silicon) a full `pytest` run
segfaults in `test_train.py::...[lightgbm]` on clean main as well; use
`pytest -k "not lightgbm"` locally and rely on Linux CI. lightgbm is imported before torch (see the
first handoff's macOS note). Executors must not leave non-terminating
background wait loops.

## Continue with T13a

| Task | Outcome and principal acceptance |
| --- | --- |
| T13a | Field heatmap (driver × lap) and model comparison views on the committed `demo/` pack; offline AppTest. Brief ready. |
| T13 | Streamlit Community Cloud deployment of `app/` reading `demo/` (D11), under human authorization; local run as fallback. |
| T14 | Final README/runbook, Colab and offline end-to-end reproduction. Every result number cites `runs/` or `reports/`. |
| T1c | Align repository agent-workflow prose (AGENTS.md etc.) with the practice used. |

## Maintain and transfer the report

The report lives in [f1-pitstop-report](https://github.com/lyskawson/f1-pitstop-report),
also a submodule of this repository (pointer `ee556297d97b80db27220276171d3eb08082c331`).
Report PR [#1](https://github.com/lyskawson/f1-pitstop-report/pull/1)
updated the whole narrative through application `1c76890` (T12b), including
the validation and test results. The team compiles the PDF in Overleaf by
importing the report repository; no local TeX build exists, so the page count
in report-state.json is filled in after an Overleaf compile. For later
updates use `prompts/update-report.md`; integrate teammates' report edits
first and update one shared narrative through the latest accepted main.

Before the next handoff, update this file: the task table, exact
source/PR/CI references, open decisions and the next copyable executor
prompt. Keep portable files free of local account paths, caches and
credentials.
