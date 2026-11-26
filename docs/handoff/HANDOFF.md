# F1 Pitstop orchestrator handoff

Updated on 2026-11-26 by michalklamecki's orchestrator, continuing the
2026-11-24 handoff. Application checkpoint:
`38a17d3a134da87a7facb3f4abec600360966692` (T7 merged, main CI green).
The T11 runs were produced at revision `f919147` (merged in `228f320`).
Later documentation commits can contain this handoff. Inspect the live main
and open PRs before starting work.

## Start with T12, after the team approves the test evaluation

The next task is `prompts/T12-locked-test-evaluation.md`, relative to this
file. It evaluates the locked test split once. D4 allows that only after
configs are frozen and a human gives explicit approval, before M3
(2026-12-20). **This handoff does not grant that approval.** The three
teammates have equal authority, so agree on it together, then record the
approval text and who gave it; the executor stores it in `metrics.json` via
`pitstop evaluate --test-approval`.

The T11 run directories that T12 freezes are not in Git (`runs/` is ignored).
They exist only on michalklamecki's computer. T12 is therefore simplest there.
On another computer, first agree whether to retrain the final runs from
`configs/final/` (CPU is deterministic per machine, but bit-identity across
machines is not verified) and re-derive their val thresholds before any test
access; record that choice in the T12 report.

T7 was done on a second computer without `runs/` or `data/`. Continue T12 on
the computer that holds the T11 runs: `git pull --ff-only` on main first.
Check `runs/` for the 18 final run directories before any test access.
`runs/` and `data/` are not in Git. T7 changed only `app/`, tests, the `app`
dependency group and CI, so the frozen T11 runs are unaffected. Run
`uv sync --locked --extra train --group app` once after pulling.

If approval is not ready, T13a can run first. It adds the D11 field heatmap
and model comparison views to `app/streamlit_app.py` on synthetic fixture
packs. It needs no test data. Write its brief from the task table below,
reusing the T7 brief's structure (in the PR #19 description).

## Purpose and status

Predict whether a driver changes tyres on lap t+1 using information available
at the end of lap t. Compare baseline models, MLP, GRU and a causal TCN, then
present predictions in Streamlit. Neural Networks course; presentation on
2027-01-17, demo before the winter break (docs/plan.md).

The data pipeline, all six models and the validation study are merged.
Remaining: the locked test evaluation, demo export, dashboard, final
documentation and the report update.

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

Earlier stages (T1, T1b, T0, T2, T2c, T3, T5, the first handoff #7 and the
report submodule #8) are listed in git history and `reports/accepted-work.md`.
That file also summarises the evidence for every stage above.

Validation result (T11, val split only; numbers from
`reports/validation-v1.md`): MLP, GRU and TCN reach val AP 0.141–0.150 at a
prevalence of 0.028, LightGBM 0.141, logreg 0.087. The networks are tied with
LightGBM in 8 of 9 paired race-bootstrap comparisons. These val numbers are
optimistically biased by selection; only T12 gives unbiased numbers.

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

Open questions for the team: the test approval and where T12 runs (above);
whether the demo export is part of T12 or a separate PR (the T12 brief keeps
it separate as T12b); the milestone dates in docs/plan.md.

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

## Continue after T12

| Task | Outcome and principal acceptance |
| --- | --- |
| T12 | Single locked test evaluation of the 18 frozen T11 runs with frozen val thresholds, human approval recorded; `reports/test-v1.{json,md}`. Brief ready. |
| T12b | Demo pack per contracts "evaluate → dashboard": first three retained test races, metadata with hashes. Only after T12. |
| T13a | Field heatmap (driver × lap) and model comparison views on fixture packs; offline AppTest. Independent of T12. |
| T13 | Dashboard on the exported demo pack, offline AppTest, deployment under human authorization. |
| T14 | Final README/runbook, Colab and offline end-to-end reproduction. Every result number cites `runs/` or `reports/`. |
| T1c | Align repository agent-workflow prose (AGENTS.md etc.) with the practice used. |

## Maintain and transfer the report

The report lives in [f1-pitstop-report](https://github.com/lyskawson/f1-pitstop-report),
also a submodule of this repository. Its main is still
`d1a908478ee49abb5cb8e8bf62787cb1516ce38c` and report-state.json still
documents application T2 at `5ba376f`, so T2c through T11 are not yet written
up. Use `prompts/update-report.md`; integrate teammates' report edits first
and update one shared narrative through the latest accepted main.

Before the next handoff, update this file: the task table, exact
source/PR/CI references, open decisions and the next copyable executor
prompt. Keep portable files free of local account paths, caches and
credentials.
