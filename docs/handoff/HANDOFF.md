# F1 Pitstop orchestrator handoff

Prepared on 2026-11-24 for a teammate continuing with separate orchestrator
and executor chats. The application code checkpoint is
`4927786152af1b6343f98720754d266e9b3a4433`. Later documentation commits
can contain this handoff. Inspect the live main and PRs before starting work.

## Start with T4c

The next task is `prompts/T4c-lap-count-contract.md`, relative to this file.
It has been prepared and has not been implemented at this checkpoint.
Give the human a copyable executor launch message in your first response.

Resolve the actual application checkout and choose a sibling notes directory
with writable `prompts/` and `reports/`. The executor needs both locations.
Suggest high reasoning effort for T4c. The established model was gpt-6.1-sol,
and the receiving human may choose an available model.

The orchestrator owns priorities, briefs, dependencies, evidence checks and
handoffs. Executors implement and test in separate chats launched by the human.
Keep the result of a task in one branch and PR. Continue ordinary fixes in the
same author chat. Give the human one clear next action after each result.

## Purpose and completed work

Predict whether a driver changes tyres on lap t+1 using information available
at the end of lap t. Compare baseline models, MLP, GRU and a causal TCN, then
present the predictions in Streamlit. A reproducible comparison is the goal.
The project is for a Neural Networks course, with a planned presentation on
2027-01-17. The original milestone dates in docs/plan.md need reassessment.

Merged work includes the repository skeleton, schemas and offline CI (T1),
the FastF1 dependency update (T1b), the source-data audit and fixtures (T0),
the offline raw snapshot/verifier (T2), the legacy acquisition-time contract
(T2c), next-lap labels and fixed splits/exclusions (T3), and metrics/evaluation
on synthetic inputs (T5).

Recent merges and evidence:

| Stage | Main commit | PR | Verification |
| --- | --- | --- | --- |
| T2c | 3de4c38022c36cb81e1076c79ecce87fad296bb9 | [#4](https://github.com/lyskawson/f1-pitstop/pull/4) | Source a79bccd, green CI, 224 local tests reported |
| T3 | 515e72edff169a05abc152d340c54a3f1d3a83e2 | [#5](https://github.com/lyskawson/f1-pitstop/pull/5) | Source e005c56, 254 tests and training audit reported |
| T5 | 4927786152af1b6343f98720754d266e9b3a4433 | [#6](https://github.com/lyskawson/f1-pitstop/pull/6) | Source 383e3e8, 278 tests and synthetic CLI smoke reported |

Main CI succeeded after [T3](https://github.com/lyskawson/f1-pitstop/actions/runs/45383833653)
and [T5](https://github.com/lyskawson/f1-pitstop/actions/runs/45399220816).
The portable `reports/accepted-work.md` records the transferred author results.
Historical private cache probes and detailed audit files are outside this
package. T4c uses the public source and the shared archive.

Features, processed training inputs, model training and dashboard remain
unfinished. T5's perfect-score smoke cases are synthetic software checks.
There are no real predictive model results and the locked test is unevaluated.

## Shared data

Download the existing versioned archive from the public
[data-v1 release](https://github.com/lyskawson/f1-pitstop/releases/tag/data-v1).
It contains 92 Race sessions from 2022–2025 and 101,290 raw lap rows.
The archive SHA-256 is
`f30bff5fdaa6c280203483e878ca1e4c9189761e7471b6e8a83d74d93e09c410`.

From the application checkout, if the files are absent:

```sh
uv sync --locked --extra train
mkdir -p data/raw
gh release download data-v1 --repo lyskawson/f1-pitstop --pattern 'v1.tar.gz*' --dir data/raw
(cd data/raw && shasum -a 256 -c v1.tar.gz.sha256) && test ! -e data/raw/v1 && tar -xzf data/raw/v1.tar.gz -C data/raw
uv run --offline --no-sync pitstop verify-data --manifest data/raw/v1/manifest.json
```

On Linux, `sha256sum -c` can replace `shasum -a 256 -c`. A manual download
of both release assets also works. Inspect existing files before downloading
or extracting. Keep the archive immutable. Verify the inside-snapshot manifest,
whose payload paths are relative to its directory. Its byte-identical tracked
copy is `manifests/data.json`.

The archive works without the original FastF1 cache, credentials or a new
source acquisition. Unknown acquisition dates for the historical cache remain
explicitly null with explanations under the approved T2c contract. Snapshot
creation time is separate. Data acquisition and predictive test evaluation
retain their own authorization requirements.

## Decisions to preserve

Read current AGENTS.md, docs/decisions.md and docs/contracts.md before editing.
All three human teammates have equal authority to propose changes, approve
decisions and reassign the A/B/C work areas. Technical ownership labels do
not grant one student exclusive control of the plan. Record agreed changes
where they belong. Decision changes need human approval and a change-log entry.
Respect the actual repository permissions of the receiving human.

D2 was approved on 2026-11-24 and merged in T3. Labels use exact t+1/t+2 joins
after generated rows are removed. A pit entry needs the exact exit lap,
usable tyre measurements, a one-step Stint increase and a compound or age
discontinuity for a positive. Used sets can increase age. A continuous tyre
history needs documented exact no-change confirmation for a negative.
Unresolved cases are omitted with reasons. In the current v1 workflow,
confirmation mappings are empty in every split. Private training cache probes
never become implicit label inputs. Follow the full D2 implementation.

The split is fixed before exclusions. Train is 2022–2024, validation is
2025 rounds 1–12 and test is rounds 13–24. Preprocessing fits on train only.
The test split is evaluated once after configurations are frozen and a human
explicitly approves it. This handoff does not grant that approval.

For T4c/T4, the existing contract requires count announcements known by
sample time T. Missing required feature inputs produce a contextual build
error. An earlier suggestion to omit pre-announcement samples was deferred.
Implement the existing error behaviour and causal lookup. The available
`lap_count_evidence.json` payload needs its reader contract and shared
validation in the dedicated T4c PR. Keep archive bytes intact.

All feature aggregation is bounded by Time <= T, across drivers and streams.
IsAccurate masks time features and never selects label rows. Do not fill labels
or use future aggregates. Preserve D3 exclusions, special-rule flags and reasons.
T0's exact-versus-mixed SC finding is documented in docs/fastf1-findings.md.
Amending the historical Leakage rule 5 prose requires approval and a log entry.

## Workflow and acceptance

Follow AGENTS.md. Tasks run sequentially, one branch and one PR per task.
Contract edits belong in a dedicated PR. Code, comments, docs and commits use
English. Write short code the students can explain. Tests and data processing
run offline. Data, caches, credentials and raw archives stay outside Git.

Task briefs state the outcome, applicable docs, allowed files, acceptance,
branch and report location. Branch prefix defaults to `codex/`.
After successful local checks, executors push their branch and open or update
its PR themselves using the report as the body. This standing task publication
permission includes ordinary PR/CI reads and correction commits. Use tool
permission mechanisms for environmental restrictions and report actual blockers.
Avoid force-push and direct application main writes.

The orchestrator checks full head SHA, allowed scope and CI against the saved
report and acceptance artifacts. Re-run checks when a concrete gap warrants it.
A human authorizes merging, and may delegate it to the agent. Squash-merge
after required checks, then update local main with --ff-only. The receiving
human can provide authorization in their own chat.

The ordinary review policy calls for an initial independent review for medium
risk tasks and targeted audits of labels, leakage and metrics for high risk
tasks. Alek explicitly waived independent reviews for work on 2026-11-24,
including completed T3 and T5. Record that historical omission. The waiver is
dated. Future process choices belong to the receiving human. Ordinary fixes
after a review use a finding-to-fix mapping and fresh affected checks. Request
another review only for a concrete unresolved finding or new risk.

Baseline local acceptance:

```sh
uv run --offline --no-sync ruff check .
uv run --offline --no-sync ruff format --check .
uv run --offline --no-sync pytest
uv run --offline --no-sync pitstop --help
```

Use a writable task-specific UV_CACHE_DIR if the default uv metadata is blocked.
On Alek's Mac, importing torch before fitting LightGBM caused a segfault with
the locked versions. Importing lightgbm first worked. Check this in T6 and
preserve a working path. CPU is the reference device.

## Continue after T4c

| Task | Outcome and principal acceptance |
| --- | --- |
| T4 | Causal features, 15-lap windows and build-dataset CLI. Test truncation of all streams, rival data after T, window provenance, train-only scaling and aligned output artifacts. |
| T6 | Shared trainer, prediction and constant/logreg/LightGBM baselines. CPU smoke on fixture inputs, finite predictions, reproducible artifacts and save/load equality. Reuse merged T5 metrics. |
| T7 | Streamlit shell on fixture predictions. Offline AppTest changes race and driver. |
| T8–T10 | MLP, GRU and causal TCN. Test shapes, small-sample overfit and save/load equality. TCN must pass causality tests. |
| T11 | Bounded validation experiments, saved runs and evidence-backed results. Agree compute budget before the larger sweep. |
| T12 | Freeze configurations and validation thresholds, obtain explicit human test approval, then perform the single test evaluation and demo export. |
| T13 | Dashboard on exported predictions and metadata, offline AppTest, deployment under human authorization. |
| T14 | Final README/runbook, Colab and offline end-to-end reproduction. |

The team may change this order where dependencies allow. Preserve the meaning
of each accepted interface. T1c's alignment of repository agent-workflow prose
is also open. Do not declare a whole phase complete while its tasks are pending.

## Maintain and transfer the report

The report is in the separate public repository
[f1-pitstop-report](https://github.com/lyskawson/f1-pitstop-report).
At handoff its main is `d1a908478ee49abb5cb8e8bf62787cb1516ce38c`.
Its report-state.json still points to application T2 at `5ba376f`.
Alek will ask his writer to update his part through merged T5.

At the end of the receiving teammate's work, give a report writer
`prompts/update-report.md`. First integrate the report remote so Alek's and
other teammates' edits are preserved. Update one shared narrative from the
last documented application commit through the latest accepted main. Reuse
the existing template and title page. Explain the implementation, technology
roles, tests, uses, limitations and goal completion in readable Polish.

The human wants natural, concise prose with only details needed to understand
the project. Remove repetition and development-process narration. Keep hashes,
full schemas, command inventories and test-case arithmetic in maintenance
evidence. Update implemented methods and conclusions in place. Build and
inspect the PDF and publish the checked sources to the existing report remote.

Before the next handoff, update the task table, exact source/PR/CI references,
open decisions and next copyable executor prompt. Keep the portable files free
of local account paths, cache content and credentials.
