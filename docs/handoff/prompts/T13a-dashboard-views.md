TASK T13a — dashboard field heatmap and model comparison views     Owner: C     Risk: 1

PROJECT_ROOT = <application checkout>. NOTES_ROOT = <sibling notes directory>.
Report: NOTES_ROOT/reports/T13a.md (written so it can serve as the PR body).
Branch: codex/t13a-dashboard-views from up-to-date main (T12b merged as 1c76890).

Outcome: `app/streamlit_app.py` shows the three D11 views on the committed
demo pack (`demo/`, T12b): the existing race view, a field heatmap and a model
comparison. The app keeps reading only precomputed files (no torch, lightgbm,
fastf1 or pitstop import, no training, no evaluation).

Read first: AGENTS.md; docs/decisions.md D11; docs/contracts.md
"evaluate → dashboard"; docs/runbook.md (T7, T12b); app/streamlit_app.py;
tests/test_app.py; demo/metadata.json; reports/test-v1.md (to label the
comparison correctly). The T7 PR #19 description explains the race view.

Required behaviour:
- Organise the page with `st.tabs` (Race view, Field heatmap, Model
  comparison). Race and model selectors are shared above the tabs; the
  driver selector belongs to the race view. Keep the race view unchanged.
- Field heatmap: for the selected race and model, drivers × target lap,
  colour = score (fixed 0..1 scale), real tyre changes (y = 1 at target_lap)
  visibly marked (e.g. outlined cells or point markers), and cells at or above
  the frozen threshold distinguishable. Use Altair, as the race view does.
  Show event names from metadata.json where available (fall back to race_id).
- Model comparison: a table per run in the pack with model name, seed, test
  AP with its 95% CI, prevalence, F1 / precision / recall at the frozen
  threshold, event k=1 F1 and false alarms per race, read from the "test"
  split, "overall" group of demo/metrics.json; plus a simple AP chart with CI
  error bars and the prevalence as a reference line. State in a caption that
  these are full test-split metrics (all test races) of the seed-42 run of
  each model, that a CI overlap means no ranking, and that the mean over
  seeds is in reports/test-v1.md. Never compute new metrics in the app.
- Missing or inconsistent metrics.json/metadata.json: an `st.info`/`st.error`
  in the affected tab only; the other views keep working.
- Keep the code short; small helper functions like the existing ones.

Tests (offline AppTest, synthetic packs only, extend tests/test_app.py):
- the three tabs render without exceptions on the synthetic pack (extend the
  existing make_pack with metrics.json and metadata.json in contract shape);
- switching race and model updates the heatmap (one chart rendered);
- the comparison table shows the AP values from the synthetic metrics.json
  exactly (no recomputation);
- missing metrics.json only affects the comparison tab;
- the existing tests keep passing, including the no-heavy-imports check.
Then run the app via AppTest on the real `demo/` once and report what renders
(the real pack has 3 races, 6 runs).

Allowed tracked files: app/streamlit_app.py, tests/test_app.py,
docs/runbook.md (short T13a note). Do NOT edit demo/, src/, contracts,
decisions, reports or dependencies (streamlit and altair are already locked).

Acceptance:
uv run --offline --no-sync ruff check . &&
uv run --offline --no-sync ruff format --check . &&
uv run --offline --no-sync pytest
plus the AppTest check on the real pack.

Deliver: commits on the branch, the report with full head SHA, scope, checks,
a short description of each view (what the real pack shows), and known gaps.
