TASK T4c — timestamped lap-count input contract     Owner: A     Risk: 2

Resolve PROJECT_ROOT from the application checkout containing this file.
Resolve NOTES_ROOT from the writable notes location supplied by the human or
orchestrator. Store the report at NOTES_ROOT/reports/T4c.md. Use actual paths.

Outcome: make the existing archived lap-count evidence a documented,
validated input for T4's causal laps_remaining calculation, preserving the
current unresolved-input error policy and immutable v1 archive.
Deliver a focused contract PR as required by AGENTS.md.

Start from up-to-date main. T5 was merged at
4927786152af1b6343f98720754d266e9b3a4433. Check live state for later work.
Read AGENTS.md, docs/contracts.md raw snapshot and dataset features,
docs/decisions.md D5 and Leakage rules, docs/fastf1-findings.md planned-lap
evidence, and ingest.verify_data's optional-payload validation.

The existing contract requires counts known by sample cutoff T and an explicit
build error for unresolved required inputs. Implement those requirements.

Required behaviour:
- Document lap_count_evidence.json's actual format and provenance. It is
  already included and hashed in data-v1. Keep archive and manifest bytes.
  Raw snapshot verification continues to support archives without this
  optional file. T4 needs count evidence for the races it processes.
- Provide shared validation for this payload. Reuse existing optional
  verifier checks and preserve their behaviour, including source identity,
  clock basis, coverage and final-count/event consistency. Test malformed
  timestamps, counts, identity and missing provenance. Support actual parser
  output such as integral float counts and null TotalLaps updates.
- Specify the downstream lookup at T. Select the last positive integral,
  non-null TotalLaps whose finite session Time <= T in the same race.
  Use timestamp order and source order to break equal-time ties.
  Null TotalLaps updates leave the last known count in effect.
- laps_remaining is the selected count minus lap t. Later announcements,
  observed lap maxima and unqualified final events.total_laps cannot supply
  an earlier sample's count.
- If no count is known at T, the future builder reports an unresolved
  required-input error with race/sample context. State this in the contract.
  Implementing full features belongs to T4.

Allowed tracked files:
docs/contracts.md, src/pitstop/contracts.py, tests/test_contracts.py,
src/pitstop/ingest.py (reuse the shared validator),
tests/test_ingest.py (affected regressions), docs/runbook.md.
Preserve D2, D5's feature meanings, the 15-feature order, dependencies and data.
Consult the human about a genuine conflict with accepted decisions. Applying
the existing contract does not require another approval.

Acceptance:
uv run --offline --no-sync ruff check . &&
uv run --offline --no-sync ruff format --check . &&
uv run --offline --no-sync pytest

Locally also run:
uv run --offline --no-sync pitstop verify-data --manifest data/raw/v1/manifest.json
Confirm the existing payload validates and raw snapshot/manifest checksums
remain identical. Tests use synthetic payloads and existing offline fixtures.
If the archive is absent, obtain the shared release described in HANDOFF.md.

Deliver branch codex/t4c-lap-count-contract, one PR and the notes report with
full commit SHA, scope, exact checks/results, compatibility evidence and gaps.
After successful checks, push and create/update the PR yourself. Verify the
head and observe CI to completion. Use a writable UV_CACHE_DIR if needed.
Review policy and merge authorization come from the receiving human's current
orchestrator instructions. The historical waiver applied on 2026-11-24.
