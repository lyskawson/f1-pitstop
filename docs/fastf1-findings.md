# T0: FastF1 findings

FastF1 **3.9.0a0**, audited **2026-11-06**. Evidence comes from
`data/t0/summary.json`, produced by
`uv run python notebooks/t0_fastf1_spike.py analyze` with FastF1 offline mode
and an additional socket guard. The human downloaded the cache. No model was
trained, no features were fitted, and the locked test split was not evaluated.
Weather is cached and checked for availability only; it is not a feature.

**D2 is not validated: Stint can increase on a penalty-only pit-lane pass.**
Stop before implementing labels and obtain a human decision on the questions
below. Decisions and contracts have not been edited.

## Checks

| Check | Assumption or question | Yes / no | Evidence in summary.json |
| --- | --- | --- | --- |
| a | PitInTime marks the in-lap and PitOutTime the out-lap. | Yes for row association; no strict timestamp-containment guarantee. | `a_pit_boundaries`: 3,469 in-lap markers and 3,467 out-lap markers. No marked row lacks timing bounds. All out-lap timestamps lie inside their lap; 97 in-lap timestamps exceed the recorded lap end. For example, `2023_13_R`, driver `1`, lap 2: PitInTime 3925.519 s, lap end 3924.644 s. The following lap carries PitOutTime. Keep the raw association; do not move markers to another row merely to enforce timestamp containment. |
| b | Stint stays unchanged on drive-through and stop/go penalties. | **No.** | `b_penalties`: `2022_08_R`, driver `6`, laps 3–4; `2024_23_R`, driver `4`, laps 45–46 and driver `44`, laps 47–48; `2025_20_R`, driver `55`, laps 56–57. Stint increases despite continuing Compound and TyreLife at the penalty pass. In contrast, `2025_08_R`, driver `63`, laps 62–63 preserves Stint. See the detailed table below. |
| c | The loader uses the last non-null total_laps; is that safe at every earlier lap? | Yes to last-non-null; **no** to a constant count across all races. | `c_total_laps`: every loaded race agrees with the last non-null announcement. Japan `2022_18_R` reports 53 planned laps while the maximum recorded lap is 28; its only non-null announcement is 53. Other races change their planned count, including retained races `2023_14_R` and `2025_11_R`. Preserve timestamped announcements before using laps_remaining. |
| d | Red-flag interruptions remain visible in laps and TrackStatus. | Yes. | `d_red_flags.lap_context`: `2022_10_R`, driver `1`, lap 1 has `125`, followed by lap 2 with `1`; `2023_03_R`, driver `1`, laps 8, 55 and 57 contain `5`; `2024_08_R`, driver `1`, lap 1 has `1254`, followed by lap 2 with `14`. Restart gaps and pit markers remain in the raw rows. These whole races are excluded. |
| e | The D2 Stint-increase proxy equals every Jolpica pit stop within ±1 lap. | No exact equality. | `e_pit_matches`: 3,339 proxy events; 3,363 Jolpica stops; 3,339 one-to-one matches, or 99.286% of Jolpica stops. Every proxy event matches, with 24 Jolpica-only stops. This measures agreement of recorded visits, not proof of tyre changes: penalties also appear in both sources. Per-driver counts, matched pairs and every mismatch are retained. |
| f | Wet/red races and special tyre rule races can be identified. | Yes. | `exclusion_reasons` identifies 18 whole-race exclusions, listed below. `special_tyre_rule_race_ids` identifies `2023_18_R`, `2025_08_R`, `2025_23_R`; a special rule is a flag, not an exclusion. |
| g | All statements about IsAccurate from the earlier FastF1 version still hold. | **No** for the broad SC statement; yes for the other masks and FastF1Generated. | `g_accuracy`: zero accurate rows among 3,469 in-laps, 3,467 out-laps, 135 generated laps and 7,270 non-green/yellow laps. After a previous status exactly `4`, all 2,806 rows are inaccurate. After any previous status containing `4`, 1,017 of 5,260 rows are accurate. Example: `2022_01_R`, driver `1`, lap 51 is accurate after a mixed SC status. FastF1Generated exists in every loaded race. |
| h | Coverage includes all requested seasons and reports failures. | Yes. | `coverage`: 92 races loaded, zero race failures; `errors` and `schedule_errors` are empty. By season: 22, 22, 24 and 24 races. Weather is present in every loaded session (`weather_rows`). Download-side completeness is recorded separately in `data/t0/download_log.json`. |

Counts in (e) use the documented pit-entry plus next-lap Stint-increase proxy
after removing generated rows. They do not filter IsAccurate and do not claim
that every proxy event is an actual tyre change. Jolpica driver IDs are joined
through driver codes to the historical car numbers in the race laps. Jolpica's
permanent-number field can reflect a later season and must not be used directly
for this join. No session.results fields are used by the audit.

## Penalty evidence

All times below are session seconds; `null` means no marker. These are raw
observations, not repaired labels. Full surrounding rows, all pit passes by each
affected driver, and the race-control messages are in `b_penalties`.

| race_id | Driver | Lap | Stint | PitInTime | PitOutTime | Compound | TyreLife |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022_08_R | 6 | 3 | 1 | 4131.373 | null | MEDIUM | 3 |
| 2022_08_R | 6 | 4 | 2 | null | 4160.747 | MEDIUM | 4 |
| 2024_23_R | 4 | 45 | 4 | 7757.138 | null | HARD | 10 |
| 2024_23_R | 4 | 46 | 5 | null | 7793.161 | HARD | 11 |
| 2024_23_R | 44 | 47 | 4 | 7942.075 | null | HARD | 13 |
| 2024_23_R | 44 | 48 | 5 | null | 7964.810 | HARD | 14 |
| 2025_20_R | 55 | 56 | 3 | 8127.048 | null | SOFT | 13 |
| 2025_20_R | 55 | 57 | 4 | null | 8144.933 | SOFT | 14 |
| 2025_08_R | 63 | 62 | 2 | 8352.534 | null | MEDIUM | 9 |
| 2025_08_R | 63 | 63 | 2 | null | 8377.454 | MEDIUM | 10 |

The race-control messages identify stop/go penalties for Latifi and Norris,
and drive-through penalties for Hamilton, Sainz and Russell. Continuing tyre
age supports the no-change interpretation; unchanged Compound alone would not
prove it. Penalty message lap numbers can refer to issuance or delayed
confirmation. For Russell the issue message is on lap 53 and the served message
on lap 78; the audit therefore preserves all passes instead of automatically
assigning a penalty to the nearest message lap.

## Changing planned lap counts

These are the non-null `TotalLaps` values, in announcement order. Exact session
timestamps and current-lap values remain in `c_total_laps.announcements`.

| race_id | Announced counts |
| --- | --- |
| 2022_07_R | 78 → 77 |
| 2023_14_R | 53 → 52 → 51 |
| 2024_21_R | 71 → 70 → 69 |
| 2025_01_R | 58 → 57 |
| 2025_11_R | 71 → 70 |

Japan shows why the reported planned count must also be distinguished from
completed laps. Do not replace planned counts with a final observed lap maximum.

## Jolpica-only stops

Each entry is `driver_number: lap`; separate entries denote separate visits.
There are no FastF1-only proxy events after matching. No automatic exclusion or
correction is inferred from this table.

| race_id | Unmatched Jolpica visits |
| --- | --- |
| 2022_07_R | 6: 19; 6: 29 |
| 2022_09_R | 22: 47 |
| 2022_11_R | 77: 39; 23: 41; 5: 44; 55: 50 |
| 2022_15_R | 22: 43 |
| 2023_08_R | 22: 34 |
| 2024_21_R | 43: 30 |
| 2025_06_R | 12: 25; 55: 25; 1: 26; 23: 26; 22: 27; 44: 28; 14: 28; 30: 28; 81: 29; 4: 29; 63: 29; 16: 29 |
| 2025_08_R | 63: 62 |
| 2025_15_R | 16: 52 |

Miami `2025_06_R` has missing Stint and TyreLife values recorded in
`driver_coverage`, so a successful session load is not a guarantee of usable
labels. Russell's unmatched Monaco visit is the unchanged-Stint penalty pass
shown above. Remaining discrepancies need ingestion-level review of retirements,
pit-lane visits and missing timing/tyre data; do not silently turn them into
negative labels or describe all Jolpica visits as tyre changes.

## Whole-race exclusions

`wet` means any INTERMEDIATE/WET lap; `red_flag` means any TrackStatus containing
`5`. This list implements only whole-race wet/red screening; later gap and
driver-level missingness rules still apply.

| race_id | Reason |
| --- | --- |
| 2022_04_R | wet |
| 2022_07_R | wet, red_flag |
| 2022_10_R | red_flag |
| 2022_17_R | wet |
| 2022_18_R | wet, red_flag |
| 2023_03_R | red_flag |
| 2023_06_R | wet |
| 2023_13_R | wet, red_flag |
| 2023_19_R | red_flag |
| 2023_20_R | red_flag |
| 2024_04_R | red_flag |
| 2024_08_R | red_flag |
| 2024_09_R | wet |
| 2024_12_R | wet |
| 2024_21_R | wet, red_flag |
| 2025_01_R | wet |
| 2025_12_R | wet |
| 2025_13_R | wet |

Special tyre rule flags: Qatar `2023_18_R`, Monaco `2025_08_R`, Qatar `2025_23_R`.

## Fixtures and reproduction

The fixtures contain the complete raw laps of Bahrain `2022_01_R` (1,125 rows)
and Saudi Arabia `2022_02_R` (820 rows), with the corresponding events records.
Both races are dry and have no red flag. Raw nulls and generated rows are
preserved. Times are float seconds; validate_raw_laps passes after Parquet
round-trip. Weather is not exported. The fixture tests need neither FastF1
cache nor credentials.

Run the following offline after the human has populated the cache:

```sh
uv run python notebooks/t0_fastf1_spike.py analyze
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

## Open questions requiring human decisions

- **D2:** Replace the claim that every Stint increase is a tyre change. Proposed
  direction: retain the existing raw-row horizon and joins, but distinguish
  verified tyre changes from penalty-only passes using tyre evidence and
  validated race-control context. Preserve same-compound tyre changes. A
  Compound-only rule or a TyreLife-reset-only rule is not established by this
  audit and must not be adopted without checking used tyre sets and missingness.
  The proposed fallback to PitInTime alone also labels penalty passes positive;
  using it would require an explicit change of target to pit-lane visits.
- **Raw contract / D5:** Add a timestamped lap-count announcement payload and
  provenance in a dedicated contract PR. Earlier-lap features must use the last
  announcement known at their cutoff, not Session.total_laps from the final
  stream. Define how to handle samples before the first known announcement.
- **Leakage rule 5:** Clarify that this FastF1 version masks the first lap after
  status exactly `4`, rather than every mixed status containing `4`. Continue
  using IsAccurate only for time features, never label-row selection.
- **Data quality:** Resolve missing tyre data and unmatched visits before
  claiming an exact tyre-change audit or a complete modelling dataset. Also
  document the sub-second PitInTime/lap-end offsets without reassigning labels.

No decision, contract, label implementation or exclusion beyond the existing
wet/red rules has been changed. T0 stops at these findings pending review.
