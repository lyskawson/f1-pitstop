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
