# Agent instructions

Read [decisions](docs/decisions.md), especially the
[Leakage rules](docs/decisions.md#leakage-rules-enforced-by-tests), and
[contracts](docs/contracts.md) before editing.

## Commands

```sh
uv sync --locked --extra train
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pitstop --help
```

## Owners

| Owner | Modules |
| --- | --- |
| A | ingest, labels, splits, features, windows, dataset, models/mlp.py |
| B | metrics, evaluate, train, predict, models/baselines.py, models/rnn.py |
| C | app/, models/tcn.py, CI |

## Rules

- One task = one branch = one PR; touch only the files the task allows. T1 alone
  bootstraps the repository on main without a PR. Tasks run one at a time.
- After T1, change `src/pitstop/contracts.py` and `docs/contracts.md` only in a
  dedicated contract PR. Keep schemas, documentation, and contract tests consistent.
- Work offline. Tests never touch the network. A human downloads FastF1 data once
  and shares a versioned archive with its SHA-256 manifest.
- Use only information available at the end of lap t. Follow the linked leakage
  rules; fit preprocessing on train only. Never evaluate the locked test split
  without explicit human approval.
- Never invent results. Every result number in README or reports must come from
  a file in `runs/` or `reports/`.
- Write short, readable code that students can explain. Use English for code,
  comments, documentation, and commits; preserve the original source document copies.
- Change `docs/decisions.md` only with human approval and a change-log entry.
- Never commit data, caches, tokens, or credentials.
