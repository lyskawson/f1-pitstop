# F1 pit stop prediction

Predict next-lap tyre stops from information available at the end of the current lap.

Python 3.12 and uv are required. The pipeline commands are currently stubs.

```sh
uv sync --locked --extra train
uv run pitstop --help
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Read the [decisions](docs/decisions.md), [project plan](docs/plan.md),
[contracts](docs/contracts.md), and [runbook](docs/runbook.md).
