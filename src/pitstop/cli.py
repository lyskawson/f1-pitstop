"""Expose the pipeline commands; implementations arrive in later tasks."""

import argparse
import sys

COMMANDS = (
    "download",
    "verify-data",
    "build-dataset",
    "train",
    "predict",
    "evaluate",
    "export-demo",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the common command-line interface."""
    parser = argparse.ArgumentParser(prog="pitstop", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparsers.add_parser(command, help="Not implemented yet.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return a nonzero status until the requested stage is implemented."""
    args = build_parser().parse_args(argv)
    print(f"pitstop {args.command}: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
