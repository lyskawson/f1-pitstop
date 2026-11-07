"""Expose the pipeline commands; implementations arrive in later tasks."""

import argparse
import json
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
        stage = subparsers.add_parser(command, help=command.replace("-", " ").capitalize())
        if command == "download":
            stage.add_argument("--config", type=str)
            stage.add_argument(
                "--from-cache", action="store_true", help="Export existing cache offline"
            )
            stage.add_argument("--snapshot", type=str)
        elif command == "verify-data":
            stage.add_argument("--manifest", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return a nonzero status until the requested stage is implemented."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download" and (args.config or args.from_cache or args.snapshot):
            from pitstop.ingest import build_snapshot

            if not (args.config and args.from_cache and args.snapshot):
                raise ValueError("download requires --config, --from-cache and --snapshot")
            result = build_snapshot(args.config, args.snapshot)
        elif args.command == "verify-data" and args.manifest:
            from pitstop.ingest import verify_data

            result = verify_data(args.manifest)
        else:
            print(f"pitstop {args.command}: not implemented yet", file=sys.stderr)
            return 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"pitstop {args.command}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
