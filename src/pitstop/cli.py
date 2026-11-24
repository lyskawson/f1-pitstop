"""Run the pit stop prediction pipeline stages."""

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
        elif command == "evaluate":
            stage.add_argument("--predictions", help="Predictions parquet (required)")
            stage.add_argument("--samples", help="Declared samples parquet (required)")
            stage.add_argument("--data-manifest", help="Dataset manifest JSON (required)")
            stage.add_argument("--output-dir", help="Run artifact directory (required)")
            stage.add_argument("--run-id", help="Required when predictions contain multiple runs")
            stage.add_argument("--split", choices=("val", "test"), default="val")
            stage.add_argument("--bootstrap-samples", type=int, default=1000)
            stage.add_argument("--seed", type=int, default=42, help="Bootstrap seed")
            stage.add_argument("--synthetic", action="store_true", help="Use a synthetic fixture")
            stage.add_argument("--test-approval", help="Reference to explicit human test approval")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a pipeline stage and return its exit status."""
    argv = sys.argv[1:] if argv is None else argv
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
        elif args.command == "evaluate" and len(argv) > 1:
            from pitstop.evaluate import evaluate_files

            if not all((args.predictions, args.samples, args.data_manifest, args.output_dir)):
                raise ValueError(
                    "evaluate requires --predictions, --samples, --data-manifest and --output-dir"
                )
            metrics = evaluate_files(
                args.predictions,
                args.samples,
                args.data_manifest,
                args.output_dir,
                run_id=args.run_id,
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                synthetic=args.synthetic,
                test_approval=args.test_approval,
            )
            result = {
                "complete": not metrics["leakage_audit"]["required"],
                "run_id": metrics["run_id"],
                "output_dir": args.output_dir,
                "leakage_audit": metrics["leakage_audit"],
            }
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
