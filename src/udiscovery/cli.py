from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paths import repository_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="udiscovery",
        description="Validate, fit, evaluate, and reproduce U-Discovery release artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate release data and paper artifacts")
    validate.add_argument("--root", type=Path, help="Release root (defaults to installed source root)")
    validate.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    train = subparsers.add_parser("train", help="Fit one configured equation candidate")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--epochs", type=int, help="Override configured epoch count")
    train.add_argument("--output-dir", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a safe NPZ reconstruction")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--reconstruction", type=Path)
    evaluate.add_argument("--output-dir", type=Path)

    reproduce = subparsers.add_parser("reproduce", help="Regenerate paper tables and numerical plots")
    reproduce.add_argument("--target", choices=("paper",), default="paper")
    reproduce.add_argument("--output-dir", type=Path)

    generate = subparsers.add_parser("generate-synthetic", help="Generate a new documented synthetic realization")
    generate.add_argument(
        "--dataset",
        type=Path,
        default=repository_root() / "data/synthetic/dataset.npz",
        help="Frozen structure bundle",
    )
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        from .validate import validate_release

        report = validate_release(args.root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("PASS" if report["ok"] else "FAIL")
            for warning in report["warnings"]:
                print(f"warning: {warning}")
            for error in report["errors"]:
                print(f"error: {error}")
        if not report["ok"]:
            raise SystemExit(1)
        return
    if args.command == "train":
        from .train import fit_candidate

        output = fit_candidate(
            args.config,
            device_override=args.device,
            epochs_override=args.epochs,
            output_dir=args.output_dir,
        )
        print(output)
        return
    if args.command == "evaluate":
        from .evaluate import evaluate_reconstruction

        output = evaluate_reconstruction(
            args.config,
            reconstruction_path=args.reconstruction,
            output_dir=args.output_dir,
        )
        print(output)
        return
    if args.command == "reproduce":
        from .reproduce import reproduce_paper

        print(reproduce_paper(args.output_dir))
        return
    if args.command == "generate-synthetic":
        from .synthetic import generate_from_frozen_structure

        print(
            generate_from_frozen_structure(
                resolve_path(args.dataset), resolve_path(args.output), seed=args.seed
            )
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
