from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paths import repository_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="udiscovery",
        description="Rank custom equations with UrbanDE-Net, or reproduce U-Discovery artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Fit and rank equations from your data and Python file")
    discover.add_argument("--data", required=True, type=Path, help="NPZ with states and distance arrays")
    discover.add_argument("--equations", required=True, type=Path, help="Trusted Python file exporting EQUATIONS")
    discover.add_argument("--output-dir", type=Path, default=Path("outputs"))
    discover.add_argument("--epochs", type=int, default=200)
    discover.add_argument("--patience", type=int, default=30)
    discover.add_argument("--history-length", type=int, help="Observed history per prediction (default: up to 12)")
    discover.add_argument("--batch-size", type=int, default=32)
    discover.add_argument("--learning-rate", type=float, default=0.001)
    discover.add_argument("--seed", type=int, default=42)
    discover.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    discover.add_argument("--forcing-period", type=float, help="Enable Fourier forcing with this period in samples")
    discover.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    discover.add_argument("--candidate", nargs="+", help="Fit only these equation IDs (default: all)")
    discover.add_argument("--continue-on-error", action="store_true", help="Record failed candidates and rank the successful ones")
    discover.add_argument("--check-only", action="store_true", help="Validate data and equation gradients without fitting")

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
    evaluate.add_argument("--split", choices=("train", "validation", "test", "all"), default="test",
                          help="Split for new indexed reconstructions (legacy archives retain their own scope)")

    reproduce = subparsers.add_parser("reproduce", help="Regenerate paper tables and numerical plots")
    reproduce.add_argument("--target", choices=("paper",), default="paper")
    reproduce.add_argument("--output-dir", type=Path)

    generate = subparsers.add_parser("generate-synthetic", help="Generate a new documented synthetic realization")
    generate.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Frozen structure bundle",
    )
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "discover":
        from .discover import discover

        try:
            output = discover(
                args.data, args.equations, output_dir=args.output_dir,
                epochs=args.epochs, patience=args.patience,
                history_length=args.history_length, batch_size=args.batch_size,
                learning_rate=args.learning_rate, seed=args.seed, device=args.device,
                forcing_period=args.forcing_period, threads=args.threads,
                candidates=args.candidate, continue_on_error=args.continue_on_error,
                check_only=args.check_only,
            )
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            parser.exit(1, f"Discovery error: {exc}\n")
        if output is not None:
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            print(f"Results: {output}")
            print(f"Best equation by validation error: {summary['best_candidate']}")
            print(f"Rankings: {output / 'rankings.csv'}")
            if summary["failures"]:
                print(f"Warning: {len(summary['failures'])} candidate(s) failed; ranking is partial.", file=sys.stderr)
        return
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
            split=args.split,
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
                resolve_path(args.dataset) if args.dataset else repository_root() / "data/synthetic/dataset.npz",
                Path(args.output).expanduser().resolve(), seed=args.seed
            )
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
