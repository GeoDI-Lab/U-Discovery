"""One-command custom-equation fitting and validation-set ranking."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import load_dataset, validate_dataset


class DiscoveryError(RuntimeError):
    """A run failed; its saved manifest contains the completed work and errors."""


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_candidates(rows: list[dict]) -> list[dict]:
    """Rank only validation error; use complexity then ID for exact ties."""
    ranked = sorted(rows, key=lambda row: (
        row["validation_mse_delta"], row["complexity"], row["candidate_id"]
    ))
    for position, row in enumerate(ranked, 1):
        row["rank"] = position
        row["pareto_optimal"] = not any(
            other["validation_mse_delta"] <= row["validation_mse_delta"]
            and other["complexity"] <= row["complexity"]
            and (other["validation_mse_delta"] < row["validation_mse_delta"]
                 or other["complexity"] < row["complexity"])
            for other in ranked
        )
    return ranked


def discover(
    data_path: str | Path,
    equations_path: str | Path,
    *,
    output_dir: str | Path = "outputs",
    epochs: int = 200,
    patience: int = 30,
    history_length: int | None = None,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    device: str = "auto",
    forcing_period: float | None = None,
    threads: int = 1,
    candidates: list[str] | None = None,
    continue_on_error: bool = False,
    check_only: bool = False,
) -> Path | None:
    """Fit all candidates and write a fresh run directory, or validate inputs.

    Equations are trusted Python code. Every fit uses the same chronological
    split, initialization seed, model settings, and training budget. Prediction
    uses observed history; no recursive forecasting is performed.
    """
    from .custom import load_equations, validate_equation
    from .train import _require_torch, fit_candidate, select_device, set_seed

    torch = _require_torch()
    for name, value in (("epochs", epochs), ("patience", patience),
                        ("batch_size", batch_size), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if forcing_period is not None and (not math.isfinite(forcing_period) or forcing_period <= 0):
        raise ValueError("forcing_period must be positive and finite (in samples)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    data_file = Path(data_path).expanduser().resolve()
    equation_file = Path(equations_path).expanduser().resolve()
    dataset = load_dataset(data_file)
    errors = validate_dataset(dataset, require_symmetric_distance=False)
    if errors:
        raise ValueError("Invalid dataset: " + "; ".join(errors))
    if history_length is None:
        history_length = min(12, dataset.train_end - 1)
    if (isinstance(history_length, bool) or not isinstance(history_length, int)
            or not 2 <= history_length < dataset.train_end):
        raise ValueError("history_length must be an integer >= 2 and smaller than train_end")
    selected_device = select_device(device)
    set_seed(seed)
    equations = load_equations(equation_file)
    if candidates is not None:
        unknown = set(candidates) - {equation.id for equation in equations}
        if unknown:
            raise ValueError(f"Unknown candidate IDs: {', '.join(sorted(unknown))}")
        equations = [equation for equation in equations if equation.id in candidates]
        if not equations:
            raise ValueError("Select at least one candidate")

    failures = []
    valid = []
    for equation in equations:
        try:
            validate_equation(equation, dataset, selected_device)
            valid.append(equation)
        except Exception as exc:
            if not continue_on_error:
                raise ValueError(f"Candidate {equation.id!r} failed input checks: {exc}") from exc
            failures.append({"candidate_id": equation.id, "stage": "input_check",
                             "error": f"{type(exc).__name__}: {exc}"})
    if check_only:
        if failures:
            raise ValueError("Equation checks failed: " + json.dumps(failures))
        print(f"Inputs valid: {len(valid)} equations, {dataset.num_timesteps} timesteps, "
              f"{dataset.num_nodes} nodes; train/validation/test boundaries "
              f"{dataset.train_end}/{dataset.val_end}/{dataset.num_timesteps}.")
        return None

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(tempfile.mkdtemp(prefix=f"discovery_{stamp}_", dir=destination))
    (run / "inputs").mkdir()
    (run / "configurations").mkdir()
    shutil.copyfile(equation_file, run / "inputs/equations.py")
    np.savez_compressed(
        run / "inputs/data.npz", states=dataset.states,
        initial_state=dataset.initial_state, distance=dataset.distance,
        adjacency=dataset.adjacency, node_ids=dataset.node_ids,
        train_end=np.asarray(dataset.train_end), val_end=np.asarray(dataset.val_end),
        dt=np.asarray(dataset.dt),
    )
    model_settings = {
        "history_length": history_length,
        "temporal_harmonics": int(forcing_period is not None),
        "encoder_dim": 32, "head_hidden": [32, 16],
        "residual_channels": 16, "dilation_channels": 16,
        "skip_channels": 32, "end_channels": 32, "blocks": 2, "layers": 2,
        "encoder_dropout": 0.1, "head_dropout": 0.1,
        "adaptive_adjacency": True,
    }
    denominators = {}
    for split, start, end in (
        ("train", history_length, dataset.train_end),
        ("validation", dataset.train_end, dataset.val_end),
        ("test", dataset.val_end, dataset.num_timesteps),
    ):
        change = (dataset.states[start:end].astype(np.float64)
                  - dataset.states[start - 1:end - 1].astype(np.float64))
        denominators[split] = float(np.mean(change ** 2))
    from . import __version__

    summary = {
        "schema_version": 1, "status": "running",
        "udiscovery_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": "teacher_forced_one_step",
        "ranking_rule": "validation_mse_delta ascending, complexity ascending, candidate_id ascending",
        "test_used_for_selection": False,
        "complexity_source": "user-supplied spatial equation complexity; shared neural architecture excluded",
        "normalization": "MSE divided by persistence (zero predicted change) MSE on the same targets; null if zero",
        "persistence_mse": denominators,
        "requested_candidates": [equation.id for equation in equations],
        "seed": seed, "device": str(selected_device), "threads": threads,
        "model": model_settings, "forcing_period_samples": forcing_period,
        "split": {"train_end": dataset.train_end, "validation_end": dataset.val_end,
                  "test_end": dataset.num_timesteps},
        "inputs": {"data": "inputs/data.npz", "equations": "inputs/equations.py",
                   "original_data_sha256": _sha256(data_file),
                   "saved_data_sha256": _sha256(run / "inputs/data.npz"),
                   "equations_sha256": _sha256(equation_file)},
        "failures": failures,
    }
    _json(run / "summary.json", summary)
    previous_threads = torch.get_num_threads()
    rows = []
    aborted = False
    try:
        torch.set_num_threads(threads)
        for position, equation in enumerate(valid, 1):
            print(f"[{position}/{len(valid)}] Fitting {equation.id} on {selected_device}...", file=sys.stderr, flush=True)
            config = {
                "name": "discovery", "dataset": "../inputs/data.npz",
                "split": summary["split"], "model": model_settings,
                "training": {
                    "candidate_id": equation.id, "initial_parameters": dict(equation.parameters),
                    "parameter_mode": "localized", "seed": seed, "deterministic": True,
                    "device": str(selected_device), "epochs": epochs, "patience": patience,
                    "batch_size": batch_size, "learning_rate": learning_rate,
                    "weight_decay": 1e-5, "min_delta": 0.0,
                    "forcing_frequency": 1.0 / forcing_period if forcing_period is not None else 1.0,
                },
                "output_dir": "../candidates",
            }
            config_file = run / "configurations" / f"{equation.id}.json"
            _json(config_file, config)
            try:
                candidate_run = fit_candidate(config_file, equation=equation)
                metrics = json.loads((candidate_run / "metrics.json").read_text(encoding="utf-8"))
                manifest = json.loads((candidate_run / "run_manifest.json").read_text(encoding="utf-8"))
                row = {
                    "candidate_id": equation.id, "name": equation.display_name,
                    "complexity": equation.complexity, "formula": equation.formula,
                    "best_epoch": manifest["best_epoch"],
                    "epochs_completed": manifest["epochs_completed"],
                    "run_directory": candidate_run.relative_to(run).as_posix(),
                }
                for split in ("train", "validation", "test"):
                    for metric, value in metrics[split].items():
                        row[f"{split}_{metric}"] = value
                    for metric in ("mse", "mse_delta"):
                        row[f"{split}_n{metric}"] = (
                            metrics[split][metric] / denominators[split]
                            if denominators[split] > 0 else None
                        )
                if not math.isfinite(row["validation_mse_delta"]):
                    raise ValueError("Candidate has no finite validation score")
                rows.append(row)
                print(f"  validation MSE(change) = {row['validation_mse_delta']:.6g}", file=sys.stderr, flush=True)
            except Exception as exc:
                failures.append({"candidate_id": equation.id, "stage": "fit_or_evaluation",
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"  Failed: {exc}", file=sys.stderr, flush=True)
                if not continue_on_error:
                    aborted = True
                    break
            summary["completed_candidates"] = [row["candidate_id"] for row in rows]
            _json(run / "summary.json", summary)
    except KeyboardInterrupt:
        failures.append({"candidate_id": equation.id, "stage": "interrupted",
                         "error": "Run interrupted by the user"})
        aborted = True
    finally:
        torch.set_num_threads(previous_threads)

    ranked = rank_candidates(rows)
    summary.update({
        "status": "failed" if aborted or not ranked else "partial" if failures else "complete",
        "completed_candidates": [row["candidate_id"] for row in ranked],
        "best_candidate": ranked[0]["candidate_id"] if ranked and not aborted else None,
        "failures": failures,
    })
    _json(run / "summary.json", summary)
    _json(run / "failed_candidates.json", failures)
    _json(run / "rankings.json", {"ranking_rule": summary["ranking_rule"],
                                 "status": summary["status"], "candidates": ranked})
    if ranked:
        fields = ["rank", "candidate_id", "name", "complexity", "pareto_optimal"]
        fields += [key for key in ranked[0] if key not in fields]
        with (run / "rankings.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ranked)
    report = ["# Equation discovery results", "", f"Run status: **{summary['status']}**.", "",
              "Ranked by validation MSE of predicted changes, then spatial equation complexity.",
              "Test metrics do not affect selection. Predictions use observed history (one step ahead).", "",
              "| Rank | Equation | Validation MSE(change) | Test MSE(change) | Complexity |",
              "| ---: | --- | ---: | ---: | ---: |"]
    report += [f"| {row['rank']} | {row['candidate_id']} | {row['validation_mse_delta']:.6g} | "
               f"{row['test_mse_delta']:.6g} | {row['complexity']} |" for row in ranked]
    report += ["", "See rankings.csv for all metrics, summary.json for settings and input hashes,",
               "and candidates/ for weights, inferred parameters, predictions and training history.",
               "Input snapshots include the equation file; retain any external helper modules it imports.",
               "Failures and incomplete comparisons are recorded in failed_candidates.json.", ""]
    (run / "README.md").write_text("\n".join(report), encoding="utf-8")
    if aborted or not ranked:
        raise DiscoveryError(f"Discovery failed. Saved results and error details: {run}")
    return run
