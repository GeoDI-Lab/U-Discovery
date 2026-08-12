from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_config, resolved_dataset_path
from .data import load_dataset
from .metrics import state_and_delta_metrics
from .paths import resolve_path


def evaluate_reconstruction(
    config_path: str | Path,
    *,
    reconstruction_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    config, config_file = load_config(config_path)
    dataset = load_dataset(resolved_dataset_path(config, config_file))
    evaluation = config.get("evaluation", {})
    requested = reconstruction_path or evaluation.get("reconstruction")
    if not requested:
        raise ValueError("No reconstruction path was supplied in CLI or configuration")
    if reconstruction_path is not None:
        supplied = Path(reconstruction_path).expanduser()
        reconstruction_file = (
            supplied.resolve() if supplied.is_absolute() else (Path.cwd() / supplied).resolve()
        )
    else:
        reconstruction_file = resolve_path(requested, base=config_file.parent)
    with np.load(reconstruction_file, allow_pickle=False) as source:
        packaged_truth = (
            np.asarray(source["ground_truth_state"])
            if "ground_truth_state" in source
            else None
        )
        if "states" in source:
            prediction = np.asarray(source["states"])
        elif "prediction" in source:
            prediction = np.asarray(source["prediction"])
        elif "teacher_forced_state" in source:
            prediction = np.asarray(source["teacher_forced_state"])
        else:
            if len(source.files) != 1:
                raise ValueError(
                    "Reconstruction NPZ contains several candidates; provide a single-candidate "
                    "NPZ such as the packaged top1_reconstruction.npz"
                )
            prediction = np.asarray(source[source.files[0]])
    truth = packaged_truth if packaged_truth is not None else dataset.states
    if prediction.shape != truth.shape:
        test_truth = truth[dataset.val_end :]
        transition_aligned_test_truth = truth[dataset.val_end + 1 :]
        if prediction.shape == test_truth.shape:
            truth = test_truth
        elif prediction.shape == transition_aligned_test_truth.shape:
            truth = transition_aligned_test_truth
        else:
            raise ValueError(
                f"Reconstruction shape {prediction.shape} does not match full {truth.shape} "
                f"or test {test_truth.shape} data"
            )
    metrics = state_and_delta_metrics(truth, prediction)
    if output_dir is not None:
        supplied_output = Path(output_dir).expanduser()
        destination = (
            supplied_output.resolve()
            if supplied_output.is_absolute()
            else (Path.cwd() / supplied_output).resolve()
        )
    else:
        destination = resolve_path(config.get("output_dir", "outputs"), base=config_file.parent)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{config['name']}_evaluation.json"
    if output.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = destination / f"{config['name']}_evaluation_{stamp}.json"
    payload = {
        "schema_version": 1,
        "configuration": str(config_file),
        "dataset": str(dataset.path),
        "reconstruction": str(reconstruction_file),
        "evaluated_shape": list(truth.shape),
        "metrics": metrics,
    }
    try:
        root = config_file.parent.parent
        payload["configuration"] = str(config_file.relative_to(root))
        payload["dataset"] = str(dataset.path.relative_to(root))
        payload["reconstruction"] = str(reconstruction_file.relative_to(root))
    except ValueError:
        pass
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output
