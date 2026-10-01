from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_config, resolved_dataset_path
from .data import load_dataset
from .metrics import correlation, mse, r2, state_and_delta_metrics
from .paths import resolve_path


def evaluate_reconstruction(
    config_path: str | Path,
    *,
    reconstruction_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    split: str = "test",
) -> Path:
    if split not in {"train", "validation", "test", "all"}:
        raise ValueError("split must be train, validation, test, or all")
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
    indexed = None
    with np.load(reconstruction_file, allow_pickle=False) as source:
        if "target_indices" in source:
            required = {"prediction", "predicted_delta", "target_indices"}
            if not required.issubset(source.files):
                raise ValueError("Indexed reconstruction requires prediction, predicted_delta and target_indices")
            indexed = {name: np.asarray(source[name]) for name in required}
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
    if indexed is not None:
        prediction = indexed["prediction"]
        targets = indexed["target_indices"]
        if (targets.ndim != 1 or targets.dtype.kind not in "iu" or len(targets) == 0
                or np.any(targets < 1) or np.any(targets >= dataset.num_timesteps)
                or np.any(np.diff(targets.astype(np.int64)) != 1)):
            raise ValueError("target_indices must be consecutive integer indices within dataset[1:]")
        truth = dataset.states[targets]
        previous = dataset.states[targets - 1]
        delta = indexed["predicted_delta"]
        if prediction.shape != truth.shape or delta.shape != truth.shape:
            raise ValueError("Indexed prediction and predicted_delta must match (targets, nodes)")
        if not np.isfinite(prediction).all() or not np.isfinite(delta).all():
            raise ValueError("Reconstruction contains non-finite predictions")
        if packaged_truth is not None and not np.array_equal(packaged_truth, truth):
            raise ValueError("Packaged ground truth does not match the configured dataset at target_indices")
        if not np.allclose(prediction, previous + delta, rtol=1e-5, atol=1e-6):
            raise ValueError("prediction must equal observed previous state + predicted_delta")
        selected = {
            "train": targets < dataset.train_end,
            "validation": (targets >= dataset.train_end) & (targets < dataset.val_end),
            "test": targets >= dataset.val_end,
            "all": np.ones(targets.shape, dtype=bool),
        }[split]
        if not selected.any():
            raise ValueError(f"No predictions for the {split} split")
        truth_delta = (truth - previous)[selected]
        truth, prediction, delta = truth[selected], prediction[selected], delta[selected]
        metrics = {
            "mse": mse(truth, prediction), "r2": r2(truth, prediction),
            "correlation": correlation(truth, prediction),
            "mse_delta": mse(truth_delta, delta), "r2_delta": r2(truth_delta, delta),
            "correlation_delta": correlation(truth_delta, delta),
        }
    elif prediction.shape != truth.shape:
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
    if indexed is None:
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError("Reconstruction contains non-finite states")
        metrics = state_and_delta_metrics(truth, prediction)
    metrics = {key: value if np.isfinite(value) else None for key, value in metrics.items()}
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
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = destination / f"{config['name']}_evaluation_{stamp}.json"
    payload = {
        "schema_version": 1,
        "configuration": str(config_file),
        "dataset": str(dataset.path),
        "reconstruction": str(reconstruction_file),
        "evaluated_shape": list(truth.shape),
        "metrics": metrics,
        "evaluation_mode": "teacher_forced_one_step" if indexed is not None else "legacy_state_sequence",
        "split": split if indexed is not None else "archive_defined",
    }
    try:
        root = config_file.parent.parent
        payload["configuration"] = str(config_file.relative_to(root))
        payload["dataset"] = str(dataset.path.relative_to(root))
        payload["reconstruction"] = str(reconstruction_file.relative_to(root))
    except ValueError:
        pass
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output
