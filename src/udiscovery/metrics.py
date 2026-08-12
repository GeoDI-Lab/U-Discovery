from __future__ import annotations

import numpy as np


def mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {prediction.shape}")
    return float(np.mean(np.square(prediction - truth)))


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    residual = np.sum(np.square(truth - prediction))
    total = np.sum(np.square(truth - truth.mean()))
    return float(1.0 - residual / total) if total > 0 else float("nan")


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Return the archived mean node-wise time correlation.

    UrbanDE-Net computes a Pearson coefficient along the time axis for every
    node and then averages the node coefficients.  Preserving that convention
    is necessary to reproduce Tables 2 and 4; flattening time and nodes gives a
    different value even when MSE and R2 are unchanged.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {prediction.shape}")
    if truth.ndim == 1:
        truth = truth[:, None]
        prediction = prediction[:, None]
    if truth.ndim != 2 or truth.shape[0] < 2:
        return float("nan")
    centered_truth = truth - truth.mean(axis=0, keepdims=True)
    centered_prediction = prediction - prediction.mean(axis=0, keepdims=True)
    numerator = np.sum(centered_truth * centered_prediction, axis=0)
    denominator = np.sqrt(
        np.sum(centered_truth**2, axis=0) * np.sum(centered_prediction**2, axis=0)
    )
    # The archived implementation adds epsilon and retains constant-node
    # coefficients as zero in the node average.
    return float(np.mean(numerator / (denominator + 1e-12)))


def state_and_delta_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.ndim != 2 or truth.shape != prediction.shape:
        raise ValueError("truth and prediction must share shape (time, node)")
    truth_delta = np.diff(truth, axis=0)
    predicted_delta = np.diff(prediction, axis=0)
    return {
        "mse": mse(truth, prediction),
        "r2": r2(truth, prediction),
        "correlation": correlation(truth, prediction),
        "mse_delta": mse(truth_delta, predicted_delta),
        "r2_delta": r2(truth_delta, predicted_delta),
        "correlation_delta": correlation(truth_delta, predicted_delta),
    }


def normalize_errors(
    metrics: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    output = dict(metrics)
    output["nmse"] = metrics["mse"] / baseline["mse"]
    output["nmse_delta"] = metrics["mse_delta"] / baseline["mse_delta"]
    return output
