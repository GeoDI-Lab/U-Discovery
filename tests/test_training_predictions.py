"""Regressions for custom fitting, one-step timing, and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from udiscovery.custom import Equation
from udiscovery.train import fit_candidate


def make_config(root, *, initial=None):
    nodes, times = 3, 14
    t = np.arange(times, dtype=np.float32)[:, None]
    states = 2.0 + np.arange(nodes, dtype=np.float32)[None, :] + 0.05 * t**2
    distance = np.abs(np.arange(nodes)[:, None] - np.arange(nodes)[None, :]).astype(np.float32)
    dataset = root / "data.npz"
    np.savez_compressed(
        dataset, states=states, distance=distance, dt=np.asarray(0.25),
        train_end=np.asarray(8), val_end=np.asarray(11),
        initial_state=states[0], adjacency=(distance > 0).astype(np.float32),
    )
    config = {
        "name": "tiny", "dataset": str(dataset),
        "split": {"train_end": 8, "validation_end": 11, "test_end": times},
        "model": {
            "history_length": 2, "temporal_harmonics": 0, "encoder_dim": 4,
            "head_hidden": [4], "residual_channels": 4, "dilation_channels": 4,
            "skip_channels": 4, "end_channels": 4, "blocks": 1, "layers": 1,
            "encoder_dropout": 0.0, "head_dropout": 0.0,
            "adaptive_adjacency": False,
        },
        "training": {
            "epochs": 2, "batch_size": 3, "seed": 7, "device": "cpu",
            "initial_parameters": initial or {},
        },
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, states


def test_parameterless_equation_exports_aligned_direct_deltas(tmp_path):
    config, states = make_config(tmp_path)
    equation = Equation(
        "constant", lambda state, distance, parameters, *, adjacency: torch.ones_like(state) * 2.0,
        parameters={}, complexity=1,
    )
    run = fit_candidate(config, equation=equation, output_dir=tmp_path / "runs")
    with np.load(run / "reconstruction.npz", allow_pickle=False) as reconstruction:
        current, target = reconstruction["current_indices"], reconstruction["target_indices"]
        np.testing.assert_array_equal(current, np.arange(1, 13))
        np.testing.assert_array_equal(target, current + 1)
        np.testing.assert_array_equal(reconstruction["ground_truth_state"], states[target])
        np.testing.assert_allclose(reconstruction["predicted_delta"], 0.5)
        np.testing.assert_allclose(reconstruction["prediction"], states[current] + 0.5)
        np.testing.assert_array_equal(reconstruction["split_code"], [0]*6 + [1]*3 + [2]*3)
        test_rows = reconstruction["split_code"] == 2
        direct = float(np.mean((0.5 - (states[target[test_rows]] - states[current[test_rows]]))**2))
        wrong = float(np.mean((np.diff(reconstruction["prediction"][test_rows], axis=0)
                               - np.diff(states[target[test_rows]], axis=0))**2))
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["test"]["mse_delta"] == pytest.approx(direct)
    assert metrics["test"]["mse_delta"] != pytest.approx(wrong)
    assert metrics["evaluation_mode"] == "teacher_forced_one_step"
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["trainable_dynamics"] is False
    assert manifest["epochs_completed"] == 1
    with np.load(run / "fitted_parameters.npz", allow_pickle=False) as parameters:
        assert parameters["temporal_sin"].shape == (12, 3, 0)


def test_custom_k_is_trainable_and_has_no_temporal_forcing(tmp_path):
    config, _ = make_config(tmp_path)
    equation = Equation(
        "decay", lambda state, distance, parameters, *, adjacency: -parameters["k"] * state,
        parameters={"k": 0.2}, complexity=2,
    )
    run = fit_candidate(config, equation=equation, output_dir=tmp_path / "runs")
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["fixed_parameters"] == {}
    assert manifest["parameter_scopes"] == {"k": "global"}
    assert manifest["trainable_dynamics"] is True
    with np.load(run / "fitted_parameters.npz", allow_pickle=False) as parameters:
        fitted = parameters["interaction__k"]
        assert fitted.shape == (12,)
        assert np.isfinite(fitted).all()
        assert np.max(np.abs(fitted - 0.2)) > 1.0e-5


def test_fit_rejects_config_split_mismatch(tmp_path):
    config, _ = make_config(tmp_path)
    payload = json.loads(config.read_text())
    payload["split"]["train_end"] = 7
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"split.train_end=.*disagrees"):
        fit_candidate(config, output_dir=tmp_path / "runs")


def test_fit_rejects_nonfinite_gradients(tmp_path):
    config, _ = make_config(tmp_path)
    equation = Equation(
        "bad_gradient",
        lambda state, distance, parameters, *, adjacency:
            torch.ones_like(state) * torch.sqrt(parameters["k"] - parameters["k"]),
        parameters={"k": 0.2}, complexity=2,
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        fit_candidate(config, equation=equation, output_dir=tmp_path / "runs")
