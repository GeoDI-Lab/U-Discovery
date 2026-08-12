from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyntheticParameters:
    beta: float = 0.488345142323785
    gamma: float = 1.2680008527060398
    noise_sigma: float = 1.678669956672565
    forcing_amplitude: float = 1.0428799862567975
    forcing_frequency: float = 0.011187321582417076
    forcing_phase_multiplier: float = 0.558590514613578


def single_constrained_power_gravity(
    state: np.ndarray,
    distance: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Evaluate the paper's production-constrained power-gravity net flow."""
    state = np.maximum(np.asarray(state, dtype=np.float64), 1e-6)
    distance = np.asarray(distance, dtype=np.float64)
    if distance.shape != (state.size, state.size):
        raise ValueError("distance must have shape (nodes, nodes)")
    safe_distance = np.where(distance == 0, np.inf, distance)
    kernel = np.power(safe_distance, -gamma)
    np.fill_diagonal(kernel, 0.0)
    attractiveness = np.power(state, beta)
    row_sum = np.maximum(kernel @ attractiveness, 1e-12)
    inflow = attractiveness * (kernel.T @ (state / row_sum))
    return inflow - state


def simulate_power_gravity_sine(
    initial_state: np.ndarray,
    distance: np.ndarray,
    spatial_amplitude: np.ndarray,
    spatial_phase: np.ndarray,
    *,
    steps: int = 500,
    dt: float = 1.0,
    seed: int = 42,
    parameters: SyntheticParameters = SyntheticParameters(),
) -> np.ndarray:
    """Generate a clean synthetic realization of the documented mechanism.

    The archived ``data/synthetic/dataset.npz`` remains authoritative for the
    paper because random-number streams can differ between implementations.
    """
    rng = np.random.default_rng(seed)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    amplitude = np.asarray(spatial_amplitude, dtype=np.float64)
    phase = np.asarray(spatial_phase, dtype=np.float64)
    if amplitude.shape != state.shape or phase.shape != state.shape:
        raise ValueError("spatial amplitude and phase must match initial_state")
    output = np.empty((steps + 1, state.size), dtype=np.float64)
    output[0] = state
    for step in range(steps):
        spatial = single_constrained_power_gravity(
            state, distance, parameters.beta, parameters.gamma
        )
        temporal = (
            parameters.forcing_amplitude
            * amplitude
            * np.sin(
                2 * np.pi * parameters.forcing_frequency * step
                + parameters.forcing_phase_multiplier * phase
            )
        )
        noise = rng.normal(0.0, parameters.noise_sigma * np.sqrt(dt), state.size)
        state = state + dt * (spatial + temporal) + noise
        output[step + 1] = state
    return output.astype(np.float32)


def generate_from_frozen_structure(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
) -> Path:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        states = simulate_power_gravity_sine(
            data["initial_state"],
            data["distance_raw"],
            data["spatial_amplitude"],
            data["spatial_phase"],
            steps=int(data["states"].shape[0] - 1),
            dt=float(data["dt"]),
            seed=seed,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, states=states)
    return output_path
