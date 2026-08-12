from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Dataset:
    states: np.ndarray
    initial_state: np.ndarray
    distance: np.ndarray
    adjacency: np.ndarray
    node_ids: np.ndarray
    train_end: int
    val_end: int
    dt: float
    path: Path
    distance_raw: np.ndarray | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.states.shape[1])

    @property
    def num_timesteps(self) -> int:
        return int(self.states.shape[0])


def load_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path).resolve()
    with np.load(dataset_path, allow_pickle=False) as source:
        states_key = "states" if "states" in source else "obs_dyn_full"
        states = np.asarray(source[states_key], dtype=np.float32)
        initial_key = "initial_state" if "initial_state" in source else "init_state"
        if "distance" in source:
            distance_key = "distance"
        elif "distance_matrix" in source:
            distance_key = "distance_matrix"
        else:
            distance_key = "d_mat"
        adjacency_key = "adjacency" if "adjacency" in source else "adjacency_mask"
        initial = np.asarray(source[initial_key], dtype=np.float32)
        distance = np.asarray(source[distance_key], dtype=np.float32)
        adjacency = np.asarray(source[adjacency_key], dtype=np.float32)
        node_ids = (
            np.asarray(source["node_ids"])
            if "node_ids" in source
            else np.arange(states.shape[1], dtype=np.int32)
        )
        train_end = int(source["train_end"])
        val_end = int(source["val_end"])
        if "dt" in source:
            dt = float(source["dt"])
        elif "interval_minutes" in source:
            dt = 1.0
        else:
            dt = 1.0
        distance_raw = (
            np.asarray(source["distance_raw"], dtype=np.float32)
            if "distance_raw" in source
            else None
        )
    return Dataset(
        states=states,
        initial_state=initial,
        distance=distance,
        adjacency=adjacency,
        node_ids=node_ids,
        train_end=train_end,
        val_end=val_end,
        dt=dt,
        path=dataset_path,
        distance_raw=distance_raw,
    )


def validate_dataset(
    dataset: Dataset,
    *,
    expected_timesteps: int | None = None,
    expected_nodes: int | None = None,
    require_symmetric_distance: bool = True,
    geometry_path: str | Path | None = None,
) -> list[str]:
    errors: list[str] = []
    states = dataset.states
    if states.ndim != 2:
        errors.append(f"states must be 2D, got {states.shape}")
        return errors
    timesteps, nodes = states.shape
    if expected_timesteps is not None and timesteps != expected_timesteps:
        errors.append(f"expected {expected_timesteps} timesteps, got {timesteps}")
    if expected_nodes is not None and nodes != expected_nodes:
        errors.append(f"expected {expected_nodes} nodes, got {nodes}")
    if dataset.initial_state.shape != (nodes,):
        errors.append(f"initial_state must have shape {(nodes,)}, got {dataset.initial_state.shape}")
    elif not np.array_equal(dataset.initial_state, states[0]):
        errors.append("initial_state does not equal states[0]")
    for name, matrix in (("distance", dataset.distance), ("adjacency", dataset.adjacency)):
        if matrix.shape != (nodes, nodes):
            errors.append(f"{name} must have shape {(nodes, nodes)}, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            errors.append(f"{name} contains non-finite values")
    if not np.isfinite(states).all():
        errors.append("states contain non-finite values")
    if dataset.node_ids.shape != (nodes,):
        errors.append(f"node_ids must have shape {(nodes,)}, got {dataset.node_ids.shape}")
    elif len(set(map(str, dataset.node_ids.tolist()))) != nodes:
        errors.append("node_ids are not unique")
    if not 1 <= dataset.train_end < dataset.val_end < timesteps:
        errors.append(
            f"invalid split boundaries train_end={dataset.train_end}, val_end={dataset.val_end}"
        )
    if dataset.distance.shape == (nodes, nodes):
        if not np.allclose(np.diag(dataset.distance), 0.0, atol=1e-7):
            errors.append("distance diagonal is not zero")
        if require_symmetric_distance and not np.allclose(
            dataset.distance, dataset.distance.T, atol=1e-6, rtol=1e-6
        ):
            errors.append("distance is not symmetric")
    if dataset.adjacency.shape == (nodes, nodes):
        if not np.allclose(np.diag(dataset.adjacency), 0.0, atol=1e-7):
            errors.append("adjacency diagonal is not zero")
        if np.any(dataset.adjacency < 0):
            errors.append("adjacency contains negative weights")
    if geometry_path is not None:
        errors.extend(validate_geometry(geometry_path, dataset.node_ids))
    return errors


def validate_geometry(path: str | Path, node_ids: np.ndarray) -> list[str]:
    geometry_path = Path(path)
    try:
        collection: dict[str, Any] = json.loads(geometry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"could not read geometry {geometry_path}: {exc}"]
    features = collection.get("features")
    if not isinstance(features, list):
        return ["geometry must be a GeoJSON FeatureCollection"]
    if len(features) != len(node_ids):
        return [f"geometry has {len(features)} features for {len(node_ids)} nodes"]
    feature_ids = [feature.get("properties", {}).get("cell_index") for feature in features]
    expected = node_ids.tolist()
    if list(map(str, feature_ids)) != list(map(str, expected)):
        return ["GeoJSON cell_index order does not match dataset node_ids"]
    return []
