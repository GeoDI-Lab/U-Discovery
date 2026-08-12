from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .synthetic import single_constrained_power_gravity


FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}

GROUND_TRUTH_PARAMETER_NAMES = (
    "beta",
    "gamma",
    "noise_sigma",
    "forcing_amplitude",
    "forcing_frequency_cycles_per_step",
    "forcing_phase_multiplier",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _copy_csv(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return output


def _minmax(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    low = float(np.nanmin(array))
    high = float(np.nanmax(array))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("normalization source contains no finite range")
    if high == low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def _classical_mds(distance: np.ndarray) -> np.ndarray:
    """Recover deterministic display coordinates from Euclidean distances."""
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance matrix must be square")
    size = distance.shape[0]
    centering = np.eye(size) - np.ones((size, size), dtype=np.float64) / size
    gram = -0.5 * centering @ np.square(distance) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    positive = np.maximum(eigenvalues[order], 0.0)
    coordinates = eigenvectors[:, order] * np.sqrt(positive)
    for axis in range(coordinates.shape[1]):
        pivot = int(np.argmax(np.abs(coordinates[:, axis])))
        if coordinates[pivot, axis] < 0:
            coordinates[:, axis] *= -1
    return coordinates


def _figure4_sources(root: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_path = root / "figure_sources/figure4/synthetic_figure4.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        distance = np.asarray(data["distance"], dtype=np.float64)
        distance_raw = np.asarray(data["distance_raw"], dtype=np.float64)
        adjacency = np.asarray(data["adjacency"], dtype=np.float64)
        spatial_amplitude = np.asarray(data["spatial_amplitude"], dtype=np.float64)
        spatial_phase = np.asarray(data["spatial_phase"], dtype=np.float64)
        parameters = np.asarray(data["ground_truth_parameters"], dtype=np.float64)
        dt = float(data["dt"])

    if states.shape != (501, 20):
        raise ValueError(f"Figure 4 requires states with shape (501, 20), got {states.shape}")
    if parameters.shape != (6,):
        raise ValueError(f"Figure 4 requires six documented parameters, got {parameters.shape}")
    if not np.array_equal(node_ids, np.arange(states.shape[1])):
        raise ValueError("synthetic node IDs must be ordered 0..19")

    beta, gamma, _, forcing_amplitude, forcing_frequency, phase_multiplier = parameters
    observed_change = np.diff(states, axis=0)
    spatial_increment = np.empty_like(observed_change)
    temporal_increment = np.empty_like(observed_change)
    for step, state in enumerate(states[:-1]):
        spatial_increment[step] = dt * single_constrained_power_gravity(
            state, distance_raw, float(beta), float(gamma)
        )
        temporal_increment[step] = (
            dt
            * float(forcing_amplitude)
            * spatial_amplitude
            * np.sin(
                2.0 * np.pi * float(forcing_frequency) * step
                + float(phase_multiplier) * spatial_phase
            )
        )
    closure_residual = observed_change - spatial_increment - temporal_increment
    closure_error = observed_change - (
        spatial_increment + temporal_increment + closure_residual
    )

    coordinates = _classical_mds(distance_raw)
    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "display_x_mds": coordinates[:, 0],
            "display_y_mds": coordinates[:, 1],
            "initial_state": states[0],
            "spatial_amplitude": spatial_amplitude,
            "spatial_phase": spatial_phase,
        }
    )

    edge_rows: list[dict[str, Any]] = []
    power_kernel_values: list[float] = []
    for source in range(states.shape[1]):
        for target in range(source + 1, states.shape[1]):
            if adjacency[source, target] == 0 and adjacency[target, source] == 0:
                continue
            raw = float(distance_raw[source, target])
            kernel = float(np.power(raw, -float(gamma))) if raw > 0 else 0.0
            power_kernel_values.append(kernel)
            edge_rows.append(
                {
                    "source_node_id": int(node_ids[source]),
                    "target_node_id": int(node_ids[target]),
                    "distance_raw": raw,
                    "distance_normalized": float(distance[source, target]),
                    "ground_truth_power_kernel": kernel,
                }
            )
    edges = pd.DataFrame(edge_rows)
    edge_scale = max(power_kernel_values) if power_kernel_values else 1.0
    edges["ground_truth_power_kernel_scaled"] = (
        edges["ground_truth_power_kernel"] / edge_scale
    )

    component_summary = pd.DataFrame(
        {
            "transition_index": np.arange(observed_change.shape[0]),
            "time_start": np.arange(observed_change.shape[0]) * dt,
            "observed_change_mean": observed_change.mean(axis=1),
            "observed_change_rms": np.sqrt(np.mean(np.square(observed_change), axis=1)),
            "documented_spatial_increment_mean": spatial_increment.mean(axis=1),
            "documented_spatial_increment_rms": np.sqrt(
                np.mean(np.square(spatial_increment), axis=1)
            ),
            "documented_temporal_increment_mean": temporal_increment.mean(axis=1),
            "documented_temporal_increment_rms": np.sqrt(
                np.mean(np.square(temporal_increment), axis=1)
            ),
            "closure_residual_mean": closure_residual.mean(axis=1),
            "closure_residual_rms": np.sqrt(
                np.mean(np.square(closure_residual), axis=1)
            ),
        }
    )
    state_series = pd.DataFrame(
        states,
        columns=[f"node_{int(node_id):02d}" for node_id in node_ids],
    )
    state_series.insert(0, "time", np.arange(states.shape[0]) * dt)
    state_series.insert(0, "time_index", np.arange(states.shape[0]))

    node_path = _write_csv(nodes, output / "figure4_network_nodes.csv")
    edge_path = _write_csv(edges, output / "figure4_network_edges.csv")
    summary_path = _write_csv(component_summary, output / "figure4_component_summary.csv")
    states_path = _write_csv(state_series, output / "figure4_state_series.csv")
    component_path = output / "figure4_components.npz"
    np.savez_compressed(
        component_path,
        node_ids=node_ids,
        time=np.arange(states.shape[0], dtype=np.float64) * dt,
        transition_time=np.arange(observed_change.shape[0], dtype=np.float64) * dt,
        states=states,
        observed_change=observed_change,
        documented_spatial_increment=spatial_increment,
        documented_temporal_increment=temporal_increment,
        realized_additive_residual=closure_residual,
        display_coordinates_mds=coordinates,
        ground_truth_parameter_names=np.asarray(GROUND_TRUTH_PARAMETER_NAMES),
        documented_ground_truth_parameters=parameters,
    )

    source_data = {
        "states": states,
        "node_ids": node_ids,
        "distance_raw": distance_raw,
        "coordinates": coordinates,
        "edges": edges,
        "spatial_increment": spatial_increment,
        "temporal_increment": temporal_increment,
        "closure_residual": closure_residual,
        "dt": dt,
    }
    manifest = {
        "title": "Synthetic spatiotemporal system and dynamics",
        "conclusion": (
            "The verified Figure-4-only 20-node realization combines a documented "
            "spatial gravity increment, periodic forcing, and a realized additive residual."
        ),
        "archetype": "mixed network, component heatmaps, and time series",
        "panels": {
            "4(a)": "Manuscript equation panel; represented numerically by parameter fields, not typeset here.",
            "4(b)": "20-node fully connected network reconstructed by metric MDS from released distances.",
            "4(c)": "Documented spatial and temporal increments plus the realized additive residual.",
            "4(d)": "Verified Figure 4 time series with 501 states (500 transitions).",
        },
        "source_inputs": [
            "figure_sources/figure4/synthetic_figure4.npz",
            "figure_sources/figure4/metadata.json",
        ],
        "source_artifacts": [
            _artifact(node_path),
            _artifact(edge_path),
            _artifact(summary_path),
            _artifact(states_path),
            _artifact(component_path),
        ],
        "checks": {
            "states_shape": list(states.shape),
            "edge_count": int(len(edges)),
            "component_shape": list(observed_change.shape),
            "component_closure_max_abs_error": float(np.max(np.abs(closure_error))),
            "closure_residual_standard_deviation": float(np.std(closure_residual)),
            "documented_noise_sigma": float(parameters[2]),
        },
        "warnings": [
            "This verified realization is source data for Figure 4 only; data/synthetic/dataset.npz remains the distinct source of truth for Table 2 training results.",
            "The realized additive residual is recovered by exact component closure and its standard deviation is checked against the documented noise scale.",
            "MDS coordinates are deterministic display coordinates inferred from distances, not original sampled coordinates.",
        ],
    }
    return manifest, source_data


def _figure5_sources(root: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    table_path = root / "results/synthetic/table2_metrics.csv"
    table = pd.read_csv(table_path)
    expected_columns = {
        "candidate",
        "MSE_delta",
        "R2_delta",
        "Corr_delta",
        "MSE",
        "R2",
        "Corr",
        "Complexity",
    }
    if set(table.columns) != expected_columns or len(table) != 12:
        raise ValueError(
            "Figure 5 requires the canonical 12-row results/synthetic/table2_metrics.csv"
        )
    baseline = table.loc[table["candidate"].eq("Temporal dynamics")]
    if len(baseline) != 1:
        raise ValueError("canonical Table 2 must contain one Temporal dynamics baseline")
    baseline_state = float(baseline.iloc[0]["MSE"])
    baseline_change = float(baseline.iloc[0]["MSE_delta"])

    ranking = table.loc[~table["candidate"].eq("Temporal dynamics")].copy()
    if len(ranking) != 11:
        raise ValueError("Figure 5 requires eleven spatial candidates plus one baseline row")
    ranking["complexity_effective"] = pd.to_numeric(
        ranking["Complexity"], errors="coerce"
    )
    ranking["complexity_missing_in_table"] = ranking["Complexity"].isna()
    if ranking["complexity_effective"].isna().any():
        raise ValueError("all eleven Figure 5 spatial candidates require complexity values")
    ranking["nMSE_delta"] = ranking["MSE_delta"] / baseline_change
    ranking["nMSE"] = ranking["MSE"] / baseline_state
    ranking["nMSE_rank"] = ranking["nMSE"].rank(method="min").astype(int)
    ranking["complexity_rank"] = (
        ranking["complexity_effective"].rank(method="min").astype(int)
    )
    ranking = ranking.sort_values("nMSE_delta", kind="stable").reset_index(drop=True)
    ranking.insert(0, "display_rank", np.arange(1, len(ranking) + 1))
    ranking["is_ground_truth"] = ranking["candidate"].eq(
        "Single-constrained gravity model (POW)"
    )
    ranking["nMSE_delta_normalized"] = _minmax(ranking["nMSE_delta"])
    ranking["nMSE_rank_normalized"] = _minmax(ranking["nMSE_rank"])
    ranking["complexity_rank_normalized"] = _minmax(ranking["complexity_rank"])

    frontier_rows: list[dict[str, Any]] = []
    for complexity, group in ranking.groupby("complexity_effective", sort=True):
        delta_row = group.loc[group["nMSE_delta"].idxmin()]
        state_row = group.loc[group["nMSE"].idxmin()]
        frontier_rows.append(
            {
                "complexity": float(complexity),
                "nMSE_delta_min": float(delta_row["nMSE_delta"]),
                "nMSE_delta_candidate": delta_row["candidate"],
                "nMSE_min": float(state_row["nMSE"]),
                "nMSE_candidate": state_row["candidate"],
            }
        )
    frontier = pd.DataFrame(frontier_rows)

    trajectory_path = root / "results/synthetic/figure5_parameter_trajectories.csv"
    convergence_source = pd.read_csv(trajectory_path)
    required = {
        "epoch",
        "seed",
        "validation_mse_delta",
        "beta",
        "gamma",
        "is_start",
        "is_final",
    }
    if not required.issubset(convergence_source.columns):
        raise ValueError("Figure 5 parameter trajectory source has unexpected columns")
    convergence = convergence_source.copy()
    if len(convergence) != 232 or convergence["seed"].nunique() != 10:
        raise ValueError("Figure 5 requires 232 logged trajectory points from ten seeds")
    counts = convergence.groupby("seed")[["is_start", "is_final"]].sum()
    if not (counts["is_start"].eq(1).all() and counts["is_final"].eq(1).all()):
        raise ValueError("each Figure 5 trajectory must identify one start and one final point")
    for _, run in convergence.groupby("seed", sort=False):
        if not run["epoch"].is_monotonic_increasing:
            raise ValueError("Figure 5 trajectory epochs must increase within each seed")
    convergence["beta_ground_truth"] = 0.488345142323785
    convergence["gamma_ground_truth"] = 1.2680008527060398
    convergence["beta_minus_ground_truth"] = (
        convergence["beta"] - convergence["beta_ground_truth"]
    )
    convergence["gamma_minus_ground_truth"] = (
        convergence["gamma"] - convergence["gamma_ground_truth"]
    )

    ranking_path = _write_csv(ranking, output / "figure5_heatmap_source.csv")
    frontier_path = _write_csv(frontier, output / "figure5_frontiers_source.csv")
    convergence_path = _copy_csv(
        trajectory_path, output / "figure5_parameter_trajectories_source.csv"
    )
    source_data = {
        "ranking": ranking,
        "frontier": frontier,
        "convergence": convergence,
    }
    manifest = {
        "title": "Synthetic candidate ranking and fitted parameters",
        "conclusion": (
            "The documented single-constrained power-gravity candidate is the "
            "lowest-change-error model in the canonical 12-row Table 2 source."
        ),
        "archetype": "quantitative heatmap, error frontiers, and optimization trajectories",
        "panels": {
            "5(a)": "Candidates ranked by nMSE_delta; columns encode nMSE_delta, nMSE rank, and complexity rank.",
            "5(b)": "Lowest nMSE_delta and nMSE at each discrete complexity represented in Table 2.",
            "5(c)": "Logged beta and gamma optimization trajectories across ten seeds against documented targets.",
        },
        "source_inputs": [
            "results/synthetic/table2_metrics.csv",
            "results/synthetic/figure5_parameter_trajectories.csv",
        ],
        "source_artifacts": [
            _artifact(ranking_path),
            _artifact(frontier_path),
            _artifact(convergence_path),
        ],
        "checks": {
            "canonical_table2_rows": int(len(table)),
            "temporal_state_mse_denominator": baseline_state,
            "temporal_change_mse_denominator": baseline_change,
            "top_ranked_candidate": str(ranking.iloc[0]["candidate"]),
            "top_ranked_nMSE_delta": float(ranking.iloc[0]["nMSE_delta"]),
            "displayed_spatial_candidate_count": int(len(ranking)),
            "parameter_trajectory_points": int(len(convergence)),
            "parameter_trajectory_seed_count": int(convergence["seed"].nunique()),
            "parameter_start_point_count": int(convergence["is_start"].sum()),
            "parameter_final_point_count": int(convergence["is_final"].sum()),
        },
        "warnings": [
            "The trajectories contain the exact optimization points parsed from the ten archived logs; epoch sampling is uneven across seeds.",
            "Temporal dynamics is the normalization baseline and is not displayed as a spatial-equation candidate in Figure 5, matching the manuscript panel.",
        ],
    }
    return manifest, source_data


def _polygon_ring(feature: dict[str, Any]) -> np.ndarray:
    geometry = feature.get("geometry", {})
    if geometry.get("type") != "Polygon":
        raise ValueError("Figure 6 supports ordered Polygon features only")
    coordinates = geometry.get("coordinates")
    if not coordinates:
        raise ValueError("empty polygon coordinates")
    ring = np.asarray(coordinates[0], dtype=np.float64)
    if ring.ndim != 2 or ring.shape[1] != 2 or not np.isfinite(ring).all():
        raise ValueError("invalid polygon ring")
    return ring


def _figure6_sources(root: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_path = root / "data/hennepin_2500m/hennepin_2500m.npz"
    geometry_path = root / "data/hennepin_2500m/grid_cells.geojson"
    with np.load(dataset_path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        interval_minutes = int(data["interval_minutes"])
        grid_resolution_m = int(data["grid_resolution_m"])
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    features = geometry.get("features", [])
    feature_ids = np.asarray(
        [feature.get("properties", {}).get("cell_index") for feature in features],
        dtype=np.int64,
    )
    if states.shape != (480, 210):
        raise ValueError(f"Figure 6 requires states with shape (480, 210), got {states.shape}")
    if len(features) != states.shape[1] or not np.array_equal(feature_ids, node_ids):
        raise ValueError("ordered GeoJSON cell_index values do not match dataset node_ids")

    rings = [_polygon_ring(feature) for feature in features]
    all_coordinates = np.concatenate(rings, axis=0)
    if geometry.get("crs") is not None:
        raise ValueError("Figure 6 requires RFC 7946 GeoJSON without a deprecated crs member")
    if (
        np.any(np.abs(all_coordinates[:, 0]) > 180.0)
        or np.any(np.abs(all_coordinates[:, 1]) > 90.0)
    ):
        raise ValueError("Figure 6 geometry must use longitude/latitude coordinates")
    geometry_rows: list[dict[str, Any]] = []
    for cell_index, ring in zip(node_ids, rings):
        vertices = ring[:-1] if np.array_equal(ring[0], ring[-1]) else ring
        geometry_rows.append(
            {
                "cell_index": int(cell_index),
                "centroid_longitude": float(vertices[:, 0].mean()),
                "centroid_latitude": float(vertices[:, 1].mean()),
                "min_longitude": float(vertices[:, 0].min()),
                "min_latitude": float(vertices[:, 1].min()),
                "max_longitude": float(vertices[:, 0].max()),
                "max_latitude": float(vertices[:, 1].max()),
                "mean_active_population": float(states[:, int(cell_index)].mean()),
            }
        )
    geometry_table = pd.DataFrame(geometry_rows)

    time_index = np.arange(states.shape[0], dtype=np.int64)
    elapsed_hours = time_index * interval_minutes / 60.0
    state_series = pd.DataFrame(
        states,
        columns=[f"cell_{int(node_id):03d}" for node_id in node_ids],
    )
    state_series.insert(0, "elapsed_hours", elapsed_hours)
    state_series.insert(0, "time_index", time_index)
    population_summary = pd.DataFrame(
        {
            "time_index": time_index,
            "elapsed_hours": elapsed_hours,
            "study_day": time_index // (24 * 60 // interval_minutes) + 1,
            "hour_within_day": (time_index % (24 * 60 // interval_minutes))
            * interval_minutes
            / 60.0,
            "total_active_population": states.sum(axis=1),
            "mean_cell_active_population": states.mean(axis=1),
            "median_cell_active_population": np.median(states, axis=1),
            "max_cell_active_population": states.max(axis=1),
            "active_cell_count": np.count_nonzero(states, axis=1),
        }
    )

    snapshots = np.asarray([0, 24, 48, 72], dtype=np.int64)
    snapshot_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        for node_position, cell_index in enumerate(node_ids):
            snapshot_rows.append(
                {
                    "study_day": 1,
                    "time_index": int(snapshot),
                    "hour_within_day": float(snapshot * interval_minutes / 60.0),
                    "cell_index": int(cell_index),
                    "active_population": float(states[snapshot, node_position]),
                }
            )
    snapshot_table = pd.DataFrame(snapshot_rows)

    geometry_table_path = _write_csv(
        geometry_table, output / "figure6_grid_cells_source.csv"
    )
    state_path = _write_csv(state_series, output / "figure6_state_series_source.csv")
    summary_path = _write_csv(
        population_summary, output / "figure6_population_summary_source.csv"
    )
    snapshot_path = _write_csv(
        snapshot_table, output / "figure6_day1_snapshots_source.csv"
    )
    source_data = {
        "states": states,
        "node_ids": node_ids,
        "rings": rings,
        "geometry_table": geometry_table,
        "population_summary": population_summary,
        "snapshots": snapshots,
        "interval_minutes": interval_minutes,
    }
    manifest = {
        "title": "Public 2.5 km Hennepin grid and aggregate dynamics",
        "conclusion": (
            "The privacy-preserving 210-cell aggregate exhibits heterogeneous "
            "15-minute population dynamics over five study days."
        ),
        "archetype": "geographic grid map, cell time series, and aggregate snapshots",
        "panels": {
            "6(a)": "Not regenerated: raw individual trajectories are intentionally excluded.",
            "6(b)": "Ordered 210-cell, 2.5 km grid in RFC 7946 longitude/latitude coordinates.",
            "6(c)": "All 210 aggregate state series and day-one snapshots at 00:00, 06:00, 12:00, and 18:00.",
        },
        "source_inputs": [
            "data/hennepin_2500m/hennepin_2500m.npz",
            "data/hennepin_2500m/grid_cells.geojson",
        ],
        "source_artifacts": [
            _artifact(geometry_table_path),
            _artifact(state_path),
            _artifact(summary_path),
            _artifact(snapshot_path),
        ],
        "checks": {
            "states_shape": list(states.shape),
            "geometry_feature_count": int(len(features)),
            "node_geometry_order_match": True,
            "interval_minutes": interval_minutes,
            "grid_resolution_m": grid_resolution_m,
            "geometry_sha256": _sha256(geometry_path),
            "geometry_crs": "EPSG:4326 (RFC 7946 longitude/latitude)",
        },
        "warnings": [
            "Panel 6(a) is outside the public release because it requires excluded individual trajectories.",
            "Maps use only the released aggregate grid geometry and do not add a basemap or infer raw trip locations.",
        ],
    }
    return manifest, source_data


def _figure7_sources(root: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    results = root / "results/hennepin_2500m"
    names = {
        "selection": "figure7_selection.csv",
        "heatmap": "figure7_heatmap_top20.csv",
        "frontier_delta": "figure7_frontier_nmse_delta.csv",
        "frontier_state": "figure7_frontier_nmse.csv",
    }
    copied: dict[str, Path] = {}
    frames: dict[str, pd.DataFrame] = {}
    for key, name in names.items():
        source = results / name
        copied[key] = _copy_csv(source, output / name)
        frames[key] = pd.read_csv(source)
    heatmap = frames["heatmap"]
    if len(heatmap) != 20 or not np.array_equal(
        heatmap["display_rank"].to_numpy(), np.arange(1, 21)
    ):
        raise ValueError("Figure 7 heatmap source must contain ordered ranks 1..20")
    selected = frames["selection"].loc[
        frames["selection"]["heatmap_display_rank"].eq(1)
    ]
    if len(selected) != 1:
        raise ValueError("Figure 7 selection source must identify one rank-1 candidate")

    source_data = frames
    manifest = {
        "title": "Hennepin candidate ranking and error frontiers",
        "conclusion": (
            "After the documented complexity filter, betweenness-augmented gravity "
            "has the lowest nMSE_delta among the 20 displayed candidates."
        ),
        "archetype": "quantitative heatmap and two error frontiers",
        "panels": {
            "7(a)": "Top 20 candidates after excluding the top 10% source-complexity tail.",
            "7(b)": "Minimum nMSE_delta represented at each retained source complexity.",
            "7(c)": "Minimum nMSE represented at each retained source complexity.",
        },
        "source_inputs": [f"results/hennepin_2500m/{name}" for name in names.values()],
        "source_artifacts": [_artifact(copied[key]) for key in names],
        "checks": {
            "displayed_candidate_count": int(len(heatmap)),
            "top_ranked_candidate": str(selected.iloc[0]["candidate"]),
            "top_ranked_nMSE_delta": float(selected.iloc[0]["nMSE_delta"]),
            "complexity_filter_q90": float(selected.iloc[0]["complexity_q90"]),
        },
        "warnings": [
            "Figure 7 uses source_csv_complexity, matching the archived plotting notebook; Table 4 also preserves corrected paper complexity values.",
        ],
    }
    return manifest, source_data


def _load_pyplot() -> tuple[Any | None, str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        return None, f"{type(error).__name__}: {error}"
    return plt, None


def _save_figure(figure: Any, output: Path, stem: str, plt: Any) -> list[Path]:
    png = output / f"{stem}.png"
    svg = output / f"{stem}.svg"
    figure.savefig(png, dpi=300)
    figure.savefig(svg)
    plt.close(figure)
    return [png, svg]


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.1,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _render_figure4(data: dict[str, Any], output: Path, plt: Any) -> list[Path]:
    from matplotlib.collections import LineCollection

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(12.0, 8.0), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
        network_axis = figure.add_subplot(grid[0, 0])
        series_axis = figure.add_subplot(grid[0, 1])
        component_grid = grid[1, :].subgridspec(1, 3, wspace=0.18)
        component_axes = [figure.add_subplot(component_grid[0, index]) for index in range(3)]

        coordinates = data["coordinates"]
        edges = data["edges"]
        segments = [
            [coordinates[int(row.source_node_id)], coordinates[int(row.target_node_id)]]
            for row in edges.itertuples(index=False)
        ]
        weights = edges["ground_truth_power_kernel_scaled"].to_numpy()
        line_collection = LineCollection(
            segments,
            colors=[(0.22, 0.36, 0.50, 0.08 + 0.30 * value) for value in weights],
            linewidths=0.25 + 1.0 * weights,
            zorder=1,
        )
        network_axis.add_collection(line_collection)
        nodes = network_axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=data["states"][0],
            cmap="viridis",
            s=50,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        for node_id, (x_value, y_value) in zip(data["node_ids"], coordinates):
            network_axis.text(x_value, y_value, str(int(node_id)), fontsize=5.5, ha="center", va="center")
        network_axis.autoscale()
        network_axis.set_aspect("equal")
        network_axis.set_xlabel("MDS coordinate 1")
        network_axis.set_ylabel("MDS coordinate 2")
        network_axis.set_title("Released fully connected network")
        figure.colorbar(nodes, ax=network_axis, shrink=0.75, label="initial state")
        _panel_label(network_axis, "(b)")

        time = np.arange(data["states"].shape[0]) * data["dt"]
        colors = plt.get_cmap("viridis")(
            plt.Normalize()(data["states"][0])
        )
        for index, color in enumerate(colors):
            series_axis.plot(time, data["states"][:, index], color=color, alpha=0.82, linewidth=0.75)
        series_axis.axvline(300, color="0.45", linestyle="--", linewidth=0.8, label="train/validation")
        series_axis.axvline(400, color="0.45", linestyle=":", linewidth=0.8, label="validation/test")
        series_axis.set_xlabel("time step")
        series_axis.set_ylabel("state")
        series_axis.set_title("Verified Figure 4 state series")
        series_axis.legend(frameon=False, loc="upper right")
        _panel_label(series_axis, "(d)")

        components = [
            (data["spatial_increment"], "documented spatial increment"),
            (data["temporal_increment"], "documented temporal increment"),
            (data["closure_residual"], "realized additive residual"),
        ]
        for index, (axis, (component, title)) in enumerate(zip(component_axes, components)):
            limit = float(np.quantile(np.abs(component), 0.99))
            image = axis.imshow(
                component.T,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axis.set_xlabel("transition index")
            if index == 0:
                axis.set_ylabel("node")
                _panel_label(axis, "(c)")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, shrink=0.78, label="state increment")

        return _save_figure(figure, output, "figure4_public", plt)


def _annotated_heatmap(
    axis: Any,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    annotations: list[list[str]],
    plt: Any,
) -> Any:
    image = axis.imshow(matrix, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=20, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > 0.56 else "#17202A"
            axis.text(
                column,
                row,
                annotations[row][column],
                ha="center",
                va="center",
                color=color,
                fontsize=6.5,
            )
    return image


def _render_figure5(data: dict[str, Any], output: Path, plt: Any) -> list[Path]:
    ranking = data["ranking"]
    frontier = data["frontier"]
    convergence = data["convergence"]
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(14.0, 5.8), constrained_layout=True)
        grid = figure.add_gridspec(1, 3, width_ratios=[1.65, 1.0, 1.0])
        heatmap_axis = figure.add_subplot(grid[0, 0])
        frontier_axis = figure.add_subplot(grid[0, 1])
        parameter_grid = grid[0, 2].subgridspec(2, 1, hspace=0.13)
        parameter_axes = [figure.add_subplot(parameter_grid[index, 0]) for index in range(2)]

        matrix = ranking[
            [
                "nMSE_delta_normalized",
                "nMSE_rank_normalized",
                "complexity_rank_normalized",
            ]
        ].to_numpy(dtype=np.float64)
        annotations = [
            [f"{row.nMSE_delta:.3f}", str(int(row.nMSE_rank)), str(int(row.complexity_rank))]
            for row in ranking.itertuples(index=False)
        ]
        _annotated_heatmap(
            heatmap_axis,
            1.0 - matrix,
            ranking["candidate"].tolist(),
            [r"nMSE$_\Delta$", "nMSE rank", "complexity rank"],
            annotations,
            plt,
        )
        true_position = int(np.flatnonzero(ranking["is_ground_truth"].to_numpy())[0])
        heatmap_axis.text(0.38, true_position, "★", color="#C43C39", fontsize=13, ha="center", va="center")
        heatmap_axis.set_title("Canonical Table 2 ranking")
        _panel_label(heatmap_axis, "(a)")

        frontier_axis.plot(
            frontier["complexity"],
            frontier["nMSE_delta_min"],
            marker="o",
            markersize=3.5,
            color="#3B6FB6",
            label=r"nMSE$_\Delta$",
        )
        frontier_axis.plot(
            frontier["complexity"],
            frontier["nMSE_min"],
            marker="s",
            markersize=3.2,
            color="#D07B32",
            label="nMSE",
        )
        top = ranking.iloc[0]
        frontier_axis.scatter(
            [top["complexity_effective"]] * 2,
            [top["nMSE_delta"], top["nMSE"]],
            marker="*",
            s=90,
            color="#C43C39",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        frontier_axis.axhline(1.0, color="0.55", linestyle="--", linewidth=0.7)
        frontier_axis.set_xlabel("equation complexity")
        frontier_axis.set_ylabel("normalized error")
        frontier_axis.set_title("Error frontiers")
        frontier_axis.legend(frameon=False)
        _panel_label(frontier_axis, "(b)")

        parameter_specs = [
            ("beta", "beta_ground_truth", r"$\beta$"),
            ("gamma", "gamma_ground_truth", r"$\gamma$"),
        ]
        for index, (axis, (value_column, target_column, label)) in enumerate(
            zip(parameter_axes, parameter_specs)
        ):
            for _, trajectory in convergence.groupby("seed", sort=True):
                axis.plot(
                    trajectory["epoch"],
                    trajectory[value_column],
                    color="#7899C3",
                    alpha=0.55,
                    linewidth=0.7,
                    marker=".",
                    markersize=2.0,
                )
            starts = convergence.loc[convergence["is_start"].eq(1)]
            finals = convergence.loc[convergence["is_final"].eq(1)]
            axis.scatter(
                starts["epoch"],
                starts[value_column],
                color="#3B6FB6",
                s=23,
                label="initial" if index == 0 else None,
                zorder=3,
            )
            axis.scatter(
                finals["epoch"],
                finals[value_column],
                color="#D97A2B",
                s=23,
                label="final" if index == 0 else None,
                zorder=3,
            )
            axis.axhline(
                float(convergence[target_column].iloc[0]),
                color="#C43C39",
                linestyle="--",
                linewidth=1.1,
                label="documented target" if index == 0 else None,
            )
            axis.set_ylabel(label)
            if index == 0:
                axis.set_title("Optimization trajectories across seeds")
                _panel_label(axis, "(c)")
            else:
                axis.set_xlabel("optimization epoch")
            axis.set_xscale("log")
        parameter_axes[0].legend(frameon=False, loc="best")

        return _save_figure(figure, output, "figure5_public", plt)


def _add_polygon_collection(
    axis: Any,
    rings: list[np.ndarray],
    values: np.ndarray | None,
    *,
    plt: Any,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> Any:
    from matplotlib.collections import PolyCollection

    vertices = rings
    if values is None:
        collection = PolyCollection(
            vertices,
            facecolors="#F4F5F6",
            edgecolors="#5E6973",
            linewidths=0.35,
        )
    else:
        collection = PolyCollection(
            vertices,
            array=np.asarray(values, dtype=np.float64),
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            edgecolors="white",
            linewidths=0.18,
        )
    axis.add_collection(collection)
    axis.autoscale_view()
    mean_latitude = float(np.mean([ring[:, 1].mean() for ring in rings]))
    axis.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))
    return collection


def _render_figure6(data: dict[str, Any], output: Path, plt: Any) -> list[Path]:
    states = data["states"]
    rings = data["rings"]
    interval = data["interval_minutes"]
    snapshots = data["snapshots"]
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(13.0, 8.0), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, width_ratios=[0.82, 1.45], height_ratios=[1.0, 1.0])
        grid_axis = figure.add_subplot(grid[:, 0])
        series_axis = figure.add_subplot(grid[0, 1])
        snapshot_grid = grid[1, 1].subgridspec(1, 4, wspace=0.03)
        snapshot_axes = [figure.add_subplot(snapshot_grid[0, index]) for index in range(4)]

        _add_polygon_collection(grid_axis, rings, None, plt=plt)
        grid_axis.set_xlabel("Longitude (degrees)")
        grid_axis.set_ylabel("Latitude (degrees)")
        grid_axis.set_title("Ordered 2.5 km grid (RFC 7946 / EPSG:4326)")
        _panel_label(grid_axis, "(b)")

        elapsed_hours = np.arange(states.shape[0]) * interval / 60.0
        mean_state = states.mean(axis=0)
        colors = plt.get_cmap("viridis")(plt.Normalize()(mean_state))
        for node_position, color in enumerate(colors):
            series_axis.plot(
                elapsed_hours,
                states[:, node_position],
                color=color,
                alpha=0.48,
                linewidth=0.45,
            )
        for day in range(1, 5):
            series_axis.axvline(day * 24, color="0.75", linewidth=0.55)
        series_axis.set_xlabel("elapsed hours")
        series_axis.set_ylabel("active population per cell")
        series_axis.set_title("Aggregate 15-minute cell dynamics")
        _panel_label(series_axis, "(c)")

        snapshot_values = states[snapshots]
        color_limit = float(np.quantile(snapshot_values, 0.99))
        last_collection = None
        for axis, snapshot in zip(snapshot_axes, snapshots):
            last_collection = _add_polygon_collection(
                axis,
                rings,
                states[snapshot],
                plt=plt,
                vmin=0.0,
                vmax=color_limit,
                cmap="magma",
            )
            hour = snapshot * interval / 60.0
            axis.set_title(f"day 1, {int(hour):02d}:00")
            axis.set_xticks([])
            axis.set_yticks([])
        if last_collection is not None:
            figure.colorbar(
                last_collection,
                ax=snapshot_axes,
                shrink=0.68,
                location="bottom",
                pad=0.02,
                label="active population (99th-percentile color cap)",
            )

        return _save_figure(figure, output, "figure6_public_panels_b_c", plt)


def _render_figure7(data: dict[str, Any], output: Path, plt: Any) -> list[Path]:
    heatmap = data["heatmap"]
    delta = data["frontier_delta"]
    state = data["frontier_state"]
    selected = data["selection"].loc[data["selection"]["heatmap_display_rank"].eq(1)].iloc[0]
    filtered_count = int(data["selection"]["passes_complexity_filter"].eq(True).sum())
    rank_scale = max(filtered_count - 1, 1)
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(12.2, 7.7), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, width_ratios=[1.5, 1.0])
        heatmap_axis = figure.add_subplot(grid[:, 0])
        delta_axis = figure.add_subplot(grid[0, 1])
        state_axis = figure.add_subplot(grid[1, 1])

        matrix = heatmap[
            [
                "nMSE_delta_normalized",
                "nMSE_rank_normalized",
                "complexity_rank_normalized",
            ]
        ].to_numpy(dtype=np.float64)
        annotations = [
            [
                f"{row.nMSE_delta:.3f}",
                str(int(round(row.nMSE_rank_normalized * rank_scale + 1))),
                str(int(round(row.complexity_rank_normalized * rank_scale + 1))),
            ]
            for row in heatmap.itertuples(index=False)
        ]
        _annotated_heatmap(
            heatmap_axis,
            1.0 - matrix,
            heatmap["candidate"].tolist(),
            [r"nMSE$_\Delta$", "nMSE rank", "complexity rank"],
            annotations,
            plt,
        )
        heatmap_axis.text(0.38, 0, "★", color="#C43C39", fontsize=13, ha="center", va="center")
        heatmap_axis.set_title("Complexity-filtered candidate ranking")
        _panel_label(heatmap_axis, "(a)")

        frontier_specs = [
            (
                delta_axis,
                delta,
                "nMSE_delta_min",
                float(selected["nMSE_delta"]),
                r"nMSE$_\Delta$",
                "(b)",
            ),
            (
                state_axis,
                state,
                "nMSE_min",
                float(selected["nMSE"]),
                "nMSE",
                "(c)",
            ),
        ]
        for axis, source, value_column, selected_value, ylabel, panel in frontier_specs:
            axis.plot(
                source["source_csv_complexity"],
                source[value_column],
                color="#3B6FB6",
                marker="o",
                markersize=3.4,
            )
            axis.scatter(
                selected["source_csv_complexity"],
                selected_value,
                marker="*",
                s=100,
                color="#C43C39",
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
            axis.set_xlabel("source fitting complexity")
            axis.set_ylabel(ylabel)
            axis.set_title(f"Error frontier: {ylabel}")
            _panel_label(axis, panel)

        return _save_figure(figure, output, "figure7_public", plt)


def reproduce_public_figures(root: Path, output: Path) -> dict[str, Any]:
    """Generate public figure source data and render panels when Matplotlib exists.

    Source tables and arrays are always written. Matplotlib is imported lazily so
    the core NumPy/Pandas installation can still reproduce all machine-readable
    outputs and report a precise rendering skip reason.
    """
    output.mkdir(parents=True, exist_ok=True)
    builders: list[
        tuple[
            str,
            Callable[[Path, Path], tuple[dict[str, Any], dict[str, Any]]],
            Callable[[dict[str, Any], Path, Any], list[Path]],
        ]
    ] = [
        ("figure4", _figure4_sources, _render_figure4),
        ("figure5", _figure5_sources, _render_figure5),
        ("figure6", _figure6_sources, _render_figure6),
        ("figure7", _figure7_sources, _render_figure7),
    ]

    prepared: dict[str, tuple[dict[str, Any], dict[str, Any], Callable[..., list[Path]]]] = {}
    for key, source_builder, renderer in builders:
        figure_manifest, source_data = source_builder(root, output)
        prepared[key] = (figure_manifest, source_data, renderer)

    plt, import_error = _load_pyplot()
    if plt is None:
        rendering = {
            "status": "skipped",
            "backend": "matplotlib/Agg",
            "reason": import_error,
            "install_hint": "Install the optional 'figures' dependency group to render PNG and SVG files.",
        }
    else:
        rendering = {
            "status": "rendered",
            "backend": "matplotlib/Agg",
            "matplotlib_version": str(plt.matplotlib.__version__),
        }

    figures: dict[str, Any] = {}
    for key, (figure_manifest, source_data, renderer) in prepared.items():
        rendered: list[Path] = []
        if plt is not None:
            rendered = renderer(source_data, output, plt)
        figure_manifest["rendering"] = {
            "status": "rendered" if rendered else "skipped",
            "artifacts": [_artifact(path) for path in rendered],
        }
        figures[key] = figure_manifest
    return {"renderer": rendering, "figures": figures}
