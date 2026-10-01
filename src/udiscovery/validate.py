from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_dataset, validate_dataset
from .metrics import state_and_delta_metrics
from .paths import repository_root


EXPECTED_TABLE2 = [
    ("Single-constrained gravity model (POW)", 5.2345, 0.7293, 0.8083, 3.2497, 0.9994, 0.9994, 17),
    ("Single-constrained gravity model (EXP)", 9.6106, 0.5030, 0.6837, 4.5115, 0.9992, 0.9992, 21),
    ("Classic gravity model (POW)", 9.6728, 0.4997, 0.6832, 4.3428, 0.9992, 0.9992, 22),
    ("Saturation-limited diffusion", 9.6912, 0.4988, 0.6838, 4.2980, 0.9992, 0.9992, 27),
    ("Classic gravity model (EXP)", 9.7263, 0.4970, 0.6824, 4.3142, 0.9992, 0.9992, 24),
    ("Distance-weighted diffusion", 9.7443, 0.4960, 0.6817, 4.3209, 0.9992, 0.9992, 10),
    ("Rank-based diffusion", 9.7628, 0.4951, 0.6816, 4.2932, 0.9992, 0.9992, 16),
    ("Local–global mixed diffusion", 9.7704, 0.4947, 0.6816, 4.2757, 0.9992, 0.9992, 25),
    ("KNN diffusion", 9.7851, 0.4939, 0.6811, 4.2700, 0.9992, 0.9992, 15),
    ("Temporal dynamics", 9.8226, 0.4920, 0.6803, 4.2350, 0.9992, 0.9992, None),
    ("Accessibility-driven drift", 9.8226, 0.4920, 0.6803, 4.2417, 0.9992, 0.9992, 23),
    ("Radiation model", 9.8851, 0.4888, 0.6790, 4.6677, 0.9992, 0.9992, 13),
]

EXPECTED_TABLE4 = [
    ("Explore–return diffusion", 0.6262, 6.9246, 47),
    ("Betweenness-augmented gravity", 0.6479, 1.1145, 26),
    ("Single-constrained gravity model (POW)", 0.6651, 1.0622, 17),
    ("Single-constrained gravity model (EXP)", 0.6718, 1.0478, 21),
    ("Classic gravity model (EXP)", 0.6947, 1.1148, 24),
    ("Edge-betweenness weighted mixed diffusion", 0.7415, 1.0332, 14),
    ("KNN two-band diffusion", 0.7436, 1.0145, 18),
    ("Two-scale piecewise diffusion", 0.7465, 1.0932, 26),
    ("Corridor-enhanced diffusion", 0.7551, 1.0468, 18),
    ("Dual-layer overlap diffusion", 0.7567, 1.0998, 20),
    ("State-adaptive diffusion", 0.7588, 1.0856, 29),
    ("Span- and cluster-aware diffusion", 0.7599, 1.0615, 30),
    ("Distance-weighted diffusion", 0.7668, 1.0789, 11),
    ("Span-adaptive diffusion", 0.7689, 1.0574, 22),
    ("Rank-based diffusion", 0.7689, 1.0748, 16),
    ("Classic gravity model (POW)", 0.7732, 1.0819, 22),
    ("Saturation-limited diffusion", 0.7755, 1.0404, 27),
    ("Cluster-constrained diffusion", 0.7769, 1.0105, 18),
    ("Local–global mixed diffusion", 0.7991, 1.0485, 25),
    ("Betweenness-modulated diffusion", 0.8039, 1.0552, 16),
    ("KNN-limited diffusion", 0.8213, 1.0654, 16),
    ("KNN diffusion", 0.8634, 1.0439, 15),
    ("Rank-truncated + tail choice diffusion", 0.9088, 0.9850, 49),
    ("Rank-choice gravity", 0.9397, 2.1683, 29),
    ("Accessibility-driven drift", 0.9504, 1.0089, 23),
    ("Radiation model", 1.0386, 1.0586, 13),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(root: str | Path) -> tuple[Path, ...]:
    """Return the deterministic inventory of portable release files.

    Local Git metadata, virtual environments, Python/test caches, packaging
    output, ``*.egg-info`` directories, and the root ``outputs/`` directory
    are outside the release. So are macOS ``.DS_Store``/``._*`` metadata,
    bytecode, and ``*.log`` files. Directory exclusions are pruned before
    traversal, so an installed environment is never recursively inspected.

    This is deliberately independent of .gitignore: unexpected source/data
    files, unsafe checkpoints, and real ``.env`` files remain in scope. Only
    ``.env.example`` is permitted as an environment template by the hygiene
    validator. The same inventory defines MANIFEST.sha256 and every release
    content scan; local installation and pipeline runs do not alter it.
    """
    release_root = Path(root).resolve()
    ignored_directories = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist",
    }
    files: list[Path] = []
    for directory, subdirectories, filenames in os.walk(release_root, followlinks=False):
        parent = Path(directory)
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name not in ignored_directories
            and not name.endswith(".egg-info")
            and name != ".DS_Store"
            and not name.startswith("._")
            and not (parent == release_root and name == "outputs")
        )
        for name in filenames:
            path = parent / name
            if name == ".DS_Store" or name.startswith("._"):
                continue
            if path.suffix.lower() in {".pyc", ".pyo", ".log"}:
                continue
            if path.is_file():
                files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(release_root).as_posix()))


def _validate_table2(root: Path) -> list[str]:
    errors: list[str] = []
    table = pd.read_csv(root / "results/synthetic/table2_metrics.csv")
    if len(table) != len(EXPECTED_TABLE2):
        errors.append(f"Table 2 must contain 12 rows, found {len(table)}")
        return errors
    columns = ("MSE_delta", "R2_delta", "Corr_delta", "MSE", "R2", "Corr")
    for position, expected in enumerate(EXPECTED_TABLE2):
        name, *values, complexity = expected
        row = table.iloc[position]
        if row["candidate"] != name:
            errors.append(
                f"Table 2 row {position + 1} candidate mismatch: {row['candidate']!r} != {name!r}"
            )
        for column, value in zip(columns, values):
            if abs(float(row[column]) - float(value)) > 1e-4:
                errors.append(f"Table 2 {column} mismatch for {name}")
        actual_complexity = row["Complexity"]
        if complexity is None:
            if not pd.isna(actual_complexity):
                errors.append(f"Table 2 temporal baseline complexity must be blank")
        elif int(actual_complexity) != complexity:
            errors.append(f"Table 2 complexity mismatch for {name}")

    with np.load(root / "results/synthetic/top1_reconstruction.npz", allow_pickle=False) as source:
        metrics = state_and_delta_metrics(source["ground_truth_state"], source["prediction"])
    top = table.iloc[0]
    for metric, column in (
        ("mse", "MSE"), ("r2", "R2"), ("correlation", "Corr"),
        ("mse_delta", "MSE_delta"), ("r2_delta", "R2_delta"),
        ("correlation_delta", "Corr_delta"),
    ):
        if abs(metrics[metric] - float(top[column])) > 1e-4:
            errors.append(f"selected synthetic reconstruction does not reproduce Table 2 {column}")
    return errors


def _validate_table4(root: Path) -> list[str]:
    errors: list[str] = []
    table = pd.read_csv(root / "results/hennepin_2500m/table4_metrics.csv")
    if len(table) != 26:
        errors.append(f"Table 4 must contain 26 candidates, found {len(table)}")
        return errors
    for position, (name, delta, state, complexity) in enumerate(EXPECTED_TABLE4):
        row = table.iloc[position]
        if int(row["rank"]) != position + 1 or row["candidate"] != name:
            errors.append(f"Table 4 row/rank mismatch at {position + 1}: {row['candidate']!r}")
        if abs(float(row["nMSE_delta"]) - delta) > 1e-4:
            errors.append(f"Table 4 nMSE_delta mismatch for {name}")
        if abs(float(row["nMSE"]) - state) > 1e-4:
            errors.append(f"Table 4 nMSE mismatch for {name}")
        if int(row["complexity"]) != complexity:
            errors.append(f"Table 4 complexity mismatch for {name}")

    metadata = json.loads(
        (root / "results/hennepin_2500m/metadata.json").read_text(encoding="utf-8")
    )
    state_denominator = float(metadata["metric_normalization"]["state_mse_denominator"])
    delta_denominator = float(metadata["metric_normalization"]["change_mse_denominator"])
    if not np.allclose(table["first_order_mse"] / state_denominator, table["nMSE"], atol=1e-12):
        errors.append("Table 4 state normalization does not match the archived denominator")
    if not np.allclose(table["second_order_mse"] / delta_denominator, table["nMSE_delta"], atol=1e-12):
        errors.append("Table 4 transition normalization does not match the archived denominator")

    with np.load(root / "results/hennepin_2500m/top1_reconstruction.npz", allow_pickle=False) as source:
        metrics = state_and_delta_metrics(source["ground_truth_state"], source["teacher_forced_state"])
    raw = pd.read_csv(root / "results/hennepin_2500m/local_proposed_eq_results_denoise.csv", index_col=0)
    selected = raw.loc["Betweenness-augmented gravity* (POW) + Osc"]
    for metric, column in (
        ("mse", "First-order MSE"), ("r2", "First-order R2"), ("correlation", "Corr"),
        ("mse_delta", "Second-order MSE"), ("r2_delta", "Second-order R2"),
        ("correlation_delta", "Corr Delta"),
    ):
        if abs(metrics[metric] - float(selected[column])) > 1e-4:
            errors.append(f"selected 2.5 km reconstruction does not reproduce {column}")
    reconstruction_metadata = metadata.get("reconstruction", {})
    if not np.isclose(
        float(reconstruction_metadata.get("mean_nodewise_time_correlation", float("nan"))),
        metrics["correlation"],
        atol=1e-4,
    ):
        errors.append("2.5 km result metadata misstates mean node-wise correlation")
    fit_metadata = json.loads(
        (root / "results/hennepin_2500m/selected_c9_fit.json").read_text(encoding="utf-8")
    )
    if fit_metadata.get("candidate", {}).get("canonical_id") != "betweenness_gravity":
        errors.append("selected C9 fit does not use the canonical registry candidate ID")
    return errors


def _validate_registry(root: Path) -> list[str]:
    errors: list[str] = []
    from .equations import list_candidates, load_registry

    registry = load_registry(root / "artifacts/candidates.json")
    if len(registry) != 26 or len(list_candidates("table_4")) != 26:
        errors.append("candidate registry must expose exactly 26 Table 4 candidates")
    if len(list_candidates("table_1")) != 11:
        errors.append("candidate registry must expose 11 Table 1 candidates")
    if len(list_candidates("table_3")) != 15:
        errors.append("candidate registry must expose 15 Table 3 candidates")
    expected_table4_ids = [
        "explore_return_diffusion", "betweenness_gravity",
        "single_constrained_gravity_power", "single_constrained_gravity_exponential",
        "classic_gravity_exponential", "edge_betweenness_diffusion",
        "knn_two_band_diffusion", "two_scale_diffusion", "corridor_diffusion",
        "dual_layer_diffusion", "state_adaptive_diffusion", "span_cluster_diffusion",
        "distance_weighted_diffusion", "span_diffusion", "rank_based_diffusion",
        "classic_gravity_power", "saturation_limited_diffusion", "cluster_diffusion",
        "local_global_diffusion", "betweenness_diffusion", "knn_limited_diffusion",
        "knn_diffusion", "rank_tail_diffusion", "rank_choice_gravity",
        "accessibility_drift", "radiation",
    ]
    actual_table4_ids = [item.id for item in list_candidates("table_4")]
    if actual_table4_ids != expected_table4_ids:
        errors.append("candidate registry Table 4 row mapping differs from the paper ranking")
    if registry != list_candidates():
        errors.append("JSON and built-in candidate registries differ")
    return errors


def _validate_archived_equation_parity(root: Path) -> list[str]:
    """Compare every public equation with archived scalar/localized outputs."""

    from .equations import evaluate

    errors: list[str] = []
    fixture_path = root / "tests/fixtures/equation_parity_v1.npz"
    metadata_path = root / "tests/fixtures/equation_parity_v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256(fixture_path) != metadata.get("fixture_sha256"):
        errors.append("archived equation-parity fixture hash mismatch")
        return errors
    registry = metadata.get("candidate_registry", {})
    registry_path = root / str(registry.get("relative_path", ""))
    if not registry_path.is_file() or sha256(registry_path) != registry.get("sha256"):
        errors.append("equation-parity candidate-registry provenance mismatch")
        return errors
    cases = metadata.get("cases", [])
    if len(cases) != 52:
        errors.append(f"equation-parity fixture must contain 52 cases, found {len(cases)}")
        return errors
    with np.load(fixture_path, allow_pickle=False) as fixture:
        if any(fixture[name].dtype.kind == "O" for name in fixture.files):
            errors.append("equation-parity fixture contains object/pickle-capable arrays")
            return errors
        context = {
            "node_betweenness": fixture["node_betweenness"],
            "edge_betweenness": fixture["edge_betweenness"],
            "cluster_labels": fixture["cluster_labels"],
        }
        for case in cases:
            candidate_id = case["candidate_id"]
            prefix = f"{case['mode']}__{candidate_id}"
            parameters = {
                name: fixture[f"{prefix}__param__{name}"]
                for name in case["parameter_names"]
            }
            actual = evaluate(
                candidate_id,
                fixture["state"],
                fixture["distance"],
                parameters,
                adjacency=fixture["adjacency"],
                **context,
            )
            if not np.allclose(
                actual, fixture[f"{prefix}__output"], rtol=3.0e-5, atol=3.0e-5
            ):
                errors.append(
                    f"archived equation parity mismatch: {candidate_id} ({case['mode']})"
                )
    return errors


def _validate_graphrag(root: Path) -> list[str]:
    errors: list[str] = []
    provenance_path = root / "graphrag/provenance.json"
    provenance: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
    def walk(value: Any):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                yield value["path"], value["sha256"]
            for nested in value.values():
                yield from walk(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from walk(nested)

    for name, expected in walk(provenance):
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"invalid GraphRAG SHA-256 value for {name}")
            continue
        path = root / "graphrag" / name
        if not path.is_file():
            errors.append(f"GraphRAG provenance file is missing: {name}")
        elif sha256(path) != expected:
            errors.append(f"GraphRAG hash mismatch: {name}")
    no_invention = json.loads(
        (root / "graphrag/outputs/no_invention_candidates_20260114.json").read_text()
    )
    two_mode = json.loads(
        (root / "graphrag/outputs/two_mode_candidates_20260119.json").read_text()
    )
    if len(no_invention.get("se_candidates", [])) != 9:
        errors.append("No-Invention output must contain 9 candidates")
    if len(two_mode.get("se_candidates_prior_only", [])) != 10:
        errors.append("two-mode output must contain 10 prior-only candidates")
    if len(two_mode.get("se_candidates_context_plus_creativity", [])) != 10:
        errors.append("two-mode output must contain 10 creativity candidates")
    mapping = json.loads((root / "graphrag/candidate_mapping.json").read_text())
    candidates = mapping.get("candidates", [])
    if len(candidates) != 29:
        errors.append("GraphRAG candidate mapping must preserve 29 raw candidates")
    selected = [item for item in candidates if item.get("paper_mapping")]
    if len(selected) != 23 or any(
        not item["paper_mapping"].get("release_candidate_id") for item in selected
    ):
        errors.append("all 23 paper-selected GraphRAG mappings must have release candidate IDs")
    return errors


def _validate_checksum_files(root: Path, files: tuple[Path, ...] | None = None) -> list[str]:
    root = root.resolve()
    files = release_files(root) if files is None else files
    errors: list[str] = []
    manifests = [path for path in files if path.name == "CHECKSUMS.sha256"]
    root_manifest = root / "MANIFEST.sha256"
    if not root_manifest.is_file():
        return ["required repository checksum manifest is missing: MANIFEST.sha256"]
    manifests.append(root_manifest)
    covered: set[Path] = set()
    for manifest in manifests:
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match:
                errors.append(f"invalid checksum line {manifest.relative_to(root)}:{number}")
                continue
            expected, name = match.groups()
            path = (manifest.parent / name).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"checksum path escapes release root: {name}")
                continue
            if not path.is_file():
                errors.append(f"checksum target is missing: {path.relative_to(root)}")
            elif sha256(path) != expected:
                errors.append(f"checksum mismatch: {path.relative_to(root)}")
            if manifest == root_manifest:
                covered.add(path)
    expected_files = {
        path.resolve()
        for path in files
        if path != root_manifest
    }
    missing = sorted(path.relative_to(root) for path in expected_files - covered)
    extra = sorted(path.relative_to(root) for path in covered - expected_files)
    if missing:
        errors.append(f"MANIFEST.sha256 does not cover release files: {missing}")
    if extra:
        errors.append(f"MANIFEST.sha256 contains unexpected files: {extra}")
    return errors


def _validate_repository_hygiene(root: Path, files: tuple[Path, ...] | None = None) -> list[str]:
    root = root.resolve()
    files = release_files(root) if files is None else files
    errors: list[str] = []
    secret_patterns = (
        re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(rb"(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}", re.I),
    )
    absolute_patterns = (
        b"/" + b"projects/",
        b"/" + b"Users/",
        b"\\" + b"Users\\",
    )
    text_suffixes = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".cff", ".toml"}
    for path in files:
        if path.name == ".env.example":
            continue
        is_environment = path.name == ".env" or path.name.startswith(".env.")
        if is_environment:
            errors.append(f"local environment/credential file found: {path.relative_to(root)}")
        if is_environment or path.suffix.lower() in text_suffixes or path.name in {"MANIFEST.sha256", "CHECKSUMS.sha256"}:
            content = path.read_bytes()
            if any(pattern.search(content) for pattern in secret_patterns):
                errors.append(f"possible plaintext credential in {path.relative_to(root)}")
            if any(pattern in content for pattern in absolute_patterns):
                errors.append(f"machine-specific absolute path in {path.relative_to(root)}")
    sensitive_headers = {"device_key", "device_id", "start_lat", "start_lon", "end_lat", "end_lon"}
    for path in files:
        if path.suffix.lower() != ".csv":
            continue
        with path.open(encoding="utf-8-sig", errors="replace") as source:
            header = source.readline().strip().lower()
        fields = {item.strip() for item in header.split(",")}
        leaked = sensitive_headers.intersection(fields)
        if leaked:
            errors.append(f"raw mobility fields {sorted(leaked)} found in {path.relative_to(root)}")
    return errors


def _validate_artifact_scope(root: Path, files: tuple[Path, ...] | None = None) -> list[str]:
    """Inspect release artifacts without treating runtime output as shipped data."""
    root = root.resolve()
    files = release_files(root) if files is None else files
    errors: list[str] = []
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".ipynb"}
    unsafe = [str(path.relative_to(root)) for path in files if path.suffix.lower() in forbidden_suffixes]
    if unsafe:
        errors.append(f"unsafe pickle-capable artifacts found: {unsafe}")
    excluded_tokens = ("1500", "2000", "figure8", "figure_8")
    leaked = [
        str(path.relative_to(root))
        for path in files
        if any(token in part.lower() for part in path.relative_to(root).parts for token in excluded_tokens)
    ]
    if leaked:
        errors.append(f"out-of-scope scale artifacts found: {leaked}")
    large = [
        (str(path.relative_to(root)), path.stat().st_size)
        for path in files
        if path.stat().st_size > 50 * 1024 * 1024
    ]
    if large:
        errors.append(f"files over 50 MiB found: {large}")
    return errors


def validate_release(root: str | Path | None = None) -> dict[str, Any]:
    release_root = Path(root).resolve() if root else repository_root()
    errors: list[str] = []
    warnings: list[str] = []
    synthetic = load_dataset(release_root / "data/synthetic/dataset.npz")
    errors.extend(
        f"synthetic: {item}"
        for item in validate_dataset(
            synthetic, expected_timesteps=501, expected_nodes=20, require_symmetric_distance=True
        )
    )
    case = load_dataset(release_root / "data/hennepin_2500m/hennepin_2500m.npz")
    errors.extend(
        f"hennepin_2500m: {item}"
        for item in validate_dataset(
            case,
            expected_timesteps=480,
            expected_nodes=210,
            require_symmetric_distance=False,
            geometry_path=release_root / "data/hennepin_2500m/grid_cells.geojson",
        )
    )
    asymmetry = float(np.max(np.abs(case.distance - case.distance.T)))
    if asymmetry > 1e-6:
        warnings.append(
            f"hennepin_2500m normalized distance is asymmetric (max |D-D.T|={asymmetry:.7f}); "
            "this is the archived column-wise preprocessing used by the fits"
        )
    if (synthetic.train_end, synthetic.val_end, synthetic.num_timesteps) != (300, 400, 501):
        errors.append("synthetic split boundaries must be 300/100/101 states")
    if (case.train_end, case.val_end, case.num_timesteps) != (288, 384, 480):
        errors.append("hennepin_2500m split boundaries must be 288/96/96 states")
    if not np.array_equal(synthetic.adjacency, np.ones((20, 20)) - np.eye(20)):
        errors.append("synthetic adjacency must be complete without self-loops")
    if not np.array_equal(case.adjacency, np.ones((210, 210)) - np.eye(210)):
        errors.append("hennepin_2500m adjacency must be complete without self-loops")
    with np.load(release_root / "data/hennepin_2500m/hennepin_2500m.npz", allow_pickle=False) as source:
        if int(source["test_end"]) != 480 or int(source["interval_minutes"]) != 15:
            errors.append("hennepin_2500m scalar time metadata is invalid")
        if int(source["grid_resolution_m"]) != 2500:
            errors.append("hennepin_2500m resolution metadata is not 2500 metres")
    with np.load(release_root / "figure_sources/figure4/synthetic_figure4.npz", allow_pickle=False) as figure4:
        if figure4["states"].shape != (501, 20):
            errors.append("Figure 4 realization must contain 501 x 20 states")
        if not np.array_equal(figure4["initial_state"], synthetic.initial_state):
            errors.append("Figure 4 and Table 2 initial states must match")
        if not np.array_equal(figure4["distance_raw"], synthetic.distance_raw):
            errors.append("Figure 4 and Table 2 raw distance matrices must match")
        if np.array_equal(figure4["states"], synthetic.states):
            errors.append("Figure 4 realization must remain distinct from the Table 2 trajectory")
    errors.extend(_validate_table2(release_root))
    errors.extend(_validate_table4(release_root))
    errors.extend(_validate_registry(release_root))
    errors.extend(_validate_archived_equation_parity(release_root))
    errors.extend(_validate_graphrag(release_root))
    files = release_files(release_root)
    errors.extend(_validate_checksum_files(release_root, files))
    errors.extend(_validate_repository_hygiene(release_root, files))
    errors.extend(_validate_artifact_scope(release_root, files))
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "datasets": {
            "synthetic": list(synthetic.states.shape),
            "hennepin_2500m": list(case.states.shape),
        },
    }
