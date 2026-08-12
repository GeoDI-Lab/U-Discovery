"""Regenerate the deterministic archived-solver parity fixture.

Run from the release root with a PyTorch-enabled environment:

    python tests/fixtures/generate_equation_parity_fixture.py

The archived working-tree solver is read only to produce reference outputs; it
is not imported by the release test suite.
"""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


RELEASE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = RELEASE_ROOT.parent
ARCHIVE_RELATIVE_PATH = Path(
    "UrbanDE_Net_1119_county_proposed_int_denoised/pde_solver.py"
)
ARCHIVE_SOLVER = WORKSPACE_ROOT / ARCHIVE_RELATIVE_PATH
REGISTRY_PATH = RELEASE_ROOT / "artifacts/candidates.json"
FIXTURE_PATH = Path(__file__).with_name("equation_parity_v1.npz")
METADATA_PATH = Path(__file__).with_name("equation_parity_v1.json")
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from udiscovery.equations import list_candidates  # noqa: E402


STATE = np.asarray([2.0, 3.5, 5.0, 7.5, 11.0], dtype=np.float32)
DISTANCE = np.asarray(
    [
        [0.0, 1.1, 2.4, 4.2, 7.0],
        [1.1, 0.0, 1.7, 3.1, 5.8],
        [2.4, 1.7, 0.0, 1.3, 3.6],
        [4.2, 3.1, 1.3, 0.0, 2.2],
        [7.0, 5.8, 3.6, 2.2, 0.0],
    ],
    dtype=np.float32,
)
ADJACENCY = np.ones((5, 5), dtype=np.float32) - np.eye(5, dtype=np.float32)
NODE_BETWEENNESS = np.asarray([0.05, 0.3, 1.0, 0.5, 0.1], dtype=np.float32)
EDGE_BETWEENNESS = np.asarray(
    [
        [0.0, 0.8, 0.4, 0.2, 0.1],
        [0.8, 0.0, 1.0, 0.6, 0.2],
        [0.4, 1.0, 0.0, 0.9, 0.5],
        [0.2, 0.6, 0.9, 0.0, 0.7],
        [0.1, 0.2, 0.5, 0.7, 0.0],
    ],
    dtype=np.float32,
)
CLUSTER_LABELS = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)


PARAMETER_VALUES = {
    "alpha": 0.85,
    "beta": 1.15,
    "gamma": 1.35,
    "lambda": 0.35,
    "eta": 1.25,
    "lambda_z": 0.45,
    "q": 1.1,
    "lambda_k": 0.55,
    "K": 2.0,
    "lambda_s": 0.4,
    "kappa": 1.2,
    "rho": 0.25,
    "Gamma": 0.3,
    "chi": 0.7,
    "delta": 0.8,
    "rho_in": 1.1,
    "rho_out": 0.35,
    "K_clusters": 2.0,
    "k": 2.0,
    "C": 0.4,
    "eta_e": 1.4,
    "eta_l": 1.1,
    "eta_c": 2.0,
    "a": 1.0,
    "r0": 2.5,
    "eta1": 0.9,
    "eta2": 1.8,
    "mu": 0.6,
    "c": 0.7,
    "R": 2.0,
    "beta_r": 0.3,
    "beta_t": 0.08,
    "b": 0.25,
}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write NPZ with sorted members and a fixed ZIP timestamp."""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def load_solver():
    if not ARCHIVE_SOLVER.is_file():
        raise FileNotFoundError(f"Archived solver not found at {ARCHIVE_RELATIVE_PATH}")
    module_spec = importlib.util.spec_from_file_location("archived_pde_solver", ARCHIVE_SOLVER)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("Could not load archived solver module")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.PDESolver("cpu")


def parameter_value(name: str, scope: str, mode: str) -> np.ndarray:
    value = np.asarray(PARAMETER_VALUES[name], dtype=np.float32)
    if mode == "localized" and scope == "node":
        value = np.linspace(float(value), float(value) * 1.15, STATE.size, dtype=np.float32)
    return value


def main() -> None:
    solver = load_solver()
    state = torch.as_tensor(STATE)
    distance = torch.as_tensor(DISTANCE)
    adjacency = torch.as_tensor(ADJACENCY)
    node_betweenness = torch.as_tensor(NODE_BETWEENNESS)
    edge_betweenness = torch.as_tensor(EDGE_BETWEENNESS)
    cluster_labels = torch.as_tensor(CLUSTER_LABELS)
    extra_args = {
        "betweenness_diffusion": [node_betweenness],
        "edge_betweenness_diffusion": [edge_betweenness],
        "cluster_diffusion": [cluster_labels],
        "span_cluster_diffusion": [cluster_labels],
        "corridor_diffusion": [edge_betweenness],
        "betweenness_gravity": [edge_betweenness],
    }

    arrays: dict[str, np.ndarray] = {
        "state": STATE,
        "distance": DISTANCE,
        "adjacency": ADJACENCY,
        "node_betweenness": NODE_BETWEENNESS,
        "edge_betweenness": EDGE_BETWEENNESS,
        "cluster_labels": CLUSTER_LABELS,
    }
    cases = []
    for candidate in list_candidates("table_4"):
        legacy_method = candidate.legacy_solver_names[0]
        function = getattr(solver, legacy_method)
        for mode in ("scalar", "localized"):
            parameters = [
                parameter_value(name, scope, mode)
                for name, scope in zip(candidate.parameter_names, candidate.parameter_scopes)
            ]
            prefix = f"{mode}__{candidate.id}"
            for name, value in zip(candidate.parameter_names, parameters):
                arrays[f"{prefix}__param__{name}"] = value
            torch_parameters = [torch.as_tensor(value) for value in parameters]
            with torch.no_grad():
                output = function(
                    state,
                    torch_parameters,
                    distance,
                    adjacency,
                    *extra_args.get(candidate.id, []),
                )
            arrays[f"{prefix}__output"] = output.detach().cpu().numpy()
            cases.append(
                {
                    "candidate_id": candidate.id,
                    "mode": mode,
                    "legacy_solver_method": legacy_method,
                    "parameter_names": list(candidate.parameter_names),
                }
            )

    write_deterministic_npz(FIXTURE_PATH, arrays)
    metadata = {
        "schema_version": 1,
        "fixture": FIXTURE_PATH.name,
        "fixture_sha256": file_hash(FIXTURE_PATH),
        "description": "Archived pde_solver.py outputs for all 26 Table 4 equations under scalar and localized parameterization.",
        "archive_solver": {
            "relative_path": ARCHIVE_RELATIVE_PATH.as_posix(),
            "sha256": file_hash(ARCHIVE_SOLVER),
        },
        "candidate_registry": {
            "relative_path": "artifacts/candidates.json",
            "sha256": file_hash(REGISTRY_PATH),
        },
        "array_serialization": "NumPy NPY members in deterministic ZIP; allow_pickle=False",
        "floating_dtype": "float32",
        "cases": cases,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE_PATH.relative_to(RELEASE_ROOT)}")
    print(f"sha256 {metadata['fixture_sha256']}")


if __name__ == "__main__":
    main()
