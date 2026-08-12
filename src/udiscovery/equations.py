"""Canonical spatial interaction equations used in the U-Discovery paper.

The public :func:`evaluate` function accepts either NumPy arrays or PyTorch
tensors.  PyTorch inputs stay in PyTorch, so gradients through the state and
continuous parameters are retained.  Distance ranks, neighbourhoods,
clustering, and centrality are graph metadata and are intentionally treated as
non-differentiable context.

The implementation follows the archived ``pde_solver.py`` parameter order and
numerical safeguards.  Only the 26 equations that appear in Table 4 are
exposed; duplicate candidates that were fitted but omitted from the paper are
not part of this release API.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # PyTorch is optional for data inspection and paper-artifact rendering.
    import torch
except ImportError:  # pragma: no cover - exercised in NumPy-only installs
    torch = None  # type: ignore[assignment]


EPS = 1.0e-12
MIN_STATE = 1.0e-6
SAFE_DISTANCE = 1.0e12


class EquationInputError(ValueError):
    """Raised when an equation receives an invalid state, graph, or parameter."""


@dataclass(frozen=True)
class CandidateSpec:
    """Machine-readable description of one paper equation candidate."""

    id: str
    display_name: str
    formula: str
    origin: str
    graphrag_id: str | None
    legacy_solver_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_scopes: tuple[str, ...]
    context_inputs: tuple[str, ...]
    paper_rows: Mapping[str, int]
    complexities: Mapping[str, int]

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)


def _spec(
    candidate_id: str,
    display_name: str,
    formula: str,
    origin: str,
    graphrag_id: str | None,
    legacy: Sequence[str],
    parameters: Sequence[tuple[str, str]],
    context: Sequence[str] = (),
    rows: Mapping[str, int] | None = None,
    complexities: Mapping[str, int] | None = None,
) -> CandidateSpec:
    return CandidateSpec(
        id=candidate_id,
        display_name=display_name,
        formula=formula,
        origin=origin,
        graphrag_id=graphrag_id,
        legacy_solver_names=tuple(legacy),
        parameter_names=tuple(name for name, _ in parameters),
        parameter_scopes=tuple(scope for _, scope in parameters),
        context_inputs=tuple(context),
        paper_rows=dict(rows or {}),
        complexities=dict(complexities or {}),
    )


# ``scope`` records the localized 2.5 km fitting contract: ``node`` parameters
# may be scalar or length N; ``global`` parameters must be scalar.  Table 1 and
# Table 4 use different complexity bookkeeping for four literature baselines,
# so complexity is deliberately stored per table.
_SPECS: tuple[CandidateSpec, ...] = (
    _spec(
        "classic_gravity_power", "Classic gravity model (POW)",
        "sum_j[p_j^alpha p_i^beta d_ji^-gamma - p_i^alpha p_j^beta d_ij^-gamma]",
        "literature_baseline", None, ("interaction_powerlaw_gravity_classic",),
        (("alpha", "node"), ("beta", "node"), ("gamma", "global")),
        rows={"table_1": 3, "table_4": 16}, complexities={"table_1": 20, "table_4": 22},
    ),
    _spec(
        "classic_gravity_exponential", "Classic gravity model (EXP)",
        "sum_j[p_j^alpha p_i^beta exp(-gamma d_ji) - p_i^alpha p_j^beta exp(-gamma d_ij)]",
        "literature_baseline", None, ("interaction_exponential_gravity_classic",),
        (("alpha", "node"), ("beta", "node"), ("gamma", "global")),
        rows={"table_1": 4, "table_4": 5}, complexities={"table_1": 22, "table_4": 24},
    ),
    _spec(
        "single_constrained_gravity_power", "Single-constrained gravity model (POW)",
        "inflow_i = p_i^beta sum_j[d_ji^-gamma p_j / sum_k(d_jk^-gamma p_k^beta)]; outflow_i=p_i",
        "literature_baseline", "SE2", ("interaction_powerlaw_gravity_v1",),
        (("beta", "node"), ("gamma", "global")),
        rows={"table_1": 1, "table_4": 3}, complexities={"table_1": 21, "table_4": 17},
    ),
    _spec(
        "single_constrained_gravity_exponential", "Single-constrained gravity model (EXP)",
        "inflow_i = p_i^beta sum_j[exp(-gamma d_ji) p_j / sum_k(exp(-gamma d_jk) p_k^beta)]; outflow_i=p_i",
        "literature_baseline", None, ("interaction_exponential_gravity_v1",),
        (("beta", "node"), ("gamma", "global")),
        rows={"table_1": 2, "table_4": 4}, complexities={"table_1": 25, "table_4": 21},
    ),
    _spec(
        "radiation", "Radiation model",
        "sum_j(T_ji-T_ij), T_ij=p_i p_j/[(p_i+s_ij)(p_i+p_j+s_ij)]",
        "literature_baseline", None, ("interaction_radiation_model",), (),
        rows={"table_1": 5, "table_4": 26}, complexities={"table_1": 13, "table_4": 13},
    ),
    _spec(
        "distance_weighted_diffusion", "Distance-weighted diffusion",
        "lambda sum_j d_ij^-eta (p_j-p_i)", "graphrag_no_invention", "SE3",
        ("interaction_se3", "interaction_SE3"), (("lambda", "node"), ("eta", "global")),
        rows={"table_1": 6, "table_4": 13}, complexities={"table_1": 11, "table_4": 11},
    ),
    _spec(
        "rank_based_diffusion", "Rank-based diffusion",
        "lambda_z sum_j 0.5[r_i(j)^-q+r_j(i)^-q](p_j-p_i)",
        "graphrag_no_invention", "SE5", ("interaction_se5", "interaction_SE5"),
        (("lambda_z", "node"), ("q", "global")),
        rows={"table_1": 7, "table_4": 15}, complexities={"table_1": 16, "table_4": 16},
    ),
    _spec(
        "knn_diffusion", "KNN diffusion",
        "lambda_k sum_{j in N_K(i)} normalized(d_ij^-eta)(p_j-p_i)",
        "graphrag_no_invention", "SE6", ("interaction_se6", "interaction_SE6"),
        (("lambda_k", "node"), ("K", "global"), ("eta", "global")),
        rows={"table_1": 8, "table_4": 22}, complexities={"table_1": 15, "table_4": 15},
    ),
    _spec(
        "saturation_limited_diffusion", "Saturation-limited diffusion",
        "lambda_s sum_j normalized(d_ij^-eta)[g(p_j)-g(p_i)], g(p)=p/(1+p/kappa)",
        "graphrag_no_invention", "SE7", ("interaction_se7", "interaction_SE7"),
        (("lambda_s", "node"), ("kappa", "node"), ("eta", "global")),
        rows={"table_1": 9, "table_4": 17}, complexities={"table_1": 27, "table_4": 27},
    ),
    _spec(
        "accessibility_drift", "Accessibility-driven drift",
        "rho p_i sum_j normalized(d_ij^-eta)(A_j-A_i), A_i=sum_k d_ik^-eta",
        "graphrag_no_invention", "SE8", ("interaction_se8", "interaction_SE8"),
        (("rho", "node"), ("eta", "global")),
        rows={"table_1": 10, "table_4": 25}, complexities={"table_1": 23, "table_4": 23},
    ),
    _spec(
        "local_global_diffusion", "Local-global mixed diffusion",
        "(1-Gamma)lambda sum_j d_ij^-eta(p_j-p_i)+Gamma chi(mean(p)-p_i)",
        "graphrag_no_invention", "SE9", ("interaction_se9", "interaction_SE9"),
        (("Gamma", "node"), ("lambda", "node"), ("chi", "node"), ("eta", "global")),
        rows={"table_1": 11, "table_4": 19}, complexities={"table_1": 25, "table_4": 25},
    ),
    _spec(
        "rank_choice_gravity", "Rank-choice gravity",
        "sum_j(T_ji-T_ij), T_ij=p_i p_j f[r_i(j)]/sum_k p_k f[r_i(k)]",
        "graphrag_prior_only", "P3", ("interaction_p3",), (("lambda", "global"),),
        rows={"table_3": 1, "table_4": 24}, complexities={"table_3": 29, "table_4": 29},
    ),
    _spec(
        "betweenness_diffusion", "Betweenness-modulated diffusion",
        "kappa sum_j sqrt(b_i b_j)d_ij^-eta(p_j-p_i)",
        "graphrag_prior_only", "P5", ("interaction_p5",),
        (("kappa", "node"), ("eta", "global")), ("node_betweenness",),
        rows={"table_3": 2, "table_4": 20}, complexities={"table_3": 16, "table_4": 16},
    ),
    _spec(
        "edge_betweenness_diffusion", "Edge-betweenness weighted mixed diffusion",
        "kappa sum_j e_ij^delta d_ij^-eta(p_j-p_i)",
        "graphrag_prior_only", "P6", ("interaction_p6",),
        (("delta", "node"), ("kappa", "node"), ("eta", "global")), ("edge_betweenness",),
        rows={"table_3": 3, "table_4": 6}, complexities={"table_3": 14, "table_4": 14},
    ),
    _spec(
        "cluster_diffusion", "Cluster-constrained diffusion",
        "kappa sum_j[rho_in I(c_i=c_j)+rho_out I(c_i!=c_j)]d_ij^-eta(p_j-p_i)",
        "graphrag_prior_only", "P7", ("interaction_p7",),
        (("rho_in", "node"), ("rho_out", "node"), ("kappa", "node"),
         ("K_clusters", "global"), ("eta", "global")), ("cluster_labels",),
        rows={"table_3": 4, "table_4": 18}, complexities={"table_3": 18, "table_4": 18},
    ),
    _spec(
        "span_diffusion", "Span-adaptive diffusion",
        "kappa sum_j[d_ij/sqrt(s_i s_j)]^-eta(p_j-p_i)",
        "graphrag_prior_only", "P8", ("interaction_p8",),
        (("kappa", "node"), ("k", "global"), ("eta", "global")),
        rows={"table_3": 5, "table_4": 14}, complexities={"table_3": 22, "table_4": 22},
    ),
    _spec(
        "knn_limited_diffusion", "KNN-limited diffusion",
        "kappa sum_j 0.5[I(j in N_k(i))+I(i in N_k(j))]d_ij^-eta(p_j-p_i)",
        "graphrag_prior_only", "P10", ("interaction_p10",),
        (("kappa", "node"), ("k", "global"), ("eta", "global")),
        rows={"table_3": 6, "table_4": 21}, complexities={"table_3": 16, "table_4": 16},
    ),
    _spec(
        "explore_return_diffusion", "Explore-return diffusion",
        "sum_j(T_ji-T_ij), T_ij=p_i[(1-P_new)w_return+P_new w_explore]",
        "graphrag_context_creativity", "C1", ("interaction_c1",),
        (("lambda", "node"), ("C", "node"), ("k", "global"), ("eta_e", "global")),
        rows={"table_3": 7, "table_4": 1}, complexities={"table_3": 47, "table_4": 47},
    ),
    _spec(
        "dual_layer_diffusion", "Dual-layer overlap diffusion",
        "kappa sum_j[Gamma d_ij^-eta_l+(1-Gamma)d_ij^-eta_c](p_j-p_i)",
        "graphrag_context_creativity", "C2", ("interaction_c2",),
        (("Gamma", "node"), ("kappa", "node"), ("eta_l", "global"), ("eta_c", "global")),
        rows={"table_3": 8, "table_4": 10}, complexities={"table_3": 20, "table_4": 20},
    ),
    _spec(
        "two_scale_diffusion", "Two-scale piecewise diffusion",
        "kappa sum_j K(d_ij)(p_j-p_i), K(d)=a d^-eta1 below r0 else a r0^(eta2-eta1)d^-eta2",
        "graphrag_context_creativity", "C3", ("interaction_c3",),
        (("a", "node"), ("kappa", "node"), ("r0", "global"),
         ("eta1", "global"), ("eta2", "global")),
        rows={"table_3": 9, "table_4": 8}, complexities={"table_3": 26, "table_4": 26},
    ),
    _spec(
        "span_cluster_diffusion", "Span- and cluster-aware diffusion",
        "kappa sum_j[d_ij/sqrt(s_i s_j)]^-eta[rho_in I(c_i=c_j)+rho_out I(c_i!=c_j)](p_j-p_i)",
        "graphrag_context_creativity", "C4", ("interaction_c4",),
        (("rho_in", "node"), ("rho_out", "node"), ("kappa", "node"),
         ("k", "global"), ("eta", "global"), ("K_clusters", "global")), ("cluster_labels",),
        rows={"table_3": 10, "table_4": 12}, complexities={"table_3": 30, "table_4": 30},
    ),
    _spec(
        "corridor_diffusion", "Corridor-enhanced diffusion",
        "kappa sum_j[1+mu e_ij/mean(e)]d_ij^-eta(p_j-p_i)",
        "graphrag_context_creativity", "C5", ("interaction_c5",),
        (("mu", "node"), ("kappa", "node"), ("eta", "global")), ("edge_betweenness",),
        rows={"table_3": 11, "table_4": 9}, complexities={"table_3": 18, "table_4": 18},
    ),
    _spec(
        "rank_tail_diffusion", "Rank-truncated + tail choice diffusion",
        "sum_j(T_ji-T_ij), T_ij=p_i p_j w_ij/sum_k p_k w_ik; rank head and power-law tail",
        "graphrag_context_creativity", "C7", ("interaction_c7",),
        (("lambda", "node"), ("c", "node"), ("R", "global"), ("eta", "global")),
        rows={"table_3": 12, "table_4": 23}, complexities={"table_3": 49, "table_4": 49},
    ),
    _spec(
        "state_adaptive_diffusion", "State-adaptive diffusion",
        "[beta_r+beta_t sum_j w_ij p_j] sum_j w_ij(p_j-p_i)",
        "graphrag_context_creativity", "C8", ("interaction_c8",),
        (("beta_r", "node"), ("beta_t", "node"), ("eta", "global")),
        rows={"table_3": 13, "table_4": 11}, complexities={"table_3": 29, "table_4": 29},
    ),
    _spec(
        "betweenness_gravity", "Betweenness-augmented gravity",
        "sum_j(T_ji-T_ij), T_ij=kappa p_i^alpha p_j^beta(e_ij+eps)^delta d_ij^-gamma",
        "graphrag_context_creativity", "C9", ("interaction_c9",),
        (("alpha", "node"), ("beta", "node"), ("kappa", "global"),
         ("delta", "global"), ("gamma", "global")), ("edge_betweenness",),
        rows={"table_3": 14, "table_4": 2}, complexities={"table_3": 26, "table_4": 26},
    ),
    _spec(
        "knn_two_band_diffusion", "KNN two-band diffusion",
        "kappa sum_j[a I(kNN union)+b I(other)]d_ij^-eta(p_j-p_i)",
        "graphrag_context_creativity", "C10", ("interaction_c10",),
        (("a", "node"), ("b", "node"), ("kappa", "node"),
         ("k", "global"), ("eta", "global")),
        rows={"table_3": 15, "table_4": 7}, complexities={"table_3": 18, "table_4": 18},
    ),
)


_BY_ID = {candidate.id: candidate for candidate in _SPECS}
_ALIASES: dict[str, str] = {}
for _candidate in _SPECS:
    for _alias in (
        _candidate.id,
        _candidate.display_name,
        _candidate.graphrag_id,
        *_candidate.legacy_solver_names,
    ):
        if _alias:
            _ALIASES[_alias.casefold()] = _candidate.id


def get_candidate(candidate_id: str) -> CandidateSpec:
    """Return a candidate by canonical ID, GraphRAG ID, or legacy solver name."""

    try:
        canonical_id = _ALIASES[str(candidate_id).casefold()]
    except KeyError as exc:
        available = ", ".join(sorted(_BY_ID))
        raise KeyError(f"Unknown candidate {candidate_id!r}. Available IDs: {available}") from exc
    return _BY_ID[canonical_id]


def list_candidates(table: str | None = None) -> tuple[CandidateSpec, ...]:
    """List all released candidates, optionally in a paper table's row order."""

    if table is None:
        return _SPECS
    table_key = str(table).lower().replace(" ", "_")
    if table_key in {"1", "table1"}:
        table_key = "table_1"
    elif table_key in {"3", "table3"}:
        table_key = "table_3"
    elif table_key in {"4", "table4"}:
        table_key = "table_4"
    selected = [candidate for candidate in _SPECS if table_key in candidate.paper_rows]
    if not selected:
        raise KeyError(f"Unknown or empty paper table {table!r}; expected table_1, table_3, or table_4")
    return tuple(sorted(selected, key=lambda candidate: candidate.paper_rows[table_key]))


def load_registry(path: str | Path | None = None) -> tuple[CandidateSpec, ...]:
    """Load registry metadata from JSON, or return the built-in release registry.

    Passing a path is useful for provenance validation.  Equation dispatch is
    intentionally restricted to the built-in, reviewed IDs; loading JSON never
    executes functions named by the file.
    """

    if path is None:
        return _SPECS
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("candidates")
    if not isinstance(records, list):
        raise EquationInputError("Registry JSON must contain a 'candidates' list")
    loaded: list[CandidateSpec] = []
    for record in records:
        parameters = record.get("parameters", [])
        loaded.append(
            _spec(
                record["id"], record["display_name"], record["formula"], record["origin"],
                record.get("graphrag_id"), record.get("legacy_solver_names", []),
                [(item["name"], item["scope_2500m"]) for item in parameters],
                record.get("context_inputs", []), record.get("paper_rows", {}),
                record.get("complexities", {}),
            )
        )
    ids = [item.id for item in loaded]
    if len(ids) != len(set(ids)):
        raise EquationInputError("Registry JSON contains duplicate candidate IDs")
    return tuple(loaded)


def _is_torch(value: Any) -> bool:
    return torch is not None and torch.is_tensor(value)


def _as_array(value: Any, like: Any, *, boolean: bool = False) -> Any:
    if _is_torch(like):
        dtype = torch.bool if boolean else like.dtype
        if _is_torch(value):
            return value.to(device=like.device, dtype=dtype)
        return torch.as_tensor(value, device=like.device, dtype=dtype)
    return np.asarray(value, dtype=bool if boolean else np.asarray(like).dtype)


def _ndim(value: Any) -> int:
    return int(value.ndim) if hasattr(value, "ndim") else np.asarray(value).ndim


def _numel(value: Any) -> int:
    return int(value.numel()) if _is_torch(value) else int(np.asarray(value).size)


def _reshape(value: Any, *shape: int) -> Any:
    return value.reshape(*shape)


def _copy(value: Any) -> Any:
    return value.clone() if _is_torch(value) else value.copy()


def _all_finite(value: Any) -> bool:
    if _is_torch(value):
        return bool(torch.isfinite(value).all().detach().cpu().item())
    return bool(np.isfinite(value).all())


def _any(value: Any) -> bool:
    if _is_torch(value):
        return bool(value.any().detach().cpu().item())
    return bool(np.any(value))


def _clamp(value: Any, minimum: float | None = None, maximum: float | None = None) -> Any:
    if _is_torch(value):
        return torch.clamp(value, min=minimum, max=maximum)
    low = -np.inf if minimum is None else minimum
    high = np.inf if maximum is None else maximum
    return np.clip(value, low, high)


def _where(condition: Any, left: Any, right: Any) -> Any:
    return torch.where(condition, left, right) if _is_torch(condition) else np.where(condition, left, right)


def _exp(value: Any) -> Any:
    return torch.exp(value) if _is_torch(value) else np.exp(value)


def _sqrt(value: Any) -> Any:
    return torch.sqrt(value) if _is_torch(value) else np.sqrt(value)


def _log(value: Any) -> Any:
    return torch.log(value) if _is_torch(value) else np.log(value)


def _mean(value: Any) -> Any:
    return value.mean()


def _sum(value: Any, axis: int, *, keepdims: bool = False) -> Any:
    if _is_torch(value):
        return torch.sum(value, dim=axis, keepdim=keepdims)
    return np.sum(value, axis=axis, keepdims=keepdims)


def _eye(n: int, like: Any) -> Any:
    if _is_torch(like):
        return torch.eye(n, dtype=torch.bool, device=like.device)
    return np.eye(n, dtype=bool)


def _ones(shape: tuple[int, ...], like: Any, *, boolean: bool = False) -> Any:
    if _is_torch(like):
        return torch.ones(shape, dtype=torch.bool if boolean else like.dtype, device=like.device)
    return np.ones(shape, dtype=bool if boolean else np.asarray(like).dtype)


def _zeros(shape: tuple[int, ...], like: Any, *, boolean: bool = False) -> Any:
    if _is_torch(like):
        return torch.zeros(shape, dtype=torch.bool if boolean else like.dtype, device=like.device)
    return np.zeros(shape, dtype=bool if boolean else np.asarray(like).dtype)


def _to_numpy(value: Any) -> np.ndarray:
    if _is_torch(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_inputs(state: Any, distance: Any, adjacency: Any | None) -> tuple[Any, Any, Any]:
    if _is_torch(state):
        if not (state.dtype.is_floating_point or state.dtype.is_complex):
            raise EquationInputError("state must have a floating-point dtype")
        p = state
    else:
        p = np.asarray(state)
        if p.dtype.kind not in "fc":
            p = p.astype(np.float64)
    if _ndim(p) != 1 or p.shape[0] < 2:
        raise EquationInputError(f"state must have shape (nodes,), got {tuple(p.shape)}")
    if not _all_finite(p):
        raise EquationInputError("state contains NaN or infinite values")

    d = _as_array(distance, p)
    n = int(p.shape[0])
    if _ndim(d) != 2 or tuple(d.shape) != (n, n):
        raise EquationInputError(f"distance must have shape ({n}, {n}), got {tuple(d.shape)}")
    if not _all_finite(d):
        raise EquationInputError("distance contains NaN or infinite values")
    if _any(d < 0):
        raise EquationInputError("distance values must be non-negative")

    if adjacency is None:
        mask = _ones((n, n), p, boolean=True) & ~_eye(n, p)
    else:
        mask = _as_array(adjacency, p, boolean=True)
        if _ndim(mask) != 2 or tuple(mask.shape) != (n, n):
            raise EquationInputError(f"adjacency must have shape ({n}, {n}), got {tuple(mask.shape)}")
        mask = _copy(mask)
        if _is_torch(mask):
            mask.fill_diagonal_(False)
        else:
            np.fill_diagonal(mask, False)
    return p, d, mask


def _prepare_params(spec: CandidateSpec, params: Mapping[str, Any] | Sequence[Any], state: Any) -> list[Any]:
    if isinstance(params, Mapping):
        missing = [name for name in spec.parameter_names if name not in params]
        extra = sorted(set(params) - set(spec.parameter_names))
        if missing or extra:
            raise EquationInputError(
                f"{spec.id} parameter mismatch; missing={missing or 'none'}, extra={extra or 'none'}"
            )
        raw_values = [params[name] for name in spec.parameter_names]
    else:
        raw_values = list(params)
        if len(raw_values) != spec.parameter_count:
            raise EquationInputError(
                f"{spec.id} expects {spec.parameter_count} parameters {spec.parameter_names}, "
                f"got {len(raw_values)}"
            )

    n = int(state.shape[0])
    prepared: list[Any] = []
    for name, scope, raw in zip(spec.parameter_names, spec.parameter_scopes, raw_values):
        value = _as_array(raw, state)
        if _ndim(value) == 1 and _numel(value) == 1:
            value = _reshape(value, ())
        elif _ndim(value) == 2 and tuple(value.shape) == (n, 1) and scope == "node":
            value = _reshape(value, n)
        if _ndim(value) > 0:
            if scope != "node" or tuple(value.shape) != (n,):
                expected = "a scalar or length-N vector" if scope == "node" else "a scalar"
                raise EquationInputError(f"parameter {name!r} must be {expected}, got {tuple(value.shape)}")
        if not _all_finite(value):
            raise EquationInputError(f"parameter {name!r} contains NaN or infinite values")
        prepared.append(value)
    return prepared


def _safe_distance(distance: Any) -> Any:
    replacement = _as_array(SAFE_DISTANCE, distance)
    return _where(distance == 0, replacement, distance)


def _laplacian(weight: Any, state: Any) -> Any:
    return weight @ state - _sum(weight, 1) * state


def _normalise_rows(weight: Any) -> Any:
    return weight / _clamp(_sum(weight, 1, keepdims=True), EPS)


def _integer(value: Any, n: int, name: str) -> int:
    if _is_torch(value):
        number = float(value.detach().cpu().item())
    else:
        number = float(np.asarray(value).item())
    if not np.isfinite(number):
        raise EquationInputError(f"discrete parameter {name!r} must be finite")
    return max(1, min(int(round(number)), max(1, n - 1)))


def _rank_matrix(distance: Any, adjacency: Any) -> Any:
    n = int(distance.shape[0])
    valid = adjacency & ~_eye(n, distance)
    masked = _where(valid, distance, _as_array(SAFE_DISTANCE, distance))
    if _is_torch(distance):
        order = torch.argsort(masked, dim=1, stable=True)
        ranks = torch.zeros((n, n), dtype=distance.dtype, device=distance.device)
        values = torch.arange(1, n + 1, dtype=distance.dtype, device=distance.device).view(1, n).expand(n, n)
        ranks.scatter_(1, order, values)
    else:
        order = np.argsort(masked, axis=1, kind="stable")
        ranks = np.zeros((n, n), dtype=distance.dtype)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, n + 1), (n, n)), axis=1)
    return _where(valid, ranks, _as_array(0.0, distance))


def _knn_mask(distance: Any, adjacency: Any, k_value: Any) -> Any:
    n = int(distance.shape[0])
    k = _integer(k_value, n, "k")
    valid = adjacency & ~_eye(n, distance)
    masked = _where(valid, distance, _as_array(SAFE_DISTANCE, distance))
    if _is_torch(distance):
        indices = torch.argsort(masked, dim=1, stable=True)[:, :k]
        result = torch.zeros((n, n), dtype=distance.dtype, device=distance.device)
        result.scatter_(1, indices, 1.0)
    else:
        indices = np.argsort(masked, axis=1, kind="stable")[:, :k]
        result = np.zeros((n, n), dtype=distance.dtype)
        np.put_along_axis(result, indices, 1.0, axis=1)
    return result * adjacency


def _local_span(distance: Any, adjacency: Any, k_value: Any) -> Any:
    mask = _knn_mask(distance, adjacency, k_value)
    count = _clamp(_sum(mask, 1), 1.0)
    return _clamp(_sum(mask * distance, 1) / count, EPS)


def _normalise_max(value: Any) -> Any:
    maximum = value.max()
    return value / _clamp(maximum, EPS)


def _context_array(
    context: Mapping[str, Any],
    name: str,
    state: Any,
    shape: tuple[int, ...],
    *,
    non_negative: bool = True,
) -> Any | None:
    if name not in context or context[name] is None:
        return None
    value = _as_array(context[name], state)
    if tuple(value.shape) != shape:
        raise EquationInputError(f"context {name!r} must have shape {shape}, got {tuple(value.shape)}")
    if not _all_finite(value) or (non_negative and _any(value < 0)):
        qualifier = "finite non-negative" if non_negative else "finite"
        raise EquationInputError(f"context {name!r} must contain {qualifier} values")
    return value


def _context_vector(context: Mapping[str, Any], name: str, state: Any, distance: Any, adjacency: Any) -> Any:
    if name in context and context[name] is not None:
        value = _as_array(context[name], state)
    elif name == "node_betweenness":
        value = _as_array(_compute_betweenness(distance, adjacency, edge=False), state)
    else:  # pragma: no branch - only the two centrality names call this helper
        value = _as_array(_compute_betweenness(distance, adjacency, edge=True), state)
    expected = (state.shape[0],) if name == "node_betweenness" else tuple(distance.shape)
    if tuple(value.shape) != tuple(expected):
        raise EquationInputError(f"context {name!r} must have shape {expected}, got {tuple(value.shape)}")
    if not _all_finite(value) or _any(value < 0):
        raise EquationInputError(f"context {name!r} must contain finite non-negative values")
    return _normalise_max(value)


def _compute_betweenness(distance: Any, adjacency: Any, *, edge: bool) -> np.ndarray:
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - context arrays are used in minimal installs
        kind = "edge_betweenness" if edge else "node_betweenness"
        raise EquationInputError(
            f"{kind} was not supplied and automatic computation requires networkx"
        ) from exc
    d = _to_numpy(distance)
    a = _to_numpy(adjacency).astype(bool)
    n = d.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(
        (i, j, float(d[i, j]))
        for i in range(n) for j in range(i + 1, n)
        if a[i, j] and d[i, j] > 0
    )
    approximation = min(200, n) if n > 300 or graph.number_of_edges() > 5000 else None
    if edge:
        values = nx.edge_betweenness_centrality(
            graph, k=approximation, normalized=True, weight="weight", seed=0
        )
        result = np.zeros((n, n), dtype=float)
        for (left, right), value in values.items():
            result[left, right] = value
            result[right, left] = value
    else:
        values = nx.betweenness_centrality(
            graph, k=approximation, normalized=True, weight="weight", seed=0
        )
        result = np.asarray([values[index] for index in range(n)], dtype=float)
    maximum = float(result.max(initial=0.0))
    return result / max(maximum, EPS)


def _cluster_labels(
    context: Mapping[str, Any], state: Any, distance: Any, adjacency: Any, cluster_count: Any
) -> Any:
    n = int(state.shape[0])
    if "cluster_labels" in context and context["cluster_labels"] is not None:
        labels_np = _to_numpy(context["cluster_labels"])
    else:
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError as exc:  # pragma: no cover - explicit context used in minimal installs
            raise EquationInputError(
                "cluster_labels were not supplied and automatic computation requires scikit-learn"
            ) from exc
        count = max(1, min(_integer(cluster_count, n + 1, "K_clusters"), n))
        d = _to_numpy(distance).copy()
        a = _to_numpy(adjacency).astype(bool)
        fill = float(d[a].max()) if np.any(a) else float(d.max())
        d[~a] = fill
        try:
            model = AgglomerativeClustering(n_clusters=count, metric="precomputed", linkage="average")
        except TypeError:  # scikit-learn < 1.2
            model = AgglomerativeClustering(n_clusters=count, affinity="precomputed", linkage="average")
        labels_np = model.fit_predict(d)
    labels_np = np.asarray(labels_np)
    if labels_np.shape != (n,) or not np.isfinite(labels_np).all():
        raise EquationInputError(f"context 'cluster_labels' must have shape ({n},) and finite values")
    return _as_array(labels_np, state)


def _radiation(state: Any, distance: Any, adjacency: Any) -> Any:
    p = _clamp(state, MIN_STATE)
    n = int(p.shape[0])
    closer = distance[:, None, :] < distance[:, :, None]
    eye = _eye(n, distance)
    valid = closer & ~eye[:, None, :] & ~eye[None, :, :]
    opportunities = _sum(p[None, None, :] * valid, 2)
    numerator = p[:, None] * p[None, :] * adjacency
    denominator = (p[:, None] + opportunities + 1.0e-8) * (
        p[:, None] + p[None, :] + opportunities + 1.0e-8
    )
    flow = numerator / denominator
    return _sum(flow, 0) - _sum(flow, 1)


def precompute_context(
    candidate_id: str,
    distance: Any,
    params: Mapping[str, Any] | Sequence[Any] = (),
    adjacency: Any | None = None,
) -> dict[str, Any]:
    """Compute immutable graph context once for repeated equation evaluation.

    The returned arrays use the same backend/device as ``distance``. Continuous
    state and parameter gradients are unaffected; ranks, neighbours,
    centralities, and clusters are structural quantities by definition.
    """

    spec = get_candidate(candidate_id)
    if _ndim(distance) != 2 or distance.shape[0] != distance.shape[1]:
        raise EquationInputError("distance must be a square matrix")
    n = int(distance.shape[0])
    state = _ones((n,), distance)
    _, d, mask_bool = _prepare_inputs(state, distance, adjacency)

    if isinstance(params, Mapping):
        parameter_map = dict(params)
    else:
        values = list(params)
        if values and len(values) != spec.parameter_count:
            raise EquationInputError(
                f"{spec.id} expects {spec.parameter_count} parameters, got {len(values)}"
            )
        parameter_map = dict(zip(spec.parameter_names, values))

    def structural_parameter(name: str) -> Any:
        if name not in parameter_map:
            raise EquationInputError(
                f"precomputing {spec.id} context requires fixed structural parameter {name!r}"
            )
        return _as_array(parameter_map[name], state)

    cid = spec.id
    result: dict[str, Any] = {}
    if cid in {"rank_based_diffusion", "rank_choice_gravity", "rank_tail_diffusion"}:
        result["rank_matrix"] = _rank_matrix(d, mask_bool)

    knn_parameter = None
    if cid == "knn_diffusion":
        knn_parameter = structural_parameter("K")
    elif cid in {"knn_limited_diffusion", "explore_return_diffusion", "knn_two_band_diffusion"}:
        knn_parameter = structural_parameter("k")
    if knn_parameter is not None:
        result["knn_mask"] = _knn_mask(d, mask_bool, knn_parameter)

    if cid in {"span_diffusion", "span_cluster_diffusion"}:
        result["local_span"] = _local_span(d, mask_bool, structural_parameter("k"))

    if cid == "betweenness_diffusion":
        result["node_betweenness"] = _as_array(
            _compute_betweenness(d, mask_bool, edge=False), state
        )
    if cid in {"edge_betweenness_diffusion", "corridor_diffusion", "betweenness_gravity"}:
        result["edge_betweenness"] = _as_array(
            _compute_betweenness(d, mask_bool, edge=True), state
        )

    if cid in {"cluster_diffusion", "span_cluster_diffusion"}:
        result["cluster_labels"] = _cluster_labels(
            {}, state, d, mask_bool, structural_parameter("K_clusters")
        )
    return result


def evaluate(
    candidate_id: str,
    state: Any,
    distance: Any,
    params: Mapping[str, Any] | Sequence[Any] = (),
    adjacency: Any | None = None,
    **context: Any,
) -> Any:
    """Evaluate one spatial interaction right-hand side.

    Parameters
    ----------
    candidate_id:
        Canonical ID, GraphRAG ID (for example ``"C9"``), or archived solver
        method name.
    state:
        One state vector with shape ``(nodes,)``.  A PyTorch tensor selects the
        differentiable PyTorch backend; other array-like values select NumPy.
    distance:
        Non-negative ``(nodes, nodes)`` distance matrix.
    params:
        Preferred form is ``{parameter_name: value}``; a sequence in the
        archived solver order is also accepted.  Node-scoped values may be
        scalars or length-N vectors.
    adjacency:
        Optional boolean/numeric edge mask.  The default is a fully connected
        graph without self-loops, matching the paper experiments.
    **context:
        Optional ``node_betweenness``, ``edge_betweenness``, or
        ``cluster_labels`` arrays.  Missing graph context is deterministically
        computed when networkx/scikit-learn is installed.
    """

    spec = get_candidate(candidate_id)
    p, d, mask_bool = _prepare_inputs(state, distance, adjacency)
    values = _prepare_params(spec, params, p)
    mask = _as_array(mask_bool, p)
    safe_d = _safe_distance(d)
    cid = spec.id

    if cid in {"classic_gravity_power", "classic_gravity_exponential"}:
        alpha, beta, gamma = values
        positive = _clamp(p, MIN_STATE)
        origin = positive**alpha
        destination = positive**beta
        kernel = (safe_d ** (-gamma) if cid.endswith("power") else _exp(-gamma * d)) * mask
        return destination * (kernel.T @ origin) - origin * (kernel @ destination)

    if cid in {"single_constrained_gravity_power", "single_constrained_gravity_exponential"}:
        beta, gamma = values
        positive = _clamp(p, MIN_STATE)
        attractiveness = positive**beta
        kernel = (safe_d ** (-gamma) if cid.endswith("power") else _exp(-gamma * d)) * mask
        row_sum = _clamp(kernel @ attractiveness, EPS)
        scale = positive / row_sum
        return attractiveness * (kernel.T @ scale) - positive

    if cid == "radiation":
        return _radiation(p, d, mask)

    if cid == "distance_weighted_diffusion":
        coefficient, eta = values
        return coefficient * _laplacian(safe_d ** (-eta) * mask, p)

    if cid == "rank_based_diffusion":
        coefficient, exponent = values
        ranks = _context_array(context, "rank_matrix", p, tuple(d.shape))
        if ranks is None:
            ranks = _rank_matrix(d, mask_bool)
        valid = ranks > 0
        log_rank = _where(valid, _log(_clamp(ranks, 1.0)), _as_array(0.0, p))
        log_weight = _clamp(-exponent * log_rank, -50.0, 50.0)
        rank_weight = _where(valid, _exp(log_weight), _as_array(0.0, p))
        weight = 0.5 * (rank_weight + rank_weight.T) * mask
        return coefficient * _laplacian(weight, p)

    if cid == "knn_diffusion":
        coefficient, k, eta = values
        neighbour_mask = _context_array(context, "knn_mask", p, tuple(d.shape))
        if neighbour_mask is None:
            neighbour_mask = _knn_mask(d, mask_bool, k)
        weight = _normalise_rows(safe_d ** (-eta) * neighbour_mask)
        return coefficient * _laplacian(weight, p)

    if cid == "saturation_limited_diffusion":
        coefficient, capacity, eta = values
        weight = _normalise_rows(safe_d ** (-eta) * mask)
        positive = _clamp(p, MIN_STATE)
        capacity = _clamp(capacity, MIN_STATE)
        transformed = positive / (1.0 + positive / capacity)
        return coefficient * (weight @ transformed - transformed)

    if cid == "accessibility_drift":
        coefficient, eta = values
        kernel = safe_d ** (-eta) * mask
        accessibility = _sum(kernel, 1)
        weight = kernel / _clamp(accessibility[:, None], EPS)
        return coefficient * p * (weight @ accessibility - accessibility)

    if cid == "local_global_diffusion":
        mixture, coefficient, global_coefficient, eta = values
        mixture = _clamp(mixture, 0.0, 1.0)
        local = _laplacian(safe_d ** (-eta) * mask, p)
        return (1.0 - mixture) * coefficient * local + mixture * global_coefficient * (_mean(p) - p)

    if cid == "rank_choice_gravity":
        (mixture,) = values
        mixture = _clamp(mixture, 0.0, 1.0)
        positive = _clamp(p, MIN_STATE)
        ranks = _context_array(context, "rank_matrix", p, tuple(d.shape))
        if ranks is None:
            ranks = _rank_matrix(d, mask_bool)
        valid = ranks > 0
        score = mixture * float(p.shape[0]) / _clamp(ranks, 1.0) + (1.0 - mixture)
        score = _where(valid, score, _as_array(0.0, p))
        scale = positive / _clamp(score @ positive, EPS)
        return positive * (score.T @ scale) - positive

    if cid == "betweenness_diffusion":
        coefficient, eta = values
        centrality = _context_vector(context, "node_betweenness", p, d, mask_bool)
        root = _sqrt(_clamp(centrality, 0.0) + EPS)
        weight = root[:, None] * root[None, :] * safe_d ** (-eta) * mask
        return coefficient * _laplacian(weight, p)

    if cid in {"edge_betweenness_diffusion", "corridor_diffusion", "betweenness_gravity"}:
        centrality = _context_vector(context, "edge_betweenness", p, d, mask_bool)
        if cid == "edge_betweenness_diffusion":
            delta, coefficient, eta = values
            weight = (centrality + EPS) ** delta * safe_d ** (-eta) * mask
            return coefficient * _laplacian(weight, p)
        if cid == "corridor_diffusion":
            enhancement, coefficient, eta = values
            valid = mask_bool & ~_eye(int(p.shape[0]), p)
            mean_edge = _clamp(centrality[valid].mean() if _any(valid) else _as_array(1.0, p), EPS)
            weight = (1.0 + enhancement * centrality / mean_edge) * safe_d ** (-eta) * mask
            return coefficient * _laplacian(weight, p)
        alpha, beta, coefficient, delta, gamma = values
        positive = _clamp(p, MIN_STATE)
        origin = positive**alpha
        destination = positive**beta
        kernel = (centrality + EPS) ** delta * safe_d ** (-gamma) * mask
        return coefficient * (destination * (kernel.T @ origin) - origin * (kernel @ destination))

    if cid in {"cluster_diffusion", "span_cluster_diffusion"}:
        if cid == "cluster_diffusion":
            within, between, coefficient, cluster_count, eta = values
            labels = _cluster_labels(context, p, d, mask_bool, cluster_count)
            cluster_weight = _where(labels[:, None] == labels[None, :], within, between)
            weight = cluster_weight * safe_d ** (-eta) * mask
        else:
            within, between, coefficient, k, eta, cluster_count = values
            labels = _cluster_labels(context, p, d, mask_bool, cluster_count)
            cluster_weight = _where(labels[:, None] == labels[None, :], within, between)
            span = _context_array(context, "local_span", p, (int(p.shape[0]),))
            if span is None:
                span = _local_span(d, mask_bool, k)
            pair_span = _sqrt(_clamp(span[:, None] * span[None, :], EPS))
            weight = (safe_d / (pair_span + EPS)) ** (-eta) * cluster_weight * mask
        return coefficient * _laplacian(weight, p)

    if cid in {"span_diffusion", "knn_limited_diffusion"}:
        coefficient, k, eta = values
        if cid == "span_diffusion":
            span = _context_array(context, "local_span", p, (int(p.shape[0]),))
            if span is None:
                span = _local_span(d, mask_bool, k)
            pair_span = _sqrt(_clamp(span[:, None] * span[None, :], EPS))
            weight = (safe_d / (pair_span + EPS)) ** (-eta) * mask
        else:
            neighbour = _context_array(context, "knn_mask", p, tuple(d.shape))
            if neighbour is None:
                neighbour = _knn_mask(d, mask_bool, k)
            weight = 0.5 * (neighbour + neighbour.T) * safe_d ** (-eta) * mask
        return coefficient * _laplacian(weight, p)

    if cid == "explore_return_diffusion":
        novelty_rate, offset, k, eta = values
        positive = _clamp(p, MIN_STATE)
        new_probability = 1.0 / (1.0 + novelty_rate * (_log(_as_array(float(p.shape[0]), p)) + offset))
        new_probability = _clamp(new_probability, 0.0, 1.0)
        neighbour = _context_array(context, "knn_mask", p, tuple(d.shape))
        if neighbour is None:
            neighbour = _knn_mask(d, mask_bool, k)
        return_weight = _normalise_rows(neighbour / (safe_d + EPS) * mask)
        explore_weight = _normalise_rows(safe_d ** (-eta) * mask)
        mixture = (1.0 - new_probability) * return_weight + new_probability * explore_weight
        return mixture.T @ positive - positive

    if cid == "dual_layer_diffusion":
        mixture, coefficient, local_eta, community_eta = values
        mixture = _clamp(mixture, 0.0, 1.0)
        weight = (mixture * safe_d ** (-local_eta) + (1.0 - mixture) * safe_d ** (-community_eta)) * mask
        return coefficient * _laplacian(weight, p)

    if cid == "two_scale_diffusion":
        amplitude, coefficient, threshold, near_eta, far_eta = values
        threshold = _clamp(threshold, EPS)
        far_amplitude = amplitude * threshold ** (far_eta - near_eta)
        kernel = _where(d < threshold, amplitude * safe_d ** (-near_eta), far_amplitude * safe_d ** (-far_eta))
        return coefficient * _laplacian(kernel * mask, p)

    if cid == "rank_tail_diffusion":
        mixture, tail_scale, cutoff, eta = values
        mixture = _clamp(mixture, 0.0, 1.0)
        positive = _clamp(p, MIN_STATE)
        cutoff_int = _integer(cutoff, int(p.shape[0]), "R")
        ranks = _context_array(context, "rank_matrix", p, tuple(d.shape))
        if ranks is None:
            ranks = _rank_matrix(d, mask_bool)
        valid = ranks > 0
        head = mixture * float(p.shape[0]) / _clamp(ranks, 1.0) + (1.0 - mixture)
        tail = tail_scale * safe_d ** (-eta)
        score = _where(ranks <= float(cutoff_int), head, tail)
        score = _where(valid, score, _as_array(0.0, p))
        scale = positive / _clamp(score @ positive, EPS)
        return positive * (score.T @ scale) - positive

    if cid == "state_adaptive_diffusion":
        baseline, state_coefficient, eta = values
        weight = _normalise_rows(safe_d ** (-eta) * mask)
        neighbour_state = weight @ p
        exchange = neighbour_state - p * _sum(weight, 1)
        return (baseline + state_coefficient * neighbour_state) * exchange

    if cid == "knn_two_band_diffusion":
        near, far, coefficient, k, eta = values
        neighbour = _context_array(context, "knn_mask", p, tuple(d.shape))
        if neighbour is None:
            neighbour = _knn_mask(d, mask_bool, k)
        ring = (neighbour > 0) | (neighbour.T > 0)
        multiplier = near * ring + far * (~ring)
        return coefficient * _laplacian(multiplier * safe_d ** (-eta) * mask, p)

    raise RuntimeError(f"Candidate {cid!r} is registered but has no evaluator")


__all__ = [
    "CandidateSpec",
    "EquationInputError",
    "evaluate",
    "get_candidate",
    "list_candidates",
    "load_registry",
    "precompute_context",
]
