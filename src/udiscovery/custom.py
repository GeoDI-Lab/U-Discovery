"""User-supplied, differentiable equations for the end-to-end discovery workflow.

Equation files are trusted Python programs, not a sandboxed data format. Loading
one executes its top-level code with the current user's permissions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import importlib.util
import math
from numbers import Integral, Real
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .data import Dataset


_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_PARAMETER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class Equation:
    """One candidate interaction equation and its trainable coefficients.

    ``function(state, distance, parameters, *, adjacency)`` receives float32
    PyTorch tensors and must return a tensor of shape ``(nodes,)`` on the same
    device. Use PyTorch operations throughout to preserve parameter gradients.
    Treat all input tensors and the parameter mapping as read-only; use a clone
    if an intermediate calculation needs an in-place operation.
    ``parameters`` maps names to positive scalar initial values; ``scopes`` may
    select ``"node"`` for a coefficient per node or ``"global"`` (the default)
    for one shared coefficient. Names in ``bounded_parameters`` are constrained
    to (0, 1), with initial values strictly inside that interval. There are no
    reserved parameter names. An empty parameter mapping is allowed.

    ``complexity`` is the positive integer used in complexity-aware ranking.
    """

    id: str
    function: Callable
    parameters: Mapping[str, float]
    complexity: int
    scopes: Mapping[str, str] = field(default_factory=dict)
    name: str = ""
    formula: str = ""
    bounded_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_definition(self)
        # Detach metadata from dictionaries that the equation author may reuse.
        object.__setattr__(self, "parameters", {key: float(value) for key, value in self.parameters.items()})
        object.__setattr__(self, "scopes", dict(self.scopes))
        object.__setattr__(self, "complexity", int(self.complexity))
        object.__setattr__(self, "bounded_parameters", tuple(self.bounded_parameters))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameters)

    @property
    def parameter_scopes(self) -> tuple[str, ...]:
        return tuple(self.scopes.get(name, "global") for name in self.parameters)

    @property
    def display_name(self) -> str:
        return self.name or self.id


def _validate_definition(equation: Equation) -> None:
    if not isinstance(equation.id, str) or not _ID_PATTERN.fullmatch(equation.id):
        raise ValueError("Equation id must start with a letter or digit and contain only letters, digits, '_' or '-'")
    prefix = f"Equation {equation.id!r}"
    if not callable(equation.function):
        raise ValueError(f"{prefix}: function must be callable")
    if isinstance(equation.complexity, bool) or not isinstance(equation.complexity, Integral) or equation.complexity < 1:
        raise ValueError(f"{prefix}: complexity must be a positive integer")
    if not isinstance(equation.name, str) or not isinstance(equation.formula, str):
        raise ValueError(f"{prefix}: name and formula must be strings")
    if not isinstance(equation.parameters, Mapping):
        raise ValueError(f"{prefix}: parameters must map names to positive scalar initial values")
    for name, value in equation.parameters.items():
        if not isinstance(name, str) or not _PARAMETER_PATTERN.fullmatch(name):
            raise ValueError(f"{prefix}: invalid parameter name {name!r}; use letters, digits and '_' and do not start with a digit")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{prefix}: initial value for parameter {name!r} must be a finite positive scalar")
    if not isinstance(equation.scopes, Mapping):
        raise ValueError(f"{prefix}: scopes must map parameter names to 'node' or 'global'")
    for name, scope in equation.scopes.items():
        if name not in equation.parameters:
            raise ValueError(f"{prefix}: scope specified for unknown parameter {name!r}")
        if not isinstance(scope, str) or scope not in {"node", "global"}:
            raise ValueError(f"{prefix}: scope for {name!r} must be 'node' or 'global'")
    if not isinstance(equation.bounded_parameters, (tuple, list)):
        raise ValueError(f"{prefix}: bounded_parameters must be a tuple or list of parameter names")
    seen: set[str] = set()
    for name in equation.bounded_parameters:
        if not isinstance(name, str) or name not in equation.parameters:
            raise ValueError(f"{prefix}: unknown bounded parameter {name!r}")
        if name in seen:
            raise ValueError(f"{prefix}: duplicate bounded parameter {name!r}")
        seen.add(name)
        if not 0 < equation.parameters[name] < 1:
            raise ValueError(f"{prefix}: bounded parameter {name!r} must have an initial value strictly between 0 and 1")


def load_equations(path: str | Path) -> list[Equation]:
    """Execute a trusted Python file and validate its nonempty ``EQUATIONS``.

    Relative sibling imports such as ``from .helpers import diffusion`` are
    supported through a private package namespace, without changing sys.path.
    This executes arbitrary Python; only load equation files you trust.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Equation file does not exist: {source}")
    if source.suffix.lower() != ".py":
        raise ValueError(f"Equation file must be a Python .py file: {source}")
    package_name = f"_udiscovery_equations_{uuid4().hex}"
    module_name = f"{package_name}.equations"
    package = ModuleType(package_name)
    package.__path__ = [str(source.parent)]
    package.__package__ = package_name
    package.__spec__ = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load equation file: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    sys.modules[module_name] = module
    try:
        # Compiling the current bytes avoids stale bytecode when an author
        # rapidly edits and reloads an equation file with the same file size.
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        equations = getattr(module, "EQUATIONS", None)
        if not isinstance(equations, (list, tuple)) or not equations:
            raise ValueError("export a nonempty list or tuple named EQUATIONS")
        seen: set[str] = set()
        for index, equation in enumerate(equations):
            if not isinstance(equation, Equation):
                raise ValueError(f"EQUATIONS[{index}] must be an Equation instance")
            _validate_definition(equation)
            # These IDs become output directories, including on Windows/macOS.
            if equation.id.casefold() in seen:
                raise ValueError(f"duplicate equation id {equation.id!r} (IDs must also be unique ignoring case)")
            seen.add(equation.id.casefold())
        return list(equations)
    except Exception as exc:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]
        raise ValueError(f"Could not load equations from {source}: {exc}") from exc
    # Successful modules remain registered for lazy relative imports and
    # normal Python class/function metadata; sys.path is never modified.


def validate_equation(equation: Equation, dataset: Dataset, device: object) -> None:
    """Check output and parameter autograd connectivity on real inputs.

    Uses the complete graph at the first and last training observations so
    functions can rely on graph structure or node-specific constants. This is
    an early compatibility check, not a proof of stability for all inputs.
    Each declared parameter gets an independent leaf tensor so one coefficient
    cannot conceal a detached or unused coefficient. A connected zero gradient
    is valid, for example for a conservation law or a locally flat derivative.
    """
    if not isinstance(equation, Equation):
        raise ValueError("Expected an Equation instance")
    _validate_definition(equation)
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Custom equation validation requires PyTorch; install u-discovery[train]") from exc

    nodes = dataset.num_nodes
    if nodes < 1 or dataset.num_timesteps < 1:
        raise ValueError("Equation validation requires a nonempty dataset")
    time_indices = sorted({0, max(0, min(dataset.train_end - 1, dataset.num_timesteps - 1))})
    for time_index in time_indices:
        prefix = f"Equation {equation.id!r} at training timestep {time_index}"
        # Fresh copies prevent in-place user code from changing the dataset or
        # one validation sample from changing another.
        state = torch.tensor(dataset.states[time_index, :nodes], dtype=torch.float32, device=device, requires_grad=True)
        distance = torch.tensor(dataset.distance[:nodes, :nodes], dtype=torch.float32, device=device)
        adjacency = torch.tensor(dataset.adjacency[:nodes, :nodes], dtype=torch.float32, device=device)
        parameters = {
            name: torch.full((nodes,) if scope == "node" else (), equation.parameters[name], dtype=state.dtype, device=state.device, requires_grad=True)
            for name, scope in zip(equation.parameter_names, equation.parameter_scopes)
        }
        original_tensors = {
            "state": state.detach().clone(),
            "distance": distance.clone(),
            "adjacency": adjacency.clone(),
        }
        # Keep the declared leaves separately: mutating the mapping inside a
        # user function must not conceal an ignored parameter in the audit.
        parameter_leaves = tuple(parameters.values())
        try:
            with torch.enable_grad():
                result = equation.function(state, distance, parameters, adjacency=adjacency)
        except Exception as exc:
            raise ValueError(f"{prefix}: function(state, distance, parameters, *, adjacency) failed: {exc}") from exc
        for name, tensor in (("state", state), ("distance", distance), ("adjacency", adjacency)):
            if not torch.equal(tensor.detach(), original_tensors[name]):
                raise ValueError(f"{prefix}: function modified its {name} input in place; inputs are read-only, so clone tensors before modifying them")
        if tuple(parameters) != equation.parameter_names or any(
            parameters.get(name) is not leaf
            for name, leaf in zip(equation.parameter_names, parameter_leaves)
        ):
            raise ValueError(f"{prefix}: function modified the parameter mapping; treat it as read-only")
        if not isinstance(result, torch.Tensor):
            raise ValueError(f"{prefix}: function must return a PyTorch Tensor, got {type(result).__name__}")
        if tuple(result.shape) != (nodes,):
            raise ValueError(f"{prefix}: output must have shape ({nodes},), got {tuple(result.shape)}")
        if result.dtype != state.dtype:
            raise ValueError(f"{prefix}: output dtype must match state dtype {state.dtype}, got {result.dtype}")
        if result.device != state.device:
            raise ValueError(f"{prefix}: output device must match state device {state.device}, got {result.device}")
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{prefix}: output contains non-finite values")
        if not parameters:
            continue
        if not result.requires_grad:
            raise ValueError(f"{prefix}: output is detached from all declared parameters; use PyTorch operations without .detach(), .item() or NumPy conversion")
        try:
            gradients = torch.autograd.grad(result, parameter_leaves, grad_outputs=torch.ones_like(result), allow_unused=True)
        except RuntimeError as exc:
            raise ValueError(f"{prefix}: could not differentiate output with respect to declared parameters: {exc}") from exc
        for name, gradient in zip(parameters, gradients):
            if gradient is None:
                raise ValueError(f"{prefix}: parameter {name!r} is unused or detached from the output; every declared parameter must stay connected to the PyTorch graph")
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError(f"{prefix}: parameter {name!r} has a non-finite gradient")


__all__ = ["Equation", "load_equations", "validate_equation"]
