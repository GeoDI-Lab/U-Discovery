"""Portable, configuration-driven UrbanDE-Net training."""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_config, resolved_dataset_path
from .data import load_dataset
from .paths import resolve_path


DISCRETE_PARAMETER_NAMES = frozenset({"k", "K", "K_clusters", "R"})


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Training requires PyTorch. Install the training extra with "
            "`python -m pip install -e '.[train]'`."
        ) from exc
    return torch


def select_device(requested: str):
    """Resolve a requested CPU/CUDA device and fail clearly if unavailable."""

    torch = _require_torch()
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({requested}) but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} was requested but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
    return device


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable CPU/CUDA runs."""

    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def build_history_windows(states, current_indices, history_length: int):
    """Gather ``(batch, history, node)`` windows ending at current indices."""

    torch = _require_torch()
    if states.ndim != 2:
        raise ValueError(f"states must have shape (time, node), got {tuple(states.shape)}")
    if history_length < 2:
        raise ValueError("history_length must be at least 2")
    indices = torch.as_tensor(current_indices, device=states.device, dtype=torch.long).reshape(-1)
    if indices.numel() == 0:
        return states.new_empty((0, history_length, states.shape[1]))
    if int(indices.min()) < history_length - 1 or int(indices.max()) >= states.shape[0]:
        raise ValueError("current indices do not admit complete history windows")
    offsets = torch.arange(
        -history_length + 1, 1, device=states.device, dtype=torch.long
    )
    return states[indices[:, None] + offsets[None, :]]


def save_model_weights_npz(model, path: str | Path) -> Path:
    """Save a PyTorch state dict as numeric arrays without pickle."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    np.savez_compressed(output, **arrays)
    return output


def load_model_weights_npz(model, path: str | Path, *, strict: bool = True) -> tuple[list[str], list[str]]:
    """Load a numeric NPZ state dict into an existing compatible model."""

    torch = _require_torch()
    expected = model.state_dict()
    with np.load(Path(path), allow_pickle=False) as archive:
        available = set(archive.files)
        expected_names = set(expected)
        missing = sorted(expected_names - available)
        unexpected = sorted(available - expected_names)
        if strict and (missing or unexpected):
            raise ValueError(f"model weight keys mismatch; missing={missing}, unexpected={unexpected}")
        with torch.no_grad():
            for name in sorted(expected_names & available):
                target = expected[name]
                source = torch.as_tensor(
                    archive[name], device=target.device, dtype=target.dtype
                )
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"model weight {name!r} has shape {tuple(source.shape)}, "
                        f"expected {tuple(target.shape)}"
                    )
                target.copy_(source)
    return missing, unexpected


def _parameter_layout(spec, mode: str) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    normalized = mode.lower()
    if normalized in {"localized", "local", "heter", "heterogeneous", "node"}:
        effective = dict(zip(spec.parameter_names, spec.parameter_scopes))
        label = "localized"
    elif normalized in {"global", "homo", "homogeneous"}:
        effective = {name: "global" for name in spec.parameter_names}
        label = "global"
    else:
        raise ValueError("training.parameter_mode must be 'localized' or 'global'")
    train_names = tuple(
        name for name in spec.parameter_names if name not in DISCRETE_PARAMETER_NAMES
    )
    train_scopes = tuple(effective[name] for name in train_names)
    effective["__mode__"] = label
    return train_names, train_scopes, effective


def _bounded_parameters(candidate_id: str, names: Sequence[str]) -> tuple[str, ...]:
    bounded = {"Gamma"}.intersection(names)
    if candidate_id in {"rank_choice_gravity", "rank_tail_diffusion"} and "lambda" in names:
        bounded.add("lambda")
    return tuple(sorted(bounded))


def _model_from_config(
    *,
    dataset,
    spec,
    parameter_names: Sequence[str],
    parameter_scopes: Sequence[str],
    initial_parameters: Mapping[str, Any],
    model_config: Mapping[str, Any],
    seed: int,
    device,
):
    torch = _require_torch()
    from .model import UrbanDENet

    history_length = int(model_config.get("history_length", 2))
    harmonics = int(model_config.get("temporal_harmonics", 1))
    adjacency = torch.as_tensor(dataset.adjacency, dtype=torch.float32, device=device)
    distance = torch.as_tensor(dataset.distance, dtype=torch.float32, device=device)
    hidden = model_config.get("head_hidden", [128, 64])
    if not isinstance(hidden, (list, tuple)) or not hidden:
        raise ValueError("model.head_hidden must be a non-empty list of widths")
    model = UrbanDENet(
        num_nodes=dataset.num_nodes,
        history_length=history_length,
        parameter_names=parameter_names,
        parameter_scopes=parameter_scopes,
        adjacency_mask=adjacency,
        distance_matrix=distance,
        temporal_harmonics=harmonics,
        encoder_dim=int(model_config.get("encoder_dim", 128)),
        head_hidden=tuple(int(width) for width in hidden),
        encoder_dropout=float(model_config.get("encoder_dropout", 0.1)),
        head_dropout=float(model_config.get("head_dropout", 0.1)),
        residual_channels=int(model_config.get("residual_channels", 32)),
        dilation_channels=int(
            model_config.get("dilation_channels", model_config.get("residual_channels", 32))
        ),
        skip_channels=int(model_config.get("skip_channels", 64)),
        end_channels=int(model_config.get("end_channels", 128)),
        kernel_size=int(model_config.get("kernel_size", 2)),
        blocks=int(model_config.get("blocks", 4)),
        layers=int(model_config.get("layers", 2)),
        adaptive_adjacency=bool(model_config.get("adaptive_adjacency", True)),
        initial_parameters=initial_parameters,
        bounded_parameter_names=_bounded_parameters(spec.id, parameter_names),
        init_seed=seed,
    )
    return model.to(device)


def _torch_context(context: Mapping[str, Any], *, device, dtype) -> dict[str, Any]:
    torch = _require_torch()
    return {
        name: torch.as_tensor(value, device=device, dtype=dtype)
        for name, value in context.items()
    }


def _select_parameters(output, batch_index: int, fixed_parameters: Mapping[str, Any]) -> dict[str, Any]:
    selected = {
        name: value[batch_index]
        for name, value in output.interaction_parameters.items()
    }
    selected.update(fixed_parameters)
    return selected


def _temporal_forcing(output, current_indices, frequency: float):
    torch = _require_torch()
    harmonics = torch.arange(
        1,
        output.temporal_sin.shape[-1] + 1,
        device=output.temporal_sin.device,
        dtype=output.temporal_sin.dtype,
    ).view(1, 1, -1)
    times = current_indices.to(dtype=output.temporal_sin.dtype).view(-1, 1, 1)
    phase = 2.0 * torch.pi * float(frequency) * times * harmonics
    return (
        output.temporal_sin * torch.sin(phase)
        + output.temporal_cos * torch.cos(phase)
    ).sum(dim=-1)


def _predict_delta(
    model,
    histories,
    current_indices,
    *,
    candidate_id: str,
    distance,
    adjacency,
    fixed_parameters: Mapping[str, Any],
    structural_context: Mapping[str, Any],
    frequency: float,
    dt: float,
):
    torch = _require_torch()
    from .equations import evaluate

    output = model(histories)
    spatial_rows = []
    for batch_index in range(histories.shape[0]):
        spatial_rows.append(
            evaluate(
                candidate_id,
                histories[batch_index, -1],
                distance,
                _select_parameters(output, batch_index, fixed_parameters),
                adjacency=adjacency,
                **structural_context,
            )
        )
    spatial = torch.stack(spatial_rows, dim=0)
    temporal = _temporal_forcing(output, current_indices, frequency)
    return float(dt) * (spatial + temporal), output


def _copy_state_to_cpu(model) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
    }


def _restore_state(model, arrays: Mapping[str, np.ndarray]) -> None:
    torch = _require_torch()
    state = model.state_dict()
    with torch.no_grad():
        for name, target in state.items():
            target.copy_(torch.as_tensor(arrays[name], device=target.device, dtype=target.dtype))


def _parameter_outputs(
    model,
    states,
    current_indices,
    *,
    batch_size: int,
    fixed_parameters: Mapping[str, Any],
    parameter_names: Sequence[str],
    parameter_scopes: Mapping[str, str],
    train_end: int,
    val_end: int,
) -> dict[str, np.ndarray]:
    torch = _require_torch()
    model.eval()
    collected: dict[str, list[np.ndarray]] = {name: [] for name in parameter_names}
    temporal_sin: list[np.ndarray] = []
    temporal_cos: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, current_indices.numel(), batch_size):
            batch_indices = current_indices[start : start + batch_size]
            histories = build_history_windows(states, batch_indices, model.history_length)
            output = model(histories)
            for name in model.parameter_names:
                collected[name].append(output.interaction_parameters[name].detach().cpu().numpy())
            temporal_sin.append(output.temporal_sin.detach().cpu().numpy())
            temporal_cos.append(output.temporal_cos.detach().cpu().numpy())

    payload: dict[str, np.ndarray] = {
        "window_current_indices": current_indices.detach().cpu().numpy().astype(np.int64),
        "window_target_indices": (current_indices + 1).detach().cpu().numpy().astype(np.int64),
        "parameter_names": np.asarray(parameter_names, dtype="U"),
        "parameter_scopes": np.asarray([parameter_scopes[name] for name in parameter_names], dtype="U"),
        "temporal_sin": np.concatenate(temporal_sin, axis=0),
        "temporal_cos": np.concatenate(temporal_cos, axis=0),
    }
    targets = payload["window_target_indices"]
    payload["split_code"] = np.where(targets < train_end, 0, np.where(targets < val_end, 1, 2)).astype(np.int8)
    for name in parameter_names:
        if name in fixed_parameters:
            value = fixed_parameters[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            payload[f"interaction__{name}"] = np.asarray(value)
        else:
            payload[f"interaction__{name}"] = np.concatenate(collected[name], axis=0)
    return payload


def _summary(payload: Mapping[str, np.ndarray], parameter_names: Sequence[str], scopes: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in parameter_names:
        values = np.asarray(payload[f"interaction__{name}"], dtype=float)
        result[name] = {
            "scope": scopes[name],
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def fit_candidate(
    config_path: str | Path,
    *,
    device_override: str | None = None,
    epochs_override: int | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Fit one configured equation with UrbanDE-Net history-window inference."""

    torch = _require_torch()
    from .equations import get_candidate, precompute_context

    config, config_file = load_config(config_path)
    dataset = load_dataset(resolved_dataset_path(config, config_file))
    training: dict[str, Any] = config["training"]
    model_config: dict[str, Any] = config.get("model", {})
    seed = int(training.get("seed", 42))
    deterministic = bool(training.get("deterministic", True))
    set_seed(seed, deterministic)
    device = select_device(device_override or str(training.get("device", "auto")))

    candidate_id = str(training.get("candidate_id", "single_constrained_gravity_power"))
    spec = get_candidate(candidate_id)
    parameter_mode = str(training.get("parameter_mode", "localized"))
    train_names, train_scopes, effective_scopes = _parameter_layout(spec, parameter_mode)
    resolved_mode = effective_scopes.pop("__mode__")
    initial_parameters = training.get("initial_parameters", {})
    if not isinstance(initial_parameters, Mapping):
        raise ValueError("training.initial_parameters must be an object")

    fixed_values: dict[str, float] = {}
    for name in spec.parameter_names:
        if name in DISCRETE_PARAMETER_NAMES:
            if name not in initial_parameters:
                raise ValueError(
                    f"Candidate {candidate_id} requires fixed structural parameter {name!r} "
                    "in training.initial_parameters"
                )
            fixed_values[name] = float(initial_parameters[name])

    history_length = int(model_config.get("history_length", 2))
    if history_length >= dataset.train_end:
        raise ValueError(
            f"history_length={history_length} leaves no training transitions before "
            f"train_end={dataset.train_end}"
        )
    harmonics = int(model_config.get("temporal_harmonics", 1))
    if harmonics < 1:
        raise ValueError("model.temporal_harmonics must be at least 1")

    states = torch.as_tensor(dataset.states, dtype=torch.float32, device=device)
    distance = torch.as_tensor(dataset.distance, dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(dataset.adjacency, dtype=torch.float32, device=device)
    fixed_parameters = {
        name: torch.as_tensor(value, dtype=states.dtype, device=device)
        for name, value in fixed_values.items()
    }

    context_parameters = dict(initial_parameters)
    structural_numpy = precompute_context(
        candidate_id,
        dataset.distance,
        context_parameters,
        adjacency=dataset.adjacency,
    )
    structural_context = _torch_context(
        structural_numpy, device=device, dtype=states.dtype
    )

    model = _model_from_config(
        dataset=dataset,
        spec=spec,
        parameter_names=train_names,
        parameter_scopes=train_scopes,
        initial_parameters=initial_parameters,
        model_config=model_config,
        seed=seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1.0e-3)),
        weight_decay=float(training.get("weight_decay", 1.0e-5)),
    )
    epochs = int(epochs_override if epochs_override is not None else training.get("epochs", 1000))
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    batch_size = int(training.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    patience = int(training.get("patience", 80))
    min_delta = float(training.get("min_delta", 1.0e-6))
    clip_norm = float(training.get("clip_norm", 1.0))
    frequency = float(training.get("forcing_frequency", 1.0 / 96.0))
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("training.forcing_frequency must be positive and finite")

    train_indices = torch.arange(
        history_length - 1, dataset.train_end - 1, device=device, dtype=torch.long
    )
    validation_indices = torch.arange(
        max(history_length - 1, dataset.train_end - 1),
        dataset.val_end - 1,
        device=device,
        dtype=torch.long,
    )
    if train_indices.numel() == 0 or validation_indices.numel() == 0:
        raise ValueError("dataset splits do not contain usable history-window transitions")

    def run_batches(indices, *, train: bool, epoch: int) -> float:
        model.train(train)
        if train:
            generator = torch.Generator().manual_seed(seed + epoch)
            permutation = torch.randperm(indices.numel(), generator=generator).to(device)
            ordered = indices[permutation]
        else:
            ordered = indices
        total = 0.0
        count = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for start in range(0, ordered.numel(), batch_size):
                batch_indices = ordered[start : start + batch_size]
                histories = build_history_windows(states, batch_indices, history_length)
                targets = states[batch_indices + 1] - states[batch_indices]
                if train:
                    optimizer.zero_grad(set_to_none=True)
                prediction, _ = _predict_delta(
                    model,
                    histories,
                    batch_indices,
                    candidate_id=candidate_id,
                    distance=distance,
                    adjacency=adjacency,
                    fixed_parameters=fixed_parameters,
                    structural_context=structural_context,
                    frequency=frequency,
                    dt=dataset.dt,
                )
                loss = torch.mean((prediction - targets) ** 2)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss for {candidate_id} at epoch {epoch + 1}"
                    )
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                size = int(batch_indices.numel())
                total += float(loss.detach()) * size
                count += size
        return total / count

    best_state: dict[str, np.ndarray] | None = None
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    history: list[tuple[int, float, float]] = []
    for epoch in range(epochs):
        training_loss = run_batches(train_indices, train=True, epoch=epoch)
        validation_loss = run_batches(validation_indices, train=False, epoch=epoch)
        history.append((epoch + 1, training_loss, validation_loss))
        if validation_loss < best_validation - min_delta:
            best_validation = validation_loss
            best_epoch = epoch + 1
            best_state = _copy_state_to_cpu(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    _restore_state(model, best_state)

    if output_dir is not None:
        supplied_output = Path(output_dir).expanduser()
        run_root = (
            supplied_output.resolve()
            if supplied_output.is_absolute()
            else (Path.cwd() / supplied_output).resolve()
        )
    else:
        run_root = resolve_path(config.get("output_dir", "outputs"), base=config_file.parent)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / f"{config['name']}_{candidate_id}_seed{seed}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = run_root / f"{config['name']}_{candidate_id}_seed{seed}_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    save_model_weights_npz(model, run_dir / "model_weights.npz")
    all_indices = torch.arange(
        history_length - 1,
        dataset.num_timesteps - 1,
        device=device,
        dtype=torch.long,
    )
    output_scopes = {name: effective_scopes[name] for name in spec.parameter_names}
    for name in fixed_values:
        output_scopes[name] = "fixed"
    parameter_payload = _parameter_outputs(
        model,
        states,
        all_indices,
        batch_size=batch_size,
        fixed_parameters=fixed_parameters,
        parameter_names=spec.parameter_names,
        parameter_scopes=output_scopes,
        train_end=dataset.train_end,
        val_end=dataset.val_end,
    )
    np.savez_compressed(run_dir / "fitted_parameters.npz", **parameter_payload)
    if structural_numpy:
        np.savez_compressed(run_dir / "structural_context.npz", **structural_numpy)
    np.savetxt(
        run_dir / "training_history.csv",
        np.asarray(history, dtype=float),
        delimiter=",",
        header="epoch,training_mse_delta,validation_mse_delta",
        comments="",
    )

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": str(config_file),
        "dataset": str(dataset.path),
        "candidate_id": candidate_id,
        "parameter_mode": resolved_mode,
        "parameter_names": list(spec.parameter_names),
        "parameter_scopes": output_scopes,
        "fixed_parameters": fixed_values,
        "fitted_parameter_summary": _summary(
            parameter_payload, spec.parameter_names, output_scopes
        ),
        "structural_context": sorted(structural_numpy),
        "architecture": model.architecture_metadata(),
        "history_window_semantics": "window ends at current state; target is the next-state delta",
        "forcing_frequency_cycles_per_step": frequency,
        "best_epoch": best_epoch,
        "validation_mse_delta": best_validation,
        "epochs_completed": len(history),
        "seed": seed,
        "deterministic": deterministic,
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pid": os.getpid(),
        "artifacts": {
            "model_weights": "model_weights.npz",
            "fitted_parameters": "fitted_parameters.npz",
            "training_history": "training_history.csv",
            "structural_context": "structural_context.npz" if structural_numpy else None,
        },
        "serialization": "numeric NPZ only; no pickle or torch.save",
    }
    try:
        root = config_file.parent.parent
        manifest["configuration"] = str(config_file.relative_to(root))
        manifest["dataset"] = str(dataset.path.relative_to(root))
    except ValueError:
        pass
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


__all__ = [
    "build_history_windows",
    "fit_candidate",
    "load_model_weights_npz",
    "save_model_weights_npz",
    "select_device",
    "set_seed",
]
