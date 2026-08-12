#!/usr/bin/env python3
"""Convert trusted, local PyTorch result dictionaries to non-executable files.

This utility must only be used on artifacts produced by the authors. PyTorch's
legacy loader can execute pickle payloads, so it is intentionally not called by
the public reproduction commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "This author-only migration utility requires PyTorch; install the training extra."
        ) from exc
    return torch


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def export_reconstructions(source: Path, output: Path, mapping_path: Path) -> None:
    torch = _torch()
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Expected a dictionary of reconstruction tensors")
    arrays = {}
    mapping = {}
    for index, (name, value) in enumerate(payload.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Reconstruction {name!r} is not a tensor")
        key = f"run_{index:03d}_{_slug(str(name))}"
        arrays[key] = value.detach().cpu().numpy()
        mapping[key] = str(name)
    np.savez_compressed(output, **arrays)
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")


def export_parameter_runs(source: Path, output: Path, metadata_path: Path, csv_path: Path) -> None:
    torch = _torch()
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Expected a dictionary of parameter runs")
    arrays = {}
    metadata = {}
    rows = []
    run_number = 0
    for source_key, value in payload.items():
        if source_key == "true_dynamics" or not isinstance(value, dict):
            continue
        run_id = f"run_{run_number:03d}"
        run_number += 1
        theta_int = value.get("theta_int")
        if isinstance(theta_int, dict):
            for scope in ("node", "global"):
                tensor = theta_int.get(scope)
                if isinstance(tensor, torch.Tensor):
                    arrays[f"{run_id}_theta_int_{scope}"] = tensor.detach().cpu().numpy()
        elif isinstance(theta_int, torch.Tensor):
            arrays[f"{run_id}_theta_int"] = theta_int.detach().cpu().numpy()
        theta_osc = value.get("theta_osc")
        if isinstance(theta_osc, torch.Tensor):
            arrays[f"{run_id}_theta_osc"] = theta_osc.detach().cpu().numpy()
        combo = value.get("combo_info") if isinstance(value.get("combo_info"), dict) else {}
        metadata[run_id] = {
            "source_key": str(source_key),
            "val_loss": float(value.get("val_loss", float("nan"))),
            "epoch": int(value.get("epoch", -1)),
            "stage": str(value.get("stage", "")),
            "updated_at": str(value.get("updated_at", "")),
            "combo_info": combo,
        }
        int_global = arrays.get(f"{run_id}_theta_int_global")
        row = {
            "run_id": run_id,
            "source_key": str(source_key),
            "seed": _seed_from_text(str(source_key)),
            "epoch": metadata[run_id]["epoch"],
            "val_loss": metadata[run_id]["val_loss"],
        }
        if int_global is not None:
            for idx, scalar in enumerate(np.asarray(int_global).reshape(-1), start=1):
                row[f"interaction_parameter_{idx}"] = float(scalar)
        rows.append(row)
    np.savez_compressed(output, **arrays)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_from_text(value: str) -> int | str:
    match = re.search(r"seed_(\d+)", value)
    return int(match.group(1)) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("reconstructions", "parameters"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    print(
        "WARNING: this command uses pickle-capable torch.load. Continue only for "
        "an author-produced artifact whose SHA-256 you have independently verified."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "reconstructions":
        if args.mapping is None:
            parser.error("--mapping is required for reconstructions")
        export_reconstructions(args.source, args.output, args.mapping)
    else:
        if args.metadata is None or args.csv is None:
            parser.error("--metadata and --csv are required for parameters")
        export_parameter_runs(args.source, args.output, args.metadata, args.csv)


if __name__ == "__main__":
    main()
