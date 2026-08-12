from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .paths import repository_root
from .plotting import reproduce_public_figures


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


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _paper_tables(root: Path, output: Path) -> dict[str, dict[str, Any]]:
    registry = json.loads((root / "artifacts/candidates.json").read_text(encoding="utf-8"))
    candidates = registry["candidates"] if isinstance(registry, dict) else registry

    def rows_for(table_key: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in candidates:
            if table_key not in item.get("paper_rows", {}):
                continue
            rows.append(
                {
                    "paper_row": item["paper_rows"][table_key],
                    "candidate_id": item["id"],
                    "display_name": item["display_name"],
                    "formula": item["formula"],
                    "complexity": item.get("complexities", {}).get(table_key, ""),
                    "graphrag_output_id": item.get("graphrag_id") or "",
                    "legacy_solver": ";".join(item.get("legacy_solver_names", [])),
                }
            )
        return sorted(rows, key=lambda row: int(row["paper_row"]))

    candidate_fields = [
        "paper_row",
        "candidate_id",
        "display_name",
        "formula",
        "complexity",
        "graphrag_output_id",
        "legacy_solver",
    ]
    table1_path = _write_csv(
        output / "table1_candidates.csv", rows_for("table_1"), candidate_fields
    )
    table3_path = _write_csv(
        output / "table3_candidates.csv", rows_for("table_3"), candidate_fields
    )

    table2_source = root / "results/synthetic/table2_metrics.csv"
    table2 = pd.read_csv(table2_source)
    expected_table2_columns = {
        "candidate",
        "MSE_delta",
        "R2_delta",
        "Corr_delta",
        "MSE",
        "R2",
        "Corr",
        "Complexity",
    }
    if set(table2.columns) != expected_table2_columns or len(table2) != 12:
        raise ValueError(
            "results/synthetic/table2_metrics.csv must be the canonical 12-row manuscript source"
        )
    distance_row = table2.loc[table2["candidate"].eq("Distance-weighted diffusion")]
    if (
        len(distance_row) != 1
        or abs(float(distance_row.iloc[0]["MSE_delta"]) - 9.7443) > 5e-5
        or abs(float(distance_row.iloc[0]["MSE"]) - 4.3209) > 5e-5
        or float(distance_row.iloc[0]["Complexity"]) != 10.0
    ):
        raise ValueError("canonical Table 2 distance-diffusion row does not match the manuscript")
    table2_path = output / "table2_metrics.csv"
    shutil.copy2(table2_source, table2_path)

    table4_path = output / "table4_metrics.csv"
    shutil.copy2(root / "results/hennepin_2500m/table4_metrics.csv", table4_path)
    return {
        "table1": _artifact(table1_path),
        "table2": _artifact(table2_path),
        "table3": _artifact(table3_path),
        "table4": _artifact(table4_path),
    }


def _generated_artifacts(
    tables: dict[str, dict[str, Any]], figure_bundle: dict[str, Any]
) -> list[dict[str, Any]]:
    records = list(tables.values())
    for figure in figure_bundle["figures"].values():
        records.extend(figure["source_artifacts"])
        records.extend(figure["rendering"]["artifacts"])
    return sorted(records, key=lambda record: record["path"])


def reproduce_paper(output_dir: str | Path | None = None) -> Path:
    """Reproduce all public paper tables and numerical/source figure panels."""
    root = repository_root()
    if output_dir is None:
        output = root / "reproduced/paper"
    else:
        requested = Path(output_dir).expanduser()
        output = (
            requested.resolve()
            if requested.is_absolute()
            else (Path.cwd() / requested).resolve()
        )
    output.mkdir(parents=True, exist_ok=True)

    tables = _paper_tables(root, output)
    figure_bundle = reproduce_public_figures(root, output)
    manifest = {
        "schema_version": 2,
        "scope": (
            "Tables 1-4 plus machine-readable and optionally rendered public "
            "numerical/source panels for Figures 4-7"
        ),
        "tables": tables,
        "figure_renderer": figure_bundle["renderer"],
        "figures": figure_bundle["figures"],
        "generated_artifacts": _generated_artifacts(tables, figure_bundle),
        "explicit_exclusions": [
            "Figure 6(a), which requires excluded individual trajectories",
            "Figure 8 and every 1.5 km or 2.0 km input/result",
            "Raw mobility records, device identifiers, and origin/destination coordinates",
        ],
        "notes": [
            "Machine-readable source artifacts are generated even when Matplotlib is unavailable.",
            "Figure 4 uses the verified Figure-4-only realization under figure_sources/figure4; it is not the Table 2 training trajectory.",
            "Figure 5 and Table 2 use the canonical 12-row manuscript source, not table2_metrics_legacy.csv.",
            "Figure 6 regenerates public panels (b-c) only from ordered aggregate grid data.",
            "Figure 7 is regenerated from the packaged selection, heatmap, and frontier sources.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output
