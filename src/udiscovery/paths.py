from __future__ import annotations

from pathlib import Path
import os


def repository_root() -> Path:
    """Return the release artifact root without depending on the current directory.

    An editable install naturally resolves from the source tree. A wheel cannot
    redistribute the paper datasets as Python package data, so users may set
    ``UDISCOVERY_ROOT`` to a downloaded release directory when invoking
    artifact-level commands from a regular installation.
    """
    configured = os.environ.get("UDISCOVERY_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "artifacts/candidates.json").is_file():
            raise FileNotFoundError(
                f"UDISCOVERY_ROOT does not identify a release checkout: {root}"
            )
        return root
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "artifacts/candidates.json").is_file():
        return source_root
    raise FileNotFoundError(
        "Release data are not inside the installed wheel. Run from an editable "
        "checkout or set UDISCOVERY_ROOT to the downloaded U-Discovery release."
    )


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is not None:
        return (base / path).resolve()
    current_candidate = (Path.cwd() / path).resolve()
    if current_candidate.exists():
        return current_candidate
    return (repository_root() / path).resolve()
