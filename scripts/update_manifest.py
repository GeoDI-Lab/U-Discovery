"""Regenerate only the root release manifest using the validator's inventory.

Run after reviewing intentional release changes. Frozen CHECKSUMS.sha256 files
inside the archived datasets/results are never rewritten by this utility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from udiscovery.validate import release_files, sha256


def update_manifest(root: str | Path) -> Path:
    """Hash the same portable files that ``udiscovery validate`` inspects."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Release root is not a directory: {root}")
    manifest = root / "MANIFEST.sha256"
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in release_files(root)
        if path != manifest
    ]
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT, help="Release checkout root")
    args = parser.parse_args()
    print(update_manifest(args.root))


if __name__ == "__main__":
    main()
