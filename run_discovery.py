"""Fit and rank your equations: python run_discovery.py --data data.npz --equations equations.py."""

from pathlib import Path
import sys

# Works from a source checkout, including when called from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from udiscovery.cli import main


if __name__ == "__main__":
    main(["discover", *sys.argv[1:]])
