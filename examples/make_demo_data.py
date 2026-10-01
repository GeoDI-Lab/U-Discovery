"""Generate the deterministic 60-step, four-node quickstart dataset.

Run from any directory with Python and NumPy installed. The default output is
demo.npz next to this script. This small example is separate from the archived
synthetic study and has no noise or periodic forcing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_demo(output: Path) -> Path:
    positions = np.asarray([0.0, 1.0, 2.4, 4.0], dtype=np.float64)
    distance = np.abs(positions[:, None] - positions[None, :])
    adjacency = np.ones((len(positions), len(positions))) - np.eye(len(positions))
    weights = adjacency / (1.0 + distance)
    states = np.empty((60, len(positions)), dtype=np.float64)
    states[0] = [1.0, 2.4, 0.7, 3.5]
    dt = 0.25
    diffusion_rate = 0.12
    for step in range(len(states) - 1):
        differences = states[step][None, :] - states[step][:, None]
        derivative = diffusion_rate * (weights * differences).sum(axis=1)
        states[step + 1] = states[step] + dt * derivative

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # states and distance are required. dt is optional and defaults to 1.
    # Omitting split boundaries selects chronological 60/20/20 splits.
    np.savez_compressed(
        output,
        states=states.astype(np.float32),
        distance=distance.astype(np.float32),
        dt=np.asarray(dt, dtype=np.float64),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("demo.npz")
    )
    print(make_demo(parser.parse_args().output))


if __name__ == "__main__":
    main()
