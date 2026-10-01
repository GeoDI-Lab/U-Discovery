"""Example candidate equations for ``run_discovery.py``.

Each function returns the derivative dx/dt for one state vector. The runner
multiplies by the dataset's dt to predict the next observed change. Use Torch
operations so gradients flow through the supplied positive parameters.
"""

from udiscovery.custom import Equation


def weighted_diffusion(state, distance, parameters, *, adjacency):
    """Exchange between connected nodes, attenuated by distance."""
    weights = adjacency / (1.0 + distance)
    differences = state.unsqueeze(0) - state.unsqueeze(1)
    return parameters["diffusion_rate"] * (weights * differences).sum(dim=1)


def mean_reversion(state, distance, parameters, *, adjacency):
    """Relaxation toward the current mean over all nodes."""
    return parameters["restoring_rate"] * (state.mean() - state)


def decay(state, distance, parameters, *, adjacency):
    """Independent exponential decay at each node."""
    return -parameters["decay_rate"] * state


# Complexity is a user-supplied, positive integer score. Use the same scoring rule
# for every candidate; these example values are illustrative operation counts.
# Parameters default to scope="global" (shared across nodes at each time).
# For node-specific coefficients, add scopes={"diffusion_rate": "node"}.
EQUATIONS = [
    Equation(
        id="weighted_diffusion",
        name="Distance-weighted diffusion",
        formula="dx_i/dt = k * sum_j A_ij/(1+d_ij) * (x_j-x_i)",
        function=weighted_diffusion,
        parameters={"diffusion_rate": 0.10},
        complexity=5,
    ),
    Equation(
        id="mean_reversion",
        name="Restoring toward the global mean",
        formula="dx_i/dt = r * (mean(x)-x_i)",
        function=mean_reversion,
        parameters={"restoring_rate": 0.10},
        complexity=3,
    ),
    Equation(
        id="decay",
        name="Independent decay",
        formula="dx_i/dt = -k*x_i",
        function=decay,
        parameters={"decay_rate": 0.01},
        complexity=1,
    ),
]
