# Fit and rank your equations

The discovery workflow loads one dataset and a list of equations, fits a separate UrbanDE-Net for each candidate, restores its best validation checkpoint, evaluates the candidates, and writes a ranked table. You do not need to edit the package or construct a separate training configuration for each equation.

## 1. Install and try the demo

From the repository root, with Python 3.9 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
python run_discovery.py \
  --data examples/demo.npz \
  --equations examples/equations.py \
  --output-dir outputs/demo \
  --epochs 3 --device cpu
```

On Windows, activate with `.venv\Scripts\activate`. The three-epoch run is a short installation and workflow check. Omit `--epochs 3` to allow the default 200 epochs, with early stopping after 30 epochs without validation improvement. You can regenerate the included deterministic 60-step, four-node dataset with `python examples/make_demo_data.py`.

The equivalent installed commands are:

```bash
python -m udiscovery discover --data examples/demo.npz --equations examples/equations.py --output-dir outputs/demo
udiscovery discover --data examples/demo.npz --equations examples/equations.py --output-dir outputs/demo
```

Use `python run_discovery.py --help` for the full interface. Paths supplied to discovery are interpreted from your current working directory.

## 2. Prepare the data

Save an NPZ archive with NumPy. Rows of `states` are equally spaced observations in chronological order; columns are nodes in a consistent order. The minimum dataset contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `states` | `(T, N)` | Finite observed state values, with at least five time points and two nodes |
| `distance` | `(N, N)` | Finite, nonnegative pairwise distances with zero diagonal; asymmetry is supported |

For example, once you have populated `states` and `distance` from your own measurements:

```python
import numpy as np

np.savez_compressed(
    "data.npz",
    states=np.asarray(states, dtype=np.float32),
    distance=np.asarray(distance, dtype=np.float32),
)
```

Optional keys provide graph, timing, and split information:

| Key | Default | Requirements |
| --- | --- | --- |
| `adjacency` | Complete graph without self-edges | `(N, N)`, finite nonnegative weights, zero diagonal |
| `dt` | `1.0` | Positive finite scalar; interval in the units used by your equation |
| `train_end`, `val_end` | `floor(0.6*T)`, `floor(0.8*T)` | Integer scalars; supply both or neither |
| `node_ids` | `0, 1, ..., N-1` | `(N,)` unique identifiers, with no Python object arrays |
| `initial_state` | `states[0]` | `(N,)`, equal to `states[0]` |

Explicit splits use target-state indices: targets before `train_end` belong to training, targets from `train_end` through `val_end-1` belong to validation, and targets from `val_end` onward belong to testing. Each prediction has an observed history ending immediately before its target. This allows the first validation or test prediction to use earlier observed states without fitting on validation or test targets. A later test prediction uses earlier observed test states, as expected for one-step evaluation.

The history length defaults to `min(12, train_end - 1)`. It must be at least two and leave at least one training target. Validation and test splits must each contain at least one target. More than the minimum number of observations will usually be needed for a useful comparison.

The runner does not impute missing observations, infer irregular time intervals, or normalize your measurement units. Prepare consistently ordered and consistently scaled arrays before running. The node order in `states`, `distance`, `adjacency`, and `node_ids` must agree.

## 3. Write the candidate equation file

Copy [examples/equations.py](../examples/equations.py), or create a file like this:

```python
from udiscovery.custom import Equation


def diffusion(state, distance, parameters, *, adjacency):
    weights = adjacency / (1.0 + distance)
    differences = state.unsqueeze(0) - state.unsqueeze(1)
    return parameters["rate"] * (weights * differences).sum(dim=1)


EQUATIONS = [
    Equation(
        id="diffusion",
        name="Distance-weighted diffusion",
        formula="dx_i/dt = k * sum_j A_ij/(1+d_ij) * (x_j-x_i)",
        function=diffusion,
        parameters={"rate": 0.1},
        complexity=5,
    ),
]
```

The Python module must export a nonempty `EQUATIONS` list (or tuple) with candidate IDs unique even when ignoring case. IDs must start with a letter or digit and contain only letters, digits, underscores, or hyphens. Each function receives `state` with shape `(N,)`, `distance` and `adjacency` with shape `(N, N)`, and a dictionary of parameter tensors. Return a finite Torch tensor with shape `(N,)`, matching the float32 dtype and device of `state`, representing **the derivative `dx/dt`**. The runner predicts `delta = dt * derivative` and then `next_state = current_state + delta`.

Treat inputs and the parameter mapping as read-only; clone a tensor before modifying it in place. Use Torch operations, such as tensor methods or `torch.exp`, to preserve gradients. Every declared parameter must remain connected to the output. Avoid converting parameters or outputs to NumPy, calling `.detach()`, or returning a Python list. The equation module is executable Python; use equations whose code you trust. Keep the module self-contained so the saved copy is sufficient to inspect the equation later; imported local helper files are not automatically copied.

Equation fields:

| Field | Meaning |
| --- | --- |
| `id` | Unique stable identifier used in output files and candidate selection |
| `function` | Callable implementing the derivative |
| `parameters` | Mapping from parameter names to finite positive scalar initial values; may be empty |
| `complexity` | Positive integer score used for tie-breaking and the Pareto flag |
| `scopes` | Optional mapping of parameter names to `"global"` or `"node"`; defaults to global |
| `name`, `formula` | Optional descriptive metadata |
| `bounded_parameters` | Optional tuple of parameter names constrained to `(0, 1)` |

For node-specific coefficients, set `scopes={"rate": "node"}`. A global coefficient is shared by all nodes for a particular history window; a node coefficient has one value per node. **Both scopes can vary with the observed history.** These are neural coefficient outputs, not a fit of one constant coefficient over all times.

All trainable parameters are positive. For a subtractive term, put the minus sign in the equation, as in the example `decay` candidate. Parameters in `bounded_parameters` also require initial values strictly between zero and one. Use the same parameter scope and complexity-counting convention across candidates when that is appropriate to your scientific comparison. The runner records your supplied spatial-equation complexity, excluding the shared neural network; it does not derive or validate a symbolic operation count.

## 4. Validate, then run

Check shapes, numerical validity, candidate definitions, and differentiability before fitting. Equation checks use all nodes at the first and last training observations; passing these checks does not guarantee numerical stability on every later input:

```bash
python run_discovery.py \
  --data data.npz --equations equations.py \
  --output-dir outputs/my_experiment --check-only
```

Fit all candidates:

```bash
python run_discovery.py \
  --data data.npz --equations equations.py \
  --output-dir outputs/my_experiment
```

Useful options:

| Option | Default | Effect |
| --- | --- | --- |
| `--epochs` | `200` | Maximum training epochs for each candidate |
| `--patience` | `30` | Early-stop after this many epochs without validation improvement |
| `--batch-size` | `32` | Number of history windows per training/evaluation batch |
| `--learning-rate` | `0.001` | Optimizer learning rate |
| `--seed` | `42` | Random initialization and training seed |
| `--history-length` | `min(12, train_end-1)` | Number of observed time points per model input |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`; auto selects CUDA when available, otherwise CPU |
| `--threads` | `1` | PyTorch CPU worker threads during fitting |
| `--forcing-period` | Off | Enable one sinusoidal/cosine forcing harmonic with period in observation steps |
| `--candidate` | All | Select one or more candidate IDs, separated by spaces |
| `--continue-on-error` | Off | Record failed candidates and rank the successful ones |
| `--check-only` | Off | Validate the inputs without training |

For example, `--candidate weighted_diffusion mean_reversion` runs just those two candidates. A forcing period of `24` represents a 24-observation cycle, regardless of the numerical value of `dt`. Leave forcing disabled for a comparison based only on your candidate equation terms. If enabled, the learned periodic component is added to each candidate derivative and can affect how differences between equations should be interpreted.

Invalid inputs fail before a run is created. By default, a failure during fitting or evaluation aborts the run and saves partial-run metadata. With `--continue-on-error`, inspect `failed_candidates.json`; a table of successful candidates is not a complete comparison of the original candidate set. Invalid shared data or an invalid equation module cannot be repaired by skipping one candidate.

Discovery uses a compact UrbanDE-Net configuration to make custom experiments practical: encoder width 32, parameter-head widths 32/16, residual and dilation widths 16, skip/end widths 32, two blocks and two layers, and dropout 0.1. These defaults differ from the full paper configurations. The exact model settings are recorded in `summary.json` and each candidate configuration.

## 5. Read the results

The output directory receives a unique timestamped run subdirectory, so subsequent runs do not overwrite earlier ones. Start with `rankings.csv`, `rankings.json`, and the run's `README.md`. The summary identifies the selected candidate and records the run settings and completion status. Input snapshots include canonical data, your equation module, resolved candidate configurations, and hashes. Each successful candidate has model weights and coefficient arrays saved as NPZ, training history, reconstruction arrays, metrics, and a run manifest. NPZ outputs can be read with `np.load(path, allow_pickle=False)`.

| Output | Contents |
| --- | --- |
| `rankings.csv`, `rankings.json` | Rank, candidate ID, complexity, Pareto flag, split metrics, and candidate output path |
| `summary.json` | Completion status, selected candidate, model settings, split boundaries, persistence denominators, and input hashes |
| `inputs/data.npz`, `inputs/equations.py` | Canonical dataset including resolved defaults, and a copy of your equation file |
| `configurations/<candidate_id>.json` | Resolved configuration for each attempted fit |
| `failed_candidates.json` | Failed candidate IDs, stages, and error messages; empty on full success |
| `candidates/<run>/model_weights.npz` | Numeric model state dictionary |
| `candidates/<run>/fitted_parameters.npz` | Parameter tensors, prediction-window indices, and split codes |
| `candidates/<run>/reconstruction.npz` | Predictions and directly predicted changes aligned to target indices |
| `candidates/<run>/training_history.csv` | Training and validation change MSE for each epoch |
| `candidates/<run>/metrics.json`, `run_manifest.json` | Split metrics, selected epoch, parameter metadata, and environment information |

For example, load `rankings.csv` with `pandas.read_csv(...)`. In `rankings.json`, rows are under the `candidates` key and run completeness is under `status`. The `run_directory` column links each row to its candidate output directory.

The ranking order is ascending `validation_mse_delta`, then ascending `complexity`, then candidate ID for a reproducible final tie-break. Test performance never selects the best epoch or the ranking order. Check the test columns to assess performance on the later held-out interval. The Pareto flag means no other successful candidate has both no greater validation error and no greater complexity, with at least one strict improvement; it does not change the ranking order.

For an observed transition from `x[t]` to `x[t+1]`:

```text
true_delta       = x[t+1] - x[t]
predicted_delta  = dt * fitted_derivative(observed_history)
predicted_state  = x[t] + predicted_delta
MSE_delta        = mean((predicted_delta - true_delta)^2)
MSE_state        = mean((predicted_state - x[t+1])^2)
```

With this aligned one-step definition, state MSE and change MSE are equal up to numerical precision. The change metric compares predicted and observed changes directly; it does not take another time difference of the predicted next states.

Normalized MSE divides the model's MSE by the same split's persistence error. Persistence predicts `x[t+1] = x[t]`, so its change prediction is zero and its denominator is `mean(true_delta**2)`. Values below one improve on persistence, and the denominator is saved as `summary.json` → `persistence_mse` for each split. If it is zero, normalized MSE is undefined and is written as JSON `null`, not infinity. Other undefined metrics are also represented as `null` in JSON.

The coefficient arrays contain values for each evaluated history window. Their averages summarize model outputs, not uniquely identified constant physical coefficients. Evaluation is teacher-forced one-step prediction using observed histories; it is not an autonomous trajectory rollout. Before drawing scientific conclusions, examine residuals, candidate design and scope, optimization stability, and performance across seeds. The three-epoch demo is only a workflow check.

## Existing paper artifacts

The new discovery workflow is separate from the frozen paper reproduction path. Existing `train`, `evaluate`, `validate`, and `reproduce` commands remain available; see [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) and [PAPER_AUDIT.md](../PAPER_AUDIT.md). In a regular wheel installation, artifact-level commands need `UDISCOVERY_ROOT` set to a release checkout. Custom-data discovery does not require that variable.
