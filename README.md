# U-Discovery

**Prepare your data and candidate equations, then run one Python command to fit and rank them.**

U-Discovery (Urban Discovery) combines scientific-prior-guided equation proposal, differentiable parameter fitting with the Urban Differential Equation Network (UrbanDE-Net), and comparison by reconstruction accuracy and mathematical complexity. Version 0.2.0 connects fitting and evaluation into a single workflow for your own data and Python equations. The repository also retains the equation registry, GraphRAG artifacts, synthetic study, and aggregated 2.5 km Hennepin County case study.

**[Try the interactive demo](https://zhongfuma-gis.com/u-discovery-geodi/)**

## Install

Python 3.9 or newer is required. Run these commands from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train]'
```

On Windows, activate with `.venv\Scripts\activate`.

## Try the complete workflow

The included small dataset and three candidate equations are ready to run:

```bash
python run_discovery.py \
  --data examples/demo.npz \
  --equations examples/equations.py \
  --output-dir outputs/demo \
  --epochs 3 --device cpu
```

This short run checks that fitting, evaluation, and ranking work together. Omit `--epochs 3` for the default maximum of 200 epochs with validation-based early stopping. A short demo is not evidence that an equation has been identified correctly.

For your own experiment, supply an NPZ dataset and a Python file exporting an `EQUATIONS` list:

```bash
python run_discovery.py \
  --data path/to/data.npz \
  --equations path/to/equations.py \
  --output-dir outputs/my_experiment
```

You can also use `python -m udiscovery discover` or `udiscovery discover` with the same arguments. Use `--check-only` to validate inputs before fitting.

Each run creates a separate timestamped directory containing `rankings.csv`, `rankings.json`, a summary, input snapshots, and each candidate's fitted model, coefficients, reconstruction, and metrics. Candidates rank by **validation next-step change MSE**; complexity breaks ties. Test metrics are reported separately. A Pareto flag identifies candidates that are not dominated jointly on validation error and complexity.

See the **[user quickstart](docs/quickstart.md)** for the data format, equation template, options, output files, and metric definitions. Start from [examples/equations.py](examples/equations.py); regenerate the demo with `python examples/make_demo_data.py`.

## What the result means

UrbanDE-Net learns coefficients from observed history. Its coefficient outputs can change over time, including coefficients shared across nodes. Evaluation predicts one next step from observed history, rather than repeatedly feeding predictions back into the model. The ranking therefore compares fitted candidates under this evaluation procedure; it does not establish a unique physical law or a free-running forecasting result. The new workflow does not reproduce the paper's archived candidate ordering.

## Existing study workflows

The original commands and frozen study artifacts remain available. For figure generation and the test suite, install the additional dependencies with `python -m pip install -e '.[all]'`:

```bash
python -m udiscovery validate
python -m udiscovery train --config configs/synthetic.yaml
python -m udiscovery train --config configs/hennepin_2500m.yaml
python -m udiscovery evaluate \
  --config configs/hennepin_2500m.yaml \
  --reconstruction results/hennepin_2500m/top1_reconstruction.npz
python -m udiscovery reproduce --target paper
```

Artifact commands require this source checkout. With a regular wheel installation, set `UDISCOVERY_ROOT` to a downloaded release checkout to use those commands. Discovery with your own data and equations works independently of the packaged paper artifacts.

## Repository guide

| Path | Contents |
| --- | --- |
| `run_discovery.py` | Single-command entry point |
| `examples/` | Ready-to-run data, equation definitions, and data generator |
| `src/udiscovery/` | UrbanDE-Net, fitting, evaluation, custom-equation interface, and ranking |
| `artifacts/`, `configs/` | Original candidate registry and study configurations |
| `data/`, `results/` | Frozen study data and results |
| `graphrag/` | Scientific-prior prompts, equation outputs, and provenance |
| `figure_sources/`, `reproduced/` | Archived figure inputs and generated study outputs |
| `tests/`, `scripts/` | Verification and maintainer utilities |

Further documentation:

- [Release review](docs/release-review.md): issues addressed and interpretation limits.
- [Changelog](CHANGELOG.md): changes in version 0.2.0.
- [DATA.md](DATA.md): study datasets, formats, and exclusions.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): original experiment and artifact instructions.
- [PAPER_AUDIT.md](PAPER_AUDIT.md): differences between archived artifacts and reported values.
- [graphrag/README.md](graphrag/README.md): GraphRAG artifact scope and provenance.

## Citation and licenses

Citation metadata are provided in [CITATION.cff](CITATION.cff). Software is released under the [MIT License](LICENSE). Original data and other research artifacts are released under [CC BY 4.0](DATA_LICENSE.md); see [LICENSING.md](LICENSING.md) for scope and third-party exclusions.
