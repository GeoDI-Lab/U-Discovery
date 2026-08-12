# U-Discovery

U-Discovery (Urban Discovery) is an interpretable equation-discovery framework for urban spatiotemporal systems. It combines scientific-prior-guided equation proposal, differentiable parameter fitting with the Urban Differential Equation Network (UrbanDE-Net), and candidate comparison based on reconstruction accuracy and mathematical complexity. This repository provides the portable implementation, equation registry, GraphRAG prompt and output artifacts, a synthetic example, and an aggregated 2.5 km Hennepin County case study.

## Project structure

```text
U-Discovery-release/
├── src/udiscovery/              # Core Python package
│   ├── equations.py             # Candidate equations and evaluators
│   ├── model.py                 # Graph WaveNet and UrbanDE-Net
│   ├── train.py                 # Model fitting
│   ├── evaluate.py              # Reconstruction evaluation
│   ├── synthetic.py             # Synthetic-data generator
│   └── validate.py              # Data and repository validation
├── artifacts/candidates.json    # Canonical equation registry
├── configs/                     # Experiment configurations
├── data/                        # Synthetic and Hennepin 2.5 km datasets
├── graphrag/                    # Prompts, equation outputs, and provenance
├── results/                     # Fitted parameters and evaluation results
├── figure_sources/              # Source data for archived visual analyses
├── reproduced/                  # Generated study outputs
├── tests/                       # Equation-parity and training tests
└── scripts/                     # Maintainer utilities
```

## Installation

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

## Basic usage

Validate the packaged code and data:

```bash
python -m udiscovery validate
```

Run either included experiment:

```bash
python -m udiscovery train --config configs/synthetic.yaml
python -m udiscovery train --config configs/hennepin_2500m.yaml
```

Evaluate a saved reconstruction:

```bash
python -m udiscovery evaluate \
  --config configs/hennepin_2500m.yaml \
  --reconstruction results/hennepin_2500m/top1_reconstruction.npz
```

Run `python -m udiscovery --help` for all commands and options.

## Documentation

- [DATA.md](DATA.md): dataset contents, formats, and exclusions.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): experiment and artifact instructions.
- [PAPER_AUDIT.md](PAPER_AUDIT.md): documented differences between archived artifacts and reported values.
- [graphrag/README.md](graphrag/README.md): GraphRAG artifact scope and provenance.

## Citation and licenses

Citation metadata are provided in [CITATION.cff](CITATION.cff). Software is
released under the [MIT License](LICENSE), while original data and other
research artifacts are released under
[CC BY 4.0](DATA_LICENSE.md). See [LICENSING.md](LICENSING.md) for the exact
scope and third-party exclusions.
