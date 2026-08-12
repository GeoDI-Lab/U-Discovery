# Reproducibility guide

## Reproduction levels

1. **Artifact reproduction:** `python -m udiscovery reproduce --target paper` regenerates Tables 1–4 and public numerical Figure 4–7 source products from frozen, non-executable files.
2. **Evaluation reproduction:** `python -m udiscovery evaluate` recomputes state and transition metrics for a safe reconstruction NPZ.
3. **Fitting reproduction:** `python -m udiscovery train` fits a configured equation with a portable differentiable implementation on CPU or CUDA.
4. **GraphRAG output reproduction:** exact prompts and resulting JSON/Markdown outputs are archived, but the corpus/index is not redistributed and an LLM rerun is not expected to be bitwise identical.

## Paper artifact map

| Paper item | Released source | Reproduction |
|---|---|---|
| Tables 1 and 3 | `artifacts/candidates.json`, GraphRAG candidate JSONs | `udiscovery reproduce` |
| Table 2 | frozen synthetic bundle and canonical `results/synthetic/table2_metrics.csv` | validation and table export; the 13-row legacy file is audit-only |
| Figure 4 | isolated archived Figure 4 realization | `udiscovery reproduce` |
| Figure 5 | Table 2 metrics and fitted parameter runs | `udiscovery reproduce` with plotting extra |
| Figure 6(a) | raw individual trajectories | excluded; manuscript panel only |
| Figure 6(b–c) | aggregated 2.5 km bundle and ordered GeoJSON | public inputs included |
| Table 4 | two localized result CSVs and derived 26-row table | `udiscovery reproduce` |
| Figure 7 | exact selection, heatmap, and frontier CSVs | `udiscovery reproduce` with plotting extra |
| Figure 8 | 1.5/2.0 km experiments | out of scope |

## Frozen versus newly trained results

The paper values come from the frozen result tables and safe exported parameter/reconstruction arrays. A newly trained model can differ because of hardware, PyTorch kernels, initialization, and early stopping. New runs use the GraphWaveNet UrbanDE-Net encoder, localized/global parameter heads, and node-wise Fourier forcing, and are stored in timestamped directories with versions, seed, metrics, and safe numeric NPZ weights/parameters.

## Environment

Core validation and table reproduction require NumPy and pandas. Training requires PyTorch and uses NetworkX/scikit-learn for candidates needing graph context; figures require Matplotlib. Dependency ranges are in `pyproject.toml`. `requirements-lock.txt` records the compatible Python 3.9 MSI environment used for full parity, plotting, and smoke tests; `requirements-core-py312-lock.txt` records the smaller Python 3.12 validation environment. GraphRAG package versions are recorded separately in `graphrag/provenance.json`.

## Integrity checks

Run:

```bash
python -m udiscovery validate --json
python -m unittest discover -s tests -v
```

Validation checks shapes, orientation, finite values, splits, geometry order, candidate counts, GraphRAG JSON structure, Table 2/Table 4 values to `1e-4`, unsafe formats, excluded resolutions, and size limits.
