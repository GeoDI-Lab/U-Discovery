# Data inventory

## Frozen synthetic experiment

`data/synthetic/dataset.npz` contains 501 states for 20 nodes (500 transitions), raw and normalized 20×20 distance matrices, full off-diagonal adjacency, ordered node IDs, 300/100/101 state splits, structural forcing covariates, and ground-truth parameters. Arrays use `(time, node)` order and can be loaded with `numpy.load(..., allow_pickle=False)`.

`data/synthetic/metadata.json` records the original array hashes and accepted generation attempt. The frozen bundle is authoritative for Table 2.

`figure_sources/figure4/synthetic_figure4.npz` is a distinct archived stochastic realization used only for the unchanged Figure 4 panels. It shares the initial state and distance matrix with the Table 2 bundle, but its trajectory must not be used to recompute Table 2.

## Hennepin 2.5 km case

`data/hennepin_2500m/hennepin_2500m.npz` contains:

- aggregated active-population dynamics, shape `(480, 210)`, at 15-minute intervals;
- the exact normalized `(210, 210)` distance matrix used for fitting;
- full off-diagonal adjacency, ordered node IDs, and 288/96/96 state splits;
- no device IDs, individual coordinates, or trajectory records.

`grid_cells.geojson` contains the corresponding 210 ordered 2.5 km cells as RFC 7946 longitude/latitude coordinates (EPSG:4326), transformed from the original EPSG:26915 grid. Its `cell_index` order equals `node_ids` exactly.

The normalized distance matrix is intentionally asymmetric (`max |D-D.T| = 0.4780116`) because the archived preprocessing independently min–max normalized each column. The raw metric distance was symmetric, but replacing the released matrix would no longer reproduce the fits.

## Explicit exclusions

This repository contains none of the following:

- raw PlaceIQ records or individual trip geometries;
- device identifiers, start/end coordinates, or raw timestamps;
- 1.5 km or 2.0 km arrays and geometries;
- other-county datasets;
- GraphRAG paper corpus, Parquet index, GraphML/GEXF graphs, embeddings, or LanceDB store;
- unsafe PyTorch/pickle checkpoints.

Each data directory has detailed metadata and SHA-256 checksums.

## License

Except where otherwise noted, the released original and derived data are
available under CC BY 4.0. See [DATA_LICENSE.md](DATA_LICENSE.md) and
[LICENSING.md](LICENSING.md) for scope and third-party exclusions.
