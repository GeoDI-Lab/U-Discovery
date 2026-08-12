# Hennepin 2.5 km data

`hennepin_2500m.npz` is the canonical, privacy-preserving input bundle for the
paper's 2.5 km case study. It contains aggregated cell counts only; no device
IDs, individual trips, or raw coordinates are present.

Load it without pickle support:

```python
import numpy as np

with np.load("data/hennepin_2500m/hennepin_2500m.npz", allow_pickle=False) as data:
    states = data["states"]                 # (480, 210), time x node
    distance = data["distance_matrix"]      # (210, 210)
    adjacency = data["adjacency_mask"]      # (210, 210)
    train_end = int(data["train_end"])      # 288
    val_end = int(data["val_end"])          # 384
```

The released normalized distance matrix is intentionally the exact fitting
input. Its source used independent column-wise min-max normalization, so it is
not symmetric even though the underlying meter-distance matrix is symmetric.
Changing or symmetrizing it changes the fitted equations. See `metadata.json`
for array hashes, split definitions, and the full preprocessing caveat.

`grid_cells.geojson` contains 210 ordered polygons in RFC 7946
longitude/latitude coordinates (EPSG:4326), transformed from the original
2.5 km NAD83 / UTM zone 15N grid. Every feature has `cell_index` 0 through 209
in exact `node_ids` order.
