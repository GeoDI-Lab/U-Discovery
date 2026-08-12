# Hennepin 2.5 km results

This directory contains the frozen localized-parameter results needed for the
paper's 2.5 km Table 4 and Figure 7.

- `local_classic_eq_results_denoise.csv` and
  `local_proposed_eq_results_denoise.csv` are byte-identical archived result
  tables.
- `table4_metrics.csv` is the 26-row paper table source, with normalized errors
  recomputed from the temporal-only (`Osc`) denominators recorded in
  `metadata.json`.
- `figure7_selection.csv` records every inclusion/exclusion decision.
  `figure7_heatmap_top20.csv` and the two `figure7_frontier_*.csv` files are the
  compact numerical plot sources.
- `selected_c9_parameters.npz` is a no-pickle export of the selected
  Betweenness-augmented gravity fit. `selected_c9_fit.json` records parameter
  meanings, training provenance, source checkpoint hash, and an exact numeric
  equivalence check.
- `top1_reconstruction_vs_truth.csv` is the transparent long-form source for
  the 95 x 210 teacher-forced test reconstruction;
  `top1_reconstruction.npz` is its compact array representation.

No `.pt` files are distributed. The original selected checkpoint was loaded
only in the author's PyTorch environment, exported to numeric NPZ arrays, and
verified with zero maximum absolute error. This directory intentionally omits
global fits, k=2/two-harmonic sensitivity, exclude-SE sensitivity, logs,
duplicate checkpoints, and large post-paper flow diagnostics.
