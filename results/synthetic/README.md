# Frozen synthetic results

`table2_metrics.csv` is the exact 12-row manuscript Table 2 source. The
historical `table2_metrics_legacy.csv` is retained for audit because it contains
two similarly named distance-diffusion rows; only the `diffusion2` values were
reported in the paper.

`top1_reconstruction.npz` is the selected seed-43 single-constrained
power-gravity reconstruction and its aligned 100-state test truth. It reproduces
the Table 2 top-row state and transition metrics to better than `1e-4`.
`reconstructions.npz` contains the other safely converted candidate/seed
reconstructions and never requires pickle loading.

`fitted_parameter_runs.npz` contains the final selected parameters for ten
ground-truth-candidate runs. `figure5_parameter_trajectories.csv` contains all
232 archived best-so-far points parsed from the corresponding logs; the logs
themselves are excluded. `figure5_parameter_convergence.csv` is the compact
final-point view.

The trajectory used by Figure 4 is intentionally isolated under
`figure_sources/figure4`. It is not the Table 2 training array.
