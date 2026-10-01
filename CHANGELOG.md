# Changelog

## 0.2.0 — 2026-10-01

### Added

- One-command data-to-ranking workflow: `python run_discovery.py` and `udiscovery discover`.
- Custom Python candidate definitions through `udiscovery.custom.Equation` and an `EQUATIONS` list.
- Minimal NPZ dataset input with documented graph, timing, and chronological-split defaults.
- Validation-ranked CSV/JSON results, separate test metrics, complexity tie-breaks, and a validation-error/complexity Pareto flag.
- Input snapshots and hashes, resolved per-candidate configurations, run summaries, and inspectable model and coefficient outputs.
- Preflight-only validation, candidate selection, explicit CPU/CUDA choice, early stopping, optional periodic forcing, and optional continuation after a candidate failure.
- A deterministic four-node demo and a user quickstart covering the equation interface, outputs, and interpretation limits.

### Fixed

- Fresh training/discovery evaluation now compares predicted next-step changes directly to their matching observed transitions.
- Persistence-normalized errors record their denominator and represent undefined values as `null` in JSON.
- Custom-data validation rejects invalid shapes, numerical values, graph constraints, split boundaries, and insufficient history before fitting.
- Installed CLI commands no longer require paper-artifact paths while constructing the argument parser.
- Release validation excludes local environments, runtime outputs, and caches through a shared portable-file inventory, while retaining checks for unexpected release files and prohibited checkpoint/credential artifacts.
- `scripts/update_manifest.py` regenerates the root release manifest using that same inventory without changing frozen dataset/result checksums.

### Compatibility

- Existing study commands, frozen datasets/results, and archived reconstruction metric conventions are retained.
- Discovery measures teacher-forced one-step performance with learned, history-dependent coefficients. Its ranking is separate from the paper's archived rankings.

## 0.1.0 — 2026-08-12

- Initial portable release of the equation registry, UrbanDE-Net fitting, synthetic and Hennepin County study artifacts, and reproduction utilities.
