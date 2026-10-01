# Review for version 0.2.0

This review covers the local public-release source and the changes needed to make a custom-data experiment usable through one command. It is a code and workflow review, not independent verification of the paper's scientific claims or a new evaluation of the archived study datasets.

## Issues addressed

| Finding in the earlier workflow | Change in 0.2.0 |
| --- | --- |
| Training and evaluation were separate commands, with no single operation fitting and ranking a user-supplied set of equations. | `discover` and `run_discovery.py` join input validation, candidate fitting, checkpoint selection, evaluation, and ranking. |
| Candidate equations were tied to the built-in registry and study configurations. | An explicit `Equation` interface loads a user's Python `EQUATIONS` list without edits to package code. |
| User data needed undocumented study fields, and some invalid numerical values or split boundaries could reach training. | Minimal `states` and `distance` inputs receive documented defaults, with checks for finite values, shapes, graph constraints, time step, splits, and available history. |
| The general state-and-delta helper computed time differences between consecutive reconstructed states. That is not the direct next-step change error of teacher-forced predictions. | Fresh fits export aligned predictions and compare each predicted change against its observed transition. Discovery also reports persistence-normalized error with its denominator. The archived reconstruction path retains its documented historical convention. |
| Some CLI defaults resolved the repository root while building the command parser. An installed wheel could therefore fail before reaching commands that need no paper files. | Paper-artifact lookup is deferred to commands that need it. Discovery accepts external input paths independently of the release artifact root. |
| Local virtual environments, generated outputs, and caches could be treated as release contents during validation. | Validation and manifest generation now share an explicit portable-file inventory, excluding local runtime material while still detecting unexpected release files. |
| A failed candidate or a repeated run could leave results difficult to interpret or locate. | Separate run directories, status summaries, candidate manifests, input snapshots, hashes, and explicit optional failure continuation make run provenance inspectable. |

## What the new ranking measures

Every candidate is fitted using the training interval. Its best checkpoint is selected using validation next-step change MSE. The final table uses that same validation metric, followed by complexity and candidate ID as deterministic tie-breaks. Test metrics are computed for reporting and do not select the checkpoint or the winner. The Pareto flag compares validation error and the user-supplied complexity score without replacing the table's sort order.

Predictions use observed history windows, including previous observed validation or test states when predicting later points in those intervals. They are one-step reconstructions, not free-running trajectories. UrbanDE-Net produces history-dependent coefficients. A `global` coefficient is shared across nodes for a window and can still vary over time. Neither a favorable rank nor a coefficient average alone establishes a uniquely identified governing law.

Periodic forcing is disabled by default in discovery. Enabling it adds a learned harmonic component to every candidate and changes the scientific comparison. Complexity is supplied by the user and should follow a consistent rule. Failed candidates are excluded from a continued run's table and are reported separately.

## Release scope

The existing train/evaluate/reproduce commands, canonical equation registry, GraphRAG material, and frozen research data/results remain available. Discovery uses a compact UrbanDE-Net configuration rather than the full paper configuration, and records the exact architecture settings in each run. The demo is a new small deterministic weighted-diffusion system for checking the public workflow. It is not the archived synthetic study, and its short-run ranking is not a reproduction of the paper.

See [PAPER_AUDIT.md](../PAPER_AUDIT.md) for previously documented differences between reported paper values and released artifacts. The 0.2.0 workflow does not resolve those scientific provenance differences by retraining or changing frozen results.

## Verification commands

Local verification on 2026-10-01 used Python 3.12.14, PyTorch 2.14.1, and NumPy 2.5.3 on CPU:

- All 62 automated tests passed, including archived equation parity and the new end-to-end workflow.
- The default demo completed all three candidates; weighted diffusion ranked first, with validation change MSE approximately `1.24e-11` and test change MSE approximately `7.39e-12`. This is a noiseless workflow example, not evidence about performance on user data.
- A built wheel ran discovery outside the source checkout with `UDISCOVERY_ROOT` unset.
- The original synthetic training command completed a one-epoch run; both its new indexed reconstruction and the archived Hennepin reconstruction evaluated successfully.
- Release validation passed. Its existing warning about asymmetric Hennepin normalized distances remains expected and documented.

The release can be checked with:

```bash
python -m pip install -e '.[train,test]'
python -m pytest
python -m udiscovery validate
python run_discovery.py --data examples/demo.npz --equations examples/equations.py --output-dir outputs/demo --check-only
python run_discovery.py --data examples/demo.npz --equations examples/equations.py --output-dir outputs/demo --epochs 3 --device cpu
```

The smoke run should create successful candidate outputs and a ranking table. Exact scientific ranking quality is not guaranteed by a three-epoch run. Package portability should additionally be checked by installing a built wheel outside the checkout and invoking discovery with external data and equation paths, without `UDISCOVERY_ROOT`.


## Maintaining the release manifest

The repository checksum inventory excludes local `.venv`/`venv` directories, Git metadata, Python/test caches, build output, package metadata, logs, macOS metadata, and the root `outputs/` directory. Installing dependencies or running the documented examples therefore does not change the release inventory. These exclusions are explicit and independent of `.gitignore`: unexpected files elsewhere in the release still fail the checksum inventory until reviewed. Pickle-capable checkpoints such as `.pt`, `.pth`, and `.ckpt`, pickle files, notebooks, and actual environment/credential files remain prohibited within the release inventory; ignoring them in Git does not make them acceptable release artifacts.

After reviewing intentional source or documentation changes, maintainers can refresh the root checksum manifest and validate the complete release:

```bash
python scripts/update_manifest.py
python -m udiscovery validate
```

The utility updates only the root `MANIFEST.sha256`. It does not rewrite the `CHECKSUMS.sha256` files for frozen research datasets or results. Investigate mismatches in those frozen artifacts rather than regenerating their hashes as part of a software release.
