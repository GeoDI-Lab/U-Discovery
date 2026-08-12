"""Focused model and portable-training tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


class UrbanDENetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def make_model(self, seed=17):
        torch = self.torch
        from udiscovery.model import UrbanDENet

        nodes = 4
        adjacency = torch.ones(nodes, nodes) - torch.eye(nodes)
        distance = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.5, 2.0],
             [2.0, 1.5, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        return UrbanDENet(
            num_nodes=nodes,
            history_length=4,
            parameter_names=("alpha", "beta", "kappa", "delta", "gamma"),
            parameter_scopes=("node", "node", "global", "global", "global"),
            adjacency_mask=adjacency,
            distance_matrix=distance,
            temporal_harmonics=1,
            encoder_dim=8,
            head_hidden=(8,),
            encoder_dropout=0.0,
            head_dropout=0.0,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=4,
            end_channels=4,
            blocks=1,
            layers=1,
            adaptive_adjacency=False,
            initial_parameters={
                "alpha": 0.8, "beta": 1.1, "kappa": 0.5,
                "delta": 0.7, "gamma": 1.3,
            },
            init_seed=seed,
        )

    def test_wrapper_emits_localized_global_and_temporal_fields(self):
        torch = self.torch
        model = self.make_model()
        output = model(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 10.0)
        self.assertEqual(output.interaction_parameters["alpha"].shape, (2, 4))
        self.assertEqual(output.interaction_parameters["beta"].shape, (2, 4))
        self.assertEqual(output.interaction_parameters["kappa"].shape, (2,))
        self.assertEqual(output.interaction_parameters["delta"].shape, (2,))
        self.assertEqual(output.interaction_parameters["gamma"].shape, (2,))
        self.assertEqual(output.temporal_sin.shape, (2, 4, 1))
        self.assertEqual(output.temporal_cos.shape, (2, 4, 1))
        self.assertTrue(torch.all(output.interaction_parameters["alpha"] > 0).item())

    def test_initialization_and_npz_weight_round_trip_are_deterministic(self):
        torch = self.torch
        from udiscovery.train import load_model_weights_npz, save_model_weights_npz

        first = self.make_model(seed=23)
        second = self.make_model(seed=23)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            before = {name: value.clone() for name, value in first.state_dict().items()}
            save_model_weights_npz(first, path)
            with torch.no_grad():
                next(first.parameters()).add_(1.0)
            self.assertEqual(load_model_weights_npz(first, path), ([], []))
            for name, value in first.state_dict().items():
                torch.testing.assert_close(value, before[name])
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(archive.files)
                self.assertTrue(all(archive[name].dtype.kind != "O" for name in archive.files))

    def test_history_window_alignment(self):
        torch = self.torch
        from udiscovery.train import build_history_windows

        states = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        windows = build_history_windows(states, torch.tensor([3, 6]), 4)
        torch.testing.assert_close(windows[0], states[0:4])
        torch.testing.assert_close(windows[1], states[3:7])


class PortableTrainingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")

    def test_one_epoch_writes_only_safe_portable_artifacts(self):
        from udiscovery.train import fit_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = 4
            times = 14
            t = np.arange(times, dtype=np.float32)[:, None]
            offsets = np.arange(nodes, dtype=np.float32)[None, :]
            states = 5.0 + offsets + 0.05 * t + 0.1 * np.sin(2.0 * np.pi * t / 6.0)
            distance = np.asarray(
                [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0],
                 [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]],
                dtype=np.float32,
            )
            adjacency = np.ones((nodes, nodes), dtype=np.float32) - np.eye(nodes, dtype=np.float32)
            dataset_path = root / "tiny.npz"
            np.savez_compressed(
                dataset_path,
                states=states,
                initial_state=states[0],
                distance=distance,
                adjacency=adjacency,
                node_ids=np.arange(nodes, dtype=np.int32),
                train_end=np.asarray(8),
                val_end=np.asarray(11),
                dt=np.asarray(1.0, dtype=np.float32),
            )
            config = {
                "name": "tiny",
                "dataset": str(dataset_path),
                "split": {"train_end": 8, "validation_end": 11, "test_end": times},
                "model": {
                    "history_length": 4,
                    "temporal_harmonics": 1,
                    "encoder_dim": 8,
                    "head_hidden": [8],
                    "encoder_dropout": 0.0,
                    "head_dropout": 0.0,
                    "residual_channels": 4,
                    "dilation_channels": 4,
                    "skip_channels": 4,
                    "end_channels": 4,
                    "blocks": 1,
                    "layers": 1,
                    "adaptive_adjacency": False,
                },
                "training": {
                    "candidate_id": "distance_weighted_diffusion",
                    "parameter_mode": "localized",
                    "initial_parameters": {"lambda": 0.2, "eta": 1.0},
                    "forcing_frequency": 1.0 / 6.0,
                    "seed": 5,
                    "device": "cpu",
                    "deterministic": True,
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "patience": 2,
                },
                "output_dir": str(root / "unused"),
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run = fit_candidate(config_path, output_dir=root / "runs")

            expected = {
                "model_weights.npz", "fitted_parameters.npz",
                "training_history.csv", "run_manifest.json",
            }
            self.assertTrue(expected.issubset({path.name for path in run.iterdir()}))
            self.assertFalse(list(run.rglob("*.pt")))
            self.assertFalse(list(run.rglob("*.pth")))
            with np.load(run / "model_weights.npz", allow_pickle=False) as weights:
                self.assertTrue(weights.files)
                self.assertTrue(all(weights[name].dtype.kind != "O" for name in weights.files))
            with np.load(run / "fitted_parameters.npz", allow_pickle=False) as fitted:
                self.assertEqual(fitted["interaction__lambda"].shape[1:], (nodes,))
                self.assertEqual(fitted["interaction__eta"].ndim, 1)
                self.assertEqual(fitted["temporal_sin"].shape[1:], (nodes, 1))
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["serialization"], "numeric NPZ only; no pickle or torch.save")
            self.assertEqual(manifest["architecture"]["class"], "UrbanDENet")


if __name__ == "__main__":
    unittest.main()
