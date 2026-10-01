"""End-to-end contracts for user data, equation ranking, and evaluation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from udiscovery.data import load_dataset, validate_dataset
from udiscovery.discover import DiscoveryError, discover, rank_candidates


class UserDataTests(unittest.TestCase):
    def test_minimal_data_defaults_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            states = np.arange(60, dtype=np.float32).reshape(20, 3)
            distance = np.ones((3, 3)) - np.eye(3)
            np.savez(path, states=states, distance=distance)
            data = load_dataset(path)
            self.assertEqual((data.train_end, data.val_end, data.dt), (12, 16, 1.0))
            np.testing.assert_equal(data.initial_state, states[0])
            np.testing.assert_equal(data.adjacency, distance)
            self.assertEqual(validate_dataset(data), [])
            for values, pattern in (
                ({"states": states}, "distance"),
                ({"states": states[:, 0], "distance": distance}, "shape"),
                ({"states": states[:, :1], "distance": distance[:1, :1]}, "node >= 2"),
                ({"states": states, "distance": distance, "train_end": 10}, "both"),
                ({"states": states, "distance": distance, "train_end": 10.5, "val_end": 15}, "integer"),
            ):
                np.savez(path, **values)
                with self.assertRaisesRegex(ValueError, pattern):
                    load_dataset(path)
            for value in (0, -1, np.nan, np.inf):
                np.savez(path, states=states, distance=distance, dt=value)
                self.assertTrue(any("dt" in error for error in validate_dataset(load_dataset(path))))
            np.save(Path(directory) / "wrong.npy", states)
            with self.assertRaisesRegex(ValueError, "NPZ"):
                load_dataset(Path(directory) / "wrong.npy")


class RankingTests(unittest.TestCase):
    def test_validation_only_order_complexity_ties_and_pareto_front(self):
        rows = [
            {"candidate_id": "b", "validation_mse_delta": 1.0, "test_mse_delta": 100, "complexity": 4},
            {"candidate_id": "a", "validation_mse_delta": 1.0, "test_mse_delta": 0.0, "complexity": 4},
            {"candidate_id": "simple", "validation_mse_delta": 2.0, "test_mse_delta": 0.0, "complexity": 1},
            {"candidate_id": "dominated", "validation_mse_delta": 3.0, "test_mse_delta": 0.0, "complexity": 5},
        ]
        ranked = rank_candidates(rows)
        self.assertEqual([row["candidate_id"] for row in ranked], ["a", "b", "simple", "dominated"])
        self.assertEqual([row["pareto_optimal"] for row in ranked], [True, True, True, False])
        rows[0]["test_mse_delta"] = -1000
        self.assertEqual([row["candidate_id"] for row in rank_candidates(rows)], ["a", "b", "simple", "dominated"])

    def test_parser_does_not_need_bundled_artifact_tree(self):
        from udiscovery.cli import _parser
        with patch("udiscovery.cli.repository_root", side_effect=FileNotFoundError("no checkout")):
            args = _parser().parse_args(["discover", "--data", "data.npz", "--equations", "eq.py"])
        self.assertEqual(args.epochs, 200)


class DiscoveryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "tiny.npz"
        self.equations = self.root / "equations.py"
        self.output = self.root / "output"
        # Constant derivative; matches exactly without fitting any parameters.
        states = np.arange(20, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
        states += np.arange(3, dtype=np.float32)[None, :] + 2
        np.savez(self.data, states=states, distance=np.ones((3, 3)) - np.eye(3), dt=0.5)
        self.equations.write_text(
            "import torch\nfrom udiscovery.custom import Equation\n"
            "def exact(state, distance, parameters, *, adjacency):\n"
            "    return torch.ones_like(state) * 2\n"
            "def zero(state, distance, parameters, *, adjacency):\n"
            "    return torch.zeros_like(state)\n"
            "EQUATIONS = [Equation('exact', exact, {}, 2), Equation('zero', zero, {}, 1)]\n",
            encoding="utf-8",
        )

    def run_discovery(self, **kwargs):
        return discover(self.data, self.equations, output_dir=self.output,
                        device="cpu", epochs=1, history_length=3, **kwargs)

    def test_full_run_ranking_artifacts_and_re_evaluation(self):
        from udiscovery.evaluate import evaluate_reconstruction
        run = self.run_discovery()
        payload = json.loads((run / "rankings.json").read_text())
        rows = payload["candidates"]
        self.assertEqual([row["candidate_id"] for row in rows], ["exact", "zero"])
        self.assertEqual(rows[0]["validation_mse_delta"], 0)
        self.assertEqual(rows[1]["test_nmse_delta"], 1)
        summary = json.loads((run / "summary.json").read_text())
        self.assertFalse(summary["test_used_for_selection"])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["best_candidate"], "exact")
        candidate = run / rows[0]["run_directory"]
        for split in ("train", "validation", "test"):
            path = evaluate_reconstruction(run / "configurations/exact.json",
                                           reconstruction_path=candidate / "reconstruction.npz",
                                           output_dir=run / "evaluations", split=split)
            evaluated = json.loads(path.read_text())
            self.assertEqual(evaluated["metrics"]["mse_delta"], rows[0][f"{split}_mse_delta"])
        with np.load(candidate / "reconstruction.npz", allow_pickle=False) as source:
            self.assertEqual(source["target_indices"][0], 3)
            self.assertEqual(source["target_indices"][-1], 19)
            self.assertTrue(np.all(source["predicted_delta"] == 1))
        self.assertEqual((run / "inputs/equations.py").read_bytes(), self.equations.read_bytes())
        repeated = self.run_discovery()
        self.assertNotEqual(run, repeated)

    def test_check_only_writes_no_output_and_candidate_selection(self):
        self.assertIsNone(self.run_discovery(check_only=True))
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "Unknown candidate"):
            self.run_discovery(candidates=["missing"])
        run = self.run_discovery(candidates=["zero"])
        self.assertEqual(json.loads((run / "summary.json").read_text())["best_candidate"], "zero")

    def test_constant_series_has_null_normalized_errors(self):
        np.savez(self.data, states=np.ones((20, 3)), distance=np.ones((3, 3)) - np.eye(3))
        run = self.run_discovery(candidates=["zero"])
        row = json.loads((run / "rankings.json").read_text())["candidates"][0]
        self.assertEqual(row["test_mse_delta"], 0)
        self.assertIsNone(row["test_nmse_delta"])
        self.assertIsNone(row["test_r2_delta"])
        self.assertNotIn("NaN", (run / "rankings.json").read_text())

    def test_fail_fast_and_partial_results_record_candidate_failure(self):
        from udiscovery.train import fit_candidate

        def fail_first(config, **kwargs):
            if kwargs["equation"].id == "exact":
                raise FloatingPointError("test numerical failure")
            return fit_candidate(config, **kwargs)

        with patch("udiscovery.train.fit_candidate", side_effect=fail_first):
            with self.assertRaisesRegex(DiscoveryError, "Saved results"):
                self.run_discovery()
            failed_run = next(self.output.iterdir())
            self.assertEqual(json.loads((failed_run / "summary.json").read_text())["status"], "failed")
            run = self.run_discovery(continue_on_error=True)
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["best_candidate"], "zero")
        self.assertEqual(summary["failures"][0]["candidate_id"], "exact")

    def test_all_bad_preflight_candidates_save_failed_manifest_when_continuing(self):
        self.equations.write_text(
            "from udiscovery.custom import Equation\n"
            "def bad(state, distance, parameters, *, adjacency):\n"
            "    return state.sum()\nEQUATIONS = [Equation('bad', bad, {}, 1)]\n"
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            self.run_discovery()
        with self.assertRaises(DiscoveryError):
            self.run_discovery(continue_on_error=True)
        run = next(self.output.iterdir())
        self.assertEqual(json.loads((run / "summary.json").read_text())["status"], "failed")

    def test_re_evaluation_rejects_wrong_dataset(self):
        from udiscovery.evaluate import evaluate_reconstruction
        run = self.run_discovery(candidates=["exact"])
        candidate = run / json.loads((run / "rankings.json").read_text())["candidates"][0]["run_directory"]
        saved_data = run / "inputs/data.npz"
        with np.load(saved_data, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        payload["states"] = payload["states"] + 100
        np.savez(saved_data, **payload)
        with self.assertRaisesRegex(ValueError, "ground truth"):
            evaluate_reconstruction(run / "configurations/exact.json",
                                    reconstruction_path=candidate / "reconstruction.npz",
                                    output_dir=run / "evaluation")

    def test_indexed_evaluation_cannot_hide_bad_predictions_with_legacy_states(self):
        from udiscovery.evaluate import evaluate_reconstruction
        config = self.root / "config.json"
        config.write_text(json.dumps({"name": "tiny", "dataset": str(self.data),
                                      "split": {}, "training": {}}))
        data = load_dataset(self.data)
        targets = np.arange(3, len(data.states))
        reconstruction = self.root / "conflicting.npz"
        np.savez(reconstruction, target_indices=targets, states=data.states[targets],
                 prediction=np.full_like(data.states[targets], np.nan),
                 predicted_delta=np.ones_like(data.states[targets]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluate_reconstruction(config, reconstruction_path=reconstruction, output_dir=self.output)

    def test_script_runs_from_another_working_directory(self):
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY / "run_discovery.py"),
             "--data", str(self.data), "--equations", str(self.equations), "--check-only"],
            cwd=self.root, capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Inputs valid", completed.stdout)


if __name__ == "__main__":
    unittest.main()
