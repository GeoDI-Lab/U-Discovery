"""Regression tests against deterministic outputs from the archived solver."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE = FIXTURES / "equation_parity_v1.npz"
METADATA = FIXTURES / "equation_parity_v1.json"
sys.path.insert(0, str(REPOSITORY / "src"))

from udiscovery.equations import evaluate  # noqa: E402


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArchivedSolverParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    def test_fixture_and_source_provenance_hashes(self):
        self.assertEqual(file_hash(FIXTURE), self.metadata["fixture_sha256"])
        source = self.metadata["archive_solver"]
        self.assertFalse(Path(source["relative_path"]).is_absolute())
        self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
        workspace_source = REPOSITORY.parent / source["relative_path"]
        if workspace_source.is_file():
            self.assertEqual(file_hash(workspace_source), source["sha256"])
        registry = self.metadata["candidate_registry"]
        registry_path = REPOSITORY / registry["relative_path"]
        self.assertFalse(Path(registry["relative_path"]).is_absolute())
        self.assertEqual(file_hash(registry_path), registry["sha256"])

    def test_all_26_equations_match_scalar_and_localized_archived_outputs(self):
        cases = self.metadata["cases"]
        self.assertEqual(len(cases), 52)
        self.assertEqual(
            {(case["candidate_id"], case["mode"]) for case in cases},
            {
                (case["candidate_id"], mode)
                for case in cases[::2]
                for mode in ("scalar", "localized")
            },
        )
        with np.load(FIXTURE, allow_pickle=False) as fixture:
            self.assertTrue(all(fixture[name].dtype.kind != "O" for name in fixture.files))
            context = {
                "node_betweenness": fixture["node_betweenness"],
                "edge_betweenness": fixture["edge_betweenness"],
                "cluster_labels": fixture["cluster_labels"],
            }
            for case in cases:
                prefix = f"{case['mode']}__{case['candidate_id']}"
                parameters = {
                    name: fixture[f"{prefix}__param__{name}"]
                    for name in case["parameter_names"]
                }
                with self.subTest(candidate=case["candidate_id"], mode=case["mode"]):
                    actual = evaluate(
                        case["candidate_id"],
                        fixture["state"],
                        fixture["distance"],
                        parameters,
                        adjacency=fixture["adjacency"],
                        **context,
                    )
                    np.testing.assert_allclose(
                        actual,
                        fixture[f"{prefix}__output"],
                        rtol=3.0e-5,
                        atol=3.0e-5,
                    )


if __name__ == "__main__":
    unittest.main()
