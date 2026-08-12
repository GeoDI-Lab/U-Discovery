"""Focused tests for the canonical paper equation registry and evaluators."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from udiscovery.equations import (  # noqa: E402
    EquationInputError,
    evaluate,
    get_candidate,
    list_candidates,
    load_registry,
    precompute_context,
)


NODES = 5
STATE = np.asarray([2.0, 3.5, 5.0, 7.5, 11.0], dtype=np.float64)
DISTANCE = np.asarray(
    [
        [0.0, 1.1, 2.4, 4.2, 7.0],
        [1.1, 0.0, 1.7, 3.1, 5.8],
        [2.4, 1.7, 0.0, 1.3, 3.6],
        [4.2, 3.1, 1.3, 0.0, 2.2],
        [7.0, 5.8, 3.6, 2.2, 0.0],
    ],
    dtype=np.float64,
)
ADJACENCY = np.ones((NODES, NODES), dtype=bool)
np.fill_diagonal(ADJACENCY, False)
NODE_BETWEENNESS = np.asarray([0.05, 0.3, 1.0, 0.5, 0.1], dtype=np.float64)
EDGE_BETWEENNESS = np.asarray(
    [
        [0.0, 0.8, 0.4, 0.2, 0.1],
        [0.8, 0.0, 1.0, 0.6, 0.2],
        [0.4, 1.0, 0.0, 0.9, 0.5],
        [0.2, 0.6, 0.9, 0.0, 0.7],
        [0.1, 0.2, 0.5, 0.7, 0.0],
    ],
    dtype=np.float64,
)
CLUSTER_LABELS = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
CONTEXT = {
    "node_betweenness": NODE_BETWEENNESS,
    "edge_betweenness": EDGE_BETWEENNESS,
    "cluster_labels": CLUSTER_LABELS,
}


PARAMETER_VALUES = {
    "alpha": 0.85,
    "beta": 1.15,
    "gamma": 1.35,
    "lambda": 0.35,
    "eta": 1.25,
    "lambda_z": 0.45,
    "q": 1.1,
    "lambda_k": 0.55,
    "K": 2.0,
    "lambda_s": 0.4,
    "kappa": 1.2,
    "rho": 0.25,
    "Gamma": 0.3,
    "chi": 0.7,
    "delta": 0.8,
    "rho_in": 1.1,
    "rho_out": 0.35,
    "K_clusters": 2.0,
    "k": 2.0,
    "C": 0.4,
    "eta_e": 1.4,
    "eta_l": 1.1,
    "eta_c": 2.0,
    "a": 1.0,
    "r0": 2.5,
    "eta1": 0.9,
    "eta2": 1.8,
    "mu": 0.6,
    "c": 0.7,
    "R": 2.0,
    "beta_r": 0.3,
    "beta_t": 0.08,
    "b": 0.25,
}


def candidate_params(candidate_id: str, *, localized: bool = True):
    spec = get_candidate(candidate_id)
    result = {}
    discrete = {"K", "K_clusters", "k", "R"}
    for name, scope in zip(spec.parameter_names, spec.parameter_scopes):
        value = PARAMETER_VALUES[name]
        if localized and scope == "node" and name not in discrete:
            value = np.linspace(value, value * 1.15, NODES, dtype=np.float64)
        result[name] = value
    return result


class RegistryTests(unittest.TestCase):
    def test_registry_has_exact_paper_rows(self):
        self.assertEqual(len(list_candidates()), 26)
        self.assertEqual([item.paper_rows["table_1"] for item in list_candidates("table_1")], list(range(1, 12)))
        self.assertEqual([item.paper_rows["table_3"] for item in list_candidates("table_3")], list(range(1, 16)))
        self.assertEqual([item.paper_rows["table_4"] for item in list_candidates("table_4")], list(range(1, 27)))

    def test_json_registry_matches_runtime_registry(self):
        loaded = load_registry(REPOSITORY / "artifacts" / "candidates.json")
        runtime = list_candidates()
        self.assertEqual([item.id for item in loaded], [item.id for item in runtime])
        for left, right in zip(loaded, runtime):
            self.assertEqual(left, right)

    def test_aliases_resolve_without_executable_dispatch(self):
        self.assertEqual(get_candidate("C9").id, "betweenness_gravity")
        self.assertEqual(get_candidate("interaction_SE3").id, "distance_weighted_diffusion")
        self.assertEqual(get_candidate("Radiation model").id, "radiation")
        with self.assertRaises(KeyError):
            get_candidate("interaction_not_a_real_candidate")

    def test_table_specific_complexity_is_preserved(self):
        candidate = get_candidate("single_constrained_gravity_power")
        self.assertEqual(candidate.complexities, {"table_1": 21, "table_4": 17})

    def test_table_4_rows_match_reported_ranking(self):
        expected = {
            "distance_weighted_diffusion": 13,
            "span_diffusion": 14,
            "rank_based_diffusion": 15,
            "classic_gravity_power": 16,
            "saturation_limited_diffusion": 17,
            "cluster_diffusion": 18,
        }
        self.assertEqual(
            {name: get_candidate(name).paper_rows["table_4"] for name in expected},
            expected,
        )


class NumpyEquationTests(unittest.TestCase):
    def test_all_26_candidates_are_finite_with_localized_parameters(self):
        for spec in list_candidates("table_4"):
            with self.subTest(candidate=spec.id):
                result = evaluate(
                    spec.id,
                    STATE,
                    DISTANCE,
                    candidate_params(spec.id),
                    adjacency=ADJACENCY,
                    **CONTEXT,
                )
                self.assertEqual(result.shape, (NODES,))
                self.assertTrue(np.isfinite(result).all())

    def test_distance_diffusion_matches_direct_formula(self):
        coefficient = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9])
        eta = 1.3
        safe_distance = np.where(DISTANCE == 0, 1.0e12, DISTANCE)
        weight = safe_distance ** (-eta) * ADJACENCY
        expected = coefficient * (weight @ STATE - weight.sum(axis=1) * STATE)
        actual = evaluate(
            "distance_weighted_diffusion",
            STATE,
            DISTANCE,
            {"lambda": coefficient, "eta": eta},
            adjacency=ADJACENCY,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_production_constrained_gravity_conserves_total_state(self):
        for candidate in ("single_constrained_gravity_power", "single_constrained_gravity_exponential"):
            result = evaluate(candidate, STATE, DISTANCE, candidate_params(candidate), adjacency=ADJACENCY)
            self.assertAlmostEqual(float(result.sum()), 0.0, places=11)

    def test_input_and_parameter_shapes_are_validated(self):
        with self.assertRaises(EquationInputError):
            evaluate("radiation", STATE[None, :], DISTANCE, ())
        with self.assertRaises(EquationInputError):
            evaluate("radiation", STATE, DISTANCE[:-1], ())
        with self.assertRaises(EquationInputError):
            evaluate("distance_weighted_diffusion", STATE, DISTANCE, {"lambda": 1.0})
        with self.assertRaises(EquationInputError):
            evaluate(
                "distance_weighted_diffusion",
                STATE,
                DISTANCE,
                {"lambda": np.ones(NODES - 1), "eta": 1.0},
            )
        with self.assertRaises(EquationInputError):
            evaluate(
                "distance_weighted_diffusion",
                STATE,
                DISTANCE,
                {"lambda": 1.0, "eta": np.ones(NODES)},
            )

    def test_precomputed_rank_and_knn_context_preserve_results(self):
        for candidate in ("rank_based_diffusion", "knn_diffusion", "span_diffusion"):
            with self.subTest(candidate=candidate):
                params = candidate_params(candidate, localized=False)
                context = precompute_context(
                    candidate, DISTANCE, params, adjacency=ADJACENCY
                )
                direct = evaluate(
                    candidate, STATE, DISTANCE, params, adjacency=ADJACENCY
                )
                precomputed = evaluate(
                    candidate,
                    STATE,
                    DISTANCE,
                    params,
                    adjacency=ADJACENCY,
                    **context,
                )
                np.testing.assert_allclose(precomputed, direct, rtol=0.0, atol=0.0)


class TorchEquationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def test_torch_matches_numpy_and_retains_state_gradients(self):
        torch = self.torch
        distance = torch.as_tensor(DISTANCE, dtype=torch.float64)
        adjacency = torch.as_tensor(ADJACENCY)
        for spec in list_candidates("table_4"):
            with self.subTest(candidate=spec.id):
                numpy_params = candidate_params(spec.id)
                expected = evaluate(
                    spec.id,
                    STATE,
                    DISTANCE,
                    numpy_params,
                    adjacency=ADJACENCY,
                    **CONTEXT,
                )
                state = torch.tensor(STATE, dtype=torch.float64, requires_grad=True)
                torch_params = {
                    name: torch.as_tensor(value, dtype=torch.float64)
                    for name, value in numpy_params.items()
                }
                actual = evaluate(
                    spec.id,
                    state,
                    distance,
                    torch_params,
                    adjacency=adjacency,
                    node_betweenness=torch.as_tensor(NODE_BETWEENNESS, dtype=torch.float64),
                    edge_betweenness=torch.as_tensor(EDGE_BETWEENNESS, dtype=torch.float64),
                    cluster_labels=torch.as_tensor(CLUSTER_LABELS),
                )
                np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-10, atol=1e-10)
                (actual.square().sum()).backward()
                self.assertIsNotNone(state.grad)
                self.assertTrue(torch.isfinite(state.grad).all().item())


if __name__ == "__main__":
    unittest.main()
