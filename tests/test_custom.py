"""Tests for user equation loading and early differentiability validation."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from udiscovery.custom import Equation, load_equations, validate_equation
from udiscovery.data import Dataset


def identity(state, distance, parameters, *, adjacency):
    return state


class EquationDefinitionTests(unittest.TestCase):
    def test_metadata_preserves_order_and_defaults(self):
        initial = {"k": 0.2, "K": 0.3, "lambda": 0.4}
        equation = Equation("custom-1", identity, initial, 5, scopes={"K": "node"})
        initial["k"] = 1.0
        self.assertEqual(equation.parameters["k"], 0.2)
        self.assertEqual(equation.parameter_names, ("k", "K", "lambda"))
        self.assertEqual(equation.parameter_scopes, ("global", "node", "global"))
        self.assertEqual(equation.display_name, "custom-1")
        named = Equation("named", identity, {}, 1, name="Readable label", formula="x")
        self.assertEqual(named.display_name, "Readable label")

    def test_invalid_definitions_are_rejected(self):
        cases = [
            ({"id": "../unsafe"}, "id"),
            ({"id": ""}, "id"),
            ({"id": "has space"}, "id"),
            ({"function": None}, "callable"),
            ({"complexity": 0}, "positive integer"),
            ({"complexity": 1.5}, "positive integer"),
            ({"complexity": True}, "positive integer"),
            ({"parameters": []}, "parameters must map"),
            ({"parameters": {"2bad": 1.0}}, "parameter name"),
            ({"parameters": {"a": 0.0}}, "finite positive scalar"),
            ({"parameters": {"a": -0.1}}, "finite positive scalar"),
            ({"parameters": {"a": float("nan")}}, "finite positive scalar"),
            ({"parameters": {"a": float("inf")}}, "finite positive scalar"),
            ({"parameters": {"a": [1.0]}}, "finite positive scalar"),
            ({"parameters": {"a": True}}, "finite positive scalar"),
            ({"scopes": {"unknown": "global"}}, "unknown parameter"),
            ({"scopes": {"a": "edge"}}, "scope"),
            ({"scopes": {"a": []}}, "scope"),
            ({"scopes": []}, "scopes must map"),
            ({"bounded_parameters": ("unknown",)}, "unknown bounded parameter"),
            ({"bounded_parameters": ("a", "a")}, "duplicate bounded parameter"),
            ({"bounded_parameters": "a"}, "tuple or list"),
            ({"bounded_parameters": ("a",), "parameters": {"a": 1.0}}, "strictly between"),
            ({"name": None}, "must be strings"),
            ({"formula": 3}, "must be strings"),
        ]
        for updates, message in cases:
            with self.subTest(updates=updates):
                kwargs = dict(id="valid", function=identity, parameters={"a": 0.4}, complexity=2)
                kwargs.update(updates)
                with self.assertRaisesRegex(ValueError, message):
                    Equation(**kwargs)

    def test_numpy_scalar_initial_values_and_bounded_names(self):
        equation = Equation("numpy", identity, {"a": np.float32(0.4)}, np.int64(2), bounded_parameters=["a"])
        self.assertIsInstance(equation.parameters["a"], float)
        self.assertIsInstance(equation.complexity, int)
        self.assertEqual(equation.bounded_parameters, ("a",))


class EquationLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "my equations.py"
        self.addCleanup(self.temporary.cleanup)

    def write(self, source):
        self.source.write_text(textwrap.dedent(source), encoding="utf-8")

    def test_loads_a_user_file_without_changing_python_path(self):
        self.write("""
            from udiscovery.custom import Equation
            def derivative(state, distance, parameters, *, adjacency):
                return parameters['a'] * state
            EQUATIONS = (Equation('linear', derivative, {'a': 0.2}, 2),)
        """)
        previous_path = list(sys.path)
        result = load_equations(self.source)
        self.assertEqual([item.id for item in result], ["linear"])
        self.assertEqual(sys.path, previous_path)

    def test_relative_helpers_work_at_import_and_call_time(self):
        (self.directory / "helper.py").write_text("FACTOR = 3\n", encoding="utf-8")
        self.write("""
            from .helper import FACTOR
            from udiscovery.custom import Equation
            def derivative(state, distance, parameters, *, adjacency):
                from .helper import FACTOR
                return FACTOR * state
            EQUATIONS = [Equation('helper', derivative, {}, 2)]
        """)
        equations = load_equations(self.source)
        self.assertEqual(equations[0].function(2, None, {}, adjacency=None), 6)

    def test_failed_imports_restore_private_module_namespace(self):
        self.write("raise RuntimeError('intentional failure')")
        before = {name for name in sys.modules if name.startswith("_udiscovery_equations_")}
        with self.assertRaisesRegex(ValueError, "intentional failure"):
            load_equations(self.source)
        after = {name for name in sys.modules if name.startswith("_udiscovery_equations_")}
        self.assertEqual(before, after)

    def test_invalid_exports_and_case_colliding_ids_are_rejected(self):
        header = "from udiscovery.custom import Equation\nf = lambda *a, **kw: None\n"
        for declaration, message in [
            ("", "nonempty list or tuple"),
            ("EQUATIONS = []", "nonempty list or tuple"),
            ("EQUATIONS = {}", "nonempty list or tuple"),
            ("EQUATIONS = [1]", "Equation instance"),
            ("EQUATIONS = [Equation('same', f, {}, 1), Equation('same', f, {}, 1)]", "duplicate equation id"),
            ("EQUATIONS = [Equation('same', f, {}, 1), Equation('SAME', f, {}, 1)]", "duplicate equation id"),
        ]:
            with self.subTest(declaration=declaration):
                self.write(header + declaration)
                with self.assertRaisesRegex(ValueError, message):
                    load_equations(self.source)

    def test_missing_and_non_python_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_equations(self.source)
        other = self.source.with_suffix(".txt")
        other.write_text("EQUATIONS = []", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, ".py file"):
            load_equations(other)

    def test_rapid_reload_uses_current_source(self):
        header = "from udiscovery.custom import Equation\nf = lambda *a, **kw: None\n"
        self.write(header + "EQUATIONS = [Equation('first', f, {}, 1)]")
        self.assertEqual(load_equations(self.source)[0].id, "first")
        self.write(header + "EQUATIONS = [Equation('other', f, {}, 1)]")
        self.assertEqual(load_equations(self.source)[0].id, "other")


class EquationAutogradTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def setUp(self):
        states = np.asarray([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]], dtype=np.float32)
        distance = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :]).astype(np.float32)
        self.dataset = Dataset(states, states[0].copy(), distance, (distance > 0).astype(np.float32), np.arange(3), 2, 4, 1.0, Path("test.npz"))

    def validate(self, function, parameters=None, **kwargs):
        equation = Equation("test", function, {"a": 0.5} if parameters is None else parameters, 2, **kwargs)
        return validate_equation(equation, self.dataset, "cpu")

    def test_each_parameter_receives_its_declared_shape_and_actual_data(self):
        observations = []
        def function(state, distance, parameters, *, adjacency):
            self.assertEqual(parameters["k"].shape, ())
            self.assertEqual(parameters["K"].shape, state.shape)
            self.assertTrue(parameters["k"].is_leaf)
            self.assertTrue(parameters["K"].is_leaf)
            self.assertEqual(float(distance[0, 2]), 2.0)
            self.assertEqual(float(adjacency[0, 0]), 0.0)
            observations.append(float(state[0].detach()))
            return parameters["k"] * state + parameters["K"]
        self.assertIsNone(self.validate(function, {"k": 0.2, "K": 0.8}, scopes={"K": "node"}))
        self.assertEqual(observations, [1.0, 2.0])

    def test_unused_parameter_is_detected_even_when_other_parameter_is_connected(self):
        def function(state, distance, parameters, *, adjacency):
            return parameters["a"] * state
        with self.assertRaisesRegex(ValueError, "parameter 'b' is unused or detached"):
            self.validate(function, {"a": 0.5, "b": 0.5})

    def test_detached_parameter_is_detected_independently(self):
        def function(state, distance, parameters, *, adjacency):
            return parameters["a"] * state + parameters["b"].detach()
        with self.assertRaisesRegex(ValueError, "parameter 'b' is unused or detached"):
            self.validate(function, {"a": 0.5, "b": 0.5})

    def test_constant_derivative_and_connected_zero_gradient_are_valid(self):
        def constant_derivative(state, distance, parameters, *, adjacency):
            return parameters["a"].expand_as(state)
        self.validate(constant_derivative)
        def conservative(state, distance, parameters, *, adjacency):
            return parameters["a"] * (state.mean() - state)
        self.validate(conservative)
        def locally_flat(state, distance, parameters, *, adjacency):
            return parameters["a"] * 0 + state
        self.validate(locally_flat)

    def test_parameter_free_equations_are_allowed(self):
        self.validate(lambda state, *args, **kwargs: self.torch.zeros_like(state), {})

    def test_detached_output_and_invalid_output_types_are_rejected(self):
        cases = [
            (lambda state, *args, **kwargs: state.detach(), "detached from all"),
            (lambda state, *args, **kwargs: np.zeros(3), "PyTorch Tensor"),
            (lambda state, *args, **kwargs: state[:, None], "output must have shape"),
            (lambda state, *args, **kwargs: state.double(), "dtype must match"),
            (lambda state, *args, **kwargs: state * float("nan"), "non-finite values"),
            (lambda state, *args, **kwargs: state.to("meta"), "device must match"),
        ]
        for function, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(function)

    def test_nonfinite_gradients_fail_even_for_finite_outputs(self):
        def function(state, distance, parameters, *, adjacency):
            return self.torch.sqrt(parameters["a"] - 0.5).expand_as(state)
        with self.assertRaisesRegex(ValueError, "non-finite gradient"):
            self.validate(function)

    def test_user_exceptions_identify_the_candidate(self):
        def function(state, distance, parameters, *, adjacency):
            raise RuntimeError("bad implementation")
        with self.assertRaisesRegex(ValueError, "Equation 'test'.*bad implementation"):
            self.validate(function)

    def test_validation_also_checks_the_final_training_state(self):
        def function(state, distance, parameters, *, adjacency):
            return parameters["a"] * state / (state[0] - 2)
        with self.assertRaisesRegex(ValueError, "timestep 1.*non-finite"):
            self.validate(function)

    def test_input_mutation_is_rejected_and_dataset_is_preserved(self):
        def function(state, distance, parameters, *, adjacency):
            distance.zero_()
            adjacency.zero_()
            return parameters["a"] * state
        old_distance = self.dataset.distance.copy()
        old_adjacency = self.dataset.adjacency.copy()
        with self.assertRaisesRegex(ValueError, "modified its distance input"):
            self.validate(function)
        np.testing.assert_array_equal(self.dataset.distance, old_distance)
        np.testing.assert_array_equal(self.dataset.adjacency, old_adjacency)

    def test_adjacency_and_no_grad_state_mutation_are_rejected(self):
        def adjacency_mutation(state, distance, parameters, *, adjacency):
            adjacency.zero_()
            return parameters["a"] * state
        with self.assertRaisesRegex(ValueError, "modified its adjacency input"):
            self.validate(adjacency_mutation)
        def state_mutation(state, distance, parameters, *, adjacency):
            with self.torch.no_grad():
                state.add_(1)
            return parameters["a"] * state
        with self.assertRaisesRegex(ValueError, "modified its state input"):
            self.validate(state_mutation)

    def test_parameter_mapping_mutation_cannot_hide_an_unused_parameter(self):
        def function(state, distance, parameters, *, adjacency):
            parameters.pop("b")
            return parameters["a"] * state
        with self.assertRaisesRegex(ValueError, "modified the parameter mapping"):
            self.validate(function, {"a": 0.5, "b": 0.5})

    def test_validation_preserves_the_full_graph_for_node_specific_constants(self):
        nodes = 11
        states = np.ones((5, nodes), dtype=np.float32)
        distance = np.ones((nodes, nodes), dtype=np.float32) - np.eye(nodes, dtype=np.float32)
        self.dataset = Dataset(states, states[0].copy(), distance, distance.copy(), np.arange(nodes), 2, 4, 1.0, Path("test.npz"))
        constants = self.torch.arange(1, nodes + 1, dtype=self.torch.float32)
        def function(state, distance, parameters, *, adjacency):
            self.assertEqual(state.shape, (nodes,))
            self.assertEqual(distance.shape, (nodes, nodes))
            return parameters["a"] * state * constants
        self.validate(function)


if __name__ == "__main__":
    unittest.main()
