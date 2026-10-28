"""Scalar observation backend selection; no survey data or target run."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cf4_r2_scalar_primal import scalar_observation_terms  # noqa: E402


class StubObservation:
    def __init__(self):
        self.calls = []

    def value(self, *args):
        self.calls.append(("value", args))
        return -12.5, (3.0, -2.0)

    def derivative(self, *args):
        self.calls.append(("derivative", args))
        return (-12.5, (3.0, -2.0)), ("unused gradient",)


class ScalarPrimalTests(unittest.TestCase):
    def test_backends_return_the_declared_primal_and_preserve_arguments(self):
        args = (object(), object(), object(), object(), (), {}, {}, 2)
        for backend, expected_method in (
                ("direct_value_only", "value"),
                ("matched_value_and_gradient_primal", "derivative")):
            with self.subTest(backend=backend):
                obs = StubObservation()
                result = scalar_observation_terms(obs, *args, backend=backend)
                self.assertEqual(result, (-12.5, (3.0, -2.0)))
                self.assertEqual([method for method, _ in obs.calls], [expected_method])
                self.assertEqual(obs.calls[0][1], args)

    def test_unknown_backend_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported scalar observation backend"):
            scalar_observation_terms(
                StubObservation(), 1, 2, 3, 4, (), {}, {}, 2, backend="surrogate")


if __name__ == "__main__":
    unittest.main()
