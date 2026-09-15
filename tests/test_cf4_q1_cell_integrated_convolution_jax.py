"""Q1 fixed-geometry contraction checks; runnable with standard unittest."""
import json
import resource
import time
import unittest

import numpy as np

from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit
from cf4_q1_cell_integrated_convolution_jax import (
    JaxOperatorInputError, assert_q1_numpy_provenance, contract_mass_field,
    contract_mass_field_from_jax, jax, jnp, mass_gradient_basis,
)


class Q1ContractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if jax is None:
            raise RuntimeError("JAX missing: this validation must not silently skip")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("Validation requires JAX_ENABLE_X64=1")
        cls.cases = []
        pos = np.array([[0.37, 1.22, 2.31], [5.71, 0.42, 3.66], [2.08, 4.90, 5.43]])
        los = np.array([[1., 0., 0.], [0., .6, .8], [1., 2., 2.]])
        los /= np.linalg.norm(los, axis=1)[:, None]
        edge = np.array([[1.5-1e-11, 1.5, 1.5+1e-11], [6.-1e-11, .42, 3.66], [.01, 2.02, 4.03]])
        for name, positions, scales in (
            ("nondegenerate", pos, np.array([.31, .77, 1.05])),
            ("seam_knot_sliver_cold", edge, np.array([.31, .77, 0.])),
            ("multiwrap_sigma_2L", pos, np.full(3, 12.)),
        ):
            basis = np.stack([cell_integrated_tsc_deposit(
                positions[i:i+1], [1.], los[i:i+1], scales[i:i+1], 4, 6.
            ) for i in range(3)])
            cls.cases.append((name, positions, los, scales, basis))

    def test_value_conservation_and_six_mass_states(self):
        for name, pos, los, scales, basis in self.cases:
            for population in range(6):
                with self.subTest(case=name, population=population):
                    mass = np.array([.7, 1.2, .3]) * (population+1)
                    expected = cell_integrated_tsc_deposit(pos, mass, los, scales, 4, 6.)
                    actual = np.asarray(contract_mass_field(basis, mass))
                    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
                    self.assertAlmostEqual(actual.sum(), mass.sum(), places=11)

    def test_jacobian_vjp_and_scalar_finite_difference(self):
        rng = np.random.default_rng(20260905)
        for name, pos, los, scales, basis in self.cases:
            with self.subTest(case=name):
                mass = jnp.array([.7, 1.2, .3])
                function = lambda m: contract_mass_field_from_jax(basis, m)
                jacobian = np.asarray(jax.jacfwd(function)(mass))
                np.testing.assert_allclose(np.moveaxis(jacobian, -1, 0), basis, atol=2e-12, rtol=0)
                np.testing.assert_array_equal(np.asarray(mass_gradient_basis(basis)), basis)
                cotangent = rng.normal(size=(4, 4, 4))
                _, pullback = jax.vjp(function, mass)
                actual = np.asarray(pullback(jnp.asarray(cotangent))[0])
                expected = np.sum(basis * cotangent, axis=(1, 2, 3))
                np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
                # Independent scalar finite differences through the NumPy oracle.
                direction = rng.normal(size=3)
                step = 1e-5
                plus = cell_integrated_tsc_deposit(pos, np.asarray(mass)+step*direction, los, scales, 4, 6.)
                minus = cell_integrated_tsc_deposit(pos, np.asarray(mass)-step*direction, los, scales, 4, 6.)
                fd = np.sum(cotangent*(plus-minus))/(2*step)
                np.testing.assert_allclose(actual @ direction, fd, atol=1e-8, rtol=1e-5)

    def test_validation_and_static_limit(self):
        assert_q1_numpy_provenance()
        basis = self.cases[0][-1]
        for value in (np.nan, -1., 1.):
            broken = basis.copy()
            broken[0, 0, 0, 0] = value
            with self.assertRaises(JaxOperatorInputError):
                contract_mass_field(broken, [.7, 1.2, .3])
        with self.assertRaises(JaxOperatorInputError):
            contract_mass_field(basis, [-1., 0., 1.])
        with self.assertRaises(JaxOperatorInputError):
            jax.eval_shape(contract_mass_field_from_jax, jax.ShapeDtypeStruct(basis.shape, np.float64), jax.ShapeDtypeStruct((2,), np.float64))
        with self.assertRaises(JaxOperatorInputError):
            jax.eval_shape(contract_mass_field_from_jax, jax.ShapeDtypeStruct((2, 256, 256, 256), np.float64), jax.ShapeDtypeStruct((2,), np.float64))
        previous_x64 = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", False)
            with self.assertRaises(JaxOperatorInputError):
                contract_mass_field(basis, [.7, 1.2, .3])
        finally:
            jax.config.update("jax_enable_x64", previous_x64)

    def test_representative_timing(self):
        basis = jnp.asarray(self.cases[0][-1])
        mass = jnp.array([.7, 1.2, .3])
        function = jax.jit(contract_mass_field_from_jax)
        start = time.perf_counter()
        function(basis, mass).block_until_ready()
        compile_seconds = time.perf_counter()-start
        start = time.perf_counter()
        function(basis, mass).block_until_ready()
        warm_seconds = time.perf_counter()-start
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        self.assertLess(warm_seconds, 1.)
        self.assertLess(peak_mib, 2000.)
        print(json.dumps({"scope": "tiny fixed-geometry fixture only", "device": str(jax.devices()[0]),
            "compile_and_first_seconds": compile_seconds, "warm_seconds": warm_seconds,
            "peak_host_mib_so_far": peak_mib, "gpu_memory": "not_applicable_CPU_only"}), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
