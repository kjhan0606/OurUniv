import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_z0_pm_bridge import enable_pmwd_type_description_compatibility
from cf4_r1_tsc_diagnostic import stencil, deposit, interpolate, force


class TSCDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_pmwd_type_description_compatibility()
        cls.rng = np.random.default_rng(2026091405)

    def test_weights_mass_periodicity_and_transpose(self):
        x = jnp.asarray(self.rng.uniform(-2, 10, (35, 3)))
        x = x.at[:3].set(jnp.array([[0., 0., 0.], [.5, .5, .5], [8., -8., 16.]]))
        index, weight = stencil(x, 8, 1.)
        np.testing.assert_allclose(weight.sum(axis=1), 1., atol=3e-15)
        self.assertGreaterEqual(float(weight.min()), 0.)
        np.testing.assert_allclose(np.sort(weight[0]), np.sort(np.array([
            a*b*c for a in (.125, .75, .125) for b in (.125, .75, .125)
            for c in (.125, .75, .125)])), atol=1e-15)
        v = jnp.asarray(self.rng.uniform(.1, 2, 35))
        field = jnp.asarray(self.rng.normal(size=(8,)*3))
        mesh = deposit(x, v, 8, 1., chunk_size=16)
        np.testing.assert_allclose(mesh.sum(), v.sum(), atol=2e-13)
        np.testing.assert_allclose(deposit(x+8, v, 8, 1., 16), mesh, atol=2e-14)
        np.testing.assert_allclose(jnp.sum(mesh*field), jnp.dot(v, interpolate(x, field, 1., 16)), atol=2e-13)
        np.testing.assert_allclose(interpolate(x, jnp.ones((8, 8, 8, 3)), 1., 16), 1., atol=2e-15)

    def test_force_self_net_and_integer_mesh_translation(self):
        x = jnp.asarray(self.rng.uniform(0, 8, (35, 3)))
        kernel = jax.jit(lambda y: force(y, 8, 1., .31, 16))
        actual = kernel(x)
        np.testing.assert_allclose(actual.sum(axis=0), 0., atol=2e-12)
        np.testing.assert_allclose(kernel(x+jnp.array([1., -2., 3.])), actual, atol=2e-12)
        np.testing.assert_allclose(force(x[:1], 8, 1., .31, 16), 0., atol=2e-12)

    def test_full_force_gradient_across_nodes_and_stencil_boundaries(self):
        x = jnp.asarray(self.rng.uniform(0, 8, (35, 3)))
        weights = jnp.asarray(self.rng.normal(size=(35, 3)))
        scalar = jax.jit(lambda t: jnp.sum(force(x.at[0, 0].set(t), 8, 1., .31, 16)*weights))
        derivative = jax.jit(jax.grad(scalar))
        rows = []
        for center in (0., .5, 8.):
            for offset in (-1e-4, 0., 1e-4):
                t = center+offset
                analytic = float(derivative(t))
                finite = float((scalar(t+1e-6)-scalar(t-1e-6))/2e-6)
                error = abs(analytic-finite)/max(1., abs(analytic), abs(finite))
                rows.append(error)
                self.assertLess(error, 2e-5)
            jump = abs(float(derivative(center+1e-7)-derivative(center-1e-7)))
            self.assertLess(jump, 2e-5)
        print('TSC full-force gradient max scaled FD error:', max(rows), flush=True)


if __name__ == '__main__':
    unittest.main()
