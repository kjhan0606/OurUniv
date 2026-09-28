import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_continuous_tracer import cell_centres
from cf4_r2_linked_singleton_target import partial_v6_count_singleton_parts


class PartialV6TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64', True)

    def test_same_field_count_and_singleton_terms_keep_holdout_readout_separate(self):
        n, box = 4, 384.
        positions = cell_centres(n, box, .5)
        axis_index = 2
        source_id = axis_index*n*n + axis_index*n + axis_index
        observer = jnp.full(3, box/2.)
        relative = positions[source_id]-observer
        r_obs = float(jnp.linalg.norm(relative))
        rtab = jnp.linspace(.001, box/2., 4001)
        source_geometry = dict(
            positions=positions, angular=jnp.ones((2, n**3)),
            radial_table=rtab, modulus_table=5*jnp.log10(rtab)+25.,
            redshift_table=rtab/3000.)
        ids = jnp.arange(n**3, dtype=jnp.int32)[None, :]
        links = {}
        for population in range(6):
            if population == 0:
                links[population] = dict(
                    candidate_source_ids=ids,
                    candidate_mask=jnp.ones_like(ids, dtype=bool),
                    association_logprob=jnp.zeros((1, 5, n**3)),
                    voxel_ijk=jnp.array([[2, 2, 2]], dtype=jnp.int32),
                    observed_radius_cMpc_h=jnp.array([r_obs]),
                    dz_row=jnp.array([1.08*r_obs]),
                    eta_mean=jnp.array([0.]), eta_std=jnp.array([.15]),
                    eta_alpha=jnp.array([.2]))
            else:
                links[population] = dict(
                    candidate_source_ids=jnp.zeros((0, n**3), dtype=jnp.int32),
                    candidate_mask=jnp.zeros((0, n**3), dtype=bool),
                    association_logprob=jnp.zeros((0, 5, n**3)),
                    voxel_ijk=jnp.zeros((0, 3), dtype=jnp.int32),
                    observed_radius_cMpc_h=jnp.zeros((0,)),
                    dz_row=jnp.zeros((0,)), eta_mean=jnp.zeros((0,)),
                    eta_std=jnp.zeros((0,)), eta_alpha=jnp.zeros((0,)))
        train_exposure = jnp.ones(6*n**3, dtype=bool)
        heldout_exposure = jnp.zeros(6*n**3, dtype=bool)
        empty_keys = jnp.zeros((0,), dtype=jnp.int32)
        empty_counts = jnp.zeros((0,), dtype=jnp.float64)

        def evaluate(width_white):
            tracer = jnp.zeros(9).at[6].set(width_white)
            parts, _ = partial_v6_count_singleton_parts(
                jnp.ones((n, n, n)), jnp.zeros((3, n, n, n)),
                jnp.zeros(4), tracer, source_geometry, links,
                empty_keys, empty_counts, train_exposure,
                empty_keys, empty_counts, heldout_exposure, box=box)
            return parts

        parts = jax.jit(evaluate)(0.)
        total_grad = jax.jit(jax.grad(lambda x: jnp.sum(evaluate(x)[:3])))(0.)
        heldout_grad = jax.jit(jax.grad(lambda x: evaluate(x)[3]))(0.)
        self.assertEqual(parts.shape, (4,))
        self.assertTrue(np.isfinite(np.asarray(parts[:3])).all())
        self.assertAlmostEqual(float(parts[3]), 0., places=12)
        self.assertTrue(np.isfinite(float(total_grad)))
        self.assertNotEqual(float(total_grad), 0.)
        self.assertAlmostEqual(float(heldout_grad), 0., places=12)


if __name__ == '__main__':
    unittest.main()
