import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_continuous_tracer import cell_centres
from cf4_r2_linked_singleton_target import partial_v6_count_singleton_parts
from cf4_r2_shell_cdf_count import predict_shell_cdf_intensity
from cf4_r2_marked_tracer_jax import (
    intrinsic_biased_source_masses, intrinsic_lf_bin_fractions,
    predict_source_marked_intensity, sparse_marked_poisson_log_likelihood,
)


class PartialV6TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64', True)

    def test_same_field_count_and_singleton_terms_keep_holdout_readout_separate(self):
        n, box = 4, 384.
        positions = cell_centres(n, box, .5)
        # A source between the frozen survey limit and half-box catches window
        # drift; this deliberately irregular geometry is a wiring fixture.
        positions = jnp.asarray(positions).at[0].set(jnp.array([377., 192., 192.]))
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
        train_keys = jnp.array([source_id], dtype=jnp.int32)
        train_counts = jnp.array([2.])

        def evaluate(width_white,mode='gh'):
            tracer = jnp.zeros(9).at[6].set(width_white)
            parts, _ = partial_v6_count_singleton_parts(
                jnp.ones((n, n, n)), jnp.zeros((3, n, n, n)),
                jnp.zeros(4), tracer, source_geometry, links,
                train_keys, train_counts, train_exposure, box=box,
                quadrature_order=3,count_integration=mode,
                count_cdf_order=4,count_cdf_segments=4)
            return parts

        parts = jax.jit(evaluate)(0.)
        total_grad = jax.jit(jax.grad(lambda x: jnp.sum(evaluate(x)[:3])))(0.)
        self.assertEqual(parts.shape, (4,))
        self.assertTrue(np.isfinite(np.asarray(parts[:3])).all())
        self.assertAlmostEqual(float(parts[3]), 0., places=12)
        self.assertTrue(np.isfinite(float(total_grad)))
        self.assertNotEqual(float(total_grad), 0.)
        # Independent count-only reference catches a missing sigma argument or
        # a 192 instead of 180 survey limit; nonempty observed keys are essential.
        reference_fraction = jnp.sum(intrinsic_lf_bin_fractions()[1:4])
        intrinsic = intrinsic_biased_source_masses(
            jnp.ones((n, n, n)), jnp.log(reference_fraction), jnp.ones(5),
            reference_interval=(-25., -21.))
        def reference(width_white, radial_max):
            intensity = predict_source_marked_intensity(
                positions, jnp.zeros_like(positions), intrinsic,
                source_geometry['angular'], observer=observer,
                box_size_cMpc_h=box, hubble_km_s_Mpc=74.6, little_h=.746,
                radius_table_cMpc_h=rtab,
                modulus_table_h=source_geometry['modulus_table'],
                redshift_table=source_geometry['redshift_table'], grid_size=n,
                sigma_los_km_s=100*jnp.exp(.5*width_white),
                radial_min_cMpc_h=5., radial_max_cMpc_h=radial_max,
                quadrature_order=3)
            return sparse_marked_poisson_log_likelihood(
                intensity, train_keys, train_counts, selected_voxel_mask=train_exposure)
        reference = jax.jit(reference, static_argnums=1)
        expected = reference(0., 180.)
        np.testing.assert_allclose(float(parts[0]), float(expected), atol=1e-10)
        self.assertGreater(abs(float(reference(0., 192.)-expected)), 1e-8)
        count_derivative = float((evaluate(.001)[0]-evaluate(-.001)[0])/.002)
        reference_derivative = float((reference(.001, 180.)-reference(-.001, 180.))/.002)
        self.assertGreater(abs(reference_derivative), 1e-10)
        np.testing.assert_allclose(count_derivative, reference_derivative, atol=1e-8)
        # Independent shell-integral reference checks the new target branch's
        # nonempty count, width and frozen survey window wiring.
        direct=predict_shell_cdf_intensity(positions,jnp.zeros_like(positions),
            intrinsic,source_geometry['angular'],observer=observer,box_size_cMpc_h=box,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=rtab,
            modulus_table_h=source_geometry['modulus_table'],
            redshift_table=source_geometry['redshift_table'],grid_size=n,
            sigma_los_km_s=100.,radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,
            order=4,segments=4)
        expected=sparse_marked_poisson_log_likelihood(direct,train_keys,train_counts,
                                                    selected_voxel_mask=train_exposure)
        shell_parts=jax.jit(lambda width:evaluate(width,'shell_cdf'))(0.)
        np.testing.assert_allclose(float(shell_parts[0]),float(expected),atol=1e-10)
        np.testing.assert_allclose(np.asarray(shell_parts[1:]),np.asarray(parts[1:]),atol=1e-12)


if __name__ == '__main__':
    unittest.main()
