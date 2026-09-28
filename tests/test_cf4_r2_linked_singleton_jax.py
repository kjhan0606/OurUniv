import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_linked_singleton_jax import linked_singleton_logfactors_for_population


class LinkedSingletonBatchTests(unittest.TestCase):
    def test_batched_factors_match_individual_and_trace_los_nuisance(self):
        positions = jnp.array([[20., 12., 12.], [17., 12., 12.]])
        velocities = jnp.zeros((2, 3))
        intrinsic = jnp.array([[.2, .3], [.4, .2], [.5, .8],
                               [.7, .4], [.9, .6]])
        angular = jnp.ones((2, 2))
        ids = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        active = jnp.ones((2, 2), dtype=bool)
        association = jnp.zeros((2, 5, 2))
        voxels = jnp.array([[3, 2, 2], [2, 2, 2]], dtype=jnp.int32)
        radius = jnp.array([8., 5.])
        dz = jnp.array([9., 6.])
        eta_mean = jnp.array([0., .02])
        eta_std = jnp.array([.12, .18])
        eta_alpha = jnp.array([0., .3])
        rtab = jnp.linspace(.001, 12., 1201)
        radial = dict(observer=jnp.array([12., 12., 12.]),
                      box_size_cMpc_h=24., hubble_km_s_Mpc=75.,
                      little_h=.75, radius_table_cMpc_h=rtab,
                      modulus_table_h=5*jnp.log10(rtab)+25.,
                      redshift_table=rtab/3000., grid_size=4,
                      radial_min_cMpc_h=.1, radial_max_cMpc_h=12.)

        def batched(log_sigma):
            factors, _ = linked_singleton_logfactors_for_population(
                positions, velocities, intrinsic, angular, ids, active,
                association, voxels, radius, dz, eta_mean, eta_std, eta_alpha,
                population=0, sigma_los_km_s=100.*jnp.exp(log_sigma),
                radial_geometry=radial)
            return factors

        factors = jax.jit(batched)(0.)
        derivative = jax.jit(jax.grad(lambda x: jnp.sum(batched(x))))(0.)
        self.assertTrue(np.isfinite(np.asarray(factors)).all())
        self.assertTrue(np.isfinite(float(derivative)))
        self.assertNotEqual(float(derivative), 0.)

        individual = []
        for i in range(2):
            one, _ = linked_singleton_logfactors_for_population(
                positions, velocities, intrinsic, angular,
                ids[i:i+1], active[i:i+1], association[i:i+1],
                voxels[i:i+1], radius[i:i+1], dz[i:i+1],
                eta_mean[i:i+1], eta_std[i:i+1], eta_alpha[i:i+1],
                population=0, sigma_los_km_s=100., radial_geometry=radial)
            individual.append(float(one[0]))
        np.testing.assert_allclose(np.asarray(factors), individual,
                                   rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
