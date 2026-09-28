import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_linked_singleton_jax import (
    linked_singleton_logfactors_for_population, cached_eta_mixture_logfactors)


class LinkedSingletonBatchTests(unittest.TestCase):
    def test_saved_endpoint_supports_both_optimizer_records(self):
        from cf4_r2_fp_restart_check import saved_fp_endpoint
        self.assertEqual(saved_fp_endpoint({'trace':[{'parts':[1.,-2.,3.]}]}),-2.)
        self.assertEqual(saved_fp_endpoint({'trace':[],
            'full_gradient_after_block':{'parts':[1.,-4.,3.]}}),-4.)
        with self.assertRaises(ValueError):
            saved_fp_endpoint({'trace':[]})

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

        def readout(zero):
            return linked_singleton_logfactors_for_population(
                positions,velocities,intrinsic,angular,ids,active,association,
                voxels,radius,dz,eta_mean,eta_std,eta_alpha,population=0,
                sigma_los_km_s=100.,radial_geometry=radial,fp_zero_dex=zero,
                return_eta_moments=True,return_eta_mixture=True)
        moments=jax.jit(readout)(0.)
        shifted=jax.jit(readout)(.04)
        np.testing.assert_allclose(np.asarray(moments[0]),np.asarray(factors),atol=1e-12)
        self.assertTrue(np.isfinite(np.asarray(moments[2:4])).all())
        self.assertTrue((np.asarray(moments[3])>=0.).all())
        # Count-conditioned distances must not be reweighted by the observed
        # FP mark/zero point: otherwise this would condition twice on that mark.
        for left,right in zip(moments[2:],shifted[2:]):
            np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
        np.testing.assert_allclose(np.exp(np.asarray(moments[5])).sum(axis=1),1.,atol=1e-12)
        for zero, expected in ((0.,moments[0]),(.04,shifted[0])):
            cached=cached_eta_mixture_logfactors(moments[4],moments[5],eta_mean,
                                                eta_std,eta_alpha,zero)
            np.testing.assert_allclose(np.asarray(cached),np.asarray(expected),atol=1e-12)
        self.assertGreater(float(jnp.max(jnp.abs(moments[0]-shifted[0]))),1e-8)
        lower=np.log10(np.asarray(dz)/8.)
        upper=np.log10(np.asarray(dz)/5.)
        self.assertTrue((np.asarray(moments[2])>=lower-1e-12).all())
        self.assertTrue((np.asarray(moments[2])<=upper+1e-12).all())


if __name__ == '__main__':
    unittest.main()
