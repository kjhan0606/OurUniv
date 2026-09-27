import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_marked_tracer_jax import (
    predict_source_marked_intensity, source_mark_transfer,
)
from cf4_r2_observed_magnitude_transfer import (
    observed_magnitude_transfer, twompp_k_correction_delta,
)


class MarkedTracerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64', True)

    def test_jax_transfer_matches_numpy_reference_with_k_shift(self):
        mu_r = np.array([32.4, 33.1])
        mu_s = np.array([32.6, 32.9])
        z_r = np.array([.010, .015])
        z_s = np.array([.011, .014])
        actual = np.asarray(source_mark_transfer(
            jnp.asarray(mu_r), jnp.asarray(mu_s),
            jnp.asarray(z_r), jnp.asarray(z_s)))
        expected = observed_magnitude_transfer(
            mu_r, mu_s, correction_shift_mag=twompp_k_correction_delta(
                z_r, z_s))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)

    def test_deposited_mass_uses_source_selection_and_velocity_gradient(self):
        box = 96.
        observer = jnp.array([48., 48., 48.])
        positions = jnp.array([[78., 48., 48.], [48., 88., 48.]])
        velocities = jnp.array([[300., 0., 0.], [0., 0., 0.]])
        intrinsic = jnp.array([[.3, .2], [.4, .5], [.6, .7],
                               [.8, .9], [1.2, .3]])
        angular = jnp.array([[1., .8], [.5, .4]])
        radii = jnp.linspace(.001, 50., 5001)
        moduli = 5*jnp.log10(radii)+25.
        redshifts = radii/3000.
        args = dict(observer=observer, box_size_cMpc_h=box,
                    hubble_km_s_Mpc=75., little_h=.75,
                    radius_table_cMpc_h=radii, modulus_table_h=moduli,
                    redshift_table=redshifts, grid_size=4,
                    radial_min_cMpc_h=5., radial_max_cMpc_h=45.)
        intensity = np.asarray(predict_source_marked_intensity(
            positions, velocities, intrinsic, angular, **args))
        self.assertEqual(intensity.shape, (6, 4, 4, 4))
        self.assertTrue(np.isfinite(intensity).all())
        self.assertGreaterEqual(float(intensity.min()), -1e-13)

        true_r = np.array([30., 40.])
        observed_r = np.array([33., 40.])
        mu_r = np.interp(true_r, np.asarray(radii), np.asarray(moduli))
        mu_s = np.interp(observed_r, np.asarray(radii), np.asarray(moduli))
        transfer = observed_magnitude_transfer(
            mu_r, mu_s, correction_shift_mag=twompp_k_correction_delta(
                true_r/3000., observed_r/3000.))
        expected = np.sum(transfer*np.asarray(intrinsic)[None, :, :], axis=1)
        expected *= np.asarray(angular)[(np.arange(6)//3), :]
        np.testing.assert_allclose(intensity.sum(axis=(1, 2, 3)),
                                   expected.sum(axis=1), rtol=2e-10, atol=2e-11)

        def selected_bright(v):
            trial = velocities.at[0, 0].set(v)
            return predict_source_marked_intensity(
                positions, trial, intrinsic, angular, **args)[0].sum()

        derivative = float(jax.grad(selected_bright)(300.))
        step = .1
        finite_difference = float((selected_bright(300.+step)
                                   -selected_bright(300.-step))/(2*step))
        self.assertTrue(np.isfinite(derivative))
        self.assertGreater(abs(derivative), 1e-8)
        self.assertAlmostEqual(derivative, finite_difference, delta=2e-5)


if __name__ == '__main__':
    unittest.main()
