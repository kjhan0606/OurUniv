import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_marked_tracer_jax import (
    conditional_single_link_logfactor,
    intrinsic_biased_source_masses, intrinsic_lf_bin_fractions,
    intrinsic_lf_reference_weights,
    predict_source_marked_intensity, predict_source_marked_key_contributions,
    predict_source_marked_radial_key_density,
    source_mark_transfer, tsc_weight_at_voxel,
    sparse_marked_poisson_log_likelihood,
)
from cf4_2mpp_joint_likelihood_jax import tsc_deposit_jax
from cf4_2mpp_joint_likelihood_jax import _gaussian_hermite_rule
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

    def test_intrinsic_rate_and_bias_are_distinct_mass_coordinates(self):
        density = jnp.array([0., .5, 1.5])
        bias = jnp.array([.6, .8, 1., 1.2, 1.4])
        log_rate = jnp.log(2.)
        source = intrinsic_biased_source_masses(density,log_rate,bias)
        fraction = intrinsic_lf_bin_fractions()
        np.testing.assert_allclose(np.asarray(source).sum(axis=1),
                                   6.*np.asarray(fraction),rtol=1e-13)
        np.testing.assert_array_equal(np.asarray(source)[:,0],0.)
        derivative = jax.grad(lambda r: intrinsic_biased_source_masses(
            density,r,bias).sum())(log_rate)
        self.assertAlmostEqual(float(derivative),6.,places=11)

    def test_reference_rate_preserves_physical_intensity_at_matched_rate(self):
        density = jnp.array([.5, 1., 1.5])
        bias = jnp.array([.6, .8, 1., 1.2, 1.4])
        for alpha in (-.94, -.99):
            all_fractions = intrinsic_lf_bin_fractions(alpha=alpha)
            reference_fraction = jnp.sum(all_fractions[1:4])
            weights = intrinsic_lf_reference_weights(alpha=alpha)
            np.testing.assert_allclose(np.asarray(weights[1:4]).sum(), 1.,
                                       rtol=1e-12)
            old = intrinsic_biased_source_masses(
                density, jnp.log(.8), bias, alpha=alpha)
            new = intrinsic_biased_source_masses(
                density, jnp.log(.8*reference_fraction), bias, alpha=alpha,
                reference_interval=(-25., -21.))
            np.testing.assert_allclose(np.asarray(new), np.asarray(old),
                                       rtol=2e-12, atol=1e-12)
        # At fixed *bright* rate, changing the faint LF slope must not move
        # the average intrinsic reference-bin count.
        for alpha in (-.94, -.99):
            masses = intrinsic_biased_source_masses(
                density, jnp.log(.1), bias, alpha=alpha,
                reference_interval=(-25., -21.))
            self.assertAlmostEqual(float(masses[1:4].sum()), .3, places=11)
        jit_mass = jax.jit(lambda a: intrinsic_biased_source_masses(
            density, jnp.log(.1), bias, alpha=a,
            reference_interval=(-25., -21.)))
        self.assertTrue(np.isfinite(np.asarray(jit_mass(-.94))).all())

    def test_sparse_marked_count_includes_all_empty_voxel_exposure(self):
        intensity = jnp.array([[.5,1.2],[2.,.7]])
        keys = jnp.array([1,2])
        counts = jnp.array([2,1])
        actual = sparse_marked_poisson_log_likelihood(intensity,keys,counts)
        expected = 2*np.log(1.2)+np.log(2.)-np.log(2.)-4.4
        self.assertAlmostEqual(float(actual),float(expected),places=12)
        absent = sparse_marked_poisson_log_likelihood(
            intensity.at[0,1].set(0.),keys,counts)
        self.assertEqual(float(absent),float('-inf'))

    def test_sky_window_integrates_only_selected_cells_and_rejects_crossing(self):
        intensity = jnp.array([[.5,1.2],[2.,.7]])
        train = jnp.array([True,False])
        keys = jnp.array([0,2])
        counts = jnp.array([2,1])
        actual = sparse_marked_poisson_log_likelihood(
            intensity,keys,counts,selected_voxel_mask=train)
        expected = 2*np.log(.5)+np.log(2.)-np.log(2.)-(.5+2.)
        self.assertAlmostEqual(float(actual),float(expected),places=12)
        wrong = sparse_marked_poisson_log_likelihood(
            intensity,jnp.array([1]),jnp.array([1]),selected_voxel_mask=train)
        self.assertEqual(float(wrong),float('-inf'))
        gradient = np.asarray(jax.grad(lambda x: sparse_marked_poisson_log_likelihood(
            x,keys,counts,selected_voxel_mask=train))(intensity))
        np.testing.assert_array_equal(gradient[:,1],0.)

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

        # Dynamic one-node accumulation is the bounded-memory equivalent of
        # the existing statically unrolled GH rule.
        from cf4_r2_marked_tracer_jax import predict_source_marked_intensity_los_node
        one_node = jax.jit(
            predict_source_marked_intensity_los_node,
            static_argnames=('grid_size','radial_min_cMpc_h',
                             'radial_max_cMpc_h'))
        nodes, weights = _gaussian_hermite_rule(3)
        accumulated = jnp.zeros_like(jnp.asarray(intensity))
        for node, weight in zip(nodes, weights):
            accumulated += one_node(positions, velocities, intrinsic, angular,
                                    node, weight, **args)
        np.testing.assert_allclose(np.asarray(accumulated), intensity,
                                   rtol=2e-12, atol=2e-12)

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

    def test_source_to_count_key_contributions_reproduce_same_deposit(self):
        box = 96.
        positions = jnp.array([[78., 48., 48.], [48., 88., 48.],
                               [95.5, 10., 10.]])
        velocity = jnp.array([[300., 0., 0.], [0., 0., 0.],
                              [-120., 0., 0.]])
        intrinsic = jnp.array([[.3, .2, .4], [.4, .5, .6], [.6, .7, .8],
                               [.8, .9, 1.], [1.2, .3, .2]])
        angular = jnp.array([[1., .8, .7], [.5, .4, .6]])
        radii = jnp.linspace(.001, 80., 8001)
        args = dict(observer=jnp.array([48., 48., 48.]),
                    box_size_cMpc_h=box, hubble_km_s_Mpc=75., little_h=.75,
                    radius_table_cMpc_h=radii,
                    modulus_table_h=5*jnp.log10(radii)+25.,
                    redshift_table=radii/3000., grid_size=4,
                    radial_min_cMpc_h=5., radial_max_cMpc_h=80.,
                    sigma_los_km_s=120.)
        intensity = predict_source_marked_intensity(
            positions, velocity, intrinsic, angular, **args)
        for population, ijk in ((0, (3, 2, 2)), (2, (0, 0, 0)),
                                (5, (2, 3, 2))):
            contrib = predict_source_marked_key_contributions(
                positions, velocity, intrinsic, angular, population, ijk,
                **args)
            self.assertEqual(contrib.shape, intrinsic.shape)
            self.assertGreaterEqual(float(jnp.min(contrib)), 0.)
            np.testing.assert_allclose(float(jnp.sum(contrib)),
                                       float(intensity[(population,)+ijk]),
                                       rtol=2e-12, atol=2e-12)
        voxel = (3, 2, 2)
        for positions_at in (positions, positions.at[2].set(jnp.array([.5, 10., 10.]))):
            weights = tsc_weight_at_voxel(positions_at, voxel, 4, box)
            for source in range(len(positions_at)):
                deposited = tsc_deposit_jax(
                    positions_at[source:source+1], jnp.ones(1), 4, box)
                np.testing.assert_allclose(float(weights[source]),
                                           float(deposited[voxel]),
                                           rtol=1e-13, atol=1e-13)
        def key_total(v):
            trial = velocity.at[0, 0].set(v)
            return jnp.sum(predict_source_marked_key_contributions(
                positions, trial, intrinsic, angular, 0, voxel, **args))
        def full_total(v):
            trial = velocity.at[0, 0].set(v)
            return predict_source_marked_intensity(
                positions, trial, intrinsic, angular, **args)[(0,)+voxel]
        self.assertAlmostEqual(float(jax.grad(key_total)(300.)),
                               float(jax.grad(full_total)(300.)), delta=2e-10)

    def test_continuous_radial_key_density_and_normalized_mark_factor(self):
        radii = jnp.linspace(.001, 50., 5001)
        positions = jnp.array([[78., 48., 48.]])
        velocity = jnp.zeros((1, 3))
        intrinsic = jnp.array([[.1], [.2], [.3], [.4], [.5]])
        angular = jnp.ones((2, 1))
        args = dict(observer=jnp.array([48., 48., 48.]),
                    box_size_cMpc_h=96., hubble_km_s_Mpc=75., little_h=.75,
                    radius_table_cMpc_h=radii,
                    modulus_table_h=5*jnp.log10(radii)+25.,
                    redshift_table=radii/3000., grid_size=4,
                    radial_min_cMpc_h=5., radial_max_cMpc_h=45.,
                    sigma_los_km_s=120.)
        key = (3, 2, 2)
        gh_count = float(jnp.sum(predict_source_marked_key_contributions(
            positions, velocity, intrinsic, angular, 0, key, **args)))
        self.assertGreater(gh_count, 0.)
        observed_r = jnp.linspace(23., 37., 113)
        radial_density = jax.vmap(lambda radius: jnp.sum(
            predict_source_marked_radial_key_density(
                positions, velocity, intrinsic, angular, 0, key, radius,
                **args)))(observed_r)
        ys, xs = np.asarray(radial_density), np.asarray(observed_r)
        integral = float(np.sum(.5*(ys[1:]+ys[:-1])*np.diff(xs)))
        self.assertAlmostEqual(integral, gh_count, delta=.02*gh_count)
        self.assertGreater(float(radial_density[56]), 0.)

        mass = jnp.array([[.1, .3], [.2, .1], [.4, .2],
                          [.1, .6], [.2, .4]])
        log_association = jnp.log(jnp.array([[.8, .7], [.6, .5], [.9, .8],
                                             [.7, .8], [.5, .6]]))
        log_mark = jnp.log(jnp.array([[1.1, .8], [1.3, .7], [1.5, .6],
                                      [1.2, .9], [.9, 1.4]]))
        weights = np.asarray(mass*jnp.exp(log_association))
        expected = np.log(np.sum(weights*np.exp(np.asarray(log_mark)))
                          /np.sum(weights))
        score = conditional_single_link_logfactor(mass,log_association,log_mark)
        self.assertAlmostEqual(float(score),float(expected),places=13)
        self.assertAlmostEqual(float(conditional_single_link_logfactor(
            10.*mass,log_association,log_mark)),float(expected),places=13)
        self.assertAlmostEqual(float(conditional_single_link_logfactor(
            mass,log_association,jnp.zeros_like(mass))),0.,places=13)
        self.assertEqual(float(conditional_single_link_logfactor(
            jnp.zeros_like(mass),log_association,log_mark)),float('-inf'))

        # A broad LOS kernel can cross the observer. Both signed-radius
        # branches then contribute to the same observed radial density.
        near_radii = jnp.linspace(.001, 12., 1201)
        near_args = dict(observer=jnp.array([12.,12.,12.]),
                         box_size_cMpc_h=24.,hubble_km_s_Mpc=75.,little_h=.75,
                         radius_table_cMpc_h=near_radii,
                         modulus_table_h=5*jnp.log10(near_radii)+25.,
                         redshift_table=near_radii/3000.,grid_size=4,
                         radial_min_cMpc_h=.1,radial_max_cMpc_h=10.,
                         sigma_los_km_s=300.)
        source = jnp.array([[13.,12.,12.]])
        true_mass = jnp.array([[0.],[1.],[0.],[0.],[0.]])
        opposite_key = (1,2,2)
        observed = predict_source_marked_radial_key_density(
            source,jnp.zeros((1,3)),true_mass,jnp.ones((2,1)),
            0,opposite_key,1.,**near_args)
        sigma_r=3.
        pdf_plus=np.exp(-.5*0.**2)/(np.sqrt(2*np.pi)*sigma_r)
        pdf_minus=np.exp(-.5*(2./sigma_r)**2)/(np.sqrt(2*np.pi)*sigma_r)
        plus=float(tsc_weight_at_voxel(source,opposite_key,4,24.)[0])
        minus=float(tsc_weight_at_voxel(
            jnp.array([[11.,12.,12.]]),opposite_key,4,24.)[0])
        self.assertGreater(pdf_minus*minus,0.)
        mu=float(5*np.log10(1.)+25.)
        transfer=float(source_mark_transfer(
            jnp.array([mu]),jnp.array([mu]),
            jnp.array([1./3000.]),jnp.array([1./3000.]))[0,1,0])
        self.assertAlmostEqual(float(observed.sum()),
                               transfer*(pdf_plus*plus+pdf_minus*minus),
                               delta=1e-12)

    def test_lf_shape_is_shared_by_intrinsic_mass_and_transfer(self):
        fractions = intrinsic_lf_bin_fractions(mstar=-23.17, alpha=-.73)
        np.testing.assert_allclose(np.asarray(fractions).sum(), 1., rtol=1e-13)
        mass = jnp.asarray(fractions)[:, None]
        position = jnp.array([[15., 12., 12.]])
        radii = jnp.linspace(.001, 30., 3001)
        kwargs = dict(observer=jnp.array([12., 12., 12.]),
                      box_size_cMpc_h=24., hubble_km_s_Mpc=75., little_h=.75,
                      radius_table_cMpc_h=radii,
                      modulus_table_h=5*jnp.log10(radii)+25.,
                      redshift_table=radii/3000., grid_size=4,
                      radial_min_cMpc_h=1., radial_max_cMpc_h=10.,
                      mstar=-23.17, alpha=-.73)
        def selected(mstar):
            true_mass = intrinsic_lf_bin_fractions(mstar=mstar, alpha=-.73)[:,None]
            return predict_source_marked_intensity(
                position, jnp.zeros((1,3)), true_mass, jnp.ones((2,1)),
                **{**kwargs, 'mstar':mstar}).sum()
        value = selected(-23.17)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.isfinite(float(jax.grad(selected)(-23.17))))
        def selected_alpha(alpha):
            true_mass = intrinsic_lf_bin_fractions(mstar=-23.17, alpha=alpha)[:,None]
            return predict_source_marked_intensity(
                position, jnp.zeros((1,3)), true_mass, jnp.ones((2,1)),
                **{**kwargs, 'alpha':alpha}).sum()
        alpha_gradient = float(jax.grad(selected_alpha)(-.73))
        finite_difference = float((selected_alpha(-.7299)
                                   -selected_alpha(-.7301))/.0002)
        self.assertTrue(np.isfinite(alpha_gradient))
        self.assertAlmostEqual(alpha_gradient, finite_difference, delta=2e-5)
        trial = predict_source_marked_intensity(
            position, jnp.zeros((1,3)), mass, jnp.ones((2,1)), **kwargs)
        np.testing.assert_allclose(np.asarray(trial).sum(), float(value), rtol=1e-12)
        # The imported low-z shape has alpha+1=0.06 and is the numerically
        # more demanding current development reference.
        def imported_alpha(alpha):
            true_mass = intrinsic_lf_bin_fractions(mstar=-23.28, alpha=alpha)[:,None]
            return predict_source_marked_intensity(
                position, jnp.zeros((1,3)), true_mass, jnp.ones((2,1)),
                **{**kwargs, 'mstar':-23.28, 'alpha':alpha}).sum()
        imported_gradient = float(jax.grad(imported_alpha)(-.94))
        imported_finite = float((imported_alpha(-.9399)
                                 -imported_alpha(-.9401))/.0002)
        self.assertTrue(np.isfinite(imported_gradient))
        self.assertAlmostEqual(imported_gradient, imported_finite, delta=2e-5)


if __name__ == '__main__':
    unittest.main()
