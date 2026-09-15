import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import aperture_moments, particle_grid, weighted_periodic_force
from cf4_z0_pm_bridge import enable_pmwd_type_description_compatibility


class ParticleEntryTest(unittest.TestCase):
    def setUp(self):
        enable_pmwd_type_description_compatibility()
        from pmwd import Configuration, SimpleLCDM
        self.conf = Configuration(ptcl_spacing=1., ptcl_grid_shape=(4,)*3,
                                  mesh_shape=2, float_dtype=jnp.float64)
        self.cosmo = SimpleLCDM(self.conf)
        rng = np.random.default_rng(917)
        self.x = jnp.asarray(rng.uniform(0.1, 3.9, (64, 3)))
        self.v = jnp.asarray(rng.normal(size=(64, 3)) * 100)
        self.m = jnp.ones(64) * 3

    def test_aperture_numpy_and_boost(self):
        centers = jnp.asarray([[.2, .4, .6], [3.8, 2.8, .1]])
        out = aperture_moments(self.x, self.v, self.m, centers, .7, 4.)
        dx = (np.asarray(self.x)[None] - np.asarray(centers)[:, None] + 2) % 4 - 2
        w = np.asarray(self.m)[None] * np.exp(-.5*np.sum((dx/.7)**2, axis=-1))
        u = w @ np.asarray(self.v) / w.sum(axis=1)[:, None]
        var = np.sum(w[..., None]*(np.asarray(self.v)[None]-u[:, None])**2, axis=1)/w.sum(axis=1)[:, None]
        np.testing.assert_allclose(out['mass_Msun_h'], w.sum(axis=1), rtol=1e-12)
        np.testing.assert_allclose(out['variance_km2_s2'], var, rtol=1e-12)
        boosted = aperture_moments(self.x, self.v+1000, self.m, centers, .7, 4.)
        np.testing.assert_allclose(boosted['mean_velocity_km_s'], u+1000, atol=1e-10)
        np.testing.assert_allclose(boosted['variance_km2_s2'], var, rtol=1e-12)

    def test_grid_integrals_fine_mesh(self):
        out = particle_grid(self.x, self.v, self.m, self.conf)
        np.testing.assert_allclose(out['mass'].sum(), self.m.sum(), rtol=1e-12)
        np.testing.assert_allclose(out['momentum'].sum(axis=(0, 1, 2)), (self.m[:, None]*self.v).sum(axis=0), atol=1e-9)
        np.testing.assert_allclose(out['second_moment'].sum(axis=(0, 1, 2)), (self.m[:, None]*self.v**2).sum(axis=0), rtol=1e-12)
        self.assertAlmostEqual(float(out['rho'].mean()), 1., places=12)

    def test_weighted_force_reference_and_mass_split(self):
        from pmwd.gravity import gravity
        from pmwd.particles import Particles
        force = weighted_periodic_force(self.x, self.m, self.conf, self.cosmo.Omega_m)
        old = gravity(1., Particles.from_pos(self.conf, self.x), self.cosmo, self.conf)
        np.testing.assert_allclose(force, old, atol=1e-12)
        split = weighted_periodic_force(jnp.repeat(self.x, 2, axis=0),
                    jnp.tile(jnp.array([1., 2.]), 64), self.conf, self.cosmo.Omega_m)
        np.testing.assert_allclose(split, jnp.repeat(force, 2, axis=0), atol=1e-12)
        long = weighted_periodic_force(self.x, self.m, self.conf, self.cosmo.Omega_m, split_radius=.6, part='long')
        short = weighted_periodic_force(self.x, self.m, self.conf, self.cosmo.Omega_m, split_radius=.6, part='short')
        np.testing.assert_allclose(long+short, force, atol=1e-12)

    def test_force_position_gradient(self):
        direction = jnp.cos(jnp.arange(self.x.size)).reshape(self.x.shape)
        direction /= jnp.linalg.norm(direction)
        mass = jnp.where(jnp.arange(64)%2, 2., 1.)
        def value(x):
            acc = weighted_periodic_force(x, mass, self.conf, self.cosmo.Omega_m)
            return jnp.sum(acc*self.v)
        automatic = jnp.sum(jax.grad(value)(self.x)*direction)
        eps = 1e-5
        finite = (value(self.x+eps*direction)-value(self.x-eps*direction))/(2*eps)
        np.testing.assert_allclose(automatic, finite, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
