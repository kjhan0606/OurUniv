import unittest
from importlib.machinery import PathFinder
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_dynamics, periodic_delta
from cf4_r1_particle_resolution import refine_lpt_state, make_state_evolution


class ParticleResolutionTest(unittest.TestCase):
    def test_cli_search_order_resolves_library_not_driver(self):
        root = Path(__file__).resolve().parents[1]
        spec = PathFinder.find_spec('cf4_r1_particle_resolution',
                                    [str(root/'scripts'), str(root/'src')])
        self.assertIsNotNone(spec)
        self.assertEqual(Path(spec.origin), root/'src/cf4_r1_particle_resolution.py')

    def test_interpolation_constant_wave_nyquist_and_nodes(self):
        def field(n):
            q = np.indices((n,)*3)/n
            return np.stack((np.ones((n,)*3)*2,
                np.sin(2*np.pi*(q[0]+q[1])), np.cos(2*np.pi*4*q[2])), axis=-1)
        coarse = field(8)
        displacement, velocity = refine_lpt_state(coarse, 100*coarse, 32)
        np.testing.assert_allclose(displacement, field(32), atol=2e-14)
        np.testing.assert_allclose(velocity, 100*field(32), atol=2e-12)
        np.testing.assert_allclose(displacement[::4, ::4, ::4], coarse, atol=1e-14)
        np.testing.assert_allclose(displacement.mean(axis=(0, 1, 2)), coarse.mean(axis=(0, 1, 2)), atol=1e-14)

    def test_supplied_state_matches_pmwd_and_total_mass(self):
        settings = dict(n=4, box_cMpc_h=12., a_start=.015625, a_stop=.125,
                        a_nbody_maxstep=.03125,
                        cosmology=dict(Om=.31, Ob=.05, h=.746, A_s_1e9=1.63, ns=.96))
        evolve, initial, _, _, mass = make_dynamics(settings, mesh_ratio=2)
        white = jnp.asarray(np.random.default_rng(931).normal(size=4**3))
        x, v = initial(white)
        q = np.indices((4,)*3).reshape(3, -1).T*3.
        displacement = np.asarray(x)-q
        external, _, _, other_mass = make_state_evolution(settings, mesh_ratio=2)
        actual_x, actual_v = external(jnp.asarray(displacement), v)
        expected_x, expected_v = evolve(white)
        np.testing.assert_allclose(periodic_delta(actual_x, expected_x, 12.), 0., atol=1e-10)
        np.testing.assert_allclose(actual_v, expected_v, atol=1e-10)
        self.assertEqual(mass, other_mass)
        _, _, _, refined_mass = make_state_evolution({**settings, 'n': 8})
        self.assertAlmostEqual(refined_mass*8**3/(mass*4**3), 1., places=14)


if __name__ == '__main__':
    unittest.main()
