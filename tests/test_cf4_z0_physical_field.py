import json
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cf4_z0_physical_field import PhysicalFieldModel, centres, read_centred, recenter_old_pm_fields, unit_mean_density
from cf4_datum_bearing_z0_phasec_pilot import _jax_tsc_deposit_one, tsc_deposit_numpy


class PhysicalFieldTests(unittest.TestCase):
    def test_density_positive_normalized_and_shift_invariant(self):
        g = jnp.asarray(np.random.default_rng(81).normal(size=(8, 8, 8)))
        rho = np.asarray(unit_mean_density(g))
        self.assertGreater(rho.min(), 0)
        self.assertAlmostEqual(rho.mean(), 1, places=12)
        np.testing.assert_allclose(rho, unit_mean_density(g + 900), rtol=2e-12)

    def test_centre_and_periodic_reads(self):
        grid = jnp.arange(64., dtype=jnp.float64).reshape(4, 4, 4)
        pos = centres(4, 16)
        np.testing.assert_allclose(read_centred(grid, pos, 16), grid, atol=1e-12)
        np.testing.assert_allclose(read_centred(grid, pos + 16, 16), grid, atol=1e-12)

    def test_recenter_mass_and_momentum(self):
        rng = np.random.default_rng(23)
        rho = np.exp(rng.normal(size=(8, 8, 8)))
        vel = rng.normal(size=(3, 8, 8, 8))
        new_rho, new_vel = recenter_old_pm_fields(rho, vel)
        self.assertGreater(new_rho.min(), 0)
        self.assertAlmostEqual(new_rho.sum(), rho.sum(), places=10)
        np.testing.assert_allclose((new_vel * new_rho).sum(axis=(1, 2, 3)), (vel * rho).sum(axis=(1, 2, 3)), rtol=1e-12)
        impulse = np.zeros((4, 4, 4)); impulse[1, 1, 1] = 1
        shifted, _ = recenter_old_pm_fields(1 + impulse, np.zeros((3, 4, 4, 4)))
        self.assertAlmostEqual(shifted[0, 1, 1] - 1, .25 * .75**2)

    def test_tsc_independent_reference_and_directional_gradient(self):
        rng = np.random.default_rng(32)
        mass = np.exp(rng.normal(size=(4, 4, 4)))
        positions = centres(4, 16) + rng.uniform(-1.2, 1.2, size=(4, 4, 4, 3))
        actual = _jax_tsc_deposit_one(jnp.asarray(mass), jnp.asarray(positions), 4, 4.)
        np.testing.assert_allclose(actual, tsc_deposit_numpy(mass, positions, 4, 16), rtol=1e-12)
        self.assertAlmostEqual(float(actual.sum()), mass.sum(), places=11)
        direction = rng.normal(size=positions.shape)
        weights = jnp.asarray(rng.normal(size=mass.shape))
        fn = lambda pos: jnp.sum(weights * _jax_tsc_deposit_one(jnp.asarray(mass), pos, 4, 4.))
        grad = float(jnp.sum(jax.grad(fn)(jnp.asarray(positions)) * direction))
        eps = 1e-5
        finite = float((fn(positions + eps * direction) - fn(positions - eps * direction)) / (2 * eps))
        self.assertAlmostEqual(grad, finite, places=7)

    def test_full_model_normalization_gradient_and_holdout_exclusion(self):
        root = Path(__file__).resolve().parents[1]
        settings = json.loads((root / "config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())["inference_model"]
        rng = np.random.default_rng(5)
        positions = centres(4, 48).reshape(-1, 3)[::4]
        relative = positions - 24
        design = {"pos": positions, "rhat": relative / np.linalg.norm(relative, axis=1)[:, None],
                  "B": np.zeros((16, 4)), "q_std": np.ones(4), "variance": np.full(16, 10000.),
                  "holdout": np.arange(16) % 4 == 0}
        transfer = np.ones((4, 4, 4)) * .3
        transfer[0, 0, 0] = 0
        model = PhysicalFieldModel(transfer, .5, 48., np.ones((6, 4, 4, 4)), design,
                                   np.arange(1, 7), np.ones(6), settings)
        x = rng.normal(size=model.size) * .1
        x[64:] = 0
        intensity, radial = jax.jit(model.forward)(jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(intensity).sum(axis=(1, 2, 3)), 64 * np.arange(1, 7), rtol=1e-12)
        counts = jnp.asarray(rng.poisson(.8 * np.asarray(intensity)))
        data = np.asarray(radial).copy()
        fn = jax.jit(lambda vector: model.nlp(vector, counts, jnp.asarray(data)))
        direction = rng.normal(size=model.size)
        direction /= np.linalg.norm(direction)
        gradient = float(jnp.dot(jax.grad(fn)(jnp.asarray(x)), direction))
        eps = 1e-5
        finite = float((fn(x + eps * direction) - fn(x - eps * direction)) / (2 * eps))
        self.assertAlmostEqual(gradient, finite, places=6)
        changed = data.copy(); changed[design["holdout"]] += 1e6
        np.testing.assert_allclose(model.nlp(jnp.asarray(x), counts, changed), model.nlp(jnp.asarray(x), counts, data), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
