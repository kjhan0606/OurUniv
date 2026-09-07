import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_pm_calibrated_z0 import PMCalibratedFieldModel, fit_covariance
from cf4_quantile_z0 import QuantileFieldModel, fit_quantile_map, gaussianize, quantile_jax, quantile_numpy
from cf4_z0_physical_field import centres

ROOT = Path(__file__).resolve().parents[1]


class QuantilePriorTest(unittest.TestCase):
    def test_identity_map_and_linear_tails(self):
        x = np.linspace(-3.5, 3.5, 65)
        mapping = dict(x=x, y=2*x+.3, slopes=np.full_like(x, 2.))
        z = np.linspace(-6, 6, 1000)
        np.testing.assert_allclose(quantile_numpy(z, mapping), 2*z+.3, atol=1e-12)
        np.testing.assert_allclose(quantile_jax(z, mapping), 2*z+.3, atol=1e-12)
        np.testing.assert_allclose(gaussianize(2*z+.3, mapping), z, atol=1e-11)

    def test_training_map_monotonic_inverse_and_endpoint_gradients(self):
        rng = np.random.default_rng(71)
        z = rng.normal(size=(16,16,16))
        mapping = fit_quantile_map([np.exp(.4*z+.04*z*z)])
        probe = np.unique(np.concatenate((np.linspace(-6, 6, 4000), mapping["x"])))
        y = quantile_numpy(probe, mapping)
        self.assertTrue(np.all(np.diff(y) > 0))
        np.testing.assert_allclose(quantile_jax(probe, mapping), y, atol=1e-12)
        np.testing.assert_allclose(gaussianize(y, mapping), probe, atol=1e-10)
        derivative = jax.jit(jax.vmap(jax.grad(lambda t: quantile_jax(t, mapping))))
        np.testing.assert_allclose(derivative(jnp.asarray(mapping["x"])), mapping["slopes"], atol=1e-10)
        with self.assertRaises(ValueError):
            fit_quantile_map([np.ones((4,4,4))])

    def test_transformed_covariance_full_likelihood_gradient(self):
        rng = np.random.default_rng(72)
        n, box = 8, 96.
        fields = [(np.exp(.4*rng.normal(size=(n,n,n))), 100*rng.normal(size=(3,n,n,n))) for _ in range(4)]
        mapping = fit_quantile_map([r for r, _ in fields])
        coords = [gaussianize(np.log(r)-np.log(r).mean(), mapping) for r, _ in fields]
        covariance, _ = fit_covariance(fields, box, 4, density_coordinates=coords)
        pos = centres(n, box, .25).reshape(-1,3)[::64]
        rel = pos-box/2
        design = dict(pos=pos, rhat=rel/np.linalg.norm(rel,axis=1)[:,None], B=np.zeros((8,4)),
                      q_std=np.ones(4), variance=np.ones(8)*10000, holdout=np.arange(8)%4==0)
        settings = json.loads((ROOT / "config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())["inference_model"]
        source = PMCalibratedFieldModel(np.ones((n,n,n)), .5, box, np.ones((6,n,n,n)), design,
            np.ones(6), np.ones(6), settings, covariance=covariance, origin_fraction=.25, tracer_curvature_sigma=.2)
        model = QuantileFieldModel(source, covariance, mapping)
        x = rng.normal(size=model.size)*.3
        _, rho, velocity = jax.jit(model.fields)(jnp.asarray(x))
        self.assertGreater(float(rho.min()), 0)
        self.assertAlmostEqual(float(rho.mean()), 1, places=12)
        np.testing.assert_allclose(source.fields(jnp.asarray(x))[2], velocity, atol=1e-10)
        counts, radial = jnp.ones((6,n,n,n)), jnp.zeros(8)
        fn = jax.jit(lambda a: model.nlp(a, counts, radial))
        direction = rng.normal(size=model.size); direction /= np.linalg.norm(direction)
        grad = float(jnp.dot(jax.grad(fn)(jnp.asarray(x)), direction))
        eps = 1e-5
        finite = float((fn(x+eps*direction)-fn(x-eps*direction))/(2*eps))
        np.testing.assert_allclose(grad, finite, rtol=1e-5, atol=1e-5)
        changed = np.zeros(8); changed[design["holdout"]] = 1e6
        np.testing.assert_allclose(model.nlp(jnp.asarray(x), counts, jnp.asarray(changed)), fn(x), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
