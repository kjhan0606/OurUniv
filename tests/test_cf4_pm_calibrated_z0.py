import json
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_pm_calibrated_z0 import fit_covariance, PMCalibratedFieldModel, wave_geometry
from cf4_z0_physical_field import centres, read_centred


class NativePriorTest(unittest.TestCase):
    def test_native_coordinates_without_resampling(self):
        grid = jnp.arange(64., dtype=jnp.float64).reshape(4,4,4)
        pos = centres(4,48,.25)
        np.testing.assert_allclose(read_centred(grid, pos, 48, .25), grid, atol=1e-12)
        np.testing.assert_allclose(pos[0,0,0], [3,3,3])

    def test_covariance_gradients_and_unfixed_velocity(self):
        rng = np.random.default_rng(67)
        fields = [(np.exp(.4*rng.normal(size=(8,8,8))), rng.normal(size=(3,8,8,8))*100) for _ in range(4)]
        covariance, _ = fit_covariance(fields, 96., bins=4)
        for name, value in covariance.items():
            self.assertTrue(np.isfinite(value).all())
            if name != "coupling": self.assertTrue(np.all(value >= 0))
        n = 8
        pos = centres(n,96,.25).reshape(-1,3)[::64]
        rel = pos - 48
        design = dict(pos=pos, rhat=rel/np.linalg.norm(rel,axis=1)[:,None], B=np.zeros((8,4)),
                      q_std=np.ones(4), variance=np.ones(8)*10000, holdout=np.arange(8)%4==0)
        settings = json.loads((ROOT / "config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())["inference_model"]
        model = PMCalibratedFieldModel(np.ones((n,n,n)), .5, 96., np.ones((6,n,n,n)), design,
                                      np.ones(6), np.ones(6), settings, covariance=covariance, origin_fraction=.25)
        x = rng.normal(size=model.size)*.2
        field_fn = jax.jit(model.fields)
        _, rho, velocity = field_fn(jnp.asarray(x))
        self.assertAlmostEqual(float(rho.mean()), 1, places=12)
        y=x.copy(); y[n**3:4*n**3] += rng.normal(size=3*n**3)
        _, rho2, velocity2 = field_fn(jnp.asarray(y))
        np.testing.assert_array_equal(rho, rho2)
        self.assertGreater(float(jnp.std(velocity2-velocity)), 1)
        # The Fourier construction must remain Hermitian (including Nyquist).
        _, direction, _ = wave_geometry(n,96)
        mode = np.fft.fftn(x[:4*n**3].reshape(4,n,n,n), axes=(1,2,3), norm="ortho")
        gk=mode[0]*covariance["log_density_amplitude"]
        longitudinal=direction*np.sum(direction*mode[1:],axis=0)
        vk=1j*direction*covariance["coupling"]*gk + covariance["velocity_longitudinal_amplitude"]*longitudinal + covariance["velocity_transverse_amplitude"]*(mode[1:]-longitudinal)
        self.assertLess(np.max(abs(np.fft.ifftn(vk,axes=(1,2,3),norm="ortho").imag)), 1e-10)
        fn = jax.jit(lambda a: model.nlp(a, jnp.ones((6,n,n,n)), jnp.zeros(8)))
        direction_x = rng.normal(size=model.size); direction_x /= np.linalg.norm(direction_x)
        grad = float(jnp.dot(jax.grad(fn)(jnp.asarray(x)), direction_x))
        eps=1e-5
        finite=float((fn(x+eps*direction_x)-fn(x-eps*direction_x))/(2*eps))
        self.assertAlmostEqual(grad,finite,places=5)


if __name__ == "__main__": unittest.main()
