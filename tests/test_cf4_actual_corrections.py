import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_actual_selection import magnitude_h, three_way_split, mark_counts
from cf4_datum_bearing_z0_twompp_datum_builder_v1 import hash_holdout_mask
from cf4_twompp_joint_information_budget_pilot_v1 import schechter_fraction
from cf4_actual_corrected_model import SurvivalFieldModel, RadialCoordinates
from cf4_z0_physical_field import centres


class CorrectedInputsTest(unittest.TestCase):
    def test_h_magnitude_and_selection_equivalence(self):
        h = .6711
        np.testing.assert_allclose(magnitude_h([-23.], h), [-23.-5*np.log10(h)])
        distance_mpc = np.array([20., 40., 80., 120.])
        # Independent physical-unit calculation: transform BOTH edges and Mstar.
        offset = 5*np.log10(h)
        physical = schechter_fraction(distance_mpc, None, 11.5,
            -25+offset, -23.6666666666667+offset, -23.28+offset, -.94)
        h_convention = schechter_fraction(distance_mpc*h, None, 11.5,
            -25, -23.6666666666667, -23.28, -.94)
        np.testing.assert_allclose(physical, h_convention, rtol=1e-12, atol=1e-12)
        wrong = schechter_fraction(distance_mpc, None, 11.5, -25, -23.6666666666667, -23.28, -.94)
        self.assertGreater(np.max(np.abs(wrong-h_convention)), .1)
        with self.assertRaises(ValueError):
            magnitude_h([-23], 0)

    def test_calibration_disjoint_and_old_holdout_preserved(self):
        recnos = np.arange(1000)
        train, calibration, held = three_way_split(recnos, "unit-test")
        np.testing.assert_array_equal(held, hash_holdout_mask(recnos, "unit-test", 1, 5))
        np.testing.assert_array_equal(train.astype(int)+calibration+held, np.ones(1000))
        pop, shell, survives = recnos%6, (recnos//6)%6, recnos%3 == 0
        first = mark_counts(pop, shell, calibration, survives)
        changed = survives.copy(); changed[~calibration] = ~changed[~calibration]
        second = mark_counts(pop, shell, calibration, changed)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(sum(a.sum() for a in first), calibration.sum())


class CorrectedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n, box = 4, 48.
        rng = np.random.default_rng(871)
        pos = centres(n,box,.25).reshape(-1,3)[::8]
        rel = pos-box/2
        design = dict(pos=pos, rhat=rel/np.linalg.norm(rel,axis=1)[:,None],
            B=rng.normal(size=(8,4)), q_std=np.array([3.,4.,5.,2.]),
            variance=np.ones(8)*10, holdout=np.arange(8)%4 == 0)
        settings = json.loads((Path(__file__).resolve().parents[1]/
            "config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())["inference_model"]
        shape = (n,)*3
        amp = np.ones(shape)*.2; amp[0,0,0] = 0
        cov = dict(log_density_amplitude=amp, coupling=np.ones(shape)*2,
            velocity_longitudinal_amplitude=np.ones(shape)*3,
            velocity_transverse_amplitude=np.ones(shape)*2)
        response = np.ones((6,)+shape)*.8
        shells = np.repeat(response[:,None]/6, 6, axis=1)
        cls.model = SurvivalFieldModel(np.ones(shape), .5, box, response, design,
            np.ones(6), np.ones(6), settings, covariance=cov, origin_fraction=.25,
            selection_shells=shells, survival_yes=np.ones((6,6))*7,
            survival_no=np.ones((6,6))*3)
        cls.x = jnp.asarray(rng.normal(size=cls.model.size)*.15)
        cls.counts = jnp.ones((6,)+shape)
        cls.radial = jnp.asarray(rng.normal(size=8)*5)

    def test_survival_density_and_gradient(self):
        m, x = self.model, self.x
        fn = jax.jit(lambda v: m.nlp(v, self.counts, self.radial))
        grad = np.asarray(jax.grad(fn)(x))
        index = m.field_size+27
        eps = jnp.zeros(m.size).at[index].set(1e-5)
        finite = float((fn(x+eps)-fn(x-eps))/(2e-5))
        self.assertAlmostEqual(finite, grad[index], places=5)
        _, rho, vel = m.fields(x)
        from cf4_pm_calibrated_z0 import PMCalibratedFieldModel
        before, _ = PMCalibratedFieldModel.observe(m, rho, vel, x[m.field_size:])
        after, _ = m.observe(rho, vel, x[m.field_size:])
        probability = jax.nn.sigmoid(m.survival_logits(x[m.field_size:])).mean(axis=1)
        np.testing.assert_allclose(after, before*probability[:,None,None,None], rtol=1e-12)
        self.assertEqual(m.count_train_fraction, .6)

    def test_exact_field_conditional_coordinates(self):
        m, x = self.model, self.x
        coordinates = RadialCoordinates(m, self.radial)
        fn = jax.jit(lambda v: coordinates.nlp(v, self.counts, self.radial))
        physical, _ = coordinates.transform(x)
        np.testing.assert_allclose(fn(x), m.nlp(physical, self.counts, self.radial), rtol=1e-12)
        start = coordinates.start
        epsilon = x[start:start+4]
        y = x.at[start:start+4].set(jnp.array([.6,-.3,.2,.1]))
        expected = .5*(jnp.sum(y[start:start+4]**2)-jnp.sum(epsilon**2))
        np.testing.assert_allclose(fn(y)-fn(x), expected, atol=1e-9, rtol=1e-8)
        gradient = jax.grad(fn)(x)
        np.testing.assert_allclose(gradient[start:start+4], epsilon, atol=1e-9)
        # Field derivative includes d mu(field)/d field, not stopped-gradient conditioning.
        delta = jnp.zeros(m.size).at[70].set(1e-5)
        self.assertAlmostEqual(float((fn(x+delta)-fn(x-delta))/(2e-5)), float(gradient[70]), places=5)
        altered = self.radial.at[~m.train].add(1e6)
        other = RadialCoordinates(m, altered)
        np.testing.assert_allclose(coordinates.transform(x)[0], other.transform(x)[0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
