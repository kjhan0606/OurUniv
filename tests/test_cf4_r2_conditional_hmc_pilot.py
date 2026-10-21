"""Driver wiring regression; not an actual-data posterior test."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cf4_r2_conditional_hmc_pilot as pilot


class PilotWiringTests(unittest.TestCase):
    def test_mode_movement_and_nuisance_split(self):
        n = 4
        wave = np.cos(2*np.pi*np.arange(n)/n)
        cube = np.broadcast_to(wave[:,None,None], (n,)*3)
        q = np.concatenate([cube.ravel(), [3.,4.]])
        row = pilot.movement(q, np.zeros_like(q), n)
        self.assertAlmostEqual(row['fundamental_cosine'][0], np.linalg.norm(cube))
        np.testing.assert_allclose(row['fundamental_cosine'][1:], 0., atol=1e-14)
        self.assertEqual(row['canonical_nuisance_jump_l2'], 5.)

    def test_invalid_step_fails_before_oracle(self):
        for step in (0., -.1, .11, np.nan):
            with self.assertRaises(ValueError):
                pilot.run(None, None, None, {}, None, step=step)

    def test_rejected_states_are_retained_and_never_best_trial_selected(self):
        initial = np.arange(32, dtype=float) / 100
        calls = []
        def oracle(q):
            calls.append(q.copy())
            return float(q @ q / 2), q.copy()
        def rejected(oracle, metric, q, value, gradient, rng, **options):
            self.assertEqual(options, dict(step=.02, steps=1))
            return q, value, gradient, dict(accepted=False, energy_error=2.,
                                          log_acceptance=-2., force_evaluations=1)
        with tempfile.TemporaryDirectory() as temporary:
            report = {}
            with patch.object(pilot, 'N', 2), patch.object(pilot, 'split_hmc_step', rejected):
                pilot.run(oracle, initial, Path(temporary), report, lambda: None)
            self.assertEqual(report['accepted_proposals'], 0)
            self.assertEqual(len(report['sampler_trace']), 2)
            self.assertTrue(all(row['canonical_jump_rms'] == 0 for row in report['sampler_trace']))
            with np.load(Path(temporary) / 'hmc_retained_state.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['q'], initial)
                self.assertFalse(bool(saved['posterior_sample']))
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
