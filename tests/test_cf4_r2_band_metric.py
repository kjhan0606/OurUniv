import unittest
import time
from unittest.mock import patch
import numpy as np
from cf4_r2_prior_split_hmc import (
    band_scaled_inverse_mass, inverse_laplacian_metric_symbol, FixedSplitMetric,
)
from cf4_r2_band_control import paired_control
from cf4_r2_band_control import run


class BandMetricTests(unittest.TestCase):
    def test_driver_complete_pairs_and_budget_stop(self):
        class Oracle:
            def __init__(self, cap):
                self.records=[]
                self.max_evaluations=cap
                self.deadline=time.monotonic()+100000
            def __call__(self,q):
                self.records.append({'present_field_readout':{'fixture':1}})
                return float(q@q/2),q.copy()
        for cap, expected in ((65,4),(1,0)):
            report={}
            oracle=Oracle(cap)
            q=np.zeros(12**3+24)
            with patch('cf4_r2_conditional_map.N',12):
                run(oracle,q,None,report,lambda:None)
            self.assertEqual(report['complete_pairs'],expected)
            self.assertEqual(len(oracle.records),1+16*expected)
            self.assertFalse(report['selected_endpoint'])
            self.assertFalse(report['original_chain_rng_consumed'])
            np.testing.assert_array_equal(q,0.)

    def test_rejected_proposal_readout_is_not_retained_movement(self):
        metric = FixedSplitMetric(np.ones((12,)*3), np.eye(2))
        q = np.zeros(12**3+2)
        def transition(oracle, metric, q, value, gradient, rng, *,
                       step, steps, proposal_readout):
            proposed = q.copy()
            proposed[0] = 1.
            info = dict(accepted=False, log_acceptance=-2.,energy_error=2.,
                        force_evaluations=steps,
                        proposal_readout=proposal_readout(proposed,q))
            return q,value,gradient,info
        with patch('cf4_r2_band_control.split_hmc_step',transition):
            rows = paired_control(None,{'A':metric,'B':metric},q,0.,q.copy(),
                np.random.default_rng(3),step=.12,steps=8)
        for row in rows.values():
            self.assertEqual(row['retained_band_squared_jump_per_force'],0.)
            self.assertGreater(row['proposed_band_squared_jump_per_force'],0.)
            self.assertAlmostEqual(row['weighted_band_squared_jump_per_force'],
                np.exp(-2)*row['proposed_band_squared_jump_per_force'])

    def test_common_random_identical_arms_preserve_input(self):
        n = 12
        q = np.linspace(-.1,.1,n**3+2)
        before = q.copy()
        metric = FixedSplitMetric(np.ones((n,)*3), np.eye(2))
        def oracle(x):
            return float(x@x/2), x.copy()
        value, gradient = oracle(q)
        rng = np.random.default_rng(20261020)
        rows = paired_control(oracle, {'A':metric,'B':metric}, q,value,gradient,
                              rng,step=.12,steps=8)
        self.assertEqual(rows['A'],rows['B'])
        np.testing.assert_array_equal(q,before)
        self.assertTrue(rows['A']['discard_for_inference'])

    def test_only_declared_band_changes_and_input_is_preserved(self):
        c = inverse_laplacian_metric_symbol(20)
        before = c.copy()
        new = band_scaled_inverse_mass(c)
        k = np.fft.fftfreq(20)*20
        k2 = k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2
        mask = (k2 > 16) & (k2 <= 64)
        np.testing.assert_array_equal(c, before)
        np.testing.assert_array_equal(new[~mask], c[~mask])
        np.testing.assert_array_equal(new[mask], 4*c[mask])
        FixedSplitMetric(new, np.eye(2))  # Includes real-field symmetry check.
        self.assertEqual(new[4,0,0], c[4,0,0])
        self.assertEqual(new[8,0,0], 4*c[8,0,0])

    def test_invalid_configuration(self):
        for kwargs in ({'factor':0}, {'factor':np.inf}, {'lower':8,'upper':4}):
            with self.assertRaises(ValueError):
                band_scaled_inverse_mass(np.ones((4,4,4)), **kwargs)

