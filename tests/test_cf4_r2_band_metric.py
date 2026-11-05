import unittest
import time
import json
import os
import tempfile
from unittest.mock import patch
import numpy as np
from cf4_r2_prior_split_hmc import (
    band_scaled_inverse_mass, inverse_laplacian_metric_symbol, FixedSplitMetric,
)
from cf4_r2_band_control import paired_control, replay_diagnostic_rng_pairs, run


class BandMetricTests(unittest.TestCase):
    def test_replay_diagnostic_rng_pairs_verifies_common_acceptance_draws(self):
        seed = 17
        dimension = 7
        source = np.random.default_rng(seed)
        rows = []
        expected = []
        for pair in range(3):
            source.standard_normal(dimension)
            uniform = float(source.uniform())
            expected.append(uniform)
            rows.append(dict(pair=pair, complete=True, arms={
                'A':dict(log_acceptance=0., accepted=True),
                'B':dict(log_acceptance=0., accepted=True),
            }))
        actual = replay_diagnostic_rng_pairs(
            np.random.default_rng(seed), dimension, rows)
        self.assertEqual(actual, expected)

        rows[1]['arms']['B']['accepted'] = False
        with self.assertRaisesRegex(ValueError, 'disagrees with pair 1 arm B'):
            replay_diagnostic_rng_pairs(np.random.default_rng(seed), dimension, rows)

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

    def test_driver_resumes_only_pair3_after_verifying_three_prior_pairs(self):
        class Oracle:
            def __init__(self):
                self.records=[]
                self.max_evaluations=17
                self.deadline=time.monotonic()+100000
            def __call__(self,q):
                self.records.append({'present_field_readout':{'fixture':1}})
                return float(q@q/2),q.copy()

        checkpoint = '/fixture/r2_hmc_equil24h_20261013/hmc_retained_state.npz'
        prior = dict(
            source_commit='666edbdff5702f7d525efec6461b7fe1cf3049a7',
            status='CONDITIONAL_BAND_CONTROL_DIAGNOSTIC_NOT_POSTERIOR',
            stop_reason='insufficient complete-pair budget', complete_pairs=3,
            diagnostic_seed=2026102001, N=12, box_cMpc_h=384.0,
            control_metric='B=4*A only for16<integer_k_squared<=64',
            control_step=.11996093316338646, control_steps=8,
            start_checkpoint=checkpoint, posterior_sample=False,
            discard_for_inference=True, exact_target_evaluations=49,
            paired_results=[dict(pair=i,complete=True,arms={
                'A':dict(log_acceptance=0.,accepted=True,
                         weighted_band_squared_jump_per_force=1.),
                'B':dict(log_acceptance=0.,accepted=True,
                         weighted_band_squared_jump_per_force=3.),
            }) for i in range(3)],
        )
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8') as stream:
            json.dump(prior,stream)
            stream.flush()
            env={
                'CF4_BAND_CONTROL_START_PAIR':'3',
                'CF4_BAND_CONTROL_PRIOR_RESULT':stream.name,
            }
            oracle=Oracle(); report={'start_checkpoint':checkpoint}
            q=np.zeros(12**3+24)
            with patch.dict(os.environ,env), patch('cf4_r2_conditional_map.N',12):
                run(oracle,q,None,report,lambda:None)
        self.assertEqual(report['complete_pairs'],4)
        self.assertEqual([row['pair'] for row in report['paired_results']],
                         [0,1,2,3])
        self.assertEqual(len(oracle.records),17)
        self.assertEqual(report['pair3_continuation']['prior_exact_target_evaluations'],49)
        self.assertEqual(
            report['pair3_continuation']['cumulative_exact_target_evaluations_across_jobs'],
            66,
        )

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

