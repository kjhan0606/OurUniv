"""Schedule/checkpoint mechanics only; not CF4 posterior validation."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import cf4_r2_hmc_transport as transport


class Oracle:
    max_evaluations = 49
    def __init__(self):
        self.records = []
        self.deadline = time.monotonic()+100000.
    def __call__(self, q):
        value = float(q@q/2)
        self.records.append({k:value+i for i,k in enumerate(transport.TERM_KEYS)})
        return value, q.copy()


class TransportTests(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            q=np.arange(88,dtype=float)/100
            rng=np.random.default_rng(345)
            original_rng=json.dumps(rng.bit_generator.state)
            start=out/'start.npz'
            np.savez(start,q=q,rng_state=original_rng)
            oracle=Oracle()
            if mode=='preflight_stop': oracle.max_evaluations=2
            calls=[]
            def transition(oracle,metric,q,value,gradient,rng,*,step,steps):
                calls.append((step,steps,float(rng.normal())))
                if mode=='mid_stop': raise transport.EvaluationBudgetStop()
                proposed=q+.001
                for _ in range(steps): new_value,new_grad=oracle(proposed)
                accepted=len(calls)%2==1
                return (proposed if accepted else q, new_value if accepted else value,
                    new_grad if accepted else gradient,
                    dict(accepted=accepted,log_acceptance=0. if accepted else -2.,
                         energy_error=-.1 if accepted else 2.,force_evaluations=steps))
            report=dict(target_rules='fixture',target_orders={'value':2,'gradient':2},
                        split_source='fixture',association_ledger_source_commit='fixture')
            with patch.object(transport,'START',start), patch.object(transport,'N',4), \
                    patch.object(transport,'split_hmc_step',transition):
                transport.run(oracle,q,out,report,lambda:None)
            with np.load(out/'hmc_retained_state.npz',allow_pickle=False) as f:
                saved={k:f[k].copy() for k in f.files}
            return report,saved,calls,original_rng,q

    def test_sequential_schedule_freeze_rejections_and_rng_resume(self):
        report,saved,calls,_,_=self.exercise('normal')
        self.assertEqual([x[1] for x in calls[:6]],[2,2,4,4,6,6])
        self.assertTrue(all(x[1] in (4,6) for x in calls[6:]))
        self.assertEqual(len(set(x[0] for x in calls[6:])),1)
        self.assertEqual(calls[0][2],float(np.random.default_rng(345).normal()))
        self.assertEqual(report['warmup_completed'],6)
        self.assertEqual(report['diagnostic_retained'],4)
        for previous,row in zip(report['sampler_trace'][::2],report['sampler_trace'][1::2]):
            self.assertEqual(row['current_terms'],previous['current_terms'])
            self.assertEqual(row['canonical_jump_rms'],0.)
        self.assertEqual(int(saved['next_index']),10)
        self.assertFalse(bool(saved['posterior_sample']))

    def test_full_trajectory_budget_guard_restores_rng(self):
        report,saved,calls,rng,q=self.exercise('preflight_stop')
        self.assertEqual(calls,[])
        self.assertEqual(str(saved['rng_state']),rng)
        np.testing.assert_array_equal(saved['q'],q)
        self.assertEqual(report['diagnostic_retained'],0)

    def test_interrupted_candidate_is_not_rejection_or_sample(self):
        report,saved,calls,rng,q=self.exercise('mid_stop')
        self.assertEqual(len(calls),1)
        self.assertEqual(report['sampler_trace'],[])
        self.assertEqual(str(saved['rng_state']),rng)
        np.testing.assert_array_equal(saved['q'],q)

    def test_fundamental_sine_projection(self):
        n=4
        cube=np.broadcast_to(np.sin(2*np.pi*np.arange(n)/n)[:,None,None],(n,)*3)
        row=transport.mode_readout(cube.ravel(),n)
        self.assertAlmostEqual(row['fundamental_sine'][0],np.linalg.norm(cube))
        np.testing.assert_allclose(row['fundamental_sine'][1:],0.,atol=1e-14)


if __name__=='__main__': unittest.main()
