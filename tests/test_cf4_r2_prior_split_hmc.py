"""Algorithm mechanics only; no CF4 posterior, calibration or mixing claim."""
import sys
from pathlib import Path
import unittest
import json
import tempfile
import time
import contextlib
import io
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cf4_r2_prior_split_hmc import (
    FixedSplitMetric,split_trajectory,split_hmc_step,canonical_from_optimizer_oracle,
    inverse_laplacian_metric_symbol,bounded_split_pilot,restore_numpy_rng)


class PriorSplitTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(270929)
        modes=np.meshgrid(*[np.fft.fftfreq(2)]*3,indexing='ij')
        symbol=1./(1.+4.*sum(k*k for k in modes))
        self.metric=FixedSplitMetric(symbol,np.array([[.5,.1],[.1,1.2]]))
        self.q=self.rng.normal(size=10); self.p=self.metric.momentum(self.rng)
        a=self.rng.normal(size=(3,10))*.2
        self.precision=np.eye(10)+a.T@a
        self.linear=self.rng.normal(size=10)*.1

    def oracle(self,q):
        return .5*q@self.precision@q-self.linear@q,self.precision@q-self.linear

    def test_optimizer_coordinate_adapter_preserves_standard_normal_prior(self):
        scale=np.r_[np.ones(2),np.full(9,100.),1.]
        def old(x):
            return .5*np.sum((x/scale)**2),x/scale**2
        oracle=canonical_from_optimizer_oracle(old,2)
        q=self.rng.normal(size=12)
        value,gradient=oracle(q)
        self.assertAlmostEqual(value,.5*q@q,places=12)
        np.testing.assert_allclose(gradient,q,rtol=0,atol=1e-14)

    def test_live_adapter_checkpoint_and_parent_reference(self):
        from cf4_r2_split_sampling_run import run_pilot,same_target_reference
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            scale=np.r_[np.ones(8),np.full(9,100.),1.]
            initial=self.rng.normal(size=18)*scale
            def oracle(x):
                return .5*np.sum((x/scale)**2),x/scale**2
            def score(x):
                return oracle(x)[0],np.zeros(3),None,1
            field=lambda _: (np.ones((2,2,2)),np.zeros((3,2,2,2)),
                             np.zeros((3,2,2,2)),np.ones((2,2,2),bool))
            parent=dict(N=2,box_cMpc_h=384.,count_integration={'family':'shell_cdf'},
                status='PARTIAL_MAP_OPTIMIZER_STOP_NOT_POSTERIOR',training_singletons=1414,
                training_counts=47121,trace=[{'parts':[0.,0.,0.]}],final_objective=oracle(initial)[0],
                nuisance_optimizer_metric={'matrix':np.diag(scale[8:]**2).tolist()})
            (out/'result.json').write_text(json.dumps(parent))
            reference=same_target_reference(out/'accepted_checkpoint.npz',parent)
            self.assertEqual(reference['objective'],oracle(initial)[0])
            report={}
            with contextlib.redirect_stdout(io.StringIO()):
                run_pilot(initial=initial,value=oracle(initial)[0],gradient=oracle(initial)[1],
                    objective=oracle,score_only=score,terminal_field=field,
                    metric_report=out/'result.json',report=report,save_report=lambda:None,
                    out=out,started=time.monotonic(),cap=1000.,n=2)
            self.assertEqual(report['retained_proposals'],16)
            self.assertEqual(report['retained_accepted'],16)
            self.assertEqual(len(report['sampler_trace']),32)
            self.assertEqual(report['status'],'SPLIT_HMC_FEASIBILITY_STOP_NOT_POSTERIOR')
            with np.load(out/'accepted_checkpoint.npz') as checkpoint:
                np.testing.assert_allclose(checkpoint['canonical_gradient'],checkpoint['parameters']/scale)
                continuation=checkpoint['parameters'].copy()
            completed=dict(parent,**report)
            (out/'result.json').write_text(json.dumps(completed))
            ref=same_target_reference(out/'accepted_checkpoint.npz',parent)
            self.assertTrue(ref['sampling_parent'])
            later=out/'long'; later.mkdir()
            followup={'same_target_sampling_reference':ref}
            with contextlib.redirect_stdout(io.StringIO()):
                run_pilot(initial=continuation,value=oracle(continuation)[0],
                    gradient=oracle(continuation)[1],objective=oracle,score_only=score,
                    terminal_field=field,metric_report=out/'result.json',report=followup,
                    save_report=lambda:None,out=later,started=time.monotonic(),cap=1000.,n=2,
                    long_trajectory=True)
            self.assertEqual(followup['proposals'],8)
            self.assertEqual(followup['sampler']['integration_steps'],8)
            self.assertEqual(followup['warmup_proposals'],0)
            self.assertTrue(all(row['step_size']==ref['frozen_step'] for row in followup['sampler_trace']))
            bad=dict(parent,training_singletons=429)
            (out/'result.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                same_target_reference(out/'accepted_checkpoint.npz',parent)

    def test_nonfinite_hamiltonian_is_error_not_acceptance(self):
        for values in ([np.nan],[np.inf],[1.,np.nan],[1.,np.inf]):
            with patch.object(self.metric,'kinetic',side_effect=values):
                with self.assertRaises(FloatingPointError):
                    split_hmc_step(self.oracle,self.metric,self.q,*self.oracle(self.q),
                        self.rng,step=.1,steps=1)

    def test_metric_and_bounded_pilot_freeze_step_and_keep_rejections(self):
        symbol=inverse_laplacian_metric_symbol(4)
        self.assertEqual(symbol[0,0,0],1.)
        self.assertAlmostEqual(symbol[1,0,0],1./6000.)
        relaxed=inverse_laplacian_metric_symbol(4,fundamental_mass=600.)
        self.assertAlmostEqual(relaxed[1,0,0],1./600.)
        FixedSplitMetric(symbol,np.eye(2))
        FixedSplitMetric(relaxed,np.eye(2))
        rows=[]
        _,_,_,trace,message=bounded_split_pilot(self.oracle,self.metric,self.q,
            *self.oracle(self.q),self.rng,warmup=3,retained=3,steps=2,
            seconds_left=lambda:100.,callback=lambda q,u,g,row:rows.append(row),
            endpoint_value=lambda q:self.oracle(q)[0])
        self.assertEqual(len(rows),6)
        self.assertEqual(len({row['step_size'] for row in trace[3:]}),1)
        self.assertEqual(message,'proposal limit')
        rows=[]
        q,_,_,trace,_=bounded_split_pilot(lambda q:(np.inf,None),self.metric,self.q,
            *self.oracle(self.q),self.rng,warmup=1,retained=2,
            seconds_left=lambda:100.,callback=lambda q,u,g,row:rows.append(q.copy()))
        np.testing.assert_array_equal(q,self.q)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(row['canonical_jump_rms']==0. for row in trace))
        with self.assertRaises(FloatingPointError):
            split_hmc_step(self.oracle,self.metric,self.q,*self.oracle(self.q),self.rng,
                step=.1,steps=1,endpoint_value=lambda q:self.oracle(q)[0]+1.)

    def test_numpy_rng_checkpoint_restores_next_draw_and_explicit_seed_replays(self):
        original=restore_numpy_rng(12345)
        original.standard_normal(7)
        saved_state=original.bit_generator.state
        expected=original.standard_normal(5)
        resumed=restore_numpy_rng(state=saved_state)
        np.testing.assert_array_equal(resumed.standard_normal(5),expected)
        replayed=restore_numpy_rng(12345)
        replayed.standard_normal(7)
        np.testing.assert_array_equal(replayed.standard_normal(5),expected)

    def test_prior_flow_energy_and_reverse(self):
        q,p=self.metric.prior_flow(self.q,self.p,.71)
        before=.5*self.q@self.q+self.metric.kinetic(self.p)
        after=.5*q@q+self.metric.kinetic(p)
        self.assertAlmostEqual(before,after,places=12)
        q,p=self.metric.prior_flow(q,p,-.71)
        np.testing.assert_allclose(q,self.q,rtol=0,atol=2e-14)
        np.testing.assert_allclose(p,self.p,rtol=0,atol=2e-14)

    def test_full_map_reversible_and_volume_preserving(self):
        def run(z):
            q,p,_,_=split_trajectory(self.oracle,self.metric,z[:10],z[10:],.3,3)
            return np.r_[q,p]
        start=np.r_[self.q,self.p]
        final=run(start)
        q,p,_,_=split_trajectory(self.oracle,self.metric,final[:10],-final[10:],.3,3)
        np.testing.assert_allclose(q,self.q,rtol=0,atol=3e-14)
        np.testing.assert_allclose(p,-self.p,rtol=0,atol=3e-14)
        eps=1e-5
        jac=np.column_stack([(run(start+eps*d)-run(start-eps*d))/(2*eps) for d in np.eye(20)])
        self.assertAlmostEqual(float(np.linalg.det(jac)),1.,places=8)

    def test_trajectory_length_paths_share_unmutated_state_and_momentum(self):
        q_start=self.q.copy(); p_start=self.p.copy()
        initial=self.oracle(q_start)
        initial_h=initial[0]+self.metric.kinetic(p_start)
        endpoints={}
        for steps in (1,2,4):
            q_end,p_end,value,gradient=split_trajectory(
                self.oracle,self.metric,q_start,p_start,.08,steps,
                initial_evaluation=initial)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(np.isfinite(gradient).all())
            delta_h=value+self.metric.kinetic(p_end)-initial_h
            self.assertTrue(np.isfinite(delta_h))
            endpoints[steps]=q_end
        np.testing.assert_array_equal(self.q,q_start)
        np.testing.assert_array_equal(self.p,p_start)
        self.assertGreater(float(np.linalg.norm(endpoints[4]-endpoints[1])),0.)

    def test_gaussian_target_mean_covariance_with_metropolis(self):
        q=self.q.copy(); value,gradient=self.oracle(q)
        samples=[]; accepted=0
        for i in range(7000):
            q,value,gradient,info=split_hmc_step(self.oracle,self.metric,q,value,gradient,
                self.rng,step=.5,steps=3)
            accepted+=info['accepted']
            if i>=1000:
                samples.append(q.copy())
        samples=np.asarray(samples)
        covariance=np.linalg.inv(self.precision)
        mean=covariance@self.linear
        # Broad fixed-seed algorithm regression, NOT a calibrated science gate.
        self.assertGreater(accepted,5000)
        np.testing.assert_allclose(samples.mean(axis=0),mean,rtol=0,atol=.12)
        np.testing.assert_allclose(np.cov(samples.T),covariance,rtol=0,atol=.13)

    def test_invalid_metric_and_true_zero_support(self):
        with self.assertRaises(ValueError):
            FixedSplitMetric(np.ones((2,2,2)),np.diag([1.,0.]))
        asymmetric=np.ones((3,3,3)); asymmetric[1,0,0]=2.
        with self.assertRaises(ValueError):
            FixedSplitMetric(asymmetric,np.eye(2))
        q,value,gradient,info=split_hmc_step(lambda _: (np.inf,None),self.metric,
            self.q,*self.oracle(self.q),self.rng,step=.5,steps=2)
        self.assertFalse(info['accepted'])
        self.assertEqual(info['force_evaluations'],1)
        np.testing.assert_array_equal(q,self.q)
        with self.assertRaises(FloatingPointError):
            split_trajectory(lambda x:(1.,np.full_like(x,np.nan)),self.metric,self.q,self.p,.1,1)


if __name__=='__main__':
    unittest.main()
