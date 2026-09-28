"""Algorithm mechanics only; no CF4 posterior, calibration or mixing claim."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cf4_r2_prior_split_hmc import (
    FixedSplitMetric,split_trajectory,split_hmc_step,canonical_from_optimizer_oracle,
    inverse_laplacian_metric_symbol,bounded_split_pilot)


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

    def test_metric_and_bounded_pilot_freeze_step_and_keep_rejections(self):
        symbol=inverse_laplacian_metric_symbol(4)
        self.assertEqual(symbol[0,0,0],1.)
        self.assertAlmostEqual(symbol[1,0,0],1./6000.)
        FixedSplitMetric(symbol,np.eye(2))
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
