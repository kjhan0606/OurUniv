import unittest
import numpy as np
from cf4_r2_prior_split_hmc import FixedSplitMetric,split_trajectory
from cf4_r2_corrected_split_hmc import corrected_split_step


def coarse(q):
    grad=q.copy();grad[-1]+=.3*(q[-1]+1.)
    return .5*q@q+.15*(q[-1]+1.)**2,grad


def fine(q):return .5*q@q+1.5*(q[-1]-1.)**2


class CorrectedSplitTests(unittest.TestCase):
    def setUp(self):self.metric=FixedSplitMetric(np.ones((2,2,2)),np.eye(1))

    def test_proposal_map_reversible_with_different_fine_target(self):
        rng=np.random.default_rng(83);q=rng.normal(size=9);p=self.metric.momentum(rng)
        q1,p1,_,_=split_trajectory(coarse,self.metric,q,p,.3,6)
        q2,p2,_,_=split_trajectory(coarse,self.metric,q1,-p1,.3,6)
        np.testing.assert_allclose(q2,q,atol=1e-12)
        np.testing.assert_allclose(p2,-p,atol=1e-12)

    def test_rejection_retains_both_energy_and_force_cache(self):
        rng=np.random.default_rng(5);q=np.zeros(9);c,g=coarse(q);e=fine(q)
        result=corrected_split_step(coarse,lambda _:np.inf,self.metric,q,e,c,g,rng,step=.3,steps=2)
        self.assertFalse(result[-1]['accepted'])
        np.testing.assert_array_equal(result[0],q);np.testing.assert_array_equal(result[3],g)
        self.assertEqual(result[1],e);self.assertEqual(result[2],c)

    def test_moments_follow_fine_not_force_posterior(self):
        rng=np.random.default_rng(2026092912);q=np.zeros(9);c,g=coarse(q);e=fine(q)
        samples=[]
        for i in range(16000):
            q,e,c,g,_=corrected_split_step(coarse,fine,self.metric,q,e,c,g,rng,step=.3,steps=6)
            if i>=1000:samples.append(q[-1])
        # Fine posterior N(.75,.25); force posterior N(-.3/1.3,1/1.3).
        self.assertAlmostEqual(float(np.mean(samples)),.75,delta=.05)
        self.assertAlmostEqual(float(np.var(samples)),.25,delta=.04)

    def test_finite_rejection_energy_attribution_uses_starting_cache(self):
        rng=np.random.default_rng(44);q=np.zeros(9);c,g=coarse(q)
        sharp=lambda x:1e6*float(x@x)
        result=corrected_split_step(coarse,sharp,self.metric,q,sharp(q),c,g,rng,step=.3,steps=2)
        info=result[-1]
        self.assertFalse(info['accepted'])
        self.assertAlmostEqual(info['energy_error'],
            info['force_hamiltonian_error']+info['fine_force_correction_change'],places=7)
        self.assertEqual(info['initial_force_energy'],c)
        self.assertEqual(info['initial_fine_energy'],0.)
        self.assertAlmostEqual(info['fine_force_correction_change'],
            info['proposed_fine_energy']-info['proposed_force_energy']+c,places=7)
        self.assertEqual(result[1],0.);self.assertEqual(result[2],c)


if __name__=='__main__':unittest.main()
