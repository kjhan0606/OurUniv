"""Small algorithm check in the same Slurm allocation, without a PM fit."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cf4_r2_v6_partial_map import bounded_lbfgs


class BoundedMapTests(unittest.TestCase):
    def test_quadratic_decreases_and_zero_support_trial_is_rejected(self):
        rejected=[]
        accepted=[]
        # 1 IC +9 scaled nuisance +1 zero coordinate. Minimum at IC=.09;
        # first .1-sized trial lies outside this toy's explicit support.
        truth=np.zeros(11); truth[0]=.09
        def objective(x):
            if x[0]>.095:
                rejected.append(x.copy())
                return np.inf,np.zeros_like(x)
            delta=x-truth
            return 50*np.dot(delta,delta),100*delta
        def callback(x):
            accepted.append(objective(x)[0])
        x,value,grad,message=bounded_lbfgs(objective,np.zeros(11),callback,
            n_ic=1,seconds_left=lambda:10.,maxiter=30)
        self.assertGreater(len(rejected),0)
        self.assertTrue(np.all(np.diff(accepted)<=0))
        self.assertEqual(message,'gradient tolerance')
        np.testing.assert_allclose(x,truth,atol=1e-6)
        self.assertLess(value,1e-10)

    def test_stiff_descent_uses_score_only_rejections(self):
        calls={'value':0,'gradient':0}
        accepted=[]
        truth=np.zeros(11); truth[0]=1e-7
        def score(x):
            calls['value']+=1
            return 5e11*np.dot(x-truth,x-truth)
        def objective(x):
            calls['gradient']+=1
            return 5e11*np.dot(x-truth,x-truth),1e12*(x-truth)
        x,value,grad,message=bounded_lbfgs(objective,np.zeros(11),
            lambda x:accepted.append(x.copy()),n_ic=1,seconds_left=lambda:10.,
            maxiter=30,value_only=score,initial_norm_cap=1e-5)
        self.assertGreater(calls['value'],len(accepted))
        self.assertEqual(calls['gradient'],1+len(accepted))
        self.assertEqual(message,'gradient tolerance')
        np.testing.assert_allclose(x,truth,atol=1e-14)

    def test_score_full_mismatch_is_not_silently_accepted(self):
        def objective(x):
            return .5*np.dot(x-1,x-1),x-1
        with self.assertRaisesRegex(FloatingPointError,'objective mismatch'):
            bounded_lbfgs(objective,np.zeros(11),lambda x:None,n_ic=1,
                seconds_left=lambda:10.,value_only=lambda x:-100.)


if __name__=='__main__':
    unittest.main()
