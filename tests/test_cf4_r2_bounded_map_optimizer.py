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


if __name__=='__main__':
    unittest.main()
