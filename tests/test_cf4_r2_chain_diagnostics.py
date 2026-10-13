import unittest
import numpy as np
from cf4_r2_chain_diagnostics import scalar_diagnostics,retained_scalar_matrix


class DiagnosticsTests(unittest.TestCase):
    def test_iid_and_displaced_chains(self):
        x=np.random.default_rng(91).normal(size=(2,10000))
        d=scalar_diagnostics(x)
        self.assertLess(d['rank_folded_split_rhat'],1.01)
        self.assertGreater(d['batch_means_ess'],10000)
        x[1]+=3
        self.assertGreater(scalar_diagnostics(x)['rank_folded_split_rhat'],1.2)

    def test_correlated_and_stuck_chains(self):
        x=np.zeros((2,10000));noise=np.random.default_rng(4).normal(size=x.shape)
        for i in range(1,x.shape[1]):x[:,i]=.95*x[:,i-1]+noise[:,i]
        self.assertLess(scalar_diagnostics(x)['batch_means_ess'],2000)
        d=scalar_diagnostics(np.ones((2,20)))
        self.assertIsNone(d['rank_folded_split_rhat']);self.assertIsNone(d['batch_means_ess'])
        with self.assertRaises(ValueError):scalar_diagnostics(np.ones((2,3)))

    def test_rejections_remain_in_retained_trace(self):
        row=dict(warmup=False,accepted=False,fine_energy=1.,white_mean_square=1.,
            inherited_low_white_mean_square=.7,nuisance_white=[0.]*24,
            fundamental_real=[0.]*3,fundamental_imag=[0.]*3,count_raw_scores=[0.]*2,
            density_octant_means=[1.]*8)
        labels,values=retained_scalar_matrix({'trace':[dict(row,warmup=True),row,row]})
        self.assertEqual(values.shape,(2,len(labels)))
        np.testing.assert_array_equal(values[0],values[1])


if __name__=='__main__':unittest.main()
