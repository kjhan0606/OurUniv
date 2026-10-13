import unittest

import numpy as np

from cf4_r2_count_operator_closure_mock import (
    aggregate_expected, aggregate_observed, table_rows, pearson_summary,
)


class CountOperatorClosureMockTest(unittest.TestCase):
    def test_final_radial_bin_is_labeled_as_unbounded_tail(self):
        observed=np.zeros((6,16),dtype=np.int64)
        tsc=np.ones((6,16),dtype=np.float64)
        ngp=np.ones((6,16),dtype=np.float64)*2
        rows=table_rows(observed,tsc,ngp)
        edge=rows[-1]
        self.assertEqual(edge['radius_bin_label'], '>=180')
        self.assertIsNone(edge['radius_upper_cMpc_h'])

    def test_aggregate_means_respect_population_exposure(self):
        intensity=np.zeros((6,8),dtype=np.float64)
        intensity[0]=np.arange(1,9)
        exposure=np.zeros_like(intensity,dtype=bool)
        exposure[0,[0,1,4,5]]=True
        radial=np.array([0,0,1,1,0,0,1,1])
        expected=aggregate_expected(intensity,exposure,radial)
        self.assertEqual(expected[0,0],1+2+5+6)
        self.assertEqual(expected[0,1],0)

    def test_poisson_diagnostic_uses_predeclared_expected_threshold(self):
        result=pearson_summary(np.array([[12,8],[0,0]]),
                               np.array([[10.,10.],[4.,2.]]), minimum_expected=5.)
        self.assertEqual(result['degrees_of_freedom'],2)
        self.assertAlmostEqual(result['statistic'],.8)
        self.assertTrue(0. < result['p_value'] < 1.)


if __name__ == '__main__':
    unittest.main()
