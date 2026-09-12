import inspect
import unittest
import numpy as np
from cf4_member_budget import support_keys, check_union_budget, evaluate, PASS, FAIL, UNAVAILABLE
from cf4_continuous_matter import boost_scale


class MemberBudgetTests(unittest.TestCase):
    def test_cell_identity_and_union_accounting(self):
        p = np.array([4.5, 4.5, 4.5])
        keys = support_keys(p, 1., dx=1., n=10)
        all_cells = np.array(np.unravel_index(np.arange(1000), (10,)*3)).T
        expected = np.flatnonzero(np.linalg.norm(all_cells+.5-p, axis=1) <= 1.)
        np.testing.assert_array_equal(keys, expected)
        self.assertEqual(len(keys), 7)
        values = np.zeros((7, len(keys)))
        values[0] = 10/len(keys)
        values[4:] = values[0]
        members = {r: dict(position_cMpc_h=p, radius_cMpc_h=1., mass_Msun=m,
                          velocity_km_s=[0., 0., 0.]) for r, m in zip(('MW', 'M31', 'M33'), (4., 4., 4.))}
        result = evaluate(keys, values, members, dx=1., n=10)
        self.assertEqual(result['summary']['hosts_mass'], PASS)
        self.assertEqual(result['summary']['all_three_mass'], FAIL)
        triple = result['unions'][-1]
        self.assertEqual(triple['union_cell_count'], 7)
        self.assertEqual(triple['sum_individual_cell_counts'], 21)
        self.assertAlmostEqual(triple['total_BOX_moments'][0], 10.)
        members['M33']['mass_Msun'] = 0
        self.assertEqual(evaluate(keys, values, members, dx=1., n=10)['summary']['all_three_mass'], UNAVAILABLE)
        self.assertIsNone(support_keys([0., 4., 4.], 1., dx=1., n=10))
        self.assertEqual(set(inspect.signature(evaluate).parameters), {'cell_keys','field_moments','members','dx','n'})

    def test_moments_zero_reservoir_and_common_boost(self):
        total = np.array([10., 0., 0., 0., 10., 20., 30.])
        valid = check_union_budget(total, [10.], [[0., 0., 0.]], reference_velocity=[0., 0., 0.])
        self.assertEqual(valid['moment_status'], PASS)  # rM=0, rQ>0 is internal dispersion.
        invalid = check_union_budget(total, [5.], [[3., 0., 0.]], reference_velocity=[0., 0., 0.])
        self.assertEqual(invalid['mass_status'], PASS)
        self.assertEqual(invalid['moment_status'], FAIL)
        boost = np.array([105., -77., 31.])
        for mass, velocity in ((10., [0.,0.,0.]), (5., [3.,0.,0.])):
            a = check_union_budget(total, [mass], [velocity], reference_velocity=[0.,0.,0.])
            b = check_union_budget(boost_scale(total, 1., boost), [mass], [np.array(velocity)+boost], reference_velocity=boost)
            self.assertEqual(a['moment_status'], b['moment_status'])
            np.testing.assert_allclose(a['normalized_min_eigenvalues'], b['normalized_min_eigenvalues'], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
