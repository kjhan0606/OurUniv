import unittest

import numpy as np
from scipy.optimize import minimize

from cf4_affine_objective import AffineObjective


class AffineObjectiveTest(unittest.TestCase):
    def test_coupled_ill_conditioned_target_and_gradient(self):
        # Stiff nuisance coupled to the field: a diagonal scalar sweep is slow.
        hessian = np.array([[2., 900.], [900., 1.e6]])
        solution = np.array([.2, -.003])
        origin = np.array([1., .01])

        def physical(q):
            delta = q - solution
            return .5 * delta @ hessian @ delta, hessian @ delta

        target = AffineObjective(physical, origin, [1., 1.e-3], offset=5.)
        z = np.array([.3, -.2])
        value, gradient = target(z)
        self.assertAlmostEqual(value + 5., physical(target.physical(z))[0])
        for i in range(2):
            dz = np.eye(2)[i] * 1.e-5
            numerical = (target(z + dz)[0] - target(z - dz)[0]) / 2.e-5
            self.assertAlmostEqual(numerical, gradient[i], places=7)
        result = minimize(target, np.zeros(2), jac=True, method='L-BFGS-B',
                          options={'gtol': 1.e-10, 'ftol': 0., 'maxiter': 100})
        np.testing.assert_allclose(target.physical(result.x), solution, atol=1.e-7)

    def test_invalid_scale_and_shape(self):
        for scale in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                AffineObjective(lambda q: None, [1.], scale)
        target = AffineObjective(lambda q: None, [1.], 1.)
        with self.assertRaises(ValueError):
            target.physical([1., 2.])


if __name__ == '__main__':
    unittest.main()
