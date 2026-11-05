import unittest

import numpy as np

from cf4_r2_lowk_component_attribution import merge_component_score_gradient


class ComponentGradientLayoutTest(unittest.TestCase):
    def test_four_score_inputs_reconstruct_canonical_and_nuisance_blocks(self):
        rho = np.array([1.0, 2.0])
        velocity = np.array([3.0, 4.0])
        tracer = np.array([5.0])
        population = np.array([6.0, 7.0])

        def field_pullback(cotangents):
            density_gradient, velocity_gradient = cotangents
            return (density_gradient+2.0*velocity_gradient,)

        canonical = merge_component_score_gradient(
            field_pullback, (rho, velocity, tracer, population))
        np.testing.assert_array_equal(canonical, [7.0, 10.0, 5.0, 6.0, 7.0])

    def test_incomplete_component_gradient_tuple_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'four input-gradient blocks'):
            merge_component_score_gradient(lambda cotangents: (cotangents[0],),
                                           (np.array([1.0]),)*3)


if __name__ == '__main__':
    unittest.main()
