import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import multivariate_normal, norm

from cf4_r2_conditional_fp import (
    CONDITION_MATRIX as A, component_terms, conditioned_log_weights, mixture_logpdf)


class ConditionalFPTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.array([.3, 2.2, 2.7])
        lower = np.array([[.3, 0, 0], [.06, .12, 0], [-.2, .03, .08]])
        self.cov = lower @ lower.T
        self.u = A @ self.mean + [.1, -.02]

    def test_joint_density_ratio_includes_transformation(self):
        terms = component_terms(self.mean, self.cov, self.u)
        transform = np.vstack((A, [1., 0., 0.]))
        self.assertAlmostEqual(abs(np.linalg.det(transform)), 1.)
        for t in (-.1, .3, .6):
            x = np.linalg.solve(transform, np.r_[self.u, t])
            expected = (multivariate_normal.logpdf(x, self.mean, self.cov)
                        - multivariate_normal.logpdf(self.u, A @ self.mean, A @ self.cov @ A.T))
            got = norm.logpdf(t, terms['mean'], np.sqrt(terms['variance']))
            self.assertAlmostEqual(got, expected, places=10)

    def test_eta_direction_and_information_decomposition(self):
        h = np.array([1., .07, -.1])
        terms = component_terms(self.mean, self.cov, self.u, mean_direction=h)
        eps = 1e-5
        fd = (component_terms(self.mean + eps*h, self.cov, self.u)['mean']
              - component_terms(self.mean - eps*h, self.cov, self.u)['mean'])/(2*eps)
        self.assertAlmostEqual(fd, terms['mean_response'], places=10)
        marginal = (A @ h) @ np.linalg.solve(A @ self.cov @ A.T, A @ h)
        self.assertAlmostEqual(terms['conditional_fisher'] + marginal,
                               terms['joint_fisher'], places=10)
        # This direction changes only U, leaving T|U without location information.
        h_empty = self.cov @ A[0]
        empty = component_terms(self.mean, self.cov, self.u, mean_direction=h_empty)
        self.assertLess(abs(empty['mean_response']), 1e-13)

    def test_mixture_normalization_and_reweighting(self):
        means = self.mean + np.array([[-.2, 0, 0], [.2, 0, 0]])
        u = A @ self.mean + [1.2, .024]
        terms = component_terms(means, .09*np.eye(3), u)
        lw = np.log([.5, .5])
        integral = quad(lambda y: np.exp(mixture_logpdf(y, terms, lw)), -3, 4)[0]
        self.assertAlmostEqual(integral, 1., places=10)
        w = np.exp(conditioned_log_weights(lw, terms['log_condition_density']))
        self.assertGreater(abs(w @ terms['mean'] - np.mean(terms['mean'])), .03)
        np.testing.assert_allclose(mixture_logpdf([.5, .8], terms, lw + 1000),
                                   mixture_logpdf([.5, .8], terms, lw), atol=1e-12)

    def test_selection_cancellation_and_failure_control(self):
        for probability in (.05, .2, .9):
            z = quad(lambda e: probability*norm.pdf(e), -10, 10)[0]
            self.assertAlmostEqual(z/probability, 1., places=12)
            self.assertAlmostEqual(quad(lambda e: e*probability*norm.pdf(e)/z,
                                        -10, 10)[0], 0., places=12)
        z = quad(lambda e: norm.pdf(e)*expit(.5*e), -10, 10)[0]
        shift = quad(lambda e: e*norm.pdf(e)*expit(.5*e)/z, -10, 10)[0]
        self.assertAlmostEqual(shift, .23604442243988, places=11)

    def test_shared_zero_point_is_not_independent_information(self):
        info = component_terms(self.mean, self.cov, self.u)['conditional_fisher']
        fisher = info*np.ones((2, 2))  # eta and zero point have the same mean direction.
        self.assertEqual(np.linalg.matrix_rank(fisher), 1)

    def test_invalid_covariance_and_support(self):
        for cov in (np.zeros((3, 3)), np.diag([1., -1., 1.]), np.eye(3)*np.nan):
            with self.assertRaises(ValueError):
                component_terms(self.mean, cov, self.u)
        for weights in ([-np.inf, -np.inf], [0., np.nan], [0., np.inf]):
            with self.assertRaises(ValueError):
                conditioned_log_weights(weights, [0., 0.])


if __name__ == '__main__':
    unittest.main()
