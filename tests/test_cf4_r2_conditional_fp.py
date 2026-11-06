import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import multivariate_normal, norm

from cf4_r2_conditional_fp import (
    CONDITION_MATRIX as A, component_terms, conditioned_log_weights, mixture_logpdf,
    grouped_log_integrals, shared_conditional_logpdf)


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

    def test_group_and_global_offsets_match_full_gaussian_conditioning(self):
        # Independent reference: analytically integrate Gaussian offsets in
        # the FULL 9-dimensional covariance, then condition on all six U's.
        row_group = np.array([0, 1, 0])
        means = np.array([self.mean, self.mean + [.02, .01, 0], self.mean - [.01, 0, .02]])
        direction = np.array([1., .08, -.12])
        x = means + [[.04, -.01, .02], [-.03, .02, -.04], [.08, 0, -.01]]
        cov = .09*np.eye(3)
        shared_sd, group_sd = .09, .07
        nodes, weights = np.polynomial.hermite.hermgauss(40)
        logw = np.log(weights/np.sqrt(np.pi))
        offsets = np.sqrt(2)*(shared_sd*nodes[:, None, None]
                               + group_sd*nodes[None, None, :])
        mu = means[None, :, None, :] + offsets[..., None]*direction
        terms = component_terms(mu, cov, (x @ A.T)[None, :, None, :])
        lt = norm.logpdf(x[None, :, 0, None], terms['mean'], np.sqrt(terms['variance']))
        base = np.broadcast_to(logw, (40, 2, 40)).copy()
        grouped = grouped_log_integrals(lt, terms['log_condition_density'], row_group, base)
        result = shared_conditional_logpdf(grouped, logw)
        full_cov = np.kron(np.eye(3), cov)
        loading = np.tile(direction, 3)
        full_cov += shared_sd**2*np.outer(loading, loading)
        for group in (0, 1):
            loading = ((row_group == group)[:, None]*direction).ravel()
            full_cov += group_sd**2*np.outer(loading, loading)
        project = np.kron(np.eye(3), A)
        full = multivariate_normal.logpdf(x.ravel(), means.ravel(), full_cov)
        covariates = multivariate_normal.logpdf((x @ A.T).ravel(),
            (means @ A.T).ravel(), project @ full_cov @ project.T)
        self.assertAlmostEqual(result['log_joint'], full, places=10)
        self.assertAlmostEqual(result['log_conditions'], covariates, places=10)
        self.assertAlmostEqual(result['logpdf'], full-covariates, places=10)
        # Source amplitudes may depend on b and group: they still cancel
        # BEFORE marginalizing b. They are not extra observations of b.
        arbitrary = np.arange(80).reshape(40, 2, 1)*17.
        rescaled = grouped_log_integrals(lt, terms['log_condition_density'], row_group,
                                         base + arbitrary)
        other = shared_conditional_logpdf(rescaled, logw + 200.)
        self.assertAlmostEqual(other['logpdf'], result['logpdf'], places=10)
        perm = np.array([2, 0, 1])
        reordered = grouped_log_integrals(lt[:, perm], terms['log_condition_density'][:, perm],
                                          row_group[perm], base)
        self.assertAlmostEqual(shared_conditional_logpdf(reordered, logw)['logpdf'],
                               result['logpdf'], places=12)

    def test_shared_prior_must_be_conditioned_on_all_groups(self):
        # An exactly enumerable 2-node law: both groups' U inform ONE b.
        lu = np.log(np.array([[.2, .3], [.8, .6]]))[:, :, None]
        lt = np.log(np.array([[.9, .4], [.3, .5]]))[:, :, None]
        groups = grouped_log_integrals(lt, lu, np.array([0, 1]), np.zeros((2, 2, 1)))
        result = shared_conditional_logpdf(groups, np.log([.4, .6]))
        expected_weights = np.array([.4*.2*.3, .6*.8*.6])
        expected_weights /= expected_weights.sum()
        np.testing.assert_allclose(np.exp(result['log_shared_given_conditions']), expected_weights)
        expected = np.log(expected_weights @ np.array([.9*.4, .3*.5]))
        self.assertAlmostEqual(result['logpdf'], expected, places=12)
        wrong = np.log(np.array([.4, .6]) @ np.array([.9*.4, .3*.5]))
        self.assertGreater(abs(expected-wrong), .3)

    def test_members_share_one_local_distance_not_independent_mixtures(self):
        lt = np.log(np.array([[[.9, .1], [.8, .2]]]))
        lu = np.zeros_like(lt)
        grouped = grouped_log_integrals(lt, lu, np.array([0, 0]), np.zeros((1, 1, 2)))
        result = shared_conditional_logpdf(grouped, np.zeros(1))
        self.assertAlmostEqual(np.exp(result['logpdf']), (.9*.8 + .1*.2)/2, places=12)
        self.assertNotAlmostEqual(np.exp(result['logpdf']), .5*.5, places=3)

    def test_grouped_invalid_inputs_and_zero_support(self):
        zeros = np.zeros((2, 2, 2))
        for groups in (np.array([0., 1.]), np.array([0, 2]), np.array([0, 0])):
            with self.assertRaises(ValueError):
                grouped_log_integrals(zeros, zeros, groups, zeros)
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                grouped_log_integrals(zeros, zeros, np.array([0, 1]), zeros + value)
        g = dict(log_joint=np.array([[-np.inf], [0.]]),
                 log_conditions=np.array([[-np.inf], [0.]]))
        result = shared_conditional_logpdf(g, [0., 0.])
        self.assertEqual(result['logpdf'], 0.)
        np.testing.assert_array_equal(np.exp(result['log_shared_given_conditions']), [0., 1.])
        g['log_joint'][:] = -np.inf
        self.assertEqual(shared_conditional_logpdf(g, [0., 0.])['logpdf'], -np.inf)
        g['log_conditions'][:] = -np.inf
        with self.assertRaisesRegex(ValueError, 'no joint support'):
            shared_conditional_logpdf(g, [0., 0.])


if __name__ == '__main__':
    unittest.main()
