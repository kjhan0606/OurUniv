import unittest

import numpy as np
from scipy.special import ndtr

from cf4_r2_raw_selected_fp import (log_box_probability, log_cdf_interval,
    optical_error_covariance, optical_cut_geometry, selected_mark_logpdf, rule)


def normalized_example(eta=(-.1, .15)):
    t, w = rule(64)
    r, k = np.meshgrid(-.2+.5*t, 10+t, indexing='ij')
    x = np.stack((r, np.zeros_like(r), np.zeros_like(r)), axis=-1)
    magnitude = k[..., None]-np.array([33., 33.3])
    args = (x, magnitude, np.log([.25, .75]), np.array(eta),
            np.array([-23., -23.3]), np.array([-22., -22.3]),
            np.zeros(3), np.array([.12, 0., 0.]), np.diag([.04, .01, .0225]),
            np.array([[1., 0., 0.], [0., 1., 0.]]), np.array([-.2, -1.]), np.array([.3, 1.]))
    density = np.exp(selected_mark_logpdf(*args))*(2*np.pi*.1*.15)
    integral = .5*np.sum(density*w[:, None]*w[None, :])
    return integral, r, k, density, args


class RawSelectedTests(unittest.TestCase):
    def test_correlated_not_product(self):
        cov = np.array([[1., .5], [.5, 1.]])
        actual = np.exp(log_box_probability(np.zeros(2), cov,
                                             [-np.inf, -np.inf], [0., 0.], order=192))
        self.assertAlmostEqual(float(actual), 1/3, delta=2e-6)
        self.assertGreater(abs(actual-.25), .08)

    def test_positive_tail_and_reflection(self):
        self.assertAlmostEqual(float(log_cdf_interval(10., 11.)),
                               float(log_cdf_interval(-11., -10.)), places=12)
        cov = np.diag([1., 1.])
        got = log_box_probability(np.zeros(2), cov, [10., -1.], [11., 2.])
        expected = log_cdf_interval(10., 11.)+np.log(ndtr(2.)-ndtr(-1.))
        self.assertAlmostEqual(float(got), float(expected), places=11)

    def test_selected_joint_normalization(self):
        for eta in ((-.1, .15), (-.4, .3), (0., 0.)):
            self.assertAlmostEqual(float(normalized_example(eta)[0]), 1., delta=2e-7)

    def test_common_weight_scale_and_support(self):
        *_, args = normalized_example()
        args = list(args)
        baseline = selected_mark_logpdf(*args)
        args[2] = args[2]+100.
        np.testing.assert_allclose(selected_mark_logpdf(*args), baseline, atol=2e-13)
        args[0] = np.array([5., 0., 0.])
        args[1] = np.array([-22.5, -22.8])
        self.assertTrue(np.isneginf(selected_mark_logpdf(*args)))

    def test_component_terms_reconstruct_selected_density(self):
        from scipy.special import logsumexp
        *_, args=normalized_example()
        numerator,denominator=selected_mark_logpdf(*args,return_component_terms=True)
        value=logsumexp(args[2]+numerator,axis=-1)-logsumexp(args[2]+denominator)
        np.testing.assert_allclose(value,selected_mark_logpdf(*args),atol=1e-13)

    def test_source_covariance_and_aperture_geometry(self):
        cov = optical_error_covariance(.01, .02, .02)
        self.assertEqual(cov[0, 2], -.0002)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-15)
        x = np.array([[.3, 2.2, 2.8], [.4, 2.3, 2.6]])
        matrix, lo, hi = optical_cut_geometry(x, [15., 16.], [150., 200.])
        self.assertTrue(((x@matrix.T >= lo) & (x@matrix.T <= hi)).all())
        _, lo_bad, _ = optical_cut_geometry(x[:1], [18.], [150.])
        self.assertLess((x[:1]@matrix.T)[0, 0], lo_bad[0, 0])

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            log_box_probability([0., 0.], [[1., 1.], [1., 1.]], [-1., -1.], [1., 1.])
        with self.assertRaises(ValueError):
            optical_error_covariance(0., .02, .01)


if __name__ == '__main__':
    unittest.main()
