import inspect
import unittest
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from cf4_lg_composite_proxy import (pair_context, fit, pair_means, pair_logpdf,
    score, latent_observables, latent_from_observables)
from cf4_lg_observation_contract import predict


class CompositeTests(unittest.TestCase):
    def setup_case(self):
        context = pair_context([[0, 0, 0], [1, .2, .1], [.5, 1, .3]],
            [[1, 2, 3], [40, -70, 20], [20, 50, -30]], [[20, 30, 40]]*3,
            [[0, 1], [0, 2]], dx=.1875, h=.6774)
        K = np.eye(4)+.2*np.ones((4, 4))
        model = dict(beta=np.array([.1, .2, -.1, .3]), K=K)
        return context, model, np.arange(12).reshape(4, 3)*15.

    def test_full_density_marginal_scales_and_mixture(self):
        c, m, y = self.setup_case()
        for rows in ((0, 1, 2, 3), (0, 2)):
            got = pair_logpdf(y, m, c, rows)
            expected = []
            for i in range(2):
                scale = np.repeat(c['scales'][i, list(rows)], 3)
                covariance = np.kron(m['K'][np.ix_(rows, rows)], np.eye(3))*scale[:, None]*scale[None]
                expected.append(multivariate_normal.logpdf(y[list(rows)].ravel(),
                    pair_means(m, c)[i, list(rows)].ravel(), covariance))
            np.testing.assert_allclose(got, expected, atol=1e-10)
        result = score(y, m, c)
        self.assertAlmostEqual(result['log_joint'], logsumexp(pair_logpdf(y, m, c))-np.log(2))
        self.assertAlmostEqual(result['log_M33_given_host'], result['log_joint']-result['log_host'])
        doubled = {k: np.concatenate([v, v]) if isinstance(v, np.ndarray) else v for k, v in c.items()}
        self.assertAlmostEqual(score(y, m, doubled)['log_joint'], result['log_joint'])
        np.testing.assert_allclose(c['scales'][:, :2], 1000*.1875/.6774)
        np.testing.assert_allclose(c['scales'][:, 2], np.sqrt(10000+2*np.mean(np.array([20,30,40])**2)))

    def test_fit_and_field_only_interface(self):
        self.assertEqual(set(inspect.signature(pair_context).parameters),
                         {'positions', 'mean_velocity', 'physical_sigma', 'pairs', 'dx', 'h'})
        c, _, _ = self.setup_case()
        rng = np.random.default_rng(190)
        y = rng.normal(size=(13, 4, 3))
        e = np.tile(c['direction'][0], (13, 1))
        m = fit(y, e)
        np.testing.assert_allclose(m['beta'], np.einsum('nga,na->ng', y, e).mean(0))
        # Same physical scale conversion in fitting and scoring.
        target = c['base'][0]+c['scales'][0, :, None]*y[0]
        got = pair_logpdf(target, m, c)[0]
        centered = (y[0]-m['beta'][:, None]*e[0]).ravel()
        want = multivariate_normal.logpdf(centered, cov=np.kron(m['K'], np.eye(3)))
        self.assertAlmostEqual(got, want-3*np.log(c['scales'][0]).sum())
        empty = pair_context(
            np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), [], dx=.1875, h=.6774)
        self.assertEqual(score(target, m, empty)['status'], 'UNRESOLVED_SUPPORT')

    def test_mock_observables_roundtrip_and_resolved_guard(self):
        y = np.array([[730., 120., 100.], [40., -120., 80.], [-70., 80., 40.], [20., -30., 60.]])
        kwargs = dict(h=.6774, solar_position_kpc=np.array([-8., 1., 2.]),
                      solar_velocity_km_s=np.array([15., 244., 5.]))
        obs = latent_observables(y, **kwargs)
        np.testing.assert_allclose(latent_from_observables(obs, **kwargs), y, atol=1e-10)
        with self.assertRaises(ValueError):
            predict({'resolved_halos': False}, {}, **kwargs)


if __name__ == '__main__':
    unittest.main()
