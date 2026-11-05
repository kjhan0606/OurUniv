import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import skewnorm
from cf4_r2_fp_distance import fp_log_likelihood_ratio, skew_parameters, predicted_eta


def test_source_moments_and_ratio():
    m, s, a = jnp.array([-.1, .2, 0.]), jnp.array([.1, .2, .15]), jnp.array([-4., 0., 3.])
    loc, scale = skew_parameters(m, s, a)
    actual_m, actual_v = skewnorm.stats(np.asarray(a), loc=np.asarray(loc), scale=np.asarray(scale), moments='mv')
    np.testing.assert_allclose(actual_m, m, atol=1e-12)
    np.testing.assert_allclose(actual_v, s*s, atol=1e-12)
    eta = np.array([.04, -.03, .12])
    expected = skewnorm.logpdf(eta, a, loc=loc, scale=scale) - skewnorm.logpdf(0., a, loc=loc, scale=scale)
    np.testing.assert_allclose(fp_log_likelihood_ratio(eta, 0., m, s, a), expected, atol=1e-12)
    assert np.isneginf(fp_log_likelihood_ratio(2., 0., 0., .1, 0.))


def test_cold_root_and_gradient():
    z = jnp.linspace(0., .2, 10001)
    d = 2997.92458*z
    dirs = jnp.array([[1., 0., 0.], [0., 1., 0.]])
    v = jnp.ones((3, 8, 8, 8))*200.
    obs = jnp.array([.02, .03])
    eta, dist, residual = predicted_eta(v, dirs, obs, z, d)
    expected_z = (1+obs)/(1+200/299792.458)-1
    np.testing.assert_allclose(dist, expected_z*2997.92458, atol=1e-9)
    np.testing.assert_allclose(residual, 0., atol=1e-12)
    f = lambda amp: jnp.sum(predicted_eta(v, dirs, obs, z, d, amplitude=amp)[0])
    np.testing.assert_allclose(jax.grad(f)(1.), (f(1.0001)-f(.9999))/.0002, rtol=1e-6)
    np.testing.assert_allclose(predicted_eta(v*0, dirs, obs, z, d)[0], 0., atol=1e-12)


if __name__ == '__main__':
    test_source_moments_and_ratio()
    test_cold_root_and_gradient()
    print('PASS: source moments/ratios and same-state root/gradient', flush=True)
