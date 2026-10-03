import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from cf4_r2_fp_group_marginal import (nonfp_modulus_logmarks,
    conditional_latent_group_scores, conditional_group_scores)


def test_same_distance_mark_connection():
    d = jnp.array([[20., 30., 40.]])
    # Only an explicit synthetic distance convention for this algebra check.
    mu = 5*jnp.log10(d/.746)+25
    groups, methods = jnp.array([0, 0]), jnp.array([0, 1])
    obs, err, offsets = jnp.array([32.7, 33.1]), jnp.array([.3, .2]), jnp.array([.1, -.1])
    extra = nonfp_modulus_logmarks(mu, groups, obs, err, methods, offsets)
    direct = sum(-.5*((float(obs[i])-np.asarray(mu[0])-float(offsets[i]))**2
                 -(float(obs[i])-35.)**2)/float(err[i])**2 for i in range(2))
    np.testing.assert_allclose(extra[0], direct, atol=1e-12)
    args = (d, jnp.zeros_like(d), jnp.zeros_like(d), jnp.array([0]),
            jnp.array([30.]), jnp.array([.03]), jnp.array([.1]),
            jnp.array([0.]), jnp.array([0.]))
    roles = (jnp.log(jnp.array([.5])), jnp.log(jnp.array([.5])),
             0., 0., jnp.array([0.]), jnp.array([0.]))
    actual = conditional_latent_group_scores(*args, *roles, extra_log_marks=extra)
    old = conditional_group_scores(*args, extra_log_marks=extra)
    np.testing.assert_allclose(actual, old, atol=1e-12)
    fp = -.5*((np.log10(30./np.asarray(d[0]))-.03)**2-.03**2)/.1**2
    np.testing.assert_allclose(actual[0, 0], logsumexp(fp+direct)-np.log(3.), atol=1e-12)
    wrong = logsumexp(fp)-np.log(3.)+logsumexp(direct)-np.log(3.)
    assert abs(float(actual[0, 0])-wrong) > 1e-3
    f = lambda a: conditional_group_scores(*args, extra_log_marks=
        nonfp_modulus_logmarks(mu+a, groups, obs, err, methods, offsets)).sum()
    np.testing.assert_allclose(jax.grad(f)(0.), (f(1e-5)-f(-1e-5))/2e-5, rtol=1e-8)


if __name__ == '__main__':
    test_same_distance_mark_connection()
    print('PASS: non-FP marks inside SAME distance integral, fixed reference and gradient', flush=True)
