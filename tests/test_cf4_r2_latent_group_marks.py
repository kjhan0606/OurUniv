import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal, norm

from cf4_r2_fp_group_marginal import (latent_central_logmark,
    conditional_latent_group_scores, conditional_group_scores)


def test_exact_roles_and_zero_support():
    sat = np.array([[.2, 0., 0.], [.3, .5, 0.], [.6, .8, .5], [.4, .6, 0.]])
    cen = np.array([[.9, .8, .7], [.4, .6, .5], [.3, .8, .7], [.6, .9, .8]])
    gid = np.array([0, 0, 0, 1])
    w0, wi = np.array([.3, .7]), np.array([.2, .4, .1, .3])
    with np.errstate(divide='ignore'):
        logsat, logcen = jnp.array(np.log(sat)), jnp.array(np.log(cen))
    actual = latent_central_logmark(logsat, logcen, jnp.array(gid), jnp.log(w0), jnp.log(wi))
    direct = []
    for g in range(2):
        rows = np.flatnonzero(gid == g)
        val = w0[g]*np.prod(sat[rows], axis=0)
        for i in rows:
            val += wi[i]*cen[i]*np.prod(sat[rows[rows != i]], axis=0)
        direct.append(val/(w0[g]+wi[rows].sum()))
    np.testing.assert_allclose(jnp.exp(actual), direct, atol=1e-14)
    assert np.isneginf(actual[0, 2])  # no invented probability floor
    # Scaling ALL hypotheses of each group cancels; row reordering does too.
    scale = jnp.array([2., 7.])
    scaled = latent_central_logmark(logsat, logcen, jnp.array(gid),
        jnp.log(w0)+scale, jnp.log(wi)+scale[gid])
    np.testing.assert_allclose(scaled, actual, atol=1e-14)
    order = np.array([3, 1, 0, 2])
    perm = latent_central_logmark(logsat[order], logcen[order], jnp.array(gid[order]),
                                  jnp.log(w0), jnp.log(wi[order]))
    np.testing.assert_allclose(perm, actual, atol=1e-14)
    f = lambda a: jnp.exp(latent_central_logmark(logsat, logcen+a,
                        jnp.array(gid), jnp.log(w0), jnp.log(wi))).sum()
    np.testing.assert_allclose(jax.grad(f)(0.), (f(1e-5)-f(-1e-5))/2e-5, rtol=1e-9)
    # Exactly zero role prior weights are legal too.
    none_only = latent_central_logmark(logsat, logcen, jnp.array(gid),
        jnp.zeros(2), jnp.full(4, -jnp.inf))
    np.testing.assert_allclose(jnp.exp(none_only), [sat[:3].prod(0), sat[3]], atol=1e-14)


def test_zero_effect_identity_and_conditional_normalization():
    d = jnp.array([[20., 30., 40.], [25., 35., 45.]])
    w = jnp.log(jnp.array([[.2, .5, .3], [.3, .4, .3]]))
    kernel = jnp.array([[-2., 0., -3.], [-1., -2., 0.]])
    args = (jnp.array([0, 0, 1]), jnp.array([30., 30., 35.]),
        jnp.array([.04, .06, -.02]), jnp.array([.1, .12, .08]),
        jnp.array([0., 2., -1.]), jnp.array([-.02, .02]))
    roles = (jnp.log(jnp.array([.4, .4])), jnp.log(jnp.array([.3, .3, .6])))
    offsets = (jnp.array([0.]), jnp.array([0.]))
    actual = conditional_latent_group_scores(d, w, kernel, *args, *roles, 0., 0., *offsets)
    baseline = conditional_group_scores(d, w, kernel, *args)
    np.testing.assert_allclose(actual, baseline, atol=2e-13)
    shifted = conditional_latent_group_scores(d, w+17., kernel+19., *args,
                                              *roles, 0., 0., *offsets)
    np.testing.assert_allclose(shifted, baseline, atol=2e-13)
    # Changed role offsets must use the SAME eta=0 source reference, otherwise
    # branch-dependent constants incorrectly reweight the central hypotheses.
    f = lambda delta: conditional_latent_group_scores(d, w, kernel, *args,
                                          *roles, delta, -.003, *offsets).sum()
    np.testing.assert_allclose(jax.grad(f)(.005), (f(.005001)-f(.004999))/2e-6,
                               rtol=1e-7, atol=1e-8)


def test_shared_offset_gaussian_reference():
    mean, std = np.array([.08, -.03]), np.array([.07, .11])
    eta, b, scatter = .04, .01, .05
    x, weights = np.polynomial.hermite.hermgauss(32)
    actual = conditional_latent_group_scores(jnp.array([[30.]]), jnp.zeros((1, 1)),
        jnp.zeros((1, 1)), jnp.zeros(2, dtype=int), jnp.full(2, 30.*10**eta),
        jnp.array(mean), jnp.array(std), jnp.zeros(2), jnp.array([b]),
        jnp.log(jnp.array([.5])), jnp.log(jnp.array([.25, .25])), 0., 0.,
        jnp.array(np.sqrt(2.)*scatter*x), jnp.log(weights))
    cov = np.diag(std**2)+scatter**2*np.ones((2, 2))
    reference = multivariate_normal.logpdf(mean, mean=np.full(2, eta+b), cov=cov)
    reference -= norm.logpdf(0., loc=mean, scale=std).sum()
    np.testing.assert_allclose(actual, reference, atol=2e-12)
    independent = norm.logpdf(mean, loc=eta+b, scale=np.sqrt(std**2+scatter**2)).sum()
    independent -= norm.logpdf(0., loc=mean, scale=std).sum()
    assert abs(float(actual[0, 0])-independent) > .01


if __name__ == '__main__':
    test_exact_roles_and_zero_support()
    test_zero_effect_identity_and_conditional_normalization()
    test_shared_offset_gaussian_reference()
    print('PASS: exact latent roles, zero support, identity, shared covariance and derivatives', flush=True)
