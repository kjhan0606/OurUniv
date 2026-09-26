import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from cf4_r2_fp_group_marginal import (redshift_sufficient, joint_redshift_logkernel,
    conditional_group_scores, shared_zero_logfactor)


def test_correlated_redshift_and_ownership():
    y = np.array([3000., 3050.])
    c = np.array([[10000., 8000.], [8000., 40000.]])
    s = redshift_sufficient(y, c, ['group:T1', '2mpp:4'])
    actual = joint_redshift_logkernel(jnp.array([[3010.]]), jnp.array([[.01]]), jnp.array(s[None]))
    expected = multivariate_normal.logpdf(y, mean=np.full(2, 3010.), cov=c*1.01**2)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    try:
        redshift_sufficient(y, c, ['same', 'same'])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate redshift accepted')


def test_conditional_normalization_and_shared_zero():
    distance = jnp.array([[20., 30., 40.], [25., 35., 45.]])
    logw = jnp.log(jnp.array([[.2, .5, .3], [.3, .4, .3]]))
    kernel = jnp.array([[-2., 0., -3.], [-1., -2., 0.]])
    args = (jnp.array([0, 0, 1]), jnp.array([30., 30., 35.]),
            jnp.array([.04, .06, -.02]), jnp.array([.1, .12, .08]), jnp.zeros(3), jnp.array([-.02, .02]))
    s = conditional_group_scores(distance, logw, kernel, *args)
    shifted = conditional_group_scores(distance, logw, kernel+17., *args)
    np.testing.assert_allclose(s, shifted, atol=1e-12)
    b = np.array([-.02, .02])
    direct = []
    for zero in b:
        groups = []
        for g, members in ((0, [0, 1]), (1, [2])):
            marks = sum(-.5*((np.log10(float(args[1][i])/np.asarray(distance[g]))+zero-float(args[2][i]))/float(args[3][i]))**2
                        +.5*(float(args[2][i])/float(args[3][i]))**2 for i in members)
            base = np.asarray(logw[g]+kernel[g])
            groups.append(logsumexp(base+marks)-logsumexp(base))
        direct.append(groups)
    np.testing.assert_allclose(s, direct, atol=1e-12)
    result = shared_zero_logfactor(s, jnp.log(jnp.array([.5,.5])), jnp.ones(2,dtype=bool))
    np.testing.assert_allclose(result, logsumexp(np.sum(direct,axis=1))-np.log(2), atol=1e-12)
    independent = sum(logsumexp(np.asarray(s)[:,g])-np.log(2) for g in range(2))
    assert abs(float(result)-independent) > 1e-5
    f = lambda x: shared_zero_logfactor(conditional_group_scores(distance, logw, kernel*x, *args), jnp.zeros(2), jnp.ones(2,dtype=bool))
    np.testing.assert_allclose(jax.grad(f)(1.), (f(1.0001)-f(.9999))/.0002, rtol=1e-6, atol=1e-9)


if __name__ == '__main__':
    test_correlated_redshift_and_ownership()
    test_conditional_normalization_and_shared_zero()
    print('PASS: joint redshift ownership, conditional denominator, shared zero and gradient', flush=True)
