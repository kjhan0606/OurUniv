import numpy as np
from scipy.stats import multivariate_normal
from cf4_r2_cross_method import (link_nonfp_rows, closed_group_holdout,
    relative_system, fit_relative_at_variance, heldout_relative_score)


def test_shared_covariance_gls_and_prediction():
    y = np.array([.2, -.1, .4, .1, -.3])
    sigma = np.array([.1, .3, .2, .4, .2])
    group, method = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 0, 1])
    vf, extra = np.array([.04, .01, .03]), .02
    x, inv, info, rhs, logdet = relative_system(y, sigma, group, vf, method, 2, extra)
    covariance = np.diag(sigma**2)
    for g in range(3):
        a = (group == g).astype(float)
        covariance += (vf[g]+extra)*np.outer(a, a)
    ci = np.linalg.inv(covariance)
    np.testing.assert_allclose(inv(np.column_stack((y, x))), ci@np.column_stack((y, x)), atol=1e-12)
    np.testing.assert_allclose(info, x.T@ci@x, atol=1e-12)
    np.testing.assert_allclose(rhs, x.T@ci@y, atol=1e-12)
    np.testing.assert_allclose(logdet, np.linalg.slogdet(covariance)[1], atol=1e-12)
    fit = fit_relative_at_variance(y, sigma, group, vf, method, 2, extra)
    beta = np.linalg.solve(x.T@ci@x, x.T@ci@y)
    np.testing.assert_allclose(fit['offset'], beta, atol=1e-12)
    # Synthetic new groups can have the same local index numbering; this test
    # is only the predictive algebra, not a real train/holdout data split.
    predictive = heldout_relative_score(y, sigma, group, vf, method, fit, extra)
    expected = multivariate_normal.logpdf(y, mean=x@beta,
        cov=covariance+x@np.linalg.inv(info)@x.T)
    np.testing.assert_allclose(predictive['logpdf'], expected, atol=1e-12)


def test_measurement_ownership_and_common_mode():
    def row(p, t, cf, rec, **marks):
        return dict(PGC=str(p), T17=str(t), **{'1PGC':str(cf)}, recno=str(rec),
                    CF3='1', DM='99', DMfp='88', **marks)
    rows = [row(1, 2, 10, 1, DMtf='31', e_DMtf='.4'),
            row(3, 2, 11, 2, DMsnIa='32', e_DMsnIa='.2'),
            row(4, 0, 10, 3, DMtf='33', e_DMtf='.3'),
            row(5, 9, 12, 4, DMtf='34', e_DMtf='.3')]
    anchors, issues = link_nonfp_rows(rows, {1:'T2', 5:'T3'}, ['T2', 'T3'])
    assert len(anchors) == 2 and [a['method'] for a in anchors] == ['tf', 'snIa']
    assert len(issues) == 1 and issues[0]['reason'] == 'conflicting_PGC_T17'
    held = closed_group_holdout(['T2', 'T3'], [10, 11], [False, True], anchors)
    assert held == {'T2':True, 'T3':True}  # new anchor closes the bridge
    contrast = np.array([[1., -1., 0.], [0., 1., -1.]])
    np.testing.assert_array_equal(contrast@np.ones(3), np.zeros(2))
    c = np.diag([.01, .02, .03])
    np.testing.assert_allclose(contrast@(c+.123*np.ones((3, 3)))@contrast.T,
                               contrast@c@contrast.T, atol=1e-15)


if __name__ == '__main__':
    test_shared_covariance_gls_and_prediction()
    test_measurement_ownership_and_common_mode()
    print('PASS: correlated GLS/prediction, exact ownership, split closure, common-mode null', flush=True)
