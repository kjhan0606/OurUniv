import numpy as np
import pytest
from scipy.stats import multivariate_normal

from cf4_r2_conditional_group_mark import conditional_group_mark_logpdf


def test_conditional_equals_joint_minus_member_marginal():
    observed = np.array([5050.0, 34.2, 5010.0, 5080.0])
    predicted = np.array([5000.0, 34.0, 5005.0, 5020.0])
    loadings = np.array([[60.0, 0.0], [0.012, 0.04],
                         [70.0, 0.0], [50.0, 0.0]])
    covariance = np.diag([40.0**2, 0.15**2, 20.0**2, 25.0**2]) + loadings @ loadings.T
    actual = conditional_group_mark_logpdf(
        *observed[:2], observed[2:], *predicted[:2], predicted[2:],
        covariance, (10, 11))
    expected = (multivariate_normal.logpdf(observed, predicted, covariance)
                - multivariate_normal.logpdf(observed[2:], predicted[2:],
                                             covariance[2:, 2:]))
    assert actual == pytest.approx(expected, abs=1e-10)


def test_no_matched_member_is_group_joint_density():
    covariance = np.diag([80.0**2, 0.2**2])
    actual = conditional_group_mark_logpdf(5000, 34.1, (), 4990, 34.0, (),
                                            covariance, ())
    expected = multivariate_normal.logpdf([5000, 34.1], [4990, 34.0], covariance)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_duplicate_member_and_singular_covariance_fail():
    covariance = np.eye(4)
    with pytest.raises(ValueError, match='unique'):
        conditional_group_mark_logpdf(1, 2, (3, 4), 1, 2, (3, 4),
                                       covariance, (8, 8))
    with pytest.raises(ValueError, match='positive definite'):
        conditional_group_mark_logpdf(1, 2, (3,), 1, 2, (3,),
                                       np.zeros((3, 3)), (8,))
