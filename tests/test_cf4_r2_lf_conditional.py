import numpy as np
from scipy.integrate import quad

from cf4_r2_lf_conditional import (conditional_logpdf, conditional_pit,
                                   observed_k_intervals)


def test_normalized_conditional_lf_and_cdf():
    lower = np.array([-24.9,-23.5])
    upper = np.array([-23.2,-22.1])
    middle = (lower+upper)/2
    for row in range(2):
        integral = quad(lambda m: np.exp(conditional_logpdf(
            np.array([m]),lower[row],upper[row],-.94,-23.28)[0]),
            lower[row],upper[row])[0]
        np.testing.assert_allclose(integral,1.,rtol=1e-12)
    pit = conditional_pit(middle,lower,upper,-.94,-23.28)
    assert np.all((pit > 0) & (pit < 1))


def test_observed_class_intervals():
    lo,hi = observed_k_intervals(np.array([34.,34.]),np.array([0,1]))
    np.testing.assert_allclose(lo,[-25.,-22.5])
    np.testing.assert_allclose(hi,[-22.5,-21.5])
