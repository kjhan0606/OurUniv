import numpy as np

from hong2021_wiener_self_consistency import (
    fourier_geometry,
    posterior_parameters,
)


def test_fourier_geometry_counts_full_real_grid_modes():
    edges = np.asarray([0.01, 100.0])
    _, valid, weights, geometry = fourier_geometry(8, 1.0, edges)
    assert np.count_nonzero(~valid) == 1  # the deliberately excluded DC mode
    assert weights[valid].sum() == 8**3 - 1
    assert geometry[0, 1] == 8**3 - 1


def test_wiener_posterior_scalar_case():
    truth_power = np.asarray([4.0])
    transfer = np.asarray([0.5])
    noise = np.asarray([3.0])
    gain, variance = posterior_parameters(truth_power, transfer, noise)
    np.testing.assert_allclose(gain, [0.5])
    np.testing.assert_allclose(variance, [3.0])
