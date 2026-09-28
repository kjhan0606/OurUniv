import numpy as np

from cf4_r2_velocity_residual import (
    residual_summary,
    tsc_deposit_2x_moments,
    tsc_interpolate_periodic,
    tsc_stencil,
)


def test_cell_centred_tsc_weights_and_periodic_gather():
    pos = np.array([[0.75, 1.5, 2.25], [74.25, 73.5, 72.75]])
    _indices, weights = tsc_stencil(pos, 75.0, 25)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-14)
    grid = np.empty((3, 25, 25, 25), dtype=np.float64)
    grid[0], grid[1], grid[2] = 12.0, -4.0, 0.25
    got = tsc_interpolate_periodic(grid, pos, 75.0)
    np.testing.assert_allclose(got, [[12.0, -4.0, .25]] * len(pos), atol=1e-13)
    np.testing.assert_allclose(
        tsc_interpolate_periodic(grid, pos + 75.0, 75.0), got, atol=1e-13)


def test_tsc_moment_deposit_conserves_all_seven_channels():
    rng = np.random.default_rng(31)
    mass = rng.uniform(.5, 2.0, size=(50, 50, 50))
    velocity = rng.normal(size=(3, 50, 50, 50))
    moments = np.concatenate((mass[None], mass[None] * velocity,
                              mass[None] * velocity**2), axis=0)
    got = tsc_deposit_2x_moments(moments)
    assert got.shape == (7, 25, 25, 25)
    np.testing.assert_allclose(got.sum(axis=(1, 2, 3)),
                               moments.sum(axis=(1, 2, 3)), rtol=2e-14, atol=2e-10)
    assert np.all(got[0] > 0)

    impulse = np.zeros((7, 50, 50, 50), dtype=np.float64)
    impulse[0, 0, 0, 0] = 8.0
    deposited = tsc_deposit_2x_moments(impulse)[0]
    np.testing.assert_allclose(deposited[24, 24, 24], 8.0 * .28125**3, atol=1e-14)
    np.testing.assert_allclose(deposited[0, 0, 0], 8.0 * .6875**3, atol=1e-14)
    np.testing.assert_allclose(deposited.sum(), 8.0, atol=1e-14)


def test_residual_summary_separates_bias_and_dispersion():
    residual = np.array([[101.0, -3.0, 7.0], [99.0, 3.0, -7.0]])
    summary = residual_summary(residual)
    np.testing.assert_allclose(summary["mean_km_s"], [100.0, 0.0, 0.0])
    np.testing.assert_allclose(summary["component_sigma_km_s"], [1.0, 3.0, 7.0])
    np.testing.assert_allclose(summary["isotropic_1d_sigma_km_s"], np.sqrt(59 / 3))
