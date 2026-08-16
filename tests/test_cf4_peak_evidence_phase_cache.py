import numpy as np

from cf4_peak_evidence import prepare_exact_peak_operator
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    impulse_spectrum,
    parent_mean_at_point_sets,
    phase_cache_metadata,
    phase_response_grid,
)


def gaussian_filter(n, radius=1.4):
    frequency = 2.0 * np.pi * np.fft.fftfreq(n)
    k2 = (
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    result = np.exp(-0.5 * radius ** 2 * k2)
    result[0, 0, 0] = 0.0
    return result


def test_analytic_impulse_spectrum_matches_unitary_fft():
    n = 12
    point = np.asarray([7, 2, 11])
    impulse = np.zeros((n, n, n))
    impulse[tuple(point)] = 1.0
    np.testing.assert_allclose(
        impulse_spectrum(n, point),
        np.fft.fftn(impulse, norm="ortho"),
        rtol=2e-14, atol=2e-14,
    )


def test_phase_response_grid_matches_exact_operator_column():
    n, coarse_n = 12, 4
    filt = gaussian_filter(n)
    phase = np.asarray([2, 1, 0])
    points = np.asarray([
        phase,
        [5, 7, 9],
        [11, 2, 4],
        [1, 10, 8],
    ])
    exact = prepare_exact_peak_operator(filt, coarse_n, points, 0.25)
    grid = phase_response_grid(filt, coarse_n, phase)
    expected_column = grid[tuple(points.T)]
    np.testing.assert_allclose(
        expected_column, exact.signal_covariance[:, 0],
        rtol=2e-12, atol=2e-14,
    )


def test_phase_cache_matches_exact_covariance_for_multiple_geometries():
    n, coarse_n = 12, 4
    filt = gaussian_filter(n)
    point_sets = [
        np.asarray([[0, 0, 0], [1, 2, 3], [11, 7, 5], [4, 8, 2]]),
        np.asarray([[5, 5, 5], [8, 2, 11], [3, 9, 6], [10, 1, 4]]),
    ]
    cached, metadata = covariance_for_point_sets(filt, coarse_n, point_sets)
    for points, covariance in zip(point_sets, cached):
        exact = prepare_exact_peak_operator(filt, coarse_n, points, 0.3)
        np.testing.assert_allclose(
            covariance, exact.signal_covariance, rtol=3e-12, atol=3e-14
        )
    assert metadata["refinement_ratio"] == 3
    assert 1 < metadata["phase_count_used"] <= 27
    assert metadata["response_grids_held_simultaneously"] == 1
    assert metadata["maximum_pre_symmetrization_asymmetry"] < 1e-12


def test_parent_mean_uses_one_exact_field_for_all_point_sets():
    rng = np.random.default_rng(67)
    n, coarse_n = 12, 4
    filt = gaussian_filter(n)
    coarse = rng.standard_normal((coarse_n, coarse_n, coarse_n))
    point_sets = [
        np.asarray([[0, 0, 0], [1, 2, 3], [11, 7, 5]]),
        np.asarray([[5, 5, 5], [8, 2, 11], [3, 9, 6]]),
    ]
    cached = parent_mean_at_point_sets(coarse, filt, point_sets)
    for points, mean in zip(point_sets, cached):
        exact = prepare_exact_peak_operator(filt, coarse_n, points, 0.2)
        np.testing.assert_allclose(
            mean, exact.predict_parent(coarse), rtol=2e-12, atol=2e-14
        )


def test_phase_cache_is_exact_but_not_yet_full_size_authorized():
    metadata = phase_cache_metadata()
    assert metadata["covariance"] == "exact AQA*; no stationary approximation"
    assert metadata["production_phase_count"] == 27
    assert metadata["memory_policy"] == "one Nfine response grid at a time"
    assert "no workers=-1" in metadata["FFT_workers"]
    assert metadata["all_parent_evidence_authorized"] is False
