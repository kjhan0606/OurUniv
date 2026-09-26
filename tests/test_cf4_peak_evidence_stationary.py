import numpy as np

from cf4_peak_evidence import prepare_exact_peak_operator
from cf4_peak_evidence_stationary import (
    approximation_metadata,
    covariance_at_points,
    stationary_null_mask_full,
    stationary_peak_log_evidence,
    stationary_signal_covariance_grid,
)


def gaussian_filter(n, radius):
    frequency = 2.0 * np.pi * np.fft.fftfreq(n)
    k2 = (
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    result = np.exp(-0.5 * radius ** 2 * k2)
    result[0, 0, 0] = 0.0
    return result


def test_stationary_mask_removes_only_strict_coarse_interior():
    mask = stationary_null_mask_full(12, 4)
    assert mask.shape == (12, 12, 12)
    assert np.count_nonzero(~mask) == 3 ** 3
    mapped = np.mod(np.asarray([-1, 0, 1]), 12)
    assert not np.any(mask[np.ix_(mapped, mapped, mapped)])
    assert mask[2, 0, 0]
    assert mask[10, 0, 0]


def test_stationary_covariance_grid_matches_direct_spectral_sum():
    n, coarse_n = 6, 2
    filt = gaussian_filter(n, 0.8)
    mask = stationary_null_mask_full(n, coarse_n)
    grid = stationary_signal_covariance_grid(filt, coarse_n)
    points = np.asarray([[0, 0, 0], [1, 2, 3], [5, 4, 1]])
    actual = covariance_at_points(grid, points)
    expected = np.empty_like(actual)
    frequencies = np.indices((n, n, n))
    for i, left in enumerate(points):
        for j, right in enumerate(points):
            phase = np.exp(
                2j * np.pi * sum(
                    frequencies[axis] * (left[axis] - right[axis]) / n
                    for axis in range(3)
                )
            )
            expected[i, j] = np.sum(
                np.abs(filt) ** 2 * mask * phase
            ).real / n ** 3
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_stationary_evidence_keeps_normalized_gaussian_terms():
    mean = np.asarray([0.2, -0.1])
    targets = np.asarray([0.5, 0.3])
    signal = np.asarray([[0.7, 0.1], [0.1, 0.4]])
    logpdf, terms = stationary_peak_log_evidence(
        mean, targets, signal, np.asarray([0.2, 0.4])
    )
    total = signal + np.diag([0.2 ** 2, 0.4 ** 2])
    residual = targets - mean
    expected = -0.5 * (
        2 * np.log(2.0 * np.pi)
        + np.linalg.slogdet(total)[1]
        + residual @ np.linalg.solve(total, residual)
    )
    np.testing.assert_allclose(logpdf, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(terms["observation_covariance"], total)
    assert "log_determinant" in terms


def test_gaussian_smoothing_does_not_rescue_stationary_approximation():
    fine_n, coarse_n = 12, 4
    # This engineering analogue has strong Gaussian suppression, yet exact Q
    # still produces material coarse-grid phase dependence.
    filt = gaussian_filter(fine_n, 2.0)
    points = np.asarray([
        [0, 0, 0], [1, 2, 3], [4, 5, 6], [11, 8, 2]
    ])
    exact = prepare_exact_peak_operator(filt, coarse_n, points, 0.25)
    grid = stationary_signal_covariance_grid(filt, coarse_n)
    approximate = covariance_at_points(grid, points)
    relative = np.linalg.norm(approximate - exact.signal_covariance) / np.linalg.norm(
        exact.signal_covariance
    )
    assert relative > 0.5
    assert np.ptp(np.diag(exact.signal_covariance)) > 1e-5
    np.testing.assert_allclose(
        np.diag(approximate), np.diag(approximate)[0], rtol=0.0, atol=1e-15
    )


def test_stationary_engine_is_rejected_for_all_parent_evidence():
    metadata = approximation_metadata()
    assert "rejected" in metadata["role"]
    assert "Nyquist" in metadata["approximation"]
    assert "order-unity" in metadata["rejection_reason"]
    assert metadata["all_parent_evidence_authorized"] is False
