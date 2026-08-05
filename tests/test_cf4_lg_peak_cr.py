import numpy as np

from src.cf4_lg_peak_cr import (
    condition_translated_constraints,
    free_rfft_mask,
    two_peak_points,
)


def test_free_mask_matches_coarse_embedding_support():
    free = free_rfft_mask(16, 8)
    assert free.shape == (16, 16, 9)
    assert not free[0, 0, 0]
    assert not free[3, 3, 3]
    assert free[4, 0, 0]  # skipped coarse Nyquist
    assert free[0, 0, 4]  # skipped coarse kz Nyquist


def test_conditioning_preserves_frozen_coefficients_and_moves_constraints():
    n, coarse = 16, 8
    rng = np.random.default_rng(4)
    base = rng.normal(size=(n, n, n))
    k = np.fft.fftfreq(n)[:, None, None] ** 2
    k = k + np.fft.fftfreq(n)[None, :, None] ** 2
    k = k + np.fft.rfftfreq(n)[None, None, :] ** 2
    filt = np.exp(-20.0 * k)
    free = free_rfft_mask(n, coarse)
    points = np.array([[7, 8, 8], [9, 8, 8]])
    before = np.fft.irfftn(
        np.fft.rfftn(base) * filt, s=base.shape, axes=(0, 1, 2))
    result, meta = condition_translated_constraints(
        base, filt, free, points, np.array([1.5, 1.5]), 0.1, 12)
    bk, rk = np.fft.rfftn(base), np.fft.rfftn(result)
    np.testing.assert_allclose(rk[~free], bk[~free], rtol=2e-5, atol=2e-5)
    after = np.fft.irfftn(rk * filt, s=base.shape, axes=(0, 1, 2))
    assert np.linalg.norm(after[tuple(points.T)] - 1.5) < np.linalg.norm(
        before[tuple(points.T)] - 1.5)
    assert meta["correction_rms"] > 0


def test_two_peak_geometry_has_fourteen_unique_probes():
    points, kinds = two_peak_points(
        64, np.array([32, 32, 32]), np.array([1.0, 1.0, 0.0]), 8, 2)
    assert points.shape == (14, 3)
    assert len(np.unique(points, axis=0)) == 14
    assert kinds.sum() == 2
