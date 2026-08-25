import numpy as np
import pytest

from cf4_lg_highk_conditional_field import (
    add_restriction_adjoint_preserve_dtype,
    conditional_field,
    prolong_white_spectrum,
    restrict_spectrum_preserve_dtype,
)
from cf4_peak_evidence import prepare_exact_peak_operator
from cf4_projection_contract import add_restriction_adjoint, restrict_spectrum


def hermitian_filter(n: int) -> np.ndarray:
    k = np.fft.fftfreq(n) * n
    k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    return np.exp(-0.08 * k2).astype(np.complex128)


def test_dtype_preserving_projection_matches_reference():
    rng = np.random.default_rng(3)
    field = rng.normal(size=(12, 12, 12))
    spectrum = np.fft.fftn(field, norm="ortho")
    target = np.fft.fftn(rng.normal(size=(4, 4, 4)), norm="ortho")
    np.testing.assert_allclose(
        restrict_spectrum_preserve_dtype(spectrum, 4),
        restrict_spectrum(spectrum, 4),
        atol=2e-13,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        add_restriction_adjoint_preserve_dtype(spectrum, target),
        add_restriction_adjoint(spectrum, target),
        atol=2e-13,
        rtol=2e-13,
    )


def test_float32_prolongation_preserves_coarse_field():
    coarse = np.random.default_rng(5).normal(size=(4, 4, 4)).astype(np.float32)
    fine_fft, coarse_fft = prolong_white_spectrum(
        coarse, 12, 71, float_dtype=np.float32, workers=1
    )
    restricted = restrict_spectrum_preserve_dtype(fine_fft, 4)
    error = np.sqrt(np.mean(np.abs(restricted - coarse_fft) ** 2))
    scale = np.sqrt(np.mean(np.abs(coarse_fft) ** 2))
    assert error / scale < 2e-6


def test_conditional_field_matches_exact_float64_reference():
    n, coarse_n = 12, 4
    filt = hermitian_filter(n)
    points = np.asarray([[1, 2, 3], [7, 8, 9], [4, 6, 2]])
    targets = np.asarray([0.4, -0.1, 0.7])
    coarse = np.random.default_rng(9).normal(size=(coarse_n,) * 3)
    operator = prepare_exact_peak_operator(filt, coarse_n, points, 0.25)
    expected, expected_meta = operator.conditional_sample(
        coarse, targets, fine_seed=81, noise_seed=91
    )
    actual, actual_meta = conditional_field(
        coarse,
        filt,
        points,
        targets,
        0.25,
        fine_seed=81,
        noise_seed=91,
        signal_covariance=operator.signal_covariance,
        float_dtype=np.float64,
        workers=1,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        actual_meta["achieved_after"], expected_meta["achieved_after"],
        atol=2e-6, rtol=2e-6,
    )
    assert actual_meta["coarse_roundtrip_relative_RMS"] < 1e-12
    assert actual_meta["correction_restriction_relative_RMS"] < 1e-12
    assert actual_meta["maximum_response_identity_error"] < 1e-12
    assert actual_meta["null_subspace_mean_square"] > 0.0
    assert actual_meta["peak_evidence_reapplied"] is False


def test_unbounded_fft_workers_are_rejected():
    coarse = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="positive explicit cap"):
        prolong_white_spectrum(coarse, 12, 1, workers=-1)
