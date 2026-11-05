import numpy as np

from cf4_peak_evidence import (
    apply_filter,
    engineering_metadata,
    minimum_norm_fine_mean,
    normalized_gaussian_logpdf,
    null_project_field,
    prepare_exact_peak_operator,
)
from cf4_projection_contract import restrict_spectrum, restrict_white_field


def smooth_filter(n):
    frequency = np.fft.fftfreq(n) * n
    k2 = (
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    result = np.exp(-0.15 * k2)
    result[0, 0, 0] = 0.0
    return result


def dense_restriction(fine_n, coarse_n):
    rows = []
    for cell in range(fine_n ** 3):
        basis = np.zeros((fine_n, fine_n, fine_n))
        basis.flat[cell] = 1.0
        restricted_fft = restrict_spectrum(
            np.fft.fftn(basis, norm="ortho"), coarse_n
        )
        rows.append(np.fft.ifftn(restricted_fft, norm="ortho").real.ravel())
    return np.asarray(rows).T


def dense_observation(fine_n, filter_full, points):
    rows = []
    for cell in range(fine_n ** 3):
        basis = np.zeros((fine_n, fine_n, fine_n))
        basis.flat[cell] = 1.0
        smoothed = apply_filter(basis, filter_full)
        rows.append(smoothed[tuple(points.T)])
    return np.asarray(rows).T


def test_null_projector_matches_dense_I_minus_Rstar_R():
    rng = np.random.default_rng(53)
    fine_n, coarse_n = 4, 2
    restriction = dense_restriction(fine_n, coarse_n)
    dense_q = np.eye(fine_n ** 3) - restriction.T @ restriction
    field = rng.standard_normal((fine_n, fine_n, fine_n))
    projected = null_project_field(field, coarse_n)
    np.testing.assert_allclose(
        projected.ravel(), dense_q @ field.ravel(), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(dense_q @ dense_q, dense_q, rtol=2e-14, atol=2e-14)
    assert np.sqrt(np.mean(restrict_white_field(projected, coarse_n) ** 2)) < 1e-7


def test_exact_peak_mean_and_covariance_match_dense_gaussian_model():
    rng = np.random.default_rng(59)
    fine_n, coarse_n = 4, 2
    points = np.asarray([[0, 0, 0], [1, 2, 3], [3, 1, 2]])
    filt = smooth_filter(fine_n)
    sigma = np.asarray([0.2, 0.3, 0.4])
    operator = prepare_exact_peak_operator(filt, coarse_n, points, sigma)

    restriction = dense_restriction(fine_n, coarse_n)
    observation = dense_observation(fine_n, filt, points)
    dense_q = np.eye(fine_n ** 3) - restriction.T @ restriction
    expected_covariance = observation @ dense_q @ observation.T
    np.testing.assert_allclose(
        operator.signal_covariance, expected_covariance,
        rtol=2e-13, atol=2e-14,
    )
    np.testing.assert_allclose(
        operator.observation_covariance,
        expected_covariance + np.diag(sigma ** 2),
        rtol=2e-13, atol=2e-14,
    )

    coarse = rng.standard_normal((coarse_n, coarse_n, coarse_n))
    expected_mean = observation @ restriction.T @ coarse.ravel()
    np.testing.assert_allclose(
        operator.predict_parent(coarse), expected_mean,
        rtol=2e-13, atol=2e-14,
    )
    np.testing.assert_allclose(
        minimum_norm_fine_mean(coarse, fine_n).ravel(),
        restriction.T @ coarse.ravel(),
        rtol=2e-14, atol=2e-14,
    )


def test_normalized_log_evidence_keeps_determinant_and_constant():
    value = np.asarray([0.3, -0.2])
    mean = np.asarray([-0.1, 0.4])
    covariance = np.asarray([[1.2, 0.2], [0.2, 0.7]])
    actual, terms = normalized_gaussian_logpdf(value, mean, covariance)
    residual = value - mean
    sign, logdet = np.linalg.slogdet(covariance)
    expected = -0.5 * (
        2 * np.log(2.0 * np.pi)
        + logdet
        + residual @ np.linalg.solve(covariance, residual)
    )
    assert sign == 1.0
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(terms["log_determinant"], logdet)
    assert terms["normalization_dimension"] == 2


def test_matheron_conditional_samples_match_dense_observation_moments():
    rng = np.random.default_rng(61)
    fine_n, coarse_n = 4, 2
    points = np.asarray([[0, 0, 0], [1, 2, 3]])
    filt = smooth_filter(fine_n)
    operator = prepare_exact_peak_operator(filt, coarse_n, points, 0.35)
    coarse = rng.standard_normal((coarse_n, coarse_n, coarse_n)).astype(np.float32)
    targets = np.asarray([0.7, -0.4])
    parent_mean = operator.predict_parent(coarse)
    signal = operator.signal_covariance
    total = operator.observation_covariance
    expected_mean = parent_mean + signal @ np.linalg.solve(
        total, targets - parent_mean
    )
    expected_covariance = signal - signal @ np.linalg.solve(total, signal)

    achieved = []
    maximum_roundtrip = 0.0
    for sample in range(1200):
        _, metadata = operator.conditional_sample(
            coarse, targets, 10000 + sample, 20000 + sample
        )
        achieved.append(metadata["achieved_after"])
        maximum_roundtrip = max(
            maximum_roundtrip,
            metadata["roundtrip"]["maximum_normalized_error"],
        )
    achieved = np.asarray(achieved)
    np.testing.assert_allclose(np.mean(achieved, axis=0), expected_mean, atol=0.025)
    np.testing.assert_allclose(
        np.cov(achieved, rowvar=False), expected_covariance, atol=0.025
    )
    assert maximum_roundtrip < 1e-6


def test_peak_evidence_is_independent_of_conditional_random_seeds():
    coarse = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 8.0
    points = np.asarray([[0, 0, 0], [2, 1, 3]])
    operator = prepare_exact_peak_operator(smooth_filter(4), 2, points, 0.25)
    targets = np.asarray([0.3, 0.8])
    _, first = operator.conditional_sample(coarse, targets, 71, 81)
    _, second = operator.conditional_sample(coarse, targets, 72, 82)
    assert first["normalized_log_evidence"] == second["normalized_log_evidence"]
    np.testing.assert_array_equal(
        first["evidence_terms"]["predicted_mean"],
        second["evidence_terms"]["predicted_mean"],
    )


def test_engineering_reference_forbids_candidate_generation():
    metadata = engineering_metadata()
    assert metadata["null_projector"] == "Q=I-R*R"
    assert "log determinant" in metadata["evidence"]
    assert "no workers=-1" in metadata["FFT"]
    assert metadata["candidate_generation_authorized"] is False
