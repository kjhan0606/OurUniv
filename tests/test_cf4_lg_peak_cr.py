import numpy as np

from src.cf4_lg_peak_cr import (
    condition_translated_constraints,
    draw_protohalo_midpoint_offset,
    free_rfft_mask,
    proposal_seed_rows,
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


def test_fixed_midpoint_is_backward_compatible():
    peak = {"protohalo_midpoint_offset_mpc_h": [0.0, -6.0, 4.0]}
    offset, metadata = draw_protohalo_midpoint_offset(peak, None)
    np.testing.assert_array_equal(offset, [0.0, -6.0, 4.0])
    assert metadata["mode"] == "fixed"


def test_latent_midpoint_is_seeded_and_uses_declared_prior():
    peak = {"protohalo_midpoint_prior": {
        "distribution": "diagonal_normal",
        "mean_mpc_h": [0.0, -6.0, 4.0],
        "sigma_mpc_h": [3.0, 2.0, 1.0],
    }}
    first, metadata = draw_protohalo_midpoint_offset(peak, 8501)
    second, _ = draw_protohalo_midpoint_offset(peak, 8501)
    different, _ = draw_protohalo_midpoint_offset(peak, 8502)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert metadata["midpoint_seed"] == 8501
    assert metadata["mode"] == "latent_diagonal_normal"
    assert metadata["log_target_prior_over_sampling_proposal"] == 0.0


def test_latent_midpoint_importance_mixture_records_exact_correction():
    prior = {
        "distribution": "diagonal_normal",
        "mean_mpc_h": [0.0, -6.0, 4.0],
        "sigma_mpc_h": [3.0, 3.0, 3.0],
    }
    peak = {
        "protohalo_midpoint_prior": prior,
        "protohalo_midpoint_sampling_proposal": {
            "distribution": "diagonal_normal_mixture",
            "components": [
                dict(prior, weight=0.5),
                {
                    "weight": 0.5,
                    "mean_mpc_h": [0.7, -6.4, 4.6],
                    "sigma_mpc_h": [1.5, 1.5, 1.5],
                },
            ],
        },
    }
    value, metadata = draw_protohalo_midpoint_offset(peak, 8569)
    repeated, _ = draw_protohalo_midpoint_offset(peak, 8569)
    np.testing.assert_array_equal(value, repeated)
    assert metadata["mode"] == "latent_importance_mixture"
    assert metadata["sampled_component_index"] in (0, 1)
    assert np.isclose(
        metadata["log_target_prior_over_sampling_proposal"],
        metadata["log_target_prior"] - metadata["log_sampling_proposal"],
    )


def test_proposal_seed_rows_rejects_silent_truncation():
    config = {
        "proposal_seeds": [1, 2],
        "geometry_seeds": [11],
        "likelihood_noise_seeds": [21, 22],
        "peak_constraints": {},
    }
    with np.testing.assert_raises(ValueError):
        proposal_seed_rows(config)


def test_proposal_seed_rows_binds_latent_midpoint_seed():
    config = {
        "proposal_seeds": [1, 2],
        "geometry_seeds": [11, 12],
        "likelihood_noise_seeds": [21, 22],
        "midpoint_seeds": [31, 32],
        "peak_constraints": {"protohalo_midpoint_prior": {
            "distribution": "diagonal_normal",
            "mean_mpc_h": [0.0, 0.0, 0.0],
            "sigma_mpc_h": 1.0,
        }},
    }
    assert proposal_seed_rows(config) == [
        (1, 11, 21, 31), (2, 12, 22, 32)
    ]


def test_proposal_seed_rows_expands_frozen_contiguous_bank():
    config = {
        "seed_bank": {
            "count": 3,
            "proposal_seed_start": 101,
            "geometry_seed_start": 201,
            "likelihood_noise_seed_start": 301,
            "midpoint_seed_start": 401,
        },
        "peak_constraints": {"protohalo_midpoint_prior": {
            "distribution": "diagonal_normal",
            "mean_mpc_h": [0.0, 0.0, 0.0],
            "sigma_mpc_h": 1.0,
        }},
    }
    assert proposal_seed_rows(config) == [
        (101, 201, 301, 401),
        (102, 202, 302, 402),
        (103, 203, 303, 403),
    ]
