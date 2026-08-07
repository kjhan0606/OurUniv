from __future__ import annotations

import numpy as np

from hong2021_v14_multiscale import inverse_standardized_residual
from scripts.hong2021_v24_tail_mechanism_audit import (
    forward_latent_diagnostics,
    recover_v14_standardized,
)


def test_recover_v14_standardized_inverts_stored_multiscale_member() -> None:
    generator = np.random.default_rng(240031)
    standardized = generator.normal(size=(8, 8, 8)).astype(np.float32)
    standardized -= standardized.mean(dtype=np.float64)
    scales = np.asarray([0.7, 1.2, 2.1, 3.0], dtype=np.float64)
    location = 0.13
    corrected_mean = generator.normal(0.0, 0.1, size=(8, 8, 8)).astype(np.float32)
    physical_residual = inverse_standardized_residual(
        standardized,
        predicted_location=location,
        predicted_scales=scales,
        voxel_mpc_h=0.3125,
    )
    sample = corrected_mean + physical_residual
    stored_mean = corrected_mean + np.float32(location)
    recovered = recover_v14_standardized(
        sample, stored_mean, scales, voxel_mpc_h=0.3125
    )
    assert np.allclose(recovered, standardized, rtol=0.0, atol=2.0e-6)


def test_forward_latent_diagnostics_counts_both_support_endpoints() -> None:
    standardized = np.asarray(
        [[[-3.0, -0.5], [0.5, 3.0]], [[0.0, 0.9], [-0.9, 0.0]]],
        dtype=np.float32,
    )
    mean = np.zeros_like(standardized)
    profile = {
        "centers": [-1.0, 1.0],
        "mu": [0.0, 0.0],
        "log_sigma": [0.0, 0.0],
    }
    transform = {
        "residual_value_knots": [-1.0, 0.0, 1.0],
        "z_knots": [-5.0, 0.0, 5.0],
    }
    u, latent, support = forward_latent_diagnostics(
        standardized, mean, profile, transform
    )
    assert np.array_equal(u, standardized)
    assert support["low_at_or_outside_count"] == 1
    assert support["high_at_or_outside_count"] == 1
    assert support["raw_latent_minimum"] == -5.0
    assert support["raw_latent_maximum"] == 5.0
    assert abs(float(latent.sum(dtype=np.float64))) / np.sqrt(latent.size) <= 1.0e-9
