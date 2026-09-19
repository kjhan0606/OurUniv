from __future__ import annotations

import h5py
import numpy as np

from hong2021_v14_multiscale import standardize_residual
from hong2021_v21_conditional_affine import transform_cube
from hong2021_v22_development_gate import latent_conditional_diagnostics


def test_v22_latent_q6_uses_exact_source_indices_and_all_members(tmp_path) -> None:
    ensemble_path = tmp_path / "ensemble.h5"
    cache_path = tmp_path / "cache.h5"
    grid = 4
    mean = np.linspace(-0.5, 0.5, grid**3, dtype=np.float32).reshape(grid, grid, grid)
    pattern = 0.05 * np.sin(np.arange(grid**3)).reshape(grid, grid, grid)
    location = 0.1
    truth = mean + location + pattern
    scales = np.ones(4)
    profile = {
        "edges": [-1.0, 0.0, 1.0], "centers": [-0.5, 0.5],
        "mu": [0.0, 0.0], "sigma": [1.0, 1.0], "log_sigma": [0.0, 0.0],
    }
    transform = {"z_knots": [-5.0, 0.0, 5.0], "residual_value_knots": [-5.0, 0.0, 5.0]}
    _, residual = standardize_residual(truth - mean, predicted_scales=scales, voxel_mpc_h=0.3125)
    truth_latent, _, _ = transform_cube(residual, mean, profile, transform)
    with h5py.File(cache_path, "w") as handle:
        values = np.zeros((3, 1, grid, grid, grid), dtype=np.float32)
        values[2, 0] = truth_latent
        handle.create_dataset("standardized_residual", data=values)
    with h5py.File(ensemble_path, "w") as handle:
        handle.create_dataset("source_index", data=np.asarray([2]))
        handle.create_dataset("sample", data=np.stack([truth, truth], axis=0)[None, :, None])
        handle.create_dataset("conditional_mean", data=(mean + location)[None, None])
        handle.create_dataset("predicted_residual_dc", data=np.asarray([location]))
        handle.create_dataset("predicted_band_scales", data=scales[None])
    report = latent_conditional_diagnostics(ensemble_path, cache_path, profile, transform)
    assert report["maximum_absolute_generated_minus_truth_mean"] < 1e-6
    assert np.allclose(report["generated_over_truth_std"], 1.0, atol=1e-6)
