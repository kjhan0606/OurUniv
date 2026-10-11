from __future__ import annotations

import h5py
import numpy as np

from hong2021_v21_development_gate import conditional_diagnostics


def test_v21_q6_uses_recovered_corrected_mean_and_all_members(tmp_path) -> None:
    path = tmp_path / "ensemble.h5"
    grid = 4
    mean = np.linspace(-0.5, 0.5, grid**3, dtype=np.float32).reshape(grid, grid, grid)
    location = 0.1
    pattern = np.sin(np.arange(grid**3, dtype=np.float32)).reshape(grid, grid, grid) * 0.05
    truth = mean + location + pattern
    sample = np.stack([truth + 0.5 * pattern, truth - 0.5 * pattern], axis=0)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample", data=sample[None, :, None])
        handle.create_dataset("truth", data=truth[None, None])
        handle.create_dataset("conditional_mean", data=(mean + location)[None, None])
        handle.create_dataset("predicted_residual_dc", data=np.asarray([location], dtype=np.float32))
        handle.create_dataset("predicted_band_scales", data=np.ones((1, 4), dtype=np.float32))
    report = conditional_diagnostics(path, np.asarray([-1.0, 0.0, 1.0]))
    assert report["selection_role"] == "none"
    assert report["truth_voxels"] == [32, 32]
    assert report["generated_voxels"] == [64, 64]
    assert np.isfinite(report["generated_over_truth_std"]).all()
