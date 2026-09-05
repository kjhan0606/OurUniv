from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_residual_v8_context import FEATURE_NAMES, ObservableContextUNet
from hong2021_residual_v11_recentered import (
    CACHE_SCHEMA,
    V11ResidualDataset,
    balanced_residual_scale,
    centered_residual,
    initialize_recentered_parent,
)


def test_centered_residual_has_exact_cube_dc_null() -> None:
    torch.manual_seed(31)
    truth = torch.randn(3, 1, 8, 8, 8)
    mean = torch.randn(3, 1, 8, 8, 8)
    value = centered_residual(truth, mean)
    torch.testing.assert_close(
        value.mean(dim=(-3, -2, -1)),
        torch.zeros(3, 1),
        atol=1.0e-7,
        rtol=0.0,
    )


def _write_data_and_cache(root: Path, rms: float) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    data = root / f"data_{rms}.h5"
    cache = root / f"cache_{rms}.h5"
    generator = np.random.default_rng(17)
    observable = generator.normal(size=(2, 2, 8, 8, 8)).astype(np.float32)
    truth = generator.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
    mean = generator.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
    residual = truth - mean
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    with h5py.File(data, "w") as handle:
        handle.create_dataset("input", data=observable)
        handle.create_dataset("target", data=truth)
        handle.attrs["voxel_mpc_h"] = 0.3125
    with h5py.File(cache, "w") as handle:
        handle.create_dataset("conditional_mean", data=mean)
        handle.create_dataset("centered_residual", data=residual)
        handle.attrs["schema"] = CACHE_SCHEMA
        handle.attrs["input_preprocessing"] = json.dumps({"mode": "faithful"})
        handle.attrs["residual_rms"] = rms
        handle.attrs["correction_checkpoint"] = "/frozen/v10.pt"
    return data, cache


def test_v11_dataset_uses_corrected_mean_and_full_residual(tmp_path: Path) -> None:
    data, cache = _write_data_and_cache(tmp_path, 0.25)
    dataset = V11ResidualDataset(data, cache, residual_scale=0.25, augment=False)
    condition, residual, mean, truth = dataset[0]
    assert condition.shape == (4, 8, 8, 8)
    torch.testing.assert_close(condition[2:3], mean)
    torch.testing.assert_close(
        residual.mean(dim=(-3, -2, -1)), torch.zeros(1), atol=2.0e-7, rtol=0
    )
    torch.testing.assert_close(
        residual * 0.25,
        centered_residual(truth[None], mean[None])[0],
        atol=2.0e-7,
        rtol=1.0e-6,
    )


def test_balanced_scale_is_equal_source_second_moment(tmp_path: Path) -> None:
    _, tng = _write_data_and_cache(tmp_path / "tng", 0.2)
    _, simba = _write_data_and_cache(tmp_path / "simba", 0.4)
    fit = balanced_residual_scale(tng, simba)
    assert np.isclose(fit["balanced_rms"], np.sqrt(0.5 * (0.2**2 + 0.4**2)))


def test_parent_loading_preserves_new_context_standardization() -> None:
    torch.manual_seed(41)
    parent = ObservableContextUNet(base_channels=8)
    expected_mean = torch.arange(len(FEATURE_NAMES), dtype=torch.float32) + 10
    expected_std = torch.arange(1, len(FEATURE_NAMES) + 1, dtype=torch.float32)
    child = ObservableContextUNet(
        base_channels=8, context_mean=expected_mean, context_std=expected_std
    )
    result = initialize_recentered_parent(child, parent.state_dict())
    assert set(result["retained_new_buffers"]) == {"context_mean", "context_std"}
    torch.testing.assert_close(child.context_mean, expected_mean)
    torch.testing.assert_close(child.context_std, expected_std)
    torch.testing.assert_close(child.input.weight, parent.input.weight)
