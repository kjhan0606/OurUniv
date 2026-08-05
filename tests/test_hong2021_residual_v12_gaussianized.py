from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_residual_v8_context import FEATURE_NAMES, ObservableContextUNet
from hong2021_residual_v12_gaussianized import (
    CACHE_SCHEMA,
    TRANSFORM_SCHEMA,
    V12ResidualDataset,
    gaussianize_numpy,
    initialize_v11_parent,
    inverse_gaussianize_numpy,
    inverse_gaussianize_torch,
    load_transform,
)


def _transform() -> dict:
    z = np.linspace(-5.0, 5.0, 101)
    residual = np.sinh(z / 3.0) * 0.2
    return {
        "schema": TRANSFORM_SCHEMA,
        "z_knots": z.tolist(),
        "residual_value_knots": residual.tolist(),
    }


def test_gaussianization_round_trip_and_torch_inverse() -> None:
    transform = _transform()
    value = np.linspace(-0.35, 0.35, 200, dtype=np.float32)
    latent = gaussianize_numpy(value, transform)
    restored = inverse_gaussianize_numpy(latent, transform)
    np.testing.assert_allclose(restored, value, atol=2.0e-5, rtol=0)
    actual = inverse_gaussianize_torch(
        torch.from_numpy(latent),
        torch.tensor(transform["z_knots"]),
        torch.tensor(transform["residual_value_knots"]),
    )
    torch.testing.assert_close(actual, torch.from_numpy(restored), atol=1e-7, rtol=0)


def test_load_transform_rejects_nonmonotone_knots(tmp_path: Path) -> None:
    value = _transform()
    value["residual_value_knots"][50] = value["residual_value_knots"][49]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    try:
        load_transform(path)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("nonmonotone transform was accepted")


def test_v12_dataset_returns_latent_and_corrected_mean(tmp_path: Path) -> None:
    data_path = tmp_path / "data.h5"
    cache_path = tmp_path / "cache.h5"
    rng = np.random.default_rng(8)
    observable = rng.normal(size=(2, 2, 8, 8, 8)).astype(np.float32)
    truth = rng.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
    mean = rng.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
    latent = rng.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("input", data=observable)
        handle.create_dataset("target", data=truth)
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("conditional_mean", data=mean)
        handle.create_dataset("gaussianized_residual", data=latent)
        handle.attrs["schema"] = CACHE_SCHEMA
        handle.attrs["input_preprocessing"] = json.dumps({"mode": "faithful"})
    dataset = V12ResidualDataset(data_path, cache_path, 1.0, False)
    condition, actual_latent, actual_mean, actual_truth = dataset[1]
    assert condition.shape == (4, 8, 8, 8)
    torch.testing.assert_close(condition[2:3], actual_mean)
    torch.testing.assert_close(actual_latent, torch.from_numpy(latent[1]))
    torch.testing.assert_close(actual_truth, torch.from_numpy(truth[1]))


def test_v11_parent_loading_retains_new_context_moments() -> None:
    torch.manual_seed(9)
    parent = ObservableContextUNet(base_channels=8)
    mean = torch.arange(len(FEATURE_NAMES), dtype=torch.float32) + 20
    std = torch.arange(1, len(FEATURE_NAMES) + 1, dtype=torch.float32)
    child = ObservableContextUNet(base_channels=8, context_mean=mean, context_std=std)
    result = initialize_v11_parent(child, parent.state_dict())
    assert set(result["retained_new_buffers"]) == {"context_mean", "context_std"}
    torch.testing.assert_close(child.context_mean, mean)
    torch.testing.assert_close(child.context_std, std)
    torch.testing.assert_close(child.output.weight, parent.output.weight)
