from __future__ import annotations

import numpy as np
import torch

from hong2021_v21_conditional_affine import apply_profile, invert_profile, invert_profile_torch


def test_v21_profile_is_piecewise_linear_and_roundtrips() -> None:
    profile = {
        "centers": [0.0, 1.0, 2.0],
        "mu": [0.0, 1.0, 3.0],
        "sigma": [0.5, 1.0, 2.0],
    }
    mean = np.asarray([-1.0, 0.5, 1.5, 3.0], dtype=np.float32)
    residual = np.asarray([0.25, 1.5, 2.0, -1.0], dtype=np.float32)
    transformed = apply_profile(residual, mean, profile)
    recovered = invert_profile(transformed, mean, profile)
    assert transformed.dtype == np.float32
    assert np.max(np.abs(recovered.astype(np.float64) - residual)) < 2e-6


def test_v21_profile_extrapolates_constantly() -> None:
    profile = {"centers": [0.0, 1.0], "mu": [2.0, 4.0], "sigma": [1.0, 2.0]}
    value = apply_profile(np.asarray([3.0], dtype=np.float32), np.asarray([-5.0], dtype=np.float32), profile)
    assert np.array_equal(value, np.asarray([1.0], dtype=np.float32))


def test_v21_torch_inverse_matches_numpy_and_is_rng_free() -> None:
    profile = {
        "centers": [-1.0, 0.0, 2.0],
        "mu": [-0.5, 0.25, 1.0],
        "sigma": [0.5, 1.0, 1.5],
    }
    value = torch.tensor([-2.0, 0.5, 1.0, 3.0], dtype=torch.float32)
    mean = torch.tensor([-2.0, -0.5, 1.0, 3.0], dtype=torch.float32)
    generator = torch.Generator().manual_seed(210021)
    state = generator.get_state().clone()
    actual = invert_profile_torch(
        value, mean,
        torch.tensor(profile["centers"]), torch.tensor(profile["mu"]),
        torch.tensor(profile["sigma"]),
    )
    expected = invert_profile(value.numpy(), mean.numpy(), profile)
    assert np.allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(state, generator.get_state())
