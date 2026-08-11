import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hong2021_v70_network import LatentSpatialUNet, edm_loss, parameter_count
from hong2021_v70_latent_cache import fit_and_holdout_indices
from hong2021_v70_preflight import (
    PROGRAM_SHA256,
    gaussianize_rank,
    representation_summary,
)


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v70_latent_spatial_score_model_program.json"


def test_v70_program_is_byte_bound_and_pair_free() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["joint_spatial_model"]["pair_or_2PCF_loss"] is False
    assert program["joint_spatial_model"]["Fourier_or_Ak_training_loss"] is False
    assert program["firewall"]["independent_gate_locked"] is True
    assert program["fixed_training"]["checkpoint_selection"] == (
        "fixed step-30000 EMA only"
    )


def test_gaussianized_rank_is_finite_and_roundtrips_after_frozen_clamp() -> None:
    rank = np.asarray([0.0, 1.0e-6, 0.5, 1.0 - 1.0e-6, 1.0])
    latent, restored, count = gaussianize_rank(rank)
    clipped = np.clip(rank, 1.0e-7, 1.0 - 1.0e-7)
    assert count == 2
    assert np.isfinite(latent).all()
    assert np.max(np.abs(restored - clipped)) < 2.0e-7


def test_representation_summary_records_clamps_and_moments() -> None:
    rank = np.asarray([[[[[0.0, 0.5, 1.0]]]]])
    latent, restored, _ = gaussianize_rank(rank)
    row = representation_summary([rank], [latent], [restored])
    assert row["objects"] == 1
    assert row["voxels"] == 3
    assert row["rank_clamp_count"] == 2
    assert row["rank_clamp_fraction"] == pytest.approx(2.0 / 3.0)
    assert row["maximum_normal_CDF_roundtrip_error"] < 2.0e-7


def test_latent_spatial_network_has_three_scale_attention_and_gradients() -> None:
    torch.manual_seed(170070)
    model = LatentSpatialUNet(base_channels=8, time_channels=32)
    latent = torch.randn(2, 1, 16, 16, 16)
    condition = torch.randn(2, 7, 16, 16, 16)
    sigma = torch.tensor([0.1, 1.0])
    loss, per_object = edm_loss(
        model, latent, condition, sigma, torch.randn_like(latent)
    )
    loss.backward()
    assert per_object.shape == (2,)
    assert torch.isfinite(loss)
    assert parameter_count(model) > 500_000
    assert model.attention.heads == 4
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_latent_spatial_network_rejects_missing_condition_channel() -> None:
    model = LatentSpatialUNet(base_channels=8, time_channels=32)
    with pytest.raises(ValueError, match="condition shape"):
        model(
            torch.zeros(1, 1, 16, 16, 16),
            torch.zeros(1, 6, 16, 16, 16),
            torch.zeros(1),
        )


def test_fit_partition_excludes_exact_mechanism_holdout() -> None:
    fit, held = fit_and_holdout_indices(8, [1, 3, 7])
    assert held.tolist() == [1, 3, 7]
    assert fit.tolist() == [0, 2, 4, 5, 6]
    assert np.intersect1d(fit, held).size == 0


def test_fit_partition_rejects_duplicate_holdout() -> None:
    with pytest.raises(ValueError, match="holdout"):
        fit_and_holdout_indices(8, [1, 1, 3])
