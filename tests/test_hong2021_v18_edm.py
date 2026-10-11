from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import hong2021_residual_v6 as residual_v6
from hong2021_v18_edm import _validate_exact_experiment


class DummyModel(nn.Module):
    def forward(self, *args):
        raise AssertionError("the patched denoiser must be used")


def _legacy_sample(
    model, condition, generator, steps, sigma_min, sigma_max, rho, sigma_data
):
    sigmas = residual_v6.karras_sigmas(
        steps, sigma_min, sigma_max, rho, condition.device
    )
    value = torch.randn(
        (len(condition), 1) + tuple(condition.shape[-3:]),
        device=condition.device, generator=generator,
    ) * sigmas[0]
    for index in range(steps):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        sigma_batch = torch.full((len(condition),), float(sigma), device=condition.device)
        denoised = residual_v6.edm_denoise(
            model, value, condition, sigma_batch, sigma_data
        )
        derivative = (value - denoised) / sigma
        euler = value + (sigma_next - sigma) * derivative
        if sigma_next > 0:
            next_batch = torch.full(
                (len(condition),), float(sigma_next), device=condition.device
            )
            next_denoised = residual_v6.edm_denoise(
                model, euler, condition, next_batch, sigma_data
            )
            next_derivative = (euler - next_denoised) / sigma_next
            value = value + (sigma_next - sigma) * 0.5 * (
                derivative + next_derivative
            )
        else:
            value = euler
    return value


def test_v18_registry_exposes_only_frozen_e3_paired_sampler() -> None:
    registry = json.loads(Path("config/hong2021_v18_development_program.json").read_text())
    experiment = registry["e6_prior_matched_initialization"]
    _validate_exact_experiment(experiment)
    assert experiment["sampling_seeds"] == {
        "TNG100": 35777, "SIMBA": 36777, "Swift": 37777,
    }
    assert experiment["training_or_checkpoint_update"] is False


def test_v18_registry_rejects_variance_or_seed_mutation() -> None:
    registry = json.loads(Path("config/hong2021_v18_development_program.json").read_text())
    experiment = registry["e6_prior_matched_initialization"]
    changed = copy.deepcopy(experiment)
    changed["initialization"]["source_balanced_per_band_mode_variance"][0] += 1.0
    with pytest.raises(ValueError, match="variances"):
        _validate_exact_experiment(changed)
    changed = copy.deepcopy(experiment)
    changed["sampling_seeds"]["TNG100"] = 45777
    with pytest.raises(ValueError, match="sampling seeds"):
        _validate_exact_experiment(changed)


def test_sample_edm_none_initializer_is_legacy_bit_identical(monkeypatch) -> None:
    def denoise(model, value, condition, sigma, sigma_data):
        return value * 0.125 + condition[:, :1] * 0.01

    monkeypatch.setattr(residual_v6, "edm_denoise", denoise)
    condition = torch.linspace(-1, 1, 3 * 4**3).reshape(3, 1, 4, 4, 4)
    first = torch.Generator().manual_seed(180063)
    second = torch.Generator().manual_seed(180063)
    expected = _legacy_sample(DummyModel(), condition, first, 7, 0.01, 4.0, 3.0, 2.0)
    actual = residual_v6.sample_edm(
        DummyModel(), condition, second, 7, 0.01, 4.0, 3.0, 2.0
    )
    assert torch.equal(expected, actual)
    assert torch.equal(first.get_state(), second.get_state())


def test_prior_matched_init_recovers_linear_gaussian_variance(monkeypatch) -> None:
    variance = 1795.0

    def posterior_mean(model, value, condition, sigma, sigma_data):
        scale = variance / (variance + sigma.square())
        return value * scale[:, None, None, None, None]

    monkeypatch.setattr(residual_v6, "edm_denoise", posterior_mean)
    condition = torch.zeros((20000, 1, 1, 1, 1))
    legacy_generator = torch.Generator().manual_seed(180064)
    corrected_generator = torch.Generator().manual_seed(180064)
    legacy = residual_v6.sample_edm(
        DummyModel(), condition, legacy_generator, 40, 0.002, 40.0, 7.0, 2.0
    )
    sigma_first = float(
        residual_v6.karras_sigmas(40, 0.002, 40.0, 7.0, torch.device("cpu"))[0]
    )
    corrected = residual_v6.sample_edm(
        DummyModel(), condition, corrected_generator, 40, 0.002, 40.0, 7.0, 2.0,
        init_transform=lambda noise: noise * (sigma_first**2 + variance) ** 0.5,
    )
    legacy_ratio = float(legacy.var(unbiased=False) / variance)
    corrected_ratio = float(corrected.var(unbiased=False) / variance)
    expected_legacy = sigma_first**2 / (sigma_first**2 + variance)
    assert abs(legacy_ratio - expected_legacy) < 0.05
    assert abs(corrected_ratio - 1.0) < 0.05
    assert torch.equal(legacy_generator.get_state(), corrected_generator.get_state())
