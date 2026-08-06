from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import hong2021_v18_edm as v18
import hong2021_v20_edm as v20
from hong2021_v14_edm import V20_E8_SCHEMA


def test_v20_registry_derives_e3_relative_noise_from_final_cache_rms() -> None:
    registry = json.loads(Path("config/hong2021_v20_development_program.json").read_text())
    experiment = registry["e8_gaussianized_marginal_retrain"]
    v20._validate_exact_experiment(experiment)
    assert experiment["training_noise"]["p_mean"] == v20.P_MEAN
    assert experiment["training_noise"]["p_std"] == v20.P_STD


@pytest.mark.parametrize("mutation", ("p_mean", "projection", "seed", "steps"))
def test_v20_exact_experiment_rejects_scientific_mutation(mutation: str) -> None:
    registry = json.loads(Path("config/hong2021_v20_development_program.json").read_text())
    experiment = copy.deepcopy(registry["e8_gaussianized_marginal_retrain"])
    if mutation == "p_mean":
        experiment["training_noise"]["p_mean"] += 0.01
    elif mutation == "projection":
        experiment["dc_projection"]["maximum_absolute_adjustment"] = 0.3
    elif mutation == "seed":
        experiment["sampler"]["sampling_seeds"]["TNG100"] += 1
    else:
        experiment["steps"] += 1
    with pytest.raises(ValueError):
        v20._validate_exact_experiment(experiment)


def test_v20_training_namespace_uses_relative_noise_and_new_caches(
    tmp_path, monkeypatch,
) -> None:
    registry = json.loads(Path("config/hong2021_v20_development_program.json").read_text())
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("stub")
    monkeypatch.setattr(v20, "load_frozen_registry", lambda path, repo: registry)
    monkeypatch.setattr(v20, "git_state", lambda repo: ("a" * 40, True))
    args = argparse.Namespace(
        repo=tmp_path, registry=registry_path, out=tmp_path / "out", device="cpu"
    )
    actual = v20.frozen_training_namespace(args)
    assert actual.run_schema == V20_E8_SCHEMA
    assert actual.edm_p_mean is None
    assert actual.edm_p_mean_sigma_data_fraction == 0.6
    assert actual.edm_p_std == 1.2
    assert actual.tng_train_cache.endswith("tng_train_gaussianized.h5")
    assert actual.steps == 10000
    assert actual.candidate_steps == "5000,10000"


def test_v20_checkpoint_accepts_the_training_emitted_relative_noise_mode(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "step_005000.pt"
    torch.save({
        "schema": V20_E8_SCHEMA,
        "step": 5000,
        "experiment_registry_sha256": v20.FROZEN_REGISTRY_SHA256,
        "worktree_clean_at_launch": True,
        "sigma_data": 0.9999915369331587,
        "edm_p_mean": v20.P_MEAN,
        "edm_p_std": v20.P_STD,
        "edm_p_mean_mode": "log_sigma_data_fraction",
        "edm_p_mean_sigma_data_fraction": 0.6,
        "decoder_upsampling": "nearest",
        "denoising_loss": {
            "coefficients": {"unweighted": 0.5, "tail_weighted": 0.5},
            "band_balanced": False,
        },
        "code_commit_at_launch": "a" * 40,
    }, checkpoint_path)
    registry = {
        "e8_gaussianized_marginal_retrain": {
            "initialization_and_normalization": {
                "sigma_data": 0.9999915369331587,
            },
        },
    }
    checkpoint, _ = v20._validate_checkpoint(
        checkpoint_path, step=5000, registry=registry
    )
    assert checkpoint["edm_p_mean_mode"] == "log_sigma_data_fraction"


class _Dataset:
    grid = 2
    voxel_mpc_h = 0.3125

    def __len__(self):
        return 1

    def __getitem__(self, index):
        condition = torch.zeros((1, 2, 2, 2))
        corrected = torch.zeros((1, 2, 2, 2))
        truth = torch.zeros((1, 2, 2, 2))
        return condition, torch.zeros_like(condition), corrected, truth

    def predicted_location_scales(self, index):
        return 0.0, np.ones(4, dtype=np.float32)


class _Initializer:
    maximum_observed_imaginary_ratio = 0.0


def test_v18_shared_writer_applies_v20_inverse_after_zero_dc(tmp_path, monkeypatch) -> None:
    def fake_sample(*args, **kwargs):
        return torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 2, 2)

    seen = []

    def fake_physical(value, **kwargs):
        seen.append(value.copy())
        return value

    monkeypatch.setattr(v18, "sample_edm", fake_sample)
    monkeypatch.setattr(v18, "inverse_standardized_residual", fake_physical)
    checkpoint = {"sigma_data": 1.0, "step": 5000, "schema": V20_E8_SCHEMA}
    output = tmp_path / "ensemble.h5"

    def inverse(value):
        return value + 2.0

    v18.write_prior_matched_ensemble(
        model=torch.nn.Identity(), dataset=_Dataset(), checkpoint=checkpoint,
        checkpoint_path=tmp_path / "checkpoint.pt", checkpoint_sha256="0" * 64,
        output=output, indices=[0], ensemble_members=2, sampling_steps=1,
        sigma_min=0.1, sigma_max=1.0, rho=1.0, seed=20,
        device=torch.device("cpu"), initializer=_Initializer(), sigma_first=1.0,
        metadata={}, progress_label="test", latent_inverse=inverse,
    )
    assert len(seen) == 2
    raw = fake_sample()[..., 0, :, :, :]
    centered = raw - raw.mean(dim=(-3, -2, -1), keepdim=True)
    expected = (centered + 2.0).numpy()
    assert np.array_equal(np.stack(seen), expected)
    with h5py.File(output, "r") as handle:
        assert bool(handle.attrs["complete"])
