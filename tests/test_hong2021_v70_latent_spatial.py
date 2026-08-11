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
from hong2021_v70_train import (
    EMA_DECAY,
    LEARNING_RATE,
    MINIMUM_LEARNING_RATE,
    STEPS,
    learning_rate,
    update_ema,
)
from hong2021_v70_train_gate import (
    PROGRAM_SHA256 as TRAIN_GATE_PROGRAM_SHA256,
    fourier_energy_score,
    project_residual_dc,
    sigma_schedule,
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


def test_gaussianized_rank_projects_only_documented_float32_CDF_roundoff() -> None:
    rank = np.asarray([-(2.0**-24), 1.0 + 2.0**-23], dtype=np.float64)
    latent, restored, count = gaussianize_rank(rank)
    assert count == 2
    assert np.isfinite(latent).all()
    assert np.all((restored > 0.0) & (restored < 1.0))
    with pytest.raises(ValueError, match="conditional ranks"):
        gaussianize_rank(np.asarray([1.0 + 1.0e-5]))


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


def test_preflight_requires_every_parameter_gradient_tensor() -> None:
    source = (REPO / "src/hong2021_v70_preflight.py").read_text()
    assert 'model["every_parameter_gradient_present"]' in source
    assert 'model["every_parameter_gradient_tensor_finite"]' in source
    assert 'model["every_parameter_gradient_tensor_nonzero"]' in source


def test_gradient_recheck_preserves_the_false_positive() -> None:
    source = (
        REPO / "scripts/hong2021_v70_preflight_gradient_recheck_lageunha.sh"
    ).read_text()
    assert "attempt1_invalid_unscaled_amp_gradient" in source
    assert "preflight_false_positive.json" in source
    assert "only 194/8771649 scalar gradients nonzero" in source


def test_latent_cache_runner_binds_corrected_preflight_and_no_validation() -> None:
    source = (REPO / "scripts/hong2021_v70_latent_cache_lageunha.sh").read_text()
    assert "5b708473534954ff45f19ae0711249dd2d7305fa7288458467b71a78b853a3c4" in source
    assert "hong2021_v70_latent_cache.py" in source
    assert "validation" not in source


def test_latent_cache_roundoff_recovery_preserves_failed_partial() -> None:
    source = (
        REPO / "scripts/hong2021_v70_latent_cache_roundoff_recovery_lageunha.sh"
    ).read_text()
    assert "cache_attempt1_strict_CDF_range" in source
    assert "train_latent.h5.partial" in source
    assert "1.000000119" in source
    assert "within 5e-7" in source


def test_fixed_training_schedule_has_frozen_endpoints() -> None:
    assert STEPS == 30_000
    assert learning_rate(1) == pytest.approx(LEARNING_RATE)
    assert learning_rate(STEPS) == pytest.approx(MINIMUM_LEARNING_RATE)
    assert learning_rate(STEPS // 2) > MINIMUM_LEARNING_RATE
    with pytest.raises(ValueError, match="step"):
        learning_rate(0)


def test_fixed_ema_update_uses_frozen_decay() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    ema = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
        ema.weight.fill_(1.0)
    update_ema(ema, model)
    assert torch.allclose(
        ema.weight,
        torch.full_like(ema.weight, EMA_DECAY + (1.0 - EMA_DECAY) * 2.0),
    )


def test_training_runner_binds_cache_and_fixed_fit() -> None:
    source = (REPO / "scripts/hong2021_v70_train_lageunha.sh").read_text()
    assert "3419206ce239546d7a2742ead01f20c9e6495c311dda0e4b82da6944a799ef76" in source
    assert "0ddc9a592bc0eb1ab08d11ce71a5da1864b1fedb241663b2cc9f309094943ad3" in source
    assert "--resume" not in source


def test_training_source_has_no_validation_or_checkpoint_selection_path() -> None:
    source = (REPO / "src/hong2021_v70_train.py").read_text()
    assert '_open_split(v35["development_domains"][domain], "train")' in source
    assert '_open_split(v35["development_domains"][domain], "validation")' not in source
    assert "minimum_validation" not in source


def test_train_gate_program_freezes_two_paired_streams_during_training() -> None:
    program = json.loads(
        (
            REPO / "config/hong2021_v70_train_joint_structure_gate_program.json"
        ).read_text()
    )
    streams = program["noise_streams"]
    assert streams["stream_A_seed"] != streams["stream_B_seed"]
    assert streams["members_per_query_and_stream"] == 16
    assert streams["inference_batch"] == 4
    assert program["sampler"]["steps"] == 40
    assert program["firewall"]["independent_gate_locked"] is True


def test_train_gate_is_pair_loss_free_and_phase_sensitive() -> None:
    program = json.loads(
        (
            REPO / "config/hong2021_v70_train_joint_structure_gate_program.json"
        ).read_text()
    )
    assert "complex" in program["fourier_measurement"]["phase_sensitive_energy_score"]
    assert "strictly lower" in program["selection_rules"]["phase_sensitive_pass"]
    assert program["resource_gate"]["training_or_refit_by_gate"] is False


def test_train_gate_program_is_byte_bound() -> None:
    path = REPO / "config/hong2021_v70_train_joint_structure_gate_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TRAIN_GATE_PROGRAM_SHA256


def test_heun_schedule_has_frozen_endpoints_and_terminal_zero() -> None:
    schedule = sigma_schedule()
    assert schedule.shape == (41,)
    assert float(schedule[0]) == pytest.approx(40.0)
    assert float(schedule[-2]) == pytest.approx(0.002)
    assert float(schedule[-1]) == 0.0
    assert torch.all(schedule[:-1][1:] < schedule[:-1][:-1])


def test_fourier_energy_score_rewards_exact_phase_amplitude_field() -> None:
    truth = np.asarray([1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 1.0j])
    exact = np.stack((truth, truth))
    displaced = np.stack((truth + 1.0j, truth + 1.0j))
    selected = np.asarray([True, True, True])
    assert fourier_energy_score(exact, truth, selected) == pytest.approx(0.0)
    assert fourier_energy_score(displaced, truth, selected) > 0.0


def test_residual_DC_projection_is_exact_and_finite() -> None:
    value = np.arange(2 * 4 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4, 4)
    residual, maximum = project_residual_dc(value)
    assert np.isfinite(residual).all()
    assert maximum < 1.0e-12
    assert np.max(np.abs(residual.mean(axis=(-3, -2, -1), dtype=np.float64))) < 1.0e-6


def test_train_gate_runner_requires_completed_fixed_fit() -> None:
    source = (REPO / "scripts/hong2021_v70_train_gate_lageunha.sh").read_text()
    assert "complete_fixed_30000_step_fit" in source
    assert "train_only_mechanism_gate_run" in source
    assert "validation_accessed" in source
    assert "hong2021_v70_train_gate.py" in source


def test_training_supervisor_advances_only_after_exact_completion_status() -> None:
    source = (
        REPO / "scripts/hong2021_v70_training_to_gate_supervisor_lageunha.sh"
    ).read_text()
    assert "sleep 60" in source
    assert "complete_V70_fixed_training_pending_train_only_gate" in source
    assert "hong2021_v70_train_gate_lageunha.sh" in source
