import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import hong2021_v72_sample as sampling
import hong2021_v72_seal as sealing
import hong2021_v72_sqt as sqt
import hong2021_v72_metadata_recovery as recovery
import hong2021_v72_gate as gating
import hong2021_v72_gate_label_recovery as gate_label_recovery
from hong2021_v18_init import sha256_file


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v72_locked_spatial_quantile_transport_program.json"


def test_v72_program_is_byte_bound_and_two_stage_locked() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == sqt.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == sqt.PROGRAM_SCHEMA
    assert program["status"] == sqt.PROGRAM_STATUS
    assert program["candidate_construction"]["strata_per_query"] == 16
    assert program["fixed_sampling"]["stage_A_noise_seed"] == 170082
    assert program["fixed_sampling"]["stage_B_noise_seed"] == 170083
    assert program["fair_tail_gate"]["unequal_sample_global_maximum_comparison"] == (
        "forbidden"
    )
    assert program["firewall"]["second_attempt_in_either_stage"] == "forbidden"


def test_v72_program_load_is_local_and_selections_are_disjoint(monkeypatch) -> None:
    visited: list[Path] = []
    original = sqt.sha256_file

    def traced(path: str | Path) -> str:
        resolved = Path(path).resolve()
        visited.append(resolved)
        return original(resolved)

    monkeypatch.setattr(sqt, "sha256_file", traced)
    program = sqt.load_program(PROGRAM, REPO)
    assert visited and all(path.is_relative_to(REPO) for path in visited)
    first = sqt.stage_selection(program, "A")
    second = sqt.stage_selection(program, "B")
    old = program["historical_consumed_selection"]
    for domain in sqt.DOMAIN_ORDER:
        assert len(first[domain]) == len(second[domain]) == 16
        assert not set(first[domain]) & set(second[domain])
        assert not set(first[domain]) & set(old[domain])
        assert not set(second[domain]) & set(old[domain])


def test_conditioning_strata_are_stable_equal_occupancy_partition() -> None:
    score = torch.zeros((1, 64, 64, 64))
    positions = sqt.conditioning_strata(score)
    assert positions.shape == (16, 16384)
    assert torch.equal(positions.reshape(-1), torch.arange(64**3))
    assert torch.equal(torch.sort(positions.reshape(-1)).values, torch.arange(64**3))


def test_sqt_preserves_each_member_stratum_multiset_and_spatial_rank() -> None:
    generator = torch.Generator().manual_seed(720072)
    rank_source = torch.randn((16, 1, 64, 64, 64), generator=generator)
    marginal = torch.randn((16, 1, 64, 64, 64), generator=generator)
    score = torch.linspace(-2.0, 2.0, 64**3).reshape(1, 64, 64, 64)
    positions = sqt.conditioning_strata(score)
    coupled, diagnostics = sqt.spatial_quantile_transport(
        rank_source, marginal, positions
    )
    source_flat = rank_source.reshape(16, -1)
    marginal_flat = marginal.reshape(16, -1)
    coupled_flat = coupled.reshape(16, -1)
    for selected in positions:
        assert torch.equal(
            torch.sort(coupled_flat[:, selected], dim=1, stable=True).values,
            torch.sort(marginal_flat[:, selected], dim=1, stable=True).values,
        )
        source_order = torch.argsort(
            source_flat[:, selected], dim=1, stable=True
        )
        coupled_order = torch.argsort(
            coupled_flat[:, selected], dim=1, stable=True
        )
        sorted_values = torch.sort(
            marginal_flat[:, selected], dim=1, stable=True
        ).values
        adjacent = sorted_values[:, 1:] == sorted_values[:, :-1]
        tied = torch.zeros_like(sorted_values, dtype=torch.bool)
        tied[:, 1:] |= adjacent
        tied[:, :-1] |= adjacent
        assert not torch.any((source_order != coupled_order) & ~tied)
    assert diagnostics["pre_inverse_stratum_multiset_equal"] is True
    assert diagnostics["maximum_pre_inverse_stratum_multiset_error"] == 0.0
    assert diagnostics["rank_disagreement_fraction_excluding_marginal_ties"] == 0.0
    assert diagnostics["marginal_tied_voxel_fraction"] <= 3.0e-4


def test_scalar_energy_score_is_proper_unbiased_form() -> None:
    perfect = np.ones(16)
    assert sqt.scalar_energy_score(perfect, 1.0) == 0.0
    shifted = np.full(16, 2.0)
    assert sqt.scalar_energy_score(shifted, 1.0) == 1.0
    dispersed = np.arange(16, dtype=np.float64)
    score = sqt.scalar_energy_score(dispersed, 7.5)
    assert np.isfinite(score) and score >= 0.0


def test_v72_ensemble_schema_has_frozen_shapes(tmp_path) -> None:
    path = tmp_path / "ensemble.h5"
    with h5py.File(path, "w") as handle:
        datasets = sampling._new_ensemble(handle)
        assert datasets["sample"].shape == (16, 16, 1, 64, 64, 64)
        assert datasets["source_index"].shape == (16,)
        assert datasets["initial_latent_sha256"].shape == (16, 16, 32)
        assert datasets["pre_inverse_stratum_multiset_equal"].shape == (16,)
        assert datasets["marginal_tied_voxel_fraction"].shape == (16,)


@pytest.mark.parametrize("stage_a_pass,stage_b_pass", [(False, None), (True, False), (True, True)])
def test_v72_terminal_seal_respects_two_stage_firewall(
    monkeypatch, stage_a_pass, stage_b_pass
) -> None:
    program = sqt.load_program(PROGRAM, REPO)
    first = {
        "stage_pass": stage_a_pass,
        "classification": (
            "V72_SQT_passes_fresh_stage_A"
            if stage_a_pass else "V72_SQT_is_not_fresh_screen_sufficient"
        ),
        "next": (
            "run_one_locked_stage_B_without_candidate_changes"
            if stage_a_pass
            else "seal_and_stop_without_stage_B_or_EAGLE_access_or_posthoc_candidate_changes"
        ),
    }
    second = None if stage_b_pass is None else {
        "stage_pass": stage_b_pass,
        "classification": (
            "V72_SQT_is_fresh_two_stage_sufficient"
            if stage_b_pass else "V72_SQT_is_not_fresh_confirmation_sufficient"
        ),
        "next": (
            "seal_and_await_new_explicit_user_approval_before_independent_EAGLE"
            if stage_b_pass
            else "seal_and_stop_before_EAGLE_without_repeating_either_stage"
        ),
    }
    monkeypatch.setattr(sealing, "load_program", lambda *_: copy.deepcopy(program))
    monkeypatch.setattr(sealing, "git_state", lambda *_: ("f" * 40, True))
    monkeypatch.setattr(sealing, "_is_ancestor", lambda *_: True)
    monkeypatch.setattr(
        sealing,
        "authorize_parent_evidence",
        lambda *_: {
            "v71_terminal_seal": "/sealed/v71.json",
            "v71_terminal_seal_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(sealing, "validate_preflight", lambda *_: {})

    def validate(*args, **kwargs):
        stage = args[2]
        return (first, "2" * 64) if stage == "A" else (second, "3" * 64)

    monkeypatch.setattr(sealing, "validate_stage_decision", validate)
    stage_b_path = Path("stage_B.json") if stage_a_pass else None
    result = sealing.seal(
        PROGRAM, REPO, Path("preflight.json"), "4" * 64,
        Path("stage_A.json"), stage_b_path,
    )
    assert result["stage_A_pass"] is stage_a_pass
    assert result["stage_B_accessed"] is stage_a_pass
    assert result["two_stage_pass"] is bool(stage_a_pass and stage_b_pass)
    assert result["independent_EAGLE_accessed"] is False
    assert result["independent_gate_locked"] is True
    assert result["explicit_user_approval_required_before_EAGLE"] is bool(
        stage_a_pass and stage_b_pass
    )


def test_v72_runner_orders_stage_B_only_after_stage_A_pass() -> None:
    source = (REPO / "scripts/hong2021_v72_sqt_two_stage_lageunha.sh").read_text()
    assert source.index("hong2021_v72_preflight.py") < source.index("run_stage A")
    branch = source.index("if [[ $(jq -r '.stage_pass' \"$stage_A/decision.json\") == true ]]")
    assert source.index("run_stage A") < branch < source.index("run_stage B")
    assert "complete_V72_stage_A_failure_stage_B_unopened_independent_gate_locked" in source
    assert "complete_V72_two_stage_pass_waiting_explicit_EAGLE_approval" in source


def test_v72_gate_source_never_uses_unequal_global_maximum_for_selection() -> None:
    source = (REPO / "src/hong2021_v72_gate.py").read_text()
    assert "cube_maximum_energy_score" in source
    assert '"unequal_sample_global_maximum_used": False' in source
    assert "generated_max_above_truth_max_dex" not in source


def test_v72_metadata_recovery_changes_only_one_attribute(tmp_path) -> None:
    path = tmp_path / "ensemble.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample", data=np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        handle.create_dataset("source_index", data=np.asarray([7, 11], dtype=np.int64))
        handle.attrs.update(
            {
                "schema": sqt.ENSEMBLE_SCHEMA,
                "v72_program_sha256": sqt.PROGRAM_SHA256,
                "stage": "A",
                "complete": True,
                "another_attribute": "unchanged",
            }
        )
    before = sha256_file(path)
    row = recovery.repair_file(path, before)
    assert row["all_datasets_byte_identical"] is True
    assert row["only_added_attribute"] == {"diagnostic_k_h_mpc": 1.0}
    with h5py.File(path, "r") as handle:
        assert float(handle.attrs["diagnostic_k_h_mpc"]) == 1.0
        assert handle.attrs["another_attribute"] == "unchanged"
        assert np.array_equal(
            handle["sample"][:], np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        )


def test_v72_recovery_runner_never_resamples_stage_A() -> None:
    source = (
        REPO / "scripts/hong2021_v72_resume_after_metadata_recovery_lageunha.sh"
    ).read_text()
    assert "hong2021_v72_metadata_recovery.py" in source
    assert "--stage A" in source
    assert "hong2021_v72_sample.py" in source
    sample_invocation = source.split("hong2021_v72_sample.py", 1)[1]
    assert "--stage B" in sample_invocation
    assert "--stage A" not in sample_invocation
    assert source.index("hong2021_v72_metadata_recovery.py") < source.index(
        "evaluate_stage A"
    )


def test_v72_gate_label_recovery_is_byte_bound() -> None:
    path = REPO / "config/hong2021_v72_stage_A_gate_label_recovery_program.json"
    assert sha256_file(path) == gate_label_recovery.RECOVERY_SHA256
    program = json.loads(path.read_text())
    assert program["schema"] == gate_label_recovery.RECOVERY_SCHEMA
    assert len(program["stage_A_artifacts"]) == 9
    assert program["firewall"]["stage_A_resampling_or_reevaluation"] == "forbidden"


def test_v72_gate_loader_requires_the_sole_sqt_candidate(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    payload = {
        "schema": gating.EVALUATION_SCHEMA,
        "voxel_mpc_h": 0.3125,
        "candidates": {"sqt": {"label": "sqt", "path": "/frozen/ensemble.h5"}},
    }
    path.write_text(json.dumps(payload))
    assert gating._load_sqt_metrics(path)["label"] == "sqt"
    payload["candidates"] = {"edm": {"label": "edm"}}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sole sqt"):
        gating._load_sqt_metrics(path)


def test_v72_gate_label_recovery_runner_never_repeats_stage_A() -> None:
    source = (
        REPO / "scripts/hong2021_v72_resume_after_gate_label_recovery_lageunha.sh"
    ).read_text()
    gate = source.index("hong2021_v72_gate_label_recovery.py")
    decision = source.index("--stage A --out \"$stage_A/decision.json\"")
    assert gate < decision
    assert "evaluate_stage A" not in source
    assert "--stage A --stage-A-decision" not in source
    assert "--stage B" in source
    assert "--gate-label-recovery \"$gate_recovery\"" in source


def test_v72_terminal_result_records_the_sealed_stage_A_failure() -> None:
    result = json.loads(
        (REPO / "config/hong2021_v72_result_record.json").read_text()
    )
    assert result["stage_A_gate"]["stage_pass"] is False
    assert result["stage_A_gate"]["explicit_spectral_all_domains"] is True
    assert result["stage_A_gate"]["explicit_residual_RMS_all_domains"] is True
    assert result["terminal_seal"]["stage_B_accessed"] is False
    assert result["firewall"]["independent_EAGLE_accessed"] is False
    assert result["authorization"]["rerun_or_modify_V72"] is False
    assert sha256_file(REPO / result["frozen_program"]["path"]) == result[
        "frozen_program"
    ]["sha256"]
