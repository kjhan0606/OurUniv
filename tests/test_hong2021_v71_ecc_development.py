import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import hong2021_v71_development_sample as sampling
import hong2021_v71_ecc as ecc
import hong2021_v71_seal as sealing


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v71_locked_ecc_development_program.json"


def test_v71_program_is_byte_bound_and_explicitly_path_b() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == ecc.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["status"] == ecc.PROGRAM_STATUS
    assert program["authorization"][
        "V71_implementation_and_single_development_attempt_authorized"
    ] is True
    assert program["authorization"]["independent_EAGLE_access_authorized"] is False
    assert program["fixed_sampling"]["noise_seed"] == 170073
    assert program["fixed_sampling"]["members_per_query"] == 16
    assert program["firewall"]["second_development_attempt"] == "forbidden"


def test_v71_program_load_is_local_and_does_not_touch_development(monkeypatch) -> None:
    visited: list[Path] = []
    original = ecc.sha256_file

    def traced(path: str | Path) -> str:
        resolved = Path(path).resolve()
        visited.append(resolved)
        return original(resolved)

    monkeypatch.setattr(ecc, "sha256_file", traced)
    ecc.load_program(PROGRAM, REPO)
    assert visited
    assert all(path.is_relative_to(REPO) for path in visited)
    assert not any("development_candidate" in str(path) for path in visited)


def test_ecc_exactly_preserves_each_voxel_member_multiset_and_rank() -> None:
    generator = torch.Generator().manual_seed(71)
    rank_source = torch.randn((16, 1, 7, 6, 5), generator=generator)
    marginal = torch.randn((16, 1, 7, 6, 5), generator=generator)
    coupled, diagnostics = ecc.ensemble_copula_couple(rank_source, marginal)
    assert torch.equal(
        torch.sort(coupled, dim=0, stable=True).values,
        torch.sort(marginal, dim=0, stable=True).values,
    )
    assert torch.equal(
        torch.argsort(coupled, dim=0, stable=True),
        torch.argsort(rank_source, dim=0, stable=True),
    )
    assert diagnostics["pre_inverse_sorted_latent_multiset_equal"] is True
    assert diagnostics["maximum_pre_inverse_sorted_latent_multiset_error"] == 0.0
    assert diagnostics[
        "candidate_rank_disagreement_fraction_excluding_control_ties"
    ] == 0.0


def test_ecc_tie_rule_is_deterministic_and_excludes_ties_from_rank_check() -> None:
    rank_source = torch.arange(16.0).view(16, 1, 1, 1, 1).flip(0)
    marginal = torch.arange(16.0).view(16, 1, 1, 1, 1)
    marginal[7] = marginal[6]
    first, diagnostics = ecc.ensemble_copula_couple(rank_source, marginal)
    second, repeated = ecc.ensemble_copula_couple(rank_source, marginal)
    assert torch.equal(first, second)
    assert diagnostics == repeated
    assert diagnostics["control_tied_voxel_fraction"] == 1.0
    assert diagnostics[
        "candidate_rank_disagreement_fraction_excluding_control_ties"
    ] == 0.0


def test_v71_ensemble_schema_has_frozen_shapes(tmp_path) -> None:
    path = tmp_path / "ensemble.h5"
    with h5py.File(path, "w") as handle:
        datasets = sampling._new_ensemble(handle)
        assert datasets["sample"].shape == (16, 16, 1, 64, 64, 64)
        assert datasets["conditional_mean"].shape == (16, 1, 64, 64, 64)
        assert datasets["truth"].shape == (16, 1, 64, 64, 64)
        assert datasets["initial_latent_sha256"].shape == (16, 16, 32)
        assert datasets["pre_inverse_sorted_latent_multiset_equal"].shape == (16,)
        assert datasets["maximum_post_DC_sorted_residual_multiset_error"].shape == (16,)
        assert datasets["initial_latent_sha256"].dtype == np.dtype("uint8")


def test_v71_unchanged_gate_sources_remain_frozen() -> None:
    program = ecc.load_program(PROGRAM, REPO)
    ecc.validate_frozen_gate_sources(program, REPO)


@pytest.mark.parametrize("passed", [False, True])
def test_v71_terminal_seal_preserves_single_use_branch(monkeypatch, passed) -> None:
    program = ecc.load_program(PROGRAM, REPO)
    branch = {
        "development_pass": passed,
        "classification": (
            "V71_tail_preserving_ECC_is_development_sufficient"
            if passed
            else "V71_tail_preserving_ECC_is_not_development_sufficient"
        ),
        "next": (
            "seal_V71_and_await_new_explicit_user_approval_before_independent_EAGLE_access"
            if passed
            else "seal_the_failure_and_stop_before_independent_EAGLE_without_repeating_development_or_tuning_from_its_results"
        ),
    }
    monkeypatch.setattr(sealing, "load_program", lambda *_: copy.deepcopy(program))
    monkeypatch.setattr(sealing, "git_state", lambda *_: ("f" * 40, True))
    monkeypatch.setattr(sealing, "_is_ancestor", lambda *_: True)
    monkeypatch.setattr(
        sealing,
        "authorize_parent_evidence",
        lambda *_: {
            "v70_train_gate_sha256": "1" * 64,
            "v70_terminal_seal_sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(sealing, "validate_preflight", lambda *_: {})
    monkeypatch.setattr(
        sealing, "validate_development", lambda *_: (branch, "3" * 64)
    )
    result = sealing.seal(
        PROGRAM, REPO, Path("preflight.json"), "4" * 64,
        Path("development.json"),
    )
    assert result["development_accessed"] is True
    assert result["development_pass"] is passed
    assert result["single_development_attempt_consumed"] is True
    assert result["independent_EAGLE_accessed"] is False
    assert result["independent_gate_locked"] is True
    assert result["explicit_user_approval_required_before_EAGLE"] is passed


def test_v71_runner_orders_preflight_before_semantic_access_and_seal() -> None:
    source = (
        REPO / "scripts/hong2021_v71_ecc_development_lageunha.sh"
    ).read_text()
    preflight = "hong2021_v71_preflight.py"
    sample = "hong2021_v71_development_sample.py"
    gate = "hong2021_v71_development_gate.py"
    seal = "hong2021_v71_seal.py"
    assert source.index(preflight) < source.index(sample)
    assert source.index(sample) < source.index(gate)
    assert source.index(gate) < source.index(seal)
    assert "complete_V71_development_pass_waiting_explicit_EAGLE_approval" in source
    assert "complete_V71_development_failure_independent_gate_locked" in source
    assert "hong2021" not in source.split("pytest -q", 1)[1].split("preflight", 1)[0]
