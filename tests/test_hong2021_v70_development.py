import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import hong2021_v70_development_sample as development
import hong2021_v70_seal as sealing
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v70_development_gate import _validate_frozen_gate_sources


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v70_locked_development_program.json"


def test_development_program_is_byte_bound_and_single_use() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == development.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["status"] == (
        "frozen_during_fixed_training_before_train_gate_result_or_development_access"
    )
    assert program["fixed_sampling"]["noise_seed"] == 170073
    assert program["fixed_sampling"]["members_per_query"] == 16
    assert program["unchanged_development_gate"][
        "diagnostic_control_excluded_from_selection"
    ] is True
    assert program["firewall"]["second_development_attempt"] == "forbidden"


def test_program_load_does_not_touch_development_artifacts(monkeypatch) -> None:
    visited: list[Path] = []
    original = development.sha256_file

    def traced(path: str | Path) -> str:
        resolved = Path(path).resolve()
        visited.append(resolved)
        return original(resolved)

    monkeypatch.setattr(development, "sha256_file", traced)
    development.load_program(PROGRAM, REPO)
    assert visited
    assert all(path.is_relative_to(REPO) for path in visited)
    assert not any("development_candidate" in str(path) for path in visited)


def _passing_gate(program: dict, path: Path) -> dict:
    parent = program["parent_programs"]
    gate = {
        "schema": parent["required_train_gate_schema"],
        "status": parent["required_train_gate_status"],
        "program_sha256": parent["v70_train_gate_program_sha256"],
        "train_mechanism_pass": parent["required_train_mechanism_pass"],
        "candidate_selected": parent["required_candidate_selected"],
        "classification": parent["required_classification"],
        "next": parent["required_next"],
        "code_commit": "0" * 40,
        "validation_accessed": False,
        "development_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    gate["decision_digest_sha256"] = canonical_digest(gate)
    path.write_text(json.dumps(gate) + "\n")
    return gate


def test_train_gate_authorization_accepts_only_canonical_pass(tmp_path, monkeypatch) -> None:
    program = copy.deepcopy(development.load_program(PROGRAM, REPO))
    path = tmp_path / "decision.json"
    program["parent_programs"]["required_train_gate_decision"] = str(path)
    expected = _passing_gate(program, path)
    monkeypatch.setattr(development, "_is_ancestor", lambda *_: True)
    actual = development.authorize_train_gate(
        program, REPO, path, sha256_file(path), "f" * 40
    )
    assert actual == expected

    rejected = dict(expected)
    rejected["candidate_selected"] = False
    rejected["decision_digest_sha256"] = canonical_digest(rejected)
    path.write_text(json.dumps(rejected) + "\n")
    with pytest.raises(ValueError, match="authorization"):
        development.authorize_train_gate(
            program, REPO, path, sha256_file(path), "f" * 40
        )


def test_v70_ensemble_schema_has_fixed_shapes(tmp_path) -> None:
    path = tmp_path / "ensemble.h5"
    with h5py.File(path, "w") as handle:
        datasets = development._new_ensemble(handle)
        assert datasets["sample"].shape == (16, 16, 1, 64, 64, 64)
        assert datasets["conditional_mean"].shape == (16, 1, 64, 64, 64)
        assert datasets["truth"].shape == (16, 1, 64, 64, 64)
        assert datasets["initial_latent_sha256"].shape == (16, 16, 32)
        assert datasets["maximum_inverse_CDF_error"].shape == (16, 16)
        assert all(dataset.dtype == np.dtype("float32") for name, dataset in datasets.items() if name not in ("initial_latent_sha256",))
        assert datasets["initial_latent_sha256"].dtype == np.dtype("uint8")


def test_unchanged_gate_implementations_remain_at_frozen_hashes() -> None:
    program = development.load_program(PROGRAM, REPO)
    _validate_frozen_gate_sources(program, REPO)


def test_train_gate_runner_auto_advances_only_on_pass() -> None:
    source = (REPO / "scripts/hong2021_v70_train_gate_lageunha.sh").read_text()
    pass_status = "complete_V70_train_only_gate_pass_locked_development_authorized"
    development_runner = "hong2021_v70_development_lageunha.sh"
    failure_status = "complete_V70_train_only_gate_rejection_development_locked"
    assert source.index(pass_status) < source.index(development_runner)
    assert source.index(development_runner) < source.index(failure_status)


def test_terminal_seal_rejection_never_validates_development(monkeypatch) -> None:
    program = development.load_program(PROGRAM, REPO)
    train = {
        "candidate_selected": False,
        "classification": (
            "query_aligned_latent_spatial_score_does_not_learn_cross_domain_joint_structure"
        ),
        "next": (
            "stop_before_development_without_posthoc_training_sampling_or_gate_tuning"
        ),
    }
    monkeypatch.setattr(sealing, "load_program", lambda *_: program)
    monkeypatch.setattr(sealing, "git_state", lambda *_: ("f" * 40, True))
    monkeypatch.setattr(
        sealing, "validate_train_gate", lambda *_: (train, "1" * 64)
    )
    monkeypatch.setattr(
        sealing,
        "validate_development",
        lambda *_: pytest.fail("rejected train gate touched development"),
    )
    result = sealing.seal(PROGRAM, REPO, Path("train.json"), None)
    assert result["status"] == "sealed_train_gate_rejection_development_not_accessed"
    assert result["development_accessed"] is False
    assert result["development_decision"] is None
    assert result["independent_gate_locked"] is True


@pytest.mark.parametrize("passed", [False, True])
def test_terminal_seal_preserves_locked_development_branch(monkeypatch, passed) -> None:
    program = development.load_program(PROGRAM, REPO)
    train = {"candidate_selected": True}
    branch = (
        {
            "development_pass": True,
            "classification": "V70_is_development_sufficient",
            "next": (
                "seal_V70_and_await_explicit_user_approval_before_independent_EAGLE_access"
            ),
        }
        if passed
        else {
            "development_pass": False,
            "classification": (
                "V70_joint_spatial_model_is_not_development_sufficient"
            ),
            "next": (
                "seal_the_failure_and_stop_before_independent_EAGLE_without_sampler_threshold_or_model_tuning"
            ),
        }
    )
    monkeypatch.setattr(sealing, "load_program", lambda *_: program)
    monkeypatch.setattr(sealing, "git_state", lambda *_: ("f" * 40, True))
    monkeypatch.setattr(
        sealing, "validate_train_gate", lambda *_: (train, "1" * 64)
    )
    monkeypatch.setattr(
        sealing, "validate_development", lambda *_: (branch, "2" * 64)
    )
    result = sealing.seal(
        PROGRAM, REPO, Path("train.json"), Path("development.json")
    )
    assert result["development_accessed"] is True
    assert result["development_pass"] is passed
    assert result["explicit_user_approval_required_before_EAGLE"] is passed
    assert result["independent_EAGLE_accessed"] is False
