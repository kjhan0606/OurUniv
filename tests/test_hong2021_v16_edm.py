from __future__ import annotations

import argparse
import hashlib
import json

import pytest
import torch

from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import (
    SCHEMA as V14_SCHEMA,
    V15_E3_SCHEMA,
    V16_E4_SCHEMA,
    decoder_upsampling_for_schema,
)
from hong2021_v15_development_gate import canonical_digest
from hong2021_v16_edm import frozen_training_namespace


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arguments(tmp_path) -> argparse.Namespace:
    repo = tmp_path / "repo"
    config = repo / "config"
    config.mkdir(parents=True)
    parent_registry = config / "v15.json"
    parent_registry.write_text("{}\n")
    decision = {
        "schema": "test-v15-decision",
        "development_pass": False,
        "Astrid_used": False,
    }
    decision["decision_digest_sha256"] = canonical_digest(decision)
    decision_path = repo / "e3_decision.json"
    decision_path.write_text(json.dumps(decision) + "\n")
    registry = {
        "schema": "hong2021-v16-predeclared-representation-redesign-program-v1",
        "status": "frozen_after_v15_e3_development_failure_before_v16_e4",
        "parent_v15": {
            "registry": "config/v15.json",
            "registry_sha256": _sha(parent_registry),
            "e3_decision": str(decision_path),
            "e3_decision_sha256": _sha(decision_path),
            "e3_decision_digest_sha256": decision["decision_digest_sha256"],
        },
        "e4_trilinear_decoder": {
            "steps": 10000,
            "candidate_steps": [5000, 10000],
            "tail_exponent": 0.25,
            "tail_maximum": 10.0,
            "training_seed": 144021,
            "validation_seeds": {"TNG100": 99173, "SIMBA": 99174, "Swift": 99175},
            "edm_p_mean": "log(0.6*sigma_data)",
            "edm_p_std": 1.2,
            "interpolation": {
                "parent": "nearest",
                "candidate": "trilinear",
                "align_corners": False,
                "antialias": False,
                "decoder_transitions": 2,
            },
            "sampler": {
                "ensemble_members": 16,
                "steps": 40,
                "sigma_min": 0.002,
                "sigma_max": 40.0,
                "rho": 7.0,
            },
        },
    }
    registry_path = config / "v16.json"
    registry_path.write_text(json.dumps(registry))
    values = {
        "registry": registry_path,
        "repo": repo,
        "out": str(tmp_path / "out"),
        "workers": 1,
        "device": "cpu",
    }
    for prefix in (
        "tng_train", "tng_validation", "simba_train", "simba_validation",
        "swift_train", "swift_validation",
    ):
        values[f"{prefix}_data"] = f"{prefix}_data.h5"
        values[f"{prefix}_cache"] = f"{prefix}_cache.h5"
    return argparse.Namespace(**values)


def test_v16_namespace_exposes_only_predeclared_trilinear_change(
    tmp_path, monkeypatch
) -> None:
    arguments = _arguments(tmp_path)
    monkeypatch.setattr(
        "hong2021_v16_edm.FROZEN_REGISTRY_SHA256", _sha(arguments.registry)
    )
    monkeypatch.setattr("hong2021_v16_edm.git_state", lambda repo: ("a" * 40, True))
    actual = frozen_training_namespace(arguments)
    assert actual.run_schema == V16_E4_SCHEMA
    assert actual.steps == 10000
    assert actual.candidate_steps == "5000,10000"
    assert actual.edm_p_mean_sigma_data_fraction == 0.6
    assert actual.edm_p_std == 1.2
    assert actual.tail_exponent == 0.25
    assert actual.seed == 144021


def test_v16_namespace_rejects_dirty_worktree(tmp_path, monkeypatch) -> None:
    arguments = _arguments(tmp_path)
    monkeypatch.setattr(
        "hong2021_v16_edm.FROZEN_REGISTRY_SHA256", _sha(arguments.registry)
    )
    monkeypatch.setattr("hong2021_v16_edm.git_state", lambda repo: ("a" * 40, False))
    with pytest.raises(RuntimeError, match="clean committed"):
        frozen_training_namespace(arguments)


def test_v16_namespace_rejects_registry_changed_after_preregistration(
    tmp_path, monkeypatch
) -> None:
    arguments = _arguments(tmp_path)
    monkeypatch.setattr("hong2021_v16_edm.FROZEN_REGISTRY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen hash"):
        frozen_training_namespace(arguments)


def test_schema_selects_decoder_interpolation_without_parameter_change() -> None:
    assert decoder_upsampling_for_schema(V14_SCHEMA) == "nearest"
    assert decoder_upsampling_for_schema(V15_E3_SCHEMA) == "nearest"
    assert decoder_upsampling_for_schema(V16_E4_SCHEMA) == "trilinear"
    nearest = ObservableContextUNet(base_channels=4, decoder_upsampling="nearest")
    trilinear = ObservableContextUNet(base_channels=4, decoder_upsampling="trilinear")
    assert sum(p.numel() for p in nearest.parameters()) == sum(
        p.numel() for p in trilinear.parameters()
    )
    trilinear.load_state_dict(nearest.state_dict())
    value = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 4, 2, 2, 2)
    assert nearest.upsample_decoder(value, (4, 4, 4)).shape == (1, 4, 4, 4, 4)
    assert not torch.equal(
        nearest.upsample_decoder(value, (4, 4, 4)),
        trilinear.upsample_decoder(value, (4, 4, 4)),
    )


def test_invalid_decoder_interpolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="nearest or trilinear"):
        ObservableContextUNet(base_channels=4, decoder_upsampling="cubic")
