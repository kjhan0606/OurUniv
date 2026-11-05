from __future__ import annotations

import json
from pathlib import Path

from hong2021_v18_init import sha256_file


def test_v25_registry_freezes_only_the_proper_unweighted_objective_change() -> None:
    registry = json.loads(
        Path("config/hong2021_v25_development_program.json").read_text()
    )
    assert registry["status"] == "frozen_before_implementation_or_execution"
    change = registry["single_change"]
    assert change["loss_coefficients_from"] == {
        "unweighted": 0.5,
        "tail_weighted": 0.5,
    }
    assert change["loss_coefficients_to"] == {
        "unweighted": 1.0,
        "tail_weighted": 0.0,
    }
    assert change["base_channels"] == 48
    assert change["parameters"] == 8133361
    assert change["candidate_steps"] == [10000, 20000, 30000]
    attestation = Path(registry["mechanism_audit"]["attestation"])
    assert sha256_file(attestation) == registry["mechanism_audit"][
        "attestation_sha256"
    ]


def test_v25_audit_attestation_keeps_independent_data_sealed() -> None:
    audit = json.loads(
        Path("config/hong2021_v24_tail_sampler_audit.json").read_text()
    )
    assert audit["Astrid_accessed"] is False
    assert audit["historical_EAGLE_accessed"] is False
    replay = audit["terminal_sampler_replay"]
    assert replay["numerically_identical_within_one_float32_epsilon_fields"] == 208
    assert replay["failed_fields_with_terminal_centered_z_at_or_above_5"] == 0
