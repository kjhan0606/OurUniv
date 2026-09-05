from __future__ import annotations

import json
from pathlib import Path

import torch

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v18_init import sha256_file
from hong2021_v27 import (
    DESIGN_AUDIT_SHA256,
    PARAMETERS,
    REGISTRY_SHA256,
    build_model,
    load_frozen_program,
)


REPO = Path(__file__).parents[1]


def test_v27_registry_hash_single_change_and_firewall():
    path = REPO / "config/hong2021_v27_development_program.json"
    assert sha256_file(path) == REGISTRY_SHA256
    registry = json.loads(path.read_text())
    assert registry["design_audit"]["sha256"] == DESIGN_AUDIT_SHA256
    assert registry["single_coherent_change"]["old_to_new_local_condition_channels"] == [4, 32]
    assert registry["condition_representation"]["target_free"] is True
    assert registry["approval_firewall"]["Astrid_attempts_remaining"] == 1
    assert registry["approval_firewall"]["historical_EAGLE_access"] == "forbidden"


def test_v27_frozen_program_and_parameter_count():
    registry, _, _, decision, haar = load_frozen_program(
        REPO / "config/hong2021_v27_development_program.json", REPO
    )
    assert decision["development_pass"] is False
    assert registry["likelihood"]["parameters"] == PARAMETERS
    model = build_model(
        haar,
        {"mean": [0.0] * len(FEATURE_NAMES), "std": [1.0] * len(FEATURE_NAMES)},
        device=torch.device("cpu"),
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS
