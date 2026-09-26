from __future__ import annotations

import json
from pathlib import Path

from hong2021_v18_init import sha256_file
from hong2021_v26 import (
    DESIGN_AUDIT_SHA256,
    HAAR_ARTIFACT_SHA256,
    PARAMETERS,
    REGISTRY_SHA256,
    build_model,
    load_frozen_program,
)


REPO = Path(__file__).parents[1]


def test_v26_registry_hash_and_firewall():
    path = REPO / "config/hong2021_v26_development_program.json"
    assert sha256_file(path) == REGISTRY_SHA256
    registry = json.loads(path.read_text())
    assert registry["design_audit"]["sha256"] == DESIGN_AUDIT_SHA256
    assert registry["approval_firewall"]["Astrid_attempts_remaining"] == 1
    assert registry["approval_firewall"]["historical_EAGLE_access"] == "forbidden"


def test_v26_frozen_program_and_model_parameter_count():
    registry, _, _, decision, haar = load_frozen_program(
        REPO / "config/hong2021_v26_development_program.json", REPO
    )
    assert decision["development_pass"] is False
    assert registry["coordinate_system"]["standardization_artifact_sha256"] == HAAR_ARTIFACT_SHA256
    feature_fit = {
        "mean": [0.0] * 8,
        "std": [1.0] * 8,
    }
    model = build_model(haar, feature_fit, device=__import__("torch").device("cpu"))
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS


def test_v26_has_no_target_weighted_or_edm_objective():
    registry = json.loads(
        (REPO / "config/hong2021_v26_development_program.json").read_text()
    )
    likelihood = registry["likelihood"]
    assert likelihood["target_or_density_dependent_weights"] is False
    assert likelihood["auxiliary_field_or_tail_losses"] is False
    assert "negative log likelihood" in likelihood["objective"]
    assert registry["single_coherent_change"]["edm_model_loss_noise_schedule_initializer_and_ode_sampler_removed"] is True
