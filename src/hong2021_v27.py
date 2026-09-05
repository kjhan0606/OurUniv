#!/usr/bin/env python
"""Train and sample the frozen V27 parent-aligned conditional Haar flow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

import hong2021_v26 as v26
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v21_edm import ARTIFACT_SHA256
from hong2021_v26 import load_frozen_program as load_v26_program
from hong2021_v27_flow import ParentAlignedConditionalHaarSplineFlow


REGISTRY_SCHEMA = (
    "hong2021-v27-parent-aligned-conditional-haar-flow-development-program-v1"
)
REGISTRY_SHA256 = "2cc9710285987d53ce3b4f4b9f8cce78c087e51bd4b1f0e00f5e8c6d70347984"
DESIGN_AUDIT_SHA256 = "8ee643d74d75702560117898f5eb329dbdca4108768b3f2da90437f35e5554bc"
CONTEXT_AUDIT_SHA256 = "482bbec97f1e19446b7b3942862048234ab8eca5d3e698cbce5bac52c4a61a5b"
HAAR_ARTIFACT_SHA256 = v26.HAAR_ARTIFACT_SHA256
MODEL_SCHEMA = "hong2021-v27-parent-aligned-conditional-haar-spline-flow-v1"
PREFLIGHT_SCHEMA = "hong2021-v27-hard-preflight-v1"
PROGRAM_LABEL = "V27"
ENSEMBLE_METHOD = "parent_aligned_conditional_haar_spline_flow"
REGISTRY_ATTRIBUTE = "v27_registry_sha256"
PARAMETERS = 3_787_032
NON_DC_DIMENSIONS = v26.NON_DC_DIMENSIONS
CANDIDATE_STEPS = v26.CANDIDATE_STEPS
DOMAIN_KEYS = v26.DOMAIN_KEYS
CACHE_KEYS = v26.CACHE_KEYS
DETAIL_DIMENSIONS_COARSE_TO_FINE = v26.DETAIL_DIMENSIONS_COARSE_TO_FINE


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"V27 {label} hash mismatch")
    return json.loads(path.read_text())


def load_frozen_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if sha256_file(path) != REGISTRY_SHA256:
        raise ValueError("V27 registry differs from its frozen hash")
    registry = json.loads(path.read_text())
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status")
        != "frozen_before_candidate_training_or_development_evaluation"
    ):
        raise ValueError("V27 registry schema or status mismatch")
    design = registry["design_audit"]
    design_path = _resolve(design["path"], repo)
    design_payload = _verified_json(
        design_path, DESIGN_AUDIT_SHA256, "design audit"
    )
    if (
        design.get("sha256") != DESIGN_AUDIT_SHA256
        or design_payload.get("selected_change", {}).get("name")
        != "parent_aligned_2x2x2_child_phase_condition"
        or design_payload.get("firewall", {}).get("Astrid_accessed") is not False
        or design_payload.get("firewall", {}).get("historical_EAGLE_accessed")
        is not False
    ):
        raise ValueError("V27 design selection or firewall mismatch")
    parent = registry["parent_evidence"]
    v26_path = _resolve(parent["v26_registry"], repo)
    if sha256_file(v26_path) != parent["v26_registry_sha256"]:
        raise ValueError("V27 V26-parent registry hash mismatch")
    _, artifacts, v20, _, haar = load_v26_program(v26_path, repo)
    decision_path = Path(parent["v26_decision"])
    decision = _verified_json(
        decision_path, parent["v26_decision_sha256"], "V26 decision"
    )
    if (
        canonical_digest(decision) != parent["v26_decision_digest_sha256"]
        or decision.get("development_pass") is not False
    ):
        raise ValueError("V27 V26 decision provenance mismatch")
    failure = _verified_json(
        Path(parent["v26_failure_audit"]),
        parent["v26_failure_audit_sha256"],
        "V26 failure audit",
    )
    mechanism = _verified_json(
        Path(parent["v26_mechanism_audit"]),
        parent["v26_mechanism_audit_sha256"],
        "V26 mechanism audit",
    )
    optimizer = _verified_json(
        Path(parent["v26_optimizer_audit"]),
        parent["v26_optimizer_audit_sha256"],
        "V26 optimizer audit",
    )
    context_path = _resolve(parent["v26_context_audit"], repo)
    context = _verified_json(
        context_path, CONTEXT_AUDIT_SHA256, "V26 context audit"
    )
    if (
        parent["v26_context_audit_sha256"] != CONTEXT_AUDIT_SHA256
        or failure.get("decision_digest_sha256")
        != decision.get("decision_digest_sha256")
        or mechanism.get("schema")
        != "hong2021-v26-trained-flow-mechanism-audit-v2"
        or mechanism.get("mechanism_summary", {}).get("trained_flow_roundtrip_stable")
        is not True
        or optimizer.get("schema")
        != "hong2021-v26-gradient-scale-conditioning-audit-v2"
        or optimizer.get("interpretation", {}).get("adam_compensates_raw_scale")
        is not True
        or context.get("context_interface_audit", {}).get("classification")
        != "parent_child_phase_aliasing_in_conditional_haar_context"
        or any(
            payload.get("Astrid_accessed") is not False
            for payload in (mechanism, optimizer, context)
        )
    ):
        raise ValueError("V27 parent audit conclusion or firewall mismatch")
    coordinate = registry["coordinate_system"]
    haar_path = Path(coordinate["standardization_artifact"])
    if (
        coordinate["standardization_artifact_sha256"] != HAAR_ARTIFACT_SHA256
        or sha256_file(haar_path) != HAAR_ARTIFACT_SHA256
        or haar.get("non_dc_dimensions") != NON_DC_DIMENSIONS
        or haar.get("Astrid_accessed") is not False
        or haar.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V27 Haar artifact provenance mismatch")
    condition = registry["condition_representation"]
    if (
        condition.get("target_free") is not True
        or condition.get("total_channels") != 41
        or condition.get("invertible_at_the_pooled_resolution") is not True
    ):
        raise ValueError("V27 parent-aligned condition freeze differs")
    likelihood = registry["likelihood"]
    exact = {
        "levels": 6,
        "detail_channels_per_level": 7,
        "coupling_layers_per_level": 4,
        "bins": 8,
        "tail_bound_standardized_units": 6.0,
        "parameters": PARAMETERS,
        "target_or_density_dependent_weights": False,
        "auxiliary_field_or_tail_losses": False,
    }
    for key, value in exact.items():
        if likelihood.get(key) != value:
            raise ValueError(f"V27 frozen likelihood differs: {key}")
    training = registry["training_protocol"]
    if (
        training.get("batch") != 6
        or training.get("steps") != 30_000
        or training.get("candidate_steps") != list(CANDIDATE_STEPS)
        or training.get("source_balance_per_batch")
        != {"TNG100": 2, "SIMBA": 2, "Swift": 2}
        or training.get("seed") != 144021
    ):
        raise ValueError("V27 training protocol differs from its freeze")
    if sha256_file(repo / "config/hong2021_v21_derived_artifacts.json") != ARTIFACT_SHA256:
        raise ValueError("V27 inherited V21 artifact attestation differs")
    return registry, artifacts, v20, decision, haar


def build_model(
    haar: dict[str, Any], feature_fit: dict[str, Any], *, device: torch.device
) -> ParentAlignedConditionalHaarSplineFlow:
    model = ParentAlignedConditionalHaarSplineFlow(
        detail_mean=haar["source_balanced_mean"],
        detail_std=haar["source_balanced_standard_deviation"],
        context_mean=feature_fit["mean"],
        context_std=feature_fit["std"],
        condition_channels=4,
        hidden_channels=32,
        levels=6,
        couplings=4,
        bins=8,
        tail_bound=6.0,
    ).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != PARAMETERS:
        raise RuntimeError(f"V27 parameter count changed: {parameters}")
    return model


def _validate_checkpoint(
    path: Path, *, step: int, artifacts: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    return v26._validate_checkpoint(
        path, step=step, artifacts=artifacts, program=sys.modules[__name__]
    )


def train(args: argparse.Namespace) -> None:
    v26.train(args, program=sys.modules[__name__])


def sample(args: argparse.Namespace) -> None:
    v26.sample(args, program=sys.modules[__name__])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    sampling = sub.add_parser("sample")
    for item in (training, sampling):
        item.add_argument("--registry", type=Path, required=True)
        item.add_argument("--repo", type=Path, required=True)
        item.add_argument("--device", default="cuda")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--preflight", type=Path, required=True)
    sampling.add_argument("--training-root", type=Path, required=True)
    sampling.add_argument("--domain", choices=tuple(DOMAIN_KEYS), required=True)
    sampling.add_argument("--step", type=int, choices=CANDIDATE_STEPS, required=True)
    sampling.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args) if args.mode == "train" else sample(args)


if __name__ == "__main__":
    main()
