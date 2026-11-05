#!/usr/bin/env python
"""Frozen V71 Path-B integrity checks and ensemble-copula coupling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = (
    "hong2021-v71-single-use-tail-preserving-ensemble-copula-"
    "development-program-v2"
)
PROGRAM_STATUS = (
    "refrozen_after_code_only_float32_tie_audit_before_implementation_"
    "commit_or_development_access"
)
PROGRAM_SHA256 = "23665bf5d06212113de6feee7cd756c0252a59eac204def9805c5f4595ddf008"
PROGRAM_FREEZE_COMMIT = "375cfa237fa10a5dfb361cffb462a3fcc7310b06"
PREFLIGHT_SCHEMA = "hong2021-v71-code-only-path-B-preflight-v1"
V70_GATE_SCHEMA = "hong2021-v70-train-only-joint-structure-mechanism-decision-v1"
V70_SEAL_SCHEMA = "hong2021-v70-terminal-sealed-result-v1"
ENSEMBLE_SCHEMA = "hong2021-v71-tail-preserving-ecc-ensemble-v1"
CANDIDATE = "tail_preserving_ECC_V70_copula_V63_marginal"
CONTROL = "independent_voxel_V63_marginal"
ARMS = (CANDIDATE, CONTROL)
METHOD = "fixed_V70_rank_copula_paired_V63_innovation_marginal"


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    """Validate local frozen V71 evidence without touching development payloads."""
    repo = repo.resolve()
    path = path.resolve()
    if sha256_file(path) != PROGRAM_SHA256:
        raise ValueError("V71 program hash differs")
    program = strict_json(path)
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("authorization", {}).get(
            "V71_implementation_and_single_development_attempt_authorized"
        )
        is not True
        or program.get("authorization", {}).get(
            "independent_EAGLE_access_authorized"
        )
        is not False
    ):
        raise ValueError("V71 program schema, status, or authorization differs")
    parent = program["parent_evidence"]
    for key in (
        "preimplementation_tie_threshold_erratum",
        "v71_feasibility_audit",
        "v70_result_record",
        "v70_model_program",
        "v70_development_program",
    ):
        if sha256_file(resolve_path(repo, parent[key])) != parent[f"{key}_sha256"]:
            raise ValueError(f"V71 local parent differs: {key}")
    result = strict_json(resolve_path(repo, parent["v70_result_record"]))
    if (
        result.get("status")
        != "complete_train_gate_rejection_development_unopened_failure_audited"
        or result.get("firewall", {}).get("development_accessed") is not False
        or result.get("firewall", {}).get("independent_EAGLE_accessed") is not False
        or result.get("firewall", {}).get("independent_gate_locked") is not True
        or result.get("authorization", {}).get("modify_or_rerun_V70") is not False
    ):
        raise ValueError("V71 V70 terminal result ancestry differs")
    for key in (
        "training_program",
        "train_gate_program",
        "locked_development_program",
    ):
        frozen = result["frozen_programs"]
        if sha256_file(resolve_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V71 V70 frozen program differs: {key}")
    return program


def authorize_parent_evidence(
    program: dict[str, Any], repo: Path, commit: str
) -> dict[str, Any]:
    """Validate fixed V70 rejection evidence; no development HDF5 is opened."""
    repo = repo.resolve()
    if not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit):
        raise ValueError("V71 code does not descend from the frozen program")
    parent = program["parent_evidence"]
    gate_path = Path(parent["v70_train_gate_decision"]).resolve()
    seal_path = Path(parent["v70_terminal_seal"]).resolve()
    if sha256_file(gate_path) != parent["v70_train_gate_decision_sha256"]:
        raise ValueError("V71 bound V70 train-gate hash differs")
    if sha256_file(seal_path) != parent["v70_terminal_seal_sha256"]:
        raise ValueError("V71 bound V70 terminal-seal hash differs")
    gate = strict_json(gate_path)
    if (
        gate.get("schema") != V70_GATE_SCHEMA
        or canonical_digest(gate) != gate.get("decision_digest_sha256")
        or gate.get("decision_digest_sha256")
        != parent["v70_train_gate_decision_digest_sha256"]
        or gate.get("candidate_selected") is not False
        or gate.get("train_mechanism_pass") is not False
        or gate.get("development_accessed") is not False
        or gate.get("independent_EAGLE_accessed") is not False
        or gate.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(gate.get("code_commit")), commit)
    ):
        raise ValueError("V71 bound V70 train-gate decision differs")
    seal = strict_json(seal_path)
    if (
        seal.get("schema") != V70_SEAL_SCHEMA
        or seal.get("status")
        != "sealed_train_gate_rejection_development_not_accessed"
        or canonical_digest(seal) != seal.get("decision_digest_sha256")
        or seal.get("train_gate_sha256")
        != parent["v70_train_gate_decision_sha256"]
        or seal.get("development_accessed") is not False
        or seal.get("development_decision") is not None
        or seal.get("independent_EAGLE_accessed") is not False
        or seal.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(seal.get("sealing_code_commit")), commit)
    ):
        raise ValueError("V71 bound V70 terminal seal differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v63_checkpoint",
        "conditioning_cache",
        "v70_checkpoint",
        "v70_training_report",
    ):
        if sha256_file(resolve_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V71 frozen input differs: {key}")
    return {
        "v70_train_gate": gate,
        "v70_terminal_seal": seal,
        "v70_train_gate_sha256": parent["v70_train_gate_decision_sha256"],
        "v70_terminal_seal_sha256": parent["v70_terminal_seal_sha256"],
    }


def load_development_definition(
    program: dict[str, Any], repo: Path
) -> dict[str, Any]:
    """Hash development inputs before reading any source index or truth dataset."""
    frozen = program["frozen_inputs"]
    path = resolve_path(repo, frozen["v35_development_definition"])
    if sha256_file(path) != frozen["v35_development_definition_sha256"]:
        raise ValueError("V71 V35 development definition differs")
    v35 = strict_json(path)
    selected = program["immutable_development_selection"]
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        selection_path = Path(selected[f"{domain}_selection"]).resolve()
        selection_sha = selected[f"{domain}_selection_sha256"]
        if (
            Path(row["phase_object_selection"]).resolve() != selection_path
            or row["phase_object_selection_sha256"] != selection_sha
            or sha256_file(selection_path) != selection_sha
        ):
            raise ValueError(f"V71 {domain} development selection differs")
        for key in ("validation_data", "validation_cache"):
            if sha256_file(Path(row[key])) != row[f"{key}_sha256"]:
                raise ValueError(f"V71 {domain} {key} differs")
    return v35


def validate_frozen_gate_sources(program: dict[str, Any], repo: Path) -> None:
    frozen = program["unchanged_development_gate"]
    for path_key, hash_key in (
        ("ensemble_evaluator", "ensemble_evaluator_sha256_at_freeze"),
        ("field_gate_source", "field_gate_source_sha256_at_freeze"),
        ("Q3_Q4_measurement_source", "Q3_Q4_measurement_source_sha256_at_freeze"),
        ("Q3_Q4_pass_source", "Q3_Q4_pass_source_sha256_at_freeze"),
    ):
        if sha256_file(resolve_path(repo, frozen[path_key])) != frozen[hash_key]:
            raise ValueError(f"V71 frozen development statistic changed: {path_key}")


def ensemble_copula_couple(
    rank_source: torch.Tensor, marginal_values: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Assign marginal order statistics by the rank-source member ordering."""
    if (
        rank_source.shape != marginal_values.shape
        or rank_source.ndim < 2
        or rank_source.shape[0] != 16
        or not torch.isfinite(rank_source).all()
        or not torch.isfinite(marginal_values).all()
    ):
        raise ValueError("V71 ECC tensor shape or finiteness differs")
    rank_order = torch.argsort(rank_source, dim=0, stable=True)
    sorted_marginal = torch.sort(marginal_values, dim=0, stable=True).values
    coupled = torch.empty_like(marginal_values)
    coupled.scatter_(0, rank_order, sorted_marginal)
    coupled_sorted = torch.sort(coupled, dim=0, stable=True).values
    exact_multiset = torch.equal(coupled_sorted, sorted_marginal)
    tied_voxel = torch.any(sorted_marginal[1:] == sorted_marginal[:-1], dim=0)
    tie_fraction = float(tied_voxel.double().mean().cpu())
    coupled_order = torch.argsort(coupled, dim=0, stable=True)
    mismatch = torch.any(coupled_order != rank_order, dim=0) & ~tied_voxel
    mismatch_fraction = float(mismatch.double().mean().cpu())
    diagnostics: dict[str, float | bool] = {
        "pre_inverse_sorted_latent_multiset_equal": exact_multiset,
        "maximum_pre_inverse_sorted_latent_multiset_error": float(
            torch.max(torch.abs(coupled_sorted - sorted_marginal)).cpu()
        ),
        "control_tied_voxel_fraction": tie_fraction,
        "candidate_rank_disagreement_fraction_excluding_control_ties": (
            mismatch_fraction
        ),
    }
    if not exact_multiset or mismatch_fraction != 0.0:
        raise RuntimeError("V71 ECC invariant differs")
    return coupled, diagnostics


def validate_preflight(
    path: Path,
    expected_sha256: str,
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    path = path.resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("V71 preflight hash differs")
    result = strict_json(path)
    if (
        result.get("schema") != PREFLIGHT_SCHEMA
        or result.get("status")
        != "complete_code_only_path_B_preflight_development_authorized"
        or result.get("program_sha256") != PROGRAM_SHA256
        or canonical_digest(result) != result.get("decision_digest_sha256")
        or result.get("preflight_pass") is not True
        or result.get("development_payload_semantics_accessed") is not False
        or result.get("development_truth_or_source_index_read") is not False
        or result.get("independent_EAGLE_accessed") is not False
        or result.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(result.get("code_commit")), commit)
    ):
        raise ValueError("V71 preflight authorization differs")
    return result
