#!/usr/bin/env python
"""Frozen V72 two-stage SQT integrity checks and transport primitives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = (
    "hong2021-v72-two-stage-conditioning-stratified-spatial-quantile-"
    "transport-program-v2"
)
PROGRAM_STATUS = (
    "refrozen_after_code_only_spatial_tie_audit_before_V72_"
    "implementation_or_fresh_payload_access"
)
PROGRAM_SHA256 = "83add01ae90cff6c3e1f656901daf7b9ed09a24c2f8bc59b6c52603531d8f9e4"
PROGRAM_FREEZE_COMMIT = "ac3977af9f87d0d558a446e46e02929620d9d438"
PREFLIGHT_SCHEMA = "hong2021-v72-code-only-two-stage-SQT-preflight-v1"
DECISION_SCHEMA = "hong2021-v72-single-stage-SQT-decision-v1"
ENSEMBLE_SCHEMA = "hong2021-v72-conditioning-stratified-spatial-quantile-ensemble-v1"
CANDIDATE = "conditioning_stratified_spatial_quantile_transport"
RAW = "raw_V70_latent_spatial_score"
CONTROL = "independent_voxel_V63_marginal"
ARMS = (CANDIDATE, RAW, CONTROL)
METHOD = "fixed_V70_within_stratum_spatial_rank_paired_V63_marginal"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    """Validate only local program ancestry; no fresh payload is touched."""
    repo = repo.resolve()
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V72 program hash differs")
    program = strict_json(path.resolve())
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("authorization", {}).get(
            "V72_program_and_single_stage_A_authorized"
        )
        is not True
        or program.get("authorization", {}).get(
            "independent_EAGLE_access_authorized"
        )
        is not False
    ):
        raise ValueError("V72 program schema, status, or authorization differs")
    parent = program["parent_evidence"]
    for key in (
        "preimplementation_spatial_tie_threshold_erratum",
        "v72_feasibility_audit",
        "v71_result_record",
        "v70_model_program",
        "v35_data_definition",
    ):
        if sha256_file(resolve_path(repo, parent[key])) != parent[f"{key}_sha256"]:
            raise ValueError(f"V72 local parent differs: {key}")
    audit = strict_json(resolve_path(repo, parent["v72_feasibility_audit"]))
    result = strict_json(resolve_path(repo, parent["v71_result_record"]))
    if (
        audit.get("status")
        != "complete_metadata_only_two_fresh_stages_available_candidate_not_yet_frozen"
        or audit.get("metadata_access", {}).get("validation_target_voxels_read")
        is not False
        or result.get("status")
        != "complete_single_use_development_failure_sealed_independent_gate_locked"
        or result.get("firewall", {}).get("independent_EAGLE_accessed") is not False
        or result.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V72 audit or V71 terminal evidence differs")
    _validate_selections(program)
    return program


def _validate_selections(program: dict[str, Any]) -> None:
    historical = program["historical_consumed_selection"]
    first = program["fresh_stage_A_selection"]
    second = program["fresh_stage_B_selection"]
    limits = {"TNG100": 93, "SIMBA": 64, "Swift": 148}
    for domain in DOMAIN_ORDER:
        old = list(map(int, historical[domain]))
        stage_a = list(map(int, first[domain]))
        stage_b = list(map(int, second[domain]))
        if (
            len(old) != 16
            or len(stage_a) != 16
            or len(stage_b) != 16
            or len(set(old)) != 16
            or len(set(stage_a)) != 16
            or len(set(stage_b)) != 16
            or set(old) & set(stage_a)
            or set(old) & set(stage_b)
            or set(stage_a) & set(stage_b)
            or min(stage_a + stage_b) < 0
            or max(stage_a + stage_b) >= limits[domain]
        ):
            raise ValueError(f"V72 {domain} fresh selections differ")


def authorize_parent_evidence(
    program: dict[str, Any], repo: Path, commit: str
) -> dict[str, Any]:
    """Byte-validate inherited fit and fresh files without reading datasets."""
    repo = repo.resolve()
    if not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit):
        raise ValueError("V72 code does not descend from the frozen program")
    frozen = program["frozen_inputs"]
    for key in (
        "v63_checkpoint",
        "conditioning_cache",
        "v70_checkpoint",
        "v70_training_report",
        "TNG100_validation_data",
        "TNG100_validation_cache",
        "SIMBA_validation_data",
        "SIMBA_validation_cache",
        "Swift_validation_data",
        "Swift_validation_cache",
    ):
        if sha256_file(resolve_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V72 frozen input differs: {key}")
    v71 = strict_json(resolve_path(repo, program["parent_evidence"]["v71_result_record"]))
    seal_row = v71["terminal_seal"]
    seal_path = Path(seal_row["path"]).resolve()
    if sha256_file(seal_path) != seal_row["sha256"]:
        raise ValueError("V72 V71 terminal seal hash differs")
    seal = strict_json(seal_path)
    if (
        seal.get("status") != "sealed_development_failure_independent_gate_locked"
        or canonical_digest(seal) != seal.get("decision_digest_sha256")
        or seal.get("decision_digest_sha256") != seal_row["decision_digest_sha256"]
        or seal.get("independent_EAGLE_accessed") is not False
        or seal.get("independent_gate_locked") is not True
    ):
        raise ValueError("V72 V71 terminal seal differs")
    return {
        "v71_terminal_seal": str(seal_path),
        "v71_terminal_seal_sha256": seal_row["sha256"],
        "v70_checkpoint_sha256": frozen["v70_checkpoint_sha256"],
        "v70_training_report_sha256": frozen["v70_training_report_sha256"],
    }


def validate_frozen_gate_sources(program: dict[str, Any], repo: Path) -> None:
    frozen = program["unchanged_field_gate"]
    for path_key, hash_key in (
        ("ensemble_evaluator", "ensemble_evaluator_sha256_at_freeze"),
        ("field_gate_source", "field_gate_source_sha256_at_freeze"),
        ("Q3_Q4_measurement_source", "Q3_Q4_measurement_source_sha256_at_freeze"),
        ("Q3_Q4_historical_pass_source", "Q3_Q4_historical_pass_source_sha256_at_freeze"),
    ):
        if sha256_file(resolve_path(repo, frozen[path_key])) != frozen[hash_key]:
            raise ValueError(f"V72 frozen gate source changed: {path_key}")


def stage_selection(program: dict[str, Any], stage: str) -> dict[str, list[int]]:
    if stage not in ("A", "B"):
        raise ValueError("V72 stage must be A or B")
    row = program[f"fresh_stage_{stage}_selection"]
    return {domain: list(map(int, row[domain])) for domain in DOMAIN_ORDER}


def conditioning_strata(score: torch.Tensor, strata: int = 16) -> torch.Tensor:
    if score.numel() != 64**3 or strata != 16 or score.numel() % strata:
        raise ValueError("V72 conditioning-score or strata shape differs")
    flat = score.reshape(-1)
    if not torch.isfinite(flat).all():
        raise ValueError("V72 conditioning score is nonfinite")
    return torch.argsort(flat, stable=True).reshape(strata, -1)


def spatial_quantile_transport(
    rank_source: torch.Tensor,
    marginal_values: torch.Tensor,
    strata_positions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Transport paired marginal values using within-member spatial ranks."""
    if (
        rank_source.shape != marginal_values.shape
        or rank_source.shape != (16, 1, 64, 64, 64)
        or strata_positions.shape != (16, 16384)
        or not torch.isfinite(rank_source).all()
        or not torch.isfinite(marginal_values).all()
    ):
        raise ValueError("V72 SQT tensor shape or finiteness differs")
    rank_flat = rank_source.reshape(16, -1)
    marginal_flat = marginal_values.reshape(16, -1)
    coupled_flat = torch.empty_like(marginal_flat)
    exact = True
    maximum_error = 0.0
    tied_values = 0
    total_values = 0
    mismatch_values = 0
    comparable_values = 0
    for positions in strata_positions:
        source = rank_flat[:, positions]
        marginal = marginal_flat[:, positions]
        source_order = torch.argsort(source, dim=1, stable=True)
        sorted_marginal = torch.sort(marginal, dim=1, stable=True).values
        coupled = torch.empty_like(marginal)
        coupled.scatter_(1, source_order, sorted_marginal)
        coupled_sorted = torch.sort(coupled, dim=1, stable=True).values
        current_exact = torch.equal(coupled_sorted, sorted_marginal)
        exact = exact and current_exact
        maximum_error = max(
            maximum_error,
            float(torch.max(torch.abs(coupled_sorted - sorted_marginal)).cpu()),
        )
        adjacent = sorted_marginal[:, 1:] == sorted_marginal[:, :-1]
        tied_slots = torch.zeros_like(sorted_marginal, dtype=torch.bool)
        tied_slots[:, 1:] |= adjacent
        tied_slots[:, :-1] |= adjacent
        coupled_order = torch.argsort(coupled, dim=1, stable=True)
        mismatch = (coupled_order != source_order) & ~tied_slots
        tied_values += int(tied_slots.sum().cpu())
        total_values += tied_slots.numel()
        mismatch_values += int(mismatch.sum().cpu())
        comparable_values += int((~tied_slots).sum().cpu())
        coupled_flat[:, positions] = coupled
    diagnostics: dict[str, float | bool] = {
        "pre_inverse_stratum_multiset_equal": exact,
        "maximum_pre_inverse_stratum_multiset_error": maximum_error,
        "marginal_tied_voxel_fraction": tied_values / total_values,
        "rank_disagreement_fraction_excluding_marginal_ties": (
            mismatch_values / comparable_values if comparable_values else 0.0
        ),
    }
    if not exact or maximum_error != 0.0 or mismatch_values:
        raise RuntimeError("V72 SQT invariant differs")
    return coupled_flat.reshape_as(marginal_values), diagnostics


def scalar_energy_score(ensemble: np.ndarray, truth: float) -> float:
    values = np.asarray(ensemble, dtype=np.float64).reshape(-1)
    if len(values) != 16 or not np.isfinite(values).all() or not np.isfinite(truth):
        raise ValueError("V72 scalar energy-score input differs")
    first = float(np.mean(np.abs(values - float(truth))))
    distance = np.abs(values[:, None] - values[None, :])
    second = float(distance[~np.eye(len(values), dtype=bool)].mean())
    return first - 0.5 * second


def validate_preflight(
    path: Path, expected_sha256: str, repo: Path, commit: str
) -> dict[str, Any]:
    if sha256_file(path.resolve()) != expected_sha256:
        raise ValueError("V72 preflight hash differs")
    result = strict_json(path.resolve())
    if (
        result.get("schema") != PREFLIGHT_SCHEMA
        or result.get("status")
        != "complete_code_only_V72_preflight_stage_A_authorized"
        or result.get("program_sha256") != PROGRAM_SHA256
        or canonical_digest(result) != result.get("decision_digest_sha256")
        or result.get("preflight_pass") is not True
        or result.get("fresh_input_or_target_dataset_read") is not False
        or result.get("stage_A_metric_read") is not False
        or result.get("stage_B_metric_read") is not False
        or result.get("independent_EAGLE_accessed") is not False
        or result.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(result.get("code_commit")), commit)
    ):
        raise ValueError("V72 preflight authorization differs")
    return result


def validate_stage_A_pass(
    program: dict[str, Any], path: Path, expected_sha256: str,
    preflight_sha256: str, repo: Path, commit: str,
) -> dict[str, Any]:
    expected_path = Path(program["output_roots"]["stage_A"]) / "decision.json"
    if path.resolve() != expected_path.resolve() or sha256_file(path) != expected_sha256:
        raise ValueError("V72 stage-A decision path or hash differs")
    decision = strict_json(path)
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("status") != "complete_single_V72_SQT_stage_gate"
        or decision.get("stage") != "A"
        or decision.get("program_sha256") != PROGRAM_SHA256
        or decision.get("preflight_sha256") != preflight_sha256
        or canonical_digest(decision) != decision.get("decision_digest_sha256")
        or decision.get("stage_pass") is not True
        or decision.get("candidate_selected") is not True
        or decision.get("stage_B_accessed") is not False
        or decision.get("independent_EAGLE_accessed") is not False
        or decision.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(decision.get("gate_code_commit")), commit)
    ):
        raise ValueError("V72 stage-A pass authorization differs")
    return decision
