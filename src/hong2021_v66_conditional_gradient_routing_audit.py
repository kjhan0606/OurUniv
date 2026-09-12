#!/usr/bin/env python
"""No-refit final-output-layer conditional gradient-routing audit for V66."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v31_copula import load_model as load_copula
from hong2021_v48_train import load_cache
from hong2021_v50_network import bounded_mixture_log_probability
from hong2021_v62_conditional_moment_gradient_audit import conditional_log_moment_score
from hong2021_v63_preflight import _path, load_program as load_v63_program
from hong2021_v63_train import _is_ancestor
from hong2021_v63_train_gate import _load_fit
from hong2021_v64_sampler_alignment_audit import _pair_indices, empirical_pair_moments
from hong2021_v65_structure_factorization_audit import (
    _close_handles,
    _open_train_handles,
    _query_batch,
    _rank_batch,
    load_program as load_v65_program,
)


PROGRAM_SHA256 = "f3aaa8ee8852341675bcdec264ba991a4d23b4ed6550cfa309e1b85d5b7730ca"
PROGRAM_SCHEMA = "hong2021-v66-train-only-final-output-layer-conditional-gradient-routing-audit-program-v1"
SCHEMA = "hong2021-v66-train-only-final-output-layer-conditional-gradient-routing-audit-v1"
PROGRAM_FREEZE_COMMIT = "e17fa27b09b969be97da28d96b716e3154dafc2d"
PAIR_COEFFICIENT = 0.1


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V66 {label} hash differs")
    return _json(path)


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V66 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v65_result_record"]),
        parent["v65_result_record_sha256"],
        "V65 result record",
    )
    causal = record.get("causal_factorization", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("candidate_selected")
        is not parent["required_candidate_selected"]
        or causal.get("rank_dependence_causal")
        is not parent["required_rank_dependence_causal"]
        or causal.get("query_parameter_spatial_arrangement_causal")
        is not parent["required_query_parameter_spatial_arrangement_causal"]
        or causal.get("direct_pair_bias_gradient_coherent")
        is not parent["required_direct_pair_bias_gradient_coherent"]
        or firewall.get("training_or_refit_performed")
        is not parent["required_training_or_refit_performed"]
        or firewall.get("new_development_accessed")
        is not parent["required_new_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V66 parent result or firewall differs")
    frozen = program["frozen_inputs"]
    for key, value in frozen.items():
        if key.endswith("_sha256"):
            continue
        digest = frozen.get(f"{key}_sha256")
        if digest is not None and sha256_file(_path(repo, value)) != digest:
            raise ValueError(f"V66 frozen input differs: {key}")
    v65_audit = _json(_path(repo, frozen["v65_audit"]))
    if (
        v65_audit.get("decision_digest_sha256")
        != frozen["v65_audit_decision_digest_sha256"]
        or canonical_digest(v65_audit)
        != frozen["v65_audit_decision_digest_sha256"]
        or v65_audit.get("candidate_selected") is not False
        or v65_audit.get("query_parameter_spatial_arrangement_causal") is not True
        or v65_audit.get("direct_pair_gradient_coherent") is not False
        or v65_audit.get("training_or_refit_performed") is not False
        or v65_audit.get("new_development_accessed") is not False
        or v65_audit.get("independent_gate_locked") is not True
    ):
        raise ValueError("V66 sealed V65 audit differs")
    _, v35, gate = load_v65_program(_path(repo, frozen["v65_program"]), repo)
    return program, v35, gate, v65_audit


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def gradient_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair = np.asarray([row["joint_pair_gradient"] for row in rows], dtype=np.float64)
    nll = np.asarray(
        [row["joint_bounded_NLL_gradient"] for row in rows], dtype=np.float64
    )
    for index, row in enumerate(rows):
        global_other = np.delete(pair, index, axis=0).mean(axis=0)
        domain_indices = [
            other
            for other, candidate in enumerate(rows)
            if other != index and candidate["domain"] == row["domain"]
        ]
        domain_other = pair[domain_indices].mean(axis=0)
        row["joint_pair_gradient_leave_one_out_global_mean_cosine"] = _cosine(
            pair[index], global_other
        )
        row["joint_pair_gradient_leave_one_out_same_domain_mean_cosine"] = _cosine(
            pair[index], domain_other
        )
    domain_rows = {}
    for domain in DOMAIN_ORDER:
        selected = [row for row in rows if row["domain"] == domain]
        global_cosines = np.asarray(
            [
                row["joint_pair_gradient_leave_one_out_global_mean_cosine"]
                for row in selected
            ],
            dtype=np.float64,
        )
        within_cosines = np.asarray(
            [
                row[
                    "joint_pair_gradient_leave_one_out_same_domain_mean_cosine"
                ]
                for row in selected
            ],
            dtype=np.float64,
        )
        domain_rows[domain] = {
            "median_leave_one_out_global_mean_cosine": float(
                np.median(global_cosines)
            ),
            "negative_global_mean_cosine_fraction": float(
                np.mean(global_cosines < 0.0)
            ),
            "median_leave_one_out_same_domain_mean_cosine": float(
                np.median(within_cosines)
            ),
            "negative_same_domain_mean_cosine_fraction": float(
                np.mean(within_cosines < 0.0)
            ),
        }
    global_cosines = np.asarray(
        [
            row["joint_pair_gradient_leave_one_out_global_mean_cosine"]
            for row in rows
        ],
        dtype=np.float64,
    )
    singular_values = np.linalg.svd(pair, compute_uv=False)
    squared = np.square(singular_values)
    pair_mean = pair.mean(axis=0)
    nll_mean = nll.mean(axis=0)
    combined = nll_mean + PAIR_COEFFICIENT * pair_mean
    pair_norm = float(np.linalg.norm(pair_mean))
    nll_norm = float(np.linalg.norm(nll_mean))
    return {
        "query_rows": rows,
        "domains": domain_rows,
        "global_median_leave_one_out_cosine": float(np.median(global_cosines)),
        "global_negative_cosine_fraction": float(np.mean(global_cosines < 0.0)),
        "singular_values": singular_values.tolist(),
        "singular_value_squared_norm_fractions": (squared / squared.sum()).tolist(),
        "aggregate_joint_pair_gradient": pair_mean.tolist(),
        "aggregate_joint_bounded_NLL_gradient": nll_mean.tolist(),
        "aggregate_joint_pair_gradient_L2": pair_norm,
        "aggregate_joint_bounded_NLL_gradient_L2": nll_norm,
        "coefficient_0_1_scaled_pair_to_bounded_NLL_L2_ratio": (
            PAIR_COEFFICIENT * pair_norm / max(nll_norm, 1.0e-300)
        ),
        "aggregate_pair_to_bounded_NLL_cosine": _cosine(pair_mean, nll_mean),
        "combined_update_pair_descent_dot_product": float(
            np.dot(pair_mean, combined)
        ),
        "combined_update_bounded_NLL_descent_dot_product": float(
            np.dot(nll_mean, combined)
        ),
    }


def selection_flags(
    summary: dict[str, Any],
    maximum_bias_difference: float,
    weight_norms: list[float],
    bias_tolerance: float,
) -> tuple[bool, bool, bool]:
    routing = bool(
        summary["global_median_leave_one_out_cosine"] >= 0.5
        and summary["global_negative_cosine_fraction"] <= 0.25
        and summary["global_negative_cosine_fraction"] < 5.0 / 12.0
        and all(
            summary["domains"][domain][
                "median_leave_one_out_global_mean_cosine"
            ]
            > 0.0
            for domain in DOMAIN_ORDER
        )
        and all(value > 0.0 for value in weight_norms)
        and maximum_bias_difference <= bias_tolerance
    )
    ratio = summary["coefficient_0_1_scaled_pair_to_bounded_NLL_L2_ratio"]
    scale = bool(0.01 <= ratio <= 100.0)
    compatible = bool(
        summary["combined_update_pair_descent_dot_product"] > 0.0
        and summary["combined_update_bounded_NLL_descent_dot_product"] > 0.0
    )
    return routing, scale, compatible


def classify(
    integrity_pass: bool,
    routing_supported: bool,
    scale_pass: bool,
    compatible: bool,
) -> tuple[str, str, bool]:
    if not integrity_pass:
        return (
            "conditional_gradient_routing_audit_failed_integrity",
            "stop_before_refit_and_preserve_the_failed_train_only_audit",
            False,
        )
    if routing_supported and scale_pass and compatible:
        return (
            "final_output_layer_conditionally_routes_a_coherent_pair_correction",
            "freeze_one_final_output_layer_only_coefficient_0_1_pair_objective_model_before_refit",
            True,
        )
    if not routing_supported:
        return (
            "final_output_layer_features_do_not_resolve_object_level_pair_gradient_conflict",
            "stop_before_refit_and_design_a_nonlocal_train_only_structure_hypothesis",
            False,
        )
    if not scale_pass:
        return (
            "conditional_pair_gradient_has_an_unsafe_optimization_scale",
            "stop_before_refit_and_reassess_the_estimating_score",
            False,
        )
    return (
        "conditional_pair_gradient_conflicts_with_the_sealed_likelihood_objective",
        "stop_before_refit_and_design_a_predeclared_gradient_compatibility_rule",
        False,
    )


def _trim_mapping(v65_audit: dict[str, Any], members: int) -> dict[str, Any]:
    result = {}
    for domain in DOMAIN_ORDER:
        result[domain] = []
        for query in v65_audit["donor_mappings"]["source_balanced"][domain][
            :4
        ]:
            result[domain].append(
                {
                    "query_position": int(query["query_position"]),
                    "query_object_index": int(query["query_object_index"]),
                    "members": query["members"][:members],
                }
            )
    return result


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, gate, v65_audit = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V66 audit requires clean Lageunha with frozen ancestry")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V66 audit requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    frozen = program["frozen_inputs"]
    v63, _, _, _, _, _, _ = load_v63_program(
        _path(repo, frozen["v63_program"]), repo
    )
    boundaries = {
        domain: float(v63["sealed_q99_9_backbone_boundaries"][domain])
        for domain in DOMAIN_ORDER
    }
    model, _ = _load_fit(
        _path(repo, frozen["v63_checkpoint"]),
        frozen["v63_checkpoint_sha256"],
        _path(repo, frozen["v63_training_report"]),
        frozen["v63_training_report_sha256"],
        frozen["v56_grid_sha256"],
        frozen["v54_threshold_selection_sha256"],
        frozen["v63_preflight_sha256"],
        frozen["conditioning_cache_sha256"],
        v63["frozen_inputs"]["support_selection_sha256"],
        boundaries,
        repo,
        commit,
    )
    expected_weight_shape = tuple(
        program["gradient_targets"]["conditional_output_weight"]["shape"]
    )
    if (
        tuple(model.output.weight.shape) != expected_weight_shape
        or model.output.weight.numel()
        != program["gradient_targets"]["conditional_output_weight"]["components"]
        or model.output.bias.numel()
        != program["gradient_targets"]["bias_control"]["components"]
    ):
        raise RuntimeError("V66 final output-layer shape differs")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.output.weight.requires_grad_(True)
    model.output.bias.requires_grad_(True)
    initial_weight = model.output.weight.detach().clone()
    initial_bias = model.output.bias.detach().clone()
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    queries = {
        domain: list(map(int, program["immutable_gradient_queries"][domain]))
        for domain in DOMAIN_ORDER
    }
    members = int(program["frozen_rank_and_pair_probe"]["rank_members_per_query"])
    mapping = _trim_mapping(v65_audit, members)
    for domain in DOMAIN_ORDER:
        if [row["query_object_index"] for row in mapping[domain]] != queries[domain]:
            raise ValueError("V66 query-to-V65 donor mapping binding differs")
    copula = load_copula(
        _path(repo, frozen["conditional_copula_artifact"]),
        frozen["conditional_copula_artifact_sha256"],
    )
    pairs = _pair_indices(
        int(program["frozen_rank_and_pair_probe"]["anchor_seed"]),
        int(program["frozen_rank_and_pair_probe"]["anchors_per_query"]),
    )
    v65_bias_rows = {
        (row["domain"], int(row["query_object_index"])): np.asarray(
            row["pair_gradient"], dtype=np.float64
        )
        for row in v65_audit["gradient_coherence"]["query_rows"]
    }
    rows: list[dict[str, Any]] = []
    maximum_inverse_error = 0.0
    maximum_bias_difference = 0.0
    rank_hasher = hashlib.sha256()
    handles = _open_train_handles(v35)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for position in range(4):
            condition, target, backbone = _query_batch(
                handles, prepared, queries, position, device
            )
            rank_numpy = _rank_batch(handles, mapping, position, copula, rank_hasher)
            ranks = torch.from_numpy(rank_numpy).to(device)
            output = model(condition).float()
            predicted, truth, inverse_error = empirical_pair_moments(
                output,
                target,
                backbone,
                ranks,
                target_mean,
                target_std,
                pairs,
                members,
            )
            maximum_inverse_error = max(maximum_inverse_error, inverse_error)
            for domain_index, domain in enumerate(DOMAIN_ORDER):
                start = 3 * domain_index
                pair_objective = conditional_log_moment_score(
                    predicted[start : start + 3], truth[start : start + 3]
                )
                nll_objective = -bounded_mixture_log_probability(
                    output[domain_index : domain_index + 1],
                    target[domain_index : domain_index + 1],
                ).mean()
                pair_weight, pair_bias = torch.autograd.grad(
                    pair_objective,
                    (model.output.weight, model.output.bias),
                    retain_graph=True,
                )
                nll_weight, nll_bias = torch.autograd.grad(
                    nll_objective,
                    (model.output.weight, model.output.bias),
                    retain_graph=domain_index < len(DOMAIN_ORDER) - 1,
                )
                vectors = (pair_weight, pair_bias, nll_weight, nll_bias)
                if not all(
                    bool(torch.isfinite(value).all().detach().cpu())
                    for value in vectors
                ):
                    raise RuntimeError("V66 nonfinite final-layer gradient")
                pair_weight_np = pair_weight.detach().double().cpu().numpy().reshape(-1)
                pair_bias_np = pair_bias.detach().double().cpu().numpy().reshape(-1)
                nll_weight_np = nll_weight.detach().double().cpu().numpy().reshape(-1)
                nll_bias_np = nll_bias.detach().double().cpu().numpy().reshape(-1)
                reference = v65_bias_rows[(domain, queries[domain][position])]
                bias_difference = float(np.max(np.abs(pair_bias_np - reference)))
                maximum_bias_difference = max(maximum_bias_difference, bias_difference)
                joint_pair = np.concatenate((pair_weight_np, pair_bias_np))
                joint_nll = np.concatenate((nll_weight_np, nll_bias_np))
                rows.append(
                    {
                        "domain": domain,
                        "query_position": position,
                        "query_object_index": queries[domain][position],
                        "pair_objective": float(pair_objective.detach().cpu()),
                        "bounded_NLL_objective": float(nll_objective.detach().cpu()),
                        "pair_weight_gradient": pair_weight_np.tolist(),
                        "pair_bias_gradient": pair_bias_np.tolist(),
                        "bounded_NLL_weight_gradient": nll_weight_np.tolist(),
                        "bounded_NLL_bias_gradient": nll_bias_np.tolist(),
                        "joint_pair_gradient": joint_pair.tolist(),
                        "joint_bounded_NLL_gradient": joint_nll.tolist(),
                        "pair_weight_gradient_L2": float(np.linalg.norm(pair_weight_np)),
                        "pair_bias_gradient_L2": float(np.linalg.norm(pair_bias_np)),
                        "bounded_NLL_weight_gradient_L2": float(
                            np.linalg.norm(nll_weight_np)
                        ),
                        "bounded_NLL_bias_gradient_L2": float(
                            np.linalg.norm(nll_bias_np)
                        ),
                        "V65_bias_gradient_maximum_absolute_difference": bias_difference,
                    }
                )
            del condition, target, backbone, ranks, output, predicted, truth
    finally:
        _close_handles(handles)
        prepared.close()
    if (
        not torch.equal(initial_weight, model.output.weight.detach())
        or not torch.equal(initial_bias, model.output.bias.detach())
    ):
        raise RuntimeError("V66 model parameters changed without authorization")
    summary = gradient_summary(rows)
    weight_norms = [row["pair_weight_gradient_L2"] for row in rows]
    bias_tolerance = float(
        program["gradient_targets"]["bias_control"][
            "required_reproduction_maximum_absolute_difference"
        ]
    )
    routing, scale, compatible = selection_flags(
        summary, maximum_bias_difference, weight_norms, bias_tolerance
    )
    peak = int(torch.cuda.max_memory_allocated(device))
    memory_pass = peak < int(program["resource_gate"]["peak_allocated_bytes_limit"])
    finite_values = [
        maximum_inverse_error,
        maximum_bias_difference,
        summary["global_median_leave_one_out_cosine"],
        summary["global_negative_cosine_fraction"],
        summary["coefficient_0_1_scaled_pair_to_bounded_NLL_L2_ratio"],
        summary["aggregate_pair_to_bounded_NLL_cosine"],
        summary["combined_update_pair_descent_dot_product"],
        summary["combined_update_bounded_NLL_descent_dot_product"],
    ]
    numerical_pass = bool(
        len(rows) == 12
        and maximum_inverse_error <= 2.0e-6
        and len(summary["aggregate_joint_pair_gradient"]) == 495
        and all(value > 0.0 for value in weight_norms)
        and all(math.isfinite(value) for value in finite_values)
    )
    integrity_pass = numerical_pass and memory_pass
    classification, next_step, selected = classify(
        integrity_pass, routing, scale, compatible
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_no_refit_train_only_conditional_gradient_routing_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_sha256": frozen["v63_checkpoint_sha256"],
        "training_report_sha256": frozen["v63_training_report_sha256"],
        "train_gate_sha256": frozen["v63_train_gate_sha256"],
        "train_gate_decision_digest_sha256": gate["decision_digest_sha256"],
        "immutable_train_queries": queries,
        "source_balanced_donor_mapping": mapping,
        "rank_field_stream_sha256": rank_hasher.hexdigest(),
        "gradient_summary": summary,
        "maximum_V65_bias_gradient_absolute_difference": maximum_bias_difference,
        "V65_bias_gradient_reproduction_tolerance": bias_tolerance,
        "conditional_routing_supported": routing,
        "optimization_scale_pass": scale,
        "combined_update_compatible": compatible,
        "maximum_inverse_CDF_error": maximum_inverse_error,
        "numerical_pass": numerical_pass,
        "peak_allocated_bytes": peak,
        "memory_pass": memory_pass,
        "candidate_selected": selected,
        "classification": classification,
        "next": next_step,
        "training_or_refit_performed": False,
        "optimizer_step_performed": False,
        "validation_accessed": False,
        "new_development_accessed": False,
        "development_rank_or_selection_accessed": False,
        "posthoc_coefficient_or_threshold_tuning_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V66 refuses existing audit output")
    result = audit(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
