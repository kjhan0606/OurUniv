#!/usr/bin/env python
"""No-refit domain-balanced population pair-objective audit for V68."""
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
from hong2021_v63_preflight import _path, load_program as load_v63_program
from hong2021_v63_train import _is_ancestor
from hong2021_v63_train_gate import _load_fit
from hong2021_v64_sampler_alignment_audit import _pair_indices, empirical_pair_moments
from hong2021_v65_structure_factorization_audit import (
    _close_handles,
    _open_train_handles,
    _query_batch,
    _rank_batch,
)


PROGRAM_SHA256 = "818433d4567b67a3f9ee0eca2271d25da720cd2baf4b4ef042e96b0ad90a852c"
PROGRAM_SCHEMA = "hong2021-v68-train-only-domain-balanced-population-pair-objective-stability-audit-program-v1"
SCHEMA = "hong2021-v68-train-only-domain-balanced-population-pair-objective-stability-audit-v1"
PROGRAM_FREEZE_COMMIT = "74fae2ba4e8ed8782c017a42266e1c7c9a66ba16"
PAIR_COEFFICIENT = 0.1


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V68 {label} hash differs")
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
        raise ValueError("V68 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v67_result_record"]),
        parent["v67_result_record_sha256"],
        "V67 result record",
    )
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("candidate_selected")
        is not parent["required_candidate_selected"]
        or firewall.get("training_or_refit_performed")
        is not parent["required_training_or_refit_performed"]
        or firewall.get("new_development_accessed")
        is not parent["required_new_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V68 parent result or firewall differs")
    frozen = program["frozen_inputs"]
    for key, value in frozen.items():
        if key.endswith("_sha256"):
            continue
        digest = frozen.get(f"{key}_sha256")
        if digest is not None and sha256_file(_path(repo, value)) != digest:
            raise ValueError(f"V68 frozen input differs: {key}")
    v65_audit = _json(_path(repo, frozen["v65_audit"]))
    if (
        v65_audit.get("decision_digest_sha256")
        != frozen["v65_audit_decision_digest_sha256"]
        or canonical_digest(v65_audit)
        != frozen["v65_audit_decision_digest_sha256"]
        or v65_audit.get("training_or_refit_performed") is not False
        or v65_audit.get("new_development_accessed") is not False
        or v65_audit.get("independent_gate_locked") is not True
    ):
        raise ValueError("V68 sealed V65 audit differs")
    v63, v35, _, _, _, _, _ = load_v63_program(
        _path(repo, frozen["v63_program"]), repo
    )
    gate = _verified_json(
        _path(repo, frozen["v63_train_gate"]),
        frozen["v63_train_gate_sha256"],
        "V63 train gate",
    )
    if (
        gate.get("train_mechanism_pass") is not True
        or gate.get("development_accessed") is not False
        or gate.get("independent_gate_locked") is not True
        or canonical_digest(gate) != gate.get("decision_digest_sha256")
    ):
        raise ValueError("V68 V63 train gate differs")
    return program, v63, v35, v65_audit


def _trim_mapping(v65_audit: dict[str, Any], members: int) -> dict[str, Any]:
    result = {}
    for domain in DOMAIN_ORDER:
        result[domain] = []
        for query in v65_audit["donor_mappings"]["source_balanced"][domain]:
            result[domain].append(
                {
                    "query_position": int(query["query_position"]),
                    "query_object_index": int(query["query_object_index"]),
                    "members": query["members"][:members],
                }
            )
    return result


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def population_row(
    positions: list[int],
    predicted: np.ndarray,
    truth: np.ndarray,
    jacobian: np.ndarray,
    nll_gradient: np.ndarray,
) -> dict[str, Any]:
    mean_predicted = predicted[:, positions].mean(axis=1)
    mean_truth = truth[:, positions].mean(axis=1)
    log_ratio = np.log(mean_predicted / mean_truth)
    mean_jacobian = jacobian[:, positions].mean(axis=1)
    pair_gradient = np.zeros(495, dtype=np.float64)
    for domain_index in range(3):
        for separation in range(3):
            pair_gradient += (
                log_ratio[domain_index, separation]
                / (9.0 * mean_predicted[domain_index, separation])
                * mean_jacobian[domain_index, separation]
            )
    nll = nll_gradient[:, positions].mean(axis=(0, 1))
    combined = nll + PAIR_COEFFICIENT * pair_gradient
    pair_norm = float(np.linalg.norm(pair_gradient))
    nll_norm = float(np.linalg.norm(nll))
    return {
        "positions": positions,
        "mean_predicted_pair": mean_predicted.tolist(),
        "mean_truth_pair": mean_truth.tolist(),
        "population_log_ratio": log_ratio.tolist(),
        "population_score": float(0.5 * np.mean(np.square(log_ratio))),
        "pair_gradient": pair_gradient.tolist(),
        "bounded_NLL_gradient": nll.tolist(),
        "pair_gradient_L2": pair_norm,
        "bounded_NLL_gradient_L2": nll_norm,
        "coefficient_0_1_pair_to_bounded_NLL_L2_ratio": (
            PAIR_COEFFICIENT * pair_norm / max(nll_norm, 1.0e-300)
        ),
        "pair_to_bounded_NLL_cosine": _cosine(pair_gradient, nll),
        "combined_update_pair_descent_dot_product": float(
            np.dot(pair_gradient, combined)
        ),
        "combined_update_bounded_NLL_descent_dot_product": float(
            np.dot(nll, combined)
        ),
    }


def classify(
    integrity_pass: bool, value_stable: bool, gradient_coherent: bool, safe: bool
) -> tuple[str, str, bool]:
    if not integrity_pass:
        return (
            "population_pair_objective_audit_failed_integrity",
            "stop_before_refit_and_preserve_the_failed_train_only_audit",
            False,
        )
    if value_stable and gradient_coherent and safe:
        return (
            "domain_balanced_population_pair_objective_is_stable_and_optimization_safe",
            "freeze_one_final_output_layer_population_pair_objective_model_before_refit",
            True,
        )
    return (
        "domain_balanced_population_pair_objective_is_not_train_split_stable",
        "stop_before_refit_and_do_not_add_a_pair_objective",
        False,
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v63, v35, v65_audit = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V68 audit requires clean Lageunha with frozen ancestry")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V68 audit requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    frozen = program["frozen_inputs"]
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
        domain: list(map(int, v65_audit["immutable_train_queries"][domain]))
        for domain in DOMAIN_ORDER
    }
    members = int(program["rank_and_pair_probe"]["rank_members_per_query"])
    mapping = _trim_mapping(v65_audit, members)
    copula = load_copula(
        _path(repo, frozen["conditional_copula_artifact"]),
        frozen["conditional_copula_artifact_sha256"],
    )
    pairs = _pair_indices(
        int(program["rank_and_pair_probe"]["anchor_seed"]),
        int(program["rank_and_pair_probe"]["anchors_per_query"]),
    )
    predicted = np.empty((3, 16, 3), dtype=np.float64)
    truth = np.empty_like(predicted)
    jacobian = np.empty((3, 16, 3, 495), dtype=np.float64)
    nll_gradient = np.empty((3, 16, 495), dtype=np.float64)
    maximum_inverse_error = 0.0
    rank_hasher = hashlib.sha256()
    handles = _open_train_handles(v35)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for position in range(16):
            condition, target, backbone = _query_batch(
                handles, prepared, queries, position, device
            )
            rank_numpy = _rank_batch(handles, mapping, position, copula, rank_hasher)
            ranks = torch.from_numpy(rank_numpy).to(device)
            output = model(condition).float()
            pair_predicted, pair_truth, inverse_error = empirical_pair_moments(
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
            pair_predicted = pair_predicted.reshape(3, 3)
            pair_truth = pair_truth.reshape(3, 3)
            predicted[:, position] = pair_predicted.detach().cpu().numpy()
            truth[:, position] = pair_truth.detach().cpu().numpy()
            for domain_index in range(3):
                for separation in range(3):
                    weight, bias = torch.autograd.grad(
                        pair_predicted[domain_index, separation],
                        (model.output.weight, model.output.bias),
                        retain_graph=True,
                    )
                    jacobian[domain_index, position, separation] = np.concatenate(
                        (
                            weight.detach().double().cpu().numpy().reshape(-1),
                            bias.detach().double().cpu().numpy().reshape(-1),
                        )
                    )
            for domain_index in range(3):
                objective = -bounded_mixture_log_probability(
                    output[domain_index : domain_index + 1],
                    target[domain_index : domain_index + 1],
                ).mean()
                weight, bias = torch.autograd.grad(
                    objective,
                    (model.output.weight, model.output.bias),
                    retain_graph=domain_index < 2,
                )
                nll_gradient[domain_index, position] = np.concatenate(
                    (
                        weight.detach().double().cpu().numpy().reshape(-1),
                        bias.detach().double().cpu().numpy().reshape(-1),
                    )
                )
            del condition, target, backbone, ranks, output, pair_predicted, pair_truth
    finally:
        _close_handles(handles)
        prepared.close()
    if (
        not torch.equal(initial_weight, model.output.weight.detach())
        or not torch.equal(initial_bias, model.output.bias.detach())
    ):
        raise RuntimeError("V68 model parameters changed without authorization")
    full = population_row(list(range(16)), predicted, truth, jacobian, nll_gradient)
    fold_positions = program["immutable_queries_and_folds"]["fold_positions"]
    folds = {
        name: population_row(list(map(int, positions)), predicted, truth, jacobian, nll_gradient)
        for name, positions in fold_positions.items()
    }
    fold_gradients = np.asarray(
        [folds[name]["pair_gradient"] for name in fold_positions], dtype=np.float64
    )
    cosines = []
    for index, name in enumerate(fold_positions):
        cosine = _cosine(
            fold_gradients[index], np.delete(fold_gradients, index, axis=0).mean(axis=0)
        )
        folds[name]["pair_gradient_leave_one_out_mean_cosine"] = cosine
        cosines.append(cosine)
    v65_predicted = np.empty_like(predicted)
    v65_truth = np.empty_like(truth)
    for domain_index, domain in enumerate(DOMAIN_ORDER):
        for position, row in enumerate(v65_audit["objects"][domain]):
            for separation, item in enumerate(
                row["controls"]["source_balanced"]["separations"]
            ):
                v65_predicted[domain_index, position, separation] = item[
                    "predicted_pair_mean"
                ]
                v65_truth[domain_index, position, separation] = item["truth_pair_mean"]
    rank16_log = np.log(v65_predicted.mean(axis=1) / v65_truth.mean(axis=1))
    rank8_log = np.asarray(full["population_log_ratio"])
    rank_reproduction_error = float(np.max(np.abs(rank8_log - rank16_log)))
    fold_value_error = max(
        float(
            np.max(
                np.abs(np.asarray(folds[name]["population_log_ratio"]) - rank8_log)
            )
        )
        for name in fold_positions
    )
    value_stable = bool(
        rank_reproduction_error <= math.log(1.25)
        and fold_value_error <= math.log(2.0)
    )
    gradient_coherent = bool(
        min(cosines) > 0.0
        and float(np.median(cosines)) >= 0.5
        and np.mean(np.asarray(cosines) < 0.0) == 0.0
    )
    safe = all(
        0.01 <= folds[name]["coefficient_0_1_pair_to_bounded_NLL_L2_ratio"] <= 100.0
        and folds[name]["combined_update_pair_descent_dot_product"] > 0.0
        and folds[name]["combined_update_bounded_NLL_descent_dot_product"] > 0.0
        for name in fold_positions
    )
    peak = int(torch.cuda.max_memory_allocated(device))
    memory_pass = peak < int(program["resource_gate"]["peak_allocated_bytes_limit"])
    finite = np.concatenate(
        (
            predicted.reshape(-1),
            truth.reshape(-1),
            jacobian.reshape(-1),
            nll_gradient.reshape(-1),
            np.asarray(cosines),
            [maximum_inverse_error, rank_reproduction_error, fold_value_error],
        )
    )
    numerical_pass = bool(
        maximum_inverse_error <= 2.0e-6
        and np.isfinite(finite).all()
        and np.all(predicted > 0.0)
        and np.all(truth > 0.0)
    )
    integrity_pass = numerical_pass and memory_pass
    classification, next_step, selected = classify(
        integrity_pass, value_stable, gradient_coherent, safe
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_no_refit_train_only_population_pair_objective_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_sha256": frozen["v63_checkpoint_sha256"],
        "immutable_train_queries": queries,
        "rank_field_stream_sha256": rank_hasher.hexdigest(),
        "individual_rank8_predicted_pair": predicted.tolist(),
        "individual_truth_pair": truth.tolist(),
        "individual_pair_moment_jacobian_sha256": hashlib.sha256(
            jacobian.tobytes()
        ).hexdigest(),
        "individual_bounded_NLL_gradient_sha256": hashlib.sha256(
            nll_gradient.tobytes()
        ).hexdigest(),
        "full_population": full,
        "folds": folds,
        "fold_gradient_median_leave_one_out_cosine": float(np.median(cosines)),
        "fold_gradient_negative_fraction": float(np.mean(np.asarray(cosines) < 0.0)),
        "maximum_rank8_to_sealed_rank16_full_population_log_ratio_difference": rank_reproduction_error,
        "maximum_fold_to_full_population_log_ratio_difference": fold_value_error,
        "population_value_stable": value_stable,
        "population_gradient_coherent": gradient_coherent,
        "optimization_safe": safe,
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
        "posthoc_fold_rank_coefficient_threshold_or_metric_tuning_used": False,
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
        raise FileExistsError("V68 refuses existing audit output")
    result = audit(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
