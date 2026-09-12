#!/usr/bin/env python
"""No-refit multi-object structure-factorization audit for V65."""
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

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v31_copula import conditional_forward, load_model as load_copula
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v50_network import bounded_mixture_log_probability
from hong2021_v62_conditional_moment_gradient_audit import conditional_log_moment_score
from hong2021_v63_preflight import _path, load_program as load_v63_program
from hong2021_v63_train import _is_ancestor
from hong2021_v63_train_gate import _load_fit
from hong2021_v64_sampler_alignment_audit import (
    _pair_indices,
    empirical_pair_moments,
    load_program as load_v64_program,
)


PROGRAM_SHA256 = "58c244e03a5f7fbb9cef29943869067fe3c202d01f3f3773d3cb69d4022bcc21"
PROGRAM_SCHEMA = "hong2021-v65-train-only-multi-object-structure-factorization-audit-program-v1"
SCHEMA = "hong2021-v65-train-only-multi-object-structure-factorization-audit-v1"
PROGRAM_FREEZE_COMMIT = "eda96043e1ea5eb3da8efef2dcc8af21577af9c3"
PAIR_COEFFICIENT = 0.1
CONTROL_ORDER = (
    "source_balanced",
    "same_domain",
    "spatially_permuted_rank",
    "rolled_parameter",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V65 {label} hash differs")
    return _json(path)


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V65 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v64_result_record"]),
        parent["v64_result_record_sha256"],
        "V64 result record",
    )
    audit_row = record.get("audit", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or audit_row.get("classification") != parent["required_classification"]
        or record.get("tail_sampler_alignment", {}).get("material_sampler_mismatch")
        is not parent["required_material_sampler_mismatch"]
        or record.get("gradient_evidence", {}).get("pair_gradient_scale_pass")
        is not parent["required_pair_gradient_scale_pass"]
        or firewall.get("training_or_refit_performed")
        is not parent["required_training_or_refit_performed"]
        or firewall.get("new_development_accessed")
        is not parent["required_new_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V65 parent result or firewall differs")
    frozen = program["frozen_inputs"]
    for key, value in frozen.items():
        if key.endswith("_sha256"):
            continue
        digest = frozen.get(f"{key}_sha256")
        if digest is not None and sha256_file(_path(repo, value)) != digest:
            raise ValueError(f"V65 frozen input differs: {key}")
    audit = _json(_path(repo, frozen["v64_audit"]))
    if (
        audit.get("decision_digest_sha256")
        != frozen["v64_audit_decision_digest_sha256"]
        or canonical_digest(audit) != frozen["v64_audit_decision_digest_sha256"]
        or audit.get("training_or_refit_performed") is not False
        or audit.get("new_development_accessed") is not False
        or audit.get("independent_gate_locked") is not True
    ):
        raise ValueError("V65 sealed V64 audit differs")
    _, v35, gate = load_v64_program(_path(repo, frozen["v64_program"]), repo)
    return program, v35, gate


def donor_mapping(
    v35: dict[str, Any],
    queries: dict[str, list[int]],
    members: int,
    seed: int,
    *,
    same_domain: bool,
) -> dict[str, list[dict[str, Any]]]:
    generator = np.random.default_rng(seed)
    result: dict[str, list[dict[str, Any]]] = {}
    for domain_index, query_domain in enumerate(DOMAIN_ORDER):
        query_rows = []
        for position, query_object in enumerate(queries[query_domain]):
            rows = []
            for member in range(members):
                donor_domain = (
                    query_domain
                    if same_domain
                    else DOMAIN_ORDER[(domain_index + member) % len(DOMAIN_ORDER)]
                )
                objects = int(
                    v35["development_domains"][donor_domain]["train_objects"]
                )
                if donor_domain == query_domain:
                    if objects <= 1 or not 0 <= query_object < objects:
                        raise ValueError("V65 query cannot be excluded from donor pool")
                    donor_index = int(generator.integers(objects - 1))
                    donor_index += int(donor_index >= query_object)
                else:
                    donor_index = int(generator.integers(objects))
                rows.append(
                    {
                        "member": member,
                        "donor_domain": donor_domain,
                        "donor_index": donor_index,
                        "signed_cube_isometry_index": int(
                            generator.integers(len(CUBE_ISOMETRIES))
                        ),
                    }
                )
            query_rows.append(
                {
                    "query_position": position,
                    "query_object_index": query_object,
                    "members": rows,
                }
            )
        result[query_domain] = query_rows
    return result


def _open_train_handles(
    v35: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    return {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }


def _close_handles(handles: dict[str, tuple[Any, Any]]) -> None:
    for data, cache in handles.values():
        data.close()
        cache.close()


def _query_batch(
    handles: dict[str, tuple[Any, Any]],
    prepared: Any,
    queries: dict[str, list[int]],
    position: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conditions = []
    targets = []
    backbones = []
    for domain in DOMAIN_ORDER:
        data, cache = handles[domain]
        condition, target, backbone = condition_cube(
            data,
            cache,
            prepared,
            domain,
            "train",
            int(queries[domain][position]),
        )
        conditions.append(condition)
        targets.append(target)
        backbones.append(backbone)
    return (
        torch.from_numpy(np.stack(conditions)).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
        torch.from_numpy(np.stack(backbones)).to(device),
    )


def _rank_batch(
    handles: dict[str, tuple[Any, Any]],
    mapping: dict[str, list[dict[str, Any]]],
    position: int,
    copula: dict[str, Any],
    hasher: Any,
) -> np.ndarray:
    domains = []
    for query_domain in DOMAIN_ORDER:
        members = []
        row = mapping[query_domain][position]
        for donor in row["members"]:
            data, cache = handles[donor["donor_domain"]]
            donor_index = int(donor["donor_index"])
            backbone = _backbone(cache, donor_index)[None]
            truth = np.asarray(data["target"][donor_index], dtype=np.float32)
            rank = conditional_forward(truth - backbone, backbone, copula)
            axes, reflections = CUBE_ISOMETRIES[
                int(donor["signed_cube_isometry_index"])
            ]
            rank = np.ascontiguousarray(
                apply_cube_isometry(rank, axes, reflections), dtype=np.float32
            )
            if not np.isfinite(rank).all() or np.any((rank < 0.0) | (rank > 1.0)):
                raise RuntimeError("V65 donor rank field differs")
            hasher.update(rank.tobytes())
            members.append(rank)
        domains.append(np.stack(members))
    return np.stack(domains)


def _permuted_rank_batch(
    ranks: np.ndarray, generator: np.random.Generator
) -> tuple[np.ndarray, bool]:
    result = np.empty_like(ranks)
    exact = True
    for domain_index in range(len(ranks)):
        for member in range(ranks.shape[1]):
            original = ranks[domain_index, member].reshape(-1)
            permutation = generator.permutation(original.size)
            permuted = original[permutation]
            restored = np.empty_like(original)
            restored[permutation] = permuted
            exact = exact and bool(np.array_equal(restored, original))
            result[domain_index, member] = permuted.reshape(
                ranks[domain_index, member].shape
            )
    return result, exact


def _pair_rows(
    predicted: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[list[dict[str, float]], float]:
    rows = []
    errors = []
    for separation, (predicted_value, truth_value) in enumerate(
        zip(predicted, truth, strict=True), start=1
    ):
        ratio = float((predicted_value / truth_value).detach().cpu())
        if not ratio > 0.0 or not math.isfinite(ratio):
            raise RuntimeError("V65 nonpositive or nonfinite pair ratio")
        signed = math.log(ratio)
        errors.append(abs(signed))
        rows.append(
            {
                "separation_cells": separation,
                "separation_mpc_h": 0.3125 * separation,
                "truth_pair_mean": float(truth_value.detach().cpu()),
                "predicted_pair_mean": float(predicted_value.detach().cpu()),
                "predicted_over_truth": ratio,
                "signed_log_ratio": signed,
            }
        )
    return rows, float(np.mean(errors))


def summarize_objects(
    objects: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        controls: dict[str, Any] = {}
        for control in CONTROL_ORDER:
            errors = np.asarray(
                [row["controls"][control]["mean_absolute_log_error"] for row in objects[domain]],
                dtype=np.float64,
            )
            separation_rows = {}
            for separation in range(3):
                signed = np.asarray(
                    [
                        row["controls"][control]["separations"][separation][
                            "signed_log_ratio"
                        ]
                        for row in objects[domain]
                    ],
                    dtype=np.float64,
                )
                separation_rows[str(separation + 1)] = {
                    "separation_mpc_h": 0.3125 * (separation + 1),
                    "median_signed_log_ratio": float(np.median(signed)),
                    "q10_signed_log_ratio": float(np.quantile(signed, 0.1)),
                    "q90_signed_log_ratio": float(np.quantile(signed, 0.9)),
                }
            controls[control] = {
                "median_mean_absolute_log_error": float(np.median(errors)),
                "q90_mean_absolute_log_error": float(np.quantile(errors, 0.9)),
                "minimum_mean_absolute_log_error": float(errors.min()),
                "maximum_mean_absolute_log_error": float(errors.max()),
                "separations": separation_rows,
            }
        summaries[domain] = controls
    return summaries


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def gradient_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair = np.asarray([row["pair_gradient"] for row in rows], dtype=np.float64)
    nll = np.asarray([row["bounded_NLL_gradient"] for row in rows], dtype=np.float64)
    cosines = []
    for index, row in enumerate(rows):
        leave_one_out = np.delete(pair, index, axis=0).mean(axis=0)
        cosine = _cosine(pair[index], leave_one_out)
        row["pair_gradient_leave_one_out_global_mean_cosine"] = cosine
        cosines.append(cosine)
    domain_rows = {}
    for domain in DOMAIN_ORDER:
        selected = [
            row["pair_gradient_leave_one_out_global_mean_cosine"]
            for row in rows
            if row["domain"] == domain
        ]
        domain_rows[domain] = {
            "median_leave_one_out_cosine": float(np.median(selected)),
            "negative_cosine_fraction": float(np.mean(np.asarray(selected) < 0.0)),
        }
    pair_mean = pair.mean(axis=0)
    nll_mean = nll.mean(axis=0)
    ratio = PAIR_COEFFICIENT * float(np.linalg.norm(pair_mean)) / max(
        float(np.linalg.norm(nll_mean)), 1.0e-300
    )
    return {
        "query_rows": rows,
        "domains": domain_rows,
        "global_median_leave_one_out_cosine": float(np.median(cosines)),
        "global_negative_cosine_fraction": float(np.mean(np.asarray(cosines) < 0.0)),
        "aggregate_pair_gradient": pair_mean.tolist(),
        "aggregate_bounded_NLL_gradient": nll_mean.tolist(),
        "aggregate_pair_gradient_L2": float(np.linalg.norm(pair_mean)),
        "aggregate_bounded_NLL_gradient_L2": float(np.linalg.norm(nll_mean)),
        "coefficient_0_1_scaled_pair_to_bounded_NLL_L2_ratio": ratio,
    }


def causal_flags(
    summaries: dict[str, dict[str, Any]], gradients: dict[str, Any]
) -> tuple[bool, bool, bool]:
    rank_causal = all(
        summaries[domain]["same_domain"]["median_mean_absolute_log_error"]
        < summaries[domain]["source_balanced"]["median_mean_absolute_log_error"]
        and summaries[domain]["same_domain"]["q90_mean_absolute_log_error"]
        < summaries[domain]["source_balanced"]["q90_mean_absolute_log_error"]
        and summaries[domain]["spatially_permuted_rank"][
            "median_mean_absolute_log_error"
        ]
        > summaries[domain]["source_balanced"]["median_mean_absolute_log_error"]
        for domain in DOMAIN_ORDER
    )
    parameter_causal = all(
        summaries[domain]["rolled_parameter"]["median_mean_absolute_log_error"]
        > summaries[domain]["source_balanced"]["median_mean_absolute_log_error"]
        for domain in DOMAIN_ORDER
    )
    ratio = gradients["coefficient_0_1_scaled_pair_to_bounded_NLL_L2_ratio"]
    gradient_coherent = bool(
        gradients["global_median_leave_one_out_cosine"] >= 0.5
        and gradients["global_negative_cosine_fraction"] <= 0.25
        and 0.01 <= ratio <= 100.0
    )
    return rank_causal, parameter_causal, gradient_coherent


def classify(
    integrity_pass: bool,
    rank_causal: bool,
    parameter_causal: bool,
    gradient_coherent: bool,
) -> tuple[str, str, bool]:
    if not integrity_pass:
        return (
            "structure_factorization_audit_failed_integrity",
            "stop_before_refit_and_preserve_the_failed_train_only_audit",
            False,
        )
    if rank_causal and parameter_causal and gradient_coherent:
        return (
            "rank_dependence_and_query_parameter_arrangement_jointly_cause_pair_error_with_a_coherent_gradient",
            "freeze_one_joint_domain_conditioned_copula_and_direct_pair_objective_model_before_refit",
            True,
        )
    if rank_causal and not gradient_coherent:
        return (
            "donor_rank_dependence_is_causal_without_a_coherent_direct_pair_gradient",
            "freeze_one_domain_conditioned_copula_model_before_refit",
            True,
        )
    if not rank_causal and parameter_causal and gradient_coherent:
        return (
            "query_parameter_spatial_arrangement_is_causal_with_a_coherent_direct_pair_gradient",
            "freeze_one_direct_pair_objective_model_before_refit",
            True,
        )
    return (
        "train_only_structure_factorization_does_not_select_a_single_model_change",
        "stop_before_refit_and_design_a_new_train_only_structure_hypothesis",
        False,
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, gate = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V65 audit requires clean Lageunha with frozen ancestry")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V65 audit requires the Lageunha Ada GPU")
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
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.output.bias.requires_grad_(True)
    initial_bias = model.output.bias.detach().clone()
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    query_spec = program["immutable_train_queries"]
    queries = {domain: list(map(int, query_spec[domain])) for domain in DOMAIN_ORDER}
    members = int(program["frozen_rank_controls"]["members_per_query_and_control"])
    source_mapping = donor_mapping(
        v35,
        queries,
        members,
        int(program["frozen_rank_controls"]["source_balanced"]["seed"]),
        same_domain=False,
    )
    same_mapping = donor_mapping(
        v35,
        queries,
        members,
        int(program["frozen_rank_controls"]["same_domain"]["seed"]),
        same_domain=True,
    )
    copula = load_copula(
        _path(repo, frozen["conditional_copula_artifact"]),
        frozen["conditional_copula_artifact_sha256"],
    )
    pairs = _pair_indices(
        int(program["pair_probe"]["anchor_seed"]),
        int(program["pair_probe"]["anchors_per_query"]),
    )
    permutation_generator = np.random.default_rng(
        int(
            program["frozen_rank_controls"]["spatially_permuted_rank_control"][
                "seed"
            ]
        )
    )
    source_hasher = hashlib.sha256()
    same_hasher = hashlib.sha256()
    objects: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}
    gradient_rows: list[dict[str, Any]] = []
    maximum_inverse_error = 0.0
    exact_multiset = True
    handles = _open_train_handles(v35)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for position in range(int(query_spec["objects_per_domain"])):
            condition, target, backbone = _query_batch(
                handles, prepared, queries, position, device
            )
            source_numpy = _rank_batch(
                handles, source_mapping, position, copula, source_hasher
            )
            same_numpy = _rank_batch(handles, same_mapping, position, copula, same_hasher)
            permuted_numpy, exact = _permuted_rank_batch(
                source_numpy, permutation_generator
            )
            exact_multiset = exact_multiset and exact
            source_ranks = torch.from_numpy(source_numpy).to(device)
            same_ranks = torch.from_numpy(same_numpy).to(device)
            permuted_ranks = torch.from_numpy(permuted_numpy).to(device)
            with torch.no_grad():
                output = model(condition).float()
                rolled = torch.roll(
                    output,
                    shifts=tuple(
                        int(value)
                        for value in program["frozen_rank_controls"][
                            "rolled_parameter_control"
                        ]["parameter_roll_cells"]
                    ),
                    dims=(-3, -2, -1),
                )
                control_tensors = {
                    "source_balanced": (output, source_ranks),
                    "same_domain": (output, same_ranks),
                    "spatially_permuted_rank": (output, permuted_ranks),
                    "rolled_parameter": (rolled, source_ranks),
                }
                measurements = {}
                reference_truth = None
                for control in CONTROL_ORDER:
                    parameters, ranks = control_tensors[control]
                    predicted, truth, inverse_error = empirical_pair_moments(
                        parameters,
                        target,
                        backbone,
                        ranks,
                        target_mean,
                        target_std,
                        pairs,
                        members,
                    )
                    maximum_inverse_error = max(maximum_inverse_error, inverse_error)
                    if reference_truth is None:
                        reference_truth = truth
                    elif not torch.equal(reference_truth, truth):
                        raise RuntimeError("V65 control truth binding differs")
                    measurements[control] = (predicted.reshape(3, 3), truth.reshape(3, 3))
            for domain_index, domain in enumerate(DOMAIN_ORDER):
                controls = {}
                for control in CONTROL_ORDER:
                    predicted, truth = measurements[control]
                    separation_rows, error = _pair_rows(
                        predicted[domain_index], truth[domain_index]
                    )
                    controls[control] = {
                        "mean_absolute_log_error": error,
                        "separations": separation_rows,
                    }
                objects[domain].append(
                    {
                        "query_position": position,
                        "query_object_index": queries[domain][position],
                        "controls": controls,
                    }
                )
            if position < 4:
                gradient_output = model(condition).float()
                gradient_predicted, gradient_truth, inverse_error = empirical_pair_moments(
                    gradient_output,
                    target,
                    backbone,
                    source_ranks,
                    target_mean,
                    target_std,
                    pairs,
                    int(program["gradient_coherence_probe"]["rank_members"]),
                )
                maximum_inverse_error = max(maximum_inverse_error, inverse_error)
                for domain_index, domain in enumerate(DOMAIN_ORDER):
                    start = 3 * domain_index
                    pair_objective = conditional_log_moment_score(
                        gradient_predicted[start : start + 3],
                        gradient_truth[start : start + 3],
                    )
                    nll_objective = -bounded_mixture_log_probability(
                        gradient_output[domain_index : domain_index + 1],
                        target[domain_index : domain_index + 1],
                    ).mean()
                    pair_gradient = torch.autograd.grad(
                        pair_objective, model.output.bias, retain_graph=True
                    )[0]
                    nll_gradient = torch.autograd.grad(
                        nll_objective,
                        model.output.bias,
                        retain_graph=domain_index < len(DOMAIN_ORDER) - 1,
                    )[0]
                    if not bool(
                        (
                            torch.isfinite(pair_gradient).all()
                            & torch.isfinite(nll_gradient).all()
                        )
                        .detach()
                        .cpu()
                    ):
                        raise RuntimeError("V65 nonfinite bias gradient")
                    gradient_rows.append(
                        {
                            "domain": domain,
                            "query_position": position,
                            "query_object_index": queries[domain][position],
                            "pair_objective": float(pair_objective.detach().cpu()),
                            "bounded_NLL_objective": float(nll_objective.detach().cpu()),
                            "pair_gradient": pair_gradient.detach().double().cpu().tolist(),
                            "bounded_NLL_gradient": nll_gradient.detach().double().cpu().tolist(),
                            "pair_gradient_L2": float(
                                torch.linalg.vector_norm(pair_gradient.double())
                                .detach()
                                .cpu()
                            ),
                            "bounded_NLL_gradient_L2": float(
                                torch.linalg.vector_norm(nll_gradient.double())
                                .detach()
                                .cpu()
                            ),
                        }
                    )
                del gradient_output, gradient_predicted, gradient_truth
            del condition, target, backbone, output, rolled
            del source_ranks, same_ranks, permuted_ranks
    finally:
        _close_handles(handles)
        prepared.close()
    if not torch.equal(initial_bias, model.output.bias.detach()):
        raise RuntimeError("V65 model parameters changed without authorization")
    summaries = summarize_objects(objects)
    gradients = gradient_summary(gradient_rows)
    rank_causal, parameter_causal, gradient_coherent = causal_flags(
        summaries, gradients
    )
    peak = int(torch.cuda.max_memory_allocated(device))
    memory_pass = peak < int(program["resource_gate"]["peak_allocated_bytes_limit"])
    finite_values = [maximum_inverse_error, float(peak)]
    for domain in DOMAIN_ORDER:
        for control in CONTROL_ORDER:
            finite_values.extend(
                (
                    summaries[domain][control]["median_mean_absolute_log_error"],
                    summaries[domain][control]["q90_mean_absolute_log_error"],
                )
            )
    finite_values.extend(
        (
            gradients["global_median_leave_one_out_cosine"],
            gradients["global_negative_cosine_fraction"],
            gradients["coefficient_0_1_scaled_pair_to_bounded_NLL_L2_ratio"],
        )
    )
    numerical_pass = bool(
        maximum_inverse_error <= 2.0e-6
        and exact_multiset
        and len(gradient_rows) == 12
        and all(row["pair_gradient_L2"] > 0.0 for row in gradient_rows)
        and all(row["bounded_NLL_gradient_L2"] > 0.0 for row in gradient_rows)
        and all(math.isfinite(value) for value in finite_values)
    )
    integrity_pass = numerical_pass and memory_pass
    classification, next_step, selected = classify(
        integrity_pass, rank_causal, parameter_causal, gradient_coherent
    )
    permutation_state = json.dumps(
        permutation_generator.bit_generator.state, sort_keys=True, separators=(",", ":")
    ).encode()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_no_refit_train_only_structure_factorization_audit",
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
        "donor_mappings": {
            "source_balanced": source_mapping,
            "same_domain": same_mapping,
        },
        "rank_field_stream_sha256": {
            "source_balanced": source_hasher.hexdigest(),
            "same_domain": same_hasher.hexdigest(),
        },
        "spatial_permutation_final_state_sha256": hashlib.sha256(
            permutation_state
        ).hexdigest(),
        "spatial_permutation_exact_rank_multiset_preserved": exact_multiset,
        "objects": objects,
        "domain_summaries": summaries,
        "gradient_coherence": gradients,
        "rank_dependence_causal": rank_causal,
        "query_parameter_spatial_arrangement_causal": parameter_causal,
        "direct_pair_gradient_coherent": gradient_coherent,
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
        raise FileExistsError("V65 refuses existing audit output")
    result = audit(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
