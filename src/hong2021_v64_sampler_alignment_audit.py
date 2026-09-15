#!/usr/bin/env python
"""No-refit train-only empirical-rank sampler-alignment audit for V64."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v31_copula import conditional_forward, load_model as load_copula
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v48_network import (
    RANK_EPSILON,
    gaussian_mixture_inverse,
    mixture_parameters,
    standard_normal_cdf,
)
from hong2021_v48_train import load_cache
from hong2021_v50_network import (
    LOWER_SUPPORT,
    SUPPORT_RANGE,
    UPPER_SUPPORT,
    bounded_mixture_log_probability,
)
from hong2021_v62_conditional_moment_gradient_audit import (
    _quadrature_rule,
    _real_batch,
    conditional_log_moment_score,
    conditional_physical_moments,
)
from hong2021_v63_preflight import _path, load_program as load_v63_program
from hong2021_v63_sample import _base_program
from hong2021_v63_train import _is_ancestor
from hong2021_v63_train_gate import _load_fit


PROGRAM_SHA256 = "f64d45a75258682ae3200605a740876c1326b80ec811a91ee54fd97bc4e946ec"
PROGRAM_SCHEMA = "hong2021-v64-train-only-empirical-rank-sampler-alignment-audit-program-v1"
SCHEMA = "hong2021-v64-train-only-empirical-rank-sampler-alignment-audit-v1"
PROGRAM_FREEZE_COMMIT = "e8452473c10681eae2279c49d8327d01d8995ba8"
FUTURE_COEFFICIENT = 0.1


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V64 {label} hash differs")
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
        raise ValueError("V64 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v63_result_record"]),
        parent["v63_result_record_sha256"],
        "V63 result record",
    )
    decision = record.get("development_decision", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or decision.get("classification") != parent["required_classification"]
        or decision.get(
            "V63_strictly_improves_all_three_over_both_V50_and_V52_every_domain"
        )
        is not parent["required_all_extremes_improve_both_references_every_domain"]
        or firewall.get("posthoc_scale_or_clipping_used")
        is not parent["required_posthoc_tuning_used"]
        or firewall.get("historical_EAGLE_accessed")
        is not parent["required_historical_EAGLE_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V64 parent result or firewall differs")
    frozen = program["frozen_inputs"]
    for key, value in frozen.items():
        if key.endswith("_sha256"):
            continue
        digest = frozen.get(f"{key}_sha256")
        if digest is not None and sha256_file(_path(repo, value)) != digest:
            raise ValueError(f"V64 frozen input differs: {key}")
    train_record = _json(_path(repo, frozen["v63_train_result_record"]))
    report = _json(_path(repo, frozen["v63_training_report"]))
    gate = _json(_path(repo, frozen["v63_train_gate"]))
    if (
        train_record.get("status")
        != "complete_train_gate_pass_authorized_locked_development"
        or train_record.get("authorization", {}).get("new_training_or_refit_allowed")
        is not False
        or report.get("development_accessed") is not False
        or canonical_digest(report) != report.get("decision_digest_sha256")
        or gate.get("train_mechanism_pass") is not True
        or gate.get("development_accessed") is not False
        or canonical_digest(gate) != gate.get("decision_digest_sha256")
        or gate.get("independent_gate_locked") is not True
    ):
        raise ValueError("V64 sealed V63 train evidence differs")
    v63, v35, _, _, _, _, _ = load_v63_program(
        _path(repo, frozen["v63_program"]), repo
    )
    effective, _, _ = _base_program(_path(repo, frozen["v63_program"]), repo)
    if (
        v63.get("schema")
        != "hong2021-v63-conditional-log-physical-moment-model-program-v1"
        or effective["inherited_inputs"].get("conditional_copula_artifact_sha256")
        != frozen["conditional_copula_artifact_sha256"]
        or _path(
            repo, effective["inherited_inputs"]["conditional_copula_artifact"]
        )
        != _path(repo, frozen["conditional_copula_artifact"])
    ):
        raise ValueError("V64 inherited copula binding differs")
    return program, v35, gate


def _differentiable_bounded_inverse(
    parameters: torch.Tensor, uniform: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen bisection forward value with an implicit quantile derivative."""
    if uniform.shape != (len(parameters), 1, *parameters.shape[-3:]):
        raise ValueError("V64 empirical rank shape differs")
    probability = uniform.float().clamp(RANK_EPSILON, 1.0 - RANK_EPSILON)
    latent0 = gaussian_mixture_inverse(parameters, probability)
    logits, locations, scales = mixture_parameters(parameters)
    weights = F.softmax(logits, dim=1)
    standardized = (latent0.float() - locations) / scales
    cdf = torch.sum(weights * standard_normal_cdf(standardized), dim=1, keepdim=True)
    density = torch.sum(
        weights
        * torch.exp(-0.5 * torch.square(standardized))
        / (scales * math.sqrt(2.0 * math.pi)),
        dim=1,
        keepdim=True,
    )
    if not bool((density > 0.0).all().detach().cpu()):
        raise RuntimeError("V64 nonpositive mixture density at empirical quantile")
    latent = latent0 - (cdf - probability) / density.detach().clamp_min(1.0e-12)
    bounded = LOWER_SUPPORT + SUPPORT_RANGE * torch.sigmoid(latent.double())
    if not bool(
        ((bounded > LOWER_SUPPORT) & (bounded < UPPER_SUPPORT)).all().detach().cpu()
    ):
        raise RuntimeError("V64 empirical inverse reached support boundary")
    return bounded, torch.abs(cdf - probability)


def _donor_mapping(
    v35: dict[str, Any], members: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    generator = np.random.default_rng(seed)
    result: dict[str, list[dict[str, Any]]] = {}
    for query_index, query_domain in enumerate(DOMAIN_ORDER):
        rows = []
        for member in range(members):
            donor_domain = DOMAIN_ORDER[(query_index + member) % len(DOMAIN_ORDER)]
            objects = int(v35["development_domains"][donor_domain]["train_objects"])
            donor_index = int(generator.integers(objects))
            if donor_domain == query_domain and donor_index == 0:
                donor_index = (donor_index + 1) % objects
            isometry = int(generator.integers(len(CUBE_ISOMETRIES)))
            rows.append(
                {
                    "member": member,
                    "donor_domain": donor_domain,
                    "donor_index": donor_index,
                    "signed_cube_isometry_index": isometry,
                }
            )
        result[query_domain] = rows
    return result


def _rank_ensemble(
    v35: dict[str, Any],
    mapping: dict[str, list[dict[str, Any]]],
    copula: dict[str, Any],
) -> tuple[np.ndarray, dict[str, list[str]]]:
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    ranks: list[np.ndarray] = []
    digests: dict[str, list[str]] = {}
    try:
        for query_domain in DOMAIN_ORDER:
            members = []
            hashes = []
            for row in mapping[query_domain]:
                donor_data, donor_cache = handles[row["donor_domain"]]
                donor_backbone = _backbone(donor_cache, row["donor_index"])[None]
                donor_truth = np.asarray(
                    donor_data["target"][row["donor_index"]], dtype=np.float32
                )
                rank = conditional_forward(
                    donor_truth - donor_backbone, donor_backbone, copula
                )
                axes, reflections = CUBE_ISOMETRIES[
                    row["signed_cube_isometry_index"]
                ]
                rank = apply_cube_isometry(rank, axes, reflections)
                if not np.isfinite(rank).all() or np.any((rank < 0.0) | (rank > 1.0)):
                    raise RuntimeError("V64 donor rank field differs")
                members.append(rank)
                hashes.append(hashlib.sha256(rank.tobytes()).hexdigest())
            ranks.append(np.stack(members))
            digests[query_domain] = hashes
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
    return np.stack(ranks), digests


def _selected_parameters(
    parameters: torch.Tensor,
    domain_index: int,
    indices: torch.Tensor,
    members: int,
) -> torch.Tensor:
    selected = parameters[domain_index : domain_index + 1].reshape(1, 15, -1)[
        :, :, indices
    ]
    return selected.expand(members, -1, -1).reshape(members, 15, 1, 1, -1)


def _selected_ranks(
    ranks: torch.Tensor,
    domain_index: int,
    indices: torch.Tensor,
    members: int,
) -> torch.Tensor:
    return ranks[domain_index, :members].reshape(members, -1)[:, indices].reshape(
        members, 1, 1, 1, -1
    )


def empirical_tail_moments(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    ranks: torch.Tensor,
    target_mean: float,
    target_std: float,
    boundaries: torch.Tensor,
    members: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], float]:
    coefficient = 4.5 * math.log(10.0)
    predicted_rows = []
    truth_rows = []
    counts = []
    maximum_cdf_error = 0.0
    for index in range(len(DOMAIN_ORDER)):
        base = backbone[index, 0].double().reshape(-1) + target_mean
        mask = base >= boundaries[index].double()
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(indices.numel())
        if count <= 0:
            raise RuntimeError("V64 empty tail mask")
        selected_parameters = _selected_parameters(parameters, index, indices, members)
        selected_ranks = _selected_ranks(ranks, index, indices, members)
        standardized, error = _differentiable_bounded_inverse(
            selected_parameters, selected_ranks
        )
        maximum_cdf_error = max(maximum_cdf_error, float(error.max().detach().cpu()))
        physical_y = base[indices][None, None, None, None, :] + target_std * standardized
        predicted = torch.square(torch.exp(coefficient * physical_y) - 1.0).mean()
        exact_y = base[indices] + target_std * target[index, 0].double().reshape(-1)[
            indices
        ]
        truth = torch.square(torch.exp(coefficient * exact_y) - 1.0).mean()
        predicted_rows.append(predicted)
        truth_rows.append(truth)
        counts.append(count)
    return (
        torch.stack(predicted_rows),
        torch.stack(truth_rows),
        counts,
        maximum_cdf_error,
    )


def _pair_indices(seed: int, anchors: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    coordinate = np.arange(3, 61, dtype=np.int64)
    grid = np.stack(np.meshgrid(coordinate, coordinate, coordinate, indexing="ij"), axis=-1).reshape(-1, 3)
    generator = np.random.default_rng(seed)
    selected = grid[generator.choice(len(grid), size=anchors, replace=False)]
    directions = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.int64,
    )
    direction = directions[np.arange(anchors) % len(directions)]
    first = np.ravel_multi_index(selected.T, (64, 64, 64))
    return {
        separation: (
            first,
            np.ravel_multi_index((selected + separation * direction).T, (64, 64, 64)),
        )
        for separation in (1, 2, 3)
    }


def empirical_pair_moments(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    ranks: torch.Tensor,
    target_mean: float,
    target_std: float,
    pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    members: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    coefficient = 4.5 * math.log(10.0)
    predicted_rows = []
    truth_rows = []
    maximum_cdf_error = 0.0
    for domain_index in range(len(DOMAIN_ORDER)):
        base = backbone[domain_index, 0].double().reshape(-1) + target_mean
        exact = base + target_std * target[domain_index, 0].double().reshape(-1)
        for separation in (1, 2, 3):
            first_np, second_np = pairs[separation]
            first = torch.as_tensor(first_np, device=parameters.device)
            second = torch.as_tensor(second_np, device=parameters.device)
            values = []
            for indices in (first, second):
                selected_parameters = _selected_parameters(
                    parameters, domain_index, indices, members
                )
                selected_ranks = _selected_ranks(
                    ranks, domain_index, indices, members
                )
                standardized, error = _differentiable_bounded_inverse(
                    selected_parameters, selected_ranks
                )
                maximum_cdf_error = max(
                    maximum_cdf_error, float(error.max().detach().cpu())
                )
                physical_y = (
                    base[indices][None, None, None, None, :]
                    + target_std * standardized
                )
                values.append(torch.exp(coefficient * physical_y) - 1.0)
            predicted = (values[0] * values[1]).mean()
            truth_delta_first = torch.exp(coefficient * exact[first]) - 1.0
            truth_delta_second = torch.exp(coefficient * exact[second]) - 1.0
            truth_pair = (truth_delta_first * truth_delta_second).mean()
            if not bool(
                ((predicted > 0.0) & (truth_pair > 0.0)).detach().cpu()
            ):
                raise RuntimeError("V64 nonpositive sub-Mpc pair moment")
            predicted_rows.append(predicted)
            truth_rows.append(truth_pair)
    return torch.stack(predicted_rows), torch.stack(truth_rows), maximum_cdf_error


def _objective_gradient(
    output: torch.Tensor, closure: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[float, torch.Tensor, dict[str, float]]:
    value = output.detach().requires_grad_(True)
    objective = closure(value)
    gradient = torch.autograd.grad(objective, value)[0]
    if not bool(
        (torch.isfinite(objective) & torch.isfinite(gradient).all()).detach().cpu()
    ):
        raise RuntimeError("V64 nonfinite objective or output gradient")
    metrics = {
        "L2": float(torch.linalg.vector_norm(gradient.double()).detach().cpu()),
        "maximum_absolute": float(gradient.abs().max().detach().cpu()),
    }
    return float(objective.detach().cpu()), gradient, metrics


def classify(
    numerical_pass: bool,
    material_mismatch: bool,
    tail_gradient_pass: bool,
    finite_rank_stability: bool,
    pair_conflict: bool,
    memory_pass: bool,
) -> tuple[str, str, bool]:
    if not numerical_pass or not memory_pass:
        return (
            "empirical_rank_sampler_alignment_audit_is_not_numerically_safe",
            "stop_before_refit_and_preserve_the_failed_train_only_audit",
            False,
        )
    if pair_conflict or not material_mismatch:
        return (
            "sampler_aligned_tail_repair_is_not_a_sufficient_single_model_change",
            "freeze_a_train_only_copula_or_joint_structure_audit_before_any_refit",
            False,
        )
    if tail_gradient_pass and finite_rank_stability:
        return (
            "finite_empirical_rank_transport_creates_a_causal_train_only_objective_mismatch",
            "freeze_one_model_replacing_only_the_V63_quadrature_moment_term_with_the_coefficient_0.1_sampler_aligned_empirical_rank_term",
            True,
        )
    return (
        "sampler_aligned_tail_objective_is_not_optimization_feasible",
        "stop_before_refit_and_reassess_the_finite_rank_estimating_score",
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
        raise RuntimeError("V64 audit requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V64 audit requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    frozen = program["frozen_inputs"]
    v63_program, _, _, _, _, _, _ = load_v63_program(
        _path(repo, frozen["v63_program"]), repo
    )
    boundaries_dict = {
        domain: float(v63_program["sealed_q99_9_backbone_boundaries"][domain])
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
        v63_program["frozen_inputs"]["support_selection_sha256"],
        boundaries_dict,
        repo,
        commit,
    )
    model = model.to(device).eval()
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    fixed = program["fixed_train_batch"]
    condition, target, backbone = _real_batch(
        v35,
        prepared,
        device,
        int(fixed["query_object_index"]),
        int(fixed["query_signed_cube_isometry_index"]),
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    prepared.close()
    boundaries = torch.tensor(
        [boundaries_dict[domain] for domain in DOMAIN_ORDER],
        dtype=torch.float64,
        device=device,
    )
    mapping = _donor_mapping(
        v35,
        int(program["frozen_empirical_rank_ensemble"]["members_per_query"]),
        int(program["frozen_empirical_rank_ensemble"]["selection_seed"]),
    )
    copula = load_copula(
        _path(repo, frozen["conditional_copula_artifact"]),
        frozen["conditional_copula_artifact_sha256"],
    )
    rank_numpy, rank_digests = _rank_ensemble(v35, mapping, copula)
    ranks = torch.from_numpy(rank_numpy).to(device)
    pairs = _pair_indices(
        int(program["train_only_sub_mpc_compatibility_diagnostic"]["anchor_seed"]),
        int(program["train_only_sub_mpc_compatibility_diagnostic"]["anchors_per_query"]),
    )
    nodes, weights = _quadrature_rule(64, device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        output = model(condition).float()
        quadrature, truth, counts = conditional_physical_moments(
            output,
            target,
            backbone,
            target_mean,
            target_std,
            boundaries,
            nodes,
            weights,
        )
        empirical8, truth8, counts8, error8 = empirical_tail_moments(
            output,
            target,
            backbone,
            ranks,
            target_mean,
            target_std,
            boundaries,
            8,
        )
        empirical16, truth16, counts16, error16 = empirical_tail_moments(
            output,
            target,
            backbone,
            ranks,
            target_mean,
            target_std,
            boundaries,
            16,
        )
        pair_predicted, pair_truth, pair_error = empirical_pair_moments(
            output,
            target,
            backbone,
            ranks,
            target_mean,
            target_std,
            pairs,
            16,
        )
    expected_counts = [
        int(fixed["expected_selected_voxels"][domain]) for domain in DOMAIN_ORDER
    ]
    if (
        counts != expected_counts
        or counts8 != counts
        or counts16 != counts
        or not torch.equal(truth, truth8)
        or not torch.equal(truth, truth16)
    ):
        raise RuntimeError("V64 fixed mask or truth binding differs")

    def nll_closure(value: torch.Tensor) -> torch.Tensor:
        return -bounded_mixture_log_probability(value, target).mean()

    def quadrature_closure(value: torch.Tensor) -> torch.Tensor:
        predicted, observed, _ = conditional_physical_moments(
            value,
            target,
            backbone,
            target_mean,
            target_std,
            boundaries,
            nodes,
            weights,
        )
        return conditional_log_moment_score(predicted, observed)

    def tail_closure(value: torch.Tensor) -> torch.Tensor:
        predicted, observed, _, _ = empirical_tail_moments(
            value,
            target,
            backbone,
            ranks,
            target_mean,
            target_std,
            boundaries,
            16,
        )
        return conditional_log_moment_score(predicted, observed)

    def pair_closure(value: torch.Tensor) -> torch.Tensor:
        predicted, observed, _ = empirical_pair_moments(
            value,
            target,
            backbone,
            ranks,
            target_mean,
            target_std,
            pairs,
            16,
        )
        return conditional_log_moment_score(predicted, observed)

    nll_value, nll_gradient, nll_metrics = _objective_gradient(output, nll_closure)
    quadrature_value, _, quadrature_metrics = _objective_gradient(
        output, quadrature_closure
    )
    tail_value, tail_gradient, tail_metrics = _objective_gradient(output, tail_closure)
    pair_value, pair_gradient, pair_metrics = _objective_gradient(output, pair_closure)
    dot = torch.sum(tail_gradient.double() * pair_gradient.double())
    cosine = dot / torch.clamp_min(
        torch.linalg.vector_norm(tail_gradient.double())
        * torch.linalg.vector_norm(pair_gradient.double()),
        1.0e-300,
    )
    dot_value = float(dot.detach().cpu())
    cosine_value = float(cosine.detach().cpu())
    del nll_gradient, tail_gradient, pair_gradient
    tail_ratio = FUTURE_COEFFICIENT * tail_metrics["L2"] / max(
        nll_metrics["L2"], 1.0e-300
    )
    pair_ratio = FUTURE_COEFFICIENT * pair_metrics["L2"] / max(
        nll_metrics["L2"], 1.0e-300
    )
    domains: dict[str, Any] = {}
    mismatch_values = []
    stability_values = []
    for index, domain in enumerate(DOMAIN_ORDER):
        mismatch = abs(float(torch.log(empirical16[index] / quadrature[index]).cpu()))
        stability = float(
            (
                torch.abs(empirical8[index] - empirical16[index])
                / torch.maximum(
                    torch.maximum(empirical8[index].abs(), empirical16[index].abs()),
                    torch.tensor(1.0e-300, dtype=torch.float64, device=device),
                )
            ).cpu()
        )
        mismatch_values.append(mismatch)
        stability_values.append(stability)
        domains[domain] = {
            "selected_voxels": counts[index],
            "truth_mean_delta_squared": float(truth[index].cpu()),
            "Gauss_Hermite_64_mean_delta_squared": float(quadrature[index].cpu()),
            "empirical_rank_8_mean_delta_squared": float(empirical8[index].cpu()),
            "empirical_rank_16_mean_delta_squared": float(empirical16[index].cpu()),
            "Gauss_Hermite_over_truth": float((quadrature[index] / truth[index]).cpu()),
            "empirical_rank_8_over_truth": float((empirical8[index] / truth[index]).cpu()),
            "empirical_rank_16_over_truth": float((empirical16[index] / truth[index]).cpu()),
            "absolute_empirical_rank_16_over_Gauss_Hermite_log_ratio": mismatch,
            "empirical_rank_8_to_16_relative_difference": stability,
        }
    pair_rows: dict[str, Any] = {}
    cursor = 0
    for domain in DOMAIN_ORDER:
        rows = {}
        for separation in (1, 2, 3):
            rows[str(separation)] = {
                "separation_mpc_h": 0.3125 * separation,
                "truth_pair_mean": float(pair_truth[cursor].cpu()),
                "empirical_rank_16_pair_mean": float(pair_predicted[cursor].cpu()),
                "predicted_over_truth": float(
                    (pair_predicted[cursor] / pair_truth[cursor]).cpu()
                ),
            }
            cursor += 1
        pair_rows[domain] = rows
    maximum_inverse_error = max(error8, error16, pair_error)
    material_mismatch = max(mismatch_values) >= math.log(1.1)
    finite_rank_stability = max(stability_values) <= 0.5
    tail_gradient_pass = 0.01 <= tail_ratio <= 100.0
    pair_gradient_pass = 0.01 <= pair_ratio <= 100.0
    pair_conflict = bool(pair_value > 0.0 and pair_gradient_pass and cosine_value < 0.0)
    peak = int(torch.cuda.max_memory_allocated(device))
    memory_pass = peak < int(program["resource_gate"]["peak_allocated_bytes_limit"])
    numerical_pass = bool(
        maximum_inverse_error <= 2.0e-6
        and all(math.isfinite(value) for value in mismatch_values + stability_values)
        and nll_metrics["L2"] > 0.0
        and tail_metrics["L2"] > 0.0
        and pair_metrics["L2"] > 0.0
        and math.isfinite(cosine_value)
    )
    classification, next_step, selected = classify(
        numerical_pass,
        material_mismatch,
        tail_gradient_pass,
        finite_rank_stability,
        pair_conflict,
        memory_pass,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_no_refit_train_only_sampler_alignment_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_sha256": frozen["v63_checkpoint_sha256"],
        "training_report_sha256": frozen["v63_training_report_sha256"],
        "train_gate_sha256": frozen["v63_train_gate_sha256"],
        "train_gate_decision_digest_sha256": gate["decision_digest_sha256"],
        "fixed_train_batch": fixed,
        "donor_mapping": mapping,
        "rank_multiset_sha256": rank_digests,
        "domains": domains,
        "sub_mpc_pair_diagnostic": pair_rows,
        "objectives": {
            "bounded_NLL": nll_value,
            "V63_Gauss_Hermite_score": quadrature_value,
            "sampler_aligned_tail_score": tail_value,
            "sub_mpc_pair_score": pair_value,
        },
        "output_gradients": {
            "bounded_NLL": nll_metrics,
            "V63_Gauss_Hermite_score": quadrature_metrics,
            "sampler_aligned_tail_score": tail_metrics,
            "sub_mpc_pair_score": pair_metrics,
            "coefficient_scaled_tail_to_NLL_L2_ratio": tail_ratio,
            "coefficient_scaled_pair_to_NLL_L2_ratio": pair_ratio,
            "tail_to_pair_dot_product": dot_value,
            "tail_to_pair_cosine": cosine_value,
        },
        "maximum_inverse_CDF_error": maximum_inverse_error,
        "maximum_empirical_rank_8_to_16_relative_difference": max(stability_values),
        "maximum_absolute_empirical_rank_16_over_Gauss_Hermite_log_ratio": max(
            mismatch_values
        ),
        "material_sampler_mismatch": material_mismatch,
        "finite_rank_stability": finite_rank_stability,
        "tail_gradient_scale_pass": tail_gradient_pass,
        "pair_gradient_scale_pass": pair_gradient_pass,
        "pair_conflict": pair_conflict,
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
        raise FileExistsError("V64 refuses existing audit output")
    result = audit(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
