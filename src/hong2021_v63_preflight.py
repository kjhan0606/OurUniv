#!/usr/bin/env python
"""Hard preflight for the frozen V63 conditional physical-moment model."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v48_train import load_cache
from hong2021_v50_network import INITIAL_BIASES, bounded_mixture_log_probability, parameter_count
from hong2021_v50_train import PARAMETERS
from hong2021_v54_train import _same_seed_model
from hong2021_v56_train import composite_loss as v56_composite_loss
from hong2021_v56_train import load_program as load_v56_program
from hong2021_v56_train import upper_survival_grid_score
from hong2021_v62_conditional_moment_gradient_audit import (
    _local_gradient,
    _quadrature_rule,
    _real_batch,
    conditional_log_moment_score,
    conditional_physical_moments,
)


PROGRAM_SHA256 = "ea41d61a2961b3f436ed69662dc39ad8ad151980aca32863c0442948d31b6a48"
PROGRAM_SCHEMA = "hong2021-v63-conditional-log-physical-moment-model-program-v1"
SCHEMA = "hong2021-v63-conditional-log-physical-moment-hard-preflight-v1"


def _path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V63 {label} hash differs")
    return _json(path)


def load_program(
    path: Path, repo: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_preflight_model_implementation_training_or_evaluation"
    ):
        raise ValueError("V63 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v62_record"]), parent["v62_record_sha256"], "V62 record"
    )
    if (
        record.get("status") != parent["required_status"]
        or record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("candidate_selected")
        is not parent["required_candidate_selected"]
        or record.get("firewall", {}).get("development_accessed")
        is not parent["required_development_accessed"]
        or record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V63 parent selection or firewall differs")
    frozen = program["frozen_inputs"]
    for key, value in frozen.items():
        if key.endswith("_sha256"):
            continue
        digest = frozen.get(f"{key}_sha256")
        if digest is not None and sha256_file(_path(repo, value)) != digest:
            raise ValueError(f"V63 frozen input differs: {key}")
    v56_preflight = _json(_path(repo, frozen["v56_preflight"]))
    v61_gate = _json(_path(repo, frozen["v61_train_gate"]))
    v62_audit = _json(_path(repo, frozen["v62_audit"]))
    if (
        canonical_digest(v56_preflight)
        != frozen["v56_preflight_decision_digest_sha256"]
        or canonical_digest(v61_gate) != frozen["v61_train_gate_decision_digest_sha256"]
        or canonical_digest(v62_audit) != frozen["v62_audit_decision_digest_sha256"]
        or v56_preflight.get("status") != "pass"
        or v62_audit.get("candidate_selected") is not True
        or v62_audit.get("training_or_refit_performed") is not False
        or v62_audit.get("development_accessed") is not False
        or v62_audit.get("independent_gate_locked") is not True
    ):
        raise ValueError("V63 inherited decision digest or firewall differs")
    _, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    grid = _json(_path(repo, frozen["v56_grid"]))
    thresholds = _json(_path(repo, frozen["v54_threshold_selection"]))
    boundaries = program["sealed_q99_9_backbone_boundaries"]
    for domain in DOMAIN_ORDER:
        if float(boundaries[domain]) != float(
            v61_gate["domains"][domain]["backbone_boundaries"][2]
        ):
            raise ValueError("V63 q99.9 boundary differs")
    return program, v35, grid, thresholds, v56_preflight, v62_audit, v61_gate


def _close(left: float, right: float, tolerance: float = 1.0e-7) -> bool:
    return abs(left - right) <= tolerance * max(abs(left), abs(right), 1.0)


def count_summary(counts: list[int]) -> dict[str, Any]:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or bool(np.any(values < 0)):
        raise ValueError("V63 mask counts differ")
    return {
        "objects": int(values.size),
        "zero_count_objects": int(np.count_nonzero(values == 0)),
        "minimum": int(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "maximum": int(values.max()),
        "total_selected_voxels": int(values.sum(dtype=np.int64)),
    }


def scan_train_masks(
    v35: dict[str, Any], target_mean: float, boundaries: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        data, cache = _open_split(row, "train")
        counts: list[int] = []
        try:
            for index in range(int(row["train_objects"])):
                base = _backbone(cache, index).astype(np.float64) + target_mean
                counts.append(int(np.count_nonzero(base >= float(boundaries[domain]))))
        finally:
            data.close()
            cache.close()
        result[domain] = count_summary(counts)
        print(
            f"[v63-preflight] mask scan {domain} "
            f"{result[domain]['objects']} objects minimum={result[domain]['minimum']}",
            flush=True,
        )
    return result


def _gradient(
    output: torch.Tensor,
    closure: Callable[[torch.Tensor], torch.Tensor],
    selected: int,
) -> tuple[float, dict[str, float]]:
    return _local_gradient(output, closure, selected)


def preflight(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, grid, thresholds, v56_preflight, v62_audit, _ = load_program(
        program_path, repo
    )
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V63 preflight requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V63 preflight requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    frozen = program["frozen_inputs"]
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    boundary_values = program["sealed_q99_9_backbone_boundaries"]
    occupancy = scan_train_masks(v35, target_mean, boundary_values)
    occupancy_pass = all(
        row["zero_count_objects"] == 0 and row["minimum"] > 0
        for row in occupancy.values()
    )
    real = program["hard_preflight"]["real_batch"]
    del real
    condition, target, backbone = _real_batch(v35, prepared, device, 0, 7)
    prepared.close()
    boundaries = torch.tensor(
        [boundary_values[domain] for domain in DOMAIN_ORDER],
        dtype=torch.float64,
        device=device,
    )
    nodes64, weights64 = _quadrature_rule(64, device)
    nodes32, weights32 = _quadrature_rule(32, device)
    v54_thresholds = torch.tensor(thresholds["common_log10rho_thresholds"], device=device)
    grid_thresholds = torch.tensor(grid["thresholds_log10rho"], device=device)
    grid_weights = torch.tensor(grid["physical_moment_weights"], device=device)
    model = _same_seed_model(device).eval()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V63 architecture differs")
    torch.cuda.reset_peak_memory_stats(device)
    output = model(condition)
    bias = torch.tensor(INITIAL_BIASES, device=device).reshape(1, 15, 1, 1, 1)
    initial_output_error = float((output.detach() - bias).abs().max().cpu())
    base = v56_composite_loss(
        output,
        target,
        backbone,
        target_mean,
        target_std,
        v54_thresholds,
        grid_thresholds,
        grid_weights,
    )
    primary, truth, counts = conditional_physical_moments(
        output,
        target,
        backbone,
        target_mean,
        target_std,
        boundaries,
        nodes64,
        weights64,
    )
    with torch.no_grad():
        control, control_truth, control_counts = conditional_physical_moments(
            output.detach(),
            target,
            backbone,
            target_mean,
            target_std,
            boundaries,
            nodes32,
            weights32,
        )
    if counts != control_counts or not torch.equal(truth.detach(), control_truth):
        raise RuntimeError("V63 quadrature control binding differs")
    candidate = conditional_log_moment_score(primary, truth)
    coefficient = float(program["single_model_change"]["coefficient"])
    composite = base[0] + coefficient * candidate
    expected_base = {
        "composite": float(v56_preflight["real_source_balanced_composite_loss"]),
        "bounded_NLL": float(v56_preflight["real_source_balanced_bounded_NLL"]),
        "V54_tail_score": float(v56_preflight["real_source_balanced_V54_tail_score"]),
        "V56_grid_score": float(v56_preflight["real_source_balanced_upper_grid_score"]),
    }
    measured_base = {
        "composite": float(base[0].detach().cpu()),
        "bounded_NLL": float(base[1].detach().cpu()),
        "V54_tail_score": float(base[2].detach().cpu()),
        "V56_grid_score": float(base[4].detach().cpu()),
    }
    base_pass = initial_output_error == 0.0 and all(
        _close(measured_base[key], expected_base[key]) for key in expected_base
    )
    expected_model = v62_audit["models"]["same_seed_initialization"]
    domains: dict[str, Any] = {}
    candidate_reproduction_pass = _close(
        float(candidate.detach().cpu()), float(expected_model["candidate_score"])
    )
    convergence_pass = True
    maximum_convergence = 0.0
    for index, domain in enumerate(DOMAIN_ORDER):
        convergence = float(
            (
                torch.abs(primary[index].detach() - control[index])
                / torch.maximum(
                    torch.maximum(primary[index].detach().abs(), control[index].abs()),
                    torch.tensor(1.0e-300, dtype=torch.float64, device=device),
                )
            ).cpu()
        )
        expected = expected_model["domains"][domain]
        domain_pass = (
            counts[index] == int(expected["selected_voxels"])
            and _close(float(primary[index].detach().cpu()), float(expected["predicted_mean_delta_squared_64"]))
            and _close(float(truth[index].detach().cpu()), float(expected["truth_mean_delta_squared"]))
        )
        candidate_reproduction_pass = bool(candidate_reproduction_pass and domain_pass)
        maximum_convergence = max(maximum_convergence, convergence)
        convergence_pass = bool(
            convergence_pass
            and convergence
            <= float(program["hard_preflight"]["maximum_32_to_64_relative_difference"])
        )
        domains[domain] = {
            "selected_voxels": counts[index],
            "truth_mean_delta_squared": float(truth[index].detach().cpu()),
            "predicted_mean_delta_squared_64": float(primary[index].detach().cpu()),
            "predicted_mean_delta_squared_32": float(control[index].cpu()),
            "predicted_over_truth_64": float((primary[index] / truth[index]).detach().cpu()),
            "quadrature_32_to_64_relative_difference": convergence,
            "reproduces_V62": domain_pass,
        }

    selected = sum(counts)

    def candidate_closure(value: torch.Tensor) -> torch.Tensor:
        predicted, observed, _ = conditional_physical_moments(
            value,
            target,
            backbone,
            target_mean,
            target_std,
            boundaries,
            nodes64,
            weights64,
        )
        return conditional_log_moment_score(predicted, observed)

    def nll_closure(value: torch.Tensor) -> torch.Tensor:
        return -bounded_mixture_log_probability(value, target).mean()

    candidate_value, candidate_gradient = _gradient(output, candidate_closure, selected)
    nll_value, nll_gradient = _gradient(output, nll_closure, selected)
    gradient_ratio = coefficient * candidate_gradient["L2"] / nll_gradient["L2"]
    interval = program["hard_preflight"]["candidate_to_NLL_output_gradient_L2_ratio_interval"]
    gradient_scale_pass = float(interval[0]) <= gradient_ratio <= float(interval[1])
    model.zero_grad(set_to_none=True)
    composite.backward()
    squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    maximum_gradient = 0.0
    gradient_finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_finite = bool(gradient_finite and torch.isfinite(parameter.grad).all().item())
        squared_norm += torch.sum(torch.square(parameter.grad.double()))
        maximum_gradient = max(
            maximum_gradient, float(parameter.grad.detach().abs().max().cpu())
        )
    gradient_l2 = float(torch.sqrt(squared_norm).detach().cpu())
    peak = int(torch.cuda.max_memory_allocated(device))
    gradient_pass = bool(
        gradient_finite
        and math.isfinite(gradient_l2)
        and gradient_l2 > 0.0
        and candidate_gradient["L2"] > 0.0
        and nll_gradient["L2"] > 0.0
    )
    memory_pass = peak < int(program["hard_preflight"]["peak_allocated_bytes_limit"])
    passed = all(
        (
            occupancy_pass,
            base_pass,
            candidate_reproduction_pass,
            convergence_pass,
            gradient_scale_pass,
            gradient_pass,
            memory_pass,
        )
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "parameters": PARAMETERS,
        "train_mask_occupancy": occupancy,
        "train_mask_occupancy_pass": occupancy_pass,
        "initial_output_maximum_error": initial_output_error,
        "V56_base_expected": expected_base,
        "V56_base_measured": measured_base,
        "V56_base_reproduction_pass": base_pass,
        "candidate_score": candidate_value,
        "candidate_coefficient": coefficient,
        "candidate_domains": domains,
        "candidate_reproduces_V62": candidate_reproduction_pass,
        "maximum_32_to_64_relative_difference": maximum_convergence,
        "quadrature_convergence_pass": convergence_pass,
        "candidate_output_gradient": candidate_gradient,
        "bounded_NLL": nll_value,
        "bounded_NLL_output_gradient": nll_gradient,
        "coefficient_scaled_candidate_to_NLL_output_gradient_L2_ratio": gradient_ratio,
        "allowed_candidate_to_NLL_gradient_ratio_interval": interval,
        "gradient_scale_pass": gradient_scale_pass,
        "composite_loss": float(composite.detach().cpu()),
        "composite_identity_absolute_error": abs(
            float(composite.detach().cpu())
            - (float(base[0].detach().cpu()) + coefficient * candidate_value)
        ),
        "full_model_gradient_finite": gradient_finite,
        "full_model_gradient_L2": gradient_l2,
        "full_model_gradient_maximum_absolute": maximum_gradient,
        "gradient_pass": gradient_pass,
        "peak_allocated_bytes": peak,
        "peak_allocated_bytes_limit": program["hard_preflight"]["peak_allocated_bytes_limit"],
        "memory_pass": memory_pass,
        "training_performed": False,
        "validation_accessed": False,
        "development_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
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
        raise FileExistsError("V63 refuses existing preflight")
    result = preflight(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
