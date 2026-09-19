#!/usr/bin/env python
"""Train-only high-backbone physical-moment gate for V56."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v46_tail_occupancy_audit import EXPECTED_OBJECTS, PROBE_VOXELS, _probe_indices
from hong2021_v50_network import LocalMixtureUNet, mixture_parameters, parameter_count
from hong2021_v51_bounded_support_audit import (
    _physical_delta_squared,
    _quadrature_object,
    _relative_difference,
    _truth_probe,
)
from hong2021_v54_train import CONTROL_QUADRATURE_ORDER, PRIMARY_QUADRATURE_ORDER
from hong2021_v56_train import (
    CHECKPOINT_SCHEMA,
    GRID_COEFFICIENT,
    PARAMETERS,
    PROGRAM_SHA256,
    REPORT_SCHEMA,
    TAIL_COEFFICIENT,
    _resolve,
    load_cache,
    load_grid,
    load_program,
)


SCHEMA = "hong2021-v56-train-only-high-backbone-mechanism-decision-v1"
STRATUM_QUANTILES = (0.9, 0.99, 0.999)
STRATUM_LABELS = ("below_q90", "q90_to_q99", "q99_to_q99_9", "q99_9_and_above")
RATIO_MINIMUM = 2.0 / 3.0
RATIO_MAXIMUM = 1.5
QUADRATURE_MAXIMUM = 0.005


def mechanism_pass(ratios: dict[str, float], convergence: dict[str, float]) -> bool:
    return all(
        RATIO_MINIMUM <= ratios.get(domain, math.inf) <= RATIO_MAXIMUM
        and convergence.get(domain, math.inf) <= QUADRATURE_MAXIMUM
        for domain in DOMAIN_ORDER
    )


def _load_fit(
    program: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha: str,
    report_path: Path,
    report_sha: str,
    grid_path: Path,
    grid_sha: str,
    threshold_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    cache_sha: str,
    repo: Path,
    commit: str,
) -> tuple[LocalMixtureUNet, dict[str, Any]]:
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("V56 checkpoint hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = json.loads(report_path.read_text())
    if sha256_file(report_path) != report_sha or sha256_file(preflight_path) != preflight_sha:
        raise ValueError("V56 report or preflight hash differs")
    source_commit = str(checkpoint.get("code_commit"))
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != PROGRAM_SHA256
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("v54_threshold_selection_sha256") != threshold_sha
        or checkpoint.get("grid_sha256") != grid_sha
        or checkpoint.get("preflight_sha256") != preflight_sha
        or checkpoint.get("conditioning_cache_sha256") != cache_sha
        or checkpoint.get("tail_coefficient") != TAIL_COEFFICIENT
        or checkpoint.get("grid_coefficient") != GRID_COEFFICIENT
        or report.get("schema") != REPORT_SCHEMA
        or report.get("checkpoint_sha256") != checkpoint_sha
        or report.get("grid_sha256") != grid_sha
        or report.get(
            "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection"
        )
        is not False
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, commit],
            cwd=repo,
            capture_output=True,
        ).returncode
    ):
        raise ValueError("V56 fit binding differs")
    load_grid(grid_path, grid_sha, source_commit, threshold_sha)
    model = LocalMixtureUNet()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V56 architecture differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model, checkpoint


def _masks(value: np.ndarray, boundaries: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        value < boundaries[0],
        (value >= boundaries[0]) & (value < boundaries[1]),
        (value >= boundaries[1]) & (value < boundaries[2]),
        value >= boundaries[2],
    )


@torch.inference_mode()
def _domain(
    model: LocalMixtureUNet,
    device: torch.device,
    v35: dict[str, Any],
    prepared: h5py.File,
    support: dict[str, Any],
    v53: dict[str, Any],
    v54_gate: dict[str, Any],
    domain: str,
    domain_index: int,
) -> dict[str, Any]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V56 gate object count differs")
    truth_probe = _truth_probe(v35, prepared, domain, domain_index)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    domain_maximum = float(support["domains"][domain]["maximum_standardized_residual"])
    nodes64, weights64 = np.polynomial.hermite.hermgauss(PRIMARY_QUADRATURE_ORDER)
    nodes32, weights32 = np.polynomial.hermite.hermgauss(CONTROL_QUADRATURE_ORDER)
    nodes64_t = torch.from_numpy(nodes64).to(device)
    weights64_t = torch.from_numpy(weights64).to(device)
    nodes32_t = torch.from_numpy(nodes32).to(device)
    weights32_t = torch.from_numpy(weights32).to(device)
    primary_parts: list[np.ndarray] = []
    primary_sum = control_sum = 0.0
    data, cache = _open_split(row, "train")
    try:
        from hong2021_v48_train import condition_cube

        for object_index in range(objects):
            condition, _, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            parameter = model(torch.from_numpy(condition[None]).to(device))
            indices = _probe_indices(domain_index, object_index)
            index_tensor = torch.from_numpy(indices).to(device)
            flat = (
                parameter.reshape(1, 15, -1)
                .index_select(2, index_tensor)
                .reshape(1, 15, 1, 1, -1)
            )
            logits, locations, scales = mixture_parameters(flat)
            mixture_weights = torch.softmax(logits, dim=1)[0, :, 0, 0]
            locations = locations[0, :, 0, 0]
            scales = scales[0, :, 0, 0]
            base = torch.from_numpy(
                backbone.reshape(-1)[indices].astype(np.float64) + target_mean
            ).to(device)
            primary = _quadrature_object(
                mixture_weights,
                locations,
                scales,
                base,
                target_std,
                nodes64_t,
                weights64_t,
                domain_maximum,
            )
            control = _quadrature_object(
                mixture_weights,
                locations,
                scales,
                base,
                target_std,
                nodes32_t,
                weights32_t,
                domain_maximum,
            )
            primary_value = primary["delta_squared"].sum(axis=0)
            control_value = control["delta_squared"].sum(axis=0)
            primary_parts.append(primary_value)
            primary_sum += float(primary_value.sum(dtype=np.float64))
            control_sum += float(control_value.sum(dtype=np.float64))
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(f"[v56-train-gate] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()
    predicted = np.concatenate(primary_parts)
    truth = _physical_delta_squared(truth_probe["physical_y"])
    variable = truth_probe["backbone_base"].astype(np.float64)
    boundaries = np.quantile(variable, STRATUM_QUANTILES)
    sealed = v53["train_only_high_backbone_probe"][domain]
    if not np.allclose(
        boundaries, sealed["V50_sealed"]["boundaries"], rtol=0.0, atol=1.0e-7
    ):
        raise ValueError("V56 gate boundaries differ")
    rows: dict[str, Any] = {}
    for label, mask in zip(STRATUM_LABELS, _masks(variable, boundaries), strict=True):
        truth_mean = float(np.mean(truth[mask], dtype=np.float64))
        predicted_mean = float(np.mean(predicted[mask], dtype=np.float64))
        ratio = predicted_mean / truth_mean
        rows[label] = {
            "count": int(mask.sum()),
            "truth_mean_delta_squared": truth_mean,
            "V56_quadrature_mean_delta_squared": predicted_mean,
            "V56_over_truth_mean_delta_squared": ratio,
            "V50_over_truth_mean_delta_squared": float(
                sealed["V50_sealed"]["strata"][label][
                    "quadrature_over_truth_mean_delta_squared"
                ]
            ),
            "V52_over_truth_mean_delta_squared": float(
                sealed["V52"]["strata"][label]["quadrature_over_truth_mean_delta_squared"]
            ),
            "V54_over_truth_mean_delta_squared": float(
                v54_gate["domains"][domain]["strata"][label][
                    "V54_over_truth_mean_delta_squared"
                ]
            ),
        }
    convergence = _relative_difference(primary_sum, control_sum)
    if convergence > QUADRATURE_MAXIMUM:
        raise RuntimeError("V56 gate quadrature convergence differs")
    top_ratio = float(rows["q99_9_and_above"]["V56_over_truth_mean_delta_squared"])
    return {
        "train_objects": objects,
        "probe_voxels": objects * PROBE_VOXELS,
        "backbone_boundaries": boundaries.tolist(),
        "strata": rows,
        "aggregate_32_to_64_relative_difference": convergence,
        "top_backbone_pass": RATIO_MINIMUM <= top_ratio <= RATIO_MAXIMUM,
    }


def evaluate(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    threshold_sha: str,
    grid_path: Path,
    grid_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    report_path: Path,
    report_sha: str,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V56 mechanism gate requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V56 mechanism gate requires Ada")
    model, checkpoint = _load_fit(
        program,
        checkpoint_path,
        checkpoint_sha,
        report_path,
        report_sha,
        grid_path,
        grid_sha,
        threshold_sha,
        preflight_path,
        preflight_sha,
        cache_sha,
        repo.resolve(),
        commit,
    )
    model = model.to("cuda").eval()
    prepared = load_cache(cache_path, cache_sha, str(checkpoint["code_commit"]))
    frozen = program["frozen_inputs"]
    support = json.loads(_resolve(repo.resolve(), frozen["support_selection"]).read_text())
    v54_program = json.loads(_resolve(repo.resolve(), frozen["v54_program"]).read_text())
    v53_path = _resolve(repo.resolve(), v54_program["frozen_inputs"]["v53_audit"])
    if sha256_file(v53_path) != v54_program["frozen_inputs"]["v53_audit_sha256"]:
        raise ValueError("V56 V53 audit hash differs")
    v53 = json.loads(v53_path.read_text())
    v54_gate = json.loads(_resolve(repo.resolve(), frozen["v54_train_gate"]).read_text())
    domains: dict[str, Any] = {}
    try:
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            domains[domain] = _domain(
                model,
                torch.device("cuda"),
                v35,
                prepared,
                support,
                v53,
                v54_gate,
                domain,
                domain_index,
            )
    finally:
        prepared.close()
    ratios = {
        domain: float(row["strata"]["q99_9_and_above"]["V56_over_truth_mean_delta_squared"])
        for domain, row in domains.items()
    }
    convergence = {
        domain: float(row["aggregate_32_to_64_relative_difference"])
        for domain, row in domains.items()
    }
    passed = mechanism_pass(ratios, convergence)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only_mechanism_gate",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint_sha256": checkpoint_sha,
        "training_report_sha256": report_sha,
        "grid_sha256": grid_sha,
        "preflight_sha256": preflight_sha,
        "domains": domains,
        "train_mechanism_pass": passed,
        "classification": (
            "train_high_backbone_physical_moments_calibrated"
            if passed
            else "proper_upper_survival_grid_does_not_calibrate_train_high_backbone_physical_moments"
        ),
        "next": (
            "proceed_to_locked_V56_development_sampling"
            if passed
            else "stop_before_development_sampling_and_audit_grid_survival_calibration_and_component_tail_amplitudes"
        ),
        "development_accessed": False,
        "training_or_refit_performed_by_gate": False,
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
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--thresholds-sha256", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--grid-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V56 refuses existing train mechanism gate")
    result = evaluate(
        args.program,
        args.repo,
        args.cache,
        args.cache_sha256,
        args.thresholds_sha256,
        args.grid,
        args.grid_sha256,
        args.preflight,
        args.preflight_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
        args.report,
        args.report_sha256,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
