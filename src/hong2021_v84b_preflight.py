#!/usr/bin/env python
"""Hard fit-only preflight for the frozen V84B spliced-tail model."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import numpy as np
import torch

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v84b_contract import DOMAIN_ORDER, load_program, validate_train_artifacts
from hong2021_v84b_network import (
    CORE_PARAMETERS,
    INITIAL_LOWER_SCALE,
    INITIAL_TAIL_MASS,
    INITIAL_UPPER_SCALE,
    LOWER_THRESHOLD,
    UPPER_THRESHOLD,
    conditional_cdf,
    conditional_icdf,
    conditional_log_probability,
    parameter_count,
    spliced_parameters,
    upper_physical_second_moment_margin,
)
from hong2021_v84b_train import seeded_model


SCHEMA = "hong2021-v84b-spliced-tail-hard-preflight-v1"
MEMORY_LIMIT = 28 * 1024**3


def preflight(
    program_path: Path,
    repo: Path,
    conditioning_cache: Path,
    cache_sha256: str,
    output_path: Path,
) -> dict:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if (
        not clean
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or not torch.cuda.is_available()
        or "ada" not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError("V84B preflight requires clean frozen Lageunha Ada")
    if output_path.exists():
        raise FileExistsError("V84B preflight refuses existing output")
    program, v35, partition = load_program(program_path, repo, commit)
    if output_path.resolve() != Path(program["output_roots"]["preflight"]).resolve():
        raise ValueError("V84B preflight output differs")
    validate_train_artifacts(program, v35)
    frozen = program["frozen_inputs"]
    if (
        conditioning_cache.resolve() != Path(frozen["conditioning_cache"]).resolve()
        or cache_sha256 != frozen["conditioning_cache_sha256"]
        or sha256_file(conditioning_cache) != cache_sha256
    ):
        raise ValueError("V84B preflight conditioning cache differs")
    prepared = load_cache(conditioning_cache, cache_sha256, commit)
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    fit_scan: dict[str, dict[str, float | int]] = {}
    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    try:
        target_mean = np.float32(prepared["target_mean"][()])
        target_std = np.float32(prepared["target_std"][()])
        for domain in DOMAIN_ORDER:
            data, cache = handles[domain]
            minimum = np.inf
            maximum = -np.inf
            lower_count = 0
            upper_count = 0
            lower_excess = 0.0
            upper_excess = 0.0
            voxel_count = 0
            for index in partition[domain]["fit"]:
                truth = np.asarray(data["target"][index, 0], dtype=np.float32)
                backbone = _backbone(cache, index).astype(np.float32)
                target = (truth - backbone - target_mean) / target_std
                if not np.isfinite(target).all():
                    raise RuntimeError(f"V84B {domain} target is nonfinite")
                lower = target < LOWER_THRESHOLD
                upper = target > UPPER_THRESHOLD
                minimum = min(minimum, float(target.min()))
                maximum = max(maximum, float(target.max()))
                lower_count += int(lower.sum())
                upper_count += int(upper.sum())
                lower_excess += float((LOWER_THRESHOLD - target[lower]).sum(dtype=np.float64))
                upper_excess += float((target[upper] - UPPER_THRESHOLD).sum(dtype=np.float64))
                voxel_count += int(target.size)
            fit_scan[domain] = {
                "fit_objects": len(partition[domain]["fit"]),
                "minimum": minimum,
                "maximum": maximum,
                "voxels": voxel_count,
                "lower_tail_voxels": lower_count,
                "upper_tail_voxels": upper_count,
                "lower_tail_fraction": lower_count / voxel_count,
                "upper_tail_fraction": upper_count / voxel_count,
                "lower_mean_excess": lower_excess / lower_count,
                "upper_mean_excess": upper_excess / upper_count,
            }
            index = partition[domain]["fit"][0]
            condition, target, _ = condition_cube(
                data, cache, prepared, domain, "train", index
            )
            axes, reflections = CUBE_ISOMETRIES[7]
            conditions.append(apply_cube_isometry(condition, axes, reflections))
            targets.append(apply_cube_isometry(target, axes, reflections))
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        prepared.close()
    tail_sample_pass = all(
        int(row["lower_tail_voxels"]) >= 100_000
        and int(row["upper_tail_voxels"]) >= 100_000
        for row in fit_scan.values()
    )
    physical_margin = upper_physical_second_moment_margin(float(target_std))
    physical_moment_pass = physical_margin > 0.0
    device = torch.device("cuda")
    model = seeded_model(device)
    expected_parameters = int(program["model"]["trainable_parameters"])
    if parameter_count(model) != expected_parameters:
        raise RuntimeError("V84B model parameter count differs")
    condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
    target_tensor = torch.from_numpy(np.stack(targets)).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        parameters = model(condition_tensor)
    _, weights, lower_scale, upper_scale = spliced_parameters(parameters)
    loss = -conditional_log_probability(parameters, target_tensor).mean()
    loss.backward()
    gradient = model.output.bias.grad.detach().float()
    gradient_norm = float(
        torch.linalg.vector_norm(
            torch.cat(
                [
                    parameter.grad.detach().float().reshape(-1)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            )
        ).cpu()
    )
    core_gradient_norm = float(torch.linalg.vector_norm(gradient[:CORE_PARAMETERS]).cpu())
    tail_gradient_norm = float(torch.linalg.vector_norm(gradient[CORE_PARAMETERS:]).cpu())
    probe_uniform = torch.tensor(
        [0.0001, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 2, 2, 2)
    probe_parameters = parameters[:1, :, :2, :2, :2].detach()
    recovered = conditional_cdf(
        probe_parameters, conditional_icdf(probe_parameters, probe_uniform)
    )
    round_trip_error = float(torch.max(torch.abs(recovered - probe_uniform)).cpu())
    peak = int(torch.cuda.max_memory_allocated(device))
    initialization_pass = (
        abs(float(weights.mean(dim=(0, 2, 3, 4))[0].detach().cpu()) - INITIAL_TAIL_MASS)
        < 2.0e-4
        and abs(
            float(weights.mean(dim=(0, 2, 3, 4))[2].detach().cpu())
            - INITIAL_TAIL_MASS
        )
        < 2.0e-4
        and abs(float(lower_scale.mean().detach().cpu()) - INITIAL_LOWER_SCALE) < 2.0e-3
        and abs(float(upper_scale.mean().detach().cpu()) - INITIAL_UPPER_SCALE) < 2.0e-3
    )
    numerical_pass = bool(
        torch.isfinite(loss).detach().cpu()
        and np.isfinite(gradient_norm)
        and gradient_norm > 0.0
        and core_gradient_norm > 0.0
        and tail_gradient_norm > 0.0
        and round_trip_error < 5.0e-4
        and initialization_pass
    )
    memory_pass = peak < MEMORY_LIMIT
    passed = tail_sample_pass and physical_moment_pass and numerical_pass and memory_pass
    result = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "program": str(program_path.resolve()),
        "program_sha256": sha256_file(program_path),
        "code_commit": commit,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "group_fit_scan": fit_scan,
        "at_least_100000_fit_voxels_in_each_tail_and_domain": tail_sample_pass,
        "target_std": float(target_std),
        "upper_physical_second_moment_margin": physical_margin,
        "upper_physical_second_moment_finite_by_construction": physical_moment_pass,
        "real_batch_domains": list(DOMAIN_ORDER),
        "real_batch_fit_indices": {
            domain: partition[domain]["fit"][0] for domain in DOMAIN_ORDER
        },
        "initial_conditional_NLL": float(loss.detach().cpu()),
        "gradient_norm": gradient_norm,
        "core_head_gradient_norm": core_gradient_norm,
        "tail_head_gradient_norm": tail_gradient_norm,
        "initialization_pass": initialization_pass,
        "CDF_ICDF_maximum_round_trip_error": round_trip_error,
        "numerical_pass": numerical_pass,
        "peak_allocated_bytes": peak,
        "peak_allocated_limit_bytes": MEMORY_LIMIT,
        "memory_pass": memory_pass,
        "training_performed": False,
        "group_holdout_payload_accessed": False,
        "validation_payload_accessed": False,
        "consumed_development_payload_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise RuntimeError("V84B hard preflight failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--conditioning-cache", type=Path, required=True)
    parser.add_argument("--conditioning-cache-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    preflight(
        args.program,
        args.repo,
        args.conditioning_cache,
        args.conditioning_cache_sha256,
        args.out,
    )


if __name__ == "__main__":
    main()
