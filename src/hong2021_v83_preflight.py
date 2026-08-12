#!/usr/bin/env python
"""Hard train-only preflight for the frozen V83 marginal spline model."""
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
from hong2021_v83_contract import (
    DOMAIN_ORDER,
    load_program,
    validate_train_artifacts,
)
from hong2021_v83_network import (
    TAIL_BOUND,
    conditional_forward,
    conditional_inverse,
    conditional_log_probability,
    parameter_count,
)
from hong2021_v83_train import seeded_model


SCHEMA = "hong2021-v83-conditional-marginal-spline-hard-preflight-v1"
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
        raise RuntimeError("V83 preflight requires clean frozen Lageunha Ada")
    if output_path.exists():
        raise FileExistsError("V83 preflight refuses an existing output")
    program, v35, partition = load_program(program_path, repo, commit)
    if output_path.resolve() != Path(program["output_roots"]["preflight"]).resolve():
        raise ValueError("V83 preflight output differs")
    validate_train_artifacts(program, v35)
    frozen = program["frozen_inputs"]
    if (
        conditioning_cache.resolve() != Path(frozen["conditioning_cache"]).resolve()
        or cache_sha256 != frozen["conditioning_cache_sha256"]
        or sha256_file(conditioning_cache) != cache_sha256
    ):
        raise ValueError("V83 preflight conditioning cache differs")
    prepared = load_cache(conditioning_cache, cache_sha256, commit)
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    extrema: dict[str, dict[str, float | int]] = {}
    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    try:
        target_mean = np.float32(prepared["target_mean"][()])
        target_std = np.float32(prepared["target_std"][()])
        for domain in DOMAIN_ORDER:
            data, cache = handles[domain]
            minimum = np.inf
            maximum = -np.inf
            outside = 0
            voxel_count = 0
            for index in range(int(v35["development_domains"][domain]["train_objects"])):
                truth = np.asarray(data["target"][index, 0], dtype=np.float32)
                backbone = _backbone(cache, index).astype(np.float32)
                target = (truth - backbone - target_mean) / target_std
                if not np.isfinite(target).all():
                    raise RuntimeError(f"V83 {domain} target is nonfinite")
                minimum = min(minimum, float(target.min()))
                maximum = max(maximum, float(target.max()))
                outside += int(np.count_nonzero(np.abs(target) >= TAIL_BOUND))
                voxel_count += int(target.size)
            extrema[domain] = {
                "minimum": minimum,
                "maximum": maximum,
                "voxels": voxel_count,
                "outside_open_spline_bound": outside,
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
    support_pass = all(
        int(row["outside_open_spline_bound"]) == 0 for row in extrema.values()
    )
    device = torch.device("cuda")
    model = seeded_model(device)
    expected_parameters = int(program["model"]["trainable_parameters"])
    if parameter_count(model) != expected_parameters:
        raise RuntimeError("V83 preflight model parameter count differs")
    condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
    target_tensor = torch.from_numpy(np.stack(targets)).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        parameters = model(condition_tensor)
    latent, logdet = conditional_forward(parameters, target_tensor)
    loss = -conditional_log_probability(parameters, target_tensor).mean()
    loss.backward()
    gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = float(torch.linalg.vector_norm(torch.cat(gradients)).cpu())
    finite_gradient = bool(
        np.isfinite(gradient_norm)
        and all(bool(torch.isfinite(value).all().cpu()) for value in gradients)
    )
    identity_error = float(torch.max(torch.abs(latent - target_tensor)).detach().cpu())
    identity_logdet = float(torch.max(torch.abs(logdet)).detach().cpu())
    cropped_parameters = parameters[:1, :, :4, :4, :4].detach()
    cropped_target = target_tensor[:1, :, :4, :4, :4]
    cropped_latent, forward_logdet = conditional_forward(
        cropped_parameters, cropped_target
    )
    recovered, inverse_logdet = conditional_inverse(
        cropped_parameters, cropped_latent
    )
    inverse_error = float(torch.max(torch.abs(recovered - cropped_target)).cpu())
    logdet_error = float(
        torch.max(torch.abs(forward_logdet + inverse_logdet)).cpu()
    )
    peak = int(torch.cuda.max_memory_allocated(device))
    numerical_pass = (
        bool(torch.isfinite(loss).detach().cpu())
        and finite_gradient
        and gradient_norm > 0.0
        and identity_error < 5.0e-5
        and identity_logdet < 5.0e-5
        and inverse_error < 5.0e-4
        and logdet_error < 5.0e-4
    )
    memory_pass = peak < MEMORY_LIMIT
    passed = support_pass and numerical_pass and memory_pass
    result = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "program": str(program_path.resolve()),
        "program_sha256": sha256_file(program_path),
        "code_commit": commit,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "fit_target_extrema": extrema,
        "all_train_targets_strictly_inside_spline_bound": support_pass,
        "real_batch_domains": list(DOMAIN_ORDER),
        "real_batch_fit_indices": {
            domain: partition[domain]["fit"][0] for domain in DOMAIN_ORDER
        },
        "real_batch_cube_isometry": 7,
        "initial_conditional_NLL": float(loss.detach().cpu()),
        "initial_identity_maximum_error": identity_error,
        "initial_identity_logdet_maximum": identity_logdet,
        "inverse_maximum_error": inverse_error,
        "forward_inverse_logdet_maximum_error": logdet_error,
        "gradient_norm": gradient_norm,
        "gradient_finite_and_nonzero": finite_gradient and gradient_norm > 0.0,
        "numerical_pass": numerical_pass,
        "peak_allocated_bytes": peak,
        "peak_allocated_limit_bytes": MEMORY_LIMIT,
        "memory_pass": memory_pass,
        "training_performed": False,
        "holdout_payload_accessed": False,
        "validation_payload_accessed": False,
        "development_payload_accessed": False,
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
        raise RuntimeError("V83 hard preflight failed")
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
