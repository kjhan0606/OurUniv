#!/usr/bin/env python
"""Hard preflight for the frozen V70 query-aligned latent spatial model."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri
import torch

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v50_network import RANK_EPSILON, bounded_mixture_cdf
from hong2021_v63_preflight import _path, load_program as load_v63_program
from hong2021_v63_train import _is_ancestor
from hong2021_v63_train_gate import _load_fit
from hong2021_v70_network import (
    ATTENTION_HEADS,
    BASE_CHANNELS,
    CONDITION_CHANNELS,
    TIME_CHANNELS,
    LatentSpatialUNet,
    edm_coefficients,
    edm_loss,
    parameter_count,
)


PROGRAM_SCHEMA = (
    "hong2021-v70-train-only-query-aligned-latent-spatial-score-model-program-v1"
)
PROGRAM_SHA256 = "79f1b5fe1462664b9b7a237bd82a821e205f3901603d64801a01b328c43f7e42"
PROGRAM_FREEZE_COMMIT = "692511d62d4f5a999b47a6bdecc417fce7df9764"
SCHEMA = "hong2021-v70-query-aligned-latent-spatial-hard-preflight-v1"
SEED = 170070
FIXED_REPRESENTATION_INDICES = (0, 1, 2, 3)
FIXED_ISOMETRY = 7
V65_PROGRAM_SHA256 = "58c244e03a5f7fbb9cef29943869067fe3c202d01f3f3773d3cb69d4022bcc21"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V70 {label} hash differs")
    return _json(path)


def _artifact_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_implementation_preflight_cache_training_sampling_or_development_evaluation"
    ):
        raise ValueError("V70 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _artifact_path(repo, parent["v69_result_record"]),
        parent["v69_result_record_sha256"],
        "V69 result record",
    )
    if (
        record.get("status") != parent["required_status"]
        or record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("next") != parent["required_next"]
        or record.get("audit", {}).get("candidate_selected")
        is not parent["required_candidate_selected"]
        or record.get("firewall", {}).get("training_or_refit_performed")
        is not parent["required_training_or_refit_performed"]
        or record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V70 V69 conclusion or firewall differs")
    for key, value in program["frozen_inputs"].items():
        if key.endswith("_sha256"):
            continue
        digest = program["frozen_inputs"].get(f"{key}_sha256")
        if digest is not None and sha256_file(_artifact_path(repo, value)) != digest:
            raise ValueError(f"V70 frozen input differs: {key}")
    historical = program["historical_V7_failure_control"]
    for key in ("configuration", "training_run", "training_history"):
        digest = historical[f"{key}_sha256"]
        if sha256_file(_artifact_path(repo, historical[key])) != digest:
            raise ValueError(f"V70 historical V7 artifact differs: {key}")
    checkpoint = (
        Path(program["historical_V7_failure_control"]["training_run"]).parent
        / "minimum_validation.pt"
    )
    if sha256_file(checkpoint) != historical["selected_checkpoint_sha256"]:
        raise ValueError("V70 historical V7 checkpoint differs")
    v63, v35, _, _, _, _, _ = load_v63_program(
        _path(repo, program["frozen_inputs"]["v63_program"]), repo
    )
    v63_record = _json(_path(repo, program["frozen_inputs"]["v63_result_record"]))
    if (
        v63_record.get("train_only_mechanism_decision", {}).get(
            "train_mechanism_pass"
        )
        is not True
        or v63_record.get("firewall", {}).get("independent_gate_locked") is not True
        or v63_record.get("firewall", {}).get("historical_EAGLE_accessed")
        is not False
    ):
        raise ValueError("V70 inherited V63 evidence differs")
    v65_path = repo / "config/hong2021_v65_structure_factorization_audit_program.json"
    if sha256_file(v65_path) != V65_PROGRAM_SHA256:
        raise ValueError("V70 immutable V65 query definition differs")
    v65 = _json(v65_path)
    return program, v35, v65


def gaussianize_rank(
    rank: np.ndarray, epsilon: float = RANK_EPSILON
) -> tuple[np.ndarray, np.ndarray, int]:
    value = np.asarray(rank, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("V70 conditional ranks differ")
    clipped = np.clip(value, epsilon, 1.0 - epsilon)
    latent = ndtri(clipped).astype(np.float32)
    reconstructed = ndtr(latent.astype(np.float64))
    count = int(np.count_nonzero(value != clipped))
    if not np.isfinite(latent).all() or not np.isfinite(reconstructed).all():
        raise RuntimeError("V70 Gaussianized ranks are nonfinite")
    return latent, reconstructed, count


def representation_summary(
    ranks: list[np.ndarray], latents: list[np.ndarray], reconstructed: list[np.ndarray]
) -> dict[str, Any]:
    rank = np.concatenate([value.reshape(-1) for value in ranks]).astype(np.float64)
    latent = np.concatenate([value.reshape(-1) for value in latents]).astype(np.float64)
    restored = np.concatenate(
        [value.reshape(-1) for value in reconstructed]
    ).astype(np.float64)
    clipped = np.clip(rank, RANK_EPSILON, 1.0 - RANK_EPSILON)
    count = int(np.count_nonzero(rank != clipped))
    return {
        "objects": len(ranks),
        "voxels": int(rank.size),
        "rank_clamp_count": count,
        "rank_clamp_fraction": count / int(rank.size),
        "rank_minimum": float(rank.min()),
        "rank_maximum": float(rank.max()),
        "latent_mean": float(latent.mean()),
        "latent_standard_deviation": float(latent.std()),
        "latent_minimum": float(latent.min()),
        "latent_maximum": float(latent.max()),
        "maximum_normal_CDF_roundtrip_error": float(
            np.max(np.abs(restored - clipped))
        ),
    }


def _model_and_inputs(
    program: dict[str, Any],
    v35: dict[str, Any],
    repo: Path,
    commit: str,
    device: torch.device,
) -> tuple[
    dict[str, Any], torch.Tensor, torch.Tensor, bool, dict[str, float]
]:
    frozen = program["frozen_inputs"]
    v63, _, _, _, _, _, _ = load_v63_program(
        _path(repo, frozen["v63_program"]), repo
    )
    boundaries = {
        domain: float(v63["sealed_q99_9_backbone_boundaries"][domain])
        for domain in DOMAIN_ORDER
    }
    marginal, _ = _load_fit(
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
    marginal = marginal.to(device).eval()
    for parameter in marginal.parameters():
        parameter.requires_grad_(False)
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    summaries: dict[str, Any] = {}
    fixed_conditions: list[np.ndarray] = []
    fixed_latents: list[np.ndarray] = []
    try:
        for domain in DOMAIN_ORDER:
            data, cache = _open_split(v35["development_domains"][domain], "train")
            ranks: list[np.ndarray] = []
            latents: list[np.ndarray] = []
            restored: list[np.ndarray] = []
            try:
                for index in FIXED_REPRESENTATION_INDICES:
                    condition, target, _ = condition_cube(
                        data, cache, prepared, domain, "train", index
                    )
                    condition_tensor = torch.from_numpy(condition[None]).to(device)
                    target_tensor = torch.from_numpy(target[None]).to(device)
                    with torch.no_grad():
                        parameters = marginal(condition_tensor).float()
                        rank = (
                            bounded_mixture_cdf(parameters, target_tensor)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                    latent, reconstructed, _ = gaussianize_rank(rank)
                    ranks.append(rank)
                    latents.append(latent)
                    restored.append(reconstructed)
                    if index == 0:
                        axes, reflections = CUBE_ISOMETRIES[FIXED_ISOMETRY]
                        joined = apply_cube_isometry(
                            np.concatenate((condition, latent[0]), axis=0),
                            axes,
                            reflections,
                        )
                        fixed_conditions.append(joined[:CONDITION_CHANNELS])
                        fixed_latents.append(joined[CONDITION_CHANNELS:])
            finally:
                data.close()
                cache.close()
            summaries[domain] = representation_summary(ranks, latents, restored)
    finally:
        prepared.close()
    marginal_grad_absent = all(
        parameter.grad is None for parameter in marginal.parameters()
    )
    del marginal
    torch.cuda.empty_cache()
    conditions = torch.from_numpy(np.stack(fixed_conditions)).to(device)
    latents = torch.from_numpy(np.stack(fixed_latents)).to(device)
    target_stats = {
        "mean": float(latents.double().mean().cpu()),
        "standard_deviation": float(latents.double().std().cpu()),
    }
    return summaries, conditions, latents, marginal_grad_absent, target_stats


def _preflight_model(
    condition: torch.Tensor, latent: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = LatentSpatialUNet().to(device)
    parameters = parameter_count(model)
    sigma = torch.tensor((0.1, 1.0, 10.0), device=device)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    noise = torch.randn(latent.shape, device=device, generator=generator)
    shape = (len(sigma),) + (1,) * (latent.ndim - 1)
    c_skip, c_out, c_in, c_noise = edm_coefficients(sigma)
    del c_skip, c_out
    noisy = latent + sigma.reshape(shape) * noise
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        response = model(c_in.reshape(shape) * noisy, condition, c_noise)
        permuted = model(
            c_in.reshape(shape) * noisy,
            torch.roll(condition, shifts=1, dims=0),
            c_noise,
        )
    model.zero_grad(set_to_none=True)
    full_loss, full_per_object = edm_loss(model, latent, condition, sigma, noise)
    full_loss.backward()
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    gradient_tensors = len(parameter_gradients)
    gradients_present = all(value is not None for value in parameter_gradients)
    gradients = [value.detach() for value in parameter_gradients if value is not None]
    gradient_tensor_finite = [
        bool(torch.isfinite(value).all().cpu()) for value in gradients
    ]
    gradient_tensor_nonzero = [
        bool(torch.count_nonzero(value).cpu()) for value in gradients
    ]
    gradient_values = torch.cat([value.float().reshape(-1) for value in gradients])
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        amp_loss, amp_per_object = edm_loss(
            model, latent, condition, sigma, noise
        )
    gradient_finite = bool(all(gradient_tensor_finite))
    gradient_nonzero = int(torch.count_nonzero(gradient_values).cpu())
    gradient_total = int(gradient_values.numel())
    peak = int(torch.cuda.max_memory_allocated(device))
    full = float(full_loss.cpu())
    amp = float(amp_loss.detach().cpu())
    relative = abs(amp - full) / max(abs(full), 1.0e-12)
    response_finite = bool(
        torch.isfinite(response).all().cpu()
        and torch.isfinite(permuted).all().cpu()
        and torch.isfinite(full_per_object).all().cpu()
        and torch.isfinite(amp_per_object).all().cpu()
    )
    sensitivity = float((response - permuted).abs().mean().cpu())
    return {
        "base_channels": BASE_CHANNELS,
        "time_channels": TIME_CHANNELS,
        "attention_heads": ATTENTION_HEADS,
        "parameters": parameters,
        "fixed_sigma": sigma.cpu().tolist(),
        "full_precision_EDM_loss": full,
        "AMP_EDM_loss": amp,
        "AMP_to_full_relative_difference": relative,
        "full_precision_per_domain": {
            domain: float(full_per_object[index].cpu())
            for index, domain in enumerate(DOMAIN_ORDER)
        },
        "AMP_per_domain": {
            domain: float(amp_per_object[index].detach().cpu())
            for index, domain in enumerate(DOMAIN_ORDER)
        },
        "condition_permutation_mean_absolute_response": sensitivity,
        "response_finite": response_finite,
        "gradient_finite": gradient_finite,
        "gradient_tensors_expected": gradient_tensors,
        "gradient_tensors_present": len(gradients),
        "gradient_tensors_with_nonzero_norm": int(sum(gradient_tensor_nonzero)),
        "every_parameter_gradient_present": gradients_present,
        "every_parameter_gradient_tensor_finite": bool(all(gradient_tensor_finite)),
        "every_parameter_gradient_tensor_nonzero": bool(all(gradient_tensor_nonzero)),
        "gradient_L2": float(torch.linalg.vector_norm(gradient_values).cpu()),
        "gradient_nonzero_values": gradient_nonzero,
        "gradient_values": gradient_total,
        "gradient_nonzero_fraction": gradient_nonzero / gradient_total,
        "peak_allocated_bytes": peak,
    }


def preflight(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V70 preflight requires clean Lageunha frozen ancestry")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V70 preflight requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    representation, condition, latent, marginal_grad_absent, target_stats = (
        _model_and_inputs(program, v35, repo, commit, device)
    )
    model = _preflight_model(condition, latent, device)
    rules = program["latent_cache"]["full_scan_requirements"]
    representation_pass = all(
        row["rank_clamp_fraction"]
        <= float(rules["maximum_rank_clamp_fraction_each_domain"])
        and abs(row["latent_mean"])
        <= float(rules["maximum_absolute_latent_mean_each_domain"])
        and float(rules["latent_standard_deviation_interval_each_domain"][0])
        <= row["latent_standard_deviation"]
        <= float(rules["latent_standard_deviation_interval_each_domain"][1])
        and row["maximum_normal_CDF_roundtrip_error"] <= 2.0e-7
        for row in representation.values()
    )
    model_pass = bool(
        model["response_finite"]
        and model["gradient_finite"]
        and model["every_parameter_gradient_present"]
        and model["every_parameter_gradient_tensor_finite"]
        and model["every_parameter_gradient_tensor_nonzero"]
        and model["gradient_L2"] > 0.0
        and model["gradient_nonzero_values"] > 0
        and model["condition_permutation_mean_absolute_response"] > 0.0
        and model["AMP_to_full_relative_difference"] <= 0.02
        and model["peak_allocated_bytes"]
        < int(program["hard_preflight"]["peak_allocated_bytes_limit"])
    )
    passed = representation_pass and model_pass and marginal_grad_absent
    source = repo / "src/hong2021_v70_network.py"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "program_sha256": PROGRAM_SHA256,
        "program_freeze_commit": PROGRAM_FREEZE_COMMIT,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "V70_network_source_sha256": sha256_file(source),
        "representation": representation,
        "fixed_model_batch_latent": target_stats,
        "representation_pass": representation_pass,
        "V63_parameters_inference_only": True,
        "V63_parameter_gradients_absent": marginal_grad_absent,
        "model": model,
        "model_pass": model_pass,
        "latent_cache_construction_authorized": passed,
        "optimizer_constructed": False,
        "optimizer_step_performed": False,
        "latent_cache_written": False,
        "validation_accessed": False,
        "development_accessed": False,
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
        raise FileExistsError(f"V70 refuses existing preflight: {args.out}")
    result = preflight(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
