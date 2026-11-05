#!/usr/bin/env python
"""Locked train-only joint-structure mechanism gate for the fixed V70 fit."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v48_train import condition_cube
from hong2021_v50_network import (
    bounded_mixture_cdf,
    bounded_mixture_inverse,
    standard_normal_cdf,
)
from hong2021_residual_evaluate import SpectralBinner, band_key
from hong2021_v63_train import _is_ancestor
from hong2021_v70_latent_cache import _frozen_marginal
from hong2021_v70_network import LatentSpatialUNet, edm_denoise, parameter_count
from hong2021_v70_preflight import PROGRAM_SHA256 as V70_PROGRAM_SHA256
from hong2021_v70_preflight import load_program as load_v70_program
from hong2021_v70_train import (
    CHECKPOINT_SCHEMA,
    EMA_DECAY,
    REPORT_SCHEMA as TRAIN_REPORT_SCHEMA,
    STEPS as TRAIN_STEPS,
)


PROGRAM_SCHEMA = "hong2021-v70-train-only-joint-structure-mechanism-gate-program-v1"
PROGRAM_SHA256 = "13ce1abfabe92a38072077637ef4f724f1951b6da6bd8214f472fd930c91f728"
PROGRAM_FREEZE_COMMIT = "dfd654cd65443521b4e0a0ab837de0479ffe180b"
SCHEMA = "hong2021-v70-train-only-joint-structure-mechanism-decision-v1"
STREAMS = ("stream_A", "stream_B")
BANDS = ((3.0, 6.0), (6.0, 10.0))
DENSITY_SCALE = 4.5


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V70 train gate {label} hash differs")
    return _json(path)


def _path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_during_fixed_training_before_checkpoint_or_mechanism_holdout_access"
    ):
        raise ValueError("V70 train-gate program schema or status differs")
    parent = program["parent_evidence"]
    cache_record = _verified_json(
        _path(repo, parent["v70_cache_record"]),
        parent["v70_cache_record_sha256"],
        "cache result record",
    )
    if (
        cache_record.get("status") != parent["required_cache_status"]
        or cache_record.get("cache", {}).get("fixed_training_authorized")
        is not parent["required_fixed_training_authorized"]
        or cache_record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V70 train-gate parent authorization differs")
    if sha256_file(_path(repo, parent["v70_program"])) != V70_PROGRAM_SHA256:
        raise ValueError("V70 train-gate parent program differs")
    v70, v35, _ = load_v70_program(_path(repo, parent["v70_program"]), repo)
    frozen = program["frozen_inputs"]
    for key in (
        "v65_query_program",
        "v63_checkpoint",
        "conditioning_cache",
        "latent_cache",
    ):
        if sha256_file(_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V70 train-gate frozen input differs: {key}")
    v65 = _json(_path(repo, frozen["v65_query_program"]))
    query = program["immutable_mechanism_queries"]
    for domain in DOMAIN_ORDER:
        if list(map(int, query[domain])) != list(
            map(int, v65["immutable_train_queries"][domain])
        ):
            raise ValueError("V70 train-gate immutable query list differs")
    return program, v70, v35, cache_record


def sigma_schedule(
    steps: int = 40,
    sigma_minimum: float = 0.002,
    sigma_maximum: float = 40.0,
    rho: float = 7.0,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if steps < 2 or not 0.0 < sigma_minimum < sigma_maximum or rho <= 0.0:
        raise ValueError("V70 sampler schedule differs")
    position = torch.linspace(0.0, 1.0, steps, dtype=torch.float64, device=device)
    values = (
        sigma_maximum ** (1.0 / rho)
        + position
        * (sigma_minimum ** (1.0 / rho) - sigma_maximum ** (1.0 / rho))
    ) ** rho
    return torch.cat((values.float(), torch.zeros(1, device=device)))


@torch.no_grad()
def heun_sample(
    model: LatentSpatialUNet,
    condition: torch.Tensor,
    innovation: torch.Tensor,
    schedule: torch.Tensor,
) -> torch.Tensor:
    value = innovation.float() * schedule[0]
    for index in range(len(schedule) - 1):
        current = schedule[index]
        following = schedule[index + 1]
        sigma = current.expand(len(value))
        with torch.amp.autocast("cuda", dtype=torch.float16):
            denoised = edm_denoise(model, value, condition, sigma, 1.0)
        derivative = (value - denoised.float()) / current
        candidate = value + (following - current) * derivative
        if float(following) != 0.0:
            next_sigma = following.expand(len(value))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                next_denoised = edm_denoise(
                    model, candidate, condition, next_sigma, 1.0
                )
            next_derivative = (candidate - next_denoised.float()) / following
            value = value + (following - current) * 0.5 * (
                derivative + next_derivative
            )
        else:
            value = candidate
    return value


def project_residual_dc(value: np.ndarray) -> tuple[np.ndarray, float]:
    residual = np.asarray(value, dtype=np.float64)
    residual = residual - residual.mean(
        axis=(-3, -2, -1), keepdims=True, dtype=np.float64
    )
    maximum = float(
        np.max(np.abs(residual.mean(axis=(-3, -2, -1), dtype=np.float64)))
    )
    return residual.astype(np.float32), maximum


def fourier_energy_score(
    ensemble: np.ndarray, truth: np.ndarray, selected: np.ndarray
) -> float:
    values = np.asarray(ensemble)[:, selected].astype(np.complex128, copy=False)
    target = np.asarray(truth)[selected].astype(np.complex128, copy=False)
    if values.ndim != 2 or len(values) < 2 or target.ndim != 1 or target.size == 0:
        raise ValueError("V70 Fourier energy-score shape differs")
    normalization = math.sqrt(target.size)
    first = float(
        np.mean(
            np.sqrt(np.sum(np.abs(values - target[None]) ** 2, axis=1))
            / normalization
        )
    )
    pair_sum = 0.0
    pairs = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            pair_sum += float(
                np.sqrt(np.sum(np.abs(values[left] - values[right]) ** 2))
                / normalization
            )
            pairs += 1
    return first - pair_sum / (len(values) * (len(values) - 1))


def _band_masks(grid: int, voxel_mpc_h: float) -> dict[str, np.ndarray]:
    kxy = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
    kz = 2.0 * np.pi * np.fft.rfftfreq(grid, d=voxel_mpc_h)
    magnitude = np.sqrt(
        kxy[:, None, None] ** 2
        + kxy[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    return {
        band_key(low, high): (magnitude >= low) & (magnitude < high)
        for low, high in BANDS
    }


def _load_fit(
    program: dict[str, Any], repo: Path, commit: str, device: torch.device
) -> tuple[LatentSpatialUNet, dict[str, Any], str, str]:
    frozen = program["frozen_inputs"]
    checkpoint_path = _path(repo, frozen["expected_training_checkpoint"])
    report_path = _path(repo, frozen["expected_training_report"])
    if not checkpoint_path.exists() or not report_path.exists():
        raise RuntimeError("V70 train gate requires completed fixed training")
    report = _json(report_path)
    if (
        report.get("schema") != TRAIN_REPORT_SCHEMA
        or report.get("status") != "complete_fixed_30000_step_fit"
        or report.get("program_sha256") != V70_PROGRAM_SHA256
        or report.get("training_complete") is not True
        or report.get("train_only_mechanism_gate_run") is not False
        or report.get("validation_accessed") is not False
        or report.get("development_accessed") is not False
        or report.get("historical_EAGLE_accessed") is not False
        or report.get("independent_EAGLE_accessed") is not False
        or report.get("independent_gate_locked") is not True
        or canonical_digest(report) != report.get("decision_digest_sha256")
    ):
        raise ValueError("V70 fixed training report differs")
    checkpoint_sha = sha256_file(checkpoint_path)
    report_sha = sha256_file(report_path)
    if report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("V70 fixed checkpoint hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != V70_PROGRAM_SHA256
        or checkpoint.get("step") != TRAIN_STEPS
        or checkpoint.get("steps") != TRAIN_STEPS
        or checkpoint.get("ema_decay") != EMA_DECAY
        or checkpoint.get("parameters") != 8_771_649
        or checkpoint.get("validation_accessed") is not False
        or checkpoint.get("development_accessed") is not False
        or checkpoint.get("independent_EAGLE_accessed") is not False
        or checkpoint.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(checkpoint.get("initial_code_commit")), commit)
        or not _is_ancestor(repo, str(checkpoint.get("completion_code_commit")), commit)
    ):
        raise ValueError("V70 fixed checkpoint metadata differs")
    model = LatentSpatialUNet().to(device).eval()
    if parameter_count(model) != int(checkpoint["parameters"]):
        raise RuntimeError("V70 train-gate architecture differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, report, checkpoint_sha, report_sha


def _band_power_ratio(
    binner: SpectralBinner,
    generated_power: np.ndarray,
    truth_power: np.ndarray,
) -> dict[str, float]:
    ratio = np.divide(
        generated_power,
        truth_power,
        out=np.full_like(generated_power, np.nan),
        where=truth_power > 0.0,
    )
    result: dict[str, float] = {}
    for low, high in BANDS:
        selected = (binner.k >= low) & (binner.k < high) & np.isfinite(ratio)
        result[band_key(low, high)] = float(
            np.average(ratio[selected], weights=binner.count[selected])
        )
    return result


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v70, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V70 train gate requires clean Lageunha frozen ancestry")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V70 train gate requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    candidate, training_report, checkpoint_sha, report_sha = _load_fit(
        program, repo, commit, device
    )
    marginal, inherited_v35, prepared = _frozen_marginal(v70, repo, commit, device)
    if inherited_v35["development_domains"] != v35["development_domains"]:
        raise ValueError("V70 train-gate source definition differs")
    latent_cache = h5py.File(_path(repo, program["frozen_inputs"]["latent_cache"]), "r")
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    query = program["immutable_mechanism_queries"]
    stream_seed = {
        "stream_A": int(program["noise_streams"]["stream_A_seed"]),
        "stream_B": int(program["noise_streams"]["stream_B_seed"]),
    }
    members = int(program["noise_streams"]["members_per_query_and_stream"])
    inference_batch = int(program["noise_streams"]["inference_batch"])
    schedule = sigma_schedule(device=device)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    boundaries = program["one_point_measurement"]["q99_9_backbone_boundaries"]
    results: dict[str, dict[str, Any]] = {stream: {} for stream in STREAMS}
    maximum_inverse_error = 0.0
    maximum_dc = 0.0
    all_finite = True
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for stream in STREAMS:
            noise_generator = torch.Generator(device=device).manual_seed(
                stream_seed[stream]
            )
            for domain in DOMAIN_ORDER:
                data, cache = handles[domain]
                voxel = float(data.attrs["voxel_mpc_h"])
                binner = SpectralBinner(64, voxel)
                masks = _band_masks(64, voxel)
                truth_power = np.zeros(32, dtype=np.float64)
                candidate_power = np.zeros(32, dtype=np.float64)
                latent_truth_sum = latent_truth_square = 0.0
                latent_candidate_sum = latent_candidate_square = 0.0
                latent_truth_count = latent_candidate_count = 0
                moment_truth_sum = moment_candidate_sum = 0.0
                moment_truth_count = moment_candidate_count = 0
                energy_candidate = {key: [] for key in masks}
                energy_baseline = {key: [] for key in masks}
                for object_index in map(int, query[domain]):
                    condition, _, backbone_cube = condition_cube(
                        data, cache, prepared, domain, "train", object_index
                    )
                    truth = np.asarray(data["target"][object_index, 0], dtype=np.float32)
                    backbone = np.asarray(backbone_cube[0], dtype=np.float32)
                    truth_residual, truth_dc = project_residual_dc(truth - backbone)
                    maximum_dc = max(maximum_dc, truth_dc)
                    truth_latent = np.asarray(
                        latent_cache[f"{domain}/latent"][object_index, 0],
                        dtype=np.float32,
                    )
                    latent_truth_sum += float(truth_latent.sum(dtype=np.float64))
                    latent_truth_square += float(
                        np.square(truth_latent.astype(np.float64)).sum(dtype=np.float64)
                    )
                    latent_truth_count += truth_latent.size
                    truth_power += binner.power(truth[None])[0]
                    truth_fourier = binner.transform(truth_residual[None])[0]
                    condition_tensor = torch.from_numpy(condition[None]).to(device)
                    with torch.no_grad():
                        parameters = marginal(condition_tensor).float()
                    candidate_fields: list[np.ndarray] = []
                    candidate_residuals: list[np.ndarray] = []
                    baseline_residuals: list[np.ndarray] = []
                    selected = backbone + target_mean >= float(boundaries[domain])
                    truth_delta2 = np.square(
                        np.power(10.0, DENSITY_SCALE * truth[selected], dtype=np.float64)
                        - 1.0
                    )
                    moment_truth_sum += float(truth_delta2.sum(dtype=np.float64))
                    moment_truth_count += truth_delta2.size
                    for start in range(0, members, inference_batch):
                        count = min(inference_batch, members - start)
                        innovation = torch.randn(
                            (count, 1, 64, 64, 64),
                            device=device,
                            generator=noise_generator,
                        )
                        expanded_condition = condition_tensor.expand(count, -1, -1, -1, -1)
                        candidate_latent = heun_sample(
                            candidate, expanded_condition, innovation, schedule
                        )
                        latent_candidate_sum += float(
                            candidate_latent.double().sum().cpu()
                        )
                        latent_candidate_square += float(
                            candidate_latent.double().square().sum().cpu()
                        )
                        latent_candidate_count += candidate_latent.numel()
                        parameter_batch = parameters.expand(count, -1, -1, -1, -1)
                        uniforms = {
                            "candidate": standard_normal_cdf(candidate_latent),
                            "baseline": standard_normal_cdf(innovation),
                        }
                        for arm, uniform in uniforms.items():
                            standardized = bounded_mixture_inverse(
                                parameter_batch, uniform
                            )
                            error = float(
                                torch.max(
                                    torch.abs(
                                        bounded_mixture_cdf(parameter_batch, standardized)
                                        - uniform.clamp(1.0e-7, 1.0 - 1.0e-7)
                                    )
                                ).cpu()
                            )
                            maximum_inverse_error = max(maximum_inverse_error, error)
                            residual = (
                                standardized.cpu().numpy().astype(np.float64)
                                * target_std
                                + target_mean
                            )
                            residual, dc = project_residual_dc(residual)
                            maximum_dc = max(maximum_dc, dc)
                            if arm == "candidate":
                                fields = backbone[None] + residual[:, 0]
                                candidate_fields.extend(fields.astype(np.float32))
                                candidate_residuals.extend(residual[:, 0])
                                delta2 = np.square(
                                    np.power(
                                        10.0,
                                        DENSITY_SCALE * fields[:, selected],
                                        dtype=np.float64,
                                    )
                                    - 1.0
                                )
                                moment_candidate_sum += float(
                                    delta2.sum(dtype=np.float64)
                                )
                                moment_candidate_count += delta2.size
                            else:
                                baseline_residuals.extend(residual[:, 0])
                    candidate_array = np.stack(candidate_fields)
                    candidate_residual_array = np.stack(candidate_residuals)
                    baseline_residual_array = np.stack(baseline_residuals)
                    candidate_power += binner.power(candidate_array).sum(axis=0)
                    candidate_fourier = binner.transform(candidate_residual_array)
                    baseline_fourier = binner.transform(baseline_residual_array)
                    for key, mask in masks.items():
                        energy_candidate[key].append(
                            fourier_energy_score(candidate_fourier, truth_fourier, mask)
                        )
                        energy_baseline[key].append(
                            fourier_energy_score(baseline_fourier, truth_fourier, mask)
                        )
                    all_finite = bool(
                        all_finite
                        and np.isfinite(candidate_array).all()
                        and np.isfinite(candidate_fourier).all()
                        and np.isfinite(baseline_fourier).all()
                    )
                    print(
                        f"[v70-train-gate] {stream} {domain} object={object_index}",
                        flush=True,
                    )
                truth_power /= int(query["objects_per_domain"])
                candidate_power /= int(query["objects_per_domain"]) * members
                truth_latent_mean = latent_truth_sum / latent_truth_count
                truth_latent_std = max(
                    latent_truth_square / latent_truth_count - truth_latent_mean**2,
                    0.0,
                ) ** 0.5
                candidate_latent_mean = latent_candidate_sum / latent_candidate_count
                candidate_latent_std = max(
                    latent_candidate_square / latent_candidate_count
                    - candidate_latent_mean**2,
                    0.0,
                ) ** 0.5
                results[stream][domain] = {
                    "candidate_latent_mean": candidate_latent_mean,
                    "truth_latent_mean": truth_latent_mean,
                    "candidate_minus_truth_latent_mean": candidate_latent_mean
                    - truth_latent_mean,
                    "candidate_latent_standard_deviation": candidate_latent_std,
                    "truth_latent_standard_deviation": truth_latent_std,
                    "candidate_over_truth_latent_standard_deviation": candidate_latent_std
                    / truth_latent_std,
                    "q99_9_physical_moment_candidate_over_truth": (
                        moment_candidate_sum / moment_candidate_count
                    )
                    / (moment_truth_sum / moment_truth_count),
                    "total_power_candidate_over_truth": _band_power_ratio(
                        binner, candidate_power, truth_power
                    ),
                    "Fourier_energy_score_candidate": {
                        key: float(np.mean(value))
                        for key, value in energy_candidate.items()
                    },
                    "Fourier_energy_score_independent_voxel_baseline": {
                        key: float(np.mean(value))
                        for key, value in energy_baseline.items()
                    },
                }
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        latent_cache.close()
        prepared.close()
    one_point_pass = all(
        abs(results[stream][domain]["candidate_minus_truth_latent_mean"]) <= 0.1
        and 0.9
        <= results[stream][domain][
            "candidate_over_truth_latent_standard_deviation"
        ]
        <= 1.1
        and (2.0 / 3.0)
        <= results[stream][domain]["q99_9_physical_moment_candidate_over_truth"]
        <= 1.5
        for stream in STREAMS
        for domain in DOMAIN_ORDER
    )
    spectral_pass = all(
        0.8 <= value <= 1.25
        for stream in STREAMS
        for domain in DOMAIN_ORDER
        for value in results[stream][domain][
            "total_power_candidate_over_truth"
        ].values()
    )
    phase_pass = all(
        results[stream][domain]["Fourier_energy_score_candidate"][key]
        < results[stream][domain][
            "Fourier_energy_score_independent_voxel_baseline"
        ][key]
        for stream in STREAMS
        for domain in DOMAIN_ORDER
        for key in results[stream][domain]["Fourier_energy_score_candidate"]
    )
    tolerance = math.log(1.15)
    reproducibility_rows: dict[str, dict[str, Any]] = {}
    reproducibility_pass = True
    for domain in DOMAIN_ORDER:
        rows: dict[str, Any] = {}
        for key in results["stream_A"][domain]["Fourier_energy_score_candidate"]:
            energy_a = results["stream_A"][domain]["Fourier_energy_score_candidate"][key]
            energy_b = results["stream_B"][domain]["Fourier_energy_score_candidate"][key]
            power_a = results["stream_A"][domain]["total_power_candidate_over_truth"][key]
            power_b = results["stream_B"][domain]["total_power_candidate_over_truth"][key]
            energy_difference = (
                abs(math.log(energy_a / energy_b))
                if energy_a > 0.0 and energy_b > 0.0
                else float("inf")
            )
            power_difference = (
                abs(math.log(power_a / power_b))
                if power_a > 0.0 and power_b > 0.0
                else float("inf")
            )
            rows[key] = {
                "absolute_log_energy_score_ratio": energy_difference,
                "absolute_log_total_power_ratio": power_difference,
            }
            reproducibility_pass = bool(
                reproducibility_pass
                and energy_difference <= tolerance
                and power_difference <= tolerance
            )
        reproducibility_rows[domain] = rows
    peak = int(torch.cuda.max_memory_allocated(device))
    numerical_pass = bool(
        all_finite
        and maximum_inverse_error <= 2.0e-6
        and maximum_dc <= 1.0e-7
        and peak < int(program["resource_gate"]["peak_allocated_bytes_limit"])
    )
    selected = bool(
        one_point_pass
        and spectral_pass
        and phase_pass
        and reproducibility_pass
        and numerical_pass
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only_joint_structure_gate",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "training_checkpoint_sha256": checkpoint_sha,
        "training_report_sha256": report_sha,
        "training_initial_code_commit": training_report["initial_code_commit"],
        "streams": results,
        "stream_reproducibility": reproducibility_rows,
        "stream_reproducibility_tolerance_ln_1_15": tolerance,
        "one_point_pass": one_point_pass,
        "spectral_pass": spectral_pass,
        "phase_sensitive_energy_score_pass": phase_pass,
        "stream_reproducibility_pass": reproducibility_pass,
        "maximum_inverse_CDF_error": maximum_inverse_error,
        "maximum_absolute_residual_DC": maximum_dc,
        "all_values_finite": all_finite,
        "peak_allocated_bytes": peak,
        "numerical_pass": numerical_pass,
        "train_mechanism_pass": selected,
        "candidate_selected": selected,
        "classification": (
            "query_aligned_latent_spatial_score_learns_cross_domain_joint_structure"
            if selected
            else "query_aligned_latent_spatial_score_does_not_learn_cross_domain_joint_structure"
        ),
        "next": (
            "run_one_locked_development_sampling_and_the_unchanged_field_Q3_Q4_gates"
            if selected
            else "stop_before_development_without_posthoc_training_sampling_or_gate_tuning"
        ),
        "gradient_computed": False,
        "optimizer_constructed": False,
        "optimizer_step_performed": False,
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
        raise FileExistsError(f"V70 train gate refuses existing output: {args.out}")
    result = evaluate(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
