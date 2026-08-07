#!/usr/bin/env python
"""Replay selected V24 draws and audit the exact terminal Heun latent.

The frozen ensemble stores only the inverse-mapped physical fields.  This
read-only audit replays the original RNG stream on the Lageunha Ada GPU,
checks that the selected fields are byte-identical, and records the latent
trajectory that was not stored by the original sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import h5py
import numpy as np
import torch

from hong2021_residual_v6 import edm_denoise, karras_sigmas, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_v14_edm import V14ResidualDataset, decoder_upsampling_for_schema
from hong2021_v14_multiscale import fourier_band_masks, inverse_standardized_residual
from hong2021_v18_edm import _indices
from hong2021_v18_init import (
    PriorMatchedInitializer,
    prior_matched_spectral_std,
    sha256_file,
)
from hong2021_v21_conditional_affine import invert_profile_torch
from hong2021_v24_edm import _validate_checkpoint, load_frozen_program


SCHEMA = "hong2021-v24-terminal-sampler-trajectory-audit-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
TRACE_STEPS = (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40)
TARGET_DENSITY_SCALE = 4.5
V24_REGISTRY = Path("config/hong2021_v24_development_program.json")
TRAINING = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
    "tng100_simba_swift_v24_e12_base48_edm"
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_statistics(
    value: torch.Tensor, coordinates: list[tuple[int, int, int]]
) -> list[dict[str, Any]]:
    flat = value[:, 0].reshape(len(value), -1)
    mean = flat.mean(dim=1)
    centered = flat - mean[:, None]
    top = torch.topk(centered, k=16, dim=1, largest=True, sorted=True).values
    bottom = torch.topk(centered, k=16, dim=1, largest=False, sorted=True).values
    coordinate_tensor = torch.as_tensor(
        coordinates, dtype=torch.long, device=value.device
    )
    member = torch.arange(len(value), device=value.device)
    selected = value[
        member,
        0,
        coordinate_tensor[:, 0],
        coordinate_tensor[:, 1],
        coordinate_tensor[:, 2],
    ]
    selected_centered = selected - mean
    rows = []
    for index in range(len(value)):
        rows.append({
            "member": index,
            "mean": float(mean[index]),
            "rms": float(torch.sqrt(torch.mean(flat[index].square()))),
            "centered_standard_deviation": float(torch.std(centered[index], unbiased=False)),
            "centered_minimum": float(bottom[index, 0]),
            "centered_maximum": float(top[index, 0]),
            "centered_top3_mean": float(top[index, :3].mean()),
            "centered_top16_mean": float(top[index].mean()),
            "centered_bottom3_mean": float(bottom[index, :3].mean()),
            "centered_count_above_4": int(torch.count_nonzero(centered[index] > 4.0)),
            "centered_count_above_4p5": int(torch.count_nonzero(centered[index] > 4.5)),
            "centered_count_at_or_above_5": int(torch.count_nonzero(centered[index] >= 5.0)),
            "centered_count_at_or_below_minus5": int(torch.count_nonzero(centered[index] <= -5.0)),
            "value_at_final_physical_maximum_coordinate": float(selected[index]),
            "centered_value_at_final_physical_maximum_coordinate": float(
                selected_centered[index]
            ),
        })
    return rows


@torch.inference_mode()
def sample_edm_with_trace(
    model: torch.nn.Module,
    condition: torch.Tensor,
    generator: torch.Generator,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    sigma_data: float,
    *,
    coordinates: list[tuple[int, int, int]],
    init_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    trace_steps: tuple[int, ...] = TRACE_STEPS,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Exact ``sample_edm`` implementation with read-only statistics hooks."""
    if trace_steps[0] != 0 or trace_steps[-1] != steps:
        raise ValueError("trace steps must include initialization and terminal state")
    sigmas = karras_sigmas(steps, sigma_min, sigma_max, rho, condition.device)
    noise = torch.randn(
        (len(condition), 1) + tuple(condition.shape[-3:]),
        device=condition.device,
        generator=generator,
    )
    value = noise * sigmas[0] if init_transform is None else init_transform(noise)
    rows: list[dict[str, Any]] = []

    def record(phase: str, step: int, sigma: float, tensor: torch.Tensor) -> None:
        for row in _tensor_statistics(tensor, coordinates):
            rows.append({"phase": phase, "step": step, "sigma": sigma, **row})

    record("state", 0, float(sigmas[0]), value)
    selected_steps = set(trace_steps)
    for index in range(steps):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        sigma_batch = torch.full(
            (len(condition),), float(sigma), device=condition.device
        )
        denoised = edm_denoise(model, value, condition, sigma_batch, sigma_data)
        derivative = (value - denoised) / sigma
        euler = value + (sigma_next - sigma) * derivative
        next_denoised = None
        if sigma_next > 0:
            next_batch = torch.full(
                (len(condition),), float(sigma_next), device=condition.device
            )
            next_denoised = edm_denoise(
                model, euler, condition, next_batch, sigma_data
            )
            next_derivative = (euler - next_denoised) / sigma_next
            value = value + (sigma_next - sigma) * 0.5 * (
                derivative + next_derivative
            )
        else:
            value = euler
        completed_step = index + 1
        if completed_step in selected_steps:
            record("predictor_denoised", completed_step, float(sigma), denoised)
            if next_denoised is not None:
                record(
                    "corrector_denoised",
                    completed_step,
                    float(sigma_next),
                    next_denoised,
                )
            record("state", completed_step, float(sigma_next), value)
    return value, rows


def _correlation(first: list[float], second: list[float]) -> float | None:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _group_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"members": 0}
    keys = (
        "centered_maximum",
        "centered_top3_mean",
        "centered_top16_mean",
        "centered_value_at_final_physical_maximum_coordinate",
        "centered_count_at_or_above_5",
    )
    return {
        "members": len(rows),
        **{
            f"mean_{key}": float(np.mean([float(row[key]) for row in rows]))
            for key in keys
        },
    }


def _trajectory_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["phase"]), int(row["step"]))].append(row)
    output = []
    for (phase, step), selected in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        high = [row for row in selected if bool(row["high_density_failure"])]
        ordinary = [row for row in selected if not bool(row["high_density_failure"])]
        output.append({
            "phase": phase,
            "step": step,
            "sigma": float(selected[0]["sigma"]),
            "high_density_failure": _group_summary(high),
            "ordinary": _group_summary(ordinary),
        })
    return output


def _selected_objects(domain: Mapping[str, Any]) -> list[int]:
    threshold = float(domain["truth"]["physical_global_maximum"]) + 0.3
    failed = sorted({
        int(row["object_index"])
        for row in domain["all_generated_field_rows"]
        if float(row["physical_maximum"]) > threshold
    })
    selected = list(failed)
    for row in domain["top_generated_fields"]:
        index = int(row["object_index"])
        if index not in selected:
            selected.append(index)
        if len(selected) >= max(3, len(failed)):
            break
    return sorted(selected)


def _terminal_band_rms(value: torch.Tensor, voxel_mpc_h: float) -> list[list[float]]:
    masks = torch.as_tensor(
        fourier_band_masks(value.shape[-1], voxel_mpc_h),
        dtype=torch.bool,
        device=value.device,
    )
    spectrum = torch.fft.fftn(value[:, 0].double(), dim=(-3, -2, -1), norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()
    return [
        [float(torch.sqrt(power[member][mask].mean())) for mask in masks]
        for member in range(len(value))
    ]


@torch.inference_mode()
def audit_domain(
    domain: str,
    tail: Mapping[str, Any],
    *,
    repo: Path,
    checkpoint: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    v20: Mapping[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    experiment = v20["e8_gaussianized_marginal_retrain"]
    data = experiment["data"][domain]["validation_data"]
    cache = artifacts["caches"][f"{domain}_validation"]
    dataset = V14ResidualDataset(data["path"], cache["path"], False)
    indices = _indices(experiment["development_objects"][domain], repo)
    if indices != [int(value) for value in tail["source_indices"]]:
        raise ValueError(f"{domain} tail-audit source order differs from registry")
    sampler = experiment["sampler"]
    seed = int(sampler["sampling_seeds"][domain])
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    sigmas = karras_sigmas(
        int(sampler["steps"]), float(sampler["sigma_min"]),
        float(sampler["sigma_max"]), float(sampler["rho"]), device,
    )
    initialization = artifacts["initialization"]
    initializer = PriorMatchedInitializer(
        prior_matched_spectral_std(
            dataset.grid, dataset.voxel_mpc_h, float(sigmas[0]),
            initialization["source_balanced_band_mode_variance"], device=device,
        ),
        maximum_imaginary_ratio=1.0e-12,
    )
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    z_knots = torch.as_tensor(
        transform["z_knots"], dtype=torch.float32, device=device
    )
    residual_knots = torch.as_tensor(
        transform["residual_value_knots"], dtype=torch.float32, device=device
    )
    centers = torch.as_tensor(profile["centers"], dtype=torch.float64, device=device)
    mu = torch.as_tensor(profile["mu"], dtype=torch.float64, device=device)
    log_sigma = torch.as_tensor(
        profile["log_sigma"], dtype=torch.float64, device=device
    )
    selected_objects = _selected_objects(tail)
    field_rows = {
        (int(row["object_index"]), int(row["member"])): row
        for row in tail["all_generated_field_rows"]
    }
    ensemble_path = Path(tail["ensemble"])
    trajectory: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    exact_fields = 0
    maximum_difference = 0.0
    with h5py.File(ensemble_path, "r") as ensemble:
        for object_index, data_index in enumerate(indices):
            if object_index not in selected_objects:
                torch.randn(
                    (16, 1, dataset.grid, dataset.grid, dataset.grid),
                    device=device,
                    generator=generator,
                )
                continue
            condition, _, corrected_mean, _ = dataset[data_index]
            condition_batch = condition[None].to(device).expand(
                16, -1, -1, -1, -1
            )
            coordinates = [
                tuple(int(value) for value in field_rows[(object_index, member)][
                    "maximum_cell"
                ]["coordinate"])
                for member in range(16)
            ]
            final, trace_rows = sample_edm_with_trace(
                model, condition_batch, generator,
                int(sampler["steps"]), float(sampler["sigma_min"]),
                float(sampler["sigma_max"]), float(sampler["rho"]),
                float(checkpoint["sigma_data"]), coordinates=coordinates,
                init_transform=initializer,
            )
            centered = final - final.mean(dim=(-3, -2, -1), keepdim=True)
            u = inverse_gaussianize_torch(centered, z_knots, residual_knots)
            restored = invert_profile_torch(
                u,
                corrected_mean[None].to(device).expand(16, -1, -1, -1, -1),
                centers,
                mu,
                log_sigma,
            )
            location, scales = dataset.predicted_location_scales(data_index)
            restored_numpy = restored[:, 0].float().cpu().numpy()
            replay = np.stack([
                corrected_mean.numpy()[0]
                + inverse_standardized_residual(
                    value,
                    predicted_location=location,
                    predicted_scales=scales,
                    voxel_mpc_h=dataset.voxel_mpc_h,
                )
                for value in restored_numpy
            ]).astype(np.float32)
            stored = np.asarray(
                ensemble["sample"][object_index, :, 0], dtype=np.float32
            )
            difference = np.max(np.abs(replay - stored), axis=(1, 2, 3))
            maximum_difference = max(maximum_difference, float(np.max(difference)))
            exact_fields += int(sum(np.array_equal(replay[m], stored[m]) for m in range(16)))
            band_rms = _terminal_band_rms(centered, dataset.voxel_mpc_h)
            final_stats = {
                int(row["member"]): row
                for row in trace_rows
                if row["phase"] == "state" and int(row["step"]) == 40
            }
            threshold = float(tail["truth"]["physical_global_maximum"]) + 0.3
            for row in trace_rows:
                member = int(row["member"])
                source = field_rows[(object_index, member)]
                trajectory.append({
                    "object_index": object_index,
                    "source_index": data_index,
                    "high_density_failure": float(source["physical_maximum"]) > threshold,
                    "physical_maximum_log10rho": float(source["physical_maximum"]),
                    **row,
                })
            for member in range(16):
                source = field_rows[(object_index, member)]
                stat = final_stats[member]
                terminal.append({
                    "object_index": object_index,
                    "source_index": data_index,
                    "member": member,
                    "high_density_failure": float(source["physical_maximum"]) > threshold,
                    "physical_maximum_log10rho": float(source["physical_maximum"]),
                    "physical_maximum_coordinate": list(coordinates[member]),
                    "replay_byte_identical": bool(np.array_equal(replay[member], stored[member])),
                    "replay_maximum_absolute_y_difference": float(difference[member]),
                    "sampler_uncentered_mean": float(stat["mean"]),
                    "sampler_centered_minimum": float(stat["centered_minimum"]),
                    "sampler_centered_maximum": float(stat["centered_maximum"]),
                    "sampler_centered_top3_mean": float(stat["centered_top3_mean"]),
                    "sampler_centered_count_at_or_above_5": int(
                        stat["centered_count_at_or_above_5"]
                    ),
                    "sampler_centered_count_at_or_below_minus5": int(
                        stat["centered_count_at_or_below_minus5"]
                    ),
                    "sampler_centered_latent_at_physical_maximum_coordinate": float(
                        stat["centered_value_at_final_physical_maximum_coordinate"]
                    ),
                    "sampler_centered_band_rms": band_rms[member],
                    "predicted_band_scales": [float(value) for value in scales],
                })
            print(
                f"[terminal-audit] {domain} object={object_index} "
                f"source={data_index} exact={sum(np.array_equal(replay[m], stored[m]) for m in range(16))}/16",
                flush=True,
            )
    physical_maxima = [float(row["physical_maximum_log10rho"]) for row in terminal]
    high = [row for row in terminal if row["high_density_failure"]]
    ordinary = [row for row in terminal if not row["high_density_failure"]]
    correlation_fields = {
        "sampler_centered_maximum": [float(row["sampler_centered_maximum"]) for row in terminal],
        "sampler_centered_top3_mean": [float(row["sampler_centered_top3_mean"]) for row in terminal],
        "sampler_centered_latent_at_physical_maximum_coordinate": [
            float(row["sampler_centered_latent_at_physical_maximum_coordinate"])
            for row in terminal
        ],
        **{
            f"sampler_centered_band_rms_{band}": [
                float(row["sampler_centered_band_rms"][band]) for row in terminal
            ]
            for band in range(4)
        },
    }
    return {
        "ensemble": str(ensemble_path),
        "ensemble_sha256": sha256_file(ensemble_path),
        "sampling_seed": seed,
        "selected_object_indices": selected_objects,
        "selected_source_indices": [indices[index] for index in selected_objects],
        "selected_fields": len(terminal),
        "high_density_failure_fields": len(high),
        "replay_integrity": {
            "byte_identical_fields": exact_fields,
            "expected_fields": len(terminal),
            "maximum_absolute_y_difference": maximum_difference,
        },
        "terminal_correlations_with_physical_maximum": {
            name: _correlation(physical_maxima, values)
            for name, values in correlation_fields.items()
        },
        "terminal_groups": {
            "high_density_failure": {
                "members": len(high),
                "mean_centered_maximum": float(np.mean([
                    row["sampler_centered_maximum"] for row in high
                ])) if high else None,
                "mean_centered_latent_at_physical_maximum_coordinate": float(np.mean([
                    row["sampler_centered_latent_at_physical_maximum_coordinate"]
                    for row in high
                ])) if high else None,
                "members_with_upper_support_overshoot": int(sum(
                    row["sampler_centered_count_at_or_above_5"] > 0 for row in high
                )),
            },
            "ordinary": {
                "members": len(ordinary),
                "mean_centered_maximum": float(np.mean([
                    row["sampler_centered_maximum"] for row in ordinary
                ])),
                "mean_centered_latent_at_physical_maximum_coordinate": float(np.mean([
                    row["sampler_centered_latent_at_physical_maximum_coordinate"]
                    for row in ordinary
                ])),
                "members_with_upper_support_overshoot": int(sum(
                    row["sampler_centered_count_at_or_above_5"] > 0 for row in ordinary
                )),
            },
        },
        "trajectory_groups": _trajectory_summary(trajectory),
        "terminal_fields": terminal,
        "trajectory_rows": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tail-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.out.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("terminal sampler replay requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("terminal sampler replay requires the Lageunha Ada GPU")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("terminal sampler replay requires a clean committed worktree")
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise RuntimeError(f"refusing to overwrite terminal sampler audit: {output}")
    tail_path = args.tail_audit.resolve()
    tail = json.loads(tail_path.read_text())
    if (
        tail.get("classification", {}).get("mechanism")
        != "multi_field_tail_overdispersion_without_dominant_forward_support_saturation"
        or tail.get("Astrid_accessed") is not False
        or tail.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("terminal replay requires the sealed V24 tail diagnosis")

    registry_path = (repo / V24_REGISTRY).resolve()
    _, artifacts, v20, _ = load_frozen_program(registry_path, repo)
    checkpoint_path = TRAINING / "validation_checkpoints/step_030000.pt"
    checkpoint, checkpoint_sha = _validate_checkpoint(
        checkpoint_path, step=30000, artifacts=artifacts
    )
    features = checkpoint["observable_context_features"]
    device = torch.device("cuda")
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"],
        context_std=features["std"],
        decoder_upsampling=decoder_upsampling_for_schema(checkpoint["schema"]),
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    domains = {}
    for domain in DOMAIN_ORDER:
        domains[domain] = audit_domain(
            domain,
            tail["domains"][domain],
            repo=repo,
            checkpoint=checkpoint,
            artifacts=artifacts,
            v20=v20,
            model=model,
            device=device,
        )
    all_exact = all(
        row["replay_integrity"]["byte_identical_fields"]
        == row["replay_integrity"]["expected_fields"]
        for row in domains.values()
    )
    high_overshoot = sum(
        row["terminal_groups"]["high_density_failure"][
            "members_with_upper_support_overshoot"
        ]
        for row in domains.values()
    )
    high_fields = sum(row["high_density_failure_fields"] for row in domains.values())
    report = {
        "schema": SCHEMA,
        "code_commit": _git(repo, "rev-parse", "HEAD"),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "execution": {
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "precision": "frozen float32 model with inherited float64 initializer and inverse",
        },
        "inputs": {
            "tail_audit": str(tail_path),
            "tail_audit_sha256": sha256_file(tail_path),
            "tail_audit_digest_sha256": tail["audit_digest_sha256"],
            "registry": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
        },
        "trace_steps": list(TRACE_STEPS),
        "domains": domains,
        "classification": {
            "replay_byte_identical": all_exact,
            "high_density_failure_fields_replayed": high_fields,
            "high_density_failure_fields_with_terminal_upper_support_overshoot": high_overshoot,
            "terminal_upper_support_overshoot_is_necessary_for_failure": bool(
                high_fields and high_overshoot == high_fields
            ),
            "next": (
                "design_v25_from_exact_terminal_trajectory"
                if all_exact else "stop_and_resolve_replay_mismatch"
            ),
        },
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["audit_digest_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps({
        "out": str(output),
        "classification": report["classification"],
        "audit_digest_sha256": report["audit_digest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
