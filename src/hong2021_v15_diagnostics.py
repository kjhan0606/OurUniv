#!/usr/bin/env python
"""Read-only V15-E0 diagnostics for the frozen, failed V14 candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from hong2021_residual_v6 import edm_denoise, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import SCHEMA as V14_SCHEMA, V14ResidualDataset
from hong2021_v14_multiscale import BAND_EDGES_H_MPC, fourier_band_masks


SCHEMA = "hong2021-v15-e0-frozen-v14-diagnostic-v1"
DOMAINS = ("TNG100", "SIMBA", "Swift")
POWER_KEYS = (
    "0.3-1_h_mpc",
    "1-3_h_mpc",
    "3-6_h_mpc",
    "6-10.0531_h_mpc",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("diagnostic ratio requires finite values and positive denominator")
    return float(numerator / denominator)


def field_trend(metrics_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path_like in metrics_paths:
        path = Path(path_like)
        payload = json.loads(path.read_text())
        candidate = payload["candidates"]["edm"]
        power = candidate["fourier_log_density"]
        truth_environment = candidate["environment"]["truth"]
        generated_environment = candidate["environment"]["generated"]
        checkpoint_step = int(candidate.get("checkpoint_step", 0))
        if checkpoint_step == 0:
            import h5py

            with h5py.File(candidate["path"], "r") as ensemble:
                checkpoint_step = int(ensemble.attrs["checkpoint_step"])
        environment_ratios = {}
        for name in (
            "local_peak_count_rho_gt_10",
            "local_peak_count_rho_gt_100",
            "local_void_count_rho_lt_0.1",
        ):
            environment_ratios[name] = _safe_ratio(
                float(generated_environment[name]["mean"]),
                float(truth_environment[name]["mean"]),
            )
        rows.append(
            {
                "step": checkpoint_step,
                "metrics": str(path.resolve()),
                "metrics_sha256": sha256_file(path),
                "generated_total_power_over_truth": {
                    name: float(power["generated_total_power_over_truth"][name])
                    for name in POWER_KEYS
                },
                "generated_residual_power_over_truth_residual": {
                    name: float(
                        power["generated_residual_power_over_truth_residual"][name]
                    )
                    for name in POWER_KEYS
                },
                "ensemble_mean_cross_correlation_with_truth": {
                    name: float(
                        power["ensemble_mean_cross_correlation_with_truth"][name]
                    )
                    for name in POWER_KEYS
                },
                "generated_over_truth_rms": float(
                    candidate["residual_calibration"]["generated_over_truth_rms"]
                ),
                "environment_generated_over_truth": environment_ratios,
            }
        )
    return sorted(rows, key=lambda row: row["step"])


def _band_power_sums(value: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fftn(value.to(torch.float32), dim=(-3, -2, -1))
    power = spectrum.real.square() + spectrum.imag.square()
    return torch.stack(
        [(power * mask).sum(dim=(-3, -2, -1)) for mask in masks], dim=-1
    )


@torch.inference_mode()
def fixed_sigma_denoising(
    model: torch.nn.Module,
    dataset: V14ResidualDataset,
    indices: list[int],
    sigmas: list[float],
    seed: int,
    sigma_data: float,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("diagnostic indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError("diagnostic index outside dataset")
    if not sigmas or any(not np.isfinite(value) or value <= 0 for value in sigmas):
        raise ValueError("fixed sigmas must be finite and positive")
    masks = torch.from_numpy(
        fourier_band_masks(dataset.grid, dataset.voxel_mpc_h)
    ).to(device=device, dtype=torch.float32)
    rows = []
    model.eval()
    for sigma_index, sigma_value in enumerate(sigmas):
        generator = torch.Generator(device=device).manual_seed(seed + sigma_index)
        error_sum = 0.0
        residual_sum = 0.0
        noisy_error_sum = 0.0
        voxels = 0
        error_band = torch.zeros(len(BAND_EDGES_H_MPC) - 1, dtype=torch.float64)
        residual_band = torch.zeros_like(error_band)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            examples = [dataset[index] for index in selected]
            condition = torch.stack([value[0] for value in examples]).to(device)
            residual = torch.stack([value[1] for value in examples]).to(device)
            sigma = torch.full(
                (len(selected),), sigma_value, device=device, dtype=residual.dtype
            )
            noise = torch.randn(
                residual.shape,
                device=device,
                dtype=residual.dtype,
                generator=generator,
            )
            noisy = residual + sigma[:, None, None, None, None] * noise
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                denoised = edm_denoise(model, noisy, condition, sigma, sigma_data)
            error = denoised.float() - residual.float()
            error_sum += float(error.square().sum())
            residual_sum += float(residual.float().square().sum())
            noisy_error_sum += float((noisy.float() - residual.float()).square().sum())
            voxels += error.numel()
            error_band += _band_power_sums(error[:, 0], masks).sum(0).cpu().double()
            residual_band += _band_power_sums(residual[:, 0], masks).sum(0).cpu().double()
        mse = error_sum / voxels
        rows.append(
            {
                "sigma": sigma_value,
                "samples": len(indices),
                "denoising_mse": mse,
                "zero_estimator_mse": residual_sum / voxels,
                "noisy_input_mse": noisy_error_sum / voxels,
                "mse_over_zero_estimator": _safe_ratio(error_sum, residual_sum),
                "mse_over_noisy_input": _safe_ratio(error_sum, noisy_error_sum),
                "band_error_over_residual_power": [
                    _safe_ratio(float(a), float(b))
                    for a, b in zip(error_band, residual_band, strict=True)
                ],
            }
        )
    return rows


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V14_SCHEMA:
        raise ValueError(f"not a frozen V14 checkpoint: {checkpoint_path}")
    feature_fit = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=feature_fit["mean"],
        context_std=feature_fit["std"],
    )
    model.load_state_dict(checkpoint["ema_model"])
    return model.eval().to(device), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--v14-decision", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    for prefix in ("tng", "simba", "swift"):
        parser.add_argument(f"--{prefix}-data", required=True)
        parser.add_argument(f"--{prefix}-cache", required=True)
        parser.add_argument(f"--{prefix}-indices", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps != sorted(set(args.steps)):
        raise ValueError("checkpoint steps must be unique and ordered")
    if args.batch <= 0:
        raise ValueError("batch must be positive")
    registry = json.loads(args.registry.read_text())
    if registry.get("schema") != "hong2021-v15-predeclared-development-program-v1":
        raise ValueError("invalid V15 registry")
    e0 = registry["e0_diagnostic"]
    if args.steps != e0["v14_checkpoint_steps"]:
        raise ValueError("checkpoint steps differ from the predeclared E0 program")
    if not np.array_equal(np.asarray(args.sigmas), np.asarray(e0["fixed_sigmas"])):
        raise ValueError("fixed sigmas differ from the predeclared E0 program")
    if args.seeds != [e0["noise_seeds"][name] for name in DOMAINS]:
        raise ValueError("noise seeds differ from the predeclared E0 program")
    decision_hash = sha256_file(args.v14_decision)
    if decision_hash != registry["v14_failure"]["decision_sha256"]:
        raise ValueError("V14 decision differs from the predeclared digest")
    decision = json.loads(args.v14_decision.read_text())
    if decision.get("development_pass") or decision.get("Astrid_used"):
        raise ValueError("E0 requires a failed V14 development decision with Astrid unused")
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite E0 diagnostic: {args.out}")
    seed_everything(min(args.seeds))
    device = torch.device(args.device)
    domain_arguments = {
        "TNG100": (args.tng_data, args.tng_cache, args.tng_indices),
        "SIMBA": (args.simba_data, args.simba_cache, args.simba_indices),
        "Swift": (args.swift_data, args.swift_cache, args.swift_indices),
    }
    datasets = {
        name: V14ResidualDataset(data, cache, False)
        for name, (data, cache, _) in domain_arguments.items()
    }
    indices = {
        name: [int(value) for value in encoded.split(",")]
        for name, (_, _, encoded) in domain_arguments.items()
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "v14_decision": str(args.v14_decision.resolve()),
        "v14_decision_sha256": decision_hash,
        "selection_role": "none",
        "independent_data_used": False,
        "astrid_used": False,
        "historical_eagle_confirmation_used": False,
        "fixed_sigmas": args.sigmas,
        "fourier_edges_h_mpc": list(BAND_EDGES_H_MPC[:-1]) + ["inf"],
        "domains": {},
    }
    for domain_index, name in enumerate(DOMAINS):
        metrics_paths = [
            args.evaluation
            / "development_candidates"
            / f"step_{step:06d}"
            / {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_eagle_dev"}[name]
            / "ensemble_evaluation"
            / "metrics.json"
            for step in args.steps
        ]
        domain_report = {
            "data": str(Path(domain_arguments[name][0]).resolve()),
            "cache": str(Path(domain_arguments[name][1]).resolve()),
            "indices": indices[name],
            "field_trend": field_trend(metrics_paths),
            "fixed_sigma": [],
        }
        for step in args.steps:
            checkpoint_path = (
                args.training / "validation_checkpoints" / f"step_{step:06d}.pt"
            )
            model, checkpoint = load_model(checkpoint_path, device)
            domain_report["fixed_sigma"].append(
                {
                    "step": step,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "sigma_data": float(checkpoint["sigma_data"]),
                    "edm_p_mean": float(checkpoint["edm_p_mean"]),
                    "edm_p_std": float(checkpoint["edm_p_std"]),
                    "measurements": fixed_sigma_denoising(
                        model,
                        datasets[name],
                        indices[name],
                        args.sigmas,
                        args.seeds[domain_index],
                        float(checkpoint["sigma_data"]),
                        device,
                        args.batch,
                    ),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["domains"][name] = domain_report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps({"out": str(args.out), "schema": SCHEMA}, indent=2))


if __name__ == "__main__":
    main()
