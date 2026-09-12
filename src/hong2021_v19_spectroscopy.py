#!/usr/bin/env python
"""Frozen CPU score spectroscopy for the V19-E7 mechanism test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_residual_v6 import edm_denoise, karras_sigmas
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import V14ResidualDataset, decoder_upsampling_for_schema
from hong2021_v14_multiscale import fourier_band_masks
from hong2021_v18_edm import _indices
from hong2021_v19_edm import _validate_checkpoint


SCHEMA = "hong2021-v19-fixed-score-spectroscopy-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")


@torch.inference_mode()
def score_spectroscopy(
    *, checkpoint_path: Path, registry: dict[str, Any], repo: Path
) -> dict[str, Any]:
    """Measure the preregistered band-0 denoiser shrinkage gaps on CPU."""
    torch.set_num_threads(32)
    experiment = registry["e7_band_anchored_noise_retrain"]
    checkpoint, checkpoint_sha = _validate_checkpoint(
        checkpoint_path.resolve(), step=10000, registry=registry
    )
    features = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"], context_std=features["std"],
        decoder_upsampling=decoder_upsampling_for_schema(checkpoint["schema"]),
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to("cpu")
    diagnostic = experiment["mechanism_diagnostics"]["Q1"]
    ladder_indices = [int(value) for value in diagnostic["karras_ladder_indices"]]
    sampler = experiment["sampler"]
    ladder = karras_sigmas(
        int(sampler["steps"]), float(sampler["sigma_min"]),
        float(sampler["sigma_max"]), float(sampler["rho"]), torch.device("cpu"),
    )[:-1]
    if max(ladder_indices) >= len(ladder):
        raise ValueError("V19 spectroscopy ladder index is outside the sampler")
    band0_mask = torch.from_numpy(fourier_band_masks(64, 0.3125)[0])
    train_power = registry["documented_program_facts"]["band0_per_mode_power"][
        "training_cache"
    ]
    domains: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = experiment["data"][domain]
        dataset = V14ResidualDataset(
            Path(row["validation_data"]), Path(row["validation_cache"]), False
        )
        selected = _indices(experiment["development_objects"][domain], repo)
        examples = [dataset[index] for index in selected]
        condition = torch.stack([value[0] for value in examples])
        residual = torch.stack([value[1] for value in examples])
        generator = torch.Generator(device="cpu").manual_seed(777000)
        epsilon = torch.randn(residual.shape, generator=generator)
        residual_spectrum = torch.fft.fftn(
            residual.double(), dim=(-3, -2, -1), norm="ortho"
        )
        rows = []
        variance = float(train_power[domain])
        for ladder_index in ladder_indices:
            sigma_value = float(ladder[ladder_index])
            noisy = residual + sigma_value * epsilon
            denoised = edm_denoise(
                model, noisy, condition,
                torch.full((len(residual),), sigma_value),
                float(checkpoint["sigma_data"]),
            )
            denoised_spectrum = torch.fft.fftn(
                denoised.double(), dim=(-3, -2, -1), norm="ortho"
            )
            noisy_spectrum = torch.fft.fftn(
                noisy.double(), dim=(-3, -2, -1), norm="ortho"
            )
            dk = denoised_spectrum[..., band0_mask]
            xk = noisy_spectrum[..., band0_mask]
            rk = residual_spectrum[..., band0_mask]
            gain = float((dk * xk.conj()).real.sum() / xk.abs().square().sum())
            ideal = variance / (variance + sigma_value**2)
            truth_gain = float((dk * rk.conj()).real.sum() / rk.abs().square().sum())
            orthogonal = dk - gain * xk
            rows.append({
                "ladder_index": ladder_index,
                "sigma": sigma_value,
                "g": gain,
                "g_train": ideal,
                "g_train_minus_g": ideal - gain,
                "truth_regression_gain": truth_gain,
                "denoised_power": float(dk.abs().square().mean()),
                "orthogonal_power_fraction": float(
                    orthogonal.abs().square().sum() / dk.abs().square().sum()
                ),
            })
        domains[domain] = {
            "samples": len(selected),
            "indices": selected,
            "training_band0_per_mode_power": variance,
            "measurements": rows,
        }
    threshold_by_index = {
        0: float(diagnostic["required_maximum_g_train_minus_g"]["index_0_sigma_40"]),
        4: float(diagnostic["required_maximum_g_train_minus_g"]["index_4_sigma_22_7"]),
        9: float(diagnostic["required_maximum_g_train_minus_g"]["index_9_sigma_10_4"]),
    }
    checks: dict[str, Any] = {}
    for domain, domain_row in domains.items():
        by_index = {row["ladder_index"]: row for row in domain_row["measurements"]}
        checks[domain] = {
            str(index): {
                "gap": by_index[index]["g_train_minus_g"],
                "maximum": maximum,
                "pass": by_index[index]["g_train_minus_g"] <= maximum,
            }
            for index, maximum in threshold_by_index.items()
        }
    passed = all(
        row["pass"] for domain in checks.values() for row in domain.values()
    )
    return {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "device": "cpu",
        "noise_seed": 777000,
        "noise_reinitialized_per_domain": True,
        "fft_dtype": "float64/complex128",
        "fft_norm": "ortho",
        "selection_role": "none",
        "domains": domains,
        "threshold_checks": checks,
        "Q1_pass": passed,
    }
