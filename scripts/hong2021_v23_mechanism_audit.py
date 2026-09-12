#!/usr/bin/env python
"""Audit V23 binning, loss gradients, and denoiser-to-sampler mismatch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_residual_v6 import edm_denoise
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import V14ResidualDataset, decoder_upsampling_for_schema
from hong2021_v18_init import sha256_file
from hong2021_v23_conditional_loss import (
    conditional_bin_indices,
    conditional_mean_statistics,
    conditional_mean_tail_edm_loss,
)


TRAINING = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
    "tng100_simba_swift_v23_e11_conditional_mean_edm"
)
V22_CHECKPOINT = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
    "tng100_simba_swift_v22_e10_long_horizon_edm/validation_checkpoints/"
    "step_030000.pt"
)
V23_CHECKPOINT = TRAINING / "validation_checkpoints/step_030000.pt"
V22_DECISION = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v22_e10_long_horizon/development_decision.json"
)
V23_DECISION = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v23_e11_conditional_mean/development_decision.json"
)
V23_FAILURE_AUDIT = V23_DECISION.parent / "automatic_failure_audit.json"
DOMAINS = ("TNG100", "SIMBA", "Swift-EAGLE")
GATE_DOMAINS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift-EAGLE": "swift_dev"}
SIGMAS = (0.002, 0.02, 0.2, 0.6, 2.0, 20.0, 40.0)


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _model(checkpoint: dict[str, Any], state_key: str, device: torch.device):
    features = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"], context_std=features["std"],
        decoder_upsampling=decoder_upsampling_for_schema(checkpoint["schema"]),
    )
    model.load_state_dict(checkpoint[state_key])
    return model.to(device)


def _batch(
    datasets: dict[str, V14ResidualDataset], indices: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    rows = []
    identity = 0.0
    occupancy: dict[str, list[list[int]]] = {}
    for domain in DOMAINS:
        domain_rows = [datasets[domain][index] for index in indices]
        rows.extend(domain_rows)
        occupancy[domain] = []
        for condition, _, mean, _ in domain_rows:
            identity = max(
                identity,
                float(torch.max(torch.abs(condition[2:3] - mean))),
            )
    condition = torch.stack([row[0] for row in rows]).to(device)
    residual = torch.stack([row[1] for row in rows]).to(device)
    truth = torch.stack([row[3] for row in rows]).to(device)
    return condition, residual, truth, {
        "maximum_absolute_condition_channel_2_minus_dataset_mean": identity,
        "occupancy": occupancy,
    }


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    pieces = []
    for parameter in model.parameters():
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float64))
        else:
            pieces.append(parameter.grad.detach().reshape(-1).cpu().to(torch.float64))
    return torch.cat(pieces)


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second))
    return float(torch.dot(first, second) / denominator) if denominator else math.nan


def gradient_audit(
    checkpoint: dict[str, Any], last: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    datasets = {
        domain: V14ResidualDataset(
            checkpoint["data"][domain]["train"],
            checkpoint["data"][domain]["train_cache"], False,
        )
        for domain in DOMAINS
    }
    edges = torch.tensor(
        checkpoint["denoising_loss"]["conditional_mean_edges"],
        dtype=torch.float64, device=device,
    )
    bin_weights = torch.tensor(
        checkpoint["tail_weight_fit"]["weights"],
        dtype=torch.float32, device=device,
    )
    model = _model(last, "model", device).train()
    batches = []
    for batch_index, indices in enumerate(((0, 1), (2, 3), (4, 5))):
        condition, residual, truth, structural = _batch(datasets, indices, device)
        assignment = conditional_bin_indices(condition[:, 2:3], edges)
        counts = []
        for sample in range(len(condition)):
            counts.append(
                torch.bincount(assignment[sample].reshape(-1), minlength=10)
                .cpu().tolist()
            )
        structural["per_sample_bin_counts"] = counts
        structural["minimum_bin_count"] = int(np.min(counts))
        structural["maximum_bin_count"] = int(np.max(counts))
        vectors = {}
        losses = None
        maximum_loss_reproduction_difference = 0.0
        for name, component in (("unweighted", 1), ("tail_weighted", 2), ("conditional_mean", 3)):
            model.zero_grad(set_to_none=True)
            generator = torch.Generator(device=device).manual_seed(230100 + batch_index)
            values = conditional_mean_tail_edm_loss(
                model, residual, condition, truth, bin_weights, edges, generator,
                float(checkpoint["sigma_data"]), float(checkpoint["edm_p_mean"]),
                float(checkpoint["edm_p_std"]), 1.0, 64,
            )
            current = [float(value.detach()) for value in values]
            if losses is None:
                losses = current
            else:
                difference = float(
                    np.max(np.abs(np.asarray(losses) - np.asarray(current)))
                )
                maximum_loss_reproduction_difference = max(
                    maximum_loss_reproduction_difference, difference
                )
                if not np.allclose(losses, current, rtol=1.0e-6, atol=1.0e-8):
                    raise RuntimeError(
                        "gradient audit failed to reproduce fixed draws within tolerance"
                    )
            values[component].backward()
            vectors[name] = _gradient_vector(model)
        assert losses is not None
        base = 0.5 * vectors["unweighted"] + 0.5 * vectors["tail_weighted"]
        conditional = vectors["conditional_mean"]
        combined = base + conditional
        norms = {
            "unweighted": float(torch.linalg.vector_norm(vectors["unweighted"])),
            "tail_weighted": float(torch.linalg.vector_norm(vectors["tail_weighted"])),
            "base_half_plus_half": float(torch.linalg.vector_norm(base)),
            "conditional_mean": float(torch.linalg.vector_norm(conditional)),
            "combined": float(torch.linalg.vector_norm(combined)),
        }
        batches.append(
            {
                "indices_per_domain": list(indices),
                "noise_seed": 230100 + batch_index,
                "maximum_repeated_forward_loss_difference": (
                    maximum_loss_reproduction_difference
                ),
                "losses": {
                    key: value
                    for key, value in zip(
                        ("combined", "unweighted", "tail_weighted", "conditional_mean"),
                        losses, strict=True,
                    )
                },
                "gradient_norms": norms,
                "conditional_to_base_gradient_norm_ratio": (
                    norms["conditional_mean"] / norms["base_half_plus_half"]
                ),
                "conditional_vs_base_gradient_cosine": _cosine(conditional, base),
                "conditional_vs_unweighted_gradient_cosine": _cosine(
                    conditional, vectors["unweighted"]
                ),
                "conditional_vs_tail_gradient_cosine": _cosine(
                    conditional, vectors["tail_weighted"]
                ),
                "combined_exceeds_clip_one": norms["combined"] > 1.0,
                "all_gradient_entries_finite": all(
                    bool(torch.isfinite(value).all()) for value in vectors.values()
                ),
                "structure": structural,
            }
        )
    del model
    torch.cuda.empty_cache()
    return {
        "precision": "float32_no_autocast",
        "state": "final_non_ema_training_model",
        "source_balance": "two objects from each of TNG100, SIMBA, Swift-EAGLE",
        "batches": batches,
        "mean_conditional_to_base_gradient_norm_ratio": float(np.mean([
            row["conditional_to_base_gradient_norm_ratio"] for row in batches
        ])),
        "mean_conditional_vs_base_gradient_cosine": float(np.mean([
            row["conditional_vs_base_gradient_cosine"] for row in batches
        ])),
    }


def _development_indices(decision: dict[str, Any], domain: str) -> list[int]:
    ensemble = Path(
        decision["candidates"][-1]["domains"][GATE_DOMAINS[domain]]["ensemble"]
    )
    with h5py.File(ensemble, "r") as handle:
        return [int(value) for value in handle["source_index"][:]]


@torch.inference_mode()
def denoising_audit(
    v22: dict[str, Any], v23: dict[str, Any], decision: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    models = {
        "v22": _model(v22, "ema_model", device).eval(),
        "v23": _model(v23, "ema_model", device).eval(),
    }
    edges = torch.tensor(
        v23["denoising_loss"]["conditional_mean_edges"],
        dtype=torch.float64, device=device,
    )
    output: dict[str, Any] = {}
    for domain_index, domain in enumerate(DOMAINS):
        dataset = V14ResidualDataset(
            v23["data"][domain]["validation"],
            v23["data"][domain]["validation_cache"], False,
        )
        indices = _development_indices(decision, domain)
        output[domain] = {"source_indices": indices, "models": {}}
        for model_index, (label, model) in enumerate(models.items()):
            sigma_rows = {}
            for sigma_index, sigma_value in enumerate(SIGMAS):
                bin_sums = torch.zeros(10, dtype=torch.float64, device=device)
                bin_counts = torch.zeros(10, dtype=torch.int64, device=device)
                per_sample_maxima = []
                squared_sum = 0.0
                voxels = 0
                generator = torch.Generator(device=device).manual_seed(
                    231000 + domain_index * 100 + sigma_index
                )
                for start in range(0, len(indices), 2):
                    rows = [dataset[index] for index in indices[start : start + 2]]
                    condition = torch.stack([row[0] for row in rows]).to(device)
                    residual = torch.stack([row[1] for row in rows]).to(device)
                    sigma = torch.full(
                        (len(rows),), sigma_value, dtype=residual.dtype, device=device
                    )
                    noise = torch.randn(
                        residual.shape, device=device, generator=generator
                    )
                    denoised = edm_denoise(
                        model,
                        residual + sigma[:, None, None, None, None] * noise,
                        condition, sigma, float(v23["sigma_data"]),
                    )
                    means, valid = conditional_mean_statistics(
                        denoised, residual, condition[:, 2:3], edges, 64
                    )
                    per_sample_maxima.extend(
                        means.abs().masked_fill(~valid, 0.0).max(dim=1).values
                        .cpu().tolist()
                    )
                    assignment = conditional_bin_indices(condition[:, 2:3], edges)
                    flat_assignment = assignment.reshape(-1)
                    error = (denoised - residual).reshape(-1).to(torch.float64)
                    bin_sums.scatter_add_(0, flat_assignment, error)
                    bin_counts.scatter_add_(
                        0, flat_assignment, torch.ones_like(flat_assignment)
                    )
                    squared_sum += float(torch.square(error).sum())
                    voxels += error.numel()
                aggregate = bin_sums / bin_counts.to(torch.float64)
                sigma_rows[str(sigma_value)] = {
                    "maximum_absolute_aggregate_bin_mean_error": float(
                        aggregate.abs().max()
                    ),
                    "aggregate_bin_mean_error": aggregate.cpu().tolist(),
                    "mean_per_sample_maximum_absolute_bin_mean_error": float(
                        np.mean(per_sample_maxima)
                    ),
                    "maximum_per_sample_maximum_absolute_bin_mean_error": float(
                        np.max(per_sample_maxima)
                    ),
                    "denoising_mse": squared_sum / voxels,
                    "noise_seed": 231000 + domain_index * 100 + sigma_index,
                }
            output[domain]["models"][label] = sigma_rows
    del models
    torch.cuda.empty_cache()
    return {
        "precision": "float32_matching_sampler",
        "state": "EMA",
        "sigmas": list(SIGMAS),
        "domains": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V23 mechanism audit requires an available CUDA device")
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite mechanism audit: {args.out}")
    v22 = torch.load(V22_CHECKPOINT, map_location="cpu", weights_only=False)
    v23 = torch.load(V23_CHECKPOINT, map_location="cpu", weights_only=False)
    last = torch.load(TRAINING / "last.pt", map_location="cpu", weights_only=False)
    decision22 = json.loads(V22_DECISION.read_text())
    decision23 = json.loads(V23_DECISION.read_text())
    failure = json.loads(V23_FAILURE_AUDIT.read_text())
    if decision23.get("development_pass") is not False:
        raise ValueError("V23 mechanism audit requires the failed frozen decision")
    if failure.get("Astrid_accessed") is not False or failure.get(
        "historical_EAGLE_accessed"
    ) is not False:
        raise ValueError("V23 failure record violated the data firewall")
    history = json.loads((TRAINING / "history.json").read_text())
    nonfinite_intervals = [
        row["step"] for row in history
        if not math.isfinite(row["gradient_diagnostic"]["mean_norm_before_fixed_clip"])
    ]
    report = {
        "schema": "hong2021-v23-mechanism-audit-v1",
        "code_commit": _git_commit(repo),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "v22_checkpoint": str(V22_CHECKPOINT),
            "v22_checkpoint_sha256": sha256_file(V22_CHECKPOINT),
            "v23_checkpoint": str(V23_CHECKPOINT),
            "v23_checkpoint_sha256": sha256_file(V23_CHECKPOINT),
            "v22_decision": str(V22_DECISION),
            "v22_decision_sha256": sha256_file(V22_DECISION),
            "v23_decision": str(V23_DECISION),
            "v23_decision_sha256": sha256_file(V23_DECISION),
            "v23_failure_audit": str(V23_FAILURE_AUDIT),
            "v23_failure_audit_sha256": sha256_file(V23_FAILURE_AUDIT),
        },
        "training_history": {
            "final_balanced_validation": history[-1]["balanced_validation"],
            "final_balanced_conditional_validation": history[-1][
                "balanced_conditional_validation"
            ],
            "final_conditional_mean_loss": history[-1]["train"][
                "conditional_mean_loss"
            ],
            "nonfinite_gradient_diagnostic_intervals": nonfinite_intervals,
            "first_interval_clip_activation_fraction": history[0][
                "gradient_diagnostic"
            ]["fixed_clip_activation_fraction"],
        },
        "final_sampler_Q6": {
            "v22": {
                domain: row["conditional_Q6_latent"][
                    "maximum_absolute_generated_minus_truth_mean"
                ]
                for domain, row in decision22["candidates"][-1]["domains"].items()
            },
            "v23": {
                domain: row["conditional_Q6_latent"][
                    "maximum_absolute_generated_minus_truth_mean"
                ]
                for domain, row in decision23["candidates"][-1]["domains"].items()
            },
        },
        "gradient_audit": gradient_audit(v23, last, device),
        "fixed_sigma_denoising_audit": denoising_audit(
            v22, v23, decision23, device
        ),
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["audit_digest_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
