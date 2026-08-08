#!/usr/bin/env python
"""Audit the failed V26 flow without tuning or opening independent data.

The frozen V26 registry promised trained-model base-z moments and numerical
roundtrip diagnostics, but the development gate recorded neither.  This
read-only audit fills that evidence gap on the already-selected TNG100, SIMBA,
and Swift development objects.  It also replays one stored physical member per
object and summarizes the scale-resolved optimization history.  Astrid and
historical EAGLE are deliberately absent from this program.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_residual_v6 import seed_everything
from hong2021_v14_edm import V14ResidualDataset
from hong2021_v14_multiscale import inverse_standardized_residual
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v21_conditional_affine import invert_profile_torch
from hong2021_v26 import (
    CACHE_KEYS,
    CANDIDATE_STEPS,
    DETAIL_DIMENSIONS_COARSE_TO_FINE,
    DOMAIN_KEYS,
    MODEL_SCHEMA,
    REGISTRY_SHA256,
    _validate_checkpoint,
    build_model,
    load_frozen_program,
)
from hong2021_v26_haar import haar_pyramid, haar_synthesis


SCHEMA = "hong2021-v26-trained-flow-mechanism-audit-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
TRAINING = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
    "tng100_simba_swift_v26_e14_conditional_haar_flow"
)
DECISION = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v26_e14_conditional_haar_flow/development_decision.json"
)
FAILURE_AUDIT = DECISION.parent / "automatic_failure_audit.json"
ABSOLUTE_THRESHOLDS = (3.0, 4.0, 5.0, 6.0)


@dataclass
class TensorMoments:
    """Streaming raw moments and fixed-tail occupancy for finite tensors."""

    count: int = 0
    sum1: float = 0.0
    sum2: float = 0.0
    sum3: float = 0.0
    sum4: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    absolute_maximum: float = 0.0
    nonfinite: int = 0
    threshold_counts: dict[float, int] | None = None

    def __post_init__(self) -> None:
        if self.threshold_counts is None:
            self.threshold_counts = {value: 0 for value in ABSOLUTE_THRESHOLDS}

    def update(self, value: torch.Tensor) -> None:
        flat = value.detach().reshape(-1).double()
        finite = torch.isfinite(flat)
        self.nonfinite += int((~finite).sum())
        flat = flat[finite]
        if not flat.numel():
            return
        self.count += int(flat.numel())
        self.sum1 += float(flat.sum())
        square = flat.square()
        self.sum2 += float(square.sum())
        self.sum3 += float((square * flat).sum())
        self.sum4 += float(square.square().sum())
        low = float(flat.min())
        high = float(flat.max())
        self.minimum = min(self.minimum, low)
        self.maximum = max(self.maximum, high)
        self.absolute_maximum = max(self.absolute_maximum, abs(low), abs(high))
        assert self.threshold_counts is not None
        absolute = flat.abs()
        for threshold in ABSOLUTE_THRESHOLDS:
            self.threshold_counts[threshold] += int((absolute >= threshold).sum())

    def report(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "nonfinite": self.nonfinite}
        mean = self.sum1 / self.count
        raw2 = self.sum2 / self.count
        raw3 = self.sum3 / self.count
        raw4 = self.sum4 / self.count
        variance = max(raw2 - mean * mean, 0.0)
        standard_deviation = math.sqrt(variance)
        central3 = raw3 - 3.0 * mean * raw2 + 2.0 * mean**3
        central4 = (
            raw4 - 4.0 * mean * raw3 + 6.0 * mean * mean * raw2 - 3.0 * mean**4
        )
        skewness = (
            central3 / standard_deviation**3 if standard_deviation > 0.0 else None
        )
        excess_kurtosis = (
            central4 / standard_deviation**4 - 3.0
            if standard_deviation > 0.0
            else None
        )
        assert self.threshold_counts is not None
        return {
            "count": self.count,
            "nonfinite": self.nonfinite,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "absolute_maximum": self.absolute_maximum,
            "absolute_tail_counts": {
                str(threshold): self.threshold_counts[threshold]
                for threshold in ABSOLUTE_THRESHOLDS
            },
            "absolute_tail_fractions": {
                str(threshold): self.threshold_counts[threshold] / self.count
                for threshold in ABSOLUTE_THRESHOLDS
            },
        }


def analyze_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the frozen global and per-scale optimization evidence."""
    rows = {int(row["step"]): row for row in history}
    required = (500, 10_000, 20_000, 25_000, 30_000)
    if any(step not in rows for step in required):
        raise ValueError("V26 history lacks a required diagnostic step")
    dimensions = np.asarray(DETAIL_DIMENSIONS_COARSE_TO_FINE, dtype=np.float64)
    shares = dimensions / dimensions.sum()
    domains = tuple(rows[30_000]["fixed_validation"])
    scale_rows: dict[str, list[dict[str, Any]]] = {}
    for domain in domains:
        output = []
        for scale, dimension in enumerate(dimensions.astype(int)):
            values = {
                str(step): float(
                    rows[step]["fixed_validation"][domain][
                        "scale_nll_coarse_to_fine"
                    ][scale]
                )
                for step in required
            }
            output.append(
                {
                    "coarse_to_fine_index": scale,
                    "dimensions": int(dimension),
                    "objective_dimension_fraction": float(shares[scale]),
                    "validation_nll": values,
                    "change_10000_to_30000": values["30000"] - values["10000"],
                    "worsened_10000_to_30000": values["30000"] > values["10000"],
                    "final_train_nll": float(
                        rows[30_000]["train_scale_nll_coarse_to_fine"][scale]
                    ),
                    "final_validation_minus_train": values["30000"]
                    - float(rows[30_000]["train_scale_nll_coarse_to_fine"][scale]),
                }
            )
        scale_rows[domain] = output
    nonfinite_gradient_steps = [
        int(row["step"])
        for row in history
        if not math.isfinite(
            float(row["gradient_diagnostic"]["mean_norm_before_fixed_clip"])
        )
    ]
    start = float(rows[25_000]["balanced_validation_nll"])
    final = float(rows[30_000]["balanced_validation_nll"])
    return {
        "dimensions_coarse_to_fine": dimensions.astype(int).tolist(),
        "objective_dimension_fractions_coarse_to_fine": shares.tolist(),
        "finest_scale_objective_fraction": float(shares[-1]),
        "balanced_validation_nll": {
            str(step): float(rows[step]["balanced_validation_nll"])
            for step in required
        },
        "relative_improvement_25000_to_30000": (start - final) / abs(start),
        "scale_resolved": scale_rows,
        "nonfinite_mean_gradient_intervals": nonfinite_gradient_steps,
    }


@torch.inference_mode()
def encode_latent(
    model: torch.nn.Module, latent: torch.Tensor, condition: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Return standardized Haar details, base-z fields, and forward logdets."""
    lowpass, details = haar_pyramid(latent.double(), levels=model.levels)
    global_context = model.standardized_global_context(condition)
    standardized: list[torch.Tensor] = []
    bases: list[torch.Tensor] = []
    logdets: list[torch.Tensor] = []
    for coarse_index, (flow, raw_detail) in enumerate(
        zip(model.flows, reversed(details), strict=True)
    ):
        level = model.levels - 1 - coarse_index
        mean = model.detail_mean[level].double()[None, :, None, None, None]
        std = model.detail_std[level].double()[None, :, None, None, None]
        detail = ((raw_detail - mean) / std).to(dtype=latent.dtype)
        context = model.scale_context(condition, lowpass, global_context)
        base, logdet = flow(detail, context)
        standardized.append(detail)
        bases.append(base)
        logdets.append(logdet)
        lowpass = haar_synthesis(lowpass, raw_detail)
    return standardized, bases, logdets


@torch.inference_mode()
def decode_bases(
    model: torch.nn.Module, bases: list[torch.Tensor], condition: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Invert base fields while rebuilding the registered coarse context."""
    if len(bases) != model.levels:
        raise ValueError("one base-z field is required per V26 scale")
    batch = len(condition)
    global_context = model.standardized_global_context(condition)
    lowpass = torch.zeros(
        (batch, 1, 1, 1, 1), device=condition.device, dtype=torch.float64
    )
    logdets = []
    for coarse_index, (flow, base) in enumerate(zip(model.flows, bases, strict=True)):
        level = model.levels - 1 - coarse_index
        context = model.scale_context(condition, lowpass, global_context)
        detail, logdet = flow.inverse(base, context)
        mean = model.detail_mean[level][None, :, None, None, None]
        std = model.detail_std[level][None, :, None, None, None]
        lowpass = haar_synthesis(lowpass, (detail * std + mean).double())
        logdets.append(logdet)
    sample = lowpass.to(dtype=condition.dtype)
    sample -= sample.mean(dim=(-3, -2, -1), keepdim=True)
    return sample, logdets


def _statistics_bank() -> dict[str, Any]:
    return {
        "truth_latent": TensorMoments(),
        "generated_latent": TensorMoments(),
        "truth_standardized_haar_coarse_to_fine": [TensorMoments() for _ in range(6)],
        "generated_standardized_haar_coarse_to_fine": [TensorMoments() for _ in range(6)],
        "truth_base_z_coarse_to_fine": [TensorMoments() for _ in range(6)],
        "generated_base_z_coarse_to_fine": [TensorMoments() for _ in range(6)],
    }


def _report_bank(bank: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            [item.report() for item in value]
            if isinstance(value, list)
            else value.report()
        )
        for key, value in bank.items()
    }


def _update_encoded(
    bank: Mapping[str, Any], prefix: str, details: list[torch.Tensor], bases: list[torch.Tensor]
) -> None:
    for accumulator, value in zip(
        bank[f"{prefix}_standardized_haar_coarse_to_fine"], details, strict=True
    ):
        accumulator.update(value)
    for accumulator, value in zip(
        bank[f"{prefix}_base_z_coarse_to_fine"], bases, strict=True
    ):
        accumulator.update(value)


@torch.inference_mode()
def audit_domain(
    *,
    domain: str,
    step: int,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    registry: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    v20: Mapping[str, Any],
    decision_candidate: Mapping[str, Any],
    repo: Path,
    device: torch.device,
) -> dict[str, Any]:
    experiment = v20["e8_gaussianized_marginal_retrain"]
    data = experiment["data"][domain]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[domain]["validation"]]
    indices = _indices(experiment["development_objects"][domain], repo)
    dataset = V14ResidualDataset(data["path"], cache["path"], False)
    ensemble_path = Path(
        decision_candidate["domains"][DOMAIN_KEYS[domain]]["ensemble"]
    )
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    centers = torch.as_tensor(profile["centers"], dtype=torch.float64, device=device)
    mu = torch.as_tensor(profile["mu"], dtype=torch.float64, device=device)
    log_sigma = torch.as_tensor(profile["log_sigma"], dtype=torch.float64, device=device)
    z_knots = torch.as_tensor(transform["z_knots"], dtype=torch.float32, device=device)
    residual_knots = torch.as_tensor(
        transform["residual_value_knots"], dtype=torch.float32, device=device
    )
    seed = int(registry["training_protocol"]["sampling_seeds"][domain])
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    bank = _statistics_bank()
    maximum_roundtrip = 0.0
    roundtrip_square_sum = 0.0
    roundtrip_values = 0
    maximum_logdet_cancellation = [0.0] * 6
    maximum_replay_difference = 0.0
    replayed_fields = 0
    with h5py.File(ensemble_path, "r") as ensemble:
        if [int(value) for value in ensemble["source_index"][:]] != indices:
            raise ValueError(f"{domain} V26 ensemble source order differs")
        for object_index, data_index in enumerate(indices):
            condition, truth_latent, corrected_mean, _ = dataset[data_index]
            condition_batch = condition[None].to(device).expand(16, -1, -1, -1, -1)
            generated = model.sample(condition_batch, generator=generator)
            truth = truth_latent[None].to(device)
            bank["truth_latent"].update(truth)
            bank["generated_latent"].update(generated)
            truth_details, truth_bases, _ = encode_latent(model, truth, condition[None].to(device))
            generated_details, generated_bases, forward_logdets = encode_latent(
                model, generated, condition_batch
            )
            _update_encoded(bank, "truth", truth_details, truth_bases)
            _update_encoded(bank, "generated", generated_details, generated_bases)
            recovered, inverse_logdets = decode_bases(model, generated_bases, condition_batch)
            difference = (recovered - generated).double()
            maximum_roundtrip = max(maximum_roundtrip, float(difference.abs().max()))
            roundtrip_square_sum += float(difference.square().sum())
            roundtrip_values += int(difference.numel())
            for scale, (forward, inverse) in enumerate(
                zip(forward_logdets, inverse_logdets, strict=True)
            ):
                maximum_logdet_cancellation[scale] = max(
                    maximum_logdet_cancellation[scale],
                    float((forward + inverse).abs().max()),
                )
            # Reproduce one member for every object through the complete frozen
            # inverse chain.  This binds the raw-latent diagnostics to the exact
            # stored ensemble without repeating all 16 CPU Fourier inversions.
            u = inverse_gaussianize_torch(generated[:1], z_knots, residual_knots)
            standardized = invert_profile_torch(
                u,
                corrected_mean[None].to(device),
                centers,
                mu,
                log_sigma,
            )[0, 0].float().cpu().numpy()
            location, scales = dataset.predicted_location_scales(data_index)
            physical = inverse_standardized_residual(
                standardized,
                predicted_location=location,
                predicted_scales=scales,
                voxel_mpc_h=dataset.voxel_mpc_h,
            )
            replay = corrected_mean.numpy()[0] + physical
            stored = np.asarray(ensemble["sample"][object_index, 0, 0], dtype=np.float32)
            maximum_replay_difference = max(
                maximum_replay_difference,
                float(np.max(np.abs(replay.astype(np.float32) - stored))),
            )
            replayed_fields += 1
            print(f"[audit] step={step} domain={domain} object={object_index + 1}/16", flush=True)
    report = _report_bank(bank)
    report.update(
        {
            "objects": len(indices),
            "generated_members_per_object": 16,
            "ensemble": str(ensemble_path),
            "ensemble_sha256": sha256_file(ensemble_path),
            "roundtrip": {
                "maximum_absolute_latent_error": maximum_roundtrip,
                "rms_latent_error": math.sqrt(roundtrip_square_sum / roundtrip_values),
                "maximum_absolute_logdet_cancellation_coarse_to_fine": maximum_logdet_cancellation,
            },
            "stored_ensemble_replay": {
                "members_replayed": replayed_fields,
                "member_selection": "member zero of every development object",
                "maximum_absolute_y_difference": maximum_replay_difference,
                "float32_machine_epsilon": float(np.finfo(np.float32).eps),
            },
            "checkpoint_balanced_validation_nll": float(
                checkpoint["balanced_validation_nll"]
            ),
        }
    )
    return report


def _mechanism_summary(
    candidates: Mapping[str, Any], optimization: Mapping[str, Any]
) -> dict[str, Any]:
    final = candidates["30000"]
    roundtrip_max = max(
        row["roundtrip"]["maximum_absolute_latent_error"] for row in final.values()
    )
    replay_max = max(
        row["stored_ensemble_replay"]["maximum_absolute_y_difference"]
        for row in final.values()
    )
    support = {}
    for domain, row in final.items():
        generated = row["generated_latent"]["absolute_tail_fractions"]["5.0"]
        truth = row["truth_latent"]["absolute_tail_fractions"]["5.0"]
        support[domain] = {
            "generated_absolute_latent_at_least_5_fraction": generated,
            "truth_absolute_latent_at_least_5_fraction": truth,
            "generated_over_truth": None if truth == 0.0 else generated / truth,
        }
    coarse_worsened = {
        domain: [
            row["coarse_to_fine_index"]
            for row in scales
            if row["worsened_10000_to_30000"]
        ]
        for domain, scales in optimization["scale_resolved"].items()
    }
    numerical_roundtrip_stable = roundtrip_max <= 2.0e-4
    stored_replay_stable = replay_max <= 2.0 * float(np.finfo(np.float32).eps)
    generated_support_excess = all(
        row["generated_absolute_latent_at_least_5_fraction"]
        > row["truth_absolute_latent_at_least_5_fraction"]
        for row in support.values()
    )
    if numerical_roundtrip_stable and stored_replay_stable and generated_support_excess:
        classification = (
            "joint_latent_tail_miscalibration_under_fine_scale_dominated_likelihood"
        )
        next_step = (
            "do_not_extend_v26; redesign the observation-to-density representation "
            "or likelihood so voxel-support and cross-scale dependence are explicit"
        )
    elif not numerical_roundtrip_stable:
        classification = "trained_spline_numerical_inversion_failure"
        next_step = "stop model redesign and repair the trained-flow inverse first"
    else:
        classification = "v26_failure_mechanism_not_isolated"
        next_step = "inspect per-object base-z and conditional-context calibration"
    return {
        "classification": classification,
        "next": next_step,
        "trained_flow_roundtrip_stable": numerical_roundtrip_stable,
        "stored_ensemble_replay_stable": stored_replay_stable,
        "maximum_final_roundtrip_error": roundtrip_max,
        "maximum_final_stored_replay_difference_y": replay_max,
        "latent_G21_support": support,
        "generated_support_excess_all_domains": generated_support_excess,
        "validation_scales_worsened_10000_to_30000": coarse_worsened,
        "finest_scale_objective_fraction": optimization[
            "finest_scale_objective_fraction"
        ],
        "nonfinite_mean_gradient_intervals": optimization[
            "nonfinite_mean_gradient_intervals"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--decision", type=Path, default=DECISION)
    parser.add_argument("--failure-audit", type=Path, default=FAILURE_AUDIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V26 trained-flow mechanism audit requires Lageunha")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V26 trained-flow mechanism audit requires the Ada GPU")
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V26 trained-flow mechanism audit requires a clean worktree")
    output = args.out.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise RuntimeError(f"refusing to overwrite V26 mechanism audit: {output}")
    registry, artifacts, v20, _, haar = load_frozen_program(
        args.registry.resolve(), repo
    )
    decision = json.loads(args.decision.read_text())
    failure = json.loads(args.failure_audit.read_text())
    if (
        decision.get("development_pass") is not False
        or decision.get("next")
        != "audit_optimization_and_scale_resolved_nll_without_extending_or_tuning"
        or failure.get("decision_digest_sha256") != decision.get("decision_digest_sha256")
    ):
        raise ValueError("V26 failed-decision provenance differs")
    training_run = json.loads((args.training / "run.json").read_text())
    if (
        training_run.get("schema") != MODEL_SCHEMA
        or training_run.get("status") != "complete"
        or training_run.get("experiment_registry_sha256") != REGISTRY_SHA256
    ):
        raise ValueError("V26 completed training provenance differs")
    optimization = analyze_history(
        json.loads((args.training / "history.json").read_text())
    )
    decision_candidates = {
        str(candidate["step"]): candidate for candidate in decision["candidates"]
    }
    device = torch.device(args.device)
    candidate_reports: dict[str, Any] = {}
    for step in CANDIDATE_STEPS:
        checkpoint_path = (
            args.training / "validation_checkpoints" / f"step_{step:06d}.pt"
        )
        checkpoint, checkpoint_sha = _validate_checkpoint(
            checkpoint_path, step=step, artifacts=artifacts
        )
        model = build_model(
            haar, checkpoint["observable_context_features"], device=device
        )
        model.load_state_dict(checkpoint["ema_model"])
        model.eval()
        domains = {}
        for domain in DOMAIN_ORDER:
            domains[domain] = audit_domain(
                domain=domain,
                step=step,
                model=model,
                checkpoint=checkpoint,
                registry=registry,
                artifacts=artifacts,
                v20=v20,
                decision_candidate=decision_candidates[str(step)],
                repo=repo,
                device=device,
            )
        candidate_reports[str(step)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "domains": domains,
        }
        del model
        torch.cuda.empty_cache()
    compact_candidates = {
        step: row["domains"] for step, row in candidate_reports.items()
    }
    report = {
        "schema": SCHEMA,
        "status": "complete_development_only_post_failure_audit",
        "purpose": "complete frozen V26 trained-flow and scale-resolved diagnostics",
        "registry": str(args.registry.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "training": str(args.training.resolve()),
        "training_run_sha256": sha256_file(args.training / "run.json"),
        "history_sha256": sha256_file(args.training / "history.json"),
        "decision": str(args.decision.resolve()),
        "decision_sha256": sha256_file(args.decision),
        "decision_digest_sha256": decision["decision_digest_sha256"],
        "failure_audit": str(args.failure_audit.resolve()),
        "failure_audit_sha256": sha256_file(args.failure_audit),
        "execution_host": socket.gethostname(),
        "execution_gpu": torch.cuda.get_device_name(0),
        "audit_code_commit": commit,
        "worktree_clean_at_audit": clean,
        "optimization": optimization,
        "candidates": candidate_reports,
        "mechanism_summary": _mechanism_summary(compact_candidates, optimization),
        "tuning_after_results": False,
        "training_horizon_extended": False,
        "thresholds_changed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report["mechanism_summary"], indent=2), flush=True)
    print(f"[out] {output}", flush=True)


if __name__ == "__main__":
    main()
