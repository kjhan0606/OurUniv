#!/usr/bin/env python
"""Train the frozen V22 long-horizon replication of V21."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from hong2021_residual_v6 import karras_sigmas, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_v14_edm import (
    V21_CONDITIONAL_AFFINE_CACHE_SCHEMA, V22_E10_SCHEMA,
    V14ResidualDataset, decoder_upsampling_for_schema, train,
)
from hong2021_v15_development_gate import canonical_digest
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _indices, _rng_pairing_self_check, write_prior_matched_ensemble
from hong2021_v18_init import PriorMatchedInitializer, prior_matched_spectral_std
from hong2021_v18_init import sha256_file
from hong2021_v21_edm import (
    ARTIFACT_SHA256, P_MEAN, P_STD,
    load_frozen_program as load_v21_program,
    frozen_training_namespace as v21_training_namespace,
)
from hong2021_v21_conditional_affine import invert_profile_torch


REGISTRY_SCHEMA = "hong2021-v22-long-horizon-development-program-v1"
REGISTRY_SHA256 = "2f2d5337ceecab413e647f54bcaa75e1502d76db3b24daafd22d1c1a2bd7cfbe"


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != REGISTRY_SHA256:
        raise ValueError("V22 registry differs from its frozen hash")
    registry = json.loads(path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("status") != "frozen_before_implementation_or_execution":
        raise ValueError("V22 registry schema or status mismatch")
    parent = registry["parent_evidence"]
    v21_registry = _resolve(parent["v21_registry"], repo)
    v21_artifacts = _resolve(parent["v21_artifacts"], repo)
    if sha256_file(v21_registry) != parent["v21_registry_sha256"] or sha256_file(v21_artifacts) != parent["v21_artifacts_sha256"]:
        raise ValueError("V22 V21 parent hash mismatch")
    decision_path = Path(parent["v21_decision"])
    if sha256_file(decision_path) != parent["v21_decision_sha256"]:
        raise ValueError("V22 V21 decision file hash mismatch")
    decision = json.loads(decision_path.read_text())
    if canonical_digest(decision) != parent["v21_decision_digest_sha256"] or decision.get("development_pass") is not False:
        raise ValueError("V22 parent decision digest or failure state mismatch")
    if decision.get("Astrid_used") is not False or decision.get("EAGLE_RefL0100N1504_used") is not False:
        raise ValueError("V22 parent decision violated the independent-data firewall")
    audit_path = _resolve(parent["failure_audit"], repo)
    if sha256_file(audit_path) != parent["failure_audit_sha256"]:
        raise ValueError("V22 failure audit hash mismatch")
    change = registry["single_change"]
    if change != {
        "description": "Train the byte-identical V21 representation/model/protocol from scratch with CosineAnnealingLR over 30000 rather than 10000 optimizer steps.",
        "training_from_scratch": True, "continuation_checkpoint": None,
        "steps": 30000, "candidate_steps": [10000, 20000, 30000],
        "schedule": "CosineAnnealingLR over exactly 30000 optimizer steps",
        "minimum_learning_rate": 0.00002,
    }:
        raise ValueError("V22 differs from its single predeclared horizon change")
    _, artifacts, v20 = load_v21_program(v21_registry, v21_artifacts, repo)
    if sha256_file(v21_artifacts) != ARTIFACT_SHA256:
        raise ValueError("V22 inherited V21 artifacts differ")
    return registry, artifacts, v20, decision


def frozen_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    repo = args.repo.resolve()
    registry, _, _, _ = load_frozen_program(args.registry.resolve(), repo)
    parent = registry["parent_evidence"]
    base = v21_training_namespace(argparse.Namespace(
        repo=repo,
        registry=_resolve(parent["v21_registry"], repo),
        artifacts=_resolve(parent["v21_artifacts"], repo),
        out=args.out,
        device=args.device,
    ))
    base.steps = 30000
    base.candidate_steps = "10000,20000,30000"
    base.run_schema = V22_E10_SCHEMA
    base.experiment_registry = str(args.registry.resolve())
    base.experiment_registry_sha256 = REGISTRY_SHA256
    return base


def _validate_checkpoint(path: Path, *, step: int, artifacts: dict[str, Any]) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V22_E10_SCHEMA or int(checkpoint.get("step", -1)) != step:
        raise ValueError("V22 checkpoint schema or step mismatch")
    if checkpoint.get("experiment_registry_sha256") != REGISTRY_SHA256 or checkpoint.get("worktree_clean_at_launch") is not True:
        raise ValueError("V22 checkpoint provenance mismatch")
    initialization = artifacts["initialization"]
    if float(checkpoint.get("sigma_data", math.nan)) != float(initialization["sigma_data"]):
        raise ValueError("V22 checkpoint sigma_data mismatch")
    if float(checkpoint.get("edm_p_mean", math.nan)) != P_MEAN or float(checkpoint.get("edm_p_std", math.nan)) != P_STD:
        raise ValueError("V22 checkpoint noise constants mismatch")
    if int(checkpoint.get("steps", -1)) != 30000 or checkpoint.get("candidate_steps") != [10000, 20000, 30000]:
        raise ValueError("V22 checkpoint horizon differs from registry")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry, artifacts, v20, _ = load_frozen_program(args.registry.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V22 sampling requires a clean committed worktree")
    domain, step = args.domain, int(args.step)
    if domain not in ("TNG100", "SIMBA", "Swift") or step not in (10000, 20000, 30000):
        raise ValueError("V22 domain or step is not preregistered")
    checkpoint_path = args.training_root.resolve() / "validation_checkpoints" / f"step_{step:06d}.pt"
    checkpoint, checkpoint_sha = _validate_checkpoint(checkpoint_path, step=step, artifacts=artifacts)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    data = experiment["data"][domain]["validation_data"]
    cache_key = {"TNG100": "TNG100_validation", "SIMBA": "SIMBA_validation", "Swift": "Swift_validation"}[domain]
    cache = artifacts["caches"][cache_key]
    data_path, cache_path = Path(data["path"]), Path(cache["path"])
    indices = _indices(experiment["development_objects"][domain], repo)
    sampler = experiment["sampler"]
    seed = int(sampler["sampling_seeds"][domain])
    seed_everything(seed)
    device = torch.device(args.device)
    features = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]), context_mean=features["mean"],
        context_std=features["std"], decoder_upsampling=decoder_upsampling_for_schema(checkpoint["schema"]),
    )
    model.load_state_dict(checkpoint["ema_model"]); model.eval().to(device)
    dataset = V14ResidualDataset(data_path, cache_path, False)
    if dataset.cache_schema != V21_CONDITIONAL_AFFINE_CACHE_SCHEMA or dataset.grid != 64 or dataset.voxel_mpc_h != 0.3125:
        raise ValueError("V22 inherited cache schema or grid mismatch")
    sigmas = karras_sigmas(int(sampler["steps"]), float(sampler["sigma_min"]), float(sampler["sigma_max"]), float(sampler["rho"]), device)
    sigma_first = float(sigmas[0])
    initialization = artifacts["initialization"]
    v19_path = Path(v20["parent_evidence"]["v19_registry"])
    if not v19_path.is_absolute():
        v19_path = repo / v19_path
    v19 = json.loads(v19_path.read_text())
    maximum_imaginary = float(v19["e7_band_anchored_noise_retrain"]["initialization"]["maximum_imaginary_over_real_rms"])
    initializer = PriorMatchedInitializer(
        prior_matched_spectral_std(dataset.grid, dataset.voxel_mpc_h, sigma_first, initialization["source_balanced_band_mode_variance"], device=device),
        maximum_imaginary_ratio=maximum_imaginary,
    )
    if not _rng_pairing_self_check(device, initializer, grid=dataset.grid):
        raise RuntimeError("V22 initializer changed the paired random stream")
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    if transform.get("profile_sha256") != artifacts["profile"]["sha256"]:
        raise ValueError("V22 transform/profile binding mismatch")
    z_knots = torch.as_tensor(transform["z_knots"], dtype=torch.float32, device=device)
    residual_knots = torch.as_tensor(transform["residual_value_knots"], dtype=torch.float32, device=device)
    centers = torch.as_tensor(profile["centers"], dtype=torch.float64, device=device)
    mu = torch.as_tensor(profile["mu"], dtype=torch.float64, device=device)
    log_sigma = torch.as_tensor(profile["log_sigma"], dtype=torch.float64, device=device)
    def conditional_inverse(value: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        return invert_profile_torch(
            inverse_gaussianize_torch(value, z_knots, residual_knots),
            mean, centers, mu, log_sigma,
        )
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    _ = conditional_inverse(torch.zeros((1, 1, 2, 2, 2), device=device), torch.zeros((1, 1, 2, 2, 2), device=device))
    if not torch.equal(cpu_state, torch.random.get_rng_state()) or (
        cuda_state is not None and not torch.equal(cuda_state, torch.cuda.get_rng_state(device))
    ):
        raise RuntimeError("V22 conditional inverse consumed RNG state")
    write_prior_matched_ensemble(
        model=model, dataset=dataset, checkpoint=checkpoint, checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha, output=args.out.resolve(), indices=indices,
        ensemble_members=int(sampler["ensemble_members"]), sampling_steps=int(sampler["steps"]),
        sigma_min=float(sampler["sigma_min"]), sigma_max=float(sampler["sigma_max"]), rho=float(sampler["rho"]),
        seed=seed, device=device, initializer=initializer, sigma_first=sigma_first,
        conditional_latent_inverse=conditional_inverse,
        metadata={
            "source_cache": str(cache_path), "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"], "v22_registry_sha256": REGISTRY_SHA256,
            "v21_artifact_attestation_sha256": ARTIFACT_SHA256,
            "v21_profile_sha256": artifacts["profile"]["sha256"],
            "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
            "init_measurement_report_sha256": initialization["measurement_sha256"],
            "init_band_mode_variances_json": json.dumps(initialization["source_balanced_band_mode_variance"]),
            "conditional_inverse_additional_rng_draws": 0, "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD, "sampling_code_commit": commit, "worktree_clean_at_sampling": clean,
        },
        progress_label=f"V22 {domain} {step}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train"); sampling = sub.add_parser("sample")
    for item in (training, sampling):
        item.add_argument("--registry", type=Path, required=True)
        item.add_argument("--repo", type=Path, required=True)
        item.add_argument("--device", default="cuda")
    training.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--training-root", type=Path, required=True)
    sampling.add_argument("--domain", choices=("TNG100", "SIMBA", "Swift"), required=True)
    sampling.add_argument("--step", type=int, choices=(10000, 20000, 30000), required=True)
    sampling.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train(frozen_training_namespace(args)) if args.mode == "train" else sample(args)


if __name__ == "__main__":
    main()
