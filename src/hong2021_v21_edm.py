#!/usr/bin/env python
"""Train and sample the frozen V21 conditional-affine experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import h5py

from hong2021_residual_v6 import karras_sigmas, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_v14_edm import (
    V21_CONDITIONAL_AFFINE_CACHE_SCHEMA,
    V21_E9_SCHEMA,
    V14ResidualDataset,
    decoder_upsampling_for_schema,
    train,
)
from hong2021_v15_development_gate import canonical_digest
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _indices, _rng_pairing_self_check, write_prior_matched_ensemble
from hong2021_v18_init import PriorMatchedInitializer, prior_matched_spectral_std, sha256_file
from hong2021_v21_conditional_affine import invert_profile_torch


REGISTRY_SCHEMA = "hong2021-v21-voxel-conditional-affine-development-program-v1"
REGISTRY_SHA256 = "e44486a52f284ded1d17a83da491b571a7987f0380cd5ff0fdc1ef5f3c29a27d"
ARTIFACT_SCHEMA = "hong2021-v21-derived-artifact-attestation-v1"
ARTIFACT_SHA256 = "5622fc5a22b7502eac433f50e6cc2b51e6253c6e89b179335b1f8eef4e6d5852"
P_MEAN = -0.5110403821419721
P_STD = 1.2
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
CACHE_KEYS = {
    "TNG100": {"train": "TNG100_train", "validation": "TNG100_validation"},
    "SIMBA": {"train": "SIMBA_train", "validation": "SIMBA_validation"},
    "Swift": {"train": "Swift_train", "validation": "Swift_validation"},
}


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_program(registry_path: Path, artifacts_path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(registry_path) != REGISTRY_SHA256:
        raise ValueError("V21 registry differs from its frozen hash")
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("status") != "frozen_before_implementation_or_execution":
        raise ValueError("V21 registry schema or status mismatch")
    parent = registry["parent_evidence"]
    v20_path = _resolve(parent["v20_registry"], repo)
    if sha256_file(v20_path) != parent["v20_registry_sha256"]:
        raise ValueError("V21 parent V20 registry hash mismatch")
    v20 = json.loads(v20_path.read_text())
    decision_path = Path(parent["v20_decision"])
    if sha256_file(decision_path) != parent["v20_decision_sha256"]:
        raise ValueError("V21 parent decision file hash mismatch")
    decision = json.loads(decision_path.read_text())
    if canonical_digest(decision) != parent["v20_decision_digest_sha256"]:
        raise ValueError("V21 parent decision digest mismatch")
    if decision.get("Astrid_used") is not False or decision.get("EAGLE_RefL0100N1504_used") is not False:
        raise ValueError("V21 parent does not prove the independent-data firewall")
    if sha256_file(_resolve(parent["conditional_audit"], repo)) != parent["conditional_audit_sha256"]:
        raise ValueError("V21 conditional audit hash mismatch")
    audit = registry["independent_audit"]
    if sha256_file(_resolve(audit["record"], repo)) != audit["record_sha256"]:
        raise ValueError("V21 independent design audit hash mismatch")
    if sha256_file(artifacts_path) != ARTIFACT_SHA256:
        raise ValueError("V21 derived artifact attestation hash mismatch")
    artifacts = json.loads(artifacts_path.read_text())
    if artifacts.get("schema") != ARTIFACT_SCHEMA or artifacts.get("astrid_accessed") is not False or artifacts.get("historical_eagle_accessed") is not False:
        raise ValueError("V21 derived artifact attestation is invalid")
    for row in (artifacts["profile"], artifacts["gaussianization"], *artifacts["caches"].values()):
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("V21 derived artifact hash mismatch")
    initialization = artifacts["initialization"]
    if sha256_file(Path(initialization["measurement_path"])) != initialization["measurement_sha256"]:
        raise ValueError("V21 initialization measurement hash mismatch")
    if abs(math.log(0.6 * float(initialization["sigma_data"])) - P_MEAN) > 1e-15:
        raise ValueError("V21 p_mean differs from its frozen sigma_data")
    if float(initialization["p_std"]) != P_STD:
        raise ValueError("V21 p_std mismatch")
    return registry, artifacts, v20


def frozen_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    repo = args.repo.resolve()
    registry, artifacts, v20 = load_frozen_program(args.registry.resolve(), args.artifacts.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V21 training requires a clean committed worktree")
    output = args.out.resolve()
    if output.exists():
        raise RuntimeError(f"V21 refuses a pre-existing training output: {output}")
    v20_experiment = v20["e8_gaussianized_marginal_retrain"]
    data = v20_experiment["data"]
    protocol = registry["inherited_training_protocol"]
    caches = artifacts["caches"]
    def data_path(domain: str, split: str) -> str:
        return data[domain][f"{split}_data"]["path"]
    def cache_path(domain: str, split: str) -> str:
        return caches[CACHE_KEYS[domain][split]]["path"]
    train_rms = []
    for domain in DOMAIN_KEYS:
        with h5py.File(cache_path(domain, "train"), "r") as handle:
            train_rms.append(float(handle.attrs["standardized_residual_rms"]))
    measured_sigma = math.sqrt(sum(value * value for value in train_rms) / 3.0)
    if measured_sigma.hex() != float(artifacts["initialization"]["sigma_data"]).hex():
        raise ValueError("V21 cache RMS metadata differs from attested sigma_data")
    return argparse.Namespace(
        tng_train_data=data_path("TNG100", "train"), tng_train_cache=cache_path("TNG100", "train"),
        tng_validation_data=data_path("TNG100", "validation"), tng_validation_cache=cache_path("TNG100", "validation"),
        simba_train_data=data_path("SIMBA", "train"), simba_train_cache=cache_path("SIMBA", "train"),
        simba_validation_data=data_path("SIMBA", "validation"), simba_validation_cache=cache_path("SIMBA", "validation"),
        swift_train_data=data_path("Swift", "train"), swift_train_cache=cache_path("Swift", "train"),
        swift_validation_data=data_path("Swift", "validation"), swift_validation_cache=cache_path("Swift", "validation"),
        out=str(output), steps=10000, candidate_steps="5000,10000",
        batch=6, validation_batch=6, workers=int(protocol["workers"]), base_channels=32,
        lr=0.0002, min_lr=0.00002, weight_decay=0.0001, ema_decay=0.999,
        gradient_clip=1.0, edm_p_mean=None, edm_p_mean_sigma_data_fraction=0.6,
        edm_p_std=P_STD, tail_exponent=0.25, tail_maximum=10.0,
        validation_every=500, validation_seeds=[99173, 99174, 99175], seed=144021,
        device=args.device, smoke_limit=None, run_schema=V21_E9_SCHEMA,
        experiment_registry=str(args.registry.resolve()), experiment_registry_sha256=REGISTRY_SHA256,
        code_commit_at_launch=commit, worktree_clean_at_launch=clean,
    )


def _validate_checkpoint(path: Path, step: int, artifacts: dict[str, Any]) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V21_E9_SCHEMA or int(checkpoint.get("step", -1)) != step:
        raise ValueError("V21 checkpoint schema or step mismatch")
    if checkpoint.get("experiment_registry_sha256") != REGISTRY_SHA256 or checkpoint.get("worktree_clean_at_launch") is not True:
        raise ValueError("V21 checkpoint provenance mismatch")
    initialization = artifacts["initialization"]
    if float(checkpoint.get("sigma_data", math.nan)) != float(initialization["sigma_data"]):
        raise ValueError("V21 checkpoint sigma_data mismatch")
    if float(checkpoint.get("edm_p_mean", math.nan)) != P_MEAN or float(checkpoint.get("edm_p_std", math.nan)) != P_STD:
        raise ValueError("V21 checkpoint noise constants mismatch")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry, artifacts, v20 = load_frozen_program(args.registry.resolve(), args.artifacts.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V21 sampling requires a clean committed worktree")
    domain, step = args.domain, int(args.step)
    if domain not in DOMAIN_KEYS or step not in (5000, 10000):
        raise ValueError("V21 domain or step is not preregistered")
    checkpoint_path = args.training_root.resolve() / "validation_checkpoints" / f"step_{step:06d}.pt"
    checkpoint, checkpoint_sha = _validate_checkpoint(checkpoint_path, step, artifacts)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    data = experiment["data"][domain]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[domain]["validation"]]
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
        raise ValueError("V21 sampling cache schema or grid mismatch")
    sigmas = karras_sigmas(int(sampler["steps"]), float(sampler["sigma_min"]), float(sampler["sigma_max"]), float(sampler["rho"]), device)
    sigma_first = float(sigmas[0])
    initialization = artifacts["initialization"]
    v19 = json.loads(_resolve(v20["parent_evidence"]["v19_registry"], repo).read_text())
    maximum_imaginary = float(v19["e7_band_anchored_noise_retrain"]["initialization"]["maximum_imaginary_over_real_rms"])
    initializer = PriorMatchedInitializer(
        prior_matched_spectral_std(dataset.grid, dataset.voxel_mpc_h, sigma_first, initialization["source_balanced_band_mode_variance"], device=device),
        maximum_imaginary_ratio=maximum_imaginary,
    )
    if not _rng_pairing_self_check(device, initializer, grid=dataset.grid):
        raise RuntimeError("V21 initializer changed the paired random stream")
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    if profile.get("schema") != "hong2021-v21-voxel-conditional-affine-profile-v1":
        raise ValueError("V21 profile schema mismatch")
    if transform.get("schema") != "hong2021-v21-equal-source-conditional-affine-gaussianization-v1":
        raise ValueError("V21 transform schema mismatch")
    if transform.get("profile_sha256") != artifacts["profile"]["sha256"]:
        raise ValueError("V21 transform was not fitted from the attested profile")
    z_knots = torch.as_tensor(transform["z_knots"], dtype=torch.float32, device=device)
    residual_knots = torch.as_tensor(transform["residual_value_knots"], dtype=torch.float32, device=device)
    centers = torch.as_tensor(profile["centers"], dtype=torch.float64, device=device)
    mu = torch.as_tensor(profile["mu"], dtype=torch.float64, device=device)
    log_sigma = torch.as_tensor(profile["log_sigma"], dtype=torch.float64, device=device)
    def conditional_inverse(value: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        u = inverse_gaussianize_torch(value, z_knots, residual_knots)
        return invert_profile_torch(u, mean, centers, mu, log_sigma)
    cpu_state = torch.random.get_rng_state().clone()
    device_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    _ = conditional_inverse(torch.zeros((1, 1, 2, 2, 2), device=device), torch.zeros((1, 1, 2, 2, 2), device=device))
    if not torch.equal(cpu_state, torch.random.get_rng_state()):
        raise RuntimeError("V21 conditional inverse consumed RNG state")
    if device_state is not None and not torch.equal(device_state, torch.cuda.get_rng_state(device)):
        raise RuntimeError("V21 conditional inverse consumed CUDA RNG state")
    write_prior_matched_ensemble(
        model=model, dataset=dataset, checkpoint=checkpoint, checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha, output=args.out.resolve(), indices=indices,
        ensemble_members=int(sampler["ensemble_members"]), sampling_steps=int(sampler["steps"]),
        sigma_min=float(sampler["sigma_min"]), sigma_max=float(sampler["sigma_max"]), rho=float(sampler["rho"]),
        seed=seed, device=device, initializer=initializer, sigma_first=sigma_first,
        conditional_latent_inverse=conditional_inverse,
        metadata={
            "source_cache": str(cache_path), "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"], "v21_registry_sha256": REGISTRY_SHA256,
            "v21_artifact_attestation_sha256": ARTIFACT_SHA256,
            "v21_profile_sha256": artifacts["profile"]["sha256"],
            "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
            "init_measurement_report_sha256": initialization["measurement_sha256"],
            "init_band_mode_variances_json": json.dumps(initialization["source_balanced_band_mode_variance"]),
            "conditional_inverse_additional_rng_draws": 0, "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD, "sampling_code_commit": commit, "worktree_clean_at_sampling": clean,
        },
        progress_label=f"V21 {domain} {step}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    sampling = sub.add_parser("sample")
    for item in (training, sampling):
        item.add_argument("--registry", type=Path, required=True)
        item.add_argument("--artifacts", type=Path, required=True)
        item.add_argument("--repo", type=Path, required=True)
        item.add_argument("--device", default="cuda")
    training.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--training-root", type=Path, required=True)
    sampling.add_argument("--domain", choices=tuple(DOMAIN_KEYS), required=True)
    sampling.add_argument("--step", type=int, choices=(5000, 10000), required=True)
    sampling.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(frozen_training_namespace(args)) if args.mode == "train" else sample(args)


if __name__ == "__main__":
    main()
