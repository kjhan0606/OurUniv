#!/usr/bin/env python
"""Launch and sample only the frozen V20-E8 Gaussianized-marginal experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_residual_v6 import karras_sigmas, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_v14_edm import (
    V20_E8_SCHEMA,
    V20_GAUSSIANIZED_CACHE_SCHEMA,
    V14ResidualDataset,
    decoder_upsampling_for_schema,
    train,
)
from hong2021_v15_development_gate import canonical_digest
from hong2021_v15_edm import git_state
from hong2021_v18_edm import (
    _indices,
    _rng_pairing_self_check,
    write_prior_matched_ensemble,
)
from hong2021_v18_init import (
    PriorMatchedInitializer,
    prior_matched_spectral_std,
    sha256_file,
)
from hong2021_v20_gaussianize import (
    FROZEN_REGISTRY_SHA256,
    load_frozen_registry as load_base_registry,
)


P_MEAN = -0.5108340868686441
P_STD = 1.2
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _validate_parent_evidence(
    registry: dict[str, Any], repo: Path,
) -> dict[str, Any]:
    parent = registry["parent_evidence"]
    v19_path = _resolve(parent["v19_registry"], repo)
    v19 = json.loads(v19_path.read_text())
    for label, row in parent["development_decisions"].items():
        path = Path(row["path"]).resolve()
        if sha256_file(path) != row["file_sha256"]:
            raise ValueError(f"V20 parent {label} decision file hash mismatch")
        decision = json.loads(path.read_text())
        if canonical_digest(decision) != row["decision_digest_sha256"]:
            raise ValueError(f"V20 parent {label} decision digest mismatch")
        if decision.get("Astrid_used") is not False:
            raise ValueError(f"V20 parent {label} does not prove Astrid unopened")
        if decision.get("EAGLE_RefL0100N1504_used") is not False:
            raise ValueError(f"V20 parent {label} used historical EAGLE")
    e7 = parent["development_decisions"]["E7"]
    e7_decision = json.loads(Path(e7["path"]).read_text())
    failure = e7_decision.get("failure_classification_10k", {})
    if failure.get("class") != e7["classification"] or e7_decision.get(
        "next"
    ) != e7["next"]:
        raise ValueError("V20 parent E7 decision did not authorize representation audit")
    if sha256_file(Path(parent["e7_run"])) != parent["e7_run_sha256"]:
        raise ValueError("V20 parent E7 run hash mismatch")
    for row in parent["e7_checkpoints"]:
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("V20 parent E7 checkpoint hash mismatch")
    return v19


def _validate_exact_experiment(experiment: dict[str, Any]) -> None:
    if experiment.get("training_from_scratch") is not True:
        raise ValueError("V20-E8 must train from scratch")
    if experiment.get("continuation_checkpoint") is not None:
        raise ValueError("V20-E8 may not continue a checkpoint")
    if experiment.get("steps") != 10000 or experiment.get("candidate_steps") != [
        5000, 10000,
    ]:
        raise ValueError("V20-E8 training budget differs from preregistration")
    if experiment.get("checkpoint_schema") != V20_E8_SCHEMA:
        raise ValueError("V20-E8 checkpoint schema differs from preregistration")
    transform = experiment["gaussianization"]
    if transform.get("source_weights") != [1 / 3, 1 / 3, 1 / 3]:
        raise ValueError("V20 transform source weights differ from equal source balance")
    if transform.get("validation_or_test_data_used") is not False:
        raise ValueError("V20 transform used validation or test data")
    if (
        transform.get("histogram_bins") != 131072
        or transform.get("knots") != 8193
        or transform.get("z_support") != [-5.0, 5.0]
    ):
        raise ValueError("V20 transform discretization differs from V12 inheritance")
    projection = experiment["dc_projection"]
    if (
        float(projection["maximum_postprojection_ortho_dc"]) != 1.0e-9
        or float(projection["inherited_maximum_ortho_dc"]) != 1.0e-6
        or float(projection["maximum_absolute_adjustment"]) != 0.03
        or projection["adjusted_voxels_per_cube"] != [0, 1]
        or int(projection["maximum_passes"]) != 2
    ):
        raise ValueError("V20 exact-zero-DC projection differs from Addendum B")
    initialization = experiment["initialization_and_normalization"]
    sigma_data = float(initialization["sigma_data"])
    noise = experiment["training_noise"]
    if noise.get("mode") != "log_sigma_data_fraction":
        raise ValueError("V20 did not restore the E3 relative-noise mode")
    if float(noise["median_sigma_data_fraction"]) != 0.6:
        raise ValueError("V20 E3 median sigma-data fraction differs")
    if abs(math.log(0.6 * sigma_data) - P_MEAN) > 1.0e-15:
        raise ValueError("V20 p_mean derivation differs from frozen sigma_data")
    if float(noise["p_mean"]) != P_MEAN or float(noise["p_std"]) != P_STD:
        raise ValueError("V20 stored relative-noise constants differ")
    if noise.get("additional_rng_draws_relative_to_E3") != 0:
        raise ValueError("V20 training noise changes the E3 RNG draw count")
    protocol = experiment["training_protocol"]
    exact = {
        "batch": 6,
        "source_balance_per_batch": {"TNG100": 2, "SIMBA": 2, "Swift": 2},
        "validation_batch": 6,
        "workers": 1,
        "architecture": "ObservableContextUNet",
        "base_channels": 32,
        "parameter_count": 3618465,
        "decoder_upsampling": "nearest",
        "learning_rate": 0.0002,
        "minimum_learning_rate": 0.00002,
        "schedule": "CosineAnnealingLR over exactly 10000 optimizer steps",
        "weight_decay": 0.0001,
        "ema_decay": 0.999,
        "gradient_clip": 1.0,
        "loss": "0.5*unweighted_EDM_MSE+0.5*tail_weighted_EDM_MSE",
        "tail_exponent": 0.25,
        "tail_maximum": 10.0,
        "band_balanced": False,
        "validation_every": 500,
        "training_seed": 144021,
        "validation_seeds": {"TNG100": 99173, "SIMBA": 99174, "Swift": 99175},
        "augmentation": "48 signed cube isometries",
        "selection": "smallest candidate step passing every unchanged V6 field check independently in all three domains; validation loss and Q3/Q4/Q5 are selection_role none",
    }
    if protocol != exact:
        raise ValueError("V20 training protocol differs from frozen E3 inheritance")
    sampler = experiment["sampler"]
    for key, expected in {
        "ensemble_members": 16,
        "steps": 40,
        "sigma_min": 0.002,
        "sigma_max": 40.0,
        "rho": 7.0,
        "integrator": "deterministic Heun probability-flow ODE",
        "sampling_seeds": {"TNG100": 35777, "SIMBA": 36777, "Swift": 37777},
        "additional_rng_draws": 0,
    }.items():
        if sampler.get(key) != expected:
            raise ValueError(f"V20 sampler differs from preregistration: {key}")
    if initialization["mode_counts_64"] != [146, 3596, 25296, 233105]:
        raise ValueError("V20 64^3 initializer mode counts differ")
    if initialization["mode_counts_80"] != [250, 6872, 49928, 454949]:
        raise ValueError("V20 80^3 initializer mode counts differ")
    diagnostics = experiment["mechanism_diagnostics"]
    if diagnostics.get("selection_role") != "none":
        raise ValueError("V20 mechanism diagnostics may not select checkpoints")
    if diagnostics["Q3"] != {
        "step": 10000,
        "per_domain_absolute_delta_q99_999_log10rho_max_dex": 0.10,
        "per_domain_generated_max_log10rho_above_truth_max_dex": 0.30,
        "pass": "both conditions in every development domain",
    }:
        raise ValueError("V20 Q3 differs from preregistration")
    if diagnostics["Q4"]["per_domain_generated_over_truth_mean_delta_squared_max"] != 1.5:
        raise ValueError("V20 Q4 differs from preregistration")


def load_frozen_registry(path: Path, repo: Path) -> dict[str, Any]:
    registry = load_base_registry(path, repo)
    _validate_parent_evidence(registry, repo)
    experiment = registry["e8_gaussianized_marginal_retrain"]
    _validate_exact_experiment(experiment)
    transform = experiment["gaussianization"]
    if sha256_file(Path(transform["path"])) != transform["sha256"]:
        raise ValueError("V20 transform hash mismatch")
    initialization = experiment["initialization_and_normalization"]
    if sha256_file(Path(initialization["measurement_report"])) != initialization[
        "measurement_report_sha256"
    ]:
        raise ValueError("V20 initialization measurement hash mismatch")
    data = experiment["data"]
    derived = experiment["derived_caches"]
    for domain in DOMAIN_KEYS:
        for split in ("train", "validation"):
            for artifact in (
                data[domain][f"{split}_data"],
                data[domain][f"source_{split}_cache"],
                derived[domain][split],
            ):
                if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError(f"V20 {domain} {split} artifact hash mismatch")
    model = data["location_scale_model"]
    if sha256_file(Path(model["path"])) != model["sha256"]:
        raise ValueError("V20 location-scale model hash mismatch")
    for domain in ("SIMBA", "Swift"):
        row = experiment["development_objects"][domain]
        if sha256_file(_resolve(row["path"], repo)) != row["sha256"]:
            raise ValueError(f"V20 {domain} development subset hash mismatch")
    for row in (experiment["unchanged_evaluator"], experiment["unchanged_gate"]):
        if sha256_file(_resolve(row["path"], repo)) != row[
            "preimplementation_sha256"
        ]:
            raise ValueError("V20 unchanged evaluator/gate hash mismatch")
    return registry


def frozen_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    repo = args.repo.resolve()
    registry_path = args.registry.resolve()
    registry = load_frozen_registry(registry_path, repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V20 training requires a clean committed worktree")
    output = args.out.resolve()
    if output.exists():
        raise RuntimeError(f"V20 refuses a pre-existing training output: {output}")
    experiment = registry["e8_gaussianized_marginal_retrain"]
    data = experiment["data"]
    caches = experiment["derived_caches"]
    protocol = experiment["training_protocol"]
    def data_path(domain: str, split: str) -> str:
        return data[domain][f"{split}_data"]["path"]
    def cache_path(domain: str, split: str) -> str:
        return caches[domain][split]["path"]
    return argparse.Namespace(
        tng_train_data=data_path("TNG100", "train"),
        tng_train_cache=cache_path("TNG100", "train"),
        tng_validation_data=data_path("TNG100", "validation"),
        tng_validation_cache=cache_path("TNG100", "validation"),
        simba_train_data=data_path("SIMBA", "train"),
        simba_train_cache=cache_path("SIMBA", "train"),
        simba_validation_data=data_path("SIMBA", "validation"),
        simba_validation_cache=cache_path("SIMBA", "validation"),
        swift_train_data=data_path("Swift", "train"),
        swift_train_cache=cache_path("Swift", "train"),
        swift_validation_data=data_path("Swift", "validation"),
        swift_validation_cache=cache_path("Swift", "validation"),
        out=str(output),
        steps=experiment["steps"],
        candidate_steps=",".join(map(str, experiment["candidate_steps"])),
        batch=protocol["batch"],
        validation_batch=protocol["validation_batch"],
        workers=protocol["workers"],
        base_channels=protocol["base_channels"],
        lr=protocol["learning_rate"],
        min_lr=protocol["minimum_learning_rate"],
        weight_decay=protocol["weight_decay"],
        ema_decay=protocol["ema_decay"],
        gradient_clip=protocol["gradient_clip"],
        edm_p_mean=None,
        edm_p_mean_sigma_data_fraction=0.6,
        edm_p_std=P_STD,
        tail_exponent=protocol["tail_exponent"],
        tail_maximum=protocol["tail_maximum"],
        validation_every=protocol["validation_every"],
        validation_seeds=[protocol["validation_seeds"][name] for name in DOMAIN_KEYS],
        seed=protocol["training_seed"],
        device=args.device,
        smoke_limit=None,
        run_schema=V20_E8_SCHEMA,
        experiment_registry=str(registry_path),
        experiment_registry_sha256=FROZEN_REGISTRY_SHA256,
        code_commit_at_launch=commit,
        worktree_clean_at_launch=clean,
    )


def _validate_checkpoint(
    checkpoint_path: Path, *, step: int, registry: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment = registry["e8_gaussianized_marginal_retrain"]
    initialization = experiment["initialization_and_normalization"]
    if checkpoint.get("schema") != V20_E8_SCHEMA or int(checkpoint.get("step", -1)) != step:
        raise ValueError("V20 checkpoint schema or step mismatch")
    if checkpoint.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("V20 checkpoint registry provenance mismatch")
    if checkpoint.get("worktree_clean_at_launch") is not True:
        raise ValueError("V20 checkpoint was not launched from a clean worktree")
    if float(checkpoint.get("sigma_data", math.nan)) != float(initialization["sigma_data"]):
        raise ValueError("V20 checkpoint sigma_data mismatch")
    if float(checkpoint.get("edm_p_mean", math.nan)) != P_MEAN:
        raise ValueError("V20 checkpoint p_mean mismatch")
    if float(checkpoint.get("edm_p_std", math.nan)) != P_STD:
        raise ValueError("V20 checkpoint p_std mismatch")
    if checkpoint.get("edm_p_mean_mode") != "sigma_data_fraction":
        raise ValueError("V20 checkpoint did not use the E3 relative-noise mode")
    if float(checkpoint.get("edm_p_mean_sigma_data_fraction", math.nan)) != 0.6:
        raise ValueError("V20 checkpoint sigma_data fraction mismatch")
    if checkpoint.get("decoder_upsampling") != "nearest":
        raise ValueError("V20 checkpoint decoder differs from E3")
    if checkpoint.get("denoising_loss") != {
        "coefficients": {"unweighted": 0.5, "tail_weighted": 0.5},
        "band_balanced": False,
    }:
        raise ValueError("V20 checkpoint loss differs from E3")
    commit = checkpoint.get("code_commit_at_launch")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("V20 checkpoint has no valid launch commit")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry = load_frozen_registry(args.registry.resolve(), repo)
    experiment = registry["e8_gaussianized_marginal_retrain"]
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V20 sampling requires a clean committed worktree")
    domain, step = args.domain, int(args.step)
    if domain not in DOMAIN_KEYS or step not in experiment["candidate_steps"]:
        raise ValueError("V20 domain or step is not preregistered")
    checkpoint_path = (
        args.training_root.resolve() / "validation_checkpoints" / f"step_{step:06d}.pt"
    )
    checkpoint, checkpoint_sha = _validate_checkpoint(
        checkpoint_path, step=step, registry=registry
    )
    data = experiment["data"][domain]["validation_data"]
    cache = experiment["derived_caches"][domain]["validation"]
    data_path, cache_path = Path(data["path"]).resolve(), Path(cache["path"]).resolve()
    indices = _indices(experiment["development_objects"][domain], repo)
    sampler = experiment["sampler"]
    seed = int(sampler["sampling_seeds"][domain])
    seed_everything(seed)
    device = torch.device(args.device)
    features = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"], context_std=features["std"],
        decoder_upsampling=decoder_upsampling_for_schema(checkpoint["schema"]),
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    dataset = V14ResidualDataset(data_path, cache_path, False)
    if dataset.cache_schema != V20_GAUSSIANIZED_CACHE_SCHEMA:
        raise ValueError("V20 sampling did not receive a Gaussianized cache")
    if dataset.grid != 64 or dataset.voxel_mpc_h != 0.3125:
        raise ValueError("V20 development cache is not the frozen 64^3 grid")
    sigmas = karras_sigmas(
        int(sampler["steps"]), float(sampler["sigma_min"]),
        float(sampler["sigma_max"]), float(sampler["rho"]), device,
    )
    sigma_first = float(sigmas[0])
    initialization = experiment["initialization_and_normalization"]
    v19 = json.loads(_resolve(registry["parent_evidence"]["v19_registry"], repo).read_text())
    maximum_imaginary = float(
        v19["e7_band_anchored_noise_retrain"]["initialization"][
            "maximum_imaginary_over_real_rms"
        ]
    )
    initializer = PriorMatchedInitializer(
        prior_matched_spectral_std(
            dataset.grid, dataset.voxel_mpc_h, sigma_first,
            initialization["source_balanced_band_mode_variance"], device=device,
        ),
        maximum_imaginary_ratio=maximum_imaginary,
    )
    if not _rng_pairing_self_check(device, initializer, grid=dataset.grid):
        raise RuntimeError("V20 initializer changed the paired random stream")
    transform_row = experiment["gaussianization"]
    transform = json.loads(Path(transform_row["path"]).read_text())
    z_knots = torch.as_tensor(transform["z_knots"], dtype=torch.float32, device=device)
    residual_knots = torch.as_tensor(
        transform["residual_value_knots"], dtype=torch.float32, device=device
    )
    def latent_inverse(value: torch.Tensor) -> torch.Tensor:
        return inverse_gaussianize_torch(value, z_knots, residual_knots)
    generator = torch.Generator(device=device).manual_seed(204602)
    state = generator.get_state().clone()
    _ = latent_inverse(torch.zeros((1, 1, 2, 2, 2), device=device))
    if not torch.equal(state, generator.get_state()):
        raise RuntimeError("V20 latent inverse consumed RNG state")
    write_prior_matched_ensemble(
        model=model, dataset=dataset, checkpoint=checkpoint,
        checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha,
        output=args.out.resolve(), indices=indices,
        ensemble_members=int(sampler["ensemble_members"]),
        sampling_steps=int(sampler["steps"]), sigma_min=float(sampler["sigma_min"]),
        sigma_max=float(sampler["sigma_max"]), rho=float(sampler["rho"]),
        seed=seed, device=device, initializer=initializer, sigma_first=sigma_first,
        latent_inverse=latent_inverse,
        metadata={
            "source_cache": str(cache_path),
            "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"],
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": initialization[
                "measurement_report_sha256"
            ],
            "init_band_edges_h_mpc_json": json.dumps([0.0, 1.0, 3.0, 6.0, "infinity"]),
            "init_mode_counts_json": json.dumps(initialization["mode_counts_64"]),
            "init_band_mode_variances_json": json.dumps(
                initialization["source_balanced_band_mode_variance"]
            ),
            "gaussianization_transform_sha256": transform_row["sha256"],
            "gaussianization_inverse": "frozen piecewise-linear torch inverse before V14 band/location inverse",
            "latent_inverse_additional_rng_draws": 0,
            "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD,
            "sampling_code_commit": commit,
            "worktree_clean_at_sampling": clean,
        },
        progress_label=f"V20 {domain} {step}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    training.add_argument("--registry", type=Path, required=True)
    training.add_argument("--repo", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--device", default="cuda")
    sampling = sub.add_parser("sample")
    sampling.add_argument("--registry", type=Path, required=True)
    sampling.add_argument("--repo", type=Path, required=True)
    sampling.add_argument("--training-root", type=Path, required=True)
    sampling.add_argument("--domain", choices=tuple(DOMAIN_KEYS), required=True)
    sampling.add_argument("--step", type=int, choices=(5000, 10000), required=True)
    sampling.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train(frozen_training_namespace(args))
    else:
        sample(args)


if __name__ == "__main__":
    main()
