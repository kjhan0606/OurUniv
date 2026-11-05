#!/usr/bin/env python
"""Launch and sample only the frozen V19-E7 band-anchored-noise experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from hong2021_residual_v6 import karras_sigmas, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import (
    V19_E7_SCHEMA,
    V14ResidualDataset,
    decoder_upsampling_for_schema,
    train,
)
from hong2021_v15_development_gate import canonical_digest
from hong2021_v15_edm import git_state
from hong2021_v18_edm import (
    _indices,
    _rng_pairing_self_check,
    load_frozen_registry as load_v18_registry,
    write_prior_matched_ensemble,
)
from hong2021_v18_init import (
    PriorMatchedInitializer,
    prior_matched_spectral_std,
    sha256_file,
)


REGISTRY_SCHEMA = "hong2021-v19-predeclared-band-anchored-noise-program-v1"
FROZEN_REGISTRY_SHA256 = "4a2b1099b696dbf580db8e8b8cd7dfe9dd6af45e18240316507c8136f5bc5011"
P_MEAN = 1.2720503130999161
P_STD = 1.8124552265001808
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _validate_parent_decision(parent: dict[str, Any]) -> dict[str, Any]:
    path = Path(parent["e6_decision"]).resolve()
    if sha256_file(path) != parent["e6_decision_sha256"]:
        raise ValueError("V19 parent E6 decision file hash mismatch")
    decision = json.loads(path.read_text())
    if canonical_digest(decision) != parent["e6_decision_digest_sha256"]:
        raise ValueError("V19 parent E6 decision canonical digest mismatch")
    if decision.get("development_pass") is not False:
        raise ValueError("V19 requires the frozen failed E6 decision")
    if decision.get("next") != parent["e6_next"]:
        raise ValueError("V19 parent E6 decision did not authorize an audit")
    if decision.get("Astrid_used") is not False:
        raise ValueError("V19 parent does not prove Astrid remained unopened")
    return decision


def derive_band_anchored_noise(
    variances: list[float], *, fraction: float = 0.6, component_std: float = 1.2
) -> tuple[float, float, list[float]]:
    if len(variances) != 4 or any(not math.isfinite(v) or v <= 0 for v in variances):
        raise ValueError("V19 requires four finite positive band variances")
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("V19 median RMS fraction must be finite and positive")
    if not math.isfinite(component_std) or component_std <= 0:
        raise ValueError("V19 component log standard deviation must be positive")
    component_means = [math.log(fraction * math.sqrt(value)) for value in variances]
    p_mean = sum(component_means) / len(component_means)
    between = sum((value - p_mean) ** 2 for value in component_means) / len(
        component_means
    )
    return p_mean, math.sqrt(component_std**2 + between), component_means


def _validate_exact_experiment(experiment: dict[str, Any]) -> None:
    if experiment.get("training_from_scratch") is not True:
        raise ValueError("V19-E7 must train from scratch")
    if experiment.get("continuation_checkpoint") is not None:
        raise ValueError("V19-E7 may not continue a checkpoint")
    if experiment.get("steps") != 10000 or experiment.get("candidate_steps") != [5000, 10000]:
        raise ValueError("V19-E7 training budget differs from preregistration")
    if experiment.get("checkpoint_schema") != V19_E7_SCHEMA:
        raise ValueError("V19-E7 checkpoint schema differs from preregistration")
    noise = experiment["training_noise"]
    derived_mean, derived_std, components = derive_band_anchored_noise(
        noise["source_balanced_per_band_mode_variance"],
        fraction=float(noise["inherited_median_rms_fraction"]),
        component_std=float(noise["inherited_component_log_standard_deviation"]),
    )
    for name, actual, expected in (
        ("p_mean", derived_mean, P_MEAN),
        ("p_std", derived_std, P_STD),
        ("stored p_mean", float(noise["p_mean"]), P_MEAN),
        ("stored p_std", float(noise["p_std"]), P_STD),
    ):
        if abs(actual - expected) > 1.0e-12 * abs(expected):
            raise ValueError(f"V19 {name} differs from the frozen derivation")
    if any(abs(a - b) > 1.0e-12 for a, b in zip(
        components, noise["component_log_medians"], strict=True
    )):
        raise ValueError("V19 component log medians differ from the frozen derivation")
    if noise.get("band_weights") != [0.25, 0.25, 0.25, 0.25]:
        raise ValueError("V19 band weights differ from the frozen equal mixture")
    if noise.get("additional_rng_draws_relative_to_e3") != 0:
        raise ValueError("V19 noise distribution changes the E3 RNG draw count")
    if noise.get("parameter_selection_uses_e3_or_e6_outcomes") is not False:
        raise ValueError("V19 noise parameters are not target-free")
    if abs(float(noise["median_sigma"]) - math.exp(P_MEAN)) > 1.0e-12:
        raise ValueError("V19 stored median sigma differs from p_mean")
    survival = lambda threshold: 0.5 * math.erfc(
        (math.log(threshold) - P_MEAN) / (P_STD * math.sqrt(2.0))
    )
    exact_probabilities = {
        "expected_fraction_sigma_gt_20": survival(20.0),
        "expected_fraction_sigma_gt_40": survival(40.0),
        "expected_fraction_sigma_20_to_40": survival(20.0) - survival(40.0),
        "expected_fraction_sigma_0_5_to_2": survival(0.5) - survival(2.0),
    }
    for key, expected in exact_probabilities.items():
        if abs(float(noise[key]) - expected) > 1.0e-12:
            raise ValueError(f"V19 stored analytic probability differs: {key}")
    protocol = experiment["training_protocol"]
    exact_protocol = {
        "batch": 6,
        "source_balance_per_batch": {"TNG100": 2, "SIMBA": 2, "Swift": 2},
        "validation_batch": 6,
        "workers": 1,
        "base_channels": 32,
        "learning_rate": 0.0002,
        "minimum_learning_rate": 0.00002,
        "schedule": "CosineAnnealingLR over exactly 10000 optimizer steps",
        "weight_decay": 0.0001,
        "ema_decay": 0.999,
        "gradient_clip": 1.0,
        "tail_exponent": 0.25,
        "tail_maximum": 10.0,
        "loss": "0.5*unweighted_EDM_MSE+0.5*tail_weighted_EDM_MSE",
        "validation_every": 500,
        "training_seed": 144021,
        "validation_seeds": {"TNG100": 99173, "SIMBA": 99174, "Swift": 99175},
        "augmentation": "48 signed cube isometries",
        "decoder_upsampling": "nearest",
        "decoder_align_corners": None,
        "selection": "gate-only at the fixed candidate steps; validation losses are report-only because their measure changes with the training-noise distribution",
    }
    if protocol != exact_protocol:
        raise ValueError("V19 training protocol differs from frozen E3 inheritance")
    sampler = experiment["sampler"]
    if sampler != {
        "ensemble_members": 16,
        "steps": 40,
        "sigma_min": 0.002,
        "sigma_max": 40.0,
        "rho": 7.0,
        "integrator": "deterministic Heun probability-flow ODE",
    }:
        raise ValueError("V19 sampler differs from frozen V18 inheritance")
    if experiment["sampling_seeds"] != {
        "TNG100": 35777, "SIMBA": 36777, "Swift": 37777,
    }:
        raise ValueError("V19 sampling seeds differ from frozen E3/E6 pairing")
    init = experiment["initialization"]
    if init.get("source_balanced_per_band_mode_variance") != noise[
        "source_balanced_per_band_mode_variance"
    ]:
        raise ValueError("V19 training and initialization band variances differ")
    if init.get("expected_mode_counts_by_grid") != {
        "64": [146, 3596, 25296, 233105],
        "80": [250, 6872, 49928, 454949],
    }:
        raise ValueError("V19 initializer mode counts differ from preregistration")
    q1 = experiment["mechanism_diagnostics"]["Q1"]
    if q1.get("karras_ladder_indices") != [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 17, 21, 26, 31]:
        raise ValueError("V19 Q1 spectroscopy ladder differs from preregistration")
    if q1.get("required_maximum_g_train_minus_g") != {
        "index_0_sigma_40": 0.045,
        "index_4_sigma_22_7": 0.03,
        "index_9_sigma_10_4": 0.03,
    }:
        raise ValueError("V19 Q1 thresholds differ from preregistration")
    q2 = experiment["mechanism_diagnostics"]["Q2"]
    if float(q2.get("required_fraction_of_ceiling", -1)) != 0.85:
        raise ValueError("V19 Q2 ceiling fraction differs from preregistration")


def load_frozen_registry(path: Path, repo: Path) -> dict[str, Any]:
    path = path.resolve()
    repo = repo.resolve()
    if sha256_file(path) != FROZEN_REGISTRY_SHA256:
        raise ValueError("V19 registry differs from its pre-implementation frozen hash")
    registry = json.loads(path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("invalid V19 development registry")
    if registry.get("status") != "frozen_before_v19_e7_implementation_or_execution":
        raise ValueError("V19 registry is not in its frozen pre-E7 state")
    parent = registry["parent_evidence"]
    v18_path = _resolve(parent["v18_registry"], repo)
    if sha256_file(v18_path) != parent["v18_registry_sha256"]:
        raise ValueError("V19 parent V18 registry hash mismatch")
    load_v18_registry(v18_path, repo)
    _validate_parent_decision(parent)
    audit = registry["independent_audit"]
    audit_path = Path(audit["record"]).resolve()
    if sha256_file(audit_path) != audit["record_sha256"]:
        raise ValueError("V19 independent audit record hash mismatch")
    experiment = registry["e7_band_anchored_noise_retrain"]
    _validate_exact_experiment(experiment)
    for domain in DOMAIN_KEYS:
        row = experiment["data"][domain]
        for key in ("train_data", "train_cache", "validation_data", "validation_cache"):
            artifact = Path(row[key]).resolve()
            if sha256_file(artifact) != row[f"{key}_sha256"]:
                raise ValueError(f"V19 {domain} {key} hash mismatch")
    for domain in ("SIMBA", "Swift"):
        specification = experiment["development_objects"][domain]
        subset = _resolve(specification["path"], repo)
        if sha256_file(subset) != specification["sha256"]:
            raise ValueError(f"V19 {domain} development subset hash mismatch")
    for specification in (experiment["unchanged_evaluator"], experiment["unchanged_gate"]):
        if sha256_file(_resolve(specification["path"], repo)) != specification[
            "preimplementation_sha256"
        ]:
            raise ValueError("V19 unchanged evaluator/gate hash mismatch")
    init = experiment["initialization"]
    if sha256_file(Path(init["measurement_report"])) != init["measurement_report_sha256"]:
        raise ValueError("V19 initialization measurement report hash mismatch")
    return registry


def frozen_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    repo = args.repo.resolve()
    registry_path = args.registry.resolve()
    registry = load_frozen_registry(registry_path, repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V19 training requires a clean committed worktree")
    output = args.out.resolve()
    if output.exists():
        raise RuntimeError(f"V19 refuses a pre-existing training output: {output}")
    experiment = registry["e7_band_anchored_noise_retrain"]
    data = experiment["data"]
    protocol = experiment["training_protocol"]
    return argparse.Namespace(
        tng_train_data=data["TNG100"]["train_data"],
        tng_train_cache=data["TNG100"]["train_cache"],
        tng_validation_data=data["TNG100"]["validation_data"],
        tng_validation_cache=data["TNG100"]["validation_cache"],
        simba_train_data=data["SIMBA"]["train_data"],
        simba_train_cache=data["SIMBA"]["train_cache"],
        simba_validation_data=data["SIMBA"]["validation_data"],
        simba_validation_cache=data["SIMBA"]["validation_cache"],
        swift_train_data=data["Swift"]["train_data"],
        swift_train_cache=data["Swift"]["train_cache"],
        swift_validation_data=data["Swift"]["validation_data"],
        swift_validation_cache=data["Swift"]["validation_cache"],
        out=str(output),
        steps=experiment["steps"],
        candidate_steps=",".join(str(value) for value in experiment["candidate_steps"]),
        batch=protocol["batch"],
        validation_batch=protocol["validation_batch"],
        workers=protocol["workers"],
        base_channels=protocol["base_channels"],
        lr=protocol["learning_rate"],
        min_lr=protocol["minimum_learning_rate"],
        weight_decay=protocol["weight_decay"],
        ema_decay=protocol["ema_decay"],
        gradient_clip=protocol["gradient_clip"],
        edm_p_mean=P_MEAN,
        edm_p_mean_sigma_data_fraction=None,
        edm_p_std=P_STD,
        tail_exponent=protocol["tail_exponent"],
        tail_maximum=protocol["tail_maximum"],
        validation_every=protocol["validation_every"],
        validation_seeds=[protocol["validation_seeds"][name] for name in DOMAIN_KEYS],
        seed=protocol["training_seed"],
        device=args.device,
        smoke_limit=None,
        run_schema=V19_E7_SCHEMA,
        experiment_registry=str(registry_path),
        experiment_registry_sha256=FROZEN_REGISTRY_SHA256,
        code_commit_at_launch=commit,
        worktree_clean_at_launch=clean,
    )


def _validate_checkpoint(
    checkpoint_path: Path, *, step: int, registry: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V19_E7_SCHEMA or int(checkpoint.get("step", -1)) != step:
        raise ValueError("V19 checkpoint schema or step mismatch")
    if checkpoint.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("V19 checkpoint registry provenance mismatch")
    if checkpoint.get("worktree_clean_at_launch") is not True:
        raise ValueError("V19 checkpoint was not launched from a clean worktree")
    if float(checkpoint.get("edm_p_mean", math.nan)) != P_MEAN:
        raise ValueError("V19 checkpoint p_mean mismatch")
    if float(checkpoint.get("edm_p_std", math.nan)) != P_STD:
        raise ValueError("V19 checkpoint p_std mismatch")
    if checkpoint.get("edm_p_mean_mode") != "fixed_log_sigma":
        raise ValueError("V19 checkpoint did not use the fixed band-anchored p_mean")
    if checkpoint.get("decoder_upsampling") != "nearest":
        raise ValueError("V19 checkpoint decoder differs from E3")
    if float(checkpoint.get("sigma_data", math.nan)) != float(
        registry["parent_evidence"]["e3_sigma_data"]
    ):
        raise ValueError("V19 checkpoint sigma_data differs from E3")
    tail = checkpoint.get("tail_weight_fit", {})
    if float(tail.get("inverse_probability_exponent", -1)) != 0.25:
        raise ValueError("V19 checkpoint tail exponent differs from E3")
    if float(tail.get("pre_normalization_maximum", -1)) != 10.0:
        raise ValueError("V19 checkpoint tail maximum differs from E3")
    if checkpoint.get("denoising_loss") != {
        "coefficients": {"unweighted": 0.5, "tail_weighted": 0.5},
        "band_balanced": False,
    }:
        raise ValueError("V19 checkpoint loss differs from E3")
    commit = checkpoint.get("code_commit_at_launch")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("V19 checkpoint has no valid launch commit")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry = load_frozen_registry(args.registry.resolve(), repo)
    experiment = registry["e7_band_anchored_noise_retrain"]
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V19 sampling requires a clean committed worktree")
    domain, step = args.domain, int(args.step)
    if domain not in DOMAIN_KEYS or step not in experiment["candidate_steps"]:
        raise ValueError("V19 domain or step is not preregistered")
    checkpoint_path = args.training_root.resolve() / "validation_checkpoints" / f"step_{step:06d}.pt"
    checkpoint, checkpoint_sha = _validate_checkpoint(
        checkpoint_path, step=step, registry=registry
    )
    row = experiment["data"][domain]
    data_path = Path(row["validation_data"]).resolve()
    cache_path = Path(row["validation_cache"]).resolve()
    indices = _indices(experiment["development_objects"][domain], repo)
    seed = int(experiment["sampling_seeds"][domain])
    sampler = experiment["sampler"]
    init = experiment["initialization"]
    seed_everything(seed)
    device = torch.device(args.device)
    features = checkpoint["observable_context_features"]
    decoder = decoder_upsampling_for_schema(checkpoint["schema"])
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"], context_std=features["std"],
        decoder_upsampling=decoder,
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    dataset = V14ResidualDataset(data_path, cache_path, False)
    if dataset.grid != 64 or dataset.voxel_mpc_h != 0.3125:
        raise ValueError("V19 development cache is not the frozen 64^3 grid")
    sigmas = karras_sigmas(
        int(sampler["steps"]), float(sampler["sigma_min"]),
        float(sampler["sigma_max"]), float(sampler["rho"]), device,
    )
    sigma_first = float(sigmas[0])
    initializer = PriorMatchedInitializer(
        prior_matched_spectral_std(
            dataset.grid, dataset.voxel_mpc_h, sigma_first,
            init["source_balanced_per_band_mode_variance"], device=device,
        ),
        maximum_imaginary_ratio=float(init["maximum_imaginary_over_real_rms"]),
    )
    if not _rng_pairing_self_check(device, initializer, grid=dataset.grid):
        raise RuntimeError("V19 initializer changed the paired random stream")
    write_prior_matched_ensemble(
        model=model, dataset=dataset, checkpoint=checkpoint,
        checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha,
        output=args.out.resolve(), indices=indices,
        ensemble_members=int(sampler["ensemble_members"]),
        sampling_steps=int(sampler["steps"]),
        sigma_min=float(sampler["sigma_min"]),
        sigma_max=float(sampler["sigma_max"]), rho=float(sampler["rho"]),
        seed=seed, device=device, initializer=initializer, sigma_first=sigma_first,
        metadata={
            "source_cache": str(cache_path),
            "source_cache_sha256": row["validation_cache_sha256"],
            "source_data_sha256": row["validation_data_sha256"],
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": init["measurement_report_sha256"],
            "init_band_edges_h_mpc_json": json.dumps([0.0, 1.0, 3.0, 6.0, "infinity"]),
            "init_mode_counts_json": json.dumps(init["expected_mode_counts_by_grid"]["64"]),
            "init_band_mode_variances_json": json.dumps(init["source_balanced_per_band_mode_variance"]),
            "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD,
            "sampling_code_commit": commit,
            "worktree_clean_at_sampling": clean,
        },
        progress_label=f"V19 {domain} {step}",
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
