#!/usr/bin/env python
"""Run only the preregistered V18-E6 prior-matched sampling experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import h5py
import numpy as np
import torch

from hong2021_residual_v6 import karras_sigmas, sample_edm, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import (
    ENSEMBLE_SCHEMA,
    V15_E3_SCHEMA,
    V14ResidualDataset,
    decoder_upsampling_for_schema,
)
from hong2021_v14_multiscale import inverse_standardized_residual
from hong2021_v15_development_gate import canonical_digest
from hong2021_v15_edm import git_state
from hong2021_v18_init import (
    EXPECTED_MODE_COUNTS,
    EXPECTED_MODE_COUNTS_BY_GRID,
    EXPECTED_NON_DC_MODES,
    PriorMatchedInitializer,
    SCHEMA as INIT_SCHEMA,
    prior_matched_spectral_std,
    sha256_file,
)


REGISTRY_SCHEMA = "hong2021-v18-predeclared-prior-matched-initialization-program-v1"
FROZEN_REGISTRY_SHA256 = "e8d8ed88c9a74bf124644edd8d112767e4581c347841d32e58077693d2f3f481"
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _validate_decision(
    path: Path, *, file_sha256: str, digest_sha256: str,
    expected_next: str,
) -> dict[str, Any]:
    if sha256_file(path) != file_sha256:
        raise ValueError(f"frozen decision file hash mismatch: {path}")
    decision = json.loads(path.read_text())
    if canonical_digest(decision) != digest_sha256:
        raise ValueError(f"frozen decision canonical digest mismatch: {path}")
    if decision.get("development_pass") is not False:
        raise ValueError("V18 requires a failed parent development decision")
    if decision.get("next") != expected_next:
        raise ValueError("parent decision did not authorize the V18 audit branch")
    if decision.get("Astrid_used") is not False:
        raise ValueError("parent decision does not prove that Astrid stayed unopened")
    return decision


def _validate_exact_experiment(experiment: dict[str, Any]) -> None:
    if experiment.get("sampling_only") is not True:
        raise ValueError("V18-E6 must remain sampling-only")
    if experiment.get("training_or_checkpoint_update") is not False:
        raise ValueError("V18-E6 may not update training or checkpoints")
    if experiment.get("candidate_steps") != [5000, 10000]:
        raise ValueError("V18 candidate steps differ from the frozen registry")
    if experiment.get("sampling_seeds") != {
        "TNG100": 35777, "SIMBA": 36777, "Swift": 37777,
    }:
        raise ValueError("V18 sampling seeds differ from the frozen E3 lineage")
    sampler = experiment.get("sampler", {})
    exact_sampler = {
        "ensemble_members": 16,
        "steps": 40,
        "sigma_min": 0.002,
        "sigma_max": 40.0,
        "rho": 7.0,
        "integrator": "deterministic Heun probability-flow ODE",
        "sigma_first_step_definition": "float(karras_sigmas(40,0.002,40.0,7.0,device)[0])",
    }
    if sampler != exact_sampler:
        raise ValueError("V18 sampler differs from the frozen E3 sampler")
    init = experiment.get("initialization", {})
    exact_init = {
        "schema": INIT_SCHEMA,
        "additional_rng_draws": 0,
        "fft_transform": "torch.fft.fftn/ifftn",
        "fft_norm": "ortho",
        "fft_compute_dtype": "float64/complex128 outside autocast",
        "output_dtype": "float32",
        "grid": 64,
        "voxel_mpc_h": 0.3125,
        "full_fft_mode_counts": list(EXPECTED_MODE_COUNTS),
        "non_dc_modes": EXPECTED_NON_DC_MODES,
    }
    for key, value in exact_init.items():
        if init.get(key) != value:
            raise ValueError(f"V18 initialization differs from the frozen {key}")
    expected_by_grid = {
        str(grid): list(counts)
        for grid, counts in EXPECTED_MODE_COUNTS_BY_GRID.items()
    }
    if init.get("expected_mode_counts_by_grid") != expected_by_grid:
        raise ValueError("V18 grid-specific mode counts differ from preregistration")
    mapping = experiment.get("inference_initialization_mapping", {})
    if (
        mapping.get("astrid_grid") != 80
        or mapping.get("astrid_voxel_mpc_h") != 0.3125
        or mapping.get("astrid_mode_counts") != list(EXPECTED_MODE_COUNTS_BY_GRID[80])
        or mapping.get("astrid_non_dc_modes") != 80**3 - 1
        or mapping.get("astrid_data_used_in_mapping") is not False
        or mapping.get("inference_data_remeasurement_forbidden") is not True
        or mapping.get("free_parameters_added") != 0
    ):
        raise ValueError("V18 inference-grid mapping differs from preregistration")
    variance = init.get("source_balanced_per_band_mode_variance")
    if variance != [
        1836.1418485297647, 73.04195496322953,
        10.368459581198783, 1.1250001849821358,
    ]:
        raise ValueError("V18 initialization variances differ from preregistration")


def load_frozen_registry(path: Path, repo: Path) -> dict[str, Any]:
    registry_path = path.resolve()
    repo = repo.resolve()
    if sha256_file(registry_path) != FROZEN_REGISTRY_SHA256:
        raise ValueError("V18 registry differs from its pre-implementation frozen hash")
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("invalid V18 development registry")
    if registry.get("status") != "frozen_after_preexecution_inference_grid_mapping_correction_before_v18_e6":
        raise ValueError("V18 registry is not in its frozen pre-E6 state")
    parent = registry["parent_evidence"]
    for name in ("v17", "v15"):
        parent_registry = _resolve(parent[f"{name}_registry"], repo)
        if sha256_file(parent_registry) != parent[f"{name}_registry_sha256"]:
            raise ValueError(f"frozen {name.upper()} registry hash mismatch")
    _validate_decision(
        _resolve(parent["e5_decision"], repo),
        file_sha256=parent["e5_decision_sha256"],
        digest_sha256=parent["e5_decision_digest_sha256"],
        expected_next="stop_unopened_and_obtain_new_independent_design_audit",
    )
    e3 = _validate_decision(
        _resolve(parent["e3_decision"], repo),
        file_sha256=parent["e3_decision_sha256"],
        digest_sha256=parent["e3_decision_digest_sha256"],
        expected_next="stop_without_opening_astrid",
    )
    if e3.get("experiment") != "e3":
        raise ValueError("V18 scientific baseline is not the frozen V15-E3 decision")
    checkpoint_rows = {int(row["step"]): row for row in parent["e3_checkpoints"]}
    if sorted(checkpoint_rows) != [5000, 10000]:
        raise ValueError("V18 registry has the wrong E3 checkpoints")
    for candidate in e3["candidates"]:
        step = int(candidate["step"])
        row = checkpoint_rows[step]
        if Path(candidate["checkpoint"]).resolve() != Path(row["path"]).resolve():
            raise ValueError("V18 E3 checkpoint path differs from the E3 decision")
        if candidate["checkpoint_sha256"] != row["sha256"]:
            raise ValueError("V18 E3 checkpoint hash differs from the E3 decision")
    experiment = registry["e6_prior_matched_initialization"]
    _validate_exact_experiment(experiment)
    init = experiment["initialization"]
    report = _resolve(init["measurement_report"], repo)
    if sha256_file(report) != init["measurement_report_sha256"]:
        raise ValueError("V18 initialization measurement report hash mismatch")
    measurement = json.loads(report.read_text())
    if measurement.get("source_balanced_per_band_mode_variance") != init[
        "source_balanced_per_band_mode_variance"
    ]:
        raise ValueError("V18 measurement report differs from frozen variances")
    for row in experiment["training_cache_sources"].values():
        if sha256_file(_resolve(row["path"], repo)) != row["sha256"]:
            raise ValueError("V18 training-cache hash mismatch")
    for specification in (
        experiment["unchanged_evaluator"], experiment["unchanged_gate"]
    ):
        if sha256_file(_resolve(specification["path"], repo)) != specification[
            "preimplementation_sha256"
        ]:
            raise ValueError("V18 unchanged evaluator/gate file was modified")
    return registry


def _indices(specification: list[int] | dict[str, str], repo: Path) -> list[int]:
    if isinstance(specification, list):
        values = [int(value) for value in specification]
    else:
        path = _resolve(specification["path"], repo)
        if sha256_file(path) != specification["sha256"]:
            raise ValueError("V18 development subset file hash mismatch")
        values = [int(value) for value in json.loads(path.read_text())["indices"]]
    if len(values) != 16 or len(set(values)) != 16 or min(values) < 0:
        raise ValueError("V18 development subset must contain 16 unique indices")
    return values


def _rng_pairing_self_check(
    device: torch.device, initializer: PriorMatchedInitializer, *, grid: int = 64
) -> bool:
    first = torch.Generator(device=device).manual_seed(918002)
    second = torch.Generator(device=device).manual_seed(918002)
    shape = (1, 1, grid, grid, grid)
    noise_first = torch.randn(shape, device=device, generator=first)
    noise_second = torch.randn(shape, device=device, generator=second)
    if not torch.equal(noise_first, noise_second):
        return False
    state_before = first.get_state().clone()
    initializer(noise_first)
    return bool(torch.equal(first.get_state(), state_before)) and bool(
        torch.equal(first.get_state(), second.get_state())
    )


@torch.inference_mode()
def write_prior_matched_ensemble(
    *,
    model: torch.nn.Module,
    dataset: V14ResidualDataset,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    output: Path,
    indices: list[int],
    ensemble_members: int,
    sampling_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    seed: int,
    device: torch.device,
    initializer: PriorMatchedInitializer,
    sigma_first: float,
    metadata: Mapping[str, Any],
    progress_label: str,
    latent_inverse: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Write one V18 ensemble using the sole shared prior-matched sampler path."""
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("V18 sample indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("V18 sample indices are outside the dataset")
    if ensemble_members <= 0:
        raise ValueError("V18 ensemble size must be positive")
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V18 ensemble: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        with h5py.File(partial, "w") as handle:
            shape = (
                len(indices), ensemble_members, 1,
                dataset.grid, dataset.grid, dataset.grid,
            )
            sample_ds = handle.create_dataset(
                "sample", shape=shape, dtype="f4",
                chunks=(1, 1, 1, dataset.grid, dataset.grid, dataset.grid),
                compression="lzf",
            )
            field_shape = (
                len(indices), 1, dataset.grid, dataset.grid, dataset.grid,
            )
            mean_ds = handle.create_dataset(
                "conditional_mean", shape=field_shape, dtype="f4", compression="lzf",
            )
            truth_ds = handle.create_dataset(
                "truth", shape=field_shape, dtype="f4", compression="lzf",
            )
            handle.create_dataset("source_index", data=np.asarray(indices, dtype=np.int64))
            location_ds = handle.create_dataset(
                "predicted_residual_dc", shape=(len(indices),), dtype="f4"
            )
            scale_ds = handle.create_dataset(
                "predicted_band_scales", shape=(len(indices), 4), dtype="f4"
            )
            for output_index, data_index in enumerate(indices):
                condition, _, corrected_mean, truth = dataset[data_index]
                location, scales = dataset.predicted_location_scales(data_index)
                condition_batch = condition[None].to(device).expand(
                    ensemble_members, -1, -1, -1, -1
                )
                standardized = sample_edm(
                    model, condition_batch, generator,
                    sampling_steps, sigma_min, sigma_max, rho,
                    float(checkpoint["sigma_data"]), init_transform=initializer,
                )
                standardized -= standardized.mean(dim=(-3, -2, -1), keepdim=True)
                if latent_inverse is not None:
                    standardized = latent_inverse(standardized)
                standardized_numpy = standardized[:, 0].float().cpu().numpy()
                physical = np.stack([
                    inverse_standardized_residual(
                        value, predicted_location=location,
                        predicted_scales=scales,
                        voxel_mpc_h=dataset.voxel_mpc_h,
                    )
                    for value in standardized_numpy
                ]).astype(np.float32)
                mean_with_location = corrected_mean.numpy() + np.float32(location)
                sample_ds[output_index, :, 0] = corrected_mean.numpy()[0] + physical
                mean_ds[output_index] = mean_with_location
                truth_ds[output_index] = truth.numpy()
                location_ds[output_index] = location
                scale_ds[output_index] = scales
                print(
                    f"[sample] {progress_label} {output_index + 1}/{len(indices)}",
                    flush=True,
                )
            attributes = {
                "schema": ENSEMBLE_SCHEMA,
                "method": "edm",
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": int(checkpoint["step"]),
                "checkpoint_schema": str(checkpoint["schema"]),
                "sigma_data": float(checkpoint["sigma_data"]),
                "sampling_steps": sampling_steps,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
                "seed": seed,
                "diagnostic_k_h_mpc": 1.0,
                "location_scale_uses_target": False,
                "init_schema": INIT_SCHEMA,
                "init_sigma_nominal": sigma_max,
                "init_sigma_effective_first_step": sigma_first,
                "init_fft_norm": "ortho",
                "init_fft_compute_dtype": "float64/complex128",
                "init_output_dtype": "float32",
                "init_additional_rng_draws": 0,
                "init_rng_pairing_self_check": True,
                "init_maximum_imaginary_over_real_rms": (
                    initializer.maximum_observed_imaginary_ratio
                ),
                "complete": True,
                **dict(metadata),
            }
            handle.attrs.update(attributes)
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    print(f"[out] {output}", flush=True)


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry_path = args.registry.resolve()
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e6_prior_matched_initialization"]
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V18 sampling requires a clean committed worktree")
    domain = args.domain
    step = int(args.step)
    if domain not in DOMAIN_KEYS or step not in (5000, 10000):
        raise ValueError("V18 domain or step is not preregistered")
    checkpoint_row = next(
        row for row in registry["parent_evidence"]["e3_checkpoints"]
        if int(row["step"]) == step
    )
    checkpoint_path = Path(checkpoint_row["path"]).resolve()
    if sha256_file(checkpoint_path) != checkpoint_row["sha256"]:
        raise ValueError("V18 E3 checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V15_E3_SCHEMA or int(checkpoint.get("step", -1)) != step:
        raise ValueError("V18 received the wrong E3 checkpoint")
    if checkpoint.get("experiment_registry_sha256") != registry["parent_evidence"][
        "v15_registry_sha256"
    ]:
        raise ValueError("V18 E3 checkpoint registry provenance mismatch")
    if checkpoint.get("worktree_clean_at_launch") is not True:
        raise ValueError("V18 E3 checkpoint was not launched from a clean worktree")
    inputs = experiment["validation_inputs"][domain]
    data_path = Path(inputs["data_path"]).resolve()
    cache_path = Path(inputs["cache_path"]).resolve()
    if sha256_file(data_path) != inputs["data_sha256"]:
        raise ValueError("V18 validation data SHA256 mismatch")
    if sha256_file(cache_path) != inputs["cache_sha256"]:
        raise ValueError("V18 validation cache SHA256 mismatch")
    indices = _indices(experiment["development_objects"][domain], repo)
    seed = int(experiment["sampling_seeds"][domain])
    sampler = experiment["sampler"]
    init = experiment["initialization"]
    seed_everything(seed)
    device = torch.device(args.device)
    feature_fit = checkpoint["observable_context_features"]
    decoder = decoder_upsampling_for_schema(checkpoint["schema"])
    if decoder != "nearest":
        raise ValueError("V18 E3 checkpoint does not use the nearest decoder")
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=feature_fit["mean"], context_std=feature_fit["std"],
        decoder_upsampling=decoder,
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    dataset = V14ResidualDataset(data_path, cache_path, False)
    if dataset.grid != int(init["grid"]) or dataset.voxel_mpc_h != float(init["voxel_mpc_h"]):
        raise ValueError("V18 dataset grid differs from initialization grid")
    sigmas = karras_sigmas(
        int(sampler["steps"]), float(sampler["sigma_min"]),
        float(sampler["sigma_max"]), float(sampler["rho"]), device,
    )
    sigma_first = float(sigmas[0])
    spectral_std = prior_matched_spectral_std(
        dataset.grid, dataset.voxel_mpc_h, sigma_first,
        init["source_balanced_per_band_mode_variance"], device=device,
    )
    initializer = PriorMatchedInitializer(
        spectral_std,
        maximum_imaginary_ratio=float(init["maximum_imaginary_over_real_rms"]),
    )
    rng_pairing = _rng_pairing_self_check(device, initializer)
    if not rng_pairing:
        raise RuntimeError("V18 initialization consumed RNG or changed the frozen draw")
    write_prior_matched_ensemble(
        model=model,
        dataset=dataset,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_row["sha256"],
        output=args.out.resolve(),
        indices=indices,
        ensemble_members=int(sampler["ensemble_members"]),
        sampling_steps=int(sampler["steps"]),
        sigma_min=float(sampler["sigma_min"]),
        sigma_max=float(sampler["sigma_max"]),
        rho=float(sampler["rho"]),
        seed=seed,
        device=device,
        initializer=initializer,
        sigma_first=sigma_first,
        metadata={
            "source_cache": str(cache_path),
            "source_cache_sha256": inputs["cache_sha256"],
            "source_data_sha256": inputs["data_sha256"],
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": init["measurement_report_sha256"],
            "init_band_edges_h_mpc_json": json.dumps(init["band_edges_h_mpc"]),
            "init_mode_counts_json": json.dumps(init["full_fft_mode_counts"]),
            "init_band_mode_variances_json": json.dumps(
                init["source_balanced_per_band_mode_variance"]
            ),
            "sampling_code_commit": commit,
            "worktree_clean_at_sampling": clean,
        },
        progress_label=f"V18 {domain} {step}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--domain", choices=tuple(DOMAIN_KEYS), required=True)
    parser.add_argument("--step", type=int, choices=(5000, 10000), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    sample(build_parser().parse_args())


if __name__ == "__main__":
    main()
