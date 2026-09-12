#!/usr/bin/env python
"""Run the sealed V20 Gaussianized-marginal sampler on one-shot Astrid data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch

from hong2021_astrid_seal import verify_astrid_seal
from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_residual_v6 import karras_sigmas, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import (
    V20_E8_SCHEMA,
    V20_GAUSSIANIZED_CACHE_SCHEMA,
    V14ResidualDataset,
    decoder_upsampling_for_schema,
)
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _rng_pairing_self_check, write_prior_matched_ensemble
from hong2021_v18_init import PriorMatchedInitializer, prior_matched_spectral_std, sha256_file
from hong2021_v20_edm import FROZEN_REGISTRY_SHA256, P_MEAN, P_STD, load_frozen_registry
from hong2021_v20_gaussianize import prepare_inference_cache


def _v20_cache_path(source: Path) -> Path:
    source = source.resolve()
    if source.name != "astrid_standardized.h5" or source.parent.name != "model":
        raise ValueError("V20 received an unexpected common Astrid cache path")
    derived = source.parents[2]
    return derived / "hong2021_v20/model/astrid_gaussianized.h5"


def _ensure_gaussianized_cache(
    source: Path, transform: Path, expected_transform_sha256: str,
) -> tuple[Path, str]:
    source = source.resolve()
    transform = transform.resolve()
    output = _v20_cache_path(source)
    source_sha = sha256_file(source)
    transform_sha = sha256_file(transform)
    if transform_sha != expected_transform_sha256:
        raise RuntimeError("sealed V20 Gaussianization transform changed")
    if not output.exists():
        prepare_inference_cache(source, output, transform)
    with h5py.File(output, "r") as handle:
        exact = {
            "schema": V20_GAUSSIANIZED_CACHE_SCHEMA,
            "complete": True,
            "source_standardized_cache": str(source),
            "source_standardized_cache_sha256": source_sha,
            "gaussianization_transform": str(transform),
            "gaussianization_transform_sha256": transform_sha,
            "latent_dc_restored_at_sampling": False,
        }
        for key, expected in exact.items():
            actual = handle.attrs.get(key)
            if isinstance(expected, bool):
                actual = bool(actual)
            else:
                actual = str(actual)
                expected = str(expected)
            if actual != expected:
                raise RuntimeError(f"existing V20 Astrid cache has wrong {key}")
        dataset = handle["standardized_residual"]
        if tuple(dataset.shape) != (27, 1, 80, 80, 80):
            raise RuntimeError("existing V20 Astrid cache has the wrong shape")
    return output, sha256_file(output)


def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    seal_path = args.seal.resolve()
    seal = verify_astrid_seal(
        seal_path, repo=repo, require_committed=True, require_unopened=False
    )
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("sealed V20 Astrid sampling requires a clean worktree")
    one_shot = seal["one_shot"]
    exact = {
        "indices": list(range(27)), "ensemble": 16, "sampling_steps": 40,
        "sigma_min": 0.002, "sigma_max": 40.0, "rho": 7.0, "seed": 28777,
    }
    requested = {
        "indices": [int(value) for value in args.indices.split(",")],
        "ensemble": args.ensemble, "sampling_steps": args.sampling_steps,
        "sigma_min": args.sigma_min, "sigma_max": args.sigma_max,
        "rho": args.rho, "seed": args.seed,
    }
    if requested != exact or any(one_shot.get(key) != value for key, value in exact.items()):
        raise ValueError("Astrid sampling request differs from sealed V20 one-shot")

    registry_path = Path(seal["provenance"]["v20_registry"]["path"]).resolve()
    if sha256_file(registry_path) != FROZEN_REGISTRY_SHA256:
        raise RuntimeError("sealed V20 registry changed before Astrid sampling")
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e8_gaussianized_marginal_retrain"]
    checkpoint_path = Path(seal["artifacts"]["edm"]["path"]).resolve()
    checkpoint_sha = seal["artifacts"]["edm"]["sha256"]
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("sealed V20 checkpoint changed before Astrid sampling")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V20_E8_SCHEMA:
        raise ValueError("sealed Astrid checkpoint is not V20-E8")
    if checkpoint.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("sealed V20 checkpoint registry provenance mismatch")
    if checkpoint.get("edm_p_mean") != P_MEAN or checkpoint.get("edm_p_std") != P_STD:
        raise ValueError("sealed V20 checkpoint noise distribution mismatch")

    transform_row = experiment["gaussianization"]
    transform_path = Path(transform_row["path"]).resolve()
    cache_path, cache_sha = _ensure_gaussianized_cache(
        args.cache, transform_path, transform_row["sha256"]
    )
    device = torch.device(args.device)
    seed_everything(args.seed)
    features = checkpoint["observable_context_features"]
    decoder = decoder_upsampling_for_schema(checkpoint["schema"])
    if decoder != "nearest":
        raise ValueError("sealed V20 checkpoint does not use nearest decoding")
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"], context_std=features["std"],
        decoder_upsampling=decoder,
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    dataset = V14ResidualDataset(args.data.resolve(), cache_path, False)
    if dataset.cache_schema != V20_GAUSSIANIZED_CACHE_SCHEMA:
        raise ValueError("V20 Astrid cache is not Gaussianized")
    if dataset.grid != 80 or dataset.voxel_mpc_h != 0.3125:
        raise ValueError("Astrid cache differs from sealed V20 80^3 mapping")
    sigmas = karras_sigmas(
        args.sampling_steps, args.sigma_min, args.sigma_max, args.rho, device
    )
    sigma_first = float(sigmas[0])
    initialization = experiment["initialization_and_normalization"]
    v19 = json.loads(Path(registry["parent_evidence"]["v19_registry"]).read_text())
    maximum_imaginary = float(
        v19["e7_band_anchored_noise_retrain"]["initialization"]
        ["maximum_imaginary_over_real_rms"]
    )
    initializer = PriorMatchedInitializer(
        prior_matched_spectral_std(
            dataset.grid, dataset.voxel_mpc_h, sigma_first,
            initialization["source_balanced_band_mode_variance"], device=device,
        ),
        maximum_imaginary_ratio=maximum_imaginary,
    )
    if not _rng_pairing_self_check(device, initializer, grid=dataset.grid):
        raise RuntimeError("V20 Astrid initializer changed the sealed random stream")
    transform = json.loads(transform_path.read_text())
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
        raise RuntimeError("V20 Astrid latent inverse consumed RNG state")
    mapping = {
        "astrid_grid": 80,
        "astrid_voxel_mpc_h": 0.3125,
        "astrid_mode_counts": initialization["mode_counts_80"],
        "inference_data_remeasurement": False,
    }
    write_prior_matched_ensemble(
        model=model, dataset=dataset, checkpoint=checkpoint,
        checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha,
        output=args.out.resolve(), indices=requested["indices"],
        ensemble_members=args.ensemble, sampling_steps=args.sampling_steps,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho,
        seed=args.seed, device=device, initializer=initializer,
        sigma_first=sigma_first, latent_inverse=latent_inverse,
        metadata={
            "source_cache": str(cache_path),
            "source_cache_sha256": cache_sha,
            "source_standardized_cache": str(args.cache.resolve()),
            "source_standardized_cache_sha256": sha256_file(args.cache),
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": initialization["measurement_report_sha256"],
            "init_band_edges_h_mpc_json": json.dumps([0.0, 1.0, 3.0, 6.0, "infinity"]),
            "init_mode_counts_json": json.dumps(mapping["astrid_mode_counts"]),
            "init_band_mode_variances_json": json.dumps(
                initialization["source_balanced_band_mode_variance"]
            ),
            "init_inference_mapping_json": json.dumps(mapping, sort_keys=True),
            "gaussianization_transform": str(transform_path),
            "gaussianization_transform_sha256": transform_row["sha256"],
            "gaussianization_inverse": "frozen piecewise-linear torch inverse before V14 band/location inverse",
            "latent_inverse_additional_rng_draws": 0,
            "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD,
            "astrid_seal": str(seal_path),
            "astrid_seal_sha256": sha256_file(seal_path),
            "sampling_code_commit": commit,
            "worktree_clean_at_sampling": clean,
        },
        progress_label="V20 Astrid one-shot",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--ensemble", type=int, required=True)
    parser.add_argument("--sampling-steps", type=int, required=True)
    parser.add_argument("--sigma-min", type=float, required=True)
    parser.add_argument("--sigma-max", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    sample(build_parser().parse_args())
