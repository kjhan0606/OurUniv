#!/usr/bin/env python
"""Run the sealed V19 prior-matched sampler on one-shot Astrid data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hong2021_astrid_seal import verify_astrid_seal
from hong2021_residual_v6 import karras_sigmas, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v14_edm import V19_E7_SCHEMA, V14ResidualDataset, decoder_upsampling_for_schema
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _rng_pairing_self_check, write_prior_matched_ensemble
from hong2021_v18_init import PriorMatchedInitializer, prior_matched_spectral_std, sha256_file
from hong2021_v19_edm import FROZEN_REGISTRY_SHA256, P_MEAN, P_STD, load_frozen_registry


def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    seal_path = args.seal.resolve()
    seal = verify_astrid_seal(
        seal_path, repo=repo, require_committed=True, require_unopened=False
    )
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("sealed V19 Astrid sampling requires a clean worktree")
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
        raise ValueError("Astrid sampling request differs from sealed V19 one-shot")
    registry_path = Path(seal["provenance"]["v19_registry"]["path"]).resolve()
    if sha256_file(registry_path) != FROZEN_REGISTRY_SHA256:
        raise RuntimeError("sealed V19 registry changed before Astrid sampling")
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e7_band_anchored_noise_retrain"]
    checkpoint_path = Path(seal["artifacts"]["edm"]["path"]).resolve()
    checkpoint_sha = seal["artifacts"]["edm"]["sha256"]
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("sealed V19 checkpoint changed before Astrid sampling")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V19_E7_SCHEMA:
        raise ValueError("sealed Astrid checkpoint is not V19-E7")
    if checkpoint.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("sealed V19 checkpoint registry provenance mismatch")
    if checkpoint.get("edm_p_mean") != P_MEAN or checkpoint.get("edm_p_std") != P_STD:
        raise ValueError("sealed V19 checkpoint noise distribution mismatch")
    device = torch.device(args.device)
    seed_everything(args.seed)
    features = checkpoint["observable_context_features"]
    decoder = decoder_upsampling_for_schema(checkpoint["schema"])
    if decoder != "nearest":
        raise ValueError("sealed V19 checkpoint does not use nearest decoding")
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"], context_std=features["std"],
        decoder_upsampling=decoder,
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    dataset = V14ResidualDataset(args.data.resolve(), args.cache.resolve(), False)
    if dataset.grid != 80 or dataset.voxel_mpc_h != 0.3125:
        raise ValueError("Astrid cache differs from sealed V19 80^3 mapping")
    sigmas = karras_sigmas(
        args.sampling_steps, args.sigma_min, args.sigma_max, args.rho, device
    )
    sigma_first = float(sigmas[0])
    init = experiment["initialization"]
    initializer = PriorMatchedInitializer(
        prior_matched_spectral_std(
            dataset.grid, dataset.voxel_mpc_h, sigma_first,
            init["source_balanced_per_band_mode_variance"], device=device,
        ),
        maximum_imaginary_ratio=float(init["maximum_imaginary_over_real_rms"]),
    )
    if not _rng_pairing_self_check(device, initializer, grid=dataset.grid):
        raise RuntimeError("V19 Astrid initializer changed the sealed random stream")
    mapping = {
        "astrid_grid": 80,
        "astrid_voxel_mpc_h": 0.3125,
        "astrid_mode_counts": init["expected_mode_counts_by_grid"]["80"],
        "inference_data_remeasurement": False,
    }
    write_prior_matched_ensemble(
        model=model, dataset=dataset, checkpoint=checkpoint,
        checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha,
        output=args.out.resolve(), indices=requested["indices"],
        ensemble_members=args.ensemble, sampling_steps=args.sampling_steps,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho,
        seed=args.seed, device=device, initializer=initializer,
        sigma_first=sigma_first,
        metadata={
            "source_cache": str(args.cache.resolve()),
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": init["measurement_report_sha256"],
            "init_band_edges_h_mpc_json": json.dumps([0.0, 1.0, 3.0, 6.0, "infinity"]),
            "init_mode_counts_json": json.dumps(mapping["astrid_mode_counts"]),
            "init_band_mode_variances_json": json.dumps(init["source_balanced_per_band_mode_variance"]),
            "init_inference_mapping_json": json.dumps(mapping, sort_keys=True),
            "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD,
            "astrid_seal": str(seal_path),
            "astrid_seal_sha256": sha256_file(seal_path),
            "sampling_code_commit": commit,
            "worktree_clean_at_sampling": clean,
        },
        progress_label="V19 Astrid one-shot",
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
