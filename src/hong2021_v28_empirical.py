#!/usr/bin/env python
"""Train-only observation-matched full-cube empirical residual control."""
from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_diffusion import radial_geometry
from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_train import apply_input_preprocessing
from hong2021_v14_multiscale import inverse_standardized_residual
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v21_conditional_affine import invert_profile_torch
from hong2021_v21_edm import ARTIFACT_SHA256
from hong2021_v26 import CACHE_KEYS
from hong2021_v27 import load_frozen_program as load_v27_program


REGISTRY_SCHEMA = (
    "hong2021-v28-train-only-empirical-joint-residual-control-development-program-v1"
)
REGISTRY_SHA256 = "7d51039e3acf3a9297ce70e9f9db19f17b9290b9501fdb72ac047422220ac3a6"
DESIGN_AUDIT_SHA256 = "b2c1fc773cc658b6b912e618c00dbefe0c22bb94d12f70237b7cd338e705cbce"
PARENT_AUDIT_SHA256 = "00ffa15d05816501c532a63821c698d2ebbe29e5e22c54974ee5fcc10da38105"
ENSEMBLE_SCHEMA = "hong2021-v28-train-only-empirical-joint-residual-ensemble-v1"
PREFLIGHT_SCHEMA = "hong2021-v28-empirical-joint-control-hard-preflight-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
DONOR_COUNTS = {"TNG100": 432, "SIMBA": 202, "Swift": 409}
LOCAL_GRID = 8
GLOBAL_PREFILTER = 64
ENSEMBLE_MEMBERS = 16


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs from V28 freeze")
    return json.loads(path.read_text())


def load_frozen_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    registry = _verified_json(path, REGISTRY_SHA256, "V28 registry")
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status")
        != "frozen_before_implementation_sampling_or_development_evaluation"
    ):
        raise ValueError("V28 registry schema or status differs")
    design = registry["design_audit"]
    design_payload = _verified_json(
        _resolve(design["path"], repo), DESIGN_AUDIT_SHA256, "V28 design audit"
    )
    if (
        design.get("sha256") != DESIGN_AUDIT_SHA256
        or design_payload.get("selected_control", {}).get("name")
        != "source_balanced_observation_matched_full_cube_latent_knn"
        or design_payload.get("firewall", {}).get("Astrid_accessed") is not False
        or design_payload.get("firewall", {}).get("historical_EAGLE_accessed")
        is not False
    ):
        raise ValueError("V28 design selection or firewall differs")
    parent = registry["parent_evidence"]
    v27_path = _resolve(parent["v27_registry"], repo)
    if sha256_file(v27_path) != parent["v27_registry_sha256"]:
        raise ValueError("V28 V27 parent registry differs")
    _, artifacts, v20, _, _ = load_v27_program(v27_path, repo)
    decision = _verified_json(
        Path(parent["v27_decision"]),
        parent["v27_decision_sha256"],
        "V27 decision",
    )
    corrected = _verified_json(
        Path(parent["v27_corrected_latent_audit"]),
        PARENT_AUDIT_SHA256,
        "V27 corrected latent audit",
    )
    summary = corrected["corrected_mechanism_summary"]
    if (
        parent["v27_corrected_latent_audit_sha256"] != PARENT_AUDIT_SHA256
        or decision.get("decision_digest_sha256")
        != parent["v27_decision_digest_sha256"]
        or summary.get("classification") != parent["required_classification"]
        or summary.get("next") != parent["required_next"]
        or corrected.get("Astrid_accessed") is not False
        or corrected.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V28 corrected-parent conclusion or firewall differs")
    inherited = registry["inherited_artifacts"]
    if (
        inherited["attestation_sha256"] != ARTIFACT_SHA256
        or sha256_file(_resolve(inherited["attestation"], repo)) != ARTIFACT_SHA256
        or artifacts["profile"]["sha256"] != inherited["profile_sha256"]
        or artifacts["gaussianization"]["sha256"]
        != inherited["gaussianization_sha256"]
    ):
        raise ValueError("V28 inherited V21 artifacts differ")
    experiment = v20["e8_gaussianized_marginal_retrain"]
    for domain in DOMAIN_ORDER:
        frozen = registry["donor_library"]["sources"][domain]
        data = experiment["data"][domain]["train_data"]
        cache = artifacts["caches"][CACHE_KEYS[domain]["train"]]
        if (
            frozen["objects"] != DONOR_COUNTS[domain]
            or frozen["data_sha256"] != data["sha256"]
            or frozen["cache_sha256"] != cache["sha256"]
            or sha256_file(Path(data["path"])) != data["sha256"]
            or sha256_file(Path(cache["path"])) != cache["sha256"]
        ):
            raise ValueError(f"V28 {domain} donor provenance differs")
    if registry["donor_library"]["total_objects"] != sum(DONOR_COUNTS.values()):
        raise ValueError("V28 total donor count differs")
    return registry, artifacts, v20


def pool_local_condition(condition: np.ndarray, grid: int = LOCAL_GRID) -> np.ndarray:
    """Pool the three informative target-free local fields to a fixed grid."""
    value = np.asarray(condition, dtype=np.float32)
    if value.shape[0] < 3 or value.shape[-3:] != (64, 64, 64) or 64 % grid:
        raise ValueError("V28 local condition has the wrong shape")
    block = 64 // grid
    local = value[:3].reshape(3, grid, block, grid, block, grid, block)
    return local.mean(axis=(2, 4, 6), dtype=np.float64).astype(np.float32)


def source_balanced_fit(rows: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    """Fit equal-source moments over all non-feature axes."""
    if tuple(rows) != DOMAIN_ORDER:
        raise ValueError("V28 fit requires the frozen source order")
    means = []
    seconds = []
    for domain in DOMAIN_ORDER:
        value = np.asarray(rows[domain], dtype=np.float64)
        if value.ndim < 2 or value.shape[0] != DONOR_COUNTS[domain]:
            raise ValueError(f"V28 {domain} descriptor count differs")
        axes = (0,) + tuple(range(2, value.ndim))
        means.append(value.mean(axis=axes))
        seconds.append(np.square(value).mean(axis=axes))
    mean = np.mean(means, axis=0)
    second = np.mean(seconds, axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 1.0e-12))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("V28 descriptor standardization is invalid")
    return {"mean": mean.tolist(), "std": std.tolist()}


def standardize_descriptor(value: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    mean = np.asarray(fit["mean"], dtype=np.float32)
    std = np.asarray(fit["std"], dtype=np.float32)
    shape = (1, len(mean)) + (1,) * max(array.ndim - 2, 0)
    if array.ndim == 1:
        return (array - mean) / std
    return (array - mean.reshape(shape)) / std.reshape(shape)


@dataclass
class DonorLibrary:
    local: dict[str, np.ndarray]
    global_features: dict[str, np.ndarray]
    local_fit: dict[str, list[float]]
    global_fit: dict[str, list[float]]
    data_paths: dict[str, Path]
    cache_paths: dict[str, Path]


def _condition_only(
    data: h5py.File, cache: h5py.File, index: int, radial: np.ndarray
) -> np.ndarray:
    preprocessing = json.loads(cache.attrs["input_preprocessing"])
    observable = apply_input_preprocessing(
        np.asarray(data["input"][index], dtype=np.float32), preprocessing
    )
    mean = np.asarray(cache["conditional_mean"][index], dtype=np.float32)
    return np.concatenate((observable, mean, radial), axis=0)


def build_donor_library(
    artifacts: Mapping[str, Any], v20: Mapping[str, Any]
) -> DonorLibrary:
    experiment = v20["e8_gaussianized_marginal_retrain"]
    radial = radial_geometry(64)[None]
    local: dict[str, np.ndarray] = {}
    global_rows: dict[str, np.ndarray] = {}
    data_paths: dict[str, Path] = {}
    cache_paths: dict[str, Path] = {}
    for domain in DOMAIN_ORDER:
        data_path = Path(experiment["data"][domain]["train_data"]["path"])
        cache_path = Path(artifacts["caches"][CACHE_KEYS[domain]["train"]]["path"])
        data_paths[domain] = data_path
        cache_paths[domain] = cache_path
        rows = np.empty((DONOR_COUNTS[domain], 3, LOCAL_GRID, LOCAL_GRID, LOCAL_GRID), dtype=np.float32)
        with h5py.File(data_path, "r") as data, h5py.File(cache_path, "r") as cache:
            if (
                len(data["input"]) != DONOR_COUNTS[domain]
                or len(cache["standardized_residual"]) != DONOR_COUNTS[domain]
                or float(cache.attrs.get("maximum_postprojection_ortho_dc", np.inf))
                > 1.0e-10
            ):
                raise ValueError(f"V28 {domain} train cache count or DC differs")
            global_rows[domain] = np.asarray(
                cache["observable_context_features"], dtype=np.float32
            )
            for index in range(DONOR_COUNTS[domain]):
                rows[index] = pool_local_condition(
                    _condition_only(data, cache, index, radial)
                )
        local[domain] = rows
        print(f"[library] {domain} {len(rows)}", flush=True)
    local_fit = source_balanced_fit(local)
    global_fit = source_balanced_fit(global_rows)
    for domain in DOMAIN_ORDER:
        local[domain] = standardize_descriptor(local[domain], local_fit)
        global_rows[domain] = standardize_descriptor(
            global_rows[domain], global_fit
        )
    return DonorLibrary(
        local=local,
        global_features=global_rows,
        local_fit=local_fit,
        global_fit=global_fit,
        data_paths=data_paths,
        cache_paths=cache_paths,
    )


def source_quota(global_query_position: int) -> dict[str, int]:
    quota = {domain: 5 for domain in DOMAIN_ORDER}
    quota[DOMAIN_ORDER[global_query_position % len(DOMAIN_ORDER)]] += 1
    return quota


def select_donors(
    query_local: np.ndarray,
    query_global: np.ndarray,
    library: DonorLibrary,
    *,
    global_query_position: int,
) -> list[dict[str, Any]]:
    """Select deterministic unique train donors without target information."""
    local_query = standardize_descriptor(query_local[None], library.local_fit)[0]
    global_query = standardize_descriptor(
        np.asarray(query_global, dtype=np.float32), library.global_fit
    )
    quota = source_quota(global_query_position)
    selected: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        global_distance = np.square(
            library.global_features[domain] - global_query[None]
        ).mean(axis=1)
        candidates = np.argsort(global_distance, kind="stable")[:GLOBAL_PREFILTER]
        best_local = np.full(len(candidates), np.inf, dtype=np.float64)
        best_isometry = np.full(len(candidates), -1, dtype=np.int64)
        candidate_local = library.local[domain][candidates]
        for transform, (permutation, reflections) in enumerate(CUBE_ISOMETRIES):
            oriented = apply_cube_isometry(
                candidate_local, permutation, reflections
            )
            distance = np.square(
                oriented.astype(np.float64) - local_query[None]
            ).mean(axis=(1, 2, 3, 4))
            update = distance < best_local
            best_local[update] = distance[update]
            best_isometry[update] = transform
        total = global_distance[candidates] + best_local
        order = np.argsort(total, kind="stable")[: quota[domain]]
        for location in order:
            selected.append(
                {
                    "source": domain,
                    "donor_index": int(candidates[location]),
                    "isometry": int(best_isometry[location]),
                    "global_distance": float(global_distance[candidates[location]]),
                    "local_distance": float(best_local[location]),
                    "total_distance": float(total[location]),
                }
            )
    if len(selected) != ENSEMBLE_MEMBERS:
        raise RuntimeError("V28 empirical ensemble size differs")
    for domain in DOMAIN_ORDER:
        indices = [row["donor_index"] for row in selected if row["source"] == domain]
        if len(indices) != quota[domain] or len(indices) != len(set(indices)):
            raise RuntimeError("V28 source quota or unique-donor rule failed")
    return selected


def _profile_tensors(
    artifacts: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, ...]:
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    return (
        torch.as_tensor(profile["centers"], dtype=torch.float64, device=device),
        torch.as_tensor(profile["mu"], dtype=torch.float64, device=device),
        torch.as_tensor(profile["log_sigma"], dtype=torch.float64, device=device),
        torch.as_tensor(transform["z_knots"], dtype=torch.float32, device=device),
        torch.as_tensor(
            transform["residual_value_knots"], dtype=torch.float32, device=device
        ),
    )


def _inverse_selected_latents(
    latents: np.ndarray,
    corrected_mean: np.ndarray,
    profile_tensors: tuple[torch.Tensor, ...],
    *,
    location: float,
    scales: np.ndarray,
    voxel_mpc_h: float,
    device: torch.device,
) -> np.ndarray:
    centers, mu, log_sigma, z_knots, residual_knots = profile_tensors
    latent = torch.from_numpy(np.ascontiguousarray(latents)).to(device)
    u = inverse_gaussianize_torch(latent, z_knots, residual_knots)
    mean = torch.from_numpy(corrected_mean[None]).to(device).expand(
        len(latent), -1, -1, -1, -1
    )
    standardized = invert_profile_torch(u, mean, centers, mu, log_sigma)
    values = standardized[:, 0].float().cpu().numpy()
    physical = np.stack(
        [
            inverse_standardized_residual(
                value,
                predicted_location=location,
                predicted_scales=scales,
                voxel_mpc_h=voxel_mpc_h,
            )
            for value in values
        ]
    ).astype(np.float32)
    return corrected_mean[0][None] + physical


def _load_selected_latents(
    selected: list[dict[str, Any]], cache_handles: Mapping[str, h5py.File]
) -> np.ndarray:
    rows = []
    for donor in selected:
        latent = np.asarray(
            cache_handles[donor["source"]]["standardized_residual"][
                donor["donor_index"]
            ],
            dtype=np.float32,
        )
        permutation, reflections = CUBE_ISOMETRIES[donor["isometry"]]
        rows.append(apply_cube_isometry(latent, permutation, reflections))
    return np.stack(rows)


def sample_all(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry, artifacts, v20 = load_frozen_program(args.registry.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V28 sampling requires a clean committed worktree")
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V28 sampling requires Lageunha")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V28 sampling requires the Lageunha Ada GPU")
    preflight = _verified_json(
        args.preflight.resolve(), args.preflight_sha256, "V28 hard preflight"
    )
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("registry_sha256") != REGISTRY_SHA256
        or preflight.get("code_commit") != commit
    ):
        raise ValueError("V28 preflight differs at sampling")
    output_root = args.out.resolve()
    if output_root.exists():
        raise RuntimeError(f"V28 refuses pre-existing output root: {output_root}")
    output_root.mkdir(parents=True)
    library = build_donor_library(artifacts, v20)
    device = torch.device(args.device)
    tensors = _profile_tensors(artifacts, device)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    radial = radial_geometry(64)[None]
    domain_offset = 0
    cache_handles = {
        domain: h5py.File(library.cache_paths[domain], "r") for domain in DOMAIN_ORDER
    }
    try:
        for domain in DOMAIN_ORDER:
            data_info = experiment["data"][domain]["validation_data"]
            cache_info = artifacts["caches"][CACHE_KEYS[domain]["validation"]]
            indices = _indices(experiment["development_objects"][domain], repo)
            domain_root = output_root / DOMAIN_KEYS[domain]
            domain_root.mkdir()
            output = domain_root / "ensemble16.h5"
            partial = output.with_suffix(".h5.partial")
            with h5py.File(data_info["path"], "r") as data, h5py.File(
                cache_info["path"], "r"
            ) as cache, h5py.File(partial, "w") as handle:
                sample_ds = handle.create_dataset(
                    "sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4",
                    chunks=(1, 1, 1, 64, 64, 64), compression="lzf",
                )
                mean_ds = handle.create_dataset(
                    "conditional_mean", shape=(16, 1, 64, 64, 64),
                    dtype="f4", compression="lzf",
                )
                truth_ds = handle.create_dataset(
                    "truth", shape=(16, 1, 64, 64, 64),
                    dtype="f4", compression="lzf",
                )
                handle.create_dataset("source_index", data=np.asarray(indices, dtype=np.int64))
                location_ds = handle.create_dataset(
                    "predicted_residual_dc", shape=(16,), dtype="f4"
                )
                scale_ds = handle.create_dataset(
                    "predicted_band_scales", shape=(16, 4), dtype="f4"
                )
                donor_source_ds = handle.create_dataset(
                    "donor_source", shape=(16, 16), dtype="i1"
                )
                donor_index_ds = handle.create_dataset(
                    "donor_index", shape=(16, 16), dtype="i4"
                )
                donor_isometry_ds = handle.create_dataset(
                    "donor_isometry", shape=(16, 16), dtype="i1"
                )
                donor_distance_ds = handle.create_dataset(
                    "donor_distance", shape=(16, 16, 3), dtype="f4"
                )
                maximum_latent_dc = 0.0
                for output_index, data_index in enumerate(indices):
                    condition = _condition_only(data, cache, data_index, radial)
                    query_local = pool_local_condition(condition)
                    query_global = np.asarray(
                        cache["observable_context_features"][data_index],
                        dtype=np.float32,
                    )
                    selected = select_donors(
                        query_local,
                        query_global,
                        library,
                        global_query_position=domain_offset + output_index,
                    )
                    latent = _load_selected_latents(selected, cache_handles)
                    maximum_latent_dc = max(
                        maximum_latent_dc,
                        float(np.max(np.abs(latent.mean(axis=(-3, -2, -1)))))
                    )
                    corrected_mean = np.asarray(
                        cache["conditional_mean"][data_index], dtype=np.float32
                    )
                    location = float(cache["predicted_residual_dc"][data_index])
                    scales = np.asarray(
                        cache["predicted_band_scales"][data_index], dtype=np.float64
                    )
                    sample = _inverse_selected_latents(
                        latent,
                        corrected_mean,
                        tensors,
                        location=location,
                        scales=scales,
                        voxel_mpc_h=float(cache.attrs["voxel_mpc_h"]),
                        device=device,
                    )
                    if not np.isfinite(sample).all():
                        raise RuntimeError("V28 physical inverse produced nonfinite density")
                    sample_ds[output_index, :, 0] = sample
                    mean_ds[output_index] = corrected_mean + np.float32(location)
                    truth_ds[output_index] = np.asarray(
                        data["target"][data_index], dtype=np.float32
                    )
                    location_ds[output_index] = location
                    scale_ds[output_index] = scales
                    donor_source_ds[output_index] = [
                        DOMAIN_ORDER.index(row["source"]) for row in selected
                    ]
                    donor_index_ds[output_index] = [
                        row["donor_index"] for row in selected
                    ]
                    donor_isometry_ds[output_index] = [
                        row["isometry"] for row in selected
                    ]
                    donor_distance_ds[output_index] = np.asarray(
                        [
                            [row["global_distance"], row["local_distance"], row["total_distance"]]
                            for row in selected
                        ],
                        dtype=np.float32,
                    )
                    print(
                        f"[sample] V28 {domain} {output_index + 1}/16",
                        flush=True,
                    )
                handle.attrs.update(
                    {
                        "schema": ENSEMBLE_SCHEMA,
                        "method": "source_balanced_observation_matched_full_cube_latent_knn",
                        "v28_registry_sha256": REGISTRY_SHA256,
                        "design_audit_sha256": DESIGN_AUDIT_SHA256,
                        "parent_audit_sha256": PARENT_AUDIT_SHA256,
                        "v21_artifact_attestation_sha256": ARTIFACT_SHA256,
                        "v21_profile_sha256": artifacts["profile"]["sha256"],
                        "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
                        "source_cache": str(Path(cache_info["path"]).resolve()),
                        "source_cache_sha256": cache_info["sha256"],
                        "source_data_sha256": data_info["sha256"],
                        "donor_cache_sha256": json.dumps(
                            {
                                source: artifacts["caches"][CACHE_KEYS[source]["train"]]["sha256"]
                                for source in DOMAIN_ORDER
                            },
                            sort_keys=True,
                        ),
                        "donor_counts": json.dumps(DONOR_COUNTS, sort_keys=True),
                        "local_descriptor_fit": json.dumps(library.local_fit, sort_keys=True),
                        "global_descriptor_fit": json.dumps(library.global_fit, sort_keys=True),
                        "global_prefilter_per_source": GLOBAL_PREFILTER,
                        "ensemble_members": ENSEMBLE_MEMBERS,
                        "diagnostic_k_h_mpc": 1.0,
                        "maximum_absolute_selected_latent_dc": maximum_latent_dc,
                        "selection_uses_validation_truth": False,
                        "location_scale_uses_target": False,
                        "direct_empirical_sampling": True,
                        "sampling_code_commit": commit,
                        "worktree_clean_at_sampling": clean,
                        "hard_preflight": str(args.preflight.resolve()),
                        "hard_preflight_sha256": args.preflight_sha256,
                        "Astrid_accessed": False,
                        "historical_EAGLE_accessed": False,
                        "complete": True,
                    }
                )
            os.replace(partial, output)
            domain_offset += len(indices)
    finally:
        for handle in cache_handles.values():
            handle.close()
    print(f"[out] {output_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sample = sub.add_parser("sample-all")
    sample.add_argument("--registry", type=Path, required=True)
    sample.add_argument("--repo", type=Path, required=True)
    sample.add_argument("--preflight", type=Path, required=True)
    sample.add_argument("--preflight-sha256", required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_all(args)


if __name__ == "__main__":
    main()
