#!/usr/bin/env python
"""Hard preflight for the frozen V28 empirical joint residual control."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import h5py
import numpy as np
import torch

from hong2021_residual_diffusion import radial_geometry
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _indices
from hong2021_v28_empirical import (
    CACHE_KEYS,
    DOMAIN_ORDER,
    DONOR_COUNTS,
    ENSEMBLE_MEMBERS,
    GLOBAL_PREFILTER,
    PREFLIGHT_SCHEMA,
    REGISTRY_SHA256,
    _condition_only,
    _inverse_selected_latents,
    _load_selected_latents,
    _profile_tensors,
    build_donor_library,
    load_frozen_program,
    pool_local_condition,
    select_donors,
    source_quota,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V28 hard preflight requires Lageunha")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V28 hard preflight requires the Lageunha Ada GPU")
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V28 hard preflight requires a clean committed worktree")
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V28 preflight: {output}")
    _, artifacts, v20 = load_frozen_program(args.registry.resolve(), repo)
    library = build_donor_library(artifacts, v20)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    domain = "TNG100"
    data_info = experiment["data"][domain]["validation_data"]
    cache_info = artifacts["caches"][CACHE_KEYS[domain]["validation"]]
    index = _indices(experiment["development_objects"][domain], repo)[0]
    radial = radial_geometry(64)[None]
    with h5py.File(data_info["path"], "r") as data, h5py.File(
        cache_info["path"], "r"
    ) as cache:
        condition = _condition_only(data, cache, index, radial)
        selected = select_donors(
            pool_local_condition(condition),
            np.asarray(cache["observable_context_features"][index], dtype=np.float32),
            library,
            global_query_position=0,
        )
        corrected_mean = np.asarray(
            cache["conditional_mean"][index], dtype=np.float32
        )
        location = float(cache["predicted_residual_dc"][index])
        scales = np.asarray(cache["predicted_band_scales"][index], dtype=np.float64)
        voxel_mpc_h = float(cache.attrs["voxel_mpc_h"])
    handles = {
        source: h5py.File(library.cache_paths[source], "r")
        for source in DOMAIN_ORDER
    }
    try:
        latent = _load_selected_latents(selected, handles)
    finally:
        for handle in handles.values():
            handle.close()
    device = torch.device(args.device)
    sample = _inverse_selected_latents(
        latent,
        corrected_mean,
        _profile_tensors(artifacts, device),
        location=location,
        scales=scales,
        voxel_mpc_h=voxel_mpc_h,
        device=device,
    )
    selected_counts = {
        source: sum(row["source"] == source for row in selected)
        for source in DOMAIN_ORDER
    }
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "code_commit": commit,
        "worktree_clean": clean,
        "registry": str(args.registry.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "full_pytest_required_by_launcher": True,
        "donor_counts": DONOR_COUNTS,
        "total_donors": sum(DONOR_COUNTS.values()),
        "local_descriptor_fit": library.local_fit,
        "global_descriptor_fit": library.global_fit,
        "global_prefilter_per_source": GLOBAL_PREFILTER,
        "real_query": {
            "domain": domain,
            "source_index": index,
            "ensemble_members": len(selected),
            "expected_ensemble_members": ENSEMBLE_MEMBERS,
            "source_quota": source_quota(0),
            "selected_counts": selected_counts,
            "unique_source_index_pairs": len(
                {(row["source"], row["donor_index"]) for row in selected}
            ),
            "isometries_in_range": all(0 <= row["isometry"] < 48 for row in selected),
            "all_distances_finite": all(
                np.isfinite(row["total_distance"]) for row in selected
            ),
            "maximum_absolute_selected_latent_dc": float(
                np.max(np.abs(latent.mean(axis=(-3, -2, -1))))
            ),
            "physical_sample_finite": bool(np.isfinite(sample).all()),
            "physical_sample_minimum_y": float(sample.min()),
            "physical_sample_maximum_y": float(sample.max()),
        },
        "descriptor_fit_uses_train_only": True,
        "selection_uses_validation_truth": False,
        "validation_target_dataset_read_during_selection": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    if (
        selected_counts != source_quota(0)
        or len(selected) != ENSEMBLE_MEMBERS
        or report["real_query"]["unique_source_index_pairs"] != ENSEMBLE_MEMBERS
        or not report["real_query"]["isometries_in_range"]
        or not report["real_query"]["all_distances_finite"]
        or report["real_query"]["maximum_absolute_selected_latent_dc"] > 1.0e-7
        or not report["real_query"]["physical_sample_finite"]
    ):
        raise RuntimeError("V28 hard preflight invariant failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
