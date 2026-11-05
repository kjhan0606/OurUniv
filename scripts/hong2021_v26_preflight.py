#!/usr/bin/env python
"""Run the frozen real-data Ada feasibility checks for V26."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path

import numpy as np
import torch

from hong2021_v14_edm import V14ResidualDataset, source_balanced_feature_standardization
from hong2021_v14_mean_correction import DOMAINS
from hong2021_v15_edm import git_state
from hong2021_v18_init import sha256_file
from hong2021_v26 import (
    HAAR_ARTIFACT_SHA256,
    NON_DC_DIMENSIONS,
    PARAMETERS,
    REGISTRY_SHA256,
    _paths,
    build_model,
    load_frozen_program,
)
from hong2021_v26_haar import haar_pyramid, inverse_haar_pyramid


SCHEMA = "hong2021-v26-hard-preflight-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V26 hard preflight requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V26 hard preflight requires the Lageunha Ada GPU")
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V26 hard preflight requires a clean committed worktree")
    registry, artifacts, v20, _, haar = load_frozen_program(
        args.registry.resolve(), repo
    )
    paths = _paths(artifacts, v20)
    feature_datasets = {
        domain: V14ResidualDataset(row[0], row[1], False)
        for domain, row in paths.items()
    }
    feature_fit = source_balanced_feature_standardization(feature_datasets)
    rows = []
    for domain in DOMAINS:
        rows.extend([feature_datasets[domain][0], feature_datasets[domain][1]])
    device = torch.device("cuda")
    condition = torch.stack([row[0] for row in rows]).to(device)
    residual = torch.stack([row[1] for row in rows]).to(device)
    model = build_model(haar, feature_fit, device=device)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", enabled=True):
        log_prob, diagnostic = model.log_prob(residual, condition)
        loss = -log_prob.mean() / NON_DC_DIMENSIONS
    loss.backward()
    gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0e9))
    lowpass, details = haar_pyramid(residual[:1].double())
    recovered = inverse_haar_pyramid(lowpass, details)
    haar_roundtrip = float((recovered - residual[:1].double()).abs().max())
    spatial_energy = float(residual[:1].double().square().sum())
    coefficient_energy = float(
        lowpass.double().square().sum()
        + sum(detail.double().square().sum() for detail in details)
    )
    parseval_relative = abs(spatial_energy - coefficient_energy) / spatial_energy
    model.eval()
    first, first_dc = model.sample_with_diagnostics(
        condition[:1], generator=torch.Generator(device=device).manual_seed(926001)
    )
    second, second_dc = model.sample_with_diagnostics(
        condition[:1], generator=torch.Generator(device=device).manual_seed(926001)
    )
    reproducible = bool(torch.equal(first, second))
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V26 preflight: {output}")
    test_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "code_commit": commit,
        "test_commit": test_commit,
        "worktree_clean": clean,
        "registry": str(args.registry.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "haar_artifact_sha256": HAAR_ARTIFACT_SHA256,
        "parameters": PARAMETERS,
        "batch": 6,
        "source_balance_per_batch": {"TNG100": 2, "SIMBA": 2, "Swift": 2},
        "non_dc_dimensions": NON_DC_DIMENSIONS,
        "real_batch_nll_per_non_dc_dimension": float(loss.detach()),
        "real_batch_log_prob_finite": bool(torch.isfinite(log_prob).all()),
        "real_batch_gradient_finite": gradient_finite,
        "real_batch_gradient_norm": gradient_norm,
        "scale_log_prob_finite": bool(
            torch.isfinite(diagnostic["scale_log_prob_coarse_to_fine"]).all()
        ),
        "haar_maximum_absolute_roundtrip_error": haar_roundtrip,
        "haar_parseval_relative_error": parseval_relative,
        "sample_seed_reproducible": reproducible,
        "sample_maximum_absolute_pre_center_dc": float(
            first_dc["pre_center_mean"].abs().max()
        ),
        "sample_maximum_absolute_post_center_dc": float(
            first_dc["post_center_mean"].abs().max()
        ),
        "sample_repeat_dc_identical": all(
            torch.equal(first_dc[key], second_dc[key]) for key in first_dc
        ),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "peak_memory_limit_gib": 8.0,
        "full_pytest_required_by_launcher": True,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    checks = (
        report["real_batch_log_prob_finite"],
        report["real_batch_gradient_finite"],
        report["scale_log_prob_finite"],
        report["sample_seed_reproducible"],
        report["sample_repeat_dc_identical"],
        report["haar_maximum_absolute_roundtrip_error"] <= 3.0e-6,
        report["haar_parseval_relative_error"] <= 1.0e-6,
        report["sample_maximum_absolute_post_center_dc"] <= 1.0e-7,
        report["peak_allocated_gib"] < report["peak_memory_limit_gib"],
    )
    if not all(checks):
        raise RuntimeError(f"V26 hard-preflight check failed: {report}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
