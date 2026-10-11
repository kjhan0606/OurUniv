#!/usr/bin/env python
"""Hard preflight for the frozen V31 physical conditional-copula control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from hong2021_v15_development_gate import git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v31_copula import (
    PREFLIGHT_SCHEMA,
    REGISTRY_SHA256,
    _baseline,
    _verified_json,
    load_model,
    load_program,
    transport_conditional_residual,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V31 preflight requires a clean committed worktree")
    registry = load_program(args.registry.resolve(), repo)
    fit_report = _verified_json(
        args.model_report.resolve(), sha256_file(args.model_report.resolve()), "V31 fit report"
    )
    model_sha = sha256_file(args.model.resolve())
    if (
        fit_report.get("schema") != "hong2021-v31-train-only-physical-conditional-copula-v1"
        or fit_report.get("status") != "complete_train_only_fit"
        or fit_report.get("artifact_sha256") != model_sha
        or fit_report.get("fit_uses_validation_truth") is not False
    ):
        raise ValueError("V31 train-only fit report differs")
    model = load_model(args.model.resolve(), model_sha)
    parent = registry["frozen_v28_selections"]["TNG100"]
    with h5py.File(parent["ensemble"], "r") as old:
        donor_source = DOMAIN_ORDER[int(old["donor_source"][0, 0])]
        donor_index = int(old["donor_index"][0, 0])
        isometry = int(old["donor_isometry"][0, 0])
        query_index = int(old["source_index"][0])
        query_cache_path = Path(str(old.attrs["source_cache"]))
        aggregate = np.zeros(3, dtype=np.int64)
        for source in DOMAIN_ORDER:
            with h5py.File(registry["frozen_v28_selections"][source]["ensemble"], "r") as handle:
                values = np.asarray(handle["donor_source"], dtype=np.int64)
                aggregate += np.bincount(values.reshape(-1), minlength=3)
    train = registry["train_only_fit"]["domains"][donor_source]
    with h5py.File(train["data"], "r") as data, h5py.File(train["cache"], "r") as cache:
        donor_truth = np.asarray(data["target"][donor_index], dtype=np.float32)
        donor_backbone = _baseline(cache, donor_index)[None]
    with h5py.File(query_cache_path, "r") as cache:
        query_backbone = _baseline(cache, query_index)[None]
    sample, maximum_dc = transport_conditional_residual(
        donor_truth, donor_backbone, query_backbone, isometry, model
    )
    if tuple(aggregate.tolist()) != (256, 256, 256):
        raise RuntimeError("V31 selected donor aggregate balance differs")
    if maximum_dc > 1.0e-7 or not np.isfinite(sample).all():
        raise RuntimeError("V31 real-data transport preflight failed")
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "code_commit": commit,
        "worktree_clean": clean,
        "registry": str(args.registry.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "model": str(args.model.resolve()),
        "model_sha256": model_sha,
        "model_report": str(args.model_report.resolve()),
        "model_report_sha256": sha256_file(args.model_report.resolve()),
        "full_pytest_required_by_launcher": True,
        "aggregate_selected_donor_counts": aggregate.tolist(),
        "real_donor": {
            "source": donor_source,
            "index": donor_index,
            "isometry": isometry,
            "maximum_absolute_transported_residual_dc": maximum_dc,
            "sample_finite": bool(np.isfinite(sample).all()),
            "sample_minimum_y": float(sample.min()),
            "sample_maximum_y": float(sample.max()),
        },
        "validation_truth_used_for_fit_or_sampling": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError("V31 refuses to overwrite preflight")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
