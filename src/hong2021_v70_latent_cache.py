#!/usr/bin/env python
"""Build and verify the preflight-authorized V70 train-only latent cache."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v50_network import bounded_mixture_cdf
from hong2021_v63_preflight import _path, load_program as load_v63_program
from hong2021_v63_train import _is_ancestor
from hong2021_v63_train_gate import _load_fit
from hong2021_v70_preflight import (
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    SCHEMA as PREFLIGHT_SCHEMA,
    gaussianize_rank,
    load_program,
)


CACHE_SCHEMA = "hong2021-v70-query-aligned-train-only-Gaussian-latent-cache-v1"
REPORT_SCHEMA = "hong2021-v70-query-aligned-train-only-latent-cache-report-v1"


def fit_and_holdout_indices(
    objects: int, holdout: list[int] | tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    held = np.asarray(sorted(map(int, holdout)), dtype=np.int64)
    if (
        objects <= 0
        or held.ndim != 1
        or held.size == 0
        or np.unique(held).size != held.size
        or np.any(held < 0)
        or np.any(held >= objects)
    ):
        raise ValueError("V70 mechanism-holdout indices differ")
    mask = np.ones(objects, dtype=bool)
    mask[held] = False
    fit = np.flatnonzero(mask).astype(np.int64)
    if np.intersect1d(fit, held).size or fit.size + held.size != objects:
        raise RuntimeError("V70 fit and holdout partition differs")
    return fit, held


def _load_authorization(
    path: Path, digest: str, repo: Path, commit: str
) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError("V70 preflight hash differs")
    row = json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if (
        row.get("schema") != PREFLIGHT_SCHEMA
        or row.get("status") != "pass"
        or row.get("program_sha256") != PROGRAM_SHA256
        or row.get("program_freeze_commit") != PROGRAM_FREEZE_COMMIT
        or row.get("worktree_clean") is not True
        or row.get("representation_pass") is not True
        or row.get("model_pass") is not True
        or row.get("latent_cache_construction_authorized") is not True
        or row.get("optimizer_constructed") is not False
        or row.get("optimizer_step_performed") is not False
        or row.get("validation_accessed") is not False
        or row.get("development_accessed") is not False
        or row.get("historical_EAGLE_accessed") is not False
        or row.get("independent_EAGLE_accessed") is not False
        or row.get("independent_gate_locked") is not True
        or canonical_digest(row) != row.get("decision_digest_sha256")
        or not _is_ancestor(repo, str(row.get("code_commit")), commit)
    ):
        raise ValueError("V70 preflight authorization differs")
    return row


def _frozen_marginal(
    program: dict[str, Any], repo: Path, commit: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], h5py.File]:
    frozen = program["frozen_inputs"]
    v63, v35, _, _, _, _, _ = load_v63_program(
        _path(repo, frozen["v63_program"]), repo
    )
    boundaries = {
        domain: float(v63["sealed_q99_9_backbone_boundaries"][domain])
        for domain in DOMAIN_ORDER
    }
    model, _ = _load_fit(
        _path(repo, frozen["v63_checkpoint"]),
        frozen["v63_checkpoint_sha256"],
        _path(repo, frozen["v63_training_report"]),
        frozen["v63_training_report_sha256"],
        frozen["v56_grid_sha256"],
        frozen["v54_threshold_selection_sha256"],
        frozen["v63_preflight_sha256"],
        frozen["conditioning_cache_sha256"],
        v63["frozen_inputs"]["support_selection_sha256"],
        boundaries,
        repo,
        commit,
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    return model, v35, prepared


def _scan_cache(
    path: Path, expected: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema")) != CACHE_SCHEMA:
            raise ValueError("V70 latent-cache schema differs")
        for domain in DOMAIN_ORDER:
            latent_set = handle[f"{domain}/latent"]
            clamp_set = handle[f"{domain}/rank_clamp_count"]
            fit_set = np.asarray(handle[f"{domain}/fit_indices"], dtype=np.int64)
            held_set = np.asarray(
                handle[f"{domain}/mechanism_holdout_indices"], dtype=np.int64
            )
            objects = int(expected[domain]["objects"])
            if latent_set.shape != (objects, 1, 64, 64, 64):
                raise ValueError("V70 latent-cache cube shape differs")
            total = 0
            total_sum = 0.0
            total_square = 0.0
            minimum = float("inf")
            maximum = float("-inf")
            for index in range(objects):
                value = np.asarray(latent_set[index], dtype=np.float32)
                if not np.isfinite(value).all():
                    raise RuntimeError("V70 latent-cache value is nonfinite")
                double = value.astype(np.float64)
                total += int(double.size)
                total_sum += float(double.sum(dtype=np.float64))
                total_square += float(np.square(double).sum(dtype=np.float64))
                minimum = min(minimum, float(double.min()))
                maximum = max(maximum, float(double.max()))
            mean = total_sum / total
            variance = max(total_square / total - mean * mean, 0.0)
            clamp = int(np.asarray(clamp_set, dtype=np.int64).sum(dtype=np.int64))
            disjoint = bool(
                np.intersect1d(fit_set, held_set).size == 0
                and fit_set.size + held_set.size == objects
            )
            result[domain] = {
                "objects": objects,
                "voxels": total,
                "rank_clamp_count": clamp,
                "rank_clamp_fraction": clamp / total,
                "latent_mean": mean,
                "latent_standard_deviation": variance**0.5,
                "latent_minimum": minimum,
                "latent_maximum": maximum,
                "fit_objects": int(fit_set.size),
                "mechanism_holdout_objects": int(held_set.size),
                "fit_holdout_disjoint_and_complete": disjoint,
                "fit_indices_sha256": sha256_file_bytes(fit_set.tobytes()),
                "mechanism_holdout_indices_sha256": sha256_file_bytes(
                    held_set.tobytes()
                ),
            }
    return result


def sha256_file_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def build_cache(
    program_path: Path,
    repo: Path,
    preflight_path: Path,
    preflight_sha: str,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    program, _, v65 = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or output.exists()
        or report_path.exists()
    ):
        raise RuntimeError("V70 cache requires clean Lageunha and new outputs")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V70 cache requires the Lageunha Ada GPU")
    authorization = _load_authorization(
        preflight_path.resolve(), preflight_sha, repo, commit
    )
    device = torch.device("cuda")
    model, v35, prepared = _frozen_marginal(program, repo, commit, device)
    query = v65["immutable_train_queries"]
    expected: dict[str, dict[str, Any]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    try:
        with h5py.File(partial, "w") as target_file:
            target_file.attrs.update(
                {
                    "schema": CACHE_SCHEMA,
                    "program_sha256": PROGRAM_SHA256,
                    "program_freeze_commit": PROGRAM_FREEZE_COMMIT,
                    "code_commit": commit,
                    "preflight_sha256": preflight_sha,
                    "v63_checkpoint_sha256": program["frozen_inputs"][
                        "v63_checkpoint_sha256"
                    ],
                    "conditioning_cache_sha256": program["frozen_inputs"][
                        "conditioning_cache_sha256"
                    ],
                    "rank_epsilon": 1.0e-7,
                    "validation_accessed": False,
                    "development_accessed": False,
                    "independent_gate_locked": True,
                    "complete": False,
                }
            )
            for domain in DOMAIN_ORDER:
                row = v35["development_domains"][domain]
                objects = int(row["train_objects"])
                fit, held = fit_and_holdout_indices(objects, query[domain])
                expected[domain] = {
                    "objects": objects,
                    "fit": fit,
                    "held": held,
                }
                group = target_file.create_group(domain)
                group.attrs.update(
                    {
                        "train_data": row["train_data"],
                        "train_data_sha256": row["train_data_sha256"],
                        "train_cache": row["train_cache"],
                        "train_cache_sha256": row["train_cache_sha256"],
                    }
                )
                latent_set = group.create_dataset(
                    "latent",
                    shape=(objects, 1, 64, 64, 64),
                    dtype="f4",
                    chunks=(1, 1, 64, 64, 64),
                    compression="lzf",
                )
                clamp_set = group.create_dataset(
                    "rank_clamp_count", shape=(objects,), dtype="i8"
                )
                group.create_dataset("fit_indices", data=fit)
                group.create_dataset("mechanism_holdout_indices", data=held)
                data, cache = _open_split(row, "train")
                try:
                    for index in range(objects):
                        condition, standardized, _ = condition_cube(
                            data, cache, prepared, domain, "train", index
                        )
                        condition_tensor = torch.from_numpy(condition[None]).to(device)
                        target_tensor = torch.from_numpy(standardized[None]).to(device)
                        with torch.no_grad():
                            parameters = model(condition_tensor).float()
                            rank = (
                                bounded_mixture_cdf(parameters, target_tensor)
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                        latent, _, clamp_count = gaussianize_rank(rank)
                        latent_set[index] = latent[0]
                        clamp_set[index] = clamp_count
                        if (index + 1) % 16 == 0 or index + 1 == objects:
                            print(
                                f"[v70-cache] {domain} {index + 1}/{objects}",
                                flush=True,
                            )
                finally:
                    data.close()
                    cache.close()
            target_file.attrs["complete"] = True
            target_file.flush()
    finally:
        prepared.close()
    os.replace(partial, output)
    scan = _scan_cache(output, expected)
    rules = program["latent_cache"]["full_scan_requirements"]
    scan_pass = all(
        row["rank_clamp_fraction"]
        <= float(rules["maximum_rank_clamp_fraction_each_domain"])
        and abs(row["latent_mean"])
        <= float(rules["maximum_absolute_latent_mean_each_domain"])
        and float(rules["latent_standard_deviation_interval_each_domain"][0])
        <= row["latent_standard_deviation"]
        <= float(rules["latent_standard_deviation_interval_each_domain"][1])
        and row["fit_holdout_disjoint_and_complete"]
        for row in scan.values()
    )
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "pass" if scan_pass else "fail",
        "program_sha256": PROGRAM_SHA256,
        "program_freeze_commit": PROGRAM_FREEZE_COMMIT,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "preflight_sha256": preflight_sha,
        "preflight_decision_digest_sha256": authorization[
            "decision_digest_sha256"
        ],
        "cache": str(output.resolve()),
        "cache_sha256": sha256_file(output),
        "domains": scan,
        "complete_scan_pass": scan_pass,
        "fixed_training_authorized": scan_pass,
        "optimizer_constructed": False,
        "optimizer_step_performed": False,
        "validation_accessed": False,
        "development_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial_report, report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = build_cache(
        args.program,
        args.repo,
        args.preflight,
        args.preflight_sha256,
        args.out,
        args.report,
    )
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
