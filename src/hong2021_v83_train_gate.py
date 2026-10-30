#!/usr/bin/env python
"""Evaluate the fixed V83 checkpoint on its sealed train-only holdout."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v83_contract import DOMAIN_ORDER, load_program
from hong2021_v83_network import conditional_cdf, conditional_log_probability
from hong2021_v83_train import CHECKPOINT_SCHEMA, REPORT_SCHEMA, seeded_model


SCHEMA = "hong2021-v83-train-holdout-conditional-calibration-gate-v1"
PIT_BINS = 10
PIT_MEAN_INTERVAL = (0.47, 0.53)
QUARTILE_PIT_MEAN_INTERVAL = (0.43, 0.57)
MAXIMUM_PIT_BIN_MASS_ERROR = 0.05
COVERAGE_INTERVALS = {
    "50": (0.43, 0.57),
    "80": (0.73, 0.87),
    "95": (0.90, 0.985),
}


def calibration_pass(row: dict[str, Any]) -> bool:
    return bool(
        row["NLL_improvement_over_standard_normal"] > 0.0
        and PIT_MEAN_INTERVAL[0] <= row["PIT_mean"] <= PIT_MEAN_INTERVAL[1]
        and all(
            QUARTILE_PIT_MEAN_INTERVAL[0]
            <= value
            <= QUARTILE_PIT_MEAN_INTERVAL[1]
            for value in row["backbone_quartile_PIT_means"]
        )
        and row["PIT_maximum_bin_mass_error"] <= MAXIMUM_PIT_BIN_MASS_ERROR
        and all(
            COVERAGE_INTERVALS[level][0]
            <= row["central_coverage"][level]
            <= COVERAGE_INTERVALS[level][1]
            for level in COVERAGE_INTERVALS
        )
    )


@torch.inference_mode()
def gate(
    program_path: Path,
    repo: Path,
    conditioning_cache: Path,
    cache_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    report_path: Path,
    report_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if (
        not clean
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or not torch.cuda.is_available()
        or "ada" not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError("V83 train gate requires clean frozen Lageunha Ada")
    if output_path.exists():
        raise FileExistsError("V83 train gate refuses an existing output")
    program, v35, partition = load_program(program_path, repo, commit)
    if output_path.resolve() != Path(program["output_roots"]["train_gate"]).resolve():
        raise ValueError("V83 train gate output differs")
    frozen = program["frozen_inputs"]
    if (
        conditioning_cache.resolve() != Path(frozen["conditioning_cache"]).resolve()
        or cache_sha256 != frozen["conditioning_cache_sha256"]
        or sha256_file(conditioning_cache) != cache_sha256
        or sha256_file(checkpoint_path) != checkpoint_sha256
        or sha256_file(report_path) != report_sha256
    ):
        raise ValueError("V83 train gate artifact hash differs")
    report = json.loads(report_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    program_sha = sha256_file(program_path)
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status")
        != "complete_fixed_12000_step_proper_conditional_spline_fit"
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or report.get("program_sha256") != program_sha
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != program_sha
        or checkpoint.get("step", checkpoint.get("steps")) != 12_000
        or checkpoint.get("validation_payload_accessed") is not False
        or checkpoint.get("independent_gate_locked") is not True
    ):
        raise ValueError("V83 checkpoint or training report differs")
    device = torch.device("cuda")
    model = seeded_model(device)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()
    prepared = load_cache(conditioning_cache, cache_sha256, commit)
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    rows: dict[str, Any] = {}
    try:
        for domain in DOMAIN_ORDER:
            histogram = np.zeros(PIT_BINS, dtype=np.int64)
            quartile_sum = np.zeros(4, dtype=np.float64)
            quartile_count = np.zeros(4, dtype=np.int64)
            coverage_count = {"50": 0, "80": 0, "95": 0}
            pit_sum = 0.0
            model_nll_sum = 0.0
            normal_nll_sum = 0.0
            voxel_count = 0
            data, cache = handles[domain]
            for index in partition[domain]["holdout"]:
                condition, target, _ = condition_cube(
                    data, cache, prepared, domain, "train", index
                )
                condition_tensor = torch.from_numpy(condition[None]).to(device)
                target_tensor = torch.from_numpy(target[None]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    parameters = model(condition_tensor)
                uniform = conditional_cdf(parameters, target_tensor)
                log_probability = conditional_log_probability(parameters, target_tensor)
                values = uniform[0, 0].float().cpu().numpy().reshape(-1)
                truth = target.reshape(-1).astype(np.float64)
                score = condition[3].reshape(-1)
                order = np.argsort(score, kind="stable")
                chunks = np.array_split(order, 4)
                for quartile, positions in enumerate(chunks):
                    quartile_sum[quartile] += float(values[positions].sum(dtype=np.float64))
                    quartile_count[quartile] += int(len(positions))
                histogram += np.histogram(values, bins=PIT_BINS, range=(0.0, 1.0))[0]
                pit_sum += float(values.sum(dtype=np.float64))
                model_nll_sum += float((-log_probability).sum().cpu())
                normal_nll_sum += float(
                    np.sum(0.5 * np.square(truth) + 0.5 * math.log(2.0 * math.pi))
                )
                coverage_count["50"] += int(np.count_nonzero(np.abs(values - 0.5) <= 0.25))
                coverage_count["80"] += int(np.count_nonzero(np.abs(values - 0.5) <= 0.40))
                coverage_count["95"] += int(np.count_nonzero(np.abs(values - 0.5) <= 0.475))
                voxel_count += int(values.size)
            probabilities = histogram.astype(np.float64) / voxel_count
            row = {
                "holdout_objects": len(partition[domain]["holdout"]),
                "holdout_indices": partition[domain]["holdout"],
                "voxels": voxel_count,
                "conditional_NLL": model_nll_sum / voxel_count,
                "standard_normal_NLL": normal_nll_sum / voxel_count,
                "NLL_improvement_over_standard_normal": (
                    normal_nll_sum - model_nll_sum
                )
                / voxel_count,
                "PIT_mean": pit_sum / voxel_count,
                "PIT_histogram_probabilities": probabilities.tolist(),
                "PIT_maximum_bin_mass_error": float(
                    np.max(np.abs(probabilities - 1.0 / PIT_BINS))
                ),
                "backbone_quartile_PIT_means": (
                    quartile_sum / quartile_count
                ).tolist(),
                "central_coverage": {
                    level: count / voxel_count
                    for level, count in coverage_count.items()
                },
            }
            row["pass"] = calibration_pass(row)
            rows[domain] = row
            print(f"[v83-train-gate] {domain} " + json.dumps(row), flush=True)
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        prepared.close()
    passed = all(row["pass"] for row in rows.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "program_sha256": program_sha,
        "code_commit": commit,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_report": str(report_path.resolve()),
        "training_report_sha256": report_sha256,
        "partition_sha256": program["partition"]["sha256"],
        "thresholds": {
            "NLL_improvement_over_standard_normal": "strictly_positive",
            "PIT_mean_interval": list(PIT_MEAN_INTERVAL),
            "backbone_quartile_PIT_mean_interval": list(
                QUARTILE_PIT_MEAN_INTERVAL
            ),
            "maximum_PIT_bin_mass_error": MAXIMUM_PIT_BIN_MASS_ERROR,
            "central_coverage_intervals": {
                level: list(interval) for level, interval in COVERAGE_INTERVALS.items()
            },
        },
        "domains": rows,
        "train_holdout_mechanism_pass": passed,
        "checkpoint_or_hyperparameter_selected_on_holdout": False,
        "validation_payload_accessed": False,
        "development_payload_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
        "next": (
            "run_one_consumed_development_V72_spatial_copula_engineering_gate"
            if passed
            else "stop_V83_before_any_development_sampling"
        ),
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--conditioning-cache", type=Path, required=True)
    parser.add_argument("--conditioning-cache-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gate(
        args.program,
        args.repo,
        args.conditioning_cache,
        args.conditioning_cache_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
        args.report,
        args.report_sha256,
        args.out,
    )


if __name__ == "__main__":
    main()
