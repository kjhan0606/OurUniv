#!/usr/bin/env python
"""Evaluate fixed V84B EMA once on its simulation-group/spatial holdout."""
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
from hong2021_v84b_contract import DOMAIN_ORDER, load_program
from hong2021_v84b_network import conditional_cdf, conditional_log_probability
from hong2021_v84b_train import CHECKPOINT_SCHEMA, REPORT_SCHEMA, seeded_model


SCHEMA = "hong2021-v84b-group-held-out-tail-calibration-gate-v1"
PIT_BINS = 100
PIT_MEAN_INTERVAL = (0.49, 0.51)
MAXIMUM_PIT_TV = 0.01
QUARTILE_PIT_MEAN_INTERVAL = (0.47, 0.53)
COVERAGE_INTERVALS = {
    "50": (0.48, 0.52),
    "80": (0.78, 0.82),
    "95": (0.94, 0.96),
}
TAIL_PROBABILITIES = (0.001, 0.0001)
TAIL_RATIO_INTERVAL = (0.8, 1.25)


def calibration_pass(row: dict[str, Any]) -> bool:
    tail = row["tail_exceedance"]
    return bool(
        row["NLL_improvement_over_standard_normal"] > 0.0
        and PIT_MEAN_INTERVAL[0] <= row["PIT_mean"] <= PIT_MEAN_INTERVAL[1]
        and row["PIT_total_variation_from_uniform"] <= MAXIMUM_PIT_TV
        and all(
            QUARTILE_PIT_MEAN_INTERVAL[0]
            <= value
            <= QUARTILE_PIT_MEAN_INTERVAL[1]
            for value in row["backbone_quartile_PIT_means"]
        )
        and all(
            COVERAGE_INTERVALS[level][0]
            <= row["central_coverage"][level]
            <= COVERAGE_INTERVALS[level][1]
            for level in COVERAGE_INTERVALS
        )
        and all(
            TAIL_RATIO_INTERVAL[0]
            <= tail[f"{probability:g}"][f"{side}_over_expected"]
            <= TAIL_RATIO_INTERVAL[1]
            for probability in TAIL_PROBABILITIES
            for side in ("lower", "upper")
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
        raise RuntimeError("V84B group gate requires clean frozen Lageunha Ada")
    if output_path.exists():
        raise FileExistsError("V84B group gate refuses existing output")
    program, v35, partition = load_program(program_path, repo, commit)
    if output_path.resolve() != Path(program["output_roots"]["group_gate"]).resolve():
        raise ValueError("V84B group gate output differs")
    frozen = program["frozen_inputs"]
    if (
        conditioning_cache.resolve() != Path(frozen["conditioning_cache"]).resolve()
        or cache_sha256 != frozen["conditioning_cache_sha256"]
        or sha256_file(conditioning_cache) != cache_sha256
        or sha256_file(checkpoint_path) != checkpoint_sha256
        or sha256_file(report_path) != report_sha256
    ):
        raise ValueError("V84B gate artifact hash differs")
    training_report = json.loads(report_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    program_sha = sha256_file(program_path)
    if (
        training_report.get("schema") != REPORT_SCHEMA
        or training_report.get("status")
        != "complete_fixed_12000_step_group_fit_spliced_tail"
        or training_report.get("checkpoint_sha256") != checkpoint_sha256
        or training_report.get("program_sha256") != program_sha
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != program_sha
        or checkpoint.get("steps") != 12_000
        or checkpoint.get("group_holdout_payload_accessed") is not False
        or checkpoint.get("validation_payload_accessed") is not False
        or checkpoint.get("independent_gate_locked") is not True
    ):
        raise ValueError("V84B checkpoint or training report differs")
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
            tail_count = {
                f"{probability:g}": {"lower": 0, "upper": 0}
                for probability in TAIL_PROBABILITIES
            }
            pit_sum = 0.0
            model_nll_sum = 0.0
            normal_nll_sum = 0.0
            voxel_count = 0
            data, cache = handles[domain]
            for position, index in enumerate(partition[domain]["holdout"]):
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
                for quartile, positions in enumerate(np.array_split(order, 4)):
                    quartile_sum[quartile] += float(
                        values[positions].sum(dtype=np.float64)
                    )
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
                for probability in TAIL_PROBABILITIES:
                    key = f"{probability:g}"
                    tail_count[key]["lower"] += int(np.count_nonzero(values < probability))
                    tail_count[key]["upper"] += int(
                        np.count_nonzero(values > 1.0 - probability)
                    )
                voxel_count += int(values.size)
                if (position + 1) % 16 == 0 or position + 1 == len(
                    partition[domain]["holdout"]
                ):
                    print(
                        f"[v84b-group-gate] {domain} {position + 1}/"
                        f"{len(partition[domain]['holdout'])}",
                        flush=True,
                    )
            probabilities = histogram.astype(np.float64) / voxel_count
            tails: dict[str, Any] = {}
            for probability in TAIL_PROBABILITIES:
                key = f"{probability:g}"
                lower_probability = tail_count[key]["lower"] / voxel_count
                upper_probability = tail_count[key]["upper"] / voxel_count
                tails[key] = {
                    "expected_probability_each_side": probability,
                    "lower_probability": lower_probability,
                    "upper_probability": upper_probability,
                    "lower_over_expected": lower_probability / probability,
                    "upper_over_expected": upper_probability / probability,
                }
            row = {
                "holdout_objects": len(partition[domain]["holdout"]),
                "holdout_indices": partition[domain]["holdout"],
                "holdout_groups": partition[domain].get("holdout_groups"),
                "voxels": voxel_count,
                "conditional_NLL": model_nll_sum / voxel_count,
                "standard_normal_NLL": normal_nll_sum / voxel_count,
                "NLL_improvement_over_standard_normal": (
                    normal_nll_sum - model_nll_sum
                )
                / voxel_count,
                "PIT_mean": pit_sum / voxel_count,
                "PIT_histogram_probabilities": probabilities.tolist(),
                "PIT_total_variation_from_uniform": float(
                    0.5 * np.abs(probabilities - 1.0 / PIT_BINS).sum()
                ),
                "backbone_quartile_PIT_means": (
                    quartile_sum / quartile_count
                ).tolist(),
                "central_coverage": {
                    level: count / voxel_count for level, count in coverage_count.items()
                },
                "tail_exceedance": tails,
            }
            row["pass"] = calibration_pass(row)
            rows[domain] = row
            print(f"[v84b-group-gate-result] {domain} " + json.dumps(row), flush=True)
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
            "maximum_PIT_total_variation": MAXIMUM_PIT_TV,
            "backbone_quartile_PIT_mean_interval": list(
                QUARTILE_PIT_MEAN_INTERVAL
            ),
            "central_coverage_intervals": {
                level: list(interval) for level, interval in COVERAGE_INTERVALS.items()
            },
            "tail_probabilities_each_side": list(TAIL_PROBABILITIES),
            "tail_observed_over_expected_interval": list(TAIL_RATIO_INTERVAL),
        },
        "domains": rows,
        "group_held_out_mechanism_pass": passed,
        "checkpoint_or_hyperparameter_selected_on_holdout": False,
        "group_holdout_accessed_exactly_once_after_fixed_training": True,
        "validation_payload_accessed": False,
        "consumed_development_payload_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
        "next": (
            "design_all_train_production_refit_without_hyperparameter_change"
            if passed
            else "stop_V84B_without_consumed_or_independent_payload_access"
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
