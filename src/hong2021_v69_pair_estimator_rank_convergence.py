#!/usr/bin/env python
"""No-refit population pair-estimator rank-convergence audit for V69."""
from __future__ import annotations

import argparse
import hashlib
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
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v31_copula import load_model as load_copula
from hong2021_v48_train import load_cache
from hong2021_v63_preflight import _path, load_program as load_v63_program
from hong2021_v63_train import _is_ancestor
from hong2021_v63_train_gate import _load_fit
from hong2021_v64_sampler_alignment_audit import _pair_indices, empirical_pair_moments
from hong2021_v65_structure_factorization_audit import (
    _close_handles,
    _open_train_handles,
    _query_batch,
    _rank_batch,
    donor_mapping,
)


PROGRAM_SHA256 = "ce8d61cd4a82623b0c755c2379d33f71b08d7eea60289e833a9d54d88ac2e940"
PROGRAM_SCHEMA = "hong2021-v69-train-only-population-pair-estimator-rank-convergence-program-v1"
SCHEMA = "hong2021-v69-train-only-population-pair-estimator-rank-convergence-audit-v1"
PROGRAM_FREEZE_COMMIT = "cd6d9f8a52fa577299d623bb8b8511d2d4f5d100"
STREAMS = ("A", "B")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V69 {label} hash differs")
    return _json(path)


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V69 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v68_result_record"]),
        parent["v68_result_record_sha256"],
        "V68 result record",
    )
    population = record.get("population_audit", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("candidate_selected")
        is not parent["required_candidate_selected"]
        or population.get("population_value_stable")
        is not parent["required_population_value_stable"]
        or population.get("optimization_safe")
        is not parent["required_optimization_safe"]
        or firewall.get("training_or_refit_performed")
        is not parent["required_training_or_refit_performed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V69 parent result or firewall differs")
    frozen = program["frozen_inputs"]
    for key, value in frozen.items():
        if key.endswith("_sha256"):
            continue
        digest = frozen.get(f"{key}_sha256")
        if digest is not None and sha256_file(_path(repo, value)) != digest:
            raise ValueError(f"V69 frozen input differs: {key}")
    v65_audit = _json(_path(repo, frozen["v65_audit"]))
    if (
        v65_audit.get("decision_digest_sha256")
        != frozen["v65_audit_decision_digest_sha256"]
        or canonical_digest(v65_audit)
        != frozen["v65_audit_decision_digest_sha256"]
        or v65_audit.get("training_or_refit_performed") is not False
        or v65_audit.get("new_development_accessed") is not False
        or v65_audit.get("independent_gate_locked") is not True
    ):
        raise ValueError("V69 sealed V65 audit differs")
    v63, v35, _, _, _, _, _ = load_v63_program(
        _path(repo, frozen["v63_program"]), repo
    )
    return program, v63, v35, v65_audit


def _mapping_digest(mapping: dict[str, Any]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def population_log_ratios(
    predicted: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    if predicted.shape != truth.shape or predicted.ndim != 3:
        raise ValueError("V69 population array shape differs")
    mean_predicted = predicted.mean(axis=1)
    mean_truth = truth.mean(axis=1)
    if np.any(mean_predicted <= 0.0) or np.any(mean_truth <= 0.0):
        raise RuntimeError("V69 nonpositive population pair moment")
    return np.log(mean_predicted / mean_truth)


def classify(
    integrity_pass: bool, adjacent_pass: bool, independent_pass: bool
) -> tuple[str, str, bool]:
    if not integrity_pass:
        return (
            "population_pair_estimator_convergence_audit_failed_integrity",
            "stop_before_pair_objective_design_and_preserve_the_failed_audit",
            False,
        )
    if adjacent_pass and independent_pass:
        return (
            "rank64_population_pair_estimator_is_train_only_reproducible",
            "freeze_a_rank64_population_pair_objective_model_program_before_refit",
            True,
        )
    return (
        "rank64_population_pair_estimator_is_not_train_only_reproducible",
        "stop_pair_objective_design_and_use_non_pair_training_evidence_only",
        False,
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v63, v35, v65_audit = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V69 audit requires clean Lageunha with frozen ancestry")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V69 audit requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    frozen = program["frozen_inputs"]
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
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    queries = {
        domain: list(map(int, v65_audit["immutable_train_queries"][domain]))
        for domain in DOMAIN_ORDER
    }
    members = int(program["donor_streams"]["members"])
    mappings = {
        stream: donor_mapping(
            v35,
            queries,
            members,
            int(program["donor_streams"][f"stream_{stream}_seed"]),
            same_domain=False,
        )
        for stream in STREAMS
    }
    copula = load_copula(
        _path(repo, frozen["conditional_copula_artifact"]),
        frozen["conditional_copula_artifact_sha256"],
    )
    pairs = _pair_indices(
        int(program["pair_probe"]["anchor_seed"]),
        int(program["pair_probe"]["anchors_per_query"]),
    )
    levels = tuple(map(int, program["donor_streams"]["rank_levels"]))
    predicted = {
        stream: {
            level: np.empty((3, 16, 3), dtype=np.float64) for level in levels
        }
        for stream in STREAMS
    }
    truth = np.empty((3, 16, 3), dtype=np.float64)
    rank_hashers = {stream: hashlib.sha256() for stream in STREAMS}
    maximum_inverse_error = 0.0
    handles = _open_train_handles(v35)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for position in range(16):
            condition, target, backbone = _query_batch(
                handles, prepared, queries, position, device
            )
            with torch.no_grad():
                output = model(condition).float()
                for stream in STREAMS:
                    rank_numpy = _rank_batch(
                        handles,
                        mappings[stream],
                        position,
                        copula,
                        rank_hashers[stream],
                    )
                    ranks = torch.from_numpy(rank_numpy).to(device)
                    for level in levels:
                        pair_predicted, pair_truth, inverse_error = empirical_pair_moments(
                            output,
                            target,
                            backbone,
                            ranks,
                            target_mean,
                            target_std,
                            pairs,
                            level,
                        )
                        predicted[stream][level][:, position] = (
                            pair_predicted.reshape(3, 3).cpu().numpy()
                        )
                        current_truth = pair_truth.reshape(3, 3).cpu().numpy()
                        if stream == "A" and level == levels[0]:
                            truth[:, position] = current_truth
                        elif not np.array_equal(truth[:, position], current_truth):
                            raise RuntimeError("V69 truth binding differs")
                        maximum_inverse_error = max(
                            maximum_inverse_error, inverse_error
                        )
                    del ranks
            del condition, target, backbone, output
    finally:
        _close_handles(handles)
        prepared.close()
    log_ratios = {
        stream: {
            level: population_log_ratios(predicted[stream][level], truth)
            for level in levels
        }
        for stream in STREAMS
    }
    adjacent_rows = {}
    adjacent_maxima = []
    for stream in STREAMS:
        difference = np.abs(log_ratios[stream][64] - log_ratios[stream][32])
        maximum = float(difference.max())
        adjacent_maxima.append(maximum)
        adjacent_rows[stream] = {
            "maximum_absolute_rank32_to_rank64_log_ratio_difference": maximum,
            "absolute_differences": difference.tolist(),
        }
    independent_difference = np.abs(log_ratios["A"][64] - log_ratios["B"][64])
    independent_maximum = float(independent_difference.max())
    tolerance = math.log(1.1)
    adjacent_pass = max(adjacent_maxima) <= tolerance
    independent_pass = independent_maximum <= tolerance
    peak = int(torch.cuda.max_memory_allocated(device))
    memory_pass = peak < int(program["resource_gate"]["peak_allocated_bytes_limit"])
    finite = np.concatenate(
        [
            truth.reshape(-1),
            *(
                log_ratios[stream][level].reshape(-1)
                for stream in STREAMS
                for level in levels
            ),
            [maximum_inverse_error, *adjacent_maxima, independent_maximum],
        ]
    )
    numerical_pass = bool(
        maximum_inverse_error <= 2.0e-6 and np.isfinite(finite).all()
    )
    integrity_pass = numerical_pass and memory_pass
    classification, next_step, selected = classify(
        integrity_pass, adjacent_pass, independent_pass
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_no_refit_train_only_population_pair_estimator_rank_convergence_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_sha256": frozen["v63_checkpoint_sha256"],
        "immutable_train_queries": queries,
        "donor_mapping_sha256": {
            stream: _mapping_digest(mappings[stream]) for stream in STREAMS
        },
        "rank_field_stream_sha256": {
            stream: rank_hashers[stream].hexdigest() for stream in STREAMS
        },
        "population_log_ratios": {
            stream: {
                str(level): log_ratios[stream][level].tolist() for level in levels
            }
            for stream in STREAMS
        },
        "adjacent_rank32_to_rank64": adjacent_rows,
        "independent_rank64": {
            "maximum_absolute_stream_A_to_B_log_ratio_difference": independent_maximum,
            "absolute_differences": independent_difference.tolist(),
        },
        "convergence_tolerance_ln_1_1": tolerance,
        "adjacent_convergence_pass": adjacent_pass,
        "independent_stream_convergence_pass": independent_pass,
        "rank64_estimator_selected": selected,
        "maximum_inverse_CDF_error": maximum_inverse_error,
        "numerical_pass": numerical_pass,
        "peak_allocated_bytes": peak,
        "memory_pass": memory_pass,
        "candidate_selected": selected,
        "classification": classification,
        "next": next_step,
        "gradient_computed": False,
        "training_or_refit_performed": False,
        "optimizer_step_performed": False,
        "validation_accessed": False,
        "new_development_accessed": False,
        "posthoc_rank_seed_tolerance_or_metric_tuning_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V69 refuses existing audit output")
    result = audit(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
