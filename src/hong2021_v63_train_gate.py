#!/usr/bin/env python
"""Locked train-only high-backbone mechanism gate for V63."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v50_network import LocalMixtureUNet, parameter_count
from hong2021_v50_train import PARAMETERS
from hong2021_v56_train_gate import (
    QUADRATURE_MAXIMUM,
    RATIO_MAXIMUM,
    RATIO_MINIMUM,
    _domain as v56_domain,
    mechanism_pass,
)
from hong2021_v63_preflight import PROGRAM_SHA256, _path
from hong2021_v63_train import (
    CANDIDATE_IMPLEMENTATION_SHA256,
    CHECKPOINT_SCHEMA,
    MOMENT_COEFFICIENT,
    PREFLIGHT_DECISION_DIGEST,
    PREFLIGHT_IMPLEMENTATION_SHA256,
    PREFLIGHT_RECORD_SHA256,
    QUADRATURE_ORDER,
    REPORT_SCHEMA,
    STEPS,
    _is_ancestor,
    _load_training_inputs,
)


SCHEMA = "hong2021-v63-train-only-high-backbone-mechanism-decision-v1"


def _rename_model_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key.replace("V56_", "V63_"): _rename_model_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_model_fields(item) for item in value]
    return value


def _load_fit(
    checkpoint_path: Path,
    checkpoint_sha: str,
    report_path: Path,
    report_sha: str,
    grid_sha: str,
    threshold_sha: str,
    preflight_sha: str,
    cache_sha: str,
    support_sha: str,
    boundaries: dict[str, float],
    repo: Path,
    commit: str,
) -> tuple[LocalMixtureUNet, dict[str, Any]]:
    if (
        sha256_file(checkpoint_path) != checkpoint_sha
        or sha256_file(report_path) != report_sha
    ):
        raise ValueError("V63 checkpoint or report hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = json.loads(
        report_path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    source_commit = str(checkpoint.get("code_commit"))
    common = (
        checkpoint.get("schema") == CHECKPOINT_SCHEMA
        and checkpoint.get("program_sha256") == PROGRAM_SHA256
        and checkpoint.get("step") == STEPS
        and checkpoint.get("parameters") == PARAMETERS
        and checkpoint.get("v54_threshold_selection_sha256") == threshold_sha
        and checkpoint.get("grid_sha256") == grid_sha
        and checkpoint.get("grid_cells") == 16
        and checkpoint.get("moment_coefficient") == MOMENT_COEFFICIENT
        and checkpoint.get("moment_quadrature_order") == QUADRATURE_ORDER
        and checkpoint.get("q99_9_backbone_boundaries") == boundaries
        and checkpoint.get("preflight_sha256") == preflight_sha
        and checkpoint.get("preflight_decision_digest_sha256")
        == PREFLIGHT_DECISION_DIGEST
        and checkpoint.get("preflight_implementation_sha256")
        == PREFLIGHT_IMPLEMENTATION_SHA256
        and checkpoint.get("candidate_implementation_sha256")
        == CANDIDATE_IMPLEMENTATION_SHA256
        and checkpoint.get("preflight_record_sha256") == PREFLIGHT_RECORD_SHA256
        and checkpoint.get("conditioning_cache_sha256") == cache_sha
        and checkpoint.get("support_selection_sha256") == support_sha
        and checkpoint.get("independent_gate_locked") is True
        and checkpoint.get(
            "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection"
        )
        is False
    )
    if (
        not common
        or report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "complete_fixed_12000_step_conditional_moment_fit"
        or report.get("code_commit") != source_commit
        or report.get("checkpoint_sha256") != checkpoint_sha
        or report.get("grid_sha256") != grid_sha
        or report.get("preflight_sha256") != preflight_sha
        or report.get("moment_coefficient") != MOMENT_COEFFICIENT
        or report.get("q99_9_backbone_boundaries") != boundaries
        or report.get("development_accessed") is not False
        or report.get("historical_EAGLE_accessed") is not False
        or report.get("independent_gate_locked") is not True
        or canonical_digest(report) != report.get("decision_digest_sha256")
        or not _is_ancestor(repo, source_commit, commit)
    ):
        raise ValueError("V63 fit binding differs")
    model = LocalMixtureUNet()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V63 architecture differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model, checkpoint


def evaluate(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    threshold_path: Path,
    threshold_sha: str,
    grid_path: Path,
    grid_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    report_path: Path,
    report_sha: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V63 mechanism gate requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V63 mechanism gate requires the Lageunha Ada GPU")
    program, v35, _, _, boundaries = _load_training_inputs(
        program_path,
        repo,
        cache_path,
        cache_sha,
        threshold_path,
        threshold_sha,
        grid_path,
        grid_sha,
        preflight_path,
        preflight_sha,
        commit,
    )
    model, checkpoint = _load_fit(
        checkpoint_path,
        checkpoint_sha,
        report_path,
        report_sha,
        grid_sha,
        threshold_sha,
        preflight_sha,
        cache_sha,
        program["frozen_inputs"]["support_selection_sha256"],
        boundaries,
        repo,
        commit,
    )
    model = model.to("cuda").eval()
    from hong2021_v48_train import load_cache

    prepared = load_cache(cache_path, cache_sha, str(checkpoint["code_commit"]))
    frozen = program["frozen_inputs"]
    v56_program_path = _path(repo, frozen["v56_program"])
    v56_program = json.loads(v56_program_path.read_text())
    v56_frozen = v56_program["frozen_inputs"]
    support = json.loads(_path(repo, frozen["support_selection"]).read_text())
    v54_program_path = _path(repo, v56_frozen["v54_program"])
    if sha256_file(v54_program_path) != v56_frozen["v54_program_sha256"]:
        raise ValueError("V63 V54 program hash differs")
    v54_program = json.loads(v54_program_path.read_text())
    v53_path = _path(repo, v54_program["frozen_inputs"]["v53_audit"])
    if sha256_file(v53_path) != v54_program["frozen_inputs"]["v53_audit_sha256"]:
        raise ValueError("V63 V53 audit hash differs")
    v53 = json.loads(v53_path.read_text())
    v54_gate_path = _path(repo, v56_frozen["v54_train_gate"])
    if sha256_file(v54_gate_path) != v56_frozen["v54_train_gate_sha256"]:
        raise ValueError("V63 V54 gate hash differs")
    v54_gate = json.loads(v54_gate_path.read_text())
    domains: dict[str, Any] = {}
    try:
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            domains[domain] = _rename_model_fields(
                v56_domain(
                    model,
                    torch.device("cuda"),
                    v35,
                    prepared,
                    support,
                    v53,
                    v54_gate,
                    domain,
                    domain_index,
                )
            )
    finally:
        prepared.close()
    ratios = {
        domain: float(
            row["strata"]["q99_9_and_above"]["V63_over_truth_mean_delta_squared"]
        )
        for domain, row in domains.items()
    }
    convergence = {
        domain: float(row["aggregate_32_to_64_relative_difference"])
        for domain, row in domains.items()
    }
    passed = mechanism_pass(ratios, convergence)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only_mechanism_gate",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_sha256": checkpoint_sha,
        "training_report_sha256": report_sha,
        "grid_sha256": grid_sha,
        "preflight_sha256": preflight_sha,
        "moment_coefficient": MOMENT_COEFFICIENT,
        "allowed_top_backbone_interval": [RATIO_MINIMUM, RATIO_MAXIMUM],
        "maximum_quadrature_relative_difference": QUADRATURE_MAXIMUM,
        "domains": domains,
        "train_mechanism_pass": passed,
        "classification": (
            "conditional_log_physical_moment_objective_calibrates_all_train_high_backbone_domains"
            if passed
            else "conditional_log_physical_moment_objective_does_not_calibrate_all_train_high_backbone_domains"
        ),
        "next": (
            "proceed_to_locked_V63_development_sampling_and_evaluation"
            if passed
            else "stop_before_development_sampling_and_audit_the_fixed_training_response_without_posthoc_tuning"
        ),
        "development_accessed": False,
        "training_or_refit_performed_by_gate": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--thresholds-sha256", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--grid-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V63 refuses existing train mechanism gate")
    result = evaluate(
        args.program,
        args.repo,
        args.cache,
        args.cache_sha256,
        args.thresholds,
        args.thresholds_sha256,
        args.grid,
        args.grid_sha256,
        args.preflight,
        args.preflight_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
        args.report,
        args.report_sha256,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
