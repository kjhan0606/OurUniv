#!/usr/bin/env python
"""Train-only 256/128-point interval re-audit of the V56 grid tail."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v56_train import load_cache, load_program as load_v56_program
from hong2021_v56_train_gate import _load_fit
from hong2021_v58_high_accuracy_grid_tail_audit import (
    _domain as _shared_domain,
    _domain_summary as _v58_domain_summary,
)


PROGRAM_SHA256 = "10f9189095df73a0605a6fe5354c4a80382923de30d993cb84805b418146b7a7"
PROGRAM_SCHEMA = "hong2021-v59-256-128-grid-tail-audit-program-v1"
SCHEMA = "hong2021-v59-256-128-grid-tail-audit-v1"
PRIMARY_ORDER = 256
CONTROL_ORDER = 128


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V59 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V59 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v58_record"]), parent["v58_record_sha256"], "V58 record"
    )
    audit_row = record.get("audit", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or audit_row.get("classification") != parent["required_classification"]
        or audit_row.get("next") != parent["required_next"]
        or firewall.get("development_accessed")
        is not parent["required_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or firewall.get("Astrid_accessed") is not False
        or firewall.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V59 parent conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v58_program",
        "v58_audit",
        "v57_program",
        "v56_program",
        "v56_checkpoint",
        "v56_training_report",
        "v56_grid",
        "v56_preflight",
        "v56_train_gate",
        "v54_threshold_selection",
        "conditioning_cache",
        "support_selection",
    ):
        if sha256_file(_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V59 frozen input differs: {key}")
    v58_audit = _verified_json(
        _path(repo, frozen["v58_audit"]), frozen["v58_audit_sha256"], "V58 audit"
    )
    v58_program = _verified_json(
        _path(repo, frozen["v58_program"]), frozen["v58_program_sha256"], "V58 program"
    )
    v57_program = _verified_json(
        _path(repo, frozen["v57_program"]), frozen["v57_program_sha256"], "V57 program"
    )
    gate = _verified_json(
        _path(repo, frozen["v56_train_gate"]),
        frozen["v56_train_gate_sha256"],
        "V56 gate",
    )
    if (
        canonical_digest(v58_audit) != frozen["v58_audit_decision_digest_sha256"]
        or v58_audit.get("classification") != parent["required_classification"]
        or v58_audit.get("numerical_requirements_pass") is not False
        or v58_audit.get("development_accessed") is not False
        or gate.get("train_mechanism_pass") is not False
        or gate.get("development_accessed") is not False
        or gate.get("independent_gate_locked") is not True
        or program["classification"]["branches"][1:]
        != v57_program["classification"]["branches"][1:]
        or v58_program["classification"]["branches"][1:]
        != v57_program["classification"]["branches"][1:]
    ):
        raise ValueError("V59 audit, gate, or classification binding differs")
    return program, v57_program, gate


_KEY_RENAMES = {
    "high_accuracy_GL128_mean_delta_squared": "high_accuracy_GL256_mean_delta_squared",
    "control_GL64_mean_delta_squared": "control_GL128_mean_delta_squared",
    "GL64_to_GL128_relative_difference": "GL128_to_GL256_relative_difference",
    "GL128_to_exact_V56_GH64_relative_difference": "GL256_to_exact_V56_GH64_relative_difference",
    "predicted_mean_delta_squared_contribution_128": "predicted_mean_delta_squared_contribution_256",
    "predicted_mean_delta_squared_contribution_64": "predicted_mean_delta_squared_contribution_128",
    "ranked_component_moment_contributions_128": "ranked_component_moment_contributions_256",
    "ranked_component_moment_contributions_64": "ranked_component_moment_contributions_128",
    "ranked_component_moment_shares_128": "ranked_component_moment_shares_256",
    "predicted_mean_delta_squared_tail_contribution_128": "predicted_mean_delta_squared_tail_contribution_256",
    "predicted_mean_delta_squared_tail_contribution_64": "predicted_mean_delta_squared_tail_contribution_128",
    "tail_moment_64_to_128_relative_difference": "tail_moment_128_to_256_relative_difference",
}


def _rename_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {_KEY_RENAMES.get(key, key): _rename_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_keys(item) for item in value]
    return value


def _domain_summary(
    truth_log10rho: np.ndarray,
    truth_delta_squared: np.ndarray,
    exact_v56_values: np.ndarray,
    primary: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    component_mass_sums: np.ndarray,
    thresholds: np.ndarray,
    grid_weights: np.ndarray,
    sealed: dict[str, Any],
    numerics: dict[str, Any],
) -> dict[str, Any]:
    mapped_numerics = {
        "maximum_exact_V56_gate_reproduction_relative_difference": numerics[
            "maximum_exact_V56_gate_reproduction_relative_difference"
        ],
        "maximum_64_to_128_complete_moment_relative_difference": numerics[
            "maximum_128_to_256_complete_moment_relative_difference"
        ],
        "maximum_64_to_128_tail_moment_relative_difference": numerics[
            "maximum_128_to_256_tail_moment_relative_difference"
        ],
        "maximum_high_accuracy_complete_moment_relative_difference_from_exact_V56_64": numerics[
            "maximum_high_accuracy_complete_moment_relative_difference_from_exact_V56_64"
        ],
        "maximum_bin_partition_relative_error": numerics[
            "maximum_bin_partition_relative_error"
        ],
        "maximum_component_partition_relative_error": numerics[
            "maximum_component_partition_relative_error"
        ],
        "maximum_log_ratio_identity_absolute_error": numerics[
            "maximum_log_ratio_identity_absolute_error"
        ],
        "minimum_empirical_exceedance_count_for_threshold_classification": numerics[
            "minimum_empirical_exceedance_count_for_threshold_classification"
        ],
    }
    row = _rename_keys(
        _v58_domain_summary(
            truth_log10rho,
            truth_delta_squared,
            exact_v56_values,
            primary,
            control,
            component_mass_sums,
            thresholds,
            grid_weights,
            sealed,
            mapped_numerics,
        )
    )
    primary_moments = np.asarray(primary["component_moment_bins"])
    control_moments = np.asarray(control["component_moment_bins"])
    probabilities = np.asarray(primary["component_probability_bins"])
    primary_total = float(primary_moments.sum(dtype=np.float64))
    control_total = float(control_moments.sum(dtype=np.float64))
    raw_mass_total = float(component_mass_sums.sum(dtype=np.float64))
    probability_total = float(probabilities.sum(dtype=np.float64))
    bin_partition_error = max(
        _relative_error(float(primary_moments.sum()), primary_total),
        _relative_error(float(control_moments.sum()), control_total),
        _relative_error(probability_total, raw_mass_total),
    )
    component_partition_error = bin_partition_error
    mass_normalization_error = _relative_error(raw_mass_total, len(truth_log10rho))
    row["bin_partition_relative_error"] = bin_partition_error
    row["component_partition_relative_error"] = component_partition_error
    row["raw_mixture_mass_relative_difference_from_exact_count"] = mass_normalization_error
    complete = row["complete_moment"]
    row["numerical_requirements_pass"] = bool(
        row["reproduces_V56_gate"]
        and row["threshold_decomposition"]["q99_999_anchor"]["ratio_available"]
        and complete["GL128_to_GL256_relative_difference"]
        <= float(numerics["maximum_128_to_256_complete_moment_relative_difference"])
        and complete["GL256_to_exact_V56_GH64_relative_difference"]
        <= float(
            numerics[
                "maximum_high_accuracy_complete_moment_relative_difference_from_exact_V56_64"
            ]
        )
        and row["tail_quadrature_convergence_pass"]
        and row["log_ratio_identity_pass"]
        and bin_partition_error <= float(numerics["maximum_bin_partition_relative_error"])
        and component_partition_error
        <= float(numerics["maximum_component_partition_relative_error"])
    )
    return row


def classify(numerical_requirements_pass: bool, tng: dict[str, Any]) -> tuple[str, str]:
    if not numerical_requirements_pass:
        return (
            "V59_256_128_grid_tail_decomposition_is_numerically_unresolved",
            "freeze_only_the_minimal_train_only_numerical_repair_without_training_or_development_access",
        )
    regions = tng["regions"]
    if regions["beyond_grid"]["positive_excess_share"] >= 0.5:
        return (
            "V56_TNG_moment_excess_lies_beyond_scored_global_train_maximum",
            "freeze_one_matched_train_only_model_that_extends_the_proper_survival_grid_over_the_immutable_reachable_output_support_without_changing_other_model_or_training_choices",
        )
    if regions["below_grid"]["positive_excess_share"] >= 0.5:
        return (
            "V56_TNG_moment_excess_lies_below_the_upper_survival_grid",
            "freeze_one_matched_train_only_model_that_extends_the_proper_survival_grid_downward_to_the_immutable_q99_9_output_threshold_without_changing_other_model_or_training_choices",
        )
    summary = tng["supported_grid_error_summary"]
    if regions["inside_grid"]["positive_excess_share"] >= 0.5 and summary["available"]:
        if (
            summary["weighted_mean_absolute_log_probability_ratio"]
            > summary["weighted_mean_absolute_log_conditional_amplitude_ratio"]
        ):
            return (
                "V56_TNG_scored_grid_survival_probabilities_remain_miscalibrated",
                "freeze_one_matched_train_only_model_that_changes_only_the_predeclared_upper_survival_score_coefficient",
            )
        return (
            "V56_TNG_scored_grid_is_too_coarse_for_conditional_tail_amplitude",
            "freeze_one_matched_train_only_model_that_changes_only_the_predeclared_upper_survival_grid_resolution",
        )
    return (
        "V56_TNG_remaining_moment_excess_is_mixed_across_grid_regions",
        "seal_the_domainwise_grid_and_component_decomposition_before_selecting_any_further_model",
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v57_program, gate = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V59 audit requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V59 audit requires Ada")
    frozen = program["frozen_inputs"]
    v56_program, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    if Path(v56_program["output_roots"]["development"]).exists():
        raise RuntimeError("V59 refuses a pre-existing V56 development directory")
    model, checkpoint = _load_fit(
        v56_program,
        _path(repo, frozen["v56_checkpoint"]),
        frozen["v56_checkpoint_sha256"],
        _path(repo, frozen["v56_training_report"]),
        frozen["v56_training_report_sha256"],
        _path(repo, frozen["v56_grid"]),
        frozen["v56_grid_sha256"],
        frozen["v54_threshold_selection_sha256"],
        _path(repo, frozen["v56_preflight"]),
        frozen["v56_preflight_sha256"],
        frozen["conditioning_cache_sha256"],
        repo,
        commit,
    )
    model = model.to("cuda").eval()
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        str(checkpoint["code_commit"]),
    )
    support = _verified_json(
        _path(repo, frozen["support_selection"]),
        frozen["support_selection_sha256"],
        "support selection",
    )
    partition = v57_program["fixed_threshold_partition"]
    thresholds = np.asarray(
        [partition["lower_anchor_log10rho"], *partition["scored_grid_thresholds_log10rho"]],
        dtype=np.float64,
    )
    grid_weights = np.asarray(
        partition["scored_grid_physical_moment_weights"], dtype=np.float64
    )
    domains: dict[str, Any] = {}
    try:
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            domains[domain] = _shared_domain(
                model,
                torch.device("cuda"),
                v35,
                prepared,
                support,
                gate,
                domain,
                domain_index,
                thresholds,
                grid_weights,
                program["numerics"],
                primary_order=PRIMARY_ORDER,
                control_order=CONTROL_ORDER,
                summary_function=_domain_summary,
                progress_label="v59-audit",
            )
    finally:
        prepared.close()
    numerical_pass = all(row["numerical_requirements_pass"] for row in domains.values())
    classification, next_action = classify(numerical_pass, domains["TNG100"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_256_128_train_only_grid_tail_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint_sha256": frozen["v56_checkpoint_sha256"],
        "v58_audit_sha256": frozen["v58_audit_sha256"],
        "v56_train_gate_sha256": frozen["v56_train_gate_sha256"],
        "primary_interval_quadrature_order": PRIMARY_ORDER,
        "control_interval_quadrature_order": CONTROL_ORDER,
        "combined_thresholds_log10rho": thresholds.tolist(),
        "domains": domains,
        "numerical_requirements_pass": numerical_pass,
        "classification": classification,
        "next": next_action,
        "training_or_refit_performed": False,
        "validation_accessed": False,
        "development_accessed": False,
        "new_development_sample_generated": False,
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V59 refuses an existing audit")
    result = audit(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
