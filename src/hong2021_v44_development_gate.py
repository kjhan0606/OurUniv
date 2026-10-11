#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V44."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v6_gate import field_gate
from hong2021_v15_development_gate import _load_metrics, canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v44_sample import ARMS, ENSEMBLE_SCHEMA
from hong2021_v44_train import PREFLIGHT_SCHEMA, PROGRAM_SHA256, _verified_json, load_program


SCHEMA = "hong2021-v44-query-local-mixture-copula-three-domain-decision-v1"
HIGH_K_RMS_CHECKS = (
    "high_k_total_power_within_10_percent",
    "residual_rms_within_10_percent",
)


def _value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate(
    path: Path, arm: str, domain: str, parent: Path, gate_commit: str
) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "train_only_query_local_logistic_mixture_empirical_rank_copula",
            "arm": arm,
            "v44_program_sha256": PROGRAM_SHA256,
            "parent_selection_sha256": sha256_file(parent),
            "ensemble_members": 16,
            "mixture_components": 5,
            "mixture_bisection_steps": 28,
            "parameter_roll": "[16, 8, 4]",
            "structure_risk_ablated": arm == "structure_risk_ablation",
            "parameters_spatially_rolled": arm == "rolled_parameter_control",
            "conditional_rank_spatial_permutation": False,
            "conditional_rank_multiset_preserved": True,
            "global_residual_scale": 1.0,
            "object_amplitude_post_calibration": False,
            "diagnostic_k_h_mpc": 1.0,
            "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
            "hard_density_or_residual_clipping": False,
            "donor_translation": False,
            "donor_reselection": False,
            "posthoc_Ak_used": False,
            "worktree_clean_at_sampling": True,
            "Astrid_accessed": False,
            "historical_EAGLE_accessed": False,
            "complete": True,
        }
        for key, expected in exact.items():
            if _value(current.attrs.get(key)) != expected:
                raise ValueError(f"V44 {domain} {arm} metadata differs: {key}")
        reused = (
            "source_index",
            "donor_source",
            "donor_index",
            "donor_isometry",
            "donor_distance",
            "predicted_residual_dc",
            "predicted_band_scales",
        )
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64) or any(
            not np.array_equal(current[name][:], old[name][:]) for name in reused
        ):
            raise ValueError("V44 ensemble shape or frozen selection differs")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        inverse_error = float(
            np.max(np.asarray(current["maximum_inverse_CDF_error"], dtype=np.float64))
        )
        if maximum_dc > 1.0e-7 or inverse_error > 2.0e-6:
            raise ValueError("V44 residual DC or inverse CDF differs")
        checkpoint = Path(str(current.attrs["checkpoint"]))
        report = Path(str(current.attrs["training_report"]))
        cache = Path(str(current.attrs["conditioning_cache"]))
        preflight = Path(str(current.attrs["preflight"]))
        checkpoint_sha = str(current.attrs["checkpoint_sha256"])
        report_sha = str(current.attrs["training_report_sha256"])
        cache_sha = str(current.attrs["conditioning_cache_sha256"])
        preflight_sha = str(current.attrs["preflight_sha256"])
        if (
            sha256_file(checkpoint) != checkpoint_sha
            or sha256_file(report) != report_sha
            or sha256_file(cache) != cache_sha
            or sha256_file(preflight) != preflight_sha
        ):
            raise ValueError("V44 artifact hash differs")
        checked = json.loads(preflight.read_text())
        sampling_commit = str(current.attrs["sampling_code_commit"])
        if (
            checked.get("schema") != PREFLIGHT_SCHEMA
            or checked.get("status") != "pass"
            or checked.get("code_commit") != sampling_commit
        ):
            raise ValueError("V44 preflight binding differs")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", sampling_commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V44 sampling commit is not an ancestor")
        rank_digest = np.asarray(
            current["conditional_rank_multiset_sha256"], dtype=np.uint8
        )
        amplitude = np.asarray(
            current["object_amplitude_prediction"], dtype=np.float64
        )
    return {
        "sampling_code_commit": sampling_commit,
        "checkpoint_sha256": checkpoint_sha,
        "training_report_sha256": report_sha,
        "conditioning_cache_sha256": cache_sha,
        "preflight_sha256": preflight_sha,
        "maximum_absolute_sample_residual_DC": maximum_dc,
        "maximum_inverse_CDF_error": inverse_error,
        "rank_digest": rank_digest,
        "object_amplitude_prediction": amplitude,
    }


def _passes(domains: dict[str, Any]) -> tuple[bool, bool, bool]:
    q3 = all(
        abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"]) <= 0.1
        and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] <= 0.3
        for row in domains.values()
    )
    q4 = all(
        row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] <= 1.5
        for row in domains.values()
    )
    high_k_rms = all(
        all(row["field_gate"]["checks"].get(check, False) for check in HIGH_K_RMS_CHECKS)
        for row in domains.values()
    )
    return q3, q4, high_k_rms


def _rolled_improves(candidate: dict[str, Any], rolled: dict[str, Any]) -> bool:
    for domain in DOMAIN_ORDER:
        first = candidate["domains"][domain]["mechanism_Q3_Q4"]
        second = rolled["domains"][domain]["mechanism_Q3_Q4"]
        comparisons = (
            abs(second["delta_q99_999_dex"]) <= abs(first["delta_q99_999_dex"]),
            abs(second["generated_max_above_truth_max_dex"])
            <= abs(first["generated_max_above_truth_max_dex"]),
            abs(math.log(second["generated_over_truth_mean_delta_squared"]))
            <= abs(math.log(first["generated_over_truth_mean_delta_squared"])),
        )
        if not all(comparisons):
            return False
    return True


def classify(
    primary: bool,
    q3: bool,
    q4: bool,
    high_k_rms: bool,
    rolled_improves: bool,
) -> tuple[str, str]:
    if primary:
        return (
            "query_local_mixture_copula_sufficient",
            "seal_v44_and_await_explicit_approval_before_independent_gate",
        )
    if q3 and q4:
        return (
            "local_marginal_likelihood_is_calibrated_but_empirical_rank_copula_limits_morphology",
            "audit_only_conditional_rank_copula_dependence_without_changing_the_marginal_likelihood",
        )
    if high_k_rms:
        return (
            "local_mixture_body_is_supported_but_extreme_likelihood_is_insufficient",
            "audit_train_only_mixture_tail_NLL_and_component_occupancy_without_threshold_tuning",
        )
    if rolled_improves:
        return (
            "query_local_parameter_alignment_is_not_causal",
            "stop_local_neural_likelihoods_and_reassess_the_observable_information_ceiling",
        )
    return (
        "query_local_mixture_copula_is_not_a_common_domain_repair",
        "audit_candidate_vs_structure_ablation_and_V31_before_any_further_generator",
    )


def evaluate(root: Path, program_path: Path, repo: Path, commit: str) -> dict[str, Any]:
    _, v35, _ = load_program(program_path, repo)
    arms: dict[str, Any] = {}
    internal: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        domains: dict[str, Any] = {}
        internal[arm] = {}
        for domain in DOMAIN_ORDER:
            domain_root = root / arm / "development_candidate" / DOMAIN_KEYS[domain]
            ensemble = domain_root / "ensemble16.h5"
            parent = Path(v35["development_domains"][domain]["phase_object_selection"])
            provenance = _validate(ensemble, arm, domain, parent, commit)
            internal[arm][domain] = provenance
            public = {
                key: value
                for key, value in provenance.items()
                if key not in ("rank_digest", "object_amplitude_prediction")
            }
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V44 metrics point elsewhere")
            domains[domain] = {
                "ensemble": str(ensemble.resolve()),
                "ensemble_sha256": sha256_file(ensemble),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble),
                "provenance": public,
            }
        q3, q4, high_k_rms = _passes(domains)
        arms[arm] = {
            "domains": domains,
            "Q3_all_domains": q3,
            "Q4_all_domains": q4,
            "high_k_power_and_residual_RMS_all_domains": high_k_rms,
            "all_three_field_pass": all(
                row["field_gate"]["pass"] for row in domains.values()
            ),
        }
    for domain in DOMAIN_ORDER:
        reference = internal["query_local_mixture_copula"][domain]
        for arm in ARMS[1:]:
            row = internal[arm][domain]
            if (
                not np.array_equal(row["rank_digest"], reference["rank_digest"])
                or not np.array_equal(
                    row["object_amplitude_prediction"],
                    reference["object_amplitude_prediction"],
                )
            ):
                raise ValueError("V44 arms changed donor ranks or object amplitude")
    candidate = arms["query_local_mixture_copula"]
    rolled = arms["rolled_parameter_control"]
    rolled_better = _rolled_improves(candidate, rolled)
    primary = bool(
        candidate["Q3_all_domains"]
        and candidate["Q4_all_domains"]
        and candidate["all_three_field_pass"]
    )
    classification, next_step = classify(
        primary,
        bool(candidate["Q3_all_domains"]),
        bool(candidate["Q4_all_domains"]),
        bool(candidate["high_k_power_and_residual_RMS_all_domains"]),
        rolled_better,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "v44_query_local_mixture_copula",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "gate_code_commit": commit,
        "worktree_clean_at_gate": True,
        "arms": arms,
        "rolled_parameter_control_improves_Q3_Q4_every_domain": rolled_better,
        "development_pass": primary,
        "classification": classification,
        "next": next_step,
        "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "spatial_rank_transport": False,
        "global_residual_scale": 1.0,
        "object_amplitude_post_calibration": False,
        "hard_density_or_residual_clipping": False,
        "donor_translation": False,
        "donor_reselection": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V44 gate requires a clean committed worktree")
    result = evaluate(args.root.resolve(), args.program.resolve(), args.repo.resolve(), commit)
    if args.out.exists():
        raise FileExistsError("V44 refuses existing decision")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
