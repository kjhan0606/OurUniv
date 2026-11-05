#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V52 matched no-risk fit."""
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
from hong2021_v48_development_gate import _passes, _rolled_improves, _value
from hong2021_v50_network import LOWER_SUPPORT, UPPER_SUPPORT
from hong2021_v52_sample import ARMS, ENSEMBLE_SCHEMA, METHOD
from hong2021_v52_train import (
    LIKELIHOOD_FAMILY,
    PREFLIGHT_SCHEMA,
    PROGRAM_SHA256,
    SUPPORT_SHA256,
    load_program,
)


SCHEMA = "hong2021-v52-matched-no-risk-bounded-mixture-three-domain-decision-v1"
CANDIDATE = "no_risk_query_local_mixture_copula"


def _validate(
    path: Path, arm: str, domain: str, parent: Path, gate_commit: str
) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": METHOD,
            "arm": arm,
            "v52_program_sha256": PROGRAM_SHA256,
            "support_selection_sha256": SUPPORT_SHA256,
            "standardized_support_lower": LOWER_SUPPORT,
            "standardized_support_upper": UPPER_SUPPORT,
            "parent_selection_sha256": sha256_file(parent),
            "ensemble_members": 16,
            "mixture_components": 5,
            "likelihood_family": LIKELIHOOD_FAMILY,
            "mixture_bisection_steps": 28,
            "parameter_roll": "[16, 8, 4]",
            "structure_risk_channel_exact_standardized_zero": True,
            "parameters_spatially_rolled": arm == "rolled_parameter_control",
            "conditional_rank_spatial_permutation": False,
            "conditional_rank_multiset_preserved": True,
            "global_residual_scale": 1.0,
            "object_amplitude_post_calibration": False,
            "diagnostic_k_h_mpc": 1.0,
            "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
            "hard_density_or_residual_clipping": False,
            "sample_clipping": False,
            "component_scale_cap": False,
            "donor_translation": False,
            "donor_reselection": False,
            "posthoc_Ak_used": False,
            "worktree_clean_at_sampling": True,
            "Astrid_accessed": False,
            "historical_EAGLE_accessed": False,
            "independent_gate_locked": True,
            "complete": True,
        }
        for key, expected in exact.items():
            if _value(current.attrs.get(key)) != expected:
                raise ValueError(f"V52 {domain} {arm} metadata differs: {key}")
        if not (
            LOWER_SUPPORT
            < float(current.attrs["minimum_pre_DC_standardized_sample"])
            <= float(current.attrs["maximum_pre_DC_standardized_sample"])
            < UPPER_SUPPORT
        ):
            raise ValueError("V52 pre-DC sample support differs")
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
            raise ValueError("V52 ensemble shape or frozen selection differs")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        inverse_error = float(
            np.max(np.asarray(current["maximum_inverse_CDF_error"], dtype=np.float64))
        )
        if maximum_dc > 1.0e-7 or inverse_error > 2.0e-6:
            raise ValueError("V52 residual DC or inverse CDF differs")
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
            raise ValueError("V52 artifact hash differs")
        checked = json.loads(preflight.read_text())
        sampling_commit = str(current.attrs["sampling_code_commit"])
        if (
            checked.get("schema") != PREFLIGHT_SCHEMA
            or checked.get("status") != "pass"
            or checked.get("code_commit") != sampling_commit
            or checked.get("risk_channel_exact_standardized_zero") is not True
            or checked.get("support_selection_sha256") != SUPPORT_SHA256
            or checked.get("sample_clipping") is not False
            or checked.get("component_scale_cap") is not False
        ):
            raise ValueError("V52 preflight binding differs")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", sampling_commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V52 sampling commit is not an ancestor")
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
        "support_selection_sha256": SUPPORT_SHA256,
        "maximum_absolute_sample_residual_DC": maximum_dc,
        "maximum_inverse_CDF_error": inverse_error,
        "rank_digest": rank_digest,
        "object_amplitude_prediction": amplitude,
    }


def _extreme_comparison(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    candidate_q = abs(float(candidate["delta_q99_999_dex"]))
    reference_q = abs(float(reference["delta_q99_999_dex"]))
    candidate_max = float(candidate["generated_max_above_truth_max_dex"])
    reference_max = float(reference["generated_max_above_truth_max_dex"])
    candidate_moment = abs(
        math.log(float(candidate["generated_over_truth_mean_delta_squared"]))
    )
    reference_moment = abs(
        math.log(float(reference["generated_over_truth_mean_delta_squared"]))
    )
    return {
        "candidate_absolute_delta_q99_999_dex": candidate_q,
        "reference_absolute_delta_q99_999_dex": reference_q,
        "q99_999_strictly_improves": candidate_q < reference_q,
        "candidate_generated_max_above_truth_max_dex": candidate_max,
        "reference_generated_max_above_truth_max_dex": reference_max,
        "maximum_excess_strictly_improves": candidate_max < reference_max,
        "candidate_absolute_log_mean_delta_squared_ratio": candidate_moment,
        "reference_absolute_log_mean_delta_squared_ratio": reference_moment,
        "mean_delta_squared_ratio_strictly_improves": candidate_moment
        < reference_moment,
        "candidate_strictly_improves_all_three": (
            candidate_q < reference_q
            and candidate_max < reference_max
            and candidate_moment < reference_moment
        ),
        "reference_equals_or_improves_all_three": (
            reference_q <= candidate_q
            and reference_max <= candidate_max
            and reference_moment <= candidate_moment
        ),
    }


def classify(
    primary: bool,
    q3: bool,
    q4: bool,
    strict_improvement: bool,
    q4_improves: bool,
    high_k_rms: bool,
    reference_better: bool,
    rolled_improves: bool,
) -> tuple[str, str]:
    if primary:
        return (
            "matched_no_risk_bounded_mixture_is_development_sufficient",
            "seal_V52_and_await_explicit_approval_before_independent_EAGLE_gate",
        )
    if q3 and q4:
        return (
            "no_risk_marginal_is_calibrated_but_empirical_rank_copula_limits_morphology",
            "audit_only_conditional_rank_copula_dependence_without_changing_the_no_risk_bounded_likelihood",
        )
    if strict_improvement:
        return (
            "structure_risk_is_a_causal_amplifier_but_removal_is_not_sufficient",
            "audit_train_only_high_backbone_conditional_calibration_and_strictly_proper_physical_tail_scores_before_one_new_model",
        )
    if q4_improves and not high_k_rms:
        return (
            "structure_risk_removal_reduces_extremes_but_damages_the_stochastic_field_body",
            "freeze_a_train_only_calibrated_continuous_risk_conditioning_test_instead_of_binary_feature_removal",
        )
    if reference_better:
        return (
            "inference_only_risk_ablation_did_not_survive_matched_retraining",
            "stop_the_risk_removal_branch_and_audit_high_backbone_conditional_calibration",
        )
    if rolled_improves:
        return (
            "no_risk_query_local_parameter_alignment_is_not_causal",
            "stop_local_neural_likelihoods_and_reassess_the_observable_information_ceiling",
        )
    return (
        "matched_no_risk_result_is_mixed_or_not_a_common_domain_repair",
        "audit_V52_vs_V50_domainwise_extremes_field_checks_and_high_backbone_strata_before_any_new_model",
    )


def evaluate(root: Path, program_path: Path, repo: Path, commit: str) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo)
    v50_path = Path(program["frozen_inputs"]["v50_development_decision"])
    v50 = json.loads(v50_path.read_text())
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
                raise ValueError("V52 metrics point elsewhere")
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
        reference = internal[CANDIDATE][domain]
        row = internal["rolled_parameter_control"][domain]
        if (
            not np.array_equal(row["rank_digest"], reference["rank_digest"])
            or not np.array_equal(
                row["object_amplitude_prediction"],
                reference["object_amplitude_prediction"],
            )
        ):
            raise ValueError("V52 arms changed donor ranks or object amplitude")
    candidate = arms[CANDIDATE]
    rolled = arms["rolled_parameter_control"]
    comparisons = {
        domain: _extreme_comparison(
            candidate["domains"][domain]["mechanism_Q3_Q4"],
            v50["arms"]["bounded_query_local_mixture_copula"]["domains"][domain][
                "mechanism_Q3_Q4"
            ],
        )
        for domain in DOMAIN_ORDER
    }
    strict_improvement = all(
        row["candidate_strictly_improves_all_three"] for row in comparisons.values()
    )
    q4_improves = all(
        row["mean_delta_squared_ratio_strictly_improves"]
        for row in comparisons.values()
    )
    reference_better = all(
        row["reference_equals_or_improves_all_three"]
        for row in comparisons.values()
    )
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
        strict_improvement,
        q4_improves,
        bool(candidate["high_k_power_and_residual_RMS_all_domains"]),
        reference_better,
        rolled_better,
    )
    sealed_reference = {
        "path": str(v50_path.resolve()),
        "sha256": sha256_file(v50_path),
        "decision_digest_sha256": v50["decision_digest_sha256"],
        "domains": {
            domain: {
                "ensemble_sha256": v50["arms"][
                    "bounded_query_local_mixture_copula"
                ]["domains"][domain]["ensemble_sha256"],
                "metrics_sha256": v50["arms"][
                    "bounded_query_local_mixture_copula"
                ]["domains"][domain]["metrics_sha256"],
                "field_pass": v50["arms"]["bounded_query_local_mixture_copula"][
                    "domains"
                ][domain]["field_gate"]["pass"],
                "mechanism_Q3_Q4": v50["arms"][
                    "bounded_query_local_mixture_copula"
                ]["domains"][domain]["mechanism_Q3_Q4"],
            }
            for domain in DOMAIN_ORDER
        },
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "v52_matched_no_risk_bounded_mixture",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "gate_code_commit": commit,
        "worktree_clean_at_gate": True,
        "arms": arms,
        "sealed_V50_risk_model_reference": sealed_reference,
        "V52_vs_V50_extreme_comparison": comparisons,
        "V52_strictly_improves_all_three_extreme_metrics_every_domain": strict_improvement,
        "V52_improves_Q4_every_domain": q4_improves,
        "sealed_V50_equals_or_improves_all_three_every_domain": reference_better,
        "rolled_parameter_control_improves_Q3_Q4_every_domain": rolled_better,
        "development_pass": primary,
        "classification": classification,
        "next": next_step,
        "support_selection_sha256": SUPPORT_SHA256,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "risk_channel_exact_standardized_zero": True,
        "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "spatial_rank_transport": False,
        "global_residual_scale": 1.0,
        "object_amplitude_post_calibration": False,
        "hard_density_or_residual_clipping": False,
        "sample_clipping": False,
        "component_scale_cap": False,
        "donor_translation": False,
        "donor_reselection": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
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
        raise RuntimeError("V52 gate requires a clean committed worktree")
    result = evaluate(
        args.root.resolve(), args.program.resolve(), args.repo.resolve(), commit
    )
    if args.out.exists():
        raise FileExistsError("V52 refuses existing decision")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
