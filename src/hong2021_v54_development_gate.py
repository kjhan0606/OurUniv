#!/usr/bin/env python
"""Locked three-domain development gate for V54 after train mechanism pass."""
from __future__ import annotations

import argparse
import json
import os
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
from hong2021_v52_development_gate import _extreme_comparison
from hong2021_v54_sample import ARMS, ENSEMBLE_SCHEMA, METHOD
from hong2021_v54_train import PROGRAM_SHA256, SUPPORT_SHA256, TAIL_COEFFICIENT, load_program
from hong2021_v54_train_gate import SCHEMA as TRAIN_GATE_SCHEMA


SCHEMA = "hong2021-v54-physical-tail-brier-three-domain-decision-v1"
CANDIDATE = "bounded_query_local_mixture_copula"


def classify(
    primary: bool,
    q3: bool,
    q4: bool,
    improves_both: bool,
    reference_dominates: bool,
    rolled_improves: bool,
) -> tuple[str, str]:
    if primary:
        return (
            "train_only_proper_tail_score_is_development_sufficient",
            "seal_V54_and_await_explicit_approval_before_independent_EAGLE_gate",
        )
    if q3 and q4:
        return (
            "proper_tail_marginal_is_calibrated_but_empirical_rank_copula_limits_morphology",
            "audit_only_conditional_rank_copula_dependence_without_changing_the_V54_likelihood_or_score",
        )
    if improves_both:
        return (
            "proper_tail_score_improves_all_extremes_but_is_not_development_sufficient",
            "seal_the_score_effect_and_audit_only_the_remaining_failed_field_or_absolute_extreme_threshold",
        )
    if reference_dominates:
        return (
            "proper_tail_score_does_not_transfer_from_train_mechanism_to_development",
            "stop_tail_score_training_and_reassess_train_to_development_condition_shift",
        )
    if rolled_improves:
        return (
            "V54_query_local_tail_parameters_are_not_causal",
            "stop_local_neural_likelihoods_and_reassess_the_observable_information_ceiling",
        )
    return (
        "V54_proper_tail_score_result_is_mixed_or_not_a_common_domain_repair",
        "seal_train_and_development_evidence_before_selecting_any_further_model",
    )


def _validate(path: Path, arm: str, domain: str, parent: Path, train_gate: dict[str, Any]) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": METHOD,
            "arm": arm,
            "v54_program_sha256": PROGRAM_SHA256,
            "support_selection_sha256": SUPPORT_SHA256,
            "standardized_support_lower": -13.78839653180272,
            "standardized_support_upper": 10.259036149654781,
            "ensemble_members": 16,
            "mixture_components": 5,
            "mixture_bisection_steps": 28,
            "parameter_roll": "[16, 8, 4]",
            "parameters_spatially_rolled": arm == "rolled_parameter_control",
            "conditional_rank_spatial_permutation": False,
            "conditional_rank_multiset_preserved": True,
            "global_residual_scale": 1.0,
            "object_amplitude_post_calibration": False,
            "tail_coefficient": TAIL_COEFFICIENT,
            "structure_risk_unchanged_from_V50": True,
            "train_mechanism_pass": True,
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
                raise ValueError(f"V54 {domain} {arm} metadata differs: {key}")
        for name in ("source_index", "donor_source", "donor_index", "donor_isometry", "donor_distance", "predicted_residual_dc", "predicted_band_scales"):
            if not np.array_equal(current[name][:], old[name][:]):
                raise ValueError(f"V54 {domain} frozen selection differs: {name}")
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V54 ensemble shape differs")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(current["conditional_mean"], dtype=np.float32)[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        inverse_error = float(np.max(current["maximum_inverse_CDF_error"][:]))
        checkpoint_sha = str(current.attrs["checkpoint_sha256"])
        report_sha = str(current.attrs["training_report_sha256"])
        preflight_sha = str(current.attrs["preflight_sha256"])
        threshold_sha = str(current.attrs["threshold_selection_sha256"])
        train_gate_sha = str(current.attrs["train_mechanism_gate_sha256"])
        checkpoint_path = Path(str(current.attrs["checkpoint"]))
        report_path = Path(str(current.attrs["training_report"]))
        preflight_path = Path(str(current.attrs["preflight"]))
        threshold_path = Path(str(current.attrs["threshold_selection"]))
        train_gate_path = Path(str(current.attrs["train_mechanism_gate"]))
    if (
        maximum_dc > 1.0e-7
        or inverse_error > 2.0e-6
        or checkpoint_sha != train_gate["checkpoint_sha256"]
        or report_sha != train_gate["training_report_sha256"]
        or preflight_sha != train_gate["preflight_sha256"]
        or threshold_sha != train_gate["threshold_selection_sha256"]
        or sha256_file(checkpoint_path) != checkpoint_sha
        or sha256_file(report_path) != report_sha
        or sha256_file(preflight_path) != preflight_sha
        or sha256_file(threshold_path) != threshold_sha
        or sha256_file(train_gate_path) != train_gate_sha
    ):
        raise ValueError("V54 sample integrity differs")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "training_report_sha256": report_sha,
        "preflight_sha256": preflight_sha,
        "threshold_selection_sha256": threshold_sha,
        "train_mechanism_gate_sha256": train_gate_sha,
        "maximum_absolute_sample_residual_DC": maximum_dc,
        "maximum_inverse_CDF_error": inverse_error,
    }


def evaluate(root: Path, program_path: Path, repo: Path, train_gate_path: Path, train_gate_sha: str) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V54 development gate requires clean worktree")
    if sha256_file(train_gate_path) != train_gate_sha:
        raise ValueError("V54 train gate hash differs")
    train_gate = json.loads(train_gate_path.read_text())
    if (
        train_gate.get("schema") != TRAIN_GATE_SCHEMA
        or train_gate.get("train_mechanism_pass") is not True
        or canonical_digest(train_gate) != train_gate.get("decision_digest_sha256")
        or train_gate.get("development_accessed") is not False
    ):
        raise ValueError("V54 train mechanism gate binding differs")
    v50 = json.loads(Path(program["frozen_inputs"]["v50_development_decision"]).read_text())
    v52 = json.loads(Path(program["frozen_inputs"]["v52_development_decision"]).read_text())
    arms: dict[str, Any] = {}
    for arm in ARMS:
        domains: dict[str, Any] = {}
        for domain in DOMAIN_ORDER:
            domain_root = root / arm / "development_candidate" / DOMAIN_KEYS[domain]
            ensemble = domain_root / "ensemble16.h5"
            parent = Path(v35["development_domains"][domain]["phase_object_selection"])
            provenance = _validate(ensemble, arm, domain, parent, train_gate)
            if provenance["train_mechanism_gate_sha256"] != train_gate_sha:
                raise ValueError("V54 ensemble train gate hash differs")
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V54 metrics point elsewhere")
            domains[domain] = {
                "ensemble": str(ensemble.resolve()),
                "ensemble_sha256": sha256_file(ensemble),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble),
                "provenance": provenance,
            }
        q3, q4, high_k = _passes(domains)
        arms[arm] = {
            "domains": domains,
            "Q3_all_domains": q3,
            "Q4_all_domains": q4,
            "high_k_power_and_residual_RMS_all_domains": high_k,
            "all_three_field_pass": all(row["field_gate"]["pass"] for row in domains.values()),
        }
    candidate, rolled = arms[CANDIDATE], arms["rolled_parameter_control"]
    comparisons: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        current = candidate["domains"][domain]["mechanism_Q3_Q4"]
        refs = {
            "V50": v50["arms"]["bounded_query_local_mixture_copula"]["domains"][domain]["mechanism_Q3_Q4"],
            "V52": v52["arms"]["no_risk_query_local_mixture_copula"]["domains"][domain]["mechanism_Q3_Q4"],
        }
        comparisons[domain] = {name: _extreme_comparison(current, value) for name, value in refs.items()}
    improves_both = all(all(row["candidate_strictly_improves_all_three"] for row in comparisons[domain].values()) for domain in DOMAIN_ORDER)
    reference_dominates = all(any(row["reference_equals_or_improves_all_three"] for row in comparisons[domain].values()) for domain in DOMAIN_ORDER)
    rolled_improves = _rolled_improves(candidate, rolled)
    primary = bool(candidate["Q3_all_domains"] and candidate["Q4_all_domains"] and candidate["all_three_field_pass"])
    classification, next_step = classify(primary, bool(candidate["Q3_all_domains"]), bool(candidate["Q4_all_domains"]), improves_both, reference_dominates, rolled_improves)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "v54_physical_tail_brier_bounded_mixture",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "gate_code_commit": commit,
        "worktree_clean_at_gate": clean,
        "train_mechanism_gate": str(train_gate_path.resolve()),
        "train_mechanism_gate_sha256": train_gate_sha,
        "train_mechanism_pass": True,
        "arms": arms,
        "comparison_to_sealed_V50_and_V52": comparisons,
        "V54_strictly_improves_all_three_over_both_references_every_domain": improves_both,
        "a_sealed_reference_equals_or_improves_all_three_every_domain": reference_dominates,
        "rolled_parameter_control_improves_Q3_Q4_every_domain": rolled_improves,
        "development_pass": primary,
        "classification": classification,
        "next": next_step,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
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
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--train-gate-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V54 refuses existing development decision")
    result = evaluate(args.root.resolve(), args.program.resolve(), args.repo.resolve(), args.train_gate.resolve(), args.train_gate_sha256)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
