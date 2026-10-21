#!/usr/bin/env python
"""Locked three-domain development gate for train-gate-approved V63."""
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
from hong2021_v50_train import SUPPORT_SHA256
from hong2021_v52_development_gate import _extreme_comparison
from hong2021_v63_preflight import PROGRAM_SHA256, _path, load_program
from hong2021_v63_sample import ARMS, ENSEMBLE_SCHEMA, METHOD
from hong2021_v63_train import MOMENT_COEFFICIENT, QUADRATURE_ORDER, _is_ancestor
from hong2021_v63_train_gate import SCHEMA as TRAIN_GATE_SCHEMA


SCHEMA = "hong2021-v63-conditional-log-physical-moment-three-domain-decision-v1"
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
            "conditional_log_physical_moment_objective_is_development_sufficient",
            "seal_V63_and_await_explicit_approval_before_independent_EAGLE_gate",
        )
    if q3 and q4:
        return (
            "conditional_moment_marginal_is_calibrated_but_empirical_rank_copula_limits_morphology",
            "audit_only_conditional_rank_copula_dependence_without_changing_the_V63_likelihood_or_objective",
        )
    if improves_both:
        return (
            "conditional_moment_objective_improves_all_extremes_but_is_not_development_sufficient",
            "seal_the_train_and_development_effect_before_selecting_any_further_model",
        )
    if reference_dominates:
        return (
            "conditional_moment_train_repair_does_not_transfer_to_development",
            "stop_before_independent_EAGLE_and_reassess_train_to_development_condition_shift",
        )
    if rolled_improves:
        return (
            "V63_query_local_parameters_are_not_causal_for_development_extremes",
            "stop_local_neural_likelihood_changes_and_reassess_the_observable_information_ceiling",
        )
    return (
        "V63_development_result_is_mixed_or_not_a_common_domain_repair",
        "seal_train_and_development_evidence_before_selecting_any_further_model",
    )


def _validate(
    path: Path,
    arm: str,
    domain: str,
    parent: Path,
    train_gate: dict[str, Any],
    train_gate_path: Path,
    train_gate_sha: str,
    repo: Path,
    gate_commit: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": METHOD,
            "arm": arm,
            "v63_program_sha256": PROGRAM_SHA256,
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
            "moment_coefficient": MOMENT_COEFFICIENT,
            "moment_quadrature_order": QUADRATURE_ORDER,
            "conditional_moment_objective": True,
            "structure_risk_unchanged_from_V50": True,
            "train_mechanism_pass": True,
            "development_sampling_authorized_by_train_gate": True,
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
                raise ValueError(f"V63 {domain} {arm} metadata differs: {key}")
        for name in (
            "source_index", "donor_source", "donor_index", "donor_isometry",
            "donor_distance", "predicted_residual_dc", "predicted_band_scales",
        ):
            if not np.array_equal(current[name][:], old[name][:]):
                raise ValueError(f"V63 {domain} frozen selection differs: {name}")
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V63 ensemble shape differs")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(
            np.max(np.abs(residual.mean(axis=(-3, -2, -1), dtype=np.float64)))
        )
        inverse_error = float(np.max(current["maximum_inverse_CDF_error"][:]))
        names = {
            "checkpoint": "checkpoint_sha256",
            "training_report": "training_report_sha256",
            "preflight": "preflight_sha256",
            "threshold_selection": "threshold_selection_sha256",
            "grid": "grid_sha256",
            "train_mechanism_gate": "train_mechanism_gate_sha256",
        }
        bindings = {
            name: (
                Path(str(current.attrs[name])),
                str(current.attrs[digest_name]),
            )
            for name, digest_name in names.items()
        }
        sampling_commit = str(current.attrs["sampling_code_commit"])
    if (
        maximum_dc > 1.0e-7
        or inverse_error > 2.0e-6
        or bindings["checkpoint"][1] != train_gate["checkpoint_sha256"]
        or bindings["training_report"][1] != train_gate["training_report_sha256"]
        or bindings["preflight"][1] != train_gate["preflight_sha256"]
        or bindings["grid"][1] != train_gate["grid_sha256"]
        or bindings["train_mechanism_gate"] != (train_gate_path, train_gate_sha)
        or any(sha256_file(file_path) != digest for file_path, digest in bindings.values())
        or not _is_ancestor(repo, str(train_gate["code_commit"]), sampling_commit)
        or not _is_ancestor(repo, sampling_commit, gate_commit)
    ):
        raise ValueError("V63 sample integrity or code ancestry differs")
    return {
        **{f"{name}_sha256": digest for name, (_, digest) in bindings.items()},
        "training_code_commit": str(train_gate["code_commit"]),
        "sampling_code_commit": sampling_commit,
        "gate_code_commit": gate_commit,
        "maximum_absolute_sample_residual_DC": maximum_dc,
        "maximum_inverse_CDF_error": inverse_error,
    }


def evaluate(
    root: Path,
    program_path: Path,
    repo: Path,
    train_gate_path: Path,
    train_gate_sha: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, _, _, _, _, _ = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V63 development gate requires clean worktree")
    if sha256_file(train_gate_path) != train_gate_sha:
        raise ValueError("V63 train gate hash differs")
    train_gate = json.loads(
        train_gate_path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if (
        train_gate.get("schema") != TRAIN_GATE_SCHEMA
        or train_gate.get("train_mechanism_pass") is not True
        or canonical_digest(train_gate) != train_gate.get("decision_digest_sha256")
        or train_gate.get("development_accessed") is not False
        or train_gate.get("historical_EAGLE_accessed") is not False
        or train_gate.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(train_gate.get("code_commit")), commit)
    ):
        raise ValueError("V63 train mechanism gate binding differs")
    v56 = json.loads(_path(repo, program["frozen_inputs"]["v56_program"]).read_text())
    frozen = v56["frozen_inputs"]
    references = {}
    for label, key in (("V50", "v50_development_decision"), ("V52", "v52_development_decision")):
        path = _path(repo, frozen[key])
        if sha256_file(path) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V63 {label} reference hash differs")
        references[label] = json.loads(path.read_text())
    arms: dict[str, Any] = {}
    for arm in ARMS:
        domains: dict[str, Any] = {}
        for domain in DOMAIN_ORDER:
            domain_root = root / arm / "development_candidate" / DOMAIN_KEYS[domain]
            ensemble = domain_root / "ensemble16.h5"
            parent = Path(v35["development_domains"][domain]["phase_object_selection"])
            provenance = _validate(
                ensemble, arm, domain, parent, train_gate, train_gate_path,
                train_gate_sha, repo, commit,
            )
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V63 metrics point elsewhere")
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
            "all_three_field_pass": all(
                row["field_gate"]["pass"] for row in domains.values()
            ),
        }
    candidate, rolled = arms[CANDIDATE], arms["rolled_parameter_control"]
    comparisons: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        current = candidate["domains"][domain]["mechanism_Q3_Q4"]
        refs = {
            "V50": references["V50"]["arms"][CANDIDATE]["domains"][domain]["mechanism_Q3_Q4"],
            "V52": references["V52"]["arms"]["no_risk_query_local_mixture_copula"]["domains"][domain]["mechanism_Q3_Q4"],
        }
        comparisons[domain] = {
            name: _extreme_comparison(current, value) for name, value in refs.items()
        }
    improves_both = all(
        all(row["candidate_strictly_improves_all_three"] for row in comparisons[domain].values())
        for domain in DOMAIN_ORDER
    )
    reference_dominates = all(
        any(row["reference_equals_or_improves_all_three"] for row in comparisons[domain].values())
        for domain in DOMAIN_ORDER
    )
    rolled_improves = _rolled_improves(candidate, rolled)
    primary = bool(
        candidate["Q3_all_domains"]
        and candidate["Q4_all_domains"]
        and candidate["all_three_field_pass"]
    )
    classification, next_step = classify(
        primary,
        bool(candidate["Q3_all_domains"]),
        bool(candidate["Q4_all_domains"]),
        improves_both,
        reference_dominates,
        rolled_improves,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "v63_conditional_log_physical_moment_bounded_mixture",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "gate_code_commit": commit,
        "worktree_clean_at_gate": clean,
        "train_mechanism_gate": str(train_gate_path.resolve()),
        "train_mechanism_gate_sha256": train_gate_sha,
        "train_mechanism_pass": True,
        "arms": arms,
        "comparison_to_sealed_V50_and_V52": comparisons,
        "V63_strictly_improves_all_three_over_both_references_every_domain": improves_both,
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
        "independent_EAGLE_access_authorized": False,
        "explicit_user_approval_required_before_EAGLE": True,
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
        raise FileExistsError("V63 refuses existing development decision")
    result = evaluate(
        args.root.resolve(), args.program.resolve(), args.repo.resolve(),
        args.train_gate.resolve(), args.train_gate_sha256,
    )
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
