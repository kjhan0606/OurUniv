#!/usr/bin/env python
"""Integrity-bound unchanged field/Q3/Q4 gate for the single V71 development."""
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
from hong2021_v48_development_gate import _passes
from hong2021_v63_train import _is_ancestor
from hong2021_v71_ecc import (
    ARMS,
    CANDIDATE,
    CONTROL,
    ENSEMBLE_SCHEMA,
    METHOD,
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    authorize_parent_evidence,
    load_development_definition,
    load_program,
    validate_frozen_gate_sources,
    validate_preflight,
)


SCHEMA = "hong2021-v71-single-use-tail-preserving-ecc-development-decision-v1"


def _value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate_ensemble(
    path: Path,
    arm: str,
    domain: str,
    parent: Path,
    preflight_path: Path,
    preflight_sha: str,
    preflight: dict[str, Any],
    program: dict[str, Any],
    repo: Path,
    gate_commit: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": METHOD if arm == CANDIDATE else "independent_voxel_V63_marginal_control",
            "arm": arm,
            "v71_development_program_sha256": PROGRAM_SHA256,
            "v71_preflight": str(preflight_path.resolve()),
            "v71_preflight_sha256": preflight_sha,
            "v71_path_B_approved": True,
            "fresh_train_only_V71_screen_available": False,
            "v70_train_gate_candidate_selected": False,
            "v70_train_gate_sha256": preflight["parent_evidence"]["v70_train_gate_sha256"],
            "v70_terminal_seal_sha256": preflight["parent_evidence"]["v70_terminal_seal_sha256"],
            "parent_selection_sha256": sha256_file(parent),
            "ensemble_members": 16,
            "noise_seed": 170073,
            "sampler_steps": 40,
            "sigma_minimum": 0.002,
            "sigma_maximum": 40.0,
            "rho": 7.0,
            "stochastic_churn": 0.0,
            "diagnostic_k_h_mpc": 1.0,
            "candidate_arm": arm == CANDIDATE,
            "control_may_affect_pass_decision": False,
            "sample_clipping": False,
            "posthoc_Ak_used": False,
            "development_sampling_authorized_by_V71_path_B_preflight": True,
            "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
            "gradient_computed": False,
            "optimizer_constructed": False,
            "optimizer_step_performed": False,
            "worktree_clean_at_sampling": True,
            "Astrid_accessed": False,
            "historical_EAGLE_accessed": False,
            "independent_EAGLE_accessed": False,
            "independent_gate_locked": True,
            "complete": True,
        }
        for key, expected in exact.items():
            if _value(current.attrs.get(key)) != expected:
                raise ValueError(f"V71 {domain} {arm} metadata differs: {key}")
        reused = (
            "source_index", "donor_source", "donor_index", "donor_isometry",
            "donor_distance", "predicted_residual_dc", "predicted_band_scales",
        )
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V71 development ensemble shape differs")
        for name in reused:
            if not np.array_equal(current[name][:], old[name][:]):
                raise ValueError(f"V71 {domain} frozen selection differs: {name}")
        if tuple(current["initial_latent_sha256"].shape) != (16, 16, 32):
            raise ValueError("V71 development innovation digest shape differs")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(
            np.max(np.abs(residual.mean(axis=(-3, -2, -1), dtype=np.float64)))
        )
        inverse_error = float(np.max(current["maximum_inverse_CDF_error"][:]))
        exact_multiset = np.asarray(
            current["pre_inverse_sorted_latent_multiset_equal"][:], dtype=np.uint8
        )
        latent_error = float(
            np.max(current["maximum_pre_inverse_sorted_latent_multiset_error"][:])
        )
        pre_dc_error = float(
            np.max(current["maximum_pre_DC_sorted_residual_multiset_error"][:])
        )
        post_dc_error = float(
            np.max(current["maximum_post_DC_sorted_residual_multiset_error"][:])
        )
        tie_fraction = float(np.max(current["control_tied_voxel_fraction"][:]))
        rank_disagreement = float(
            np.max(
                current[
                    "candidate_rank_disagreement_fraction_excluding_control_ties"
                ][:]
            )
        )
        if (
            maximum_dc > 1.0e-7
            or inverse_error > 2.0e-6
            or not np.all(exact_multiset == 1)
            or latent_error != 0.0
            or pre_dc_error != 0.0
            or tie_fraction
            > float(
                program["code_only_preflight_and_numerical_requirements"]
                ["required_during_single_sampling"]
                ["maximum_control_tied_voxel_fraction"]
            )
            or rank_disagreement != 0.0
            or not np.isfinite(post_dc_error)
        ):
            raise ValueError("V71 development ECC or numerical invariant differs")
        bindings = {
            "checkpoint": (
                Path(str(current.attrs["checkpoint"])),
                str(current.attrs["checkpoint_sha256"]),
            ),
            "training_report": (
                Path(str(current.attrs["training_report"])),
                str(current.attrs["training_report_sha256"]),
            ),
            "source_data": (
                Path(str(current.attrs["source_data"])),
                str(current.attrs["source_data_sha256"]),
            ),
            "source_cache": (
                Path(str(current.attrs["source_cache"])),
                str(current.attrs["source_cache_sha256"]),
            ),
        }
        sampling_commit = str(current.attrs["sampling_code_commit"])
        innovation_digest = np.asarray(current["initial_latent_sha256"], dtype=np.uint8)
    frozen = program["frozen_inputs"]
    if (
        any(sha256_file(file_path) != digest for file_path, digest in bindings.values())
        or bindings["checkpoint"][1] != frozen["v70_checkpoint_sha256"]
        or bindings["training_report"][1] != frozen["v70_training_report_sha256"]
        or bindings["checkpoint"][0].resolve() != Path(frozen["v70_checkpoint"]).resolve()
        or bindings["training_report"][0].resolve()
        != Path(frozen["v70_training_report"]).resolve()
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, sampling_commit)
        or not _is_ancestor(repo, sampling_commit, gate_commit)
    ):
        raise ValueError("V71 development artifact hash or ancestry differs")
    return {
        **{f"{name}_sha256": digest for name, (_, digest) in bindings.items()},
        "sampling_code_commit": sampling_commit,
        "gate_code_commit": gate_commit,
        "maximum_absolute_sample_residual_DC": maximum_dc,
        "maximum_inverse_CDF_error": inverse_error,
        "maximum_pre_inverse_sorted_latent_multiset_error": latent_error,
        "maximum_pre_DC_sorted_residual_multiset_error": pre_dc_error,
        "maximum_post_DC_sorted_residual_multiset_error": post_dc_error,
        "maximum_control_tied_voxel_fraction": tie_fraction,
        "maximum_candidate_rank_disagreement_fraction_excluding_control_ties": rank_disagreement,
        "initial_latent_sha256": innovation_digest,
    }


def evaluate(
    root: Path,
    program_path: Path,
    repo: Path,
    preflight_path: Path,
    preflight_sha: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit):
        raise RuntimeError("V71 development gate requires clean frozen ancestry")
    authorize_parent_evidence(program, repo, commit)
    preflight = validate_preflight(preflight_path, preflight_sha, repo, commit)
    if root.resolve() != Path(program["output_roots"]["development"]).resolve():
        raise ValueError("V71 gate output root differs")
    validate_frozen_gate_sources(program, repo)
    v35 = load_development_definition(program, repo)
    arms: dict[str, Any] = {}
    private: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        domains: dict[str, Any] = {}
        private[arm] = {}
        for domain in DOMAIN_ORDER:
            domain_root = root / arm / "development_candidate" / DOMAIN_KEYS[domain]
            ensemble = domain_root / "ensemble16.h5"
            parent = Path(v35["development_domains"][domain]["phase_object_selection"])
            provenance = _validate_ensemble(
                ensemble, arm, domain, parent, preflight_path, preflight_sha,
                preflight, program, repo, commit,
            )
            private[arm][domain] = provenance
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V71 development metrics point elsewhere")
            domains[domain] = {
                "ensemble": str(ensemble.resolve()),
                "ensemble_sha256": sha256_file(ensemble),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble),
                "provenance": {
                    key: value for key, value in provenance.items()
                    if key != "initial_latent_sha256"
                },
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
    for domain in DOMAIN_ORDER:
        if not np.array_equal(
            private[CANDIDATE][domain]["initial_latent_sha256"],
            private[CONTROL][domain]["initial_latent_sha256"],
        ):
            raise ValueError("V71 development arms are not innovation-paired")
    candidate = arms[CANDIDATE]
    primary = bool(
        candidate["Q3_all_domains"]
        and candidate["Q4_all_domains"]
        and candidate["all_three_field_pass"]
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_single_locked_V71_ECC_three_domain_development_gate",
        "experiment": "v71_tail_preserving_ensemble_copula_coupling",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "gate_code_commit": commit,
        "worktree_clean_at_gate": clean,
        "fresh_train_only_V71_screen_available": False,
        "V70_gate_used_as_V71_selection_evidence": False,
        "arms": arms,
        "candidate_arm": CANDIDATE,
        "diagnostic_control_arm": CONTROL,
        "diagnostic_control_used_for_selection": False,
        "development_pass": primary,
        "classification": (
            "V71_tail_preserving_ECC_is_development_sufficient"
            if primary
            else "V71_tail_preserving_ECC_is_not_development_sufficient"
        ),
        "next": (
            "seal_V71_and_await_new_explicit_user_approval_before_independent_EAGLE_access"
            if primary
            else "seal_the_failure_and_stop_before_independent_EAGLE_without_repeating_development_or_tuning_from_its_results"
        ),
        "single_locked_development_attempt": True,
        "training_or_gradient_performed_by_development": False,
        "checkpoint_sampler_seed_member_count_metric_threshold_or_gate_tuned": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
        "independent_EAGLE_access_authorized": False,
        "explicit_user_approval_required_before_EAGLE": primary,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V71 refuses an existing development decision")
    result = evaluate(
        args.root.resolve(), args.program.resolve(), args.repo.resolve(),
        args.preflight.resolve(), args.preflight_sha256,
    )
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
