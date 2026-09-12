#!/usr/bin/env python
"""Apply the frozen fair-tail and unchanged field gate to one V72 stage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v6_gate import field_gate
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v63_train import _is_ancestor
from hong2021_v72_sqt import (
    ARMS,
    CANDIDATE,
    CONTROL,
    DECISION_SCHEMA,
    DOMAIN_KEYS,
    DOMAIN_ORDER,
    ENSEMBLE_SCHEMA,
    METHOD,
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    RAW,
    authorize_parent_evidence,
    load_program,
    scalar_energy_score,
    stage_selection,
    strict_json,
    validate_preflight,
    validate_frozen_gate_sources,
    validate_stage_A_pass,
)


DENSITY_SCALE = 4.5
POWER_BANDS = ("3-6_h_mpc", "6-10.0531_h_mpc")
EVALUATION_SCHEMA = "hong2021-stochastic-residual-ensemble-evaluation-v1"


def _load_sqt_metrics(path: Path) -> dict[str, Any]:
    """Load the sole V72 evaluator row without the legacy V15 EDM assumption."""
    payload = strict_json(path)
    candidates = payload.get("candidates")
    if (
        payload.get("schema") != EVALUATION_SCHEMA
        or not isinstance(candidates, dict)
        or list(candidates) != ["sqt"]
        or not isinstance(candidates["sqt"], dict)
        or candidates["sqt"].get("label") != "sqt"
    ):
        raise ValueError("V72 evaluator schema or sole sqt candidate differs")
    return candidates["sqt"]


def _value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _method(arm: str) -> str:
    return {
        CANDIDATE: METHOD,
        RAW: "raw_fixed_step30000_V70_latent_spatial_score_control",
        CONTROL: "independent_voxel_V63_marginal_control",
    }[arm]


def _validate_ensemble(
    path: Path,
    arm: str,
    domain: str,
    stage: str,
    expected_indices: list[int],
    preflight_path: Path,
    preflight_sha: str,
    program: dict[str, Any],
    repo: Path,
    gate_commit: str,
    stage_A_decision_path: Path | None,
    stage_A_decision_sha: str | None,
) -> dict[str, Any]:
    with h5py.File(path, "r") as current:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": _method(arm),
            "arm": arm,
            "stage": stage,
            "v72_program_sha256": PROGRAM_SHA256,
            "preflight": str(preflight_path.resolve()),
            "preflight_sha256": preflight_sha,
            "stage_A_decision": (
                str(stage_A_decision_path.resolve())
                if stage_A_decision_path is not None else ""
            ),
            "stage_A_decision_sha256": stage_A_decision_sha or "",
            "ensemble_members": 16,
            "conditioning_strata": 16,
            "voxels_per_stratum": 16384,
            "noise_seed": int(
                program["fixed_sampling"][f"stage_{stage}_noise_seed"]
            ),
            "sampler_steps": 40,
            "sigma_minimum": 0.002,
            "sigma_maximum": 40.0,
            "rho": 7.0,
            "stochastic_churn": 0.0,
            "diagnostic_k_h_mpc": 1.0,
            "candidate_arm": arm == CANDIDATE,
            "diagnostic_control_may_affect_selection": False,
            "unequal_sample_global_maximum_used": False,
            "sample_clipping": False,
            "posthoc_Ak_used": False,
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
                raise ValueError(f"V72 {stage} {domain} {arm} metadata differs: {key}")
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V72 ensemble sample shape differs")
        if list(map(int, current["source_index"][:])) != expected_indices:
            raise ValueError("V72 fresh source indices differ")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(
            np.max(np.abs(residual.mean(axis=(-3, -2, -1), dtype=np.float64)))
        )
        inverse_error = float(np.max(current["maximum_inverse_CDF_error"][:]))
        exact_multiset = np.asarray(
            current["pre_inverse_stratum_multiset_equal"][:], dtype=np.uint8
        )
        multiset_error = float(
            np.max(current["maximum_pre_inverse_stratum_multiset_error"][:])
        )
        tie_fraction = float(np.max(current["marginal_tied_voxel_fraction"][:]))
        rank_disagreement = float(
            np.max(
                current[
                    "rank_disagreement_fraction_excluding_marginal_ties"
                ][:]
            )
        )
        pre_physical = float(
            np.max(current["maximum_physical_pre_DC_sorted_residual_difference"][:])
        )
        post_physical = float(
            np.max(current["maximum_physical_post_DC_sorted_residual_difference"][:])
        )
        numerical = program["numerical_requirements"]
        if (
            maximum_dc > float(numerical["maximum_absolute_residual_DC"])
            or inverse_error > float(numerical["maximum_inverse_CDF_error"])
            or not np.all(exact_multiset == 1)
            or multiset_error != 0.0
            or tie_fraction > float(numerical["maximum_marginal_tied_voxel_fraction"])
            or rank_disagreement != 0.0
            or not np.isfinite(pre_physical)
            or not np.isfinite(post_physical)
        ):
            raise ValueError("V72 SQT or numerical invariant differs")
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
        or bindings["source_data"][1] != frozen[f"{domain}_validation_data_sha256"]
        or bindings["source_cache"][1] != frozen[f"{domain}_validation_cache_sha256"]
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, sampling_commit)
        or not _is_ancestor(repo, sampling_commit, gate_commit)
    ):
        raise ValueError("V72 artifact hash or ancestry differs")
    return {
        **{f"{name}_sha256": digest for name, (_, digest) in bindings.items()},
        "sampling_code_commit": sampling_commit,
        "gate_code_commit": gate_commit,
        "maximum_absolute_sample_residual_DC": maximum_dc,
        "maximum_inverse_CDF_error": inverse_error,
        "maximum_pre_inverse_stratum_multiset_error": multiset_error,
        "maximum_marginal_tied_voxel_fraction": tie_fraction,
        "maximum_rank_disagreement_fraction_excluding_marginal_ties": rank_disagreement,
        "maximum_physical_pre_DC_sorted_residual_difference": pre_physical,
        "maximum_physical_post_DC_sorted_residual_difference": post_physical,
        "initial_latent_sha256": innovation_digest,
    }


def cube_maximum_energy_score(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        sample = np.asarray(handle["sample"], dtype=np.float32)
        truth = np.asarray(handle["truth"], dtype=np.float32)
    generated_max = DENSITY_SCALE * sample.max(axis=(-3, -2, -1))[:, :, 0]
    truth_max = DENSITY_SCALE * truth.max(axis=(-3, -2, -1))[:, 0]
    per_query = [
        scalar_energy_score(generated_max[index], float(truth_max[index]))
        for index in range(16)
    ]
    return {
        "mean_unbiased_energy_score": float(np.mean(per_query)),
        "per_query_unbiased_energy_score": per_query,
        "truth_cubes": 16,
        "generated_members_per_truth_query": 16,
        "unequal_sample_global_maximum_used": False,
    }


def evaluate_stage(
    root: Path,
    program_path: Path,
    repo: Path,
    preflight_path: Path,
    preflight_sha: str,
    stage: str,
    stage_A_decision_path: Path | None = None,
    stage_A_decision_sha: str | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit):
        raise RuntimeError("V72 gate requires clean frozen ancestry")
    authorize_parent_evidence(program, repo, commit)
    validate_preflight(preflight_path, preflight_sha, repo, commit)
    if stage == "B":
        if stage_A_decision_path is None or stage_A_decision_sha is None:
            raise ValueError("V72 stage-B gate requires stage-A pass")
        validate_stage_A_pass(
            program, stage_A_decision_path, stage_A_decision_sha,
            preflight_sha, repo, commit,
        )
    elif stage != "A" or stage_A_decision_path is not None or stage_A_decision_sha is not None:
        raise ValueError("V72 stage gate authorization differs")
    if root.resolve() != Path(program["output_roots"][f"stage_{stage}"]).resolve():
        raise ValueError("V72 stage root differs")
    validate_frozen_gate_sources(program, repo)
    selection = stage_selection(program, stage)
    arms: dict[str, Any] = {}
    private: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        domains: dict[str, Any] = {}
        private[arm] = {}
        for domain in DOMAIN_ORDER:
            domain_root = root / arm / "fresh_candidate" / DOMAIN_KEYS[domain]
            ensemble = domain_root / "ensemble16.h5"
            provenance = _validate_ensemble(
                ensemble, arm, domain, stage, selection[domain],
                preflight_path, preflight_sha, program, repo, commit,
                stage_A_decision_path, stage_A_decision_sha,
            )
            private[arm][domain] = provenance
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_sqt_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V72 metrics point elsewhere")
            marginal = marginal_diagnostics(ensemble)
            power = metrics["fourier_log_density"]["generated_total_power_over_truth"]
            residual_rms = metrics["residual_calibration"]["generated_over_truth_rms"]
            domains[domain] = {
                "ensemble": str(ensemble.resolve()),
                "ensemble_sha256": sha256_file(ensemble),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "tail": marginal,
                "cube_maximum_energy_score": cube_maximum_energy_score(ensemble),
                "explicit_spectral_pass": all(
                    0.9 <= float(power[band]) <= 1.1 for band in POWER_BANDS
                ),
                "explicit_residual_RMS_pass": 0.9 <= float(residual_rms) <= 1.1,
                "provenance": {
                    key: value for key, value in provenance.items()
                    if key != "initial_latent_sha256"
                },
            }
        arms[arm] = {"domains": domains}
    for domain in DOMAIN_ORDER:
        reference = private[CANDIDATE][domain]["initial_latent_sha256"]
        if any(
            not np.array_equal(reference, private[arm][domain]["initial_latent_sha256"])
            for arm in (RAW, CONTROL)
        ):
            raise ValueError("V72 arms are not innovation-paired")
    candidate_domains = arms[CANDIDATE]["domains"]
    control_domains = arms[CONTROL]["domains"]
    q99_pass = all(
        abs(float(row["tail"]["delta_q99_999_dex"])) <= 0.1
        for row in candidate_domains.values()
    )
    q4_pass = all(
        (2.0 / 3.0)
        <= float(row["tail"]["generated_over_truth_mean_delta_squared"])
        <= 1.5
        for row in candidate_domains.values()
    )
    maximum_energy_pass = all(
        candidate_domains[domain]["cube_maximum_energy_score"][
            "mean_unbiased_energy_score"
        ]
        < control_domains[domain]["cube_maximum_energy_score"][
            "mean_unbiased_energy_score"
        ]
        for domain in DOMAIN_ORDER
    )
    field_pass = all(row["field_gate"]["pass"] for row in candidate_domains.values())
    spectral_pass = all(
        row["explicit_spectral_pass"] for row in candidate_domains.values()
    )
    rms_pass = all(
        row["explicit_residual_RMS_pass"] for row in candidate_domains.values()
    )
    primary = bool(
        q99_pass and q4_pass and maximum_energy_pass
        and field_pass and spectral_pass and rms_pass
    )
    if stage == "A":
        classification = (
            "V72_SQT_passes_fresh_stage_A"
            if primary else "V72_SQT_is_not_fresh_screen_sufficient"
        )
        next_step = (
            "run_one_locked_stage_B_without_candidate_changes"
            if primary
            else "seal_and_stop_without_stage_B_or_EAGLE_access_or_posthoc_candidate_changes"
        )
    else:
        classification = (
            "V72_SQT_is_fresh_two_stage_sufficient"
            if primary else "V72_SQT_is_not_fresh_confirmation_sufficient"
        )
        next_step = (
            "seal_and_await_new_explicit_user_approval_before_independent_EAGLE"
            if primary
            else "seal_and_stop_before_EAGLE_without_repeating_either_stage"
        )
    result: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "status": "complete_single_V72_SQT_stage_gate",
        "stage": stage,
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "stage_A_decision": (
            str(stage_A_decision_path.resolve())
            if stage_A_decision_path is not None else None
        ),
        "stage_A_decision_sha256": stage_A_decision_sha,
        "gate_code_commit": commit,
        "worktree_clean_at_gate": clean,
        "fresh_selection": selection,
        "arms": arms,
        "candidate_arm": CANDIDATE,
        "diagnostic_arms": [RAW, CONTROL],
        "diagnostic_arms_used_for_selection": False,
        "fair_q99_999_pass": q99_pass,
        "symmetric_Q4_pass": q4_pass,
        "cube_maximum_energy_score_better_than_control_all_domains": maximum_energy_pass,
        "all_three_field_pass": field_pass,
        "explicit_spectral_pass": spectral_pass,
        "explicit_residual_RMS_pass": rms_pass,
        "unequal_sample_global_maximum_used": False,
        "stage_pass": primary,
        "candidate_selected": primary,
        "classification": classification,
        "next": next_step,
        "single_use_stage": True,
        "stage_A_accessed": True,
        "stage_B_accessed": stage == "B",
        "candidate_strata_seed_member_sampler_metric_or_threshold_tuned": False,
        "training_gradient_or_optimizer_performed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
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
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument("--stage-A-decision", type=Path)
    parser.add_argument("--stage-A-decision-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V72 refuses an existing stage decision")
    result = evaluate_stage(
        args.root, args.program, args.repo, args.preflight,
        args.preflight_sha256, args.stage, args.stage_A_decision,
        args.stage_A_decision_sha256,
    )
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
