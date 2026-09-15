#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V41."""
from __future__ import annotations

import argparse
import json
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
from hong2021_v41_two_stage import (
    ARMS,
    BLOCKS,
    ENSEMBLE_SCHEMA,
    PREFLIGHT_SCHEMA,
    PROGRAM_SHA256,
    _verified_json,
    load_program,
)


SCHEMA = "hong2021-v41-two-stage-structure-amplitude-three-domain-decision-v1"


def _validate(path: Path, arm: str, domain: str, parent: Path, gate_commit: str) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "train_only_two_stage_structure_amplitude",
            "arm": arm,
            "v41_program_sha256": PROGRAM_SHA256,
            "parent_selection_sha256": sha256_file(parent),
            "block_factor": 4,
            "block_grid": 16,
            "diagnostic_k_h_mpc": 1.0,
            "ensemble_members": 16,
            "conditional_rank_multiset_preserved_before_inverse": True,
            "validation_truth_used_for_risk_or_amplitude": False,
            "donor_translation": False,
            "donor_reselection": False,
            "density_field_clipping": False,
            "posthoc_Ak_used": False,
            "worktree_clean_at_sampling": True,
            "Astrid_accessed": False,
            "historical_EAGLE_accessed": False,
            "complete": True,
        }
        for key, expected in exact.items():
            actual = current.attrs.get(key)
            actual = actual.item() if isinstance(actual, np.generic) else actual
            if actual != expected:
                raise ValueError(f"V41 {domain} {arm} metadata differs: {key}")
        reused = (
            "source_index", "donor_source", "donor_index", "donor_isometry",
            "donor_distance", "predicted_residual_dc", "predicted_band_scales",
        )
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64) or any(
            not np.array_equal(current[name][:], old[name][:]) for name in reused
        ):
            raise ValueError("V41 ensemble shape or frozen selection differs")
        seed_count = int(current.attrs["seed_block_count"])
        block_map = np.asarray(current["block_permutation"], dtype=np.int64)
        expected_map = np.broadcast_to(np.arange(BLOCKS), block_map.shape)
        nonidentity = np.count_nonzero(block_map != expected_map, axis=-1)
        if (
            block_map.shape != (16, 16, BLOCKS)
            or not np.array_equal(np.sort(block_map, axis=-1), expected_map)
            or np.any(nonidentity > 2 * seed_count)
            or not np.array_equal(nonidentity, np.asarray(current["nonidentity_blocks"], dtype=np.int64))
        ):
            raise ValueError("V41 block permutation invariant differs")
        source_index = np.asarray(current["source_index"], dtype=np.int64)
        risk_reference = np.asarray(current["risk_reference_query_index"], dtype=np.int64)
        amplitude_reference = np.asarray(current["amplitude_reference_query_index"], dtype=np.int64)
        expected_amplitude = np.roll(source_index, -1) if arm == "shuffled_amplitude_control" else source_index
        if not np.array_equal(risk_reference, source_index) or not np.array_equal(amplitude_reference, expected_amplitude):
            raise ValueError("V41 risk or amplitude reference differs")
        scale = np.asarray(current["amplitude_scale"], dtype=np.float64)
        predicted = np.asarray(current["predicted_log10_mean_delta_squared"], dtype=np.float64)
        achieved = np.asarray(current["achieved_log10_mean_delta_squared"], dtype=np.float64)
        error = np.asarray(current["absolute_amplitude_error"], dtype=np.float64)
        if (
            np.any(scale < 0) or np.any(scale > 2) or not np.isfinite(scale).all()
            or not np.allclose(error, np.abs(achieved - predicted), atol=2e-6, rtol=1e-6)
        ):
            raise ValueError("V41 amplitude calibration diagnostics differ")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(current["conditional_mean"], dtype=np.float32)[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        if maximum_dc > 1e-7:
            raise ValueError("V41 residual DC differs")
        artifact = Path(str(current.attrs["fit_artifact"])); report = Path(str(current.attrs["fit_report"])); preflight = Path(str(current.attrs["preflight"]))
        artifact_sha = str(current.attrs["fit_artifact_sha256"]); report_sha = str(current.attrs["fit_report_sha256"]); preflight_sha = str(current.attrs["preflight_sha256"])
        if sha256_file(artifact) != artifact_sha or sha256_file(report) != report_sha or sha256_file(preflight) != preflight_sha:
            raise ValueError("V41 fit or preflight hash differs")
        checked = json.loads(preflight.read_text()); commit = str(current.attrs["sampling_code_commit"])
        if checked.get("schema") != PREFLIGHT_SCHEMA or checked.get("status") != "pass" or checked.get("code_commit") != commit:
            raise ValueError("V41 preflight binding differs")
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, gate_commit], capture_output=True).returncode:
            raise ValueError("V41 sampling commit is not an ancestor")
        rank_digest = np.asarray(current["conditional_rank_multiset_sha256"], dtype=np.uint8)
    return {
        "sampling_code_commit": commit,
        "fit_artifact_sha256": artifact_sha,
        "fit_report_sha256": report_sha,
        "preflight_sha256": preflight_sha,
        "seed_block_count": seed_count,
        "maximum_absolute_sample_residual_dc": maximum_dc,
        "mean_nonidentity_blocks": float(nonidentity.mean()),
        "mean_amplitude_scale": float(scale.mean()),
        "amplitude_scale_boundary_fraction": float(np.mean((scale == 0) | (scale == 2))),
        "mean_absolute_amplitude_calibration_error": float(error.mean()),
        "maximum_absolute_amplitude_calibration_error": float(error.max()),
        "rank_digest": rank_digest,
        "block_map": block_map,
        "predicted_amplitude": predicted,
    }


def _passes(domains: dict[str, Any]) -> tuple[bool, bool]:
    q3 = all(
        abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"]) <= 0.1
        and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] <= 0.3
        for row in domains.values()
    )
    q4 = all(row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] <= 1.5 for row in domains.values())
    return q3, q4


def classify(primary: bool, q3: bool, q4: bool) -> tuple[str, str]:
    if primary:
        return "two_stage_structure_amplitude_model_sufficient", "seal_v41_and_await_explicit_approval_before_independent_gate"
    if q3 and q4:
        return "supervised_tails_repaired_but_sparse_block_transport_limits_morphology", "freeze_overlap_consistent_seam_repair_without_changing_risk_or_amplitude_models"
    if q4:
        return "object_amplitude_calibration_supported_but_structure_seeding_is_insufficient", "freeze_train_only_within_block_extreme_location_model_without_changing_object_amplitude"
    if q3:
        return "structure_seeding_supported_but_object_amplitude_calibration_is_insufficient", "audit_predicted_to_achieved_amplitude_calibration_before_any_refit"
    return "two_stage_supervised_intervention_is_not_a_common_domain_repair", "audit_candidate_vs_backbone_roll_and_amplitude_controls_before_any_further_generator"


def evaluate(root: Path, program_path: Path, repo: Path, commit: str) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo)
    arms: dict[str, Any] = {}
    internal: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        domains, internal[arm] = {}, {}
        for domain in DOMAIN_ORDER:
            domain_root = root / arm / "development_candidate" / DOMAIN_KEYS[domain]
            ensemble = domain_root / "ensemble16.h5"
            parent = Path(v35["development_domains"][domain]["phase_object_selection"])
            provenance = _validate(ensemble, arm, domain, parent, commit)
            internal[arm][domain] = provenance
            public_provenance = {key: value for key, value in provenance.items() if key not in ("rank_digest", "block_map", "predicted_amplitude")}
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V41 metrics point elsewhere")
            domains[domain] = {
                "ensemble": str(ensemble.resolve()), "ensemble_sha256": sha256_file(ensemble),
                "metrics": str(metrics_path.resolve()), "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics), "mechanism_Q3_Q4": marginal_diagnostics(ensemble),
                "provenance": public_provenance,
            }
        q3, q4 = _passes(domains)
        arms[arm] = {
            "domains": domains, "Q3_all_domains": q3, "Q4_all_domains": q4,
            "all_three_field_pass": all(row["field_gate"]["pass"] for row in domains.values()),
        }
    for domain in DOMAIN_ORDER:
        reference = internal["two_stage"][domain]
        for arm in ARMS[1:]:
            row = internal[arm][domain]
            if not np.array_equal(row["rank_digest"], reference["rank_digest"]):
                raise ValueError("V41 arms do not reuse conditional-rank donors")
        if not np.array_equal(internal["shuffled_amplitude_control"][domain]["block_map"], reference["block_map"]):
            raise ValueError("V41 shuffled-amplitude arm changed structure permutation")
        for arm in ("backbone_risk_ablation", "rolled_risk_control"):
            if not np.array_equal(internal[arm][domain]["predicted_amplitude"], reference["predicted_amplitude"]):
                raise ValueError("V41 risk control changed amplitude prediction")
    inherited = program["inherited_inputs"]
    v31 = _verified_json((repo / inherited["v31_record"]).resolve(), inherited["v31_record_sha256"], "V41 V31 record")
    comparison = {}
    for domain in DOMAIN_ORDER:
        old = v31["paired_v29_to_v31"][domain]
        comparison[domain] = {
            "Q3_delta_q99_999_dex_V31_and_arms": [
                float(old["Q3_delta_q99_999_dex"][1]),
                *[arms[arm]["domains"][domain]["mechanism_Q3_Q4"]["delta_q99_999_dex"] for arm in ARMS],
            ],
            "Q3_maximum_excess_dex_V31_and_arms": [
                float(old["Q3_maximum_excess_dex"][1]),
                *[arms[arm]["domains"][domain]["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] for arm in ARMS],
            ],
            "Q4_V31_and_arms": [
                float(old["Q4"][1]),
                *[arms[arm]["domains"][domain]["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] for arm in ARMS],
            ],
            "order": ["V31", *ARMS],
        }
    selected = arms["two_stage"]
    primary = selected["Q3_all_domains"] and selected["Q4_all_domains"] and selected["all_three_field_pass"]
    classification, next_step = classify(bool(primary), bool(selected["Q3_all_domains"]), bool(selected["Q4_all_domains"]))
    result: dict[str, Any] = {
        "schema": SCHEMA, "experiment": "v41_two_stage_structure_amplitude",
        "program": str(program_path.resolve()), "program_sha256": PROGRAM_SHA256,
        "gate_code_commit": commit, "worktree_clean_at_gate": True,
        "arms": arms, "comparison_to_v31_and_controls": comparison,
        "development_pass": bool(primary), "classification": classification, "next": next_step,
        "validation_truth_used_for_risk_or_amplitude": False,
        "donor_translation": False, "donor_reselection": False,
        "density_field_clipping": False, "posthoc_Ak_used": False,
        "Astrid_accessed": False, "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V41 gate requires a clean committed worktree")
    result = evaluate(args.root.resolve(), args.program.resolve(), args.repo.resolve(), commit)
    if args.out.exists():
        raise FileExistsError("V41 refuses existing decision")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
