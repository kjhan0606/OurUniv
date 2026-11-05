#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V42."""
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
from hong2021_v42_tail_body import (
    ARMS,
    BLOCK,
    BLOCKS,
    ENSEMBLE_SCHEMA,
    LAMBDA_KNOTS,
    PREFLIGHT_SCHEMA,
    PROGRAM_SHA256,
    TAIL_QUANTILE,
    _verified_json,
    load_program,
)


SCHEMA = "hong2021-v42-within-block-tail-body-three-domain-decision-v1"
HIGH_K_RMS_CHECKS = (
    "high_k_total_power_within_10_percent",
    "residual_rms_within_10_percent",
)


def _value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate(
    path: Path,
    arm: str,
    domain: str,
    parent: Path,
    gate_commit: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "train_only_within_block_tail_body",
            "arm": arm,
            "v42_program_sha256": PROGRAM_SHA256,
            "parent_selection_sha256": sha256_file(parent),
            "block_factor": 4,
            "block_grid": 16,
            "tail_quantile": TAIL_QUANTILE,
            "tail_lambda_minimum": LAMBDA_KNOTS[0],
            "global_residual_scale": 1.0,
            "diagnostic_k_h_mpc": 1.0,
            "ensemble_members": 16,
            "conditional_rank_multiset_preserved_before_inverse": True,
            "validation_truth_used_for_risk_or_amplitude": False,
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
                raise ValueError(f"V42 {domain} {arm} metadata differs: {key}")

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
            raise ValueError("V42 ensemble shape or frozen selection differs")

        seed_count = int(current.attrs["seed_block_count"])
        native_count = int(current.attrs["native_seed_voxels_per_seed_block"])
        block_map = np.asarray(current["block_permutation"], dtype=np.int64)
        block_identity = np.broadcast_to(np.arange(BLOCKS), block_map.shape)
        nonidentity = np.count_nonzero(block_map != block_identity, axis=-1)
        if (
            seed_count != 11
            or not 1 <= native_count <= BLOCK**3
            or block_map.shape != (16, 16, BLOCKS)
            or not np.array_equal(np.sort(block_map, axis=-1), block_identity)
            or np.any(nonidentity > 2 * seed_count)
        ):
            raise ValueError("V42 block permutation invariant differs")

        local_map = np.asarray(current["local_permutation"], dtype=np.int64)
        local_identity = np.broadcast_to(
            np.arange(BLOCK**3), local_map.shape
        )
        if (
            local_map.shape != (16, 16, seed_count, BLOCK**3)
            or not np.array_equal(np.sort(local_map, axis=-1), local_identity)
            or (
                arm == "block_only_tail_control"
                and not np.array_equal(local_map, local_identity)
            )
        ):
            raise ValueError("V42 local permutation invariant differs")

        tail_lambda = np.asarray(current["tail_lambda"], dtype=np.float64)
        predicted = np.asarray(
            current["predicted_log10_mean_delta_squared"], dtype=np.float64
        )
        achieved = np.asarray(
            current["achieved_log10_mean_delta_squared"], dtype=np.float64
        )
        error = np.asarray(current["absolute_amplitude_error"], dtype=np.float64)
        body_error = np.asarray(
            current["maximum_non_tail_error_after_undoing_DC"], dtype=np.float64
        )
        if (
            not np.isfinite(tail_lambda).all()
            or np.any(tail_lambda < LAMBDA_KNOTS[0])
            or np.any(tail_lambda > LAMBDA_KNOTS[-1])
            or (
                arm == "tail_calibration_disabled_control"
                and not np.allclose(tail_lambda, 1.0)
            )
            or not np.allclose(error, np.abs(achieved - predicted), atol=2e-6, rtol=1e-6)
            or float(body_error.max()) > 1.0e-7
        ):
            raise ValueError("V42 tail calibration invariant differs")

        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        if maximum_dc > 1.0e-7:
            raise ValueError("V42 residual DC differs")

        artifact = Path(str(current.attrs["fit_artifact"]))
        report = Path(str(current.attrs["fit_report"]))
        preflight = Path(str(current.attrs["preflight"]))
        artifact_sha = str(current.attrs["fit_artifact_sha256"])
        report_sha = str(current.attrs["fit_report_sha256"])
        preflight_sha = str(current.attrs["preflight_sha256"])
        if (
            sha256_file(artifact) != artifact_sha
            or sha256_file(report) != report_sha
            or sha256_file(preflight) != preflight_sha
        ):
            raise ValueError("V42 fit or preflight hash differs")
        checked = json.loads(preflight.read_text())
        sampling_commit = str(current.attrs["sampling_code_commit"])
        if (
            checked.get("schema") != PREFLIGHT_SCHEMA
            or checked.get("status") != "pass"
            or checked.get("code_commit") != sampling_commit
        ):
            raise ValueError("V42 preflight binding differs")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", sampling_commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V42 sampling commit is not an ancestor")
        rank_digest = np.asarray(
            current["conditional_rank_multiset_sha256"], dtype=np.uint8
        )

    return {
        "sampling_code_commit": sampling_commit,
        "fit_artifact_sha256": artifact_sha,
        "fit_report_sha256": report_sha,
        "preflight_sha256": preflight_sha,
        "seed_block_count": seed_count,
        "native_seed_voxels_per_seed_block": native_count,
        "maximum_absolute_sample_residual_dc": maximum_dc,
        "maximum_non_tail_error_after_undoing_DC": float(body_error.max()),
        "mean_nonidentity_blocks": float(nonidentity.mean()),
        "mean_nonidentity_native_voxels": float(
            np.count_nonzero(local_map != local_identity, axis=-1).sum(axis=-1).mean()
        ),
        "mean_tail_lambda": float(tail_lambda.mean()),
        "tail_lambda_boundary_fraction": float(
            np.mean(
                np.isclose(tail_lambda, LAMBDA_KNOTS[0])
                | np.isclose(tail_lambda, LAMBDA_KNOTS[-1])
            )
        ),
        "mean_absolute_amplitude_calibration_error": float(error.mean()),
        "maximum_absolute_amplitude_calibration_error": float(error.max()),
        "rank_digest": rank_digest,
        "block_map": block_map,
        "local_map": local_map,
        "predicted_amplitude": predicted,
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


def classify(
    primary: bool,
    q3: bool,
    q4: bool,
    high_k_rms: bool,
) -> tuple[str, str]:
    if primary:
        return (
            "within_block_tail_body_model_sufficient",
            "seal_v42_and_await_explicit_approval_before_independent_gate",
        )
    if q3 and q4 and high_k_rms:
        return (
            "tail_and_body_repaired_remaining_failure_is_field_morphology_or_calibration",
            "audit_only_the_remaining_named_field_checks_before_any_generator_change",
        )
    if q3 and q4:
        return (
            "tail_only_calibration_still_damages_stochastic_body",
            "audit_conditional_tail_threshold_and_DC_coupling_without_refitting_structure_models",
        )
    if q4:
        return (
            "native_extreme_location_is_still_insufficient",
            "stop_spatial_rearrangement_and_fit_a_train_only_conditional_extreme_value_tail_likelihood",
        )
    return (
        "body_preserving_tail_intervention_is_not_a_common_domain_repair",
        "audit_tail_threshold_support_and_object_amplitude_target_compatibility",
    )


def evaluate(root: Path, program_path: Path, repo: Path, commit: str) -> dict[str, Any]:
    program, v35, _, _ = load_program(program_path, repo)
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
            public_provenance = {
                key: value
                for key, value in provenance.items()
                if key
                not in ("rank_digest", "block_map", "local_map", "predicted_amplitude")
            }
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V42 metrics point elsewhere")
            domains[domain] = {
                "ensemble": str(ensemble.resolve()),
                "ensemble_sha256": sha256_file(ensemble),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble),
                "provenance": public_provenance,
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
        reference = internal["within_block_tail_body"][domain]
        for arm in ARMS[1:]:
            row = internal[arm][domain]
            if not np.array_equal(row["rank_digest"], reference["rank_digest"]):
                raise ValueError("V42 arms do not reuse conditional-rank donors")
            if not np.array_equal(row["block_map"], reference["block_map"]):
                raise ValueError("V42 control changed coarse block transport")
            if not np.array_equal(
                row["predicted_amplitude"], reference["predicted_amplitude"]
            ):
                raise ValueError("V42 control changed amplitude prediction")
        if not np.array_equal(
            internal["tail_calibration_disabled_control"][domain]["local_map"],
            reference["local_map"],
        ):
            raise ValueError("V42 tail-disabled control changed native transport")

    parent = program["parent_evidence"]
    v41_record = _verified_json(
        (repo / parent["v41_record"]).resolve(),
        parent["v41_record_sha256"],
        "V42 V41 record",
    )
    v41_decision = _verified_json(
        Path(v41_record["decision"]["path"]),
        v41_record["decision"]["sha256"],
        "V42 V41 decision",
    )
    inherited = program["inherited_inputs"]
    v31_record = _verified_json(
        (repo / inherited["v31_record"]).resolve(),
        inherited["v31_record_sha256"],
        "V42 V31 record",
    )
    comparison: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        old31 = v31_record["paired_v29_to_v31"][domain]
        old41 = v41_decision["arms"]["two_stage"]["domains"][domain]
        comparison[domain] = {
            "order": ["V31", "V41", *ARMS],
            "Q3_delta_q99_999_dex": [
                float(old31["Q3_delta_q99_999_dex"][1]),
                float(old41["mechanism_Q3_Q4"]["delta_q99_999_dex"]),
                *[
                    arms[arm]["domains"][domain]["mechanism_Q3_Q4"][
                        "delta_q99_999_dex"
                    ]
                    for arm in ARMS
                ],
            ],
            "Q3_maximum_excess_dex": [
                float(old31["Q3_maximum_excess_dex"][1]),
                float(
                    old41["mechanism_Q3_Q4"][
                        "generated_max_above_truth_max_dex"
                    ]
                ),
                *[
                    arms[arm]["domains"][domain]["mechanism_Q3_Q4"][
                        "generated_max_above_truth_max_dex"
                    ]
                    for arm in ARMS
                ],
            ],
            "Q4_generated_over_truth": [
                float(old31["Q4"][1]),
                float(
                    old41["mechanism_Q3_Q4"][
                        "generated_over_truth_mean_delta_squared"
                    ]
                ),
                *[
                    arms[arm]["domains"][domain]["mechanism_Q3_Q4"][
                        "generated_over_truth_mean_delta_squared"
                    ]
                    for arm in ARMS
                ],
            ],
        }

    selected = arms["within_block_tail_body"]
    primary = bool(
        selected["Q3_all_domains"]
        and selected["Q4_all_domains"]
        and selected["all_three_field_pass"]
    )
    classification, next_step = classify(
        primary,
        bool(selected["Q3_all_domains"]),
        bool(selected["Q4_all_domains"]),
        bool(selected["high_k_power_and_residual_RMS_all_domains"]),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "v42_within_block_tail_body",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "gate_code_commit": commit,
        "worktree_clean_at_gate": True,
        "arms": arms,
        "comparison_to_v31_v41_and_controls": comparison,
        "development_pass": primary,
        "classification": classification,
        "next": next_step,
        "validation_truth_used_for_risk_or_amplitude": False,
        "global_residual_scale": 1.0,
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
        raise RuntimeError("V42 gate requires a clean committed worktree")
    result = evaluate(
        args.root.resolve(), args.program.resolve(), args.repo.resolve(), commit
    )
    if args.out.exists():
        raise FileExistsError("V42 refuses existing decision")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
