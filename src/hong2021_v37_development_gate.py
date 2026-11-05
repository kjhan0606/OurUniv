#!/usr/bin/env python
"""Integrity-bound development gate for V37 query-aligned transport."""
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
from hong2021_v37_query_alignment import (
    ARMS,
    ENSEMBLE_SCHEMA,
    MAX_SHIFT,
    PREFLIGHT_SCHEMA,
    PROGRAM_SHA256,
    load_program,
)


SCHEMA = "hong2021-v37-query-aligned-copula-three-domain-decision-v1"


def _validate_ensemble(
    path: Path,
    *,
    arm: str,
    domain: str,
    parent: Path,
    gate_commit: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "bounded_query_aligned_conditional_copula",
            "arm": arm,
            "v37_program_sha256": PROGRAM_SHA256,
            "parent_selection_sha256": sha256_file(parent),
            "pool_factor": 4,
            "maximum_shift_coarse_cells": MAX_SHIFT,
            "diagnostic_k_h_mpc": 1.0,
            "ensemble_members": 16,
            "donor_reselection": False,
            "validation_truth_used_for_alignment_fit_or_shift_selection": False,
            "field_clipping": False,
            "posthoc_Ak_used": False,
            "worktree_clean_at_sampling": True,
            "Astrid_accessed": False,
            "historical_EAGLE_accessed": False,
            "complete": True,
        }
        for key, expected in exact.items():
            actual = current.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                raise ValueError(f"V37 {domain} {arm} metadata differs: {key}")
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V37 ensemble shape differs")
        reused = (
            "source_index",
            "donor_source",
            "donor_index",
            "donor_isometry",
            "donor_distance",
            "predicted_residual_dc",
            "predicted_band_scales",
        )
        if any(not np.array_equal(current[name][:], old[name][:]) for name in reused):
            raise ValueError("V37 did not reuse the frozen V31 selection exactly")
        source_index = np.asarray(current["source_index"], dtype=np.int64)
        alignment_reference = np.asarray(
            current["alignment_reference_query_index"], dtype=np.int64
        )
        expected_reference = (
            source_index if arm == "aligned" else np.roll(source_index, -1)
        )
        if not np.array_equal(alignment_reference, expected_reference):
            raise ValueError("V37 alignment reference query differs")
        shifts = np.asarray(current["alignment_shift_coarse"], dtype=np.int64)
        before = np.asarray(
            current["alignment_descriptor_mse_before"], dtype=np.float64
        )
        after = np.asarray(
            current["alignment_descriptor_mse_after"], dtype=np.float64
        )
        if (
            shifts.shape != (16, 16, 3)
            or before.shape != (16, 16)
            or after.shape != before.shape
            or np.any(np.abs(shifts) > MAX_SHIFT)
            or not np.isfinite(before).all()
            or not np.isfinite(after).all()
            or np.any(after > before + 1.0e-5)
        ):
            raise ValueError("V37 alignment diagnostics differ")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        if maximum_dc > 1.0e-7:
            raise ValueError("V37 residual DC differs")
        descriptor_path = Path(str(current.attrs["descriptor"]))
        preflight_path = Path(str(current.attrs["preflight"]))
        descriptor_sha = str(current.attrs["descriptor_sha256"])
        preflight_sha = str(current.attrs["preflight_sha256"])
        if (
            sha256_file(descriptor_path) != descriptor_sha
            or sha256_file(preflight_path) != preflight_sha
        ):
            raise ValueError("V37 descriptor or preflight hash differs")
        preflight = json.loads(preflight_path.read_text())
        sampling_commit = str(current.attrs.get("sampling_code_commit", ""))
        if (
            preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("status") != "pass"
            or preflight.get("code_commit") != sampling_commit
            or preflight.get("descriptor_sha256") != descriptor_sha
        ):
            raise ValueError("V37 preflight binding differs")
        if subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                sampling_commit,
                gate_commit,
            ],
            capture_output=True,
        ).returncode:
            raise ValueError("V37 sampling commit is not an ancestor of gate commit")
    lengths = np.sqrt(np.square(shifts, dtype=np.float64).sum(axis=-1))
    return {
        "sampling_code_commit": sampling_commit,
        "descriptor": str(descriptor_path.resolve()),
        "descriptor_sha256": descriptor_sha,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "maximum_absolute_sample_residual_dc": maximum_dc,
        "nonzero_shift_fraction": float(np.mean(np.any(shifts != 0, axis=-1))),
        "mean_shift_coarse_cells": float(lengths.mean()),
        "maximum_shift_coarse_cells": float(lengths.max()),
        "mean_descriptor_mse_before": float(before.mean()),
        "mean_descriptor_mse_after": float(after.mean()),
        "mean_descriptor_mse_after_over_before": float(after.mean() / before.mean()),
        "donor_selection_exactly_reused": True,
    }


def _q_pass(domains: dict[str, Any]) -> tuple[bool, bool]:
    q3 = all(
        abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"]) <= 0.10
        and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] <= 0.30
        for row in domains.values()
    )
    q4 = all(
        row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] <= 1.5
        for row in domains.values()
    )
    return q3, q4


def classify(
    *, primary_pass: bool, q3: bool, q4: bool, causal_material: bool
) -> tuple[str, str]:
    if primary_pass:
        return (
            "bounded_query_alignment_sufficient",
            "seal_v37_and_await_explicit_approval_before_independent_gate",
        )
    if q3 and q4:
        return (
            "global_query_alignment_repairs_tails_but_not_morphology",
            "freeze_train_only_local_bijective_multiscale_registration_without_changing_the_marginal_coordinate",
        )
    if causal_material:
        return (
            "global_query_alignment_is_causal_but_insufficient",
            "replace_one_global_shift_by_train_only_local_bijective_multiscale_registration",
        )
    return (
        "bounded_global_translation_does_not_explain_joint_coupling",
        "fit_a_train_only_query_conditioned_gaussian_copula_innovation_field_without_donor_translation_search",
    )


def evaluate(
    root: Path, program_path: Path, repo: Path, gate_commit: str
) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo)
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        domains: dict[str, Any] = {}
        for domain in DOMAIN_ORDER:
            domain_root = (
                root / arm / "development_candidate" / DOMAIN_KEYS[domain]
            )
            ensemble = domain_root / "ensemble16.h5"
            parent = Path(v35["development_domains"][domain]["phase_object_selection"])
            alignment = _validate_ensemble(
                ensemble,
                arm=arm,
                domain=domain,
                parent=parent,
                gate_commit=gate_commit,
            )
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve():
                raise ValueError("V37 metrics refer to another ensemble")
            domains[domain] = {
                "ensemble": str(ensemble.resolve()),
                "ensemble_sha256": sha256_file(ensemble),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble),
                "alignment_diagnostics": alignment,
            }
        q3, q4 = _q_pass(domains)
        arms[arm] = {
            "domains": domains,
            "Q3_all_domains": q3,
            "Q4_all_domains": q4,
            "all_three_field_pass": all(
                row["field_gate"]["pass"] for row in domains.values()
            ),
        }
    v31_record = _verified_v31_record(program, repo)
    comparison: dict[str, Any] = {}
    causal_rows = []
    for domain in DOMAIN_ORDER:
        aligned = arms["aligned"]["domains"][domain]["mechanism_Q3_Q4"]
        shuffled = arms["shuffled_query_control"]["domains"][domain][
            "mechanism_Q3_Q4"
        ]
        old = v31_record["paired_v29_to_v31"][domain]
        v31_q4 = float(old["Q4"][1])
        shuffled_q4 = float(shuffled["generated_over_truth_mean_delta_squared"])
        aligned_q4 = float(aligned["generated_over_truth_mean_delta_squared"])
        q4_over_v31 = aligned_q4 / v31_q4
        q4_over_shuffled = aligned_q4 / shuffled_q4
        causal_rows.append(q4_over_v31 <= 0.75 and q4_over_shuffled <= 0.90)
        comparison[domain] = {
            "Q3_delta_q99_999_dex_V31_aligned_shuffled": [
                float(old["Q3_delta_q99_999_dex"][1]),
                aligned["delta_q99_999_dex"],
                shuffled["delta_q99_999_dex"],
            ],
            "Q3_maximum_excess_dex_V31_aligned_shuffled": [
                float(old["Q3_maximum_excess_dex"][1]),
                aligned["generated_max_above_truth_max_dex"],
                shuffled["generated_max_above_truth_max_dex"],
            ],
            "Q4_V31_aligned_shuffled": [v31_q4, aligned_q4, shuffled_q4],
            "aligned_Q4_over_V31": q4_over_v31,
            "aligned_Q4_over_shuffled": q4_over_shuffled,
        }
    causal_material = all(causal_rows)
    aligned = arms["aligned"]
    primary_pass = bool(
        aligned["Q3_all_domains"]
        and aligned["Q4_all_domains"]
        and aligned["all_three_field_pass"]
    )
    classification, next_step = classify(
        primary_pass=primary_pass,
        q3=bool(aligned["Q3_all_domains"]),
        q4=bool(aligned["Q4_all_domains"]),
        causal_material=causal_material,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "v37_bounded_query_aligned_conditional_copula",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "gate_code_commit": gate_commit,
        "worktree_clean_at_gate": True,
        "arms": arms,
        "comparison_to_v31_and_shuffled_control": comparison,
        "causal_alignment_material": causal_material,
        "development_pass": primary_pass,
        "classification": classification,
        "next": next_step,
        "validation_truth_used_for_alignment_fit_or_shift_selection": False,
        "donor_reselection": False,
        "field_clipping": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def _verified_v31_record(program: dict[str, Any], repo: Path) -> dict[str, Any]:
    inherited = program["inherited_inputs"]
    path = (repo / inherited["v31_record"]).resolve()
    if sha256_file(path) != inherited["v31_record_sha256"]:
        raise ValueError("V37 V31 record hash differs at gate")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V37 gate requires a clean worktree")
    report = evaluate(
        args.root.resolve(), args.program.resolve(), args.repo.resolve(), commit
    )
    if args.out.exists():
        raise RuntimeError("V37 refuses to overwrite its decision")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
