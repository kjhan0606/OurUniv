#!/usr/bin/env python
"""Integrity-bound development gate for V39 bijective patch copula."""
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
from hong2021_v39_patch_copula import (
    ARMS, BLOCKS, ENSEMBLE_SCHEMA, PREFLIGHT_SCHEMA, PROGRAM_SHA256, load_program,
)


SCHEMA = "hong2021-v39-bijective-patch-copula-three-domain-decision-v1"


def _validate(path: Path, arm: str, domain: str, parent: Path, gate_commit: str) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "train_only_bijective_local_patch_copula",
            "arm": arm,
            "v39_program_sha256": PROGRAM_SHA256,
            "parent_selection_sha256": sha256_file(parent),
            "block_factor": 8,
            "block_grid": 8,
            "diagnostic_k_h_mpc": 1.0,
            "ensemble_members": 16,
            "conditional_rank_multiset_preserved": True,
            "additive_query_predictor": False,
            "donor_translation": False,
            "donor_reselection": False,
            "validation_truth_used_for_descriptor_fit_or_assignment": False,
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
                raise ValueError(f"V39 {domain} {arm} metadata differs: {key}")
        reused = (
            "source_index", "donor_source", "donor_index", "donor_isometry",
            "donor_distance", "predicted_residual_dc", "predicted_band_scales",
        )
        if tuple(current["sample"].shape) != (16, 16, 1, 64, 64, 64) or any(
            not np.array_equal(current[name][:], old[name][:]) for name in reused
        ):
            raise ValueError("V39 ensemble shape or frozen selection differs")
        block_map = np.asarray(current["block_permutation"], dtype=np.int64)
        expected_map = np.broadcast_to(np.arange(BLOCKS), block_map.shape)
        if block_map.shape != (16, 16, BLOCKS) or not np.array_equal(
            np.sort(block_map, axis=-1), expected_map
        ):
            raise ValueError("V39 block assignment is not bijective")
        source_index = np.asarray(current["source_index"], dtype=np.int64)
        reference = np.asarray(current["alignment_reference_query_index"], dtype=np.int64)
        expected_reference = source_index if arm == "aligned_patch" else np.roll(source_index, -1)
        if not np.array_equal(reference, expected_reference):
            raise ValueError("V39 alignment reference differs")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        if maximum_dc > 1e-7:
            raise ValueError("V39 residual DC differs")
        descriptor = Path(str(current.attrs["descriptor"])); preflight = Path(str(current.attrs["preflight"]))
        descriptor_sha = str(current.attrs["descriptor_sha256"]); preflight_sha = str(current.attrs["preflight_sha256"])
        if sha256_file(descriptor) != descriptor_sha or sha256_file(preflight) != preflight_sha:
            raise ValueError("V39 descriptor/preflight hash differs")
        checked = json.loads(preflight.read_text()); commit = str(current.attrs["sampling_code_commit"])
        if checked.get("schema") != PREFLIGHT_SCHEMA or checked.get("status") != "pass" or checked.get("code_commit") != commit:
            raise ValueError("V39 preflight binding differs")
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, gate_commit], capture_output=True).returncode:
            raise ValueError("V39 sampling commit is not an ancestor")
        nonidentity = np.asarray(current["nonidentity_fraction"], dtype=np.float64)
        descriptor_cost = np.asarray(current["mean_descriptor_cost"], dtype=np.float64)
        spatial_cost = np.asarray(current["mean_spatial_cost"], dtype=np.float64)
    return {
        "sampling_code_commit": commit,
        "descriptor_sha256": descriptor_sha,
        "preflight_sha256": preflight_sha,
        "maximum_absolute_sample_residual_dc": maximum_dc,
        "mean_nonidentity_fraction": float(nonidentity.mean()),
        "mean_descriptor_cost": float(descriptor_cost.mean()),
        "mean_spatial_cost": float(spatial_cost.mean()),
        "donor_selection_exactly_reused": True,
        "all_block_assignments_bijective": True,
    }


def _passes(domains: dict[str, Any]) -> tuple[bool, bool]:
    q3 = all(abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"]) <= .1 and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] <= .3 for row in domains.values())
    q4 = all(row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] <= 1.5 for row in domains.values())
    return q3, q4


def classify(primary: bool, q3: bool, q4: bool, material: bool) -> tuple[str, str]:
    if primary:
        return "bijective_local_patch_copula_sufficient", "seal_v39_and_await_explicit_approval_before_independent_gate"
    if q3 and q4:
        return "patch_copula_repairs_tails_but_block_seams_limit_morphology", "freeze_overlap_add_rank_preserving_seam_repair_without_changing_assignments"
    if material:
        return "bijective_patch_alignment_is_causal_but_insufficient", "freeze_two_level_coarse_to_fine_bijective_patch_copula"
    return "fixed_local_patch_copula_is_not_a_common_domain_repair", "audit_backbone_observable_sufficiency_at_object_and_structure_level_before_any_further_generator"


def evaluate(root: Path, program_path: Path, repo: Path, commit: str) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo); arms = {}
    for arm in ARMS:
        domains = {}
        for domain in DOMAIN_ORDER:
            domain_root = root / arm / "development_candidate" / DOMAIN_KEYS[domain]
            ensemble = domain_root / "ensemble16.h5"; parent = Path(v35["development_domains"][domain]["phase_object_selection"])
            provenance = _validate(ensemble, arm, domain, parent, commit)
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"; metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble.resolve(): raise ValueError("V39 metrics point elsewhere")
            domains[domain] = {"ensemble": str(ensemble.resolve()), "ensemble_sha256": sha256_file(ensemble), "metrics": str(metrics_path.resolve()), "metrics_sha256": sha256_file(metrics_path), "field_gate": field_gate(metrics), "mechanism_Q3_Q4": marginal_diagnostics(ensemble), "provenance": provenance}
        q3, q4 = _passes(domains); arms[arm] = {"domains": domains, "Q3_all_domains": q3, "Q4_all_domains": q4, "all_three_field_pass": all(row["field_gate"]["pass"] for row in domains.values())}
    inherited = program["inherited_inputs"]; v31_path = (repo / inherited["v31_record"]).resolve()
    if sha256_file(v31_path) != inherited["v31_record_sha256"]: raise ValueError("V39 V31 record differs")
    v31 = json.loads(v31_path.read_text()); comparison = {}; material_rows = []
    for domain in DOMAIN_ORDER:
        aligned = arms["aligned_patch"]["domains"][domain]["mechanism_Q3_Q4"]; shuffled = arms["shuffled_query_control"]["domains"][domain]["mechanism_Q3_Q4"]; old = v31["paired_v29_to_v31"][domain]; oq = float(old["Q4"][1]); aq = float(aligned["generated_over_truth_mean_delta_squared"]); sq = float(shuffled["generated_over_truth_mean_delta_squared"]); material_rows.append(aq / oq <= .75 and aq / sq <= .90)
        comparison[domain] = {"Q3_delta_q99_999_dex_V31_aligned_shuffled": [float(old["Q3_delta_q99_999_dex"][1]), aligned["delta_q99_999_dex"], shuffled["delta_q99_999_dex"]], "Q3_maximum_excess_dex_V31_aligned_shuffled": [float(old["Q3_maximum_excess_dex"][1]), aligned["generated_max_above_truth_max_dex"], shuffled["generated_max_above_truth_max_dex"]], "Q4_V31_aligned_shuffled": [oq, aq, sq], "aligned_Q4_over_V31": aq / oq, "aligned_Q4_over_shuffled": aq / sq}
    selected = arms["aligned_patch"]; material = all(material_rows); primary = selected["Q3_all_domains"] and selected["Q4_all_domains"] and selected["all_three_field_pass"]; classification, next_step = classify(bool(primary), bool(selected["Q3_all_domains"]), bool(selected["Q4_all_domains"]), material)
    result = {"schema": SCHEMA, "experiment": "v39_bijective_local_patch_copula", "program": str(program_path.resolve()), "program_sha256": PROGRAM_SHA256, "gate_code_commit": commit, "worktree_clean_at_gate": True, "arms": arms, "comparison_to_v31_and_shuffled_control": comparison, "patch_alignment_material": material, "development_pass": bool(primary), "classification": classification, "next": next_step, "validation_truth_used_for_descriptor_fit_or_assignment": False, "additive_query_predictor": False, "donor_translation": False, "donor_reselection": False, "density_field_clipping": False, "posthoc_Ak_used": False, "Astrid_accessed": False, "historical_EAGLE_accessed": False}
    result["decision_digest_sha256"] = canonical_digest(result); return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--program", type=Path, required=True); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); commit, clean = git_state(args.repo.resolve())
    if not clean: raise RuntimeError("V39 gate requires clean worktree")
    result = evaluate(args.root.resolve(), args.program.resolve(), args.repo.resolve(), commit)
    if args.out.exists(): raise RuntimeError("V39 refuses existing decision")
    partial = args.out.with_suffix(args.out.suffix + ".partial"); partial.write_text(json.dumps(result, indent=2) + "\n"); os.replace(partial, args.out); print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
