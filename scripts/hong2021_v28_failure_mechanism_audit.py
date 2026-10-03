#!/usr/bin/env python
"""Audit V21/V14 reconstruction, donor matching, transfer, and 2PCF after V28."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

from hong2021_v14_multiscale import inverse_standardized_residual
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v21_conditional_affine import inverse_cube
from hong2021_v26 import CACHE_KEYS
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER, load_frozen_program
from hong2021_v28_failure_audit import (
    AUDIT_PROGRAM_SHA256,
    PhysicalTailAccumulator,
    classify,
    compare_physical_tails,
)


SCHEMA = "hong2021-v28-representation-backbone-failure-mechanism-audit-v1"


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs from frozen V28 failure audit")
    return json.loads(path.read_text())


def _error_report(square_sum: float, count: int, maximum: float) -> dict[str, float | int]:
    return {
        "values": count,
        "rms": float(np.sqrt(square_sum / count)),
        "maximum_absolute": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V28 failure mechanism audit requires a clean worktree")
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V28 mechanism audit: {output}")
    program = _verified_json(
        args.audit_program.resolve(), AUDIT_PROGRAM_SHA256, "V28 audit program"
    )
    if (
        program.get("schema")
        != "hong2021-v28-representation-backbone-failure-audit-program-v1"
        or program.get("status") != "frozen_before_mechanism_audit_execution"
        or program.get("firewall", {}).get("Astrid_accessed") is not False
        or program.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V28 failure-audit program or firewall differs")
    parent = program["parent"]
    registry_path = (repo / parent["v28_registry"]).resolve()
    if sha256_file(registry_path) != parent["v28_registry_sha256"]:
        raise ValueError("V28 failure-audit registry differs")
    _, artifacts, v20 = load_frozen_program(registry_path, repo)
    preflight = _verified_json(
        Path(parent["preflight"]), parent["preflight_sha256"], "V28 preflight"
    )
    decision = _verified_json(
        Path(parent["decision"]), parent["decision_sha256"], "V28 decision"
    )
    if (
        decision.get("decision_digest_sha256") != parent["decision_digest_sha256"]
        or decision.get("classification", {}).get("class") != parent["failure_class"]
        or decision.get("next") != parent["required_next"]
        or preflight.get("status") != "pass"
    ):
        raise ValueError("V28 failure decision or preflight conclusion differs")

    experiment = v20["e8_gaussianized_marginal_retrain"]
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    evaluation = Path(parent["decision"]).parent / "development_candidate"
    train_data_handles = {
        source: h5py.File(
            experiment["data"][source]["train_data"]["path"], "r"
        )
        for source in DOMAIN_ORDER
    }
    domains: dict[str, Any] = {}
    try:
        for source in DOMAIN_ORDER:
            domain = DOMAIN_KEYS[source]
            frozen = program["frozen_inputs"][domain]
            domain_root = evaluation / domain
            ensemble_path = domain_root / "ensemble16.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            if (
                sha256_file(ensemble_path) != frozen["ensemble_sha256"]
                or sha256_file(metrics_path) != frozen["metrics_sha256"]
            ):
                raise ValueError(f"V28 {domain} frozen ensemble or metrics differ")
            data_info = experiment["data"][source]
            v21_info = artifacts["caches"][CACHE_KEYS[source]["validation"]]
            indices = _indices(experiment["development_objects"][source], repo)
            truth_bank = PhysicalTailAccumulator()
            self_bank = PhysicalTailAccumulator()
            donor_bank = PhysicalTailAccumulator()
            generated_bank = PhysicalTailAccumulator()
            donor_by_source = {name: PhysicalTailAccumulator() for name in DOMAIN_ORDER}
            generated_by_source = {name: PhysicalTailAccumulator() for name in DOMAIN_ORDER}
            standardized_square = physical_square = 0.0
            standardized_count = physical_count = 0
            standardized_max = physical_max = 0.0
            distances: list[float] = []
            generated_maxima: list[float] = []
            with h5py.File(data_info["validation_data"]["path"], "r") as data, h5py.File(
                v21_info["path"], "r"
            ) as v21, h5py.File(
                data_info["source_validation_cache"]["path"], "r"
            ) as v14, h5py.File(ensemble_path, "r") as ensemble:
                if [int(value) for value in ensemble["source_index"][:]] != indices:
                    raise ValueError(f"V28 {domain} source indices differ in audit")
                for object_index, data_index in enumerate(indices):
                    truth = np.asarray(data["target"][data_index, 0], dtype=np.float32)
                    truth_bank.update(truth)
                    latent = np.asarray(
                        v21["standardized_residual"][data_index, 0], dtype=np.float32
                    )
                    corrected_mean = np.asarray(
                        v21["conditional_mean"][data_index, 0], dtype=np.float32
                    )
                    reconstructed_standardized = inverse_cube(
                        latent, corrected_mean, profile, transform
                    )
                    original_standardized = np.asarray(
                        v14["standardized_residual"][data_index, 0], dtype=np.float32
                    )
                    difference = reconstructed_standardized.astype(np.float64) - original_standardized.astype(np.float64)
                    standardized_square += float(np.square(difference).sum())
                    standardized_count += difference.size
                    standardized_max = max(standardized_max, float(np.abs(difference).max()))
                    location = float(v21["predicted_residual_dc"][data_index])
                    scales = np.asarray(
                        v21["predicted_band_scales"][data_index], dtype=np.float64
                    )
                    physical_residual = inverse_standardized_residual(
                        reconstructed_standardized,
                        predicted_location=location,
                        predicted_scales=scales,
                        voxel_mpc_h=float(v21.attrs["voxel_mpc_h"]),
                    )
                    reconstructed_y = corrected_mean + physical_residual
                    self_bank.update(reconstructed_y)
                    physical_difference = reconstructed_y.astype(np.float64) - truth.astype(np.float64)
                    physical_square += float(np.square(physical_difference).sum())
                    physical_count += physical_difference.size
                    physical_max = max(physical_max, float(np.abs(physical_difference).max()))
                    for member in range(16):
                        donor_source = DOMAIN_ORDER[
                            int(ensemble["donor_source"][object_index, member])
                        ]
                        donor_index = int(
                            ensemble["donor_index"][object_index, member]
                        )
                        donor_truth = np.asarray(
                            train_data_handles[donor_source]["target"][donor_index, 0],
                            dtype=np.float32,
                        )
                        generated = np.asarray(
                            ensemble["sample"][object_index, member, 0],
                            dtype=np.float32,
                        )
                        donor_bank.update(donor_truth)
                        generated_bank.update(generated)
                        donor_by_source[donor_source].update(donor_truth)
                        generated_by_source[donor_source].update(generated)
                        distances.append(
                            float(ensemble["donor_distance"][object_index, member, 2])
                        )
                        generated_maxima.append(float(4.5 * generated.max()))
            truth_report = truth_bank.report()
            self_report = self_bank.report()
            donor_report = donor_bank.report()
            generated_report = generated_bank.report()
            metrics = json.loads(metrics_path.read_text())["candidates"]["edm"]
            two_point = metrics["two_point_cosmic_mean"]
            correlation = spearmanr(distances, generated_maxima)
            domains[domain] = {
                "source": source,
                "objects": len(indices),
                "members_per_object": 16,
                "validation_truth": truth_report,
                "self_reconstructed_validation": self_report,
                "selected_donor_train_truths": donor_report,
                "cross_inverted_generated": generated_report,
                "self_reconstruction_vs_truth": compare_physical_tails(
                    truth_report, self_report
                ),
                "selected_donor_truths_vs_validation_truth": compare_physical_tails(
                    truth_report, donor_report
                ),
                "cross_generated_vs_selected_donor_truths": compare_physical_tails(
                    donor_report, generated_report
                ),
                "cross_generated_vs_validation_truth": compare_physical_tails(
                    truth_report, generated_report
                ),
                "representation_value_errors": {
                    "V21_inverse_vs_V14_standardized_residual": _error_report(
                        standardized_square, standardized_count, standardized_max
                    ),
                    "complete_self_inverse_y_vs_target_y": _error_report(
                        physical_square, physical_count, physical_max
                    ),
                },
                "by_selected_donor_source": {
                    name: {
                        "donor_train_truth": donor_by_source[name].report(),
                        "cross_inverted_generated": generated_by_source[name].report(),
                        "cross_vs_donor": compare_physical_tails(
                            donor_by_source[name].report(),
                            generated_by_source[name].report(),
                        ),
                    }
                    for name in DOMAIN_ORDER
                },
                "matching_distance_vs_generated_maximum_spearman": {
                    "statistic": float(correlation.statistic),
                    "pvalue": float(correlation.pvalue),
                    "pairs": len(distances),
                    "selection_role": "none",
                },
                "two_point_improves_deterministic_all_scales": decision["candidate"]["domains"][domain]["field_gate"]["checks"]["two_point_improves_deterministic_all_scales"],
                "two_point_cosmic_mean": two_point,
            }
            print(f"[audit] {source} complete", flush=True)
    finally:
        for handle in train_data_handles.values():
            handle.close()
    classification = classify(domains)
    report = {
        "schema": SCHEMA,
        "status": "complete_development_only_failure_mechanism_audit",
        "audit_program": str(args.audit_program.resolve()),
        "audit_program_sha256": AUDIT_PROGRAM_SHA256,
        "registry": str(registry_path),
        "registry_sha256": parent["v28_registry_sha256"],
        "decision": parent["decision"],
        "decision_sha256": parent["decision_sha256"],
        "decision_digest_sha256": parent["decision_digest_sha256"],
        "execution_host": socket.gethostname(),
        "audit_code_commit": commit,
        "worktree_clean_at_audit": clean,
        "domains": domains,
        "classification": classification,
        "new_sampling": False,
        "donor_reselection": False,
        "thresholds_changed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(classification, indent=2))
    print(f"[out] {output}")


if __name__ == "__main__":
    main()
