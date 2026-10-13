#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V20-E8."""
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
from hong2021_v14_edm import V20_E8_SCHEMA
from hong2021_v15_development_gate import (
    DOMAINS,
    _load_metrics,
    _validate_ensemble,
    canonical_digest,
    git_state,
    select_candidate_rows,
)
from hong2021_v18_edm import _indices
from hong2021_v18_init import (
    SCHEMA as INIT_SCHEMA,
    measure_band_mode_variances,
    sha256_file,
)
from hong2021_v20_edm import (
    FROZEN_REGISTRY_SHA256,
    P_MEAN,
    P_STD,
    _validate_checkpoint,
    load_frozen_registry,
)
from hong2021_v20_gaussianize import CACHE_SCHEMA


SCHEMA = "hong2021-v20-integrity-bound-three-domain-decision-v1"
REGISTRY_DOMAINS = {"tng": "TNG100", "simba_dev": "SIMBA", "swift_dev": "Swift"}
Q5_CHECKS = (
    "density_pdf_improves_deterministic",
    "exact_dc_projection",
    "finite_ensemble_coverage_within_0.03",
    "high_k_total_power_within_10_percent",
    "rank_histogram_tv_at_most_0.05",
    "residual_rms_within_10_percent",
)


def _source_indices(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        return [int(value) for value in handle["source_index"][:]]


def _sampling_commit_is_ancestor(sampling_commit: str, gate_commit: str) -> bool:
    if len(sampling_commit) != 40 or len(gate_commit) != 40:
        return False
    return subprocess.run(
        [
            "git", "-C", str(Path.cwd()), "merge-base", "--is-ancestor",
            sampling_commit, gate_commit,
        ],
        capture_output=True,
    ).returncode == 0


def _parent_e7_decision(registry: dict[str, Any]) -> dict[str, Any]:
    row = registry["parent_evidence"]["development_decisions"]["E7"]
    path = Path(row["path"]).resolve()
    if sha256_file(path) != row["file_sha256"]:
        raise ValueError("V20 stored E7 decision hash mismatch")
    decision = json.loads(path.read_text())
    if canonical_digest(decision) != row["decision_digest_sha256"]:
        raise ValueError("V20 stored E7 decision digest mismatch")
    return decision


def _candidate(decision: dict[str, Any], step: int) -> dict[str, Any]:
    return next(row for row in decision["candidates"] if int(row["step"]) == step)


def _validate_draw_pairing(
    *, registry: dict[str, Any], step: int, domain: str,
    ensemble_path: Path, expected_indices: list[int],
) -> dict[str, Any]:
    e7 = _candidate(_parent_e7_decision(registry), step)["domains"][domain]
    baseline = Path(e7["ensemble"]).resolve()
    if sha256_file(baseline) != e7["ensemble_sha256"]:
        raise ValueError("V20 stored E7 draw-pairing ensemble hash mismatch")
    expected_seed = int(
        registry["e8_gaussianized_marginal_retrain"]["sampler"]["sampling_seeds"][
            REGISTRY_DOMAINS[domain]
        ]
    )
    with h5py.File(baseline, "r") as handle:
        baseline_seed = int(handle.attrs["seed"])
        baseline_indices = [int(value) for value in handle["source_index"][:]]
    with h5py.File(ensemble_path, "r") as handle:
        actual_seed = int(handle.attrs["seed"])
    if (
        baseline_seed != expected_seed
        or actual_seed != expected_seed
        or baseline_indices != expected_indices
    ):
        raise ValueError("V20 ensemble is not draw-pairable to E3/E6/E7")
    return {
        "paired": True,
        "seed": expected_seed,
        "E7_baseline": {"ensemble": str(baseline), "sha256": e7["ensemble_sha256"]},
        "E7_proves_pairing_to_E3_and_E6": bool(
            e7["draw_pairing_to_e3_and_e6"]["paired"]
        ),
    }


def _validate_v20_ensemble(
    path: Path, *, registry: dict[str, Any], domain: str, step: int,
    checkpoint_path: Path, checkpoint_sha: str, expected_indices: list[int],
    gate_commit: str,
) -> dict[str, Any]:
    experiment = registry["e8_gaussianized_marginal_retrain"]
    registry_domain = REGISTRY_DOMAINS[domain]
    seed = int(experiment["sampler"]["sampling_seeds"][registry_domain])
    _validate_ensemble(
        path, checkpoint=checkpoint_path, checkpoint_schema=V20_E8_SCHEMA,
        step=step, seed=seed,
    )
    if _source_indices(path) != expected_indices:
        raise ValueError("V20 ensemble source indices differ from frozen subset")
    data = experiment["data"][registry_domain]["validation_data"]
    cache = experiment["derived_caches"][registry_domain]["validation"]
    initialization = experiment["initialization_and_normalization"]
    transform = experiment["gaussianization"]
    with h5py.File(path, "r") as handle:
        if tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V20 development ensemble is not exactly 16x16x1x64^3")
        sampling_code_commit = str(handle.attrs.get("sampling_code_commit", ""))
        if not _sampling_commit_is_ancestor(sampling_code_commit, gate_commit):
            raise ValueError(
                "V20 ensemble sampling commit is not an ancestor of gate commit"
            )
        exact = {
            "checkpoint_sha256": checkpoint_sha,
            "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"],
            "init_schema": INIT_SCHEMA,
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": initialization[
                "measurement_report_sha256"
            ],
            "init_band_edges_h_mpc_json": json.dumps([0.0, 1.0, 3.0, 6.0, "infinity"]),
            "init_mode_counts_json": json.dumps(initialization["mode_counts_64"]),
            "init_band_mode_variances_json": json.dumps(
                initialization["source_balanced_band_mode_variance"]
            ),
            "init_sigma_nominal": 40.0,
            "init_fft_norm": "ortho",
            "init_fft_compute_dtype": "float64/complex128",
            "init_output_dtype": "float32",
            "init_additional_rng_draws": 0,
            "init_rng_pairing_self_check": True,
            "gaussianization_transform_sha256": transform["sha256"],
            "latent_inverse_additional_rng_draws": 0,
            "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD,
            "worktree_clean_at_sampling": True,
            "sampling_code_commit": sampling_code_commit,
        }
        for key, expected in exact.items():
            actual = handle.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                raise ValueError(f"V20 ensemble metadata differs: {key}")
        effective_sigma = float(handle.attrs.get("init_sigma_effective_first_step", -1))
        if not np.isfinite(effective_sigma) or abs(effective_sigma - 40.0) > 1.0e-4:
            raise ValueError("V20 effective first sigma is invalid")
        imaginary_ratio = float(
            handle.attrs.get("init_maximum_imaginary_over_real_rms", np.inf)
        )
        if imaginary_ratio > 1.0e-12:
            raise ValueError("V20 ensemble exceeded inherited imaginary tolerance")
    return {
        "seed": seed,
        "effective_sigma_first_step": effective_sigma,
        "maximum_imaginary_over_real_rms": imaginary_ratio,
        "sampling_code_commit": sampling_code_commit,
    }


def _remeasure_variance(registry: dict[str, Any]) -> dict[str, Any]:
    experiment = registry["e8_gaussianized_marginal_retrain"]
    caches = experiment["derived_caches"]
    specifications = {
        domain: {
            "path": caches[domain]["train"]["path"],
            "sha256": caches[domain]["train"]["sha256"],
            "objects": caches[domain]["train"]["objects"],
            "domain_attribute": {"TNG100": "TNG100", "SIMBA": "SIMBA", "Swift": "Swift-EAGLE"}[domain],
        }
        for domain in ("TNG100", "SIMBA", "Swift")
    }
    measured = measure_band_mode_variances(
        specifications,
        parseval_relative_tolerance=1.0e-12,
        maximum_absolute_ortho_dc=1.0e-9,
        allowed_schemas=(CACHE_SCHEMA,),
    )
    expected = np.asarray(
        experiment["initialization_and_normalization"][
            "source_balanced_band_mode_variance"
        ],
        dtype=np.float64,
    )
    actual = np.asarray(measured["source_balanced"], dtype=np.float64)
    relative = float(np.max(np.abs(actual - expected) / expected))
    if relative > 1.0e-9:
        raise ValueError("V20 train-cache variance remeasurement differs from registry")
    return {**measured, "maximum_relative_difference_from_registry": relative}


def marginal_diagnostics(path: Path) -> dict[str, Any]:
    """Compute frozen Q3/Q4 directly from one unchanged ensemble file."""
    with h5py.File(path, "r") as handle:
        truth_y = np.asarray(handle["truth"][:, 0], dtype=np.float64)
        truth_log10rho = 4.5 * truth_y
        truth_q = float(np.quantile(truth_log10rho, 0.99999))
        truth_max = float(truth_log10rho.max())
        truth_delta = np.power(10.0, truth_log10rho.astype(np.float64)) - 1.0
        truth_mean_delta_squared = float(np.mean(np.square(truth_delta)))
        sample = handle["sample"]
        generated_log10rho = np.empty(sample.size, dtype=np.float64)
        generated_delta_squared_sum = 0.0
        offset = 0
        for object_index in range(sample.shape[0]):
            values = np.asarray(sample[object_index, :, 0], dtype=np.float64)
            log_values = 4.5 * values
            flattened = log_values.reshape(-1)
            generated_log10rho[offset:offset + flattened.size] = flattened
            offset += flattened.size
            delta = np.power(10.0, log_values.astype(np.float64)) - 1.0
            generated_delta_squared_sum += float(np.square(delta).sum())
    generated_q = float(np.quantile(generated_log10rho, 0.99999))
    generated_max = float(generated_log10rho.max())
    generated_mean_delta_squared = generated_delta_squared_sum / generated_log10rho.size
    return {
        "truth_q99_999_log10rho": truth_q,
        "generated_q99_999_log10rho": generated_q,
        "delta_q99_999_dex": generated_q - truth_q,
        "truth_max_log10rho": truth_max,
        "generated_max_log10rho": generated_max,
        "generated_max_above_truth_max_dex": generated_max - truth_max,
        "truth_mean_delta_squared": truth_mean_delta_squared,
        "generated_mean_delta_squared": generated_mean_delta_squared,
        "generated_over_truth_mean_delta_squared": (
            generated_mean_delta_squared / truth_mean_delta_squared
        ),
        "definition": "log10rho=4.5*y; delta=10**log10rho-1; truth pooled over 16 unique cubes and generated pooled over 16x16 cubes",
    }


def _mechanism_passes(
    registry: dict[str, Any], final: dict[str, Any],
) -> tuple[bool, bool, bool]:
    diagnostics = registry["e8_gaussianized_marginal_retrain"][
        "mechanism_diagnostics"
    ]
    q3 = diagnostics["Q3"]
    q4 = diagnostics["Q4"]
    q3_pass = all(
        abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"])
        <= q3["per_domain_absolute_delta_q99_999_log10rho_max_dex"]
        and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"]
        <= q3["per_domain_generated_max_log10rho_above_truth_max_dex"]
        for row in final["domains"].values()
    )
    q4_pass = all(
        row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"]
        <= q4["per_domain_generated_over_truth_mean_delta_squared_max"]
        for row in final["domains"].values()
    )
    q5_pass = all(
        all(row["field_gate"]["checks"].get(check, False) for check in Q5_CHECKS)
        for row in final["domains"].values()
    )
    return q3_pass, q4_pass, q5_pass


def evaluate(
    *, root: Path, training: Path, registry_path: Path, repo: Path,
    gate_code_commit: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e8_gaussianized_marginal_retrain"]
    run_path = training / "run.json"
    run = json.loads(run_path.read_text())
    initialization = experiment["initialization_and_normalization"]
    if run.get("status") != "complete" or run.get("schema") != V20_E8_SCHEMA:
        raise ValueError("V20 training run is incomplete or has the wrong schema")
    if run.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("V20 training run registry provenance mismatch")
    if run.get("edm_p_mean") != P_MEAN or run.get("edm_p_std") != P_STD:
        raise ValueError("V20 training run noise distribution mismatch")
    if run.get("edm_p_mean_mode") != "log_sigma_data_fraction":
        raise ValueError("V20 training run p_mean mode mismatch")
    if run.get("sigma_data") != initialization["sigma_data"]:
        raise ValueError("V20 training run sigma_data mismatch")
    variance = _remeasure_variance(registry)
    expected_indices = {
        domain: _indices(
            experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo
        )
        for domain in DOMAINS
    }
    candidates = []
    for step in experiment["candidate_steps"]:
        checkpoint_path = training / "validation_checkpoints" / f"step_{step:06d}.pt"
        checkpoint, checkpoint_sha = _validate_checkpoint(
            checkpoint_path, step=step, registry=registry
        )
        training_commit = str(checkpoint.get("code_commit_at_launch", ""))
        if len(training_commit) != 40 or subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", training_commit, gate_code_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V20 training commit is not an ancestor of gate commit")
        domains: dict[str, Any] = {}
        for domain in DOMAINS:
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            init_metadata = _validate_v20_ensemble(
                ensemble_path, registry=registry, domain=domain, step=step,
                checkpoint_path=checkpoint_path, checkpoint_sha=checkpoint_sha,
                expected_indices=expected_indices[domain], gate_commit=gate_code_commit,
            )
            pairing = _validate_draw_pairing(
                registry=registry, step=step, domain=domain,
                ensemble_path=ensemble_path, expected_indices=expected_indices[domain],
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V20 metrics refer to a different ensemble")
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()),
                "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "draw_pairing_to_e3_e6_e7": pairing,
                "initialization_metadata": init_metadata,
                "mechanism_Q3_Q4": (
                    marginal_diagnostics(ensemble_path) if step == 10000 else None
                ),
            }
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_training_code_commit": training_commit,
            "gradient_diagnostic": checkpoint.get("gradient_diagnostic"),
            "domains": domains,
            "all_three_pass": all(row["field_gate"]["pass"] for row in domains.values()),
        })
    selected = select_candidate_rows(candidates)
    final = candidates[-1]
    q3_pass, q4_pass, q5_pass = _mechanism_passes(registry, final)
    if selected is not None:
        classification = None
        next_step = "freeze_exact_v20_hashes_before_astrid_one_shot"
    elif q3_pass and q4_pass:
        classification = {
            "class": "marginal_constrained_spatial_structure_limited",
            "Q3_all_domains": True,
            "Q4_all_domains": True,
            "Q5_all_domains": q5_pass,
            "next": "stop_unopened_and_audit_joint_spatial_structure_not_marginals",
        }
        next_step = classification["next"]
    else:
        classification = {
            "class": "gaussianized_marginal_falsified_audit_conditional_structure",
            "Q3_all_domains": q3_pass,
            "Q4_all_domains": q4_pass,
            "Q5_all_domains": q5_pass,
            "next": "stop_unopened_and_audit_conditional_structure",
        }
        next_step = classification["next"]
    report = {
        "schema": SCHEMA,
        "experiment": "e8_gaussianized_marginal_retrain",
        "registry": str(registry_path.resolve()),
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "transform_sha256": experiment["gaussianization"]["sha256"],
        "training": str(training.resolve()),
        "training_run_sha256": sha256_file(run_path),
        "gate_code_commit": gate_code_commit,
        "worktree_clean_at_gate": True,
        "EAGLE_RefL0100N1504_used": False,
        "Astrid_used": False,
        "independent_data_paths_accessed_by_gate": False,
        "initialization_variance_remeasurement": variance,
        "predeclared_steps": list(experiment["candidate_steps"]),
        "selection_rule": experiment["gate_selection"],
        "mechanism_diagnostics_selection_role": "none",
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "mechanism_10k": {
            "Q3_all_domains": q3_pass,
            "Q4_all_domains": q4_pass,
            "Q5_all_domains": q5_pass,
        },
        "failure_classification_10k": classification,
        "next": next_step,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V20 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root, training=args.training, registry_path=args.registry,
        repo=args.repo, gate_code_commit=commit,
    )
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if canonical_digest(existing) != existing.get("decision_digest_sha256"):
            raise RuntimeError("existing V20 decision has an invalid digest")
        if existing != report:
            raise RuntimeError("existing V20 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
