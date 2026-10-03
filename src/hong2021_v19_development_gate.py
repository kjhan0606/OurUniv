#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V19-E7."""
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
from hong2021_v14_edm import V19_E7_SCHEMA
from hong2021_v14_freeze import astrid_files
from hong2021_v15_development_gate import (
    DOMAINS,
    _load_metrics,
    _validate_ensemble,
    canonical_digest,
    git_state,
    select_candidate_rows,
)
from hong2021_v18_development_gate import _only_swift_peak_void
from hong2021_v18_edm import _indices
from hong2021_v18_init import (
    SCHEMA as INIT_SCHEMA,
    measure_band_mode_variances,
    sha256_file,
)
from hong2021_v19_edm import (
    FROZEN_REGISTRY_SHA256,
    P_MEAN,
    P_STD,
    _validate_checkpoint,
    load_frozen_registry,
)
from hong2021_v19_spectroscopy import score_spectroscopy


SCHEMA = "hong2021-v19-integrity-bound-three-domain-decision-v1"
POWER_BANDS = ("0.3-1_h_mpc", "1-3_h_mpc", "3-6_h_mpc", "6-10.0531_h_mpc")
REGISTRY_DOMAINS = {"tng": "TNG100", "simba_dev": "SIMBA", "swift_dev": "Swift"}
DEFAULT_ASTRID = Path("/gpfs/kjhan/CAMELS/Astrid/L25n256")


def _source_indices(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        return [int(value) for value in handle["source_index"][:]]


def _stored_decisions(
    registry: dict[str, Any], repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = registry["parent_evidence"]
    e6_path = Path(parent["e6_decision"]).resolve()
    if sha256_file(e6_path) != parent["e6_decision_sha256"]:
        raise ValueError("stored E6 decision changed after V19 registry validation")
    e6 = json.loads(e6_path.read_text())
    if canonical_digest(e6) != parent["e6_decision_digest_sha256"]:
        raise ValueError("stored E6 decision digest changed")
    v18_path = (repo / parent["v18_registry"]).resolve()
    if sha256_file(v18_path) != parent["v18_registry_sha256"]:
        raise ValueError("stored V18 registry changed after V19 validation")
    v18 = json.loads(v18_path.read_text())
    e3_path = Path(v18["parent_evidence"]["e3_decision"])
    if sha256_file(e3_path) != v18["parent_evidence"]["e3_decision_sha256"]:
        raise ValueError("stored E3 decision changed after V19 registry validation")
    e3 = json.loads(e3_path.read_text())
    if canonical_digest(e3) != v18["parent_evidence"]["e3_decision_digest_sha256"]:
        raise ValueError("stored E3 decision digest changed")
    return e3, e6


def _candidate(decision: dict[str, Any], step: int) -> dict[str, Any]:
    return next(row for row in decision["candidates"] if int(row["step"]) == step)


def _validate_draw_pairing(
    *, registry: dict[str, Any], step: int, domain: str,
    ensemble_path: Path, expected_indices: list[int], repo: Path,
) -> dict[str, Any]:
    e3, e6 = _stored_decisions(registry, repo)
    expected_seed = int(
        registry["e7_band_anchored_noise_retrain"]["sampling_seeds"][
            REGISTRY_DOMAINS[domain]
        ]
    )
    baselines = {}
    for label, decision in (("E3", e3), ("E6", e6)):
        row = _candidate(decision, step)["domains"][domain]
        path = Path(row["ensemble"]).resolve()
        digest = sha256_file(path)
        if digest != row["ensemble_sha256"]:
            raise ValueError(f"stored {label} ensemble hash mismatch")
        with h5py.File(path, "r") as handle:
            seed = int(handle.attrs["seed"])
            indices = [int(value) for value in handle["source_index"][:]]
        if seed != expected_seed or indices != expected_indices:
            raise ValueError(f"stored {label} ensemble is not draw-pairable to E7")
        baselines[label] = {"ensemble": str(path), "sha256": digest, "seed": seed}
    with h5py.File(ensemble_path, "r") as handle:
        if int(handle.attrs["seed"]) != expected_seed:
            raise ValueError("V19 ensemble seed differs from E3/E6 pairing")
    return {"paired": True, "seed": expected_seed, "baselines": baselines}


def _validate_e7_ensemble(
    path: Path, *, registry: dict[str, Any], domain: str, step: int,
    checkpoint_path: Path, checkpoint_sha: str, expected_indices: list[int],
    gate_commit: str,
) -> dict[str, Any]:
    experiment = registry["e7_band_anchored_noise_retrain"]
    registry_domain = REGISTRY_DOMAINS[domain]
    seed = int(experiment["sampling_seeds"][registry_domain])
    _validate_ensemble(
        path, checkpoint=checkpoint_path, checkpoint_schema=V19_E7_SCHEMA,
        step=step, seed=seed,
    )
    if _source_indices(path) != expected_indices:
        raise ValueError("V19 ensemble source indices differ from frozen subset")
    data = experiment["data"][registry_domain]
    init = experiment["initialization"]
    with h5py.File(path, "r") as handle:
        if tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V19 development ensemble is not exactly 16x16x1x64^3")
        if tuple(handle["conditional_mean"].shape) != (16, 1, 64, 64, 64):
            raise ValueError("V19 conditional mean shape mismatch")
        if tuple(handle["truth"].shape) != (16, 1, 64, 64, 64):
            raise ValueError("V19 truth shape mismatch")
        exact = {
            "checkpoint_sha256": checkpoint_sha,
            "source_cache_sha256": data["validation_cache_sha256"],
            "source_data_sha256": data["validation_data_sha256"],
            "init_schema": INIT_SCHEMA,
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": init["measurement_report_sha256"],
            "init_band_edges_h_mpc_json": json.dumps([0.0, 1.0, 3.0, 6.0, "infinity"]),
            "init_mode_counts_json": json.dumps(init["expected_mode_counts_by_grid"]["64"]),
            "init_band_mode_variances_json": json.dumps(init["source_balanced_per_band_mode_variance"]),
            "init_sigma_nominal": 40.0,
            "init_fft_norm": "ortho",
            "init_fft_compute_dtype": "float64/complex128",
            "init_output_dtype": "float32",
            "init_additional_rng_draws": 0,
            "init_rng_pairing_self_check": True,
            "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD,
            "worktree_clean_at_sampling": True,
            "sampling_code_commit": gate_commit,
        }
        for key, value in exact.items():
            actual = handle.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != value:
                raise ValueError(f"V19 ensemble metadata differs: {key}")
        effective_sigma = float(handle.attrs.get("init_sigma_effective_first_step", -1))
        if not np.isfinite(effective_sigma) or abs(effective_sigma - 40.0) > 1.0e-4:
            raise ValueError("V19 effective first sigma is invalid")
        imaginary_ratio = float(
            handle.attrs.get("init_maximum_imaginary_over_real_rms", np.inf)
        )
        if imaginary_ratio > float(init["maximum_imaginary_over_real_rms"]):
            raise ValueError("V19 ensemble exceeded imaginary-leakage tolerance")
    return {
        "seed": seed,
        "effective_sigma_first_step": effective_sigma,
        "maximum_imaginary_over_real_rms": imaginary_ratio,
        "sampling_code_commit": gate_commit,
    }


def _remeasure_variance(registry: dict[str, Any]) -> dict[str, Any]:
    experiment = registry["e7_band_anchored_noise_retrain"]
    source = experiment["data"]
    specifications = {
        "TNG100": {
            "path": source["TNG100"]["train_cache"],
            "sha256": source["TNG100"]["train_cache_sha256"],
            "objects": 432, "domain_attribute": "TNG100",
        },
        "SIMBA": {
            "path": source["SIMBA"]["train_cache"],
            "sha256": source["SIMBA"]["train_cache_sha256"],
            "objects": 202, "domain_attribute": "SIMBA",
        },
        "Swift": {
            "path": source["Swift"]["train_cache"],
            "sha256": source["Swift"]["train_cache_sha256"],
            "objects": 409, "domain_attribute": "Swift-EAGLE",
        },
    }
    measured = measure_band_mode_variances(specifications)
    expected = np.asarray(
        experiment["initialization"]["source_balanced_per_band_mode_variance"],
        dtype=np.float64,
    )
    actual = np.asarray(measured["source_balanced"], dtype=np.float64)
    relative = float(np.max(np.abs(actual - expected) / expected))
    if relative > 1.0e-9:
        raise ValueError("V19 train-cache variance remeasurement differs from registry")
    return {**measured, "maximum_relative_difference_from_registry": relative}


def _load_e3_10k(registry: dict[str, Any], repo: Path) -> dict[str, Any]:
    e3, _ = _stored_decisions(registry, repo)
    return _candidate(e3, 10000)


def _previously_passing_regressions(
    registry: dict[str, Any], final: dict[str, Any], repo: Path
) -> list[str]:
    baseline = _load_e3_10k(registry, repo)
    regressions = []
    for domain in DOMAINS:
        before = baseline["domains"][domain]["field_gate"]["checks"]
        after = final["domains"][domain]["field_gate"]["checks"]
        for check, passed in before.items():
            if passed and not after.get(check, False):
                regressions.append(f"{domain}:{check}")
    return regressions


def _tng_fails_only_zero_to_one(final: dict[str, Any]) -> bool:
    domains = final["domains"]
    checks = domains["tng"]["field_gate"]["checks"]
    two_name = "two_point_improves_deterministic_all_scales"
    if checks.get(two_name) is not False:
        return False
    if not all(value for name, value in checks.items() if name != two_name):
        return False
    scales = domains["tng"]["two_point_by_scale"]
    return bool(
        not scales["0-1_mpc_h"]["improves_deterministic"]
        and scales["1-3_mpc_h"]["improves_deterministic"]
        and scales["3-10_mpc_h"]["improves_deterministic"]
    )


def classify_failure(
    *, registry: dict[str, Any], final: dict[str, Any], q1_pass: bool, repo: Path
) -> dict[str, Any]:
    q2_pass = all(
        row["mechanism_Q2"]["pass"] for row in final["domains"].values()
    )
    regressions = _previously_passing_regressions(registry, final, repo)
    tng_signature = _tng_fails_only_zero_to_one(final)
    if q1_pass and q2_pass and tng_signature:
        label = "sigma_coverage_confirmed_exhausted_audit_symmetric_2pcf_and_nonlinear_predictability"
        next_step = "stop_unopened_and_audit_symmetric_2pcf_plus_nonlinear_stage1_band0_predictability"
    elif not q1_pass:
        label = "sigma_coverage_falsified_audit_architecture_representation"
        next_step = "stop_unopened_and_audit_architecture_or_representation"
    elif regressions:
        label = "sigma_coverage_passed_previously_passing_checks_regressed_discard_v19_distribution"
        next_step = "stop_unopened_discard_v19_distribution_without_interpolation"
    else:
        label = "predeclared_unclassified_failure_obtain_independent_audit"
        next_step = "stop_unopened_and_obtain_new_independent_design_audit"
    return {
        "class": label,
        "Q1_all_domains": q1_pass,
        "Q2_all_domains": q2_pass,
        "TNG_only_0_1_two_point_signature": tng_signature,
        "previously_passing_check_regressions": regressions,
        "next": next_step,
    }


def evaluate(
    *, root: Path, training: Path, registry_path: Path, repo: Path,
    gate_code_commit: str, astrid_root: Path = DEFAULT_ASTRID,
) -> dict[str, Any]:
    repo = repo.resolve()
    if astrid_files(astrid_root.resolve()):
        raise RuntimeError("Astrid must remain unopened during the V19 gate")
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e7_band_anchored_noise_retrain"]
    run_path = training / "run.json"
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete" or run.get("schema") != V19_E7_SCHEMA:
        raise ValueError("V19 training run is incomplete or has the wrong schema")
    if run.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("V19 training run registry provenance mismatch")
    if run.get("edm_p_mean") != P_MEAN or run.get("edm_p_std") != P_STD:
        raise ValueError("V19 training run noise distribution mismatch")
    if run.get("edm_p_mean_mode") != "fixed_log_sigma":
        raise ValueError("V19 training run p_mean mode mismatch")
    variance = _remeasure_variance(registry)
    expected_indices = {
        domain: _indices(
            experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo
        ) for domain in DOMAINS
    }
    q2_spec = experiment["mechanism_diagnostics"]["Q2"]
    candidates = []
    checkpoints: dict[int, tuple[Path, dict[str, Any], str]] = {}
    for step in experiment["candidate_steps"]:
        checkpoint_path = training / "validation_checkpoints" / f"step_{step:06d}.pt"
        checkpoint, checkpoint_sha = _validate_checkpoint(
            checkpoint_path, step=step, registry=registry
        )
        checkpoints[step] = (checkpoint_path, checkpoint, checkpoint_sha)
        training_commit = str(checkpoint.get("code_commit_at_launch", ""))
        if len(training_commit) != 40 or subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", training_commit, gate_code_commit]
        ).returncode:
            raise ValueError("V19 training commit is not an ancestor of gate commit")
        domains: dict[str, Any] = {}
        for domain in DOMAINS:
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            init_metadata = _validate_e7_ensemble(
                ensemble_path, registry=registry, domain=domain, step=step,
                checkpoint_path=checkpoint_path, checkpoint_sha=checkpoint_sha,
                expected_indices=expected_indices[domain], gate_commit=gate_code_commit,
            )
            pairing = _validate_draw_pairing(
                registry=registry, step=step, domain=domain,
                ensemble_path=ensemble_path, expected_indices=expected_indices[domain],
                repo=repo,
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V19 metrics refer to a different ensemble")
            registry_domain = REGISTRY_DOMAINS[domain]
            residual_power = metrics["fourier_log_density"][
                "generated_residual_power_over_truth_residual"
            ]
            band0 = float(residual_power[POWER_BANDS[0]])
            threshold = float(q2_spec["required_band0_ratio"][registry_domain])
            two = metrics["two_point_cosmic_mean"]
            by_scale = {}
            for scale in ("0-1_mpc_h", "1-3_mpc_h", "3-10_mpc_h"):
                generated = float(two["generated_vs_truth_ks"]["by_scale"][scale]["mean"])
                deterministic = float(two["deterministic_vs_truth_ks"]["by_scale"][scale]["mean"])
                by_scale[scale] = {
                    "generated": generated,
                    "deterministic": deterministic,
                    "improves_deterministic": generated < deterministic,
                }
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()),
                "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "draw_pairing_to_e3_and_e6": pairing,
                "initialization_metadata": init_metadata,
                "report_only_residual_power_over_truth": {
                    band: float(residual_power[band]) for band in POWER_BANDS
                },
                "mechanism_Q2": {
                    "band0_ratio": band0,
                    "required_ratio": threshold,
                    "predeclared_ceiling": float(
                        q2_spec["predeclared_ceiling"][registry_domain]
                    ),
                    "pass": band0 >= threshold,
                    "selection_role": "none",
                },
                "two_point_by_scale": by_scale,
            }
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_training_code_commit": checkpoint["code_commit_at_launch"],
            "gradient_diagnostic": checkpoint.get("gradient_diagnostic"),
            "domains": domains,
            "all_three_pass": all(row["field_gate"]["pass"] for row in domains.values()),
        })
    spectroscopy = score_spectroscopy(
        checkpoint_path=checkpoints[10000][0], registry=registry, repo=repo
    )
    selected = select_candidate_rows(candidates)
    final = candidates[-1]
    if selected is not None:
        classification = None
        next_step = "freeze_exact_v19_hashes_before_astrid_one_shot"
    elif _only_swift_peak_void(final):
        classification = {
            "class": "only_swift_peak_void_reserved_conditional_scale_branch",
            "Q1_all_domains": spectroscopy["Q1_pass"],
            "Q2_all_domains": all(
                row["mechanism_Q2"]["pass"] for row in final["domains"].values()
            ),
        }
        next_step = "stop_unopened_then_predeclare_all_band_target_free_conditional_scale"
    else:
        classification = classify_failure(
            registry=registry, final=final, q1_pass=bool(spectroscopy["Q1_pass"]),
            repo=repo,
        )
        next_step = classification["next"]
    report = {
        "schema": SCHEMA,
        "experiment": "e7_band_anchored_noise_retrain",
        "registry": str(registry_path.resolve()),
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "training": str(training.resolve()),
        "training_run_sha256": sha256_file(run_path),
        "gate_code_commit": gate_code_commit,
        "worktree_clean_at_gate": True,
        "EAGLE_RefL0100N1504_used": False,
        "Astrid_used": False,
        "initialization_variance_remeasurement": variance,
        "predeclared_steps": list(experiment["candidate_steps"]),
        "selection_rule": experiment["gate_selection"],
        "mechanism_diagnostics_selection_role": "none",
        "score_spectroscopy_Q1": spectroscopy,
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
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
    parser.add_argument("--astrid-root", type=Path, default=DEFAULT_ASTRID)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V19 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root, training=args.training, registry_path=args.registry,
        repo=args.repo, gate_code_commit=commit, astrid_root=args.astrid_root,
    )
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if canonical_digest(existing) != existing.get("decision_digest_sha256"):
            raise RuntimeError("existing V19 decision has an invalid digest")
        if existing != report:
            raise RuntimeError("existing V19 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
