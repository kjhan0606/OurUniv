#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V18-E6."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v6_gate import field_gate
from hong2021_v14_edm import V15_E3_SCHEMA
from hong2021_v14_freeze import astrid_files
from hong2021_v15_development_gate import (
    DOMAINS,
    _load_metrics,
    _validate_checkpoint as _validate_v15_checkpoint,
    _validate_ensemble,
    canonical_digest,
    git_state,
    select_candidate_rows,
)
from hong2021_v18_edm import FROZEN_REGISTRY_SHA256, load_frozen_registry
from hong2021_v18_init import (
    PriorMatchedInitializer,
    SCHEMA as INIT_SCHEMA,
    measure_band_mode_variances,
    prior_matched_spectral_std,
    sha256_file,
)


SCHEMA = "hong2021-v18-integrity-bound-three-domain-decision-v1"
POWER_BANDS = ("0.3-1_h_mpc", "1-3_h_mpc", "3-6_h_mpc", "6-10.0531_h_mpc")
REGISTRY_DOMAINS = {"tng": "TNG100", "simba_dev": "SIMBA", "swift_dev": "Swift"}
DEFAULT_ASTRID = Path("/gpfs/kjhan/CAMELS/Astrid/L25n256")


def _expected_indices(specification: list[int] | dict[str, str], repo: Path) -> list[int]:
    if isinstance(specification, list):
        values = [int(value) for value in specification]
    else:
        path = (repo / specification["path"]).resolve()
        if sha256_file(path) != specification["sha256"]:
            raise ValueError("V18 development subset file differs from its frozen hash")
        values = [int(value) for value in json.loads(path.read_text())["indices"]]
    if len(values) != 16 or len(set(values)) != 16 or min(values) < 0:
        raise ValueError("V18 development subset must contain 16 unique indices")
    return values


def _source_indices(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        return [int(value) for value in handle["source_index"][:]]


def _checkpoint_row(registry: dict[str, Any], step: int) -> dict[str, Any]:
    return next(
        row for row in registry["parent_evidence"]["e3_checkpoints"]
        if int(row["step"]) == step
    )


def _validate_checkpoint(
    registry: dict[str, Any], step: int
) -> tuple[Path, dict[str, Any]]:
    row = _checkpoint_row(registry, step)
    path = Path(row["path"]).resolve()
    if sha256_file(path) != row["sha256"]:
        raise ValueError("V18 E3 checkpoint hash mismatch")
    checkpoint = _validate_v15_checkpoint(
        path, step=step, experiment="e3",
        registry_sha256=registry["parent_evidence"]["v15_registry_sha256"],
    )
    if checkpoint.get("schema") != V15_E3_SCHEMA:
        raise ValueError("V18 checkpoint is not V15-E3")
    if checkpoint.get("code_commit_at_launch") != registry["parent_evidence"][
        "e3_training_code_commit"
    ]:
        raise ValueError("V18 checkpoint training commit mismatch")
    tail = checkpoint.get("tail_weight_fit", {})
    if float(tail.get("pre_normalization_maximum", -1)) != 10.0:
        raise ValueError("V18 checkpoint tail maximum mismatch")
    return path, checkpoint


def _validate_e6_ensemble(
    path: Path, *, registry: dict[str, Any], registry_path: Path,
    domain: str, step: int, checkpoint_path: Path, expected_indices: list[int],
) -> dict[str, Any]:
    experiment = registry["e6_prior_matched_initialization"]
    registry_domain = REGISTRY_DOMAINS[domain]
    seed = int(experiment["sampling_seeds"][registry_domain])
    _validate_ensemble(
        path, checkpoint=checkpoint_path, checkpoint_schema=V15_E3_SCHEMA,
        step=step, seed=seed,
    )
    if _source_indices(path) != expected_indices:
        raise ValueError("V18 ensemble source indices differ from the frozen subset")
    init = experiment["initialization"]
    inputs = experiment["validation_inputs"][registry_domain]
    with h5py.File(path, "r") as handle:
        expected_sample_shape = (16, 16, 1, 64, 64, 64)
        expected_field_shape = (16, 1, 64, 64, 64)
        if tuple(handle["sample"].shape) != expected_sample_shape:
            raise ValueError("V18 development ensemble is not exactly 16x16x1x64^3")
        if tuple(handle["conditional_mean"].shape) != expected_field_shape:
            raise ValueError("V18 development conditional mean is not exactly 16x1x64^3")
        if tuple(handle["truth"].shape) != expected_field_shape:
            raise ValueError("V18 development truth is not exactly 16x1x64^3")
        exact = {
            "checkpoint_sha256": _checkpoint_row(registry, step)["sha256"],
            "source_cache_sha256": inputs["cache_sha256"],
            "source_data_sha256": inputs["data_sha256"],
            "init_schema": INIT_SCHEMA,
            "init_registry_sha256": FROZEN_REGISTRY_SHA256,
            "init_measurement_report_sha256": init["measurement_report_sha256"],
            "init_band_edges_h_mpc_json": json.dumps(init["band_edges_h_mpc"]),
            "init_mode_counts_json": json.dumps(init["full_fft_mode_counts"]),
            "init_band_mode_variances_json": json.dumps(
                init["source_balanced_per_band_mode_variance"]
            ),
            "init_sigma_nominal": 40.0,
            "init_fft_norm": "ortho",
            "init_fft_compute_dtype": "float64/complex128",
            "init_output_dtype": "float32",
            "init_additional_rng_draws": 0,
            "init_rng_pairing_self_check": True,
            "worktree_clean_at_sampling": True,
        }
        for key, value in exact.items():
            actual = handle.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != value:
                raise ValueError(f"V18 ensemble initialization metadata differs: {key}")
        effective_sigma = float(handle.attrs.get("init_sigma_effective_first_step", -1))
        if not np.isfinite(effective_sigma) or abs(effective_sigma - 40.0) > 1.0e-4:
            raise ValueError("V18 effective first sigma is invalid")
        imaginary_ratio = float(
            handle.attrs.get("init_maximum_imaginary_over_real_rms", np.inf)
        )
        if imaginary_ratio > float(init["maximum_imaginary_over_real_rms"]):
            raise ValueError("V18 ensemble exceeded imaginary-leakage tolerance")
        commit = str(handle.attrs.get("sampling_code_commit", ""))
        if len(commit) != 40:
            raise ValueError("V18 ensemble has no valid sampling commit")
    return {
        "seed": seed,
        "effective_sigma_first_step": effective_sigma,
        "maximum_imaginary_over_real_rms": imaginary_ratio,
        "sampling_code_commit": commit,
        "registry_path": str(registry_path.resolve()),
    }


def _e3_candidate(registry: dict[str, Any], step: int) -> dict[str, Any]:
    parent = registry["parent_evidence"]
    payload = Path(parent["e3_decision"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != parent["e3_decision_sha256"]:
        raise ValueError("stored E3 decision changed after registry validation")
    decision = json.loads(payload)
    if canonical_digest(decision) != parent["e3_decision_digest_sha256"]:
        raise ValueError("stored E3 decision digest changed after registry validation")
    return next(row for row in decision["candidates"] if int(row["step"]) == step)


def _validate_draw_pairing(
    registry: dict[str, Any], *, step: int, domain: str,
    e6_path: Path, expected_indices: list[int],
) -> dict[str, Any]:
    baseline = _e3_candidate(registry, step)["domains"][domain]
    baseline_path = Path(baseline["ensemble"]).resolve()
    baseline_sha256 = sha256_file(baseline_path)
    if baseline_sha256 != baseline["ensemble_sha256"]:
        raise ValueError("stored E3 baseline ensemble differs from its frozen decision")
    with h5py.File(baseline_path, "r") as handle:
        baseline_seed = int(handle.attrs["seed"])
        baseline_indices = [int(value) for value in handle["source_index"][:]]
    with h5py.File(e6_path, "r") as handle:
        e6_seed = int(handle.attrs["seed"])
    registry_seed = int(
        registry["e6_prior_matched_initialization"]["sampling_seeds"][
            REGISTRY_DOMAINS[domain]
        ]
    )
    if baseline_seed != registry_seed or e6_seed != registry_seed:
        raise ValueError("V18 ensemble is not seed-paired to stored E3")
    if baseline_indices != expected_indices:
        raise ValueError("stored E3 baseline subset differs from V18")
    return {
        "paired": True, "seed": registry_seed,
        "e3_ensemble": str(baseline_path),
        "e3_ensemble_sha256": baseline_sha256,
    }


def _remeasure_variance(registry: dict[str, Any]) -> dict[str, Any]:
    experiment = registry["e6_prior_matched_initialization"]
    measured = measure_band_mode_variances(
        experiment["training_cache_sources"],
        parseval_relative_tolerance=float(
            experiment["initialization"]["parseval_relative_tolerance"]
        ),
        maximum_absolute_ortho_dc=float(
            experiment["initialization"]["maximum_absolute_ortho_dc"]
        ),
    )
    expected = np.asarray(
        experiment["initialization"]["source_balanced_per_band_mode_variance"],
        dtype=np.float64,
    )
    actual = np.asarray(measured["source_balanced"], dtype=np.float64)
    relative = np.max(np.abs(actual - expected) / expected)
    if relative > float(
        experiment["initialization"]["measurement_reverification_relative_tolerance"]
    ):
        raise ValueError("V18 train-cache variance remeasurement differs from registry")
    for domain, specification in experiment["training_cache_sources"].items():
        observed = np.asarray(measured["per_domain"][domain], dtype=np.float64)
        frozen = np.asarray(specification["per_band_mode_variance"], dtype=np.float64)
        if np.max(np.abs(observed - frozen) / frozen) > 1.0e-9:
            raise ValueError(f"V18 {domain} variance remeasurement mismatch")
    return {**measured, "maximum_relative_difference_from_registry": float(relative)}


def _rng_pairing_self_check(registry: dict[str, Any]) -> dict[str, Any]:
    experiment = registry["e6_prior_matched_initialization"]
    generator_a = torch.Generator(device="cpu").manual_seed(918003)
    generator_b = torch.Generator(device="cpu").manual_seed(918003)
    shape = (1, 1, 64, 64, 64)
    noise_a = torch.randn(shape, generator=generator_a)
    noise_b = torch.randn(shape, generator=generator_b)
    if not torch.equal(noise_a, noise_b):
        raise RuntimeError("V18 RNG pairing probe draws differ")
    state = generator_a.get_state().clone()
    std = prior_matched_spectral_std(
        64, 0.3125, 40.0,
        experiment["initialization"]["source_balanced_per_band_mode_variance"],
        device=torch.device("cpu"),
    )
    initializer = PriorMatchedInitializer(
        std,
        maximum_imaginary_ratio=float(
            experiment["initialization"]["maximum_imaginary_over_real_rms"]
        ),
    )
    initializer(noise_a)
    passed = torch.equal(generator_a.get_state(), state) and torch.equal(
        generator_a.get_state(), generator_b.get_state()
    )
    if not passed:
        raise RuntimeError("V18 initialization changed RNG state")
    return {
        "passed": True, "probe_seed": 918003, "additional_rng_draws": 0,
        "maximum_imaginary_over_real_rms": (
            initializer.maximum_observed_imaginary_ratio
        ),
    }


def _only_swift_peak_void(final: dict[str, Any]) -> bool:
    domains = final["domains"]
    swift_checks = domains["swift_dev"]["field_gate"]["checks"]
    return bool(
        domains["tng"]["field_gate"]["pass"]
        and domains["simba_dev"]["field_gate"]["pass"]
        and not swift_checks["all_selected_peak_void_statistics_improve_deterministic"]
        and all(
            value for name, value in swift_checks.items()
            if name != "all_selected_peak_void_statistics_improve_deterministic"
        )
    )


def _failure_classification(final: dict[str, Any]) -> dict[str, Any]:
    p1a = all(
        row["report_only_residual_power_over_truth"]["P1a_delta_at_least_0.15"]
        for row in final["domains"].values()
    )
    p1b = all(
        row["report_only_residual_power_over_truth"]["P1b_ratio_at_least_0.75"]
        for row in final["domains"].values()
    )
    two_point = final["domains"]["tng"]["two_point_0_1_mpc_h"]
    adjudication = bool(p1a and p1b and 0.17 <= two_point["generated"] <= 0.22)
    if not p1a:
        label = "P1a_failed_discard_initialization_hypothesis"
    elif not p1b:
        label = "P1a_passed_P1b_failed_audit_score_sigma_coverage"
    else:
        label = "P1a_P1b_passed_audit_remaining_conditional_structure"
    return {
        "P1a_all_domains": p1a,
        "P1b_all_domains": p1b,
        "class": label,
        "pooled_ks_adjudication_signature_triggered": adjudication,
    }


def evaluate(
    *, root: Path, registry_path: Path, repo: Path,
    gate_code_commit: str, astrid_root: Path = DEFAULT_ASTRID,
) -> dict[str, Any]:
    repo = repo.resolve()
    if astrid_files(astrid_root.resolve()):
        raise RuntimeError("Astrid must remain unopened during the V18 development gate")
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e6_prior_matched_initialization"]
    variance = _remeasure_variance(registry)
    rng_pairing = _rng_pairing_self_check(registry)
    steps = [int(value) for value in experiment["candidate_steps"]]
    expected_indices = {
        domain: _expected_indices(
            experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo
        ) for domain in DOMAINS
    }
    candidates = []
    for step in steps:
        checkpoint_path, checkpoint = _validate_checkpoint(registry, step)
        baseline_power = experiment["e3_baseline_residual_power_over_truth"][str(step)]
        domains: dict[str, Any] = {}
        for domain in DOMAINS:
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            init_metadata = _validate_e6_ensemble(
                ensemble_path, registry=registry, registry_path=registry_path,
                domain=domain, step=step, checkpoint_path=checkpoint_path,
                expected_indices=expected_indices[domain],
            )
            if init_metadata["sampling_code_commit"] != gate_code_commit:
                raise ValueError(
                    "V18 ensemble sampling commit differs from the gate commit"
                )
            pairing = _validate_draw_pairing(
                registry, step=step, domain=domain, e6_path=ensemble_path,
                expected_indices=expected_indices[domain],
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V18 metrics refer to a different ensemble")
            power = metrics["fourier_log_density"]["generated_total_power_over_truth"]
            residual_power = metrics["fourier_log_density"][
                "generated_residual_power_over_truth_residual"
            ]
            registry_domain = REGISTRY_DOMAINS[domain]
            baseline = baseline_power[registry_domain]
            band0 = float(residual_power[POWER_BANDS[0]])
            delta = band0 - float(baseline[0])
            two_point = metrics["two_point_cosmic_mean"]
            two_generated = float(
                two_point["generated_vs_truth_ks"]["by_scale"]["0-1_mpc_h"]["mean"]
            )
            two_deterministic = float(
                two_point["deterministic_vs_truth_ks"]["by_scale"]["0-1_mpc_h"]["mean"]
            )
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()),
                "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "draw_pairing_to_e3": pairing,
                "initialization_metadata": init_metadata,
                "report_only_total_power_over_truth": {
                    band: float(power[band]) for band in POWER_BANDS
                },
                "report_only_residual_power_over_truth": {
                    "bands": {band: float(residual_power[band]) for band in POWER_BANDS},
                    "e3_same_step_bands": {
                        band: float(value) for band, value in zip(
                            POWER_BANDS, baseline, strict=True
                        )
                    },
                    "band0_delta_from_e3": delta,
                    "P1a_delta_at_least_0.15": delta >= 0.15,
                    "P1b_ratio_at_least_0.75": band0 >= 0.75,
                    "selection_role": "none",
                },
                "two_point_0_1_mpc_h": {
                    "generated": two_generated,
                    "deterministic": two_deterministic,
                    "P2_improves_deterministic": two_generated < two_deterministic,
                },
            }
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _checkpoint_row(registry, step)["sha256"],
            "checkpoint_training_code_commit": checkpoint["code_commit_at_launch"],
            "domains": domains,
            "all_three_pass": all(row["field_gate"]["pass"] for row in domains.values()),
        })
    selected = select_candidate_rows(candidates)
    classification = None if selected is not None else _failure_classification(candidates[-1])
    if selected is not None:
        next_step = "freeze_exact_v18_hashes_before_astrid_one_shot"
    elif _only_swift_peak_void(candidates[-1]):
        next_step = "stop_unopened_then_predeclare_all_band_target_free_conditional_scale"
    else:
        next_step = "stop_unopened_and_obtain_new_independent_design_audit"
    report = {
        "schema": SCHEMA,
        "experiment": "e6_prior_matched_initialization",
        "sampling_only": True,
        "registry": str(registry_path.resolve()),
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "gate_code_commit": gate_code_commit,
        "worktree_clean_at_gate": True,
        "EAGLE_RefL0100N1504_used": False,
        "Astrid_used": False,
        "initialization_variance_remeasurement": variance,
        "rng_pairing_self_check": rng_pairing,
        "predeclared_steps": steps,
        "selection_rule": "smallest predeclared step passing all unchanged V6 field checks independently in TNG100, SIMBA development, and Swift development",
        "report_only_diagnostics_selection_role": "none",
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
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--astrid-root", type=Path, default=DEFAULT_ASTRID)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V18 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root, registry_path=args.registry, repo=args.repo,
        gate_code_commit=commit, astrid_root=args.astrid_root,
    )
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if canonical_digest(existing) != existing.get("decision_digest_sha256"):
            raise RuntimeError("existing V18 decision has an invalid digest")
        if existing != report:
            raise RuntimeError("existing V18 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
