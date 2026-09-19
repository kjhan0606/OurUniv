#!/usr/bin/env python
"""Audit one global conditional-null gate for the compatible 32-query design."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np

import hong2021_v73_gate_attainability as v73
import hong2021_v77_fresh32_partition_compatibility as v77
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v74_gate_redesign import probability_row


PROGRAM_SCHEMA = "hong2021-v78-global-conditional-null-redesign-audit-program-v1"
PROGRAM_STATUS = "frozen_before_V78_implementation_or_independent_holdout_bootstrap_draws"
PROGRAM_SHA256 = "ef6bc9f66065412467ce1b8cfd6d726857b8456179eb219f6be141326e0ed82a"
PROGRAM_FREEZE_COMMIT = "78f70aa59326db286f853b295a613409d3f9fedd"
RESULT_SCHEMA = "hong2021-v78-global-conditional-null-redesign-audit-result-v1"
DOMAIN_ORDER = v73.DOMAIN_ORDER
PE_ALPHA = 0.025
GLOBAL_ALPHA = 0.05
FAMILY_ORDER = (
    "q99_999",
    "Q4",
    "high_k_power",
    "residual_RMS",
    "density_PDF",
    "two_point",
    "environment",
    "energy",
)
CONTINUOUS_FAMILIES = FAMILY_ORDER[:6]
FAMILY_KEYS = {
    "q99_999": tuple((domain, "q_delta") for domain in DOMAIN_ORDER),
    "Q4": tuple((domain, "q4_ratio") for domain in DOMAIN_ORDER),
    "high_k_power": tuple(
        (domain, key)
        for domain in DOMAIN_ORDER
        for key in ("power_3_6", "power_6_10")
    ),
    "residual_RMS": tuple((domain, "rms_ratio") for domain in DOMAIN_ORDER),
    "density_PDF": tuple((domain, "oracle_pdf_tv") for domain in DOMAIN_ORDER),
    "two_point": tuple(
        (domain, key)
        for domain in DOMAIN_ORDER
        for key in ("oracle_ks_0_1", "oracle_ks_1_3", "oracle_ks_3_10")
    ),
}
LOG_FAMILIES = {"Q4", "high_k_power", "residual_RMS"}
POWER_ALTERNATIVES = (
    "coherent_q99_999_bias",
    "coherent_Q4_excess",
    "coherent_high_k_excess",
    "coherent_RMS_excess",
    "coherent_PDF_TV_excess",
    "clear_two_point_KS_excess",
    "environment_failure_all_domains",
    "coherent_energy_inferiority",
    "diffuse_multimetric_shift",
)


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = path.resolve()
    repo = repo.resolve()
    if sha256_file(path) != PROGRAM_SHA256:
        raise ValueError("V78 program hash differs")
    program = strict_json(path)
    limits = program.get("scope_limits", {})
    rule = program.get("complete_global_rule", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or limits.get("this_is_not_a_generator") is not True
        or limits.get("validation_input_or_target_payload_access") is not False
        or limits.get("raw_fit_train_truth_reread") is not False
        or float(rule.get("global_alpha", -1)) != GLOBAL_ALPHA
    ):
        raise ValueError("V78 schema, firewall, or alpha differs")
    parent = program["parent_evidence"]
    local = ("V77_result_record", "V76_result_record")
    gpfs = (
        "V77_audit_result",
        "V77_calibration_arrays",
        "V77_verification_arrays",
        "V76_null_p_arrays",
        "V73_summary_record",
        "V73_summary_cache",
    )
    paths = {key: resolve_path(repo, parent[key]) for key in local}
    paths.update({key: Path(parent[key]).resolve() for key in gpfs})
    for key, bound in paths.items():
        if sha256_file(bound) != parent[f"{key}_sha256"]:
            raise ValueError(f"V78 parent differs: {key}")
    v77_record = strict_json(paths["V77_result_record"])
    if (
        v77_record["decision"]["classification"]
        != "fresh32_partition_or_gate_requires_additional_redesign"
        or v77_record["authorization"]["freeze_or_execute_a_complete_gate"]
        is not False
    ):
        raise ValueError("V78 V77 boundary differs")
    summary = strict_json(paths["V73_summary_record"])
    if summary.get("summary_cache_sha256") != parent["V73_summary_cache_sha256"]:
        raise ValueError("V78 summary provenance differs")
    return program, paths


def upper_tail_p(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("V78 p-value input is nonfinite")
    ordered = np.sort(reference)
    exceed = len(ordered) - np.searchsorted(ordered, values, side="left")
    return (1.0 + exceed) / (len(ordered) + 1.0)


def global_p_value(pe_p: np.ndarray, rank_coverage_p: np.ndarray) -> np.ndarray:
    pe_p = np.asarray(pe_p, dtype=np.float64)
    rank_coverage_p = np.asarray(rank_coverage_p, dtype=np.float64)
    return np.minimum(1.0, 2.0 * np.minimum(pe_p, rank_coverage_p))


def _array(phase: dict[str, np.ndarray], domain: str, key: str) -> np.ndarray:
    return np.asarray(phase[f"{domain}__{key}"])


def family_matrix(
    phase: dict[str, np.ndarray], family: str, alternative: str | None = None
) -> np.ndarray:
    values = np.stack(
        [_array(phase, domain, key) for domain, key in FAMILY_KEYS[family]], axis=1
    ).astype(np.float64)
    if family in LOG_FAMILIES:
        if np.any(values <= 0):
            raise ValueError("V78 logarithmic family has nonpositive value")
        values = np.log(values)
    shift = 0.0
    if alternative == "coherent_q99_999_bias" and family == "q99_999":
        shift = 0.15
    elif alternative == "coherent_Q4_excess" and family == "Q4":
        shift = np.log(1.5)
    elif alternative == "coherent_high_k_excess" and family == "high_k_power":
        shift = np.log(1.1)
    elif alternative == "coherent_RMS_excess" and family == "residual_RMS":
        shift = np.log(1.1)
    elif alternative == "coherent_PDF_TV_excess" and family == "density_PDF":
        shift = 0.03
    elif alternative == "clear_two_point_KS_excess" and family == "two_point":
        shift = 0.20
    elif alternative == "diffuse_multimetric_shift":
        shift = {
            "q99_999": 0.075,
            "Q4": 0.5 * np.log(1.5),
            "high_k_power": 0.5 * np.log(1.1),
            "residual_RMS": 0.5 * np.log(1.1),
            "density_PDF": 0.015,
            "two_point": 0.05,
        }[family]
    return values + shift


def mahalanobis_model(values: np.ndarray) -> dict[str, np.ndarray | float]:
    values = np.asarray(values, dtype=np.float64)
    center = values.mean(axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    ridge = max(1e-12, 1e-6 * float(np.trace(covariance)) / values.shape[1])
    inverse = np.linalg.pinv(covariance + np.eye(values.shape[1]) * ridge)
    return {"center": center, "inverse": inverse, "ridge": ridge}


def mahalanobis_score(values: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - model["center"]
    return np.einsum("ni,ij,nj->n", centered, model["inverse"], centered)


def fit_physical_energy(
    calibration: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, Any]:
    models = {}
    calibration_scores = {}
    for family in CONTINUOUS_FAMILIES:
        values = family_matrix(calibration, family)
        model = mahalanobis_model(values)
        models[family] = model
        calibration_scores[family] = mahalanobis_score(values, model)
    calibration_environment = np.stack(
        [1 - _array(calibration, domain, "environment") for domain in DOMAIN_ORDER],
        axis=1,
    ).sum(axis=1)
    calibration_energy = np.stack(
        [_array(calibration, domain, "energy_A_minus_B") for domain in DOMAIN_ORDER],
        axis=1,
    )
    energy_center = calibration_energy.mean(axis=0)
    energy_scale = calibration_energy.std(axis=0)
    if np.any(energy_scale <= 0):
        raise ValueError("V78 energy calibration scale differs")
    calibration_energy_score = (
        (calibration_energy - energy_center) / energy_scale
    ).max(axis=1)
    fit = {
        "models": models,
        "calibration_scores": calibration_scores,
        "calibration_environment": calibration_environment,
        "energy_center": energy_center,
        "energy_scale": energy_scale,
        "calibration_energy_score": calibration_energy_score,
    }
    reference_components = physical_energy_components(reference, fit)
    fit["reference_sparse"] = reference_components["sparse"]
    fit["reference_dense"] = reference_components["dense"]
    return fit


def physical_energy_components(
    phase: dict[str, np.ndarray],
    fit: dict[str, Any],
    alternative: str | None = None,
) -> dict[str, np.ndarray]:
    p_values = []
    for family in CONTINUOUS_FAMILIES:
        score = mahalanobis_score(
            family_matrix(phase, family, alternative), fit["models"][family]
        )
        p_values.append(upper_tail_p(fit["calibration_scores"][family], score))
    environment = np.stack(
        [1 - _array(phase, domain, "environment") for domain in DOMAIN_ORDER],
        axis=1,
    ).sum(axis=1)
    if alternative == "environment_failure_all_domains":
        environment[:] = len(DOMAIN_ORDER)
    p_values.append(upper_tail_p(fit["calibration_environment"], environment))
    energy = np.stack(
        [_array(phase, domain, "energy_A_minus_B") for domain in DOMAIN_ORDER],
        axis=1,
    ).astype(np.float64)
    if alternative == "coherent_energy_inferiority":
        energy += 0.03
    elif alternative == "diffuse_multimetric_shift":
        energy += 0.015
    energy_score = ((energy - fit["energy_center"]) / fit["energy_scale"]).max(
        axis=1
    )
    p_values.append(upper_tail_p(fit["calibration_energy_score"], energy_score))
    family_p = np.stack(p_values, axis=1)
    if family_p.shape[1] != len(FAMILY_ORDER):
        raise ValueError("V78 family count differs")
    sparse = (-np.log(family_p)).max(axis=1)
    dense = (-2.0 * np.log(family_p)).sum(axis=1)
    return {"family_p": family_p, "sparse": sparse, "dense": dense}


def physical_energy_p(
    phase: dict[str, np.ndarray], fit: dict[str, Any], alternative: str | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    components = physical_energy_components(phase, fit, alternative)
    sparse_p = upper_tail_p(fit["reference_sparse"], components["sparse"])
    dense_p = upper_tail_p(fit["reference_dense"], components["dense"])
    block_p = np.minimum(1.0, 2.0 * np.minimum(sparse_p, dense_p))
    components.update(
        {"sparse_p": sparse_p, "dense_p": dense_p, "block_p": block_p}
    )
    return block_p, components


def rank_coverage_block_samples(
    arrays: Any, scenario: str, trials: int, rng: np.random.Generator
) -> np.ndarray:
    rank = np.asarray(arrays[f"null__{scenario}__rank_tv_p"], dtype=np.float64)
    coverage = np.asarray(
        arrays[f"null__{scenario}__coverage_deviation_p"], dtype=np.float64
    )
    if rank.shape != coverage.shape or rank.shape != (100000,):
        raise ValueError("V78 V76 p-value array differs")
    indices = rng.integers(0, len(rank), size=(trials, len(DOMAIN_ORDER)))
    minimum = np.ones(trials, dtype=np.float64)
    for domain_index in range(len(DOMAIN_ORDER)):
        current = indices[:, domain_index]
        minimum = np.minimum(minimum, rank[current])
        minimum = np.minimum(minimum, coverage[current])
    return np.minimum(1.0, 6.0 * minimum)


def null_calibration(
    program: dict[str, Any], pe_p: np.ndarray, v76_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = program["null_calibration_audit"]
    pe_rejection = probability_row(pe_p <= PE_ALPHA)
    pe_calibrated = bool(
        pe_rejection["probability"] <= PE_ALPHA
        and pe_rejection["Wilson_95"][1] <= 0.03
    )
    result = {
        "physical_energy": {
            "rejection": pe_rejection,
            "alpha": PE_ALPHA,
            "calibrated": pe_calibrated,
            "p_quantiles_0_2p5_50_97p5_100": np.quantile(
                pe_p, [0, 0.025, 0.5, 0.975, 1]
            ).tolist(),
        },
        "complete_scenarios": {},
    }
    output_arrays = {"null__physical_energy_p": pe_p}
    rng = np.random.default_rng(int(spec["complete_pairing_seed"]))
    trials = int(spec["complete_trials_per_scenario"])
    with np.load(v76_path) as arrays:
        for scenario in spec["rank_coverage_scenarios"]:
            pe_index = rng.integers(0, len(pe_p), size=trials)
            rc_p = rank_coverage_block_samples(arrays, scenario, trials, rng)
            complete_p = global_p_value(pe_p[pe_index], rc_p)
            rejection = probability_row(complete_p <= GLOBAL_ALPHA)
            calibrated = bool(
                rejection["probability"] <= GLOBAL_ALPHA
                and rejection["Wilson_95"][1] <= 0.055
            )
            result["complete_scenarios"][scenario] = {
                "rejection": rejection,
                "calibrated": calibrated,
                "global_p_quantiles_0_2p5_50_97p5_100": np.quantile(
                    complete_p, [0, 0.025, 0.5, 0.975, 1]
                ).tolist(),
            }
            output_arrays[f"null__{scenario}__rank_coverage_block_p"] = rc_p
            output_arrays[f"null__{scenario}__complete_global_p"] = complete_p
    result["complete_calibrated"] = all(
        row["calibrated"] for row in result["complete_scenarios"].values()
    )
    result["physical_energy_calibrated"] = pe_calibrated
    result["mathematical_FWER_upper"] = GLOBAL_ALPHA
    return result, output_arrays


def power_audit(
    program: dict[str, Any], audit_phase: dict[str, np.ndarray], fit: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = program["frozen_power_audit"]
    result = {}
    arrays = {}
    for alternative in POWER_ALTERNATIVES:
        p_value, components = physical_energy_p(audit_phase, fit, alternative)
        detected = p_value <= PE_ALPHA
        row = probability_row(detected)
        sufficient = bool(
            row["probability"] >= float(spec["minimum_detection_probability"])
            and row["Wilson_95"][0] >= float(spec["minimum_Wilson_95_lower"])
        )
        result[alternative] = {
            "detection": row,
            "p_quantiles_2p5_50_97p5": np.quantile(
                p_value, [0.025, 0.5, 0.975]
            ).tolist(),
            "sparse_detection": probability_row(components["sparse_p"] <= PE_ALPHA / 2),
            "dense_detection": probability_row(components["dense_p"] <= PE_ALPHA / 2),
            "power_sufficient": sufficient,
        }
        arrays[f"power__{alternative}__physical_energy_p"] = p_value
    return {
        "alternatives": result,
        "power_sufficient": all(row["power_sufficient"] for row in result.values()),
    }, arrays


def arrays_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        output = {key: np.asarray(source[key]) for key in source.files}
    if any(not np.isfinite(value).all() for value in output.values()):
        raise ValueError("V78 parent array is nonfinite")
    return output


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def run(program_path: Path, repo: Path, output_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, parents = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V78 requires a clean frozen Lageunha checkout")
    paths = {
        key: Path(program["outputs"][key]).resolve()
        for key in ("root", "independent_audit_arrays", "p_value_arrays", "audit_result")
    }
    if output_root.resolve() != paths["root"]:
        raise ValueError("V78 output root differs")
    if output_root.exists() or any(paths[key].exists() for key in paths if key != "root"):
        raise FileExistsError("V78 refuses an existing output")
    output_root.mkdir(parents=True)
    calibration = arrays_dict(parents["V77_calibration_arrays"])
    reference = arrays_dict(parents["V77_verification_arrays"])
    fit = fit_physical_energy(calibration, reference)
    with np.load(parents["V73_summary_cache"], allow_pickle=False) as cache:
        summaries = {domain: v73._domain_summary(cache, domain) for domain in DOMAIN_ORDER}
        k = np.asarray(cache["fourier_k"], dtype=np.float64)
        count = np.asarray(cache["fourier_mode_count"], dtype=np.int64)
        radius = np.asarray(cache["radius_mpc_h"], dtype=np.float64)
    audit_spec = program["physical_energy_block"]["independent_audit_phase"]
    audit_nested = v77.run_phase(
        summaries,
        k,
        count,
        radius,
        int(audit_spec["trials_per_domain"]),
        int(audit_spec["seed"]),
        "independent-audit",
    )
    audit_phase = v77.flatten_phase(audit_nested)
    write_npz(paths["independent_audit_arrays"], audit_phase)
    pe_p, pe_components = physical_energy_p(audit_phase, fit)
    calibration_result, null_arrays = null_calibration(
        program, pe_p, parents["V76_null_p_arrays"]
    )
    power_result, power_arrays = power_audit(program, audit_phase, fit)
    p_arrays = {
        **null_arrays,
        **power_arrays,
        "audit__family_p": pe_components["family_p"],
        "audit__sparse_statistic": pe_components["sparse"],
        "audit__dense_statistic": pe_components["dense"],
        "audit__sparse_p": pe_components["sparse_p"],
        "audit__dense_p": pe_components["dense_p"],
    }
    write_npz(paths["p_value_arrays"], p_arrays)
    invariant = bool(
        np.array_equal(
            pe_components["sparse"],
            (-np.log(pe_components["family_p"])).max(axis=1),
        )
        and np.array_equal(
            pe_p,
            np.minimum(
                1.0,
                2.0 * np.minimum(pe_components["sparse_p"], pe_components["dense_p"]),
            ),
        )
    )
    selected = bool(
        calibration_result["physical_energy_calibrated"]
        and calibration_result["complete_calibrated"]
        and power_result["power_sufficient"]
        and invariant
    )
    classification = (
        "single_global_conditional_null_gate_selected_complete_specification_may_be_frozen"
        if selected
        else "global_conditional_null_gate_requires_additional_redesign"
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_V78_global_conditional_null_redesign_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "family_order": list(FAMILY_ORDER),
        "null_calibration": calibration_result,
        "power_audit": power_result,
        "implementation_invariants": {
            "sparse_parentheses_exact": invariant,
            "one_global_alpha": GLOBAL_ALPHA,
            "physical_energy_block_alpha": PE_ALPHA,
            "rank_coverage_block_alpha": PE_ALPHA,
            "rank_coverage_per_metric_threshold": 1.0 / 240.0,
        },
        "decision": {
            "global_rule_selected": selected,
            "classification": classification,
            "complete_candidate_agnostic_specification_may_be_frozen": selected,
            "candidate_or_fresh_payload_execution_authorized": False,
            "next": (
                "freeze_complete_candidate_agnostic_V79_gate_then_await_explicit_approval"
                if selected
                else "stop_before_complete_gate_or_candidate_and_report_failed_requirement"
            ),
        },
        "artifacts": {
            "independent_audit_arrays": str(paths["independent_audit_arrays"]),
            "independent_audit_arrays_sha256": sha256_file(paths["independent_audit_arrays"]),
            "p_value_arrays": str(paths["p_value_arrays"]),
            "p_value_arrays_sha256": sha256_file(paths["p_value_arrays"]),
        },
        "validation_input_or_target_payload_accessed": False,
        "training_or_model_sampling_performed": False,
        "raw_fit_train_truth_reread": False,
        "V72_stage_B_accessed": False,
        "Astrid_accessed": False,
        "historical_or_independent_EAGLE_accessed": False,
        "new_candidate_authorized": False,
        "V72_verdict_changed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    partial = paths["audit_result"].with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, paths["audit_result"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.program, args.repo, args.output_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
