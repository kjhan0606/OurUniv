#!/usr/bin/env python3
"""Corrected aggregation for the 64 grouped-CF4 population truth mocks.

The N32 density domain contains three self-conjugate Nyquist-axis modes in
merged bin 11.  The frozen theta definition deliberately excludes every
Nyquist plane, so theta has no bin-11 mode.  This aggregator evaluates each
domain on its actual mode support and maps absent theta bin 11 to fail-closed
when forming the joint contiguous frontier.  No member, seed, threshold,
population model, posterior, or untouched validation mock is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_population_calibration as base
from cf4_kf_bin_manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA = "ouruniv-cf4-bgc-population-calibration-result-v2"
RESULT_STATUS = "COMPLETE_64_MOCK_DEVELOPMENT_CALIBRATION_V2_NO_SCIENCE_CLAIM"
EXPECTED_FILES = {"metrics.npz", "result.json", "manifest.json", "COMPLETE"}
EXPECTED_DELTA_BIN_IDS = np.arange(12, dtype=np.int64)
EXPECTED_THETA_BIN_IDS = np.arange(11, dtype=np.int64)


def expand_gate_to_union(
    domain_bin_ids: np.ndarray, domain_gate: np.ndarray, union_bin_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map a domain gate to the union; unavailable bins remain false."""

    domain_bin_ids = np.asarray(domain_bin_ids)
    domain_gate = np.asarray(domain_gate)
    union_bin_ids = np.asarray(union_bin_ids)
    if (
        domain_bin_ids.ndim != 1
        or union_bin_ids.ndim != 1
        or domain_gate.shape != domain_bin_ids.shape
        or domain_gate.dtype != np.dtype(bool)
        or np.any(np.diff(domain_bin_ids) <= 0)
        or np.any(np.diff(union_bin_ids) <= 0)
        or np.any(~np.isin(domain_bin_ids, union_bin_ids))
    ):
        raise base.CalibrationError("domain-to-union gate mapping is invalid")
    available = np.isin(union_bin_ids, domain_bin_ids)
    expanded = np.zeros(union_bin_ids.size, dtype=bool)
    lookup = {int(value): bool(gate) for value, gate in zip(domain_bin_ids, domain_gate)}
    expanded[available] = [lookup[int(value)] for value in union_bin_ids[available]]
    return expanded, available


def _load_members(
    program_sha256: str, members_root: str | Path, member_implementation_commit: str
) -> tuple[
    dict[str, dict[str, list[np.ndarray]]],
    dict[str, dict[str, np.ndarray]],
    list[list[dict[str, object]]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    root = Path(members_root)
    if not root.is_dir():
        raise base.CalibrationError("member root is absent")
    expected_names = {f"member-{index:02d}" for index in range(base.MOCK_COUNT)}
    if {path.name for path in root.iterdir()} != expected_names:
        raise base.CalibrationError("member directory set is not exact")
    domain_store = {
        "delta": {"truth": [], "mean": [], "draws": []},
        "theta": {"truth": [], "mean": [], "draws": []},
    }
    metadata: dict[str, dict[str, np.ndarray]] = {}
    heldout_rows = []
    fidelity_rows = []
    member_hashes = []
    for index in range(base.MOCK_COUNT):
        member = root / f"member-{index:02d}"
        validation = base.validate_member(member, expected_index=index)
        result = json.loads((member / "result.json").read_bytes())
        if result["program_sha256"] != program_sha256:
            raise base.CalibrationError("member program binding mismatch")
        if result["implementation_commit"] != member_implementation_commit:
            raise base.CalibrationError("member implementation commit mismatch")
        member_hashes.append(validation)
        fidelity_rows.append(result["catalog_fidelity"])
        heldout_rows.append(result["heldout_cumulative_prediction"])
        with np.load(member / "fields.npz", allow_pickle=False) as fields:
            for domain in ("delta", "theta"):
                domain_store[domain]["truth"].append(np.array(fields[f"truth_{domain}_modes"]))
                domain_store[domain]["mean"].append(
                    np.array(fields[f"posterior_mean_{domain}_modes"])
                )
                domain_store[domain]["draws"].append(
                    np.array(fields[f"posterior_draws_{domain}_modes"])
                )
                current = {
                    "assignment": np.array(fields[f"{domain}_mode_bin_index"]),
                    "prior_variance": np.array(fields[f"{domain}_prior_variance"]),
                    "self_conjugate": np.array(fields[f"{domain}_self_conjugate"]),
                }
                if domain not in metadata:
                    metadata[domain] = current
                else:
                    for key, value in current.items():
                        if not np.array_equal(value, metadata[domain][key]):
                            raise base.CalibrationError(f"{domain} member metadata changed")
    return domain_store, metadata, heldout_rows, fidelity_rows, member_hashes


def aggregate_members(
    program: Mapping[str, object],
    program_sha256: str,
    members_root: str | Path,
    member_implementation_commit: str,
    aggregation_runtime_commit: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    for label, commit in (
        ("member implementation", member_implementation_commit),
        ("aggregation runtime", aggregation_runtime_commit),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise base.CalibrationError(f"{label} commit must be lowercase 40-hex")
    domain_store, metadata, heldout_rows, fidelity_rows, member_hashes = _load_members(
        program_sha256, members_root, member_implementation_commit
    )
    domain_bins = {
        domain: np.unique(metadata[domain]["assignment"])
        for domain in ("delta", "theta")
    }
    if not np.array_equal(domain_bins["delta"], EXPECTED_DELTA_BIN_IDS):
        raise base.CalibrationError("corrected aggregate density bin support changed")
    if not np.array_equal(domain_bins["theta"], EXPECTED_THETA_BIN_IDS):
        raise base.CalibrationError("corrected aggregate theta bin support changed")
    if (
        np.count_nonzero(metadata["delta"]["assignment"] == 11) != 3
        or np.any(metadata["theta"]["assignment"] == 11)
    ):
        raise base.CalibrationError("frozen three-mode Nyquist-axis support changed")
    union_bins = np.union1d(domain_bins["delta"], domain_bins["theta"])

    heldout = np.empty((base.MOCK_COUNT, union_bins.size), dtype=np.float64)
    for mock, rows in enumerate(heldout_rows):
        row_map = {int(row["merged_bin_index"]): row for row in rows}
        if set(row_map) != set(union_bins.tolist()):
            raise base.CalibrationError("heldout cumulative bin set changed")
        heldout[mock] = [
            row_map[int(bin_id)]["per_row_improvement"] for bin_id in union_bins
        ]
    gates = program["aggregate_gates"]
    bootstrap_indices = base._bootstrap_indices(
        base.MOCK_COUNT,
        int(gates["mock_cluster_bootstrap_replicates"]),
        base.BOOTSTRAP_SEED,
    )
    heldout_point, heldout_lower, heldout_upper = base._bootstrap_interval(
        heldout, bootstrap_indices, statistic="mean"
    )
    heldout_union_pass = heldout_lower > float(
        gates["heldout_per_row_improvement_lower_min_exclusive"]
    )

    all_metrics: dict[str, object] = {}
    all_arrays: dict[str, np.ndarray] = {
        "bin_ids": union_bins,
        "delta_bin_ids": domain_bins["delta"],
        "theta_bin_ids": domain_bins["theta"],
        "bootstrap_mock_indices": bootstrap_indices,
        "heldout_per_mock_per_row_improvement": heldout,
        "heldout_mean_per_row_improvement": heldout_point,
        "heldout_bootstrap_lower_2p5": heldout_lower,
        "heldout_bootstrap_upper_97p5": heldout_upper,
        "heldout_pass": heldout_union_pass,
    }
    expanded_strict = {}
    availability = {}
    for offset, domain in enumerate(("delta", "theta")):
        bins = domain_bins[domain]
        heldout_pass = heldout_union_pass[np.isin(union_bins, bins)]
        store = domain_store[domain]
        metrics, arrays = base.compute_domain_calibration(
            domain_id=(
                "global_z0_density_delta"
                if domain == "delta"
                else "global_discrete_normalized_velocity_divergence_theta"
            ),
            truth=np.stack(store["truth"]),
            mean=np.stack(store["mean"]),
            draws=np.stack(store["draws"]),
            prior_variance=metadata[domain]["prior_variance"],
            assignment=metadata[domain]["assignment"],
            self_conjugate=metadata[domain]["self_conjugate"],
            bin_ids=bins,
            heldout_pass=heldout_pass,
            bootstrap_indices=bootstrap_indices,
            gates=gates,
            phase_seed=base.PHASE_NULL_SEED + offset,
        )
        metrics["available_merged_bin_ids"] = bins.tolist()
        all_metrics[domain] = metrics
        all_arrays.update({f"{domain}_{key}": value for key, value in arrays.items()})
        expanded, available = expand_gate_to_union(
            bins, np.asarray(metrics["strict_gate"], dtype=bool), union_bins
        )
        expanded_strict[domain] = expanded
        availability[domain] = available
        all_arrays[f"{domain}_available_on_union"] = available
        all_arrays[f"{domain}_strict_gate_on_union"] = expanded

    manifest_path = ROOT / program["inputs"]["bin_manifest"]["path"]
    manifest_body, manifest_body_sha, _ = base.fixed.load_bin_manifest(manifest_path)
    upper_edges = base._merged_upper_edges(manifest_body)
    upper_k = np.asarray([upper_edges[int(bin_id)] for bin_id in union_bins])
    all_arrays["upper_k_h_Mpc"] = upper_k
    field_frontier = base.frontier.evaluate_field_frontiers(
        upper_k, expanded_strict["delta"], expanded_strict["theta"]
    )

    fidelity_names = (
        "BGc_selected_group_count",
        "observed_distance_KS",
        "distance_error_mag_KS",
        "redshift_velocity_KS",
        "angular_histogram_total_variation",
    )
    fidelity_arrays = {
        name: np.asarray([row[name] for row in fidelity_rows]) for name in fidelity_names
    }
    all_arrays.update({f"fidelity_{key}": value for key, value in fidelity_arrays.items()})
    fidelity_member_pass = np.asarray(
        [
            base.population_fidelity_gates(row, program["population_fidelity_gates"])[
                "all_pass"
            ]
            for row in fidelity_rows
        ],
        dtype=bool,
    )
    all_arrays["fidelity_member_pass"] = fidelity_member_pass
    generator_fidelity = {
        "all_64_members_pass": bool(np.all(fidelity_member_pass)),
        "passing_member_count": int(np.count_nonzero(fidelity_member_pass)),
        "BGc_selected_group_count_range": [
            int(np.min(fidelity_arrays["BGc_selected_group_count"])),
            int(np.max(fidelity_arrays["BGc_selected_group_count"])),
        ],
        "metric_maxima": {
            name: float(np.max(fidelity_arrays[name]))
            for name in fidelity_names
            if name != "BGc_selected_group_count"
        },
    }
    joint_strict = expanded_strict["delta"] & expanded_strict["theta"]
    all_strict = bool(
        generator_fidelity["all_64_members_pass"] and np.all(joint_strict)
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "program_sha256": program_sha256,
        "member_implementation_commit": member_implementation_commit,
        "aggregation_runtime_commit": aggregation_runtime_commit,
        "aggregation_correction_source_sha256": base.sha256_file(__file__),
        "frozen_member_source_sha256": base.sha256_file(base.__file__),
        "member_count": base.MOCK_COUNT,
        "posterior_draw_count": base.POSTERIOR_DRAW_COUNT,
        "member_artifact_hashes": member_hashes,
        "bin_manifest_body_sha256": manifest_body_sha,
        "union_merged_bin_ids": union_bins.tolist(),
        "domain_available_merged_bin_ids": {
            "delta": domain_bins["delta"].tolist(),
            "theta": domain_bins["theta"].tolist(),
        },
        "theta_bin_11_disposition": {
            "density_mode_count": 3,
            "theta_mode_count": 0,
            "cause": "all_three_density_modes_are_self_conjugate_Nyquist_axis_modes_excluded_by_frozen_theta_definition",
            "joint_gate": False,
            "policy": "unavailable_domain_mode_support_fails_closed_not_omitted",
        },
        "cumulative_upper_k_h_Mpc": upper_k.tolist(),
        "population_generator_fidelity": generator_fidelity,
        "heldout_cumulative_prediction": {
            "mean_per_row_improvement": heldout_point.tolist(),
            "bootstrap_95_interval": np.column_stack(
                (heldout_lower, heldout_upper)
            ).tolist(),
            "pass": heldout_union_pass.tolist(),
        },
        "domain_metrics": all_metrics,
        "development_strict_frontier_diagnostic": {
            "density_delta": base._frontier_payload(field_frontier.density_delta),
            "velocity_divergence_theta": base._frontier_payload(
                field_frontier.velocity_divergence_theta
            ),
            "joint": base._frontier_payload(field_frontier.joint),
            "all_union_bins_and_generator_fidelity_pass": all_strict,
            "semantics": "development_diagnostic_only_not_a_promoted_constraint_frontier",
        },
        "selection_semantics": "empirical_grouped_CF4_selection_conditioned_on_clean_group_count",
        "full_survey_selection_normalization_modeled": False,
        "observed_vpec_or_vobs_used": False,
        "observed_v3k_used_for_generation": False,
        "development_only": True,
        "untouched_256_mock_validation_executed": False,
        "frontier_or_science_claim_allowed": False,
        "target_0p3_cMpc_h_claim_allowed": False,
        "next_action_requires_user_approval": True,
    }
    return result, all_arrays


def _expected_array_names() -> set[str]:
    names = {
        "bin_ids",
        "delta_bin_ids",
        "theta_bin_ids",
        "upper_k_h_Mpc",
        "bootstrap_mock_indices",
        "heldout_per_mock_per_row_improvement",
        "heldout_mean_per_row_improvement",
        "heldout_bootstrap_lower_2p5",
        "heldout_bootstrap_upper_97p5",
        "heldout_pass",
        "fidelity_BGc_selected_group_count",
        "fidelity_observed_distance_KS",
        "fidelity_distance_error_mag_KS",
        "fidelity_redshift_velocity_KS",
        "fidelity_angular_histogram_total_variation",
        "fidelity_member_pass",
    }
    suffixes = {
        "response",
        "correlation_r",
        "residual_power_ratio",
        "per_mock_variance_ratio_median",
        "variance_ratio_median",
        "variance_bootstrap_lower_2p5",
        "variance_bootstrap_upper_97p5",
        "phase_null_p_value",
        "phase_null_cross",
        "per_mock_coverage68",
        "coverage68",
        "coverage68_bootstrap_lower_2p5",
        "coverage68_bootstrap_upper_97p5",
        "per_mock_coverage95",
        "coverage95",
        "coverage95_bootstrap_lower_2p5",
        "coverage95_bootstrap_upper_97p5",
        "strict_gate",
        "available_on_union",
        "strict_gate_on_union",
    }
    names |= {
        f"{domain}_{suffix}"
        for domain in ("delta", "theta")
        for suffix in suffixes
    }
    return names


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise base.CalibrationError("corrected aggregate artifact file set is not exact")
    result_payload = (root / "result.json").read_bytes()
    metrics_payload = (root / "metrics.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise base.CalibrationError("corrected aggregate result is not canonical JSON")
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != RESULT_STATUS:
        raise base.CalibrationError("corrected aggregate result schema/status mismatch")
    if result.get("member_count") != base.MOCK_COUNT:
        raise base.CalibrationError("corrected aggregate member count mismatch")
    if result.get("domain_available_merged_bin_ids") != {
        "delta": EXPECTED_DELTA_BIN_IDS.tolist(),
        "theta": EXPECTED_THETA_BIN_IDS.tolist(),
    }:
        raise base.CalibrationError("corrected aggregate domain support mismatch")
    disposition = result.get("theta_bin_11_disposition", {})
    if disposition.get("theta_mode_count") != 0 or disposition.get("joint_gate") is not False:
        raise base.CalibrationError("absent theta bin 11 did not fail closed")
    if result.get("untouched_256_mock_validation_executed") is not False or result.get(
        "frontier_or_science_claim_allowed"
    ) is not False:
        raise base.CalibrationError("corrected aggregate crosses the science firewall")
    expected_payloads = {
        "metrics.npz": {
            "sha256": hashlib.sha256(metrics_payload).hexdigest(),
            "bytes": len(metrics_payload),
        },
        "result.json": {
            "sha256": hashlib.sha256(result_payload).hexdigest(),
            "bytes": len(result_payload),
        },
    }
    if manifest.get("payloads") != expected_payloads:
        raise base.CalibrationError("corrected aggregate payload binding mismatch")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest() or complete.get(
        "COMPLETE_written_last"
    ) is not True:
        raise base.CalibrationError("corrected aggregate COMPLETE binding mismatch")
    with np.load(io.BytesIO(metrics_payload), allow_pickle=False) as metrics:
        if set(metrics.files) != _expected_array_names():
            raise base.CalibrationError("corrected aggregate metric array set is not exact")
        if not np.array_equal(metrics["delta_bin_ids"], EXPECTED_DELTA_BIN_IDS) or not np.array_equal(
            metrics["theta_bin_ids"], EXPECTED_THETA_BIN_IDS
        ):
            raise base.CalibrationError("corrected aggregate metric domain bins changed")
        if metrics["theta_available_on_union"].tolist() != [True] * 11 + [False]:
            raise base.CalibrationError("theta union availability is not fail-closed")
        if bool(metrics["theta_strict_gate_on_union"][-1]):
            raise base.CalibrationError("absent theta terminal bin passed")
        for name in metrics.files:
            if not np.all(np.isfinite(metrics[name])):
                raise base.CalibrationError(f"corrected aggregate metric is nonfinite: {name}")
    return {
        "status": "PASS",
        "member_count": base.MOCK_COUNT,
        "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--program", required=True, type=Path)
    aggregate.add_argument("--members-root", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--member-implementation-commit", required=True)
    aggregate.add_argument("--aggregation-runtime-commit", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "aggregate":
            program, program_sha = base.load_program(args.program)
            result, arrays = aggregate_members(
                program,
                program_sha,
                args.members_root,
                args.member_implementation_commit,
                args.aggregation_runtime_commit,
            )
            base.publish_directory(args.output, result, arrays, kind="aggregate")
            report = validate_aggregate(args.output)
        else:
            report = validate_aggregate(args.directory)
    except (OSError, ValueError, base.CalibrationError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
