#!/usr/bin/env python3
"""Consumed-only V8 Local-Group support, weight, margin, and lineage autopsy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from cf4_p2_screen import load_config


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays without altering their numerical value."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def pair_key(pair: dict[str, Any]) -> tuple[int, int]:
    return tuple(sorted((int(pair["halo_i"]), int(pair["halo_j"]))))


def logmeanexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.exp(array - maximum).mean()))


def weight_summary(log_weights: list[float]) -> dict[str, Any]:
    values = np.asarray(log_weights, dtype=np.float64)
    finite = np.isfinite(values)
    weights = np.zeros(values.size, dtype=np.float64)
    if np.any(finite):
        selected = values[finite]
        maximum = float(np.max(selected))
        normalizer = maximum + math.log(float(np.exp(selected - maximum).sum()))
        weights[finite] = np.exp(selected - normalizer)
    ess = float(1.0 / np.sum(weights**2)) if np.any(weights) else 0.0
    return {
        "n_total": int(values.size),
        "n_nonzero": int(np.count_nonzero(finite)),
        "effective_sample_size": ess,
        "maximum_normalized_weight": float(weights.max(initial=0.0)),
        "normalized_weights": weights.tolist(),
    }


def hard_p2_margins(pair: dict[str, Any], screen: dict[str, Any]) -> dict[str, float]:
    masses = pair.get("masses_msun_h")
    if masses is None:
        masses = [pair["m1_fof_msun_h"], pair["m2_fof_msun_h"]]
    low, high = map(float, screen["pair_member_mass_range_msun_h"])
    sep_low, sep_high = map(float, screen["pair_separation_range_mpc_h"])
    return {
        "member_mass_lower_msun_h": float(min(masses) - low),
        "member_mass_upper_msun_h": float(high - max(masses)),
        "mass_ratio": float(screen["pair_mass_ratio_max"] - pair["mass_ratio"]),
        "separation_lower_mpc_h": float(pair["separation_mpc_h"] - sep_low),
        "separation_upper_mpc_h": float(sep_high - pair["separation_mpc_h"]),
        "midpoint_offset_mpc_h": float(
            screen["pair_midpoint_max_offset_mpc_h"]
            - pair["midpoint_offset_mpc_h"]
        ),
        "isolation_mpc_h": float(
            pair["isolation_mpc_h"] - screen["isolation_radius_mpc_h"]
        ),
    }


def hard_p2_pass_from_margins(margins: dict[str, float]) -> bool:
    return all(value >= 0.0 for value in margins.values())


def recentered_p1_margins(
    metrics: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for name in ("Virgo", "Coma"):
        observed = metrics["clusters"][name]
        spec = config["clusters"][name]
        rows[name] = {
            "target_delta": float(observed["target_delta"]),
            "target_percentile": float(
                observed["target_shell_percentile"]
                - spec["minimum_target_percentile"]
            ),
            "peak_percentile": float(
                observed["peak_shell_percentile"]
                - spec["minimum_peak_percentile"]
            ),
            "peak_separation_mpc_h": float(
                spec["search_radius_mpc_h"]
                - observed["peak_separation_mpc_h"]
            ),
        }
    local = metrics["local_void"]
    local_spec = config["local_void"]
    rows["LocalVoid"] = {
        "underdense_probe_count": float(
            local["n_underdense"] - local_spec["minimum_underdense_probes"]
        ),
        "negative_probe_mean": float(-local["probe_mean_delta"]),
        "median_percentile": float(
            local_spec["maximum_median_shell_percentile"]
            - local["median_centre_shell_percentile"]
        ),
    }
    bootes = metrics["bootes_void"]
    bootes_spec = config["bootes_void"]
    rows["BootesVoid"] = {
        "centre_percentile": float(
            bootes_spec["maximum_center_shell_percentile"]
            - bootes["centre_shell_percentile"]
        ),
        **{
            f"negative_mean_radius_{float(radius):g}_mpc_h": float(
                -bootes["mean_delta_profile"][str(float(radius))]
            )
            for radius in bootes_spec["require_negative_mean_at_radii_mpc_h"]
        },
    }
    observer = metrics["observer_environment"]
    observer_spec = config["observer_environment"]
    observer_rows = {
        f"excess_mass_radius_{radius}_msun_h": float(
            sphere["maximum_excess_mass_msun_h"] - sphere["excess_mass_msun_h"]
        )
        for radius, sphere in observer["spheres"].items()
    }
    sheet = observer["spheres"][str(float(
        observer_spec["local_sheet_radius_mpc_h"]
    ))]
    observer_rows["local_sheet_mean_delta"] = float(
        sheet["mean_delta"] - observer_spec["minimum_local_sheet_mean_delta"]
    )
    rows["ObserverEnvironment"] = observer_rows
    return rows


def _hash_entries(
    label: str, entries: list[dict[str, Any]], progress_every: int = 32
) -> list[dict[str, str]]:
    mismatches = []
    for number, row in enumerate(entries, 1):
        path = Path(row["field"])
        expected = row["field_sha256"]
        if not path.is_file():
            mismatches.append({
                "label": label, "path": str(path), "error": "missing"
            })
        else:
            actual = sha256_file(path)
            if actual != expected:
                mismatches.append({
                    "label": label,
                    "path": str(path),
                    "expected": expected,
                    "actual": actual,
                })
        if number % progress_every == 0 or number == len(entries):
            print(f"[lineage] {label} {number}/{len(entries)}", flush=True)
    return mismatches


def _spearman_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs = [
        "q_x", "q_y", "q_z", "q_radius", "axis_x", "axis_y", "axis_z"
    ]
    outputs = [
        "soft_log_likelihood", "best_pair_log_likelihood",
        "best_pair_midpoint_offset", "best_pair_separation",
        "best_pair_mass_ratio", "best_pair_hard_minimum_margin",
        "best_recentered_gate_count", "best_recentered_massive_count",
    ]
    results = []
    for source in inputs:
        for target in outputs:
            selected = [
                row for row in rows
                if np.isfinite(row.get(source, math.nan))
                and np.isfinite(row.get(target, math.nan))
            ]
            if len(selected) < 8:
                continue
            statistic = spearmanr(
                [row[source] for row in selected],
                [row[target] for row in selected],
            )
            results.append({
                "input": source,
                "output": target,
                "n": len(selected),
                "rho": float(statistic.statistic),
                "pvalue_descriptive_only": float(statistic.pvalue),
            })
    results.sort(key=lambda row: abs(row["rho"]), reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = json.loads(args.program.read_text())
    if program.get("status") != "frozen_before_detailed_v8_pairwise_attribution":
        parser.error("autopsy program is not frozen")

    for key in ("parent_terminal_record", "fable_independent_audit"):
        spec = program[key]
        path = ROOT / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            parser.error(f"{key} hash mismatch")
        if json.loads(path.read_text()).get("status") != spec["required_status"]:
            parser.error(f"{key} status mismatch")
    terminal = json.loads((ROOT / program["parent_terminal_record"]["path"]).read_text())
    evidence = terminal["immutable_evidence"]
    loaded = {}
    for name, spec in evidence.items():
        path = Path(spec["path"])
        if not path.is_absolute():
            path = ROOT / path
        if sha256_file(path) != spec["sha256"]:
            parser.error(f"terminal evidence hash mismatch: {name}")
        loaded[name] = (path, json.loads(path.read_text()))

    proposal_path, proposal = loaded["proposal_manifest"]
    projection_path, projection = loaded["parent_projection_manifest"]
    p1_path, p1 = loaded["conditioned_p1_result"]
    p2_config_path, _ = loaded["derived_p2_config"]
    p2_path, p2 = loaded["p2_result"]
    score_path, score = loaded["z0_importance_score"]
    pair_input_path, pair_input = loaded["pair_recenter_input"]
    preview_path, preview = loaded["recentered_p1_result"]
    gate_path, gate = loaded["final_gate"]
    v8_program_path, v8_program = loaded["v8_program"]
    p1_config_path = ROOT / "config/p1_targets_v2_observer.json"
    p1_config = json.loads(p1_config_path.read_text())
    resolved_p2 = load_config(p2_config_path)
    screen = resolved_p2["screen"]

    proposal_by_seed = {
        int(row["proposal_seed"]): row for row in proposal["entries"]
    }
    projection_by_seed = {
        int(row["proposal_seed"]): row for row in projection["entries"]
    }
    p1_by_seed = {int(row["seed"]): row for row in p1["members"]}
    p2_by_seed = {
        int(row["small_scale_seed"]): row for row in p2["results"]
    }
    score_by_seed = {
        int(row["small_scale_seed"]): row for row in score["rows"]
    }
    preview_by_seed = {
        int(row["small_scale_seed"]): row for row in preview["rows"]
    }
    expected_seeds = list(range(5269, 5525))

    lineage_checks = {
        "proposal_seed_set_exact": sorted(proposal_by_seed) == expected_seeds,
        "projection_seed_set_exact": sorted(projection_by_seed) == expected_seeds,
        "P1_seed_set_exact": sorted(p1_by_seed) == expected_seeds,
        "P2_seed_set_equals_P1_survivors": (
            sorted(p2_by_seed) == sorted(map(int, p1["passing_seeds"]))
        ),
        "score_seed_set_equals_P2": sorted(score_by_seed) == sorted(p2_by_seed),
        "proposal_manifest_program_hash": (
            proposal["config_sha256"] == sha256_file(v8_program_path)
        ),
        "projection_source_manifest_hash": (
            projection["source_proposal_manifest_sha256"]
            == sha256_file(proposal_path)
        ),
        "P1_projection_manifest_hash": (
            p1["manifest_sha256"] == sha256_file(projection_path)
        ),
        "P2_proposal_manifest_hash": (
            p2["conditioned_proposal_manifest_sha256"]
            == sha256_file(proposal_path)
        ),
        "P2_conditioned_P1_hash": (
            p2["conditioned_p1_result_sha256"] == sha256_file(p1_path)
        ),
        "score_proposal_manifest_hash": (
            score["proposal_manifest_sha256"] == sha256_file(proposal_path)
        ),
        "score_P2_hash": score["p2_result_sha256"] == sha256_file(p2_path),
        "pair_input_score_hash": (
            pair_input["source_likelihood_result_sha256"] == sha256_file(score_path)
        ),
        "pair_input_P2_hash": (
            pair_input["source_p2_result_sha256"] == sha256_file(p2_path)
        ),
        "preview_pair_input_hash": (
            preview["p2_result_sha256"] == sha256_file(pair_input_path)
        ),
        "preview_P1_hash": (
            preview["conditioned_p1_result_sha256"] == sha256_file(p1_path)
        ),
        "gate_score_hash": (
            gate["likelihood_result_sha256"] == sha256_file(score_path)
        ),
        "gate_preview_hash": (
            gate["recentered_P1_result_sha256"] == sha256_file(preview_path)
        ),
        "gate_P2_hash": gate["hard_P2_result_sha256"] == sha256_file(p2_path),
    }
    lineage_mismatches = []
    lineage_mismatches += _hash_entries("N576_proposal", proposal["entries"])
    lineage_mismatches += _hash_entries("N192_projection", projection["entries"])
    for row in score["rows"]:
        path = Path(row["catalog"])
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != row["catalog_sha256"]:
            lineage_mismatches.append({
                "label": "P2_catalog", "path": str(path),
                "expected": row["catalog_sha256"], "actual": actual,
            })
    print(f"[lineage] P2_catalog {len(score['rows'])}/{len(score['rows'])}", flush=True)
    lineage_checks["all_rehashed_fields_match"] = not lineage_mismatches

    hard_failure_counts: Counter[str] = Counter()
    hard_pair_recentered_rows = []
    recentered_pass_hard_rows = []
    all_rows = []
    stage_logs = {name: [] for name in (
        "likelihood_only", "soft", "hard_P2", "recentered_P1", "joint",
        "best_pair_soft",
    )}
    for seed in expected_seeds:
        proposal_row = proposal_by_seed[seed]
        q = np.asarray(
            proposal_row["protohalo_midpoint_offset_draw_mpc_h"], dtype=np.float64
        )
        axis = np.asarray(proposal_row["axis"], dtype=np.float64)
        p1_pass = bool(p1_by_seed[seed]["pass"])
        source = score_by_seed.get(seed)
        candidates = source["candidate_pairs"] if source else []
        candidate_by_key = {pair_key(pair): pair for pair in candidates}
        hard_pairs = p2_by_seed.get(seed, {}).get("screen_pairs", [])
        hard_keys = {pair_key(pair) for pair in hard_pairs}
        preview_rows = preview_by_seed.get(seed, {}).get("pair_rows", [])
        recentered_keys = {
            pair_key(row["screen_pair"]) for row in preview_rows
            if row["preview_pass"]
        }
        correction = (
            float(source["midpoint_importance"][
                "log_target_prior_over_sampling_proposal"
            ]) if source else math.nan
        )
        soft_ll = logmeanexp([pair["log_likelihood"] for pair in candidates])
        hard_ll = logmeanexp([
            pair["log_likelihood"] for key, pair in candidate_by_key.items()
            if key in hard_keys
        ])
        recentered_ll = logmeanexp([
            pair["log_likelihood"] for key, pair in candidate_by_key.items()
            if key in recentered_keys
        ])
        joint_keys = hard_keys & recentered_keys
        joint_ll = logmeanexp([
            pair["log_likelihood"] for key, pair in candidate_by_key.items()
            if key in joint_keys
        ])
        best_ll = max(
            [pair["log_likelihood"] for pair in candidates], default=-math.inf
        )
        likelihood_only = soft_ll
        stage_values = {
            "likelihood_only": likelihood_only,
            "soft": soft_ll + correction if np.isfinite(soft_ll) else -math.inf,
            "hard_P2": hard_ll + correction if np.isfinite(hard_ll) else -math.inf,
            "recentered_P1": (
                recentered_ll + correction if np.isfinite(recentered_ll) else -math.inf
            ),
            "joint": joint_ll + correction if np.isfinite(joint_ll) else -math.inf,
            "best_pair_soft": (
                best_ll + correction if np.isfinite(best_ll) else -math.inf
            ),
        }
        for name, value in stage_values.items():
            stage_logs[name].append(value)

        best_pair = source["best_pair"] if source else None
        best_hard_margins = hard_p2_margins(best_pair, screen) if best_pair else {}
        best_preview = preview_by_seed.get(seed, {}).get("best_recentered_pair")
        diagnostic = {
            "seed": seed,
            "parent_P1_pass": p1_pass,
            "loose_pair_count": len(candidates),
            "hard_P2_pair_count": len(hard_keys),
            "recentered_P1_pair_count": len(recentered_keys),
            "joint_pair_count": len(joint_keys),
            "importance_log_correction": correction,
            "q_x": float(q[0]), "q_y": float(q[1]), "q_z": float(q[2]),
            "q_radius": float(np.linalg.norm(q)),
            "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]),
            "soft_log_likelihood": soft_ll,
            "best_pair_log_likelihood": best_ll,
            "best_pair_midpoint_offset": (
                float(best_pair["midpoint_offset_mpc_h"]) if best_pair else math.nan
            ),
            "best_pair_separation": (
                float(best_pair["separation_mpc_h"]) if best_pair else math.nan
            ),
            "best_pair_mass_ratio": (
                float(best_pair["mass_ratio"]) if best_pair else math.nan
            ),
            "best_pair_hard_minimum_margin": (
                min(best_hard_margins.values()) if best_hard_margins else math.nan
            ),
            "best_recentered_gate_count": (
                int(best_preview["p1_recentered"]["n_gates_passed"])
                if best_preview else math.nan
            ),
            "best_recentered_massive_count": (
                len(best_preview["massive_screen_halos_within_8_mpc_h"])
                if best_preview else math.nan
            ),
            "stage_log_weights": stage_values,
        }
        all_rows.append(diagnostic)

        preview_by_key = {
            pair_key(row["screen_pair"]): row for row in preview_rows
        }
        for hard_pair in hard_pairs:
            key = pair_key(hard_pair)
            recentered = preview_by_key[key]
            gates = recentered["p1_recentered"]["gates"]
            for name, passed in gates.items():
                if not passed:
                    hard_failure_counts[name] += 1
            if recentered["massive_screen_halos_within_8_mpc_h"]:
                hard_failure_counts["MassiveHaloVeto"] += 1
            hard_pair_recentered_rows.append({
                "seed": seed,
                "pair": list(key),
                "recentered_preview_pass": bool(recentered["preview_pass"]),
                "gates": gates,
                "n_gates_passed": int(
                    recentered["p1_recentered"]["n_gates_passed"]
                ),
                "massive_halo_count": len(
                    recentered["massive_screen_halos_within_8_mpc_h"]
                ),
                "P1_margins": recentered_p1_margins(
                    recentered["p1_recentered"], p1_config
                ),
                "hard_P2_margins": hard_p2_margins(hard_pair, screen),
            })
        for key in recentered_keys:
            pair = candidate_by_key[key]
            margins = hard_p2_margins(pair, screen)
            recentered_pass_hard_rows.append({
                "seed": seed,
                "pair": list(key),
                "hard_P2_pass_reconstructed": hard_p2_pass_from_margins(margins),
                "hard_P2_margins": margins,
            })

    stage_summaries = {name: weight_summary(values) for name, values in stage_logs.items()}
    reproduced = {
        "generated": len(expected_seeds),
        "parent_P1": sum(row["parent_P1_pass"] for row in all_rows),
        "finite_soft_likelihood": stage_summaries["soft"]["n_nonzero"],
        "hard_P2_realizations": sum(row["hard_P2_pair_count"] > 0 for row in all_rows),
        "hard_P2_pairs": sum(row["hard_P2_pair_count"] for row in all_rows),
        "recentered_P1_realizations": sum(
            row["recentered_P1_pair_count"] > 0 for row in all_rows
        ),
        "recentered_P1_pairs": sum(
            row["recentered_P1_pair_count"] for row in all_rows
        ),
        "joint_realizations": sum(row["joint_pair_count"] > 0 for row in all_rows),
        "joint_pairs": sum(row["joint_pair_count"] for row in all_rows),
    }
    expected = program["information_firewall"]["already_consumed_terminal_summary"]
    funnel_checks = {
        key: reproduced[key] == expected[key]
        for key in (
            "generated", "parent_P1", "finite_soft_likelihood",
            "hard_P2_realizations", "recentered_P1_realizations",
            "joint_realizations",
        )
    }
    funnel_checks["joint_seed"] = [
        row["seed"] for row in all_rows if row["joint_pair_count"] > 0
    ] == [int(expected["joint_seed"])]
    funnel_checks["final_ESS"] = np.isclose(
        stage_summaries["joint"]["effective_sample_size"],
        float(expected["final_ESS"]),
    )
    all_lineage = all(lineage_checks.values()) and not lineage_mismatches
    exact_reproduction = all(funnel_checks.values())

    soft_weights = np.asarray(stage_summaries["soft"]["normalized_weights"])
    best_weights = np.asarray(stage_summaries["best_pair_soft"]["normalized_weights"])
    sensitivity = {
        "soft_uniform_pair_mixture_ESS": stage_summaries["soft"][
            "effective_sample_size"
        ],
        "soft_best_pair_ESS": stage_summaries["best_pair_soft"][
            "effective_sample_size"
        ],
        "total_variation_between_normalized_weights": float(
            0.5 * np.abs(soft_weights - best_weights).sum()
        ),
    }
    report = {
        "schema": "ouruniv-cf4-lg-v8-joint-support-autopsy-result-v1",
        "status": (
            "complete_pass_authorize_CF4_mode_release_audit"
            if all_lineage and exact_reproduction
            else "complete_integrity_failure_stop"
        ),
        "program": str(args.program.resolve()),
        "program_sha256": sha256_file(args.program),
        "terminal_record_sha256": sha256_file(
            ROOT / program["parent_terminal_record"]["path"]
        ),
        "lineage": {
            "checks": lineage_checks,
            "mismatches": lineage_mismatches,
            "all_pass": all_lineage,
        },
        "funnel": reproduced,
        "funnel_checks": funnel_checks,
        "exact_terminal_reproduction": exact_reproduction,
        "stagewise_importance": {
            name: {key: value for key, value in summary.items()
                   if key != "normalized_weights"}
            for name, summary in stage_summaries.items()
        },
        "pair_prior_sensitivity": sensitivity,
        "hard_P2_pair_recentered_P1_attribution": {
            "failure_counts": dict(hard_failure_counts),
            "rows": hard_pair_recentered_rows,
        },
        "recentered_P1_pass_pair_hard_P2_attribution": recentered_pass_hard_rows,
        "latent_associations_descriptive_only": _spearman_rows(all_rows),
        "rows": all_rows,
        "decision": {
            "authorize_CF4_mode_release_audit": all_lineage and exact_reproduction,
            "fresh_v9_authorized": False,
            "RAMSES_authorized": False,
            "same_model_seed_extension_authorized": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    print(json.dumps({
        "status": report["status"],
        "lineage_all_pass": all_lineage,
        "funnel": reproduced,
        "stagewise_importance": report["stagewise_importance"],
        "pair_prior_sensitivity": sensitivity,
        "hard_P2_recentered_failure_counts": dict(hard_failure_counts),
        "authorize_CF4_mode_release_audit": report["decision"][
            "authorize_CF4_mode_release_audit"
        ],
    }, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
