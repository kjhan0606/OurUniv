#!/usr/bin/env python
"""Test a consumed-only LOO conditional Gaussian location-scale control."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import hong2021_v75_rank_coverage_exact_null as v75
import hong2021_v82a_consumed_autopsy as v82a
import hong2021_v82b_gaussian_control as v82b
from hong2021_residual_evaluate import SpectralBinner
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = "hong2021-v82c-consumed-loo-location-scale-Gaussian-program-v1"
PROGRAM_STATUS = "frozen_and_pushed_before_V82C_draw_or_report"
REPORT_SCHEMA = "hong2021-v82c-consumed-loo-location-scale-Gaussian-v1"
DOMAIN_ORDER = v82b.DOMAIN_ORDER
QUERIES = v82b.QUERIES
MEMBERS = v82b.MEMBERS
GRID = v82b.GRID
VOXEL_MPC_H = v82b.VOXEL_MPC_H
REFERENCE_THRESHOLD = v82b.INDIVIDUAL_REFERENCE_THRESHOLD


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _bound_artifact(row: dict[str, Any], label: str) -> Path:
    if set(row) != {"path", "sha256"}:
        raise ValueError(f"V82C {label} artifact keys differ")
    path = Path(row["path"]).resolve()
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"V82C {label} artifact hash differs")
    return path


def program_freeze_commit(path: Path, repo: Path) -> str:
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(path.resolve())],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("V82C freeze commit cannot be resolved")
    return commit


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    program = strict_json(path.resolve())
    boundary = program.get("scope_boundary", {})
    authorization = program.get("authorization", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("engineering_only") is not True
        or program.get("statistically_independent") is not False
        or boundary.get("same_V82B_seeds_and_LOO_power") is not True
        or boundary.get("only_new_component_is_LOO_quartile_location") is not True
        or boundary.get("train_neural_network") is not False
        or boundary.get("open_new_independent_data") is not False
        or authorization.get("user_approved_automatic_final_Gaussian_control") is not True
    ):
        raise ValueError("V82C boundary differs")
    for label, row in program["implementation_sources"].items():
        if sha256_file(resolve_path(repo, row["path"])) != row["sha256"]:
            raise ValueError(f"V82C implementation source differs: {label}")
    for label, row in program["support_artifacts"].items():
        _bound_artifact(row, f"support/{label}")
    if set(program["input_artifacts"]) != set(DOMAIN_ORDER):
        raise ValueError("V82C domains differ")
    for domain in DOMAIN_ORDER:
        if set(program["input_artifacts"][domain]) != {"consumed_ensemble"}:
            raise ValueError(f"V82C {domain} input set differs")
        _bound_artifact(
            program["input_artifacts"][domain]["consumed_ensemble"], domain
        )
    return program


def quartile_sums(
    residuals: np.ndarray, means: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((len(residuals), 4), dtype=np.float64)
    counts = np.zeros((len(residuals), 4), dtype=np.int64)
    for query in range(len(residuals)):
        labels = v82a.equal_count_strata(means[query])
        for label in range(4):
            selected = labels == label
            sums[query, label] = residuals[query][selected].sum()
            counts[query, label] = np.count_nonzero(selected)
    return sums, counts


def loo_quartile_location(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    location = (sums.sum(axis=0)[None] - sums) / (
        counts.sum(axis=0)[None] - counts
    )
    if not np.isfinite(location).all():
        raise ValueError("V82C LOO quartile location is invalid")
    return location


def apply_quartile_location_scale(
    residual: np.ndarray,
    mean: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    labels = v82a.equal_count_strata(mean)
    output = np.asarray(residual, dtype=np.float64) * scale[labels][None]
    output += location[labels][None]
    output -= output.mean(axis=(-3, -2, -1), keepdims=True)
    return output


def _summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def inspect_domain(
    domain: str,
    path: Path,
    seed: int,
    program_sha: str,
    expected_V82B_scale_histogram: list[int],
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        truth = np.asarray(handle["truth"][:, 0], dtype=np.float64)
        mean = np.asarray(handle["conditional_mean"][:, 0], dtype=np.float64)
        source_indices = handle["source_index"][:].astype(np.int64).tolist()
    if truth.shape != (QUERIES, GRID, GRID, GRID) or mean.shape != truth.shape:
        raise ValueError(f"V82C {domain} input shape differs")
    truth_residual = v82b.centered_truth_residuals(truth, mean)
    shells = v82b.PeriodicShells(GRID, VOXEL_MPC_H)
    loo_power, _ = v82b.loo_periodic_mode_power(truth_residual, shells)
    square_sum, square_count = v82b.quartile_sumsquares(truth_residual, mean)
    scales, _ = v82b.loo_quartile_scales(square_sum, square_count)
    sums, counts = quartile_sums(truth_residual, mean)
    location = loo_quartile_location(sums, counts)
    aggregate = np.zeros(MEMBERS + 1, dtype=np.int64)
    scale_crosscheck = np.zeros(MEMBERS + 1, dtype=np.int64)
    aggregate_strata = [np.zeros(MEMBERS + 1, dtype=np.int64) for _ in range(4)]
    tables = []
    query_rows = []
    generator = np.random.default_rng(seed)
    binner = SpectralBinner(GRID, VOXEL_MPC_H)
    for query in range(QUERIES):
        stationary = v82b.gaussian_residual_draws(
            generator, loo_power[query], shells
        )
        scale_only = v82b.apply_quartile_scale(
            stationary, mean[query], scales[query]
        )
        generated = apply_quartile_location_scale(
            stationary, mean[query], location[query], scales[query]
        )
        strata = v82a.equal_count_strata(mean[query])
        measured = v82a.rank_and_coverage(
            generated, truth_residual[query], strata
        )
        aggregate += np.asarray(measured["shape"]["histogram"])
        scale_measured = v82a.rank_and_coverage(
            scale_only, truth_residual[query], strata
        )
        scale_crosscheck += np.asarray(scale_measured["shape"]["histogram"])
        for label in range(4):
            aggregate_strata[label] += np.asarray(
                measured["conditional_mean_quartiles"][label]["histogram"]
            )
        tie_seed = v75.derived_seed(
            program_sha, domain, str(query), "location_scale_tie_priority"
        )
        tables.append(
            v75.query_label_table(
                np.concatenate((generated, truth_residual[query][None])),
                np.random.default_rng(tie_seed),
            )
        )
        spectrum = v82a.spectral_query(
            binner, generated, scale_only, truth_residual[query]
        )["candidate"]
        query_rows.append(
            {
                "query_position": query,
                "source_index": source_indices[query],
                "LOO_location_low_to_high": location[query].tolist(),
                "LOO_scale_low_to_high": scales[query].tolist(),
                "rank_and_coverage": measured,
                "spectrum": spectrum,
            }
        )
        print(f"[v82c] {domain} query {query + 1}/{QUERIES}", flush=True)
    if scale_crosscheck.tolist() != list(map(int, expected_V82B_scale_histogram)):
        raise ValueError(f"V82C {domain} does not reproduce V82B scale-only draws")
    exact = v82b.exact_rank_null(tables, program_sha, domain, "location_scale")
    return {
        "source_indices": source_indices,
        "random_seed_reproduces_V82B": seed,
        "V82B_quartile_scale_histogram_crosscheck_pass": True,
        "LOO_location_summary": _summary(
            [
                {f"quartile_{label}": float(location[query, label]) for label in range(4)}
                for query in range(QUERIES)
            ]
        ),
        "LOO_scale_summary": _summary(
            [
                {f"quartile_{label}": float(scales[query, label]) for label in range(4)}
                for query in range(QUERIES)
            ]
        ),
        "aggregate_rank_shape": v82a.rank_shape(aggregate),
        "aggregate_rank_shape_by_conditional_mean_quartile": [
            v82a.rank_shape(value) for value in aggregate_strata
        ],
        "per_query_coverage": _summary(
            [
                {
                    "coverage68_minus_expected": row["rank_and_coverage"][
                        "coverage68_minus_expected"
                    ],
                    "coverage95_minus_expected": row["rank_and_coverage"][
                        "coverage95_minus_expected"
                    ],
                }
                for row in query_rows
            ]
        ),
        "spectrum": {
            metric: _summary([row["spectrum"][metric] for row in query_rows])
            for metric in (
                "member_mean_residual_power_over_truth",
                "ensemble_mean_truth_phase_cosine",
                "member_truth_phase_cosine_mean",
            )
        },
        "exact_rank_coverage_null": exact,
        "per_query": query_rows,
    }


def decide(domains: dict[str, Any]) -> dict[str, Any]:
    values = []
    rows = {}
    for domain in DOMAIN_ORDER:
        p_values = domains[domain]["exact_rank_coverage_null"][
            "conditional_p_values"
        ]
        rows[domain] = p_values
        values.extend((p_values["rank_tv"], p_values["coverage_deviation"]))
    compatible = bool(all(value > REFERENCE_THRESHOLD for value in values))
    return {
        "domains": rows,
        "six_individual_p_values": values,
        "individual_reference_threshold": REFERENCE_THRESHOLD,
        "all_six_above_reference_threshold": compatible,
        "branch": (
            "conditional_Gaussian_location_scale_sufficient_no_flow"
            if compatible
            else "conditional_location_scale_Gaussian_fails_nonGaussian_model_warranted"
        ),
        "conditional_flow_or_equivalent_nonGaussian_model_warranted": not compatible,
        "automatic_training_authorized": False,
        "engineering_only_not_validation": True,
    }


def run(program_path: Path, repo: Path, output_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program_path = program_path.resolve()
    output_path = output_path.resolve()
    program = load_program(program_path, repo)
    if output_path != Path(program["outputs"]["report"]).resolve():
        raise ValueError("V82C output binding differs")
    commit, clean = git_state(repo)
    freeze_commit = program_freeze_commit(program_path, repo)
    if not clean or not _is_ancestor(repo, freeze_commit, commit):
        raise ValueError("V82C requires a clean descendant of its freeze commit")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise ValueError("V82C must execute on lageunha")
    if output_path.exists() or output_path.with_suffix(".json.partial").exists():
        raise FileExistsError("V82C refuses existing output")
    v82b_report = strict_json(
        Path(program["support_artifacts"]["V82B_report"]["path"])
    )
    program_sha = sha256_file(program_path)
    domains = {}
    for domain in DOMAIN_ORDER:
        expected = v82b_report["domains"][domain]["baselines"]["quartile_scale"][
            "aggregate_rank_shape"
        ]["histogram"]
        domains[domain] = inspect_domain(
            domain,
            Path(program["input_artifacts"][domain]["consumed_ensemble"]["path"]),
            int(program["fixed_random_seeds"][domain]),
            program_sha,
            expected,
        )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_consumed_only_LOO_location_scale_Gaussian_control",
        "program": str(program_path),
        "program_sha256": program_sha,
        "program_freeze_commit": freeze_commit,
        "execution_commit": commit,
        "hostname": socket.gethostname(),
        "engineering_only": True,
        "statistically_independent": False,
        "domains": domains,
        "decision": decide(domains),
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.program, args.repo, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "decision_digest_sha256": report["decision_digest_sha256"],
                "out": str(args.out.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
