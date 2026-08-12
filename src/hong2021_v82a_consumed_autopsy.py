#!/usr/bin/env python
"""Diagnose V80 rank and residual-spectrum failures on consumed ensembles only.

V82A is an engineering autopsy, not a validation gate.  It refuses unbound
inputs and writes no model, ensemble, calibration, or source-data artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_residual_evaluate import BANDS, SpectralBinner, band_key
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = "hong2021-v82a-consumed-rank-phase-autopsy-program-v1"
PROGRAM_STATUS = (
    "frozen_and_pushed_after_support_hash_correction_before_bound_ensemble_payload_inspection"
)
REPORT_SCHEMA = "hong2021-v82a-consumed-rank-phase-autopsy-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
ARM_ORDER = ("candidate", "control")
QUERIES = 32
MEMBERS = 16
GRID = 64
VOXEL_MPC_H = 0.3125
EXPECTED_COVERAGE_68 = (0.84 - 0.16) * (MEMBERS - 1) / (MEMBERS + 1)
EXPECTED_COVERAGE_95 = (0.975 - 0.025) * (MEMBERS - 1) / (MEMBERS + 1)


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def program_freeze_commit(program_path: Path, repo: Path) -> str:
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(program_path.resolve())],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("V82A program freeze commit cannot be resolved")
    return commit


def _bound_artifact(row: dict[str, Any], label: str) -> tuple[Path, str]:
    if set(row) != {"path", "sha256"}:
        raise ValueError(f"V82A {label} artifact keys differ")
    path = Path(str(row["path"])).resolve()
    digest = str(row["sha256"])
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"V82A {label} artifact hash differs")
    return path, digest


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program = strict_json(path.resolve())
    authorization = program.get("authorization", {})
    boundary = program.get("scope_boundary", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("engineering_only") is not True
        or program.get("statistically_independent") is not False
        or authorization.get("user_approved_consumed_only_autopsy") is not True
        or boundary.get("write_or_mutate_input") is not False
        or boundary.get("train_or_select_model") is not False
        or boundary.get("open_new_independent_data") is not False
        or float(program.get("voxel_mpc_h", -1.0)) != VOXEL_MPC_H
    ):
        raise ValueError("V82A scope boundary differs")
    for label, row in program["implementation_sources"].items():
        source = resolve_path(repo, str(row["path"]))
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"V82A implementation source differs: {label}")
    for label, row in program["support_artifacts"].items():
        _bound_artifact(row, f"support/{label}")
    required = {"candidate_ensemble", "control_ensemble", "candidate_metrics", "control_metrics"}
    if set(program["input_artifacts"]) != set(DOMAIN_ORDER):
        raise ValueError("V82A domains differ")
    seen: set[Path] = set()
    for domain in DOMAIN_ORDER:
        rows = program["input_artifacts"][domain]
        if set(rows) != required:
            raise ValueError(f"V82A {domain} input artifact set differs")
        for label, row in rows.items():
            artifact, _ = _bound_artifact(row, f"{domain}/{label}")
            if artifact in seen:
                raise ValueError("V82A input path reused")
            seen.add(artifact)
    return program


def rank_shape(histogram: np.ndarray) -> dict[str, float | list[int]]:
    counts = np.asarray(histogram, dtype=np.int64)
    if counts.shape != (MEMBERS + 1,) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("rank histogram shape or counts differ")
    probability = counts / counts.sum()
    rank = np.arange(MEMBERS + 1, dtype=np.float64)
    mean = float(np.dot(probability, rank))
    variance = float(np.dot(probability, np.square(rank - mean)))
    expected = 1.0 / (MEMBERS + 1)
    edge = float(probability[[0, MEMBERS]].sum())
    center = float(probability[7:10].sum())
    outer4 = float(probability[[0, 1, MEMBERS - 1, MEMBERS]].sum())
    return {
        "histogram": counts.tolist(),
        "total_variation_from_uniform": float(0.5 * np.abs(probability - expected).sum()),
        "mean_rank_minus_8": mean - MEMBERS / 2,
        "rank_variance_over_uniform": variance / 24.0,
        "edge_fraction": edge,
        "edge_fraction_over_uniform": edge / (2.0 / 17.0),
        "central_7_8_9_fraction": center,
        "central_fraction_over_uniform": center / (3.0 / 17.0),
        "outer4_minus_central3_excess": outer4 / (4.0 / 17.0) - center / (3.0 / 17.0),
        "left_minus_right_fraction": float(probability[:8].sum() - probability[9:].sum()),
    }


def equal_count_strata(field: np.ndarray, strata: int = 4) -> np.ndarray:
    """Return deterministic within-query conditional-mean rank strata."""
    flat = np.asarray(field).reshape(-1)
    order = np.argsort(flat, kind="stable")
    labels = np.empty(len(flat), dtype=np.uint8)
    for label, indices in enumerate(np.array_split(order, strata)):
        labels[indices] = label
    return labels.reshape(np.asarray(field).shape)


def rank_and_coverage(
    residual: np.ndarray, target_residual: np.ndarray, strata: np.ndarray
) -> dict[str, Any]:
    rank = np.sum(residual < target_residual[None], axis=0)
    histogram = np.bincount(rank.ravel(), minlength=MEMBERS + 1)
    lower68, upper68 = np.quantile(residual, [0.16, 0.84], axis=0)
    lower95, upper95 = np.quantile(residual, [0.025, 0.975], axis=0)
    by_stratum = []
    for label in range(4):
        selected = strata == label
        by_stratum.append(
            rank_shape(np.bincount(rank[selected], minlength=MEMBERS + 1))
        )
    return {
        "shape": rank_shape(histogram),
        "coverage68": float(np.mean((target_residual >= lower68) & (target_residual <= upper68))),
        "coverage95": float(np.mean((target_residual >= lower95) & (target_residual <= upper95))),
        "coverage68_minus_expected": float(
            np.mean((target_residual >= lower68) & (target_residual <= upper68))
            - EXPECTED_COVERAGE_68
        ),
        "coverage95_minus_expected": float(
            np.mean((target_residual >= lower95) & (target_residual <= upper95))
            - EXPECTED_COVERAGE_95
        ),
        "conditional_mean_quartiles": by_stratum,
    }


def _band_average(binner: SpectralBinner, value: np.ndarray) -> dict[str, float]:
    result = {}
    for low, high in BANDS:
        selected = (binner.k >= low) & (binner.k < high) & np.isfinite(value)
        result[band_key(low, high)] = float(
            np.average(value[selected], weights=binner.count[selected])
        )
    return result


def phase_cosine(binner: SpectralBinner, first_k: np.ndarray, second_k: np.ndarray) -> dict[str, float]:
    denominator = np.abs(first_k) * np.abs(second_k)
    cosine = np.divide(
        np.real(first_k * second_k.conjugate()),
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    flat = cosine.reshape((-1, binner.bin.size))
    numerator = []
    denominator_count = []
    for row in flat:
        finite = binner.valid & np.isfinite(row)
        numerator.append(
            np.bincount(
                binner.bin[finite], weights=row[finite], minlength=binner.grid // 2
            )
        )
        denominator_count.append(
            np.bincount(binner.bin[finite], minlength=binner.grid // 2)
        )
    mean = np.divide(
        np.sum(numerator, axis=0),
        np.sum(denominator_count, axis=0),
        out=np.full(binner.grid // 2, np.nan),
        where=np.sum(denominator_count, axis=0) > 0,
    )
    return _band_average(binner, mean)


def spectral_query(
    binner: SpectralBinner,
    candidate: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    truth_k = binner.transform(target[None])[0]
    transformed = {
        "candidate": binner.transform(candidate),
        "control": binner.transform(control),
    }
    truth_power = binner.radial_sum(np.abs(truth_k[None]) ** 2)[0]
    result: dict[str, Any] = {}
    for arm in ARM_ORDER:
        current = transformed[arm]
        member_power = binner.radial_sum(np.abs(current) ** 2).mean(axis=0)
        ratio = np.divide(
            member_power,
            truth_power,
            out=np.full_like(member_power, np.nan),
            where=truth_power > 0,
        )
        mean_k = current.mean(axis=0)
        result[arm] = {
            "member_mean_residual_power_over_truth": _band_average(binner, ratio),
            "ensemble_mean_truth_phase_cosine": phase_cosine(
                binner, mean_k[None], truth_k[None]
            ),
            "member_truth_phase_cosine_mean": phase_cosine(
                binner, current, np.broadcast_to(truth_k, current.shape)
            ),
        }
    result["paired_candidate_control_member_phase_cosine"] = phase_cosine(
        binner, transformed["candidate"], transformed["control"]
    )
    return result


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


def _nested_band_summary(rows: list[dict[str, Any]], arm: str, metric: str) -> dict[str, Any]:
    return _summary([row["spectrum"][arm][metric] for row in rows])


def inspect_domain(domain: str, rows: dict[str, Any]) -> dict[str, Any]:
    paths = {
        arm: Path(rows[f"{arm}_ensemble"]["path"]).resolve() for arm in ARM_ORDER
    }
    metrics = {
        arm: strict_json(Path(rows[f"{arm}_metrics"]["path"]).resolve()) for arm in ARM_ORDER
    }
    aggregate_hist = {arm: np.zeros(MEMBERS + 1, dtype=np.int64) for arm in ARM_ORDER}
    aggregate_strata = {
        arm: [np.zeros(MEMBERS + 1, dtype=np.int64) for _ in range(4)] for arm in ARM_ORDER
    }
    query_rows: list[dict[str, Any]] = []
    binner = SpectralBinner(GRID, VOXEL_MPC_H)
    with h5py.File(paths["candidate"], "r") as candidate, h5py.File(paths["control"], "r") as control:
        expected_sample = (QUERIES, MEMBERS, 1, GRID, GRID, GRID)
        expected_field = (QUERIES, 1, GRID, GRID, GRID)
        for handle in (candidate, control):
            if tuple(handle["sample"].shape) != expected_sample or tuple(handle["truth"].shape) != expected_field or tuple(handle["conditional_mean"].shape) != expected_field:
                raise ValueError(f"V82A {domain} ensemble shape differs")
        for name in ("truth", "conditional_mean", "source_index", "initial_latent_sha256"):
            if not np.array_equal(candidate[name][:], control[name][:]):
                raise ValueError(f"V82A {domain} paired {name} differs")
        source_indices = candidate["source_index"][:].astype(np.int64).tolist()
        for query in range(QUERIES):
            mean = np.asarray(candidate["conditional_mean"][query, 0], dtype=np.float64)
            target = np.asarray(candidate["truth"][query, 0], dtype=np.float64) - mean
            target -= target.mean()
            strata = equal_count_strata(mean)
            current: dict[str, Any] = {
                "query_position": query,
                "source_index": source_indices[query],
                "conditional_mean": {
                    "mean": float(mean.mean()),
                    "standard_deviation": float(mean.std()),
                    "q05_q50_q95": np.quantile(mean, [0.05, 0.5, 0.95]).tolist(),
                },
            }
            residuals = {}
            for arm, handle in (("candidate", candidate), ("control", control)):
                residual = np.asarray(handle["sample"][query, :, 0], dtype=np.float64) - mean
                residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
                residuals[arm] = residual
                current[arm] = rank_and_coverage(residual, target, strata)
                aggregate_hist[arm] += np.asarray(current[arm]["shape"]["histogram"])
                for label in range(4):
                    aggregate_strata[arm][label] += np.asarray(
                        current[arm]["conditional_mean_quartiles"][label]["histogram"]
                    )
            current["spectrum"] = spectral_query(
                binner, residuals["candidate"], residuals["control"], target
            )
            query_rows.append(current)
            print(f"[v82a] {domain} query {query + 1}/{QUERIES}", flush=True)
    arm_summary = {}
    for arm in ARM_ORDER:
        expected_metric_hist = next(iter(metrics[arm]["candidates"].values()))[
            "residual_calibration"
        ]["rank_histogram"]
        if aggregate_hist[arm].tolist() != list(map(int, expected_metric_hist)):
            raise ValueError(f"V82A {domain} {arm} rank histogram differs from sealed evaluator")
        arm_summary[arm] = {
            "aggregate_rank_shape": rank_shape(aggregate_hist[arm]),
            "aggregate_rank_shape_by_conditional_mean_quartile": [
                rank_shape(value) for value in aggregate_strata[arm]
            ],
            "per_query_rank_tv": _summary(
                [{"value": row[arm]["shape"]["total_variation_from_uniform"]} for row in query_rows]
            )["value"],
            "per_query_coverage": _summary(
                [
                    {
                        "coverage68_minus_expected": row[arm]["coverage68_minus_expected"],
                        "coverage95_minus_expected": row[arm]["coverage95_minus_expected"],
                    }
                    for row in query_rows
                ]
            ),
            "spectrum": {
                metric: _nested_band_summary(query_rows, arm, metric)
                for metric in (
                    "member_mean_residual_power_over_truth",
                    "ensemble_mean_truth_phase_cosine",
                    "member_truth_phase_cosine_mean",
                )
            },
        }
    paired = {
        "rank_tv_candidate_minus_control": _summary(
            [
                {
                    "value": row["candidate"]["shape"]["total_variation_from_uniform"]
                    - row["control"]["shape"]["total_variation_from_uniform"]
                }
                for row in query_rows
            ]
        )["value"],
        "candidate_rank_tv_worse_query_count": int(
            sum(
                row["candidate"]["shape"]["total_variation_from_uniform"]
                > row["control"]["shape"]["total_variation_from_uniform"]
                for row in query_rows
            )
        ),
        "matched_member_phase_cosine": _summary(
            [row["spectrum"]["paired_candidate_control_member_phase_cosine"] for row in query_rows]
        ),
    }
    return {
        "source_indices": source_indices,
        "arms": arm_summary,
        "paired_candidate_control": paired,
        "per_query": query_rows,
    }


def root_cause_summary(domains: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for domain in DOMAIN_ORDER:
        candidate = domains[domain]["arms"]["candidate"]["aggregate_rank_shape"]
        control = domains[domain]["arms"]["control"]["aggregate_rank_shape"]
        rows.append(
            {
                "domain": domain,
                "candidate_rank_tv": candidate["total_variation_from_uniform"],
                "control_rank_tv": control["total_variation_from_uniform"],
                "candidate_minus_control_rank_tv": candidate["total_variation_from_uniform"]
                - control["total_variation_from_uniform"],
                "candidate_variance_ratio": candidate["rank_variance_over_uniform"],
                "control_variance_ratio": control["rank_variance_over_uniform"],
                "candidate_mean_rank_bias": candidate["mean_rank_minus_8"],
                "control_mean_rank_bias": control["mean_rank_minus_8"],
            }
        )
    return {
        "descriptive_domain_table": rows,
        "interpretation_rule_frozen_before_payload_access": {
            "shared_failure": "candidate and control have similar rank-shape deviations; prioritize inherited conditional marginal/residual calibration",
            "candidate_specific_failure": "candidate is consistently worse than paired control; prioritize V72 spatial transport",
            "domain_specific_failure": "shape or amplitude failure changes materially by domain or conditional-mean quartile; prioritize conditional/domain modeling",
            "amplitude_not_phase": "residual-power ratios fail while held-out truth phase cosines remain near their paired-control reference; prioritize scale calibration, not phase copying",
            "phase_note": "held-out stochastic truth phases are not required to match individual generated-member phases",
        },
        "automatic_model_decision": False,
        "statistical_validation_claim": False,
    }


def run(program_path: Path, repo: Path, output_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program_path = program_path.resolve()
    output_path = output_path.resolve()
    program = load_program(program_path, repo)
    expected_output = Path(program["outputs"]["report"]).resolve()
    if output_path != expected_output or program["outputs"].get("refuse_existing") is not True:
        raise ValueError("V82A output binding differs")
    commit, clean = git_state(repo)
    freeze_commit = program_freeze_commit(program_path, repo)
    if not clean or not _is_ancestor(repo, freeze_commit, commit):
        raise ValueError("V82A requires a clean descendant of its freeze commit")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise ValueError("V82A must execute on lageunha")
    if output_path.exists():
        raise FileExistsError(f"V82A refuses existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    domains = {
        domain: inspect_domain(domain, program["input_artifacts"][domain])
        for domain in DOMAIN_ORDER
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_consumed_only_engineering_autopsy",
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": freeze_commit,
        "execution_commit": commit,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "engineering_only": True,
        "statistically_independent": False,
        "rank_definitions": program["frozen_diagnostics"]["rank_shape"],
        "spectrum_definitions": program["frozen_diagnostics"]["residual_spectrum"],
        "domains": domains,
        "root_cause_summary": root_cause_summary(domains),
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"V82A refuses existing partial output: {temporary}")
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
                "decision_digest_sha256": report["decision_digest_sha256"],
                "out": str(args.out.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
