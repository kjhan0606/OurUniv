#!/usr/bin/env python
"""Run consumed-only leave-one-query-out Gaussian residual controls."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import hong2021_v75_rank_coverage_exact_null as v75
import hong2021_v82a_consumed_autopsy as v82a
from hong2021_residual_evaluate import SpectralBinner
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = "hong2021-v82b-consumed-loo-gaussian-controls-program-v1"
PROGRAM_STATUS = "frozen_and_pushed_before_V82B_generated_draw_or_report"
REPORT_SCHEMA = "hong2021-v82b-consumed-loo-gaussian-controls-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
BASELINE_ORDER = ("stationary", "quartile_scale")
QUERIES = 32
MEMBERS = 16
GRID = 64
VOXEL_MPC_H = 0.3125
RELABELINGS = 99_999
INDIVIDUAL_REFERENCE_THRESHOLD = 1.0 / 240.0


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
        raise ValueError("V82B program freeze commit cannot be resolved")
    return commit


def _bound_artifact(row: dict[str, Any], label: str) -> Path:
    if set(row) != {"path", "sha256"}:
        raise ValueError(f"V82B {label} artifact keys differ")
    path = Path(str(row["path"])).resolve()
    if len(str(row["sha256"])) != 64 or sha256_file(path) != row["sha256"]:
        raise ValueError(f"V82B {label} artifact hash differs")
    return path


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program = strict_json(path.resolve())
    boundary = program.get("scope_boundary", {})
    authorization = program.get("authorization", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("engineering_only") is not True
        or program.get("statistically_independent") is not False
        or boundary.get("leave_one_query_out_truth_fit") is not True
        or boundary.get("train_neural_network") is not False
        or boundary.get("open_new_independent_data") is not False
        or boundary.get("modify_or_stop_RAMSES") is not False
        or authorization.get("user_approved_automatic_consumed_Gaussian_control") is not True
    ):
        raise ValueError("V82B scope boundary differs")
    for label, row in program["implementation_sources"].items():
        if sha256_file(resolve_path(repo, row["path"])) != row["sha256"]:
            raise ValueError(f"V82B implementation source differs: {label}")
    for label, row in program["support_artifacts"].items():
        _bound_artifact(row, f"support/{label}")
    if set(program["input_artifacts"]) != set(DOMAIN_ORDER):
        raise ValueError("V82B input domains differ")
    seen: set[Path] = set()
    for domain in DOMAIN_ORDER:
        if set(program["input_artifacts"][domain]) != {"consumed_ensemble"}:
            raise ValueError(f"V82B {domain} input artifact set differs")
        artifact = _bound_artifact(
            program["input_artifacts"][domain]["consumed_ensemble"], domain
        )
        if artifact in seen:
            raise ValueError("V82B input path reused")
        seen.add(artifact)
    return program


@dataclass
class PeriodicShells:
    grid: int
    voxel_mpc_h: float

    def __post_init__(self) -> None:
        kxy = 2.0 * np.pi * np.fft.fftfreq(self.grid, d=self.voxel_mpc_h)
        kz = 2.0 * np.pi * np.fft.rfftfreq(self.grid, d=self.voxel_mpc_h)
        magnitude = np.sqrt(
            kxy[:, None, None] ** 2
            + kxy[None, :, None] ** 2
            + kz[None, None, :] ** 2
        )
        fundamental = 2.0 * np.pi / (self.grid * self.voxel_mpc_h)
        self.shell = np.rint(magnitude / fundamental).astype(np.int32)
        self.shell_count = int(self.shell.max()) + 1
        self.count = np.bincount(
            self.shell.ravel(), minlength=self.shell_count
        ).astype(np.int64)
        self.voxels = self.grid**3

    def radial_sum(self, value: np.ndarray) -> np.ndarray:
        flat = np.asarray(value).reshape((-1, self.shell.size))
        return np.asarray(
            [
                np.bincount(
                    self.shell.ravel(), weights=row, minlength=self.shell_count
                )
                for row in flat
            ]
        )


def centered_truth_residuals(truth: np.ndarray, mean: np.ndarray) -> np.ndarray:
    residual = np.asarray(truth, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    return residual


def loo_periodic_mode_power(
    residuals: np.ndarray, shells: PeriodicShells
) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.fft.rfftn(residuals, axes=(-3, -2, -1))
    sums = shells.radial_sum(np.abs(transformed) ** 2)
    total = sums.sum(axis=0)
    loo = (total[None] - sums) / ((len(residuals) - 1) * shells.count[None])
    loo[:, 0] = 0.0
    if not np.isfinite(loo).all() or np.any(loo < 0.0):
        raise ValueError("V82B LOO mode power is invalid")
    return loo, sums


def quartile_sumsquares(
    residuals: np.ndarray, means: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((len(residuals), 4), dtype=np.float64)
    counts = np.zeros((len(residuals), 4), dtype=np.int64)
    for query in range(len(residuals)):
        labels = v82a.equal_count_strata(means[query])
        for label in range(4):
            selected = labels == label
            sums[query, label] = np.square(residuals[query][selected]).sum()
            counts[query, label] = np.count_nonzero(selected)
    return sums, counts


def loo_quartile_scales(
    sums: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total_sum = sums.sum(axis=0)
    total_count = counts.sum(axis=0)
    loo_variance = (total_sum[None] - sums) / (total_count[None] - counts)
    global_variance = np.sum(total_sum[None] - sums, axis=1) / np.sum(
        total_count[None] - counts, axis=1
    )
    scale = np.sqrt(loo_variance / global_variance[:, None])
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("V82B LOO quartile scale is invalid")
    return scale, loo_variance


def gaussian_residual_draws(
    generator: np.random.Generator,
    mode_power: np.ndarray,
    shells: PeriodicShells,
    members: int = MEMBERS,
) -> np.ndarray:
    noise = generator.standard_normal(
        (members, shells.grid, shells.grid, shells.grid), dtype=np.float64
    )
    noise -= noise.mean(axis=(-3, -2, -1), keepdims=True)
    transformed = np.fft.rfftn(noise, axes=(-3, -2, -1))
    multiplier = np.sqrt(mode_power[shells.shell] / shells.voxels)
    transformed *= multiplier[None]
    transformed[:, 0, 0, 0] = 0.0
    residual = np.fft.irfftn(
        transformed,
        s=(shells.grid, shells.grid, shells.grid),
        axes=(-3, -2, -1),
    ).real
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    return residual


def apply_quartile_scale(
    residual: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    labels = v82a.equal_count_strata(mean)
    output = np.asarray(residual, dtype=np.float64) * scale[labels][None]
    output -= output.mean(axis=(-3, -2, -1), keepdims=True)
    return output


def moment_summary(values: np.ndarray) -> dict[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = flattened.mean()
    centered = flattened - mean
    variance = np.mean(np.square(centered))
    if variance <= 0.0:
        raise ValueError("V82B moment variance is nonpositive")
    return {
        "mean": float(mean),
        "standard_deviation": float(np.sqrt(variance)),
        "skewness": float(np.mean(centered**3) / variance**1.5),
        "excess_kurtosis": float(np.mean(centered**4) / variance**2 - 3.0),
    }


def exact_rank_null(
    tables: list[dict[str, Any]], program_sha: str, domain: str, baseline: str
) -> dict[str, Any]:
    table = v75.stack_query_tables(tables)
    observed = v75.observed_statistics(table, label=MEMBERS)
    seed = v75.derived_seed(program_sha, domain, baseline, "relabel")
    labels = np.random.default_rng(seed).integers(
        0, MEMBERS + 1, size=(RELABELINGS, QUERIES), dtype=np.int8
    )
    null = v75.assignment_statistics(table, labels)
    p_values = v75.conditional_p_values(observed, null)
    return {
        "observed": observed,
        "conditional_p_values": {
            "rank_tv": p_values["rank_tv"],
            "coverage_deviation": p_values["coverage_deviation"],
        },
        "random_relabelings": RELABELINGS,
        "relabel_seed": seed,
        "adjacent_tie_fraction": table["adjacent_tie_fraction"],
        "null_quantiles_2p5_50_97p5": {
            key: np.quantile(null[key], [0.025, 0.5, 0.975]).tolist()
            for key in ("rank_tv", "coverage_deviation")
        },
    }


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
    domain: str, path: Path, seed: int, program_sha: str
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if (
            tuple(handle["truth"].shape) != (QUERIES, 1, GRID, GRID, GRID)
            or tuple(handle["conditional_mean"].shape)
            != (QUERIES, 1, GRID, GRID, GRID)
        ):
            raise ValueError(f"V82B {domain} input shape differs")
        truth = np.asarray(handle["truth"][:, 0], dtype=np.float64)
        mean = np.asarray(handle["conditional_mean"][:, 0], dtype=np.float64)
        source_indices = handle["source_index"][:].astype(np.int64).tolist()
    residuals = centered_truth_residuals(truth, mean)
    shells = PeriodicShells(GRID, VOXEL_MPC_H)
    loo_power, periodic_power_sums = loo_periodic_mode_power(residuals, shells)
    quartile_sum, quartile_count = quartile_sumsquares(residuals, mean)
    scales, loo_quartile_variance = loo_quartile_scales(quartile_sum, quartile_count)
    aggregate_hist = {
        baseline: np.zeros(MEMBERS + 1, dtype=np.int64) for baseline in BASELINE_ORDER
    }
    aggregate_strata = {
        baseline: [np.zeros(MEMBERS + 1, dtype=np.int64) for _ in range(4)]
        for baseline in BASELINE_ORDER
    }
    label_tables: dict[str, list[dict[str, Any]]] = {
        baseline: [] for baseline in BASELINE_ORDER
    }
    query_rows = []
    binner = SpectralBinner(GRID, VOXEL_MPC_H)
    generator = np.random.default_rng(seed)
    for query in range(QUERIES):
        stationary = gaussian_residual_draws(generator, loo_power[query], shells)
        quartile = apply_quartile_scale(stationary, mean[query], scales[query])
        generated = {"stationary": stationary, "quartile_scale": quartile}
        strata = v82a.equal_count_strata(mean[query])
        current: dict[str, Any] = {
            "query_position": query,
            "source_index": source_indices[query],
            "LOO_quartile_scale_low_to_high": scales[query].tolist(),
        }
        for baseline in BASELINE_ORDER:
            current[baseline] = v82a.rank_and_coverage(
                generated[baseline], residuals[query], strata
            )
            aggregate_hist[baseline] += np.asarray(
                current[baseline]["shape"]["histogram"]
            )
            for label in range(4):
                aggregate_strata[baseline][label] += np.asarray(
                    current[baseline]["conditional_mean_quartiles"][label]["histogram"]
                )
            tie_seed = v75.derived_seed(
                program_sha, domain, baseline, str(query), "tie_priority"
            )
            label_tables[baseline].append(
                v75.query_label_table(
                    np.concatenate((generated[baseline], residuals[query][None])),
                    np.random.default_rng(tie_seed),
                )
            )
        spectrum = v82a.spectral_query(
            binner, stationary, quartile, residuals[query]
        )
        current["spectrum"] = {
            "stationary": spectrum["candidate"],
            "quartile_scale": spectrum["control"],
            "paired_stationary_quartile_member_phase_cosine": spectrum[
                "paired_candidate_control_member_phase_cosine"
            ],
        }
        query_rows.append(current)
        print(f"[v82b] {domain} query {query + 1}/{QUERIES}", flush=True)
    arms = {}
    for baseline in BASELINE_ORDER:
        arms[baseline] = {
            "aggregate_rank_shape": v82a.rank_shape(aggregate_hist[baseline]),
            "aggregate_rank_shape_by_conditional_mean_quartile": [
                v82a.rank_shape(value) for value in aggregate_strata[baseline]
            ],
            "per_query_rank_TV": _summary(
                [
                    {"value": row[baseline]["shape"]["total_variation_from_uniform"]}
                    for row in query_rows
                ]
            )["value"],
            "per_query_coverage": _summary(
                [
                    {
                        "coverage68_minus_expected": row[baseline][
                            "coverage68_minus_expected"
                        ],
                        "coverage95_minus_expected": row[baseline][
                            "coverage95_minus_expected"
                        ],
                    }
                    for row in query_rows
                ]
            ),
            "spectrum": {
                metric: _summary(
                    [row["spectrum"][baseline][metric] for row in query_rows]
                )
                for metric in (
                    "member_mean_residual_power_over_truth",
                    "ensemble_mean_truth_phase_cosine",
                    "member_truth_phase_cosine_mean",
                )
            },
            "exact_rank_coverage_null": exact_rank_null(
                label_tables[baseline], program_sha, domain, baseline
            ),
        }
    truth_moments_by_quartile = []
    for label in range(4):
        values = []
        for query in range(QUERIES):
            labels = v82a.equal_count_strata(mean[query])
            values.append(residuals[query][labels == label])
        truth_moments_by_quartile.append(moment_summary(np.concatenate(values)))
    return {
        "source_indices": source_indices,
        "random_seed_shared_by_both_controls": seed,
        "periodic_shell_count": shells.shell_count,
        "LOO_periodic_power_fit_finite": bool(np.isfinite(loo_power).all()),
        "LOO_truth_periodic_power_sum_sha256": canonical_digest(
            {"values": periodic_power_sums.tolist()}
        ),
        "LOO_quartile_variance_sha256": canonical_digest(
            {"values": loo_quartile_variance.tolist()}
        ),
        "LOO_quartile_scale_summary": _summary(
            [
                {f"quartile_{label}": float(scales[query, label]) for label in range(4)}
                for query in range(QUERIES)
            ]
        ),
        "truth_residual_moments_by_conditional_mean_quartile": truth_moments_by_quartile,
        "baselines": arms,
        "paired_stationary_quartile_member_phase_cosine": _summary(
            [
                row["spectrum"]["paired_stationary_quartile_member_phase_cosine"]
                for row in query_rows
            ]
        ),
        "per_query": query_rows,
    }


def decide(domains: dict[str, Any]) -> dict[str, Any]:
    compatibility = {}
    for baseline in BASELINE_ORDER:
        values = []
        rows = {}
        for domain in DOMAIN_ORDER:
            exact = domains[domain]["baselines"][baseline][
                "exact_rank_coverage_null"
            ]
            p_values = exact["conditional_p_values"]
            values.extend((p_values["rank_tv"], p_values["coverage_deviation"]))
            rows[domain] = p_values
        compatibility[baseline] = {
            "domains": rows,
            "six_individual_p_values": values,
            "all_six_above_V79_reference_threshold": bool(
                all(value > INDIVIDUAL_REFERENCE_THRESHOLD for value in values)
            ),
        }
    stationary = compatibility["stationary"][
        "all_six_above_V79_reference_threshold"
    ]
    quartile = compatibility["quartile_scale"][
        "all_six_above_V79_reference_threshold"
    ]
    if stationary:
        branch = "stationary_Gaussian_sufficient_for_rank_control_no_flow_yet"
        flow = False
    elif quartile:
        branch = "conditional_scale_required_but_Gaussian_sufficient_no_flow_yet"
        flow = False
    else:
        branch = "both_Gaussian_controls_fail_conditional_nonGaussian_model_warranted"
        flow = True
    return {
        "compatibility": compatibility,
        "individual_reference_threshold": INDIVIDUAL_REFERENCE_THRESHOLD,
        "branch": branch,
        "conditional_flow_or_equivalent_nonGaussian_model_warranted": flow,
        "engineering_only_not_validation": True,
        "automatic_training_authorized": False,
    }


def run(program_path: Path, repo: Path, output_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program_path = program_path.resolve()
    output_path = output_path.resolve()
    program = load_program(program_path, repo)
    if output_path != Path(program["outputs"]["report"]).resolve():
        raise ValueError("V82B output binding differs")
    commit, clean = git_state(repo)
    freeze_commit = program_freeze_commit(program_path, repo)
    if not clean or not _is_ancestor(repo, freeze_commit, commit):
        raise ValueError("V82B requires a clean descendant of its freeze commit")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise ValueError("V82B must execute on lageunha")
    if output_path.exists() or output_path.with_suffix(".json.partial").exists():
        raise FileExistsError("V82B refuses existing output")
    program_sha = sha256_file(program_path)
    domains = {
        domain: inspect_domain(
            domain,
            Path(program["input_artifacts"][domain]["consumed_ensemble"]["path"]),
            int(program["fixed_random_seeds"][domain]),
            program_sha,
        )
        for domain in DOMAIN_ORDER
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_consumed_only_LOO_Gaussian_controls",
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
    result = run(args.program, args.repo, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "decision_digest_sha256": result["decision_digest_sha256"],
                "out": str(args.out.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
