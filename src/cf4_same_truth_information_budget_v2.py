#!/usr/bin/env python3
"""Numerically corrected runner for the frozen CF4 information budget.

Version 1 used a normalized data-space CG with 500 iterations.  The finite
low-noise scenarios exhausted that limit before satisfying the unchanged
residual gate.  This correction changes no geometry, scenario, random stream,
draw, estimand, threshold, or science firewall.  It solves the equivalent
unnormalized covariance system, carries each draw's solution from noise scale
1 to 0.3 to 0.1 as a warm start, estimates the diagonal with 16 probes from
the same preconditioner stream, and permits at most 4000 CG iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_same_truth_information_budget as v1


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SCHEMA = "ouruniv-cf4-same-truth-information-budget-program-v2"
PRECONDITIONER_PROBES = 16
CG_MAXITER = 4000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise v1.InformationError("v2 repository binding escapes the repository")
    return path


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise v1.InformationError("cannot parse v2 information-budget program") from exc
    if program.get("schema") != PROGRAM_SCHEMA:
        raise v1.InformationError("v2 information-budget program schema mismatch")
    authorization = program.get("authorization", {})
    required_true = {
        "same_truth_information_budget_audit",
        "assistant_solver_correction_v2",
        "single_member_technical_pilot",
        "production_after_pilot_pass",
        "Slurm_member_array_submission",
        "Slurm_dependent_aggregation_submission",
        "GPFS_declared_read_write",
    }
    required_false = {
        "truth_array_generation_or_deserialization",
        "likelihood_datum_consumed_by_inference",
        "population_generator_retuning",
        "untouched_256_mock_validation",
        "resolution_increase",
        "ML_training",
        "frontier_promotion",
        "IC_PM_HOP_RAMSES",
        "automatic_retry",
        "automatic_follow_on_after_aggregate",
    }
    if any(authorization.get(key) is not True for key in required_true):
        raise v1.InformationError("required v2 correction authorization is absent")
    if any(authorization.get(key) is not False for key in required_false):
        raise v1.InformationError("v2 correction science firewall changed")
    design = program.get("design", {})
    if design.get("mock_count") != v1.base.MOCK_COUNT:
        raise v1.InformationError("v2 information mock count changed")
    if design.get("posterior_draw_count") != v1.base.POSTERIOR_DRAW_COUNT:
        raise v1.InformationError("v2 posterior draw count changed")
    if tuple(design.get("scenario_order", ())) != v1.SCENARIOS:
        raise v1.InformationError("v2 scenario order changed")
    if design.get("known_nuisance_noise_standard_deviation_scales") != v1.NOISE_SCALES:
        raise v1.InformationError("v2 finite noise scales changed")
    if design.get("new_truth_seed_count") != 0 or design.get("new_random_seed_count") != 0:
        raise v1.InformationError("v2 correction introduced a new random stream")
    solver = program.get("solver_correction", {})
    exact_solver = {
        "failed_v1_member_array_job_id": 329405,
        "failed_v1_aggregate_job_id": 329406,
        "system": "unnormalized_data_covariance_AAT_plus_N",
        "noise_scale_continuation_order": [1.0, 0.3, 0.1],
        "warm_start_previous_noise_scale": True,
        "preconditioner_probe_count": PRECONDITIONER_PROBES,
        "CG_tolerance": v1.base.fixed.CG_TOL,
        "CG_max_iterations": CG_MAXITER,
        "science_or_random_stream_changed": False,
    }
    if solver != exact_solver:
        raise v1.InformationError("v2 solver correction contract changed")
    for collection in ("repository_bindings", "source_bindings"):
        records = program.get(collection, {})
        if not isinstance(records, Mapping) or not records:
            raise v1.InformationError(f"v2 {collection} is absent")
        for record in records.values():
            if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
                raise v1.InformationError(f"v2 {collection} record is not exact")
            source = _repo_path(str(record["path"]))
            if sha256_file(source) != record["sha256"]:
                raise v1.InformationError(f"v2 SHA256 mismatch: {record['path']}")
    for label in ("prior_ablation", "completed_D"):
        completed = program.get(label, {})
        aggregate = Path(str(completed["aggregate_directory"]))
        if sha256_file(aggregate / "result.json") != completed["aggregate_result_sha256"]:
            raise v1.InformationError(f"v2 {label} aggregate result binding changed")
        if sha256_file(aggregate / "metrics.npz") != completed["aggregate_metrics_sha256"]:
            raise v1.InformationError(f"v2 {label} aggregate metrics binding changed")
    execution = program.get("execution", {})
    if execution.get("technical_pilot_index") != 0:
        raise v1.InformationError("v2 technical pilot index changed")
    if execution.get("production_member_array") != "0-63%8":
        raise v1.InformationError("v2 production array changed")
    if execution.get("target_0p3_cMpc_h_reached") is not False:
        raise v1.InformationError("v2 makes a forbidden resolution claim")
    return program, hashlib.sha256(payload).hexdigest()


def solve_known_covariances_v2(
    *,
    positions: np.ndarray,
    directions: np.ndarray,
    variance: np.ndarray,
    train: np.ndarray,
    args: argparse.Namespace,
    seeds: Mapping[str, object],
    transfer: np.ndarray,
    modes: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Equivalent unnormalized covariance solve with scale continuation."""

    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    forward_all, adjoint_all, _, dtype = v1.base.linear.build_forward(
        positions, directions, args
    )
    train_index = np.flatnonzero(np.asarray(train, dtype=bool))
    if train_index.size == 0 or train_index.size == train.size:
        raise v1.InformationError("v2 information geometry lacks train or holdout rows")
    train_jax = jnp.asarray(train_index)
    A_train = jax.jit(lambda field: forward_all(field)[train_jax])

    @jax.jit
    def AT_train(values):
        expanded = jnp.zeros(train.size, dtype=dtype)
        return adjoint_all(expanded.at[train_jax].set(values))

    standard_deviation = np.sqrt(np.asarray(variance, dtype=np.float64)[train_index])
    if np.any(standard_deviation <= 0.0) or not np.all(np.isfinite(standard_deviation)):
        raise v1.InformationError("v2 information geometry noise scale is invalid")
    adjoint_rng = np.random.default_rng(int(seeds["adjoint"]))
    sx = jnp.asarray(adjoint_rng.standard_normal((v1.base.fixed.N,) * 3), dtype=dtype)
    dy = jnp.asarray(adjoint_rng.standard_normal(train_index.size), dtype=dtype)
    lhs = float(jnp.vdot(A_train(sx), dy))
    rhs = float(jnp.vdot(sx, AT_train(dy)))
    adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)

    probe_rng = np.random.default_rng(int(seeds["preconditioner"]))
    signal_diagonal = np.zeros(train_index.size, dtype=np.float64)
    for _ in range(PRECONDITIONER_PROBES):
        probe = jnp.asarray(
            probe_rng.standard_normal((v1.base.fixed.N,) * 3), dtype=dtype
        )
        signal_diagonal += np.asarray(A_train(probe), dtype=np.float64) ** 2
    signal_diagonal /= PRECONDITIONER_PROBES

    operators = {}
    for scenario in v1.NEW_SCENARIOS:
        noise_scale = v1.NOISE_SCALES[scenario]
        noise_variance = jnp.asarray(
            (noise_scale * standard_deviation) ** 2, dtype=dtype
        )
        diagonal = jnp.asarray(
            np.asarray(noise_variance, dtype=np.float64) + signal_diagonal,
            dtype=dtype,
        )

        @jax.jit
        def covariance(values, noise_variance=noise_variance):
            return noise_variance * values + A_train(AT_train(values))

        precondition = jax.jit(lambda values, diagonal=diagonal: values / diagonal)
        operators[scenario] = (covariance, precondition)

    flat = np.asarray(modes["flat"], dtype=np.int64)
    theta_keep = np.asarray(modes["theta_keep"], dtype=bool)
    transfer_modes = np.asarray(transfer, dtype=np.float64).ravel()[flat]
    scenario_modes = {scenario: [] for scenario in v1.NEW_SCENARIOS}
    residuals = {scenario: [] for scenario in v1.NEW_SCENARIOS}
    for seed in seeds["posterior_draws"]:
        rng = np.random.default_rng(int(seed))
        xi = jnp.asarray(rng.standard_normal((v1.base.fixed.N,) * 3), dtype=dtype)
        rng.standard_normal(4)  # Preserve completed arm-A epsilon pairing.
        epsilon = jnp.asarray(rng.standard_normal(train_index.size), dtype=dtype)
        raw_signal = A_train(xi)
        beta_start = None
        for scenario in v1.NEW_SCENARIOS:
            noise_scale = v1.NOISE_SCALES[scenario]
            covariance, precondition = operators[scenario]
            rhs_values = -raw_signal - jnp.asarray(
                noise_scale * standard_deviation, dtype=dtype
            ) * epsilon
            beta, _ = cg(
                covariance,
                rhs_values,
                x0=beta_start,
                tol=v1.base.fixed.CG_TOL,
                atol=0.0,
                maxiter=CG_MAXITER,
                M=precondition,
            )
            beta.block_until_ready()
            residual = rhs_values - covariance(beta)
            relative = float(
                jnp.linalg.norm(residual)
                / jnp.maximum(jnp.linalg.norm(rhs_values), 1.0e-30)
            )
            posterior = np.asarray(xi + AT_train(beta), dtype=np.float64)
            white_modes = np.fft.fftn(posterior, norm="ortho").ravel()[flat]
            scenario_modes[scenario].append(white_modes * transfer_modes)
            residuals[scenario].append(relative)
            beta_start = beta

    results = {}
    arrays = {}
    for scenario in v1.NEW_SCENARIOS:
        draws_delta = np.stack(scenario_modes[scenario])
        scenario_residuals = residuals[scenario]
        numerical = {
            "adjoint_relative_error": adjoint_error,
            "adjoint_max_inclusive": v1.base.fixed.ADJOINT_MAX,
            "adjoint_pass": bool(adjoint_error <= v1.base.fixed.ADJOINT_MAX),
            "all_16_sample_cg_relative_residuals": scenario_residuals,
            "sample_cg_max_inclusive": v1.base.fixed.CG_RESIDUAL_MAX,
            "all_16_sample_cg_pass": bool(
                np.all(np.asarray(scenario_residuals) <= v1.base.fixed.CG_RESIDUAL_MAX)
            ),
            "posterior_mean_solved": False,
            "solver": "unnormalized_data_covariance_with_scale_continuation",
            "preconditioner_probe_count": PRECONDITIONER_PROBES,
            "CG_max_iterations": CG_MAXITER,
        }
        numerical["all_pass"] = bool(
            numerical["adjoint_pass"] and numerical["all_16_sample_cg_pass"]
        )
        if not numerical["all_pass"]:
            raise v1.InformationError(
                f"{scenario} v2 covariance numerical gate failed; "
                f"maximum residual={max(scenario_residuals):.9e}"
            )
        arrays[f"scenario_{scenario}_posterior_draws_delta_modes"] = draws_delta
        arrays[f"scenario_{scenario}_posterior_draws_theta_modes"] = draws_delta[
            :, theta_keep
        ]
        results[scenario] = {
            "nuisance": "known_and_subtracted",
            "noise_standard_deviation_scale": v1.NOISE_SCALES[scenario],
            "noise_variance_scale": v1.NOISE_SCALES[scenario] ** 2,
            "numerical_gates": numerical,
        }
    return results, arrays


def solve_member(
    program: Mapping[str, object],
    program_sha256: str,
    mock_index: int,
    implementation_commit: str,
):
    original = v1.solve_known_covariances
    v1.solve_known_covariances = solve_known_covariances_v2
    try:
        result, arrays = v1.solve_member(
            program, program_sha256, mock_index, implementation_commit
        )
    finally:
        v1.solve_known_covariances = original
    result["implementation_source_sha256"] = sha256_file(__file__)
    result["solver_correction_v2"] = program["solver_correction"]
    return result, arrays


def aggregate_members(
    program: Mapping[str, object],
    program_sha256: str,
    members_root: str | Path,
    member_implementation_commit: str,
    aggregation_runtime_commit: str,
):
    result, arrays = v1.aggregate_members(
        program,
        program_sha256,
        members_root,
        member_implementation_commit,
        aggregation_runtime_commit,
    )
    result["implementation_source_sha256"] = sha256_file(__file__)
    result["solver_correction_v2"] = program["solver_correction"]
    return result, arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    member = sub.add_parser("run-member")
    member.add_argument("--program", required=True, type=Path)
    member.add_argument("--mock-index", required=True, type=int)
    member.add_argument("--output", required=True, type=Path)
    member.add_argument("--implementation-commit", required=True)
    validate_member = sub.add_parser("validate-member")
    validate_member.add_argument("--directory", required=True, type=Path)
    validate_member.add_argument("--expected-index", type=int)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--program", required=True, type=Path)
    aggregate.add_argument("--members-root", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--member-implementation-commit", required=True)
    aggregate.add_argument("--aggregation-runtime-commit", required=True)
    validate_aggregate = sub.add_parser("validate-aggregate")
    validate_aggregate.add_argument("--directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-member":
            program, program_sha = load_program(args.program)
            result, arrays = solve_member(
                program, program_sha, args.mock_index, args.implementation_commit
            )
            v1.publish_directory(args.output, result, arrays, kind="member")
            report = v1.validate_member(args.output, args.mock_index)
        elif args.command == "validate-member":
            report = v1.validate_member(args.directory, args.expected_index)
        elif args.command == "aggregate":
            program, program_sha = load_program(args.program)
            result, arrays = aggregate_members(
                program,
                program_sha,
                args.members_root,
                args.member_implementation_commit,
                args.aggregation_runtime_commit,
            )
            v1.publish_directory(args.output, result, arrays, kind="aggregate")
            report = v1.validate_aggregate(args.output)
        else:
            report = v1.validate_aggregate(args.directory)
    except (OSError, ValueError, v1.InformationError, v1.base.CalibrationError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
