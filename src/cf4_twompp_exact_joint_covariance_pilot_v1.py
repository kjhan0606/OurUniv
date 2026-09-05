#!/usr/bin/env python3
"""Exact N32 velocity-plus-2M++ covariance pilot for one development geometry.

This covariance-only calculation reconstructs the exact marginalized arm-A
velocity precision for completed development geometry 0 and adds the frozen
2M++ Gaussianized-Poisson density Fisher operator.  It never materializes a
32768-square covariance, consumes a velocity or galaxy-position likelihood
datum, generates a truth field, opens locked validation, or infers a field.

The old binwise isotropic velocity spectrum is used only as a CG
preconditioner and comparison.  Every reported trace is obtained from the
exact anisotropic matrix-free velocity operator.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_population_calibration as velocity_base
import cf4_twompp_joint_information_budget_pilot_v1 as joint_v1


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SCHEMA = "ouruniv-cf4-twompp-exact-joint-covariance-pilot-program-v1"
RESULT_SCHEMA = "ouruniv-cf4-twompp-exact-joint-covariance-pilot-result-v1"
MANIFEST_SCHEMA = "ouruniv-cf4-twompp-exact-joint-covariance-pilot-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-twompp-exact-joint-covariance-pilot-complete-v1"
EXPECTED_FILES = {"metrics.npz", "result.json", "manifest.json", "COMPLETE"}
SCENARIOS = (
    "velocity_only_exact_marginalized_nuisance",
    "known_selection_reference_bias",
    "normalization_marginalized_reference_bias",
)
DOMAINS = ("delta", "theta")


class ExactPilotError(ValueError):
    """The exact one-geometry covariance contract failed closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise ExactPilotError(f"bound {label} record is not exact")
    path = Path(str(binding["path"]))
    if not path.is_file():
        raise ExactPilotError(f"bound {label} is absent: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise ExactPilotError(f"bound {label} size changed")
    if sha256_file(path) != str(binding["sha256"]):
        raise ExactPilotError(f"bound {label} hash changed")
    return path


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    payload = Path(path).read_bytes()
    program = json.loads(payload)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise ExactPilotError("unexpected exact joint-covariance program schema")
    authorization = program.get("authorization", {})
    required_true = {
        "single_geometry_exact_joint_covariance_pilot",
        "single_Slurm_submission",
        "completed_development_truth_geometry_deserialization",
        "GPFS_read",
        "GPFS_write",
    }
    required_false = {
        "truth_field_array_generation_or_deserialization",
        "likelihood_datum_consumed_by_inference",
        "observational_CF4_field_inference",
        "galaxy_positions_consumed_as_field_likelihood_datum",
        "new_truth_or_validation_seed",
        "untouched_256_mock_validation",
        "parent_posterior_promotion",
        "resolution_increase",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    }
    if any(authorization.get(key) is not True for key in required_true):
        raise ExactPilotError("required exact-pilot authorization is absent")
    if any(authorization.get(key) is not False for key in required_false):
        raise ExactPilotError("exact-pilot science firewall changed")
    design = program.get("design", {})
    if design.get("geometry_index") != 0:
        raise ExactPilotError("v1 is frozen to development geometry 0")
    if tuple(design.get("scenario_order", ())) != SCENARIOS:
        raise ExactPilotError("exact-pilot scenario order changed")
    if design.get("grid_N") != 32 or design.get("box_size_cMpc_h") != 384.0:
        raise ExactPilotError("exact-pilot N32 geometry changed")
    if design.get("trace_probe_count") != 32:
        raise ExactPilotError("exact-pilot trace-probe count changed")
    gates = design.get("information_gates", {})
    expected_gates = {
        "material": {
            "information_point_min_inclusive": 0.5,
            "information_numerical_95_lower_min_inclusive": 0.5,
            "expected_correlation_r_min_inclusive": 0.7,
            "expected_residual_power_ratio_max_inclusive": 0.5,
        },
        "strong_stretch": {
            "information_point_min_inclusive": 0.8,
            "information_numerical_95_lower_min_inclusive": 0.8,
            "expected_correlation_r_min_inclusive": math.sqrt(0.8),
            "expected_residual_power_ratio_max_inclusive": 0.2,
        },
    }
    if gates != expected_gates:
        raise ExactPilotError("exact-pilot information gates changed")
    for label, binding in program.get("bindings", {}).items():
        _verify_binding(binding, label)
    if set(program.get("bindings", {})) != {
        "implementation",
        "gate_audit",
        "isotropic_joint_program",
        "isotropic_joint_result_record",
        "isotropic_joint_implementation",
        "population_calibration_implementation",
        "fixed_field_kernels",
        "bin_manifest",
        "frozen_CF4_catalog",
        "frozen_2Mpp_selection",
        "completed_geometry0_fields",
        "completed_geometry0_result",
        "completed_geometry_aggregate_result",
        "velocity_isotropic_preconditioner_metrics",
    }:
        raise ExactPilotError("exact-pilot binding set changed")
    execution = program.get("execution", {})
    if execution.get("requested_memory_MiB", 0) < math.ceil(
        1.2 * execution.get("expected_peak_memory_MiB", math.inf)
    ):
        raise ExactPilotError("requested memory lacks 20 percent expected headroom")
    if execution.get("manual_syntax_or_syn101_numerical_execution_allowed") is not False:
        raise ExactPilotError("manual syntax/syn101 execution was enabled")
    return program, hashlib.sha256(payload).hexdigest()


def nuisance_woodbury_small_inverse(
    normalized_templates: np.ndarray, nuisance_variance: np.ndarray
) -> np.ndarray:
    """Return (Lambda^-1 + B^T B)^-1 for R=I+B Lambda B^T."""

    templates = np.asarray(normalized_templates, dtype=np.float64)
    variance = np.asarray(nuisance_variance, dtype=np.float64)
    if templates.ndim != 2 or variance.shape != (templates.shape[1],):
        raise ExactPilotError("nuisance Woodbury shapes are inconsistent")
    if np.any(variance <= 0.0) or not np.all(np.isfinite(templates)):
        raise ExactPilotError("nuisance Woodbury inputs are invalid")
    matrix = np.diag(1.0 / variance) + templates.T @ templates
    inverse = np.linalg.inv(matrix)
    if not np.all(np.isfinite(inverse)):
        raise ExactPilotError("nuisance Woodbury inverse is invalid")
    return inverse


def apply_nuisance_r_inverse(
    values: np.ndarray,
    normalized_templates: np.ndarray,
    small_inverse: np.ndarray,
) -> np.ndarray:
    """Apply (I+B Lambda B^T)^-1 using a four-column Woodbury solve."""

    vector = np.asarray(values, dtype=np.float64)
    templates = np.asarray(normalized_templates, dtype=np.float64)
    inverse = np.asarray(small_inverse, dtype=np.float64)
    if vector.shape != (templates.shape[0],):
        raise ExactPilotError("nuisance R inverse vector has wrong shape")
    return vector - templates @ (inverse @ (templates.T @ vector))


def _load_geometry(program: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct exact arm-A geometry without loading any truth field."""

    fields_path = Path(str(program["bindings"]["completed_geometry0_fields"]["path"]))
    result_path = Path(str(program["bindings"]["completed_geometry0_result"]["path"]))
    aggregate_path = Path(
        str(program["bindings"]["completed_geometry_aggregate_result"]["path"])
    )
    member_result = json.loads(result_path.read_bytes())
    aggregate_result = json.loads(aggregate_path.read_bytes())
    if (
        member_result.get("mock_index") != 0
        or member_result.get("truth_seed") != 2026083000
        or member_result.get("implementation_commit")
        != "342bb7c77ac60801a303cb81311a30b3506c8f1d"
    ):
        raise ExactPilotError("completed geometry-0 member lineage changed")
    validation = {
        "status": "PASS",
        "mock_index": 0,
        "result_sha256": sha256_file(result_path),
        "fields_sha256": sha256_file(fields_path),
    }
    if aggregate_result.get("member_artifact_hashes", [None])[0] != validation:
        raise ExactPilotError("completed geometry-0 member is not aggregate-bound")
    required = {
        "mock_cz",
        "mock_observed_distance",
        "mock_distance_error_mag",
        "mock_direction",
        "mock_true_distance",
        "mock_true_position",
        "train_raw_idx",
        "holdout_raw_idx",
    }
    with np.load(fields_path, allow_pickle=False) as loaded:
        if not required.issubset(loaded.files):
            raise ExactPilotError("completed geometry artifact lacks required arrays")
        fields = {name: np.array(loaded[name]) for name in required}
    count = fields["mock_cz"].size
    if count != 22136:
        raise ExactPilotError("completed geometry catalog row count changed")
    catalog = {
        "H0": np.array(74.6, dtype=np.float64),
        "v3k": fields["mock_cz"],
        "dist": fields["mock_observed_distance"],
        "e_dm": fields["mock_distance_error_mag"],
        "nhat": fields["mock_direction"],
        "pgc": np.arange(1, count + 1, dtype=np.int64),
    }
    args = velocity_base.fixed.frozen_args(ROOT / "data/cf4_clean.npz")
    bgc = velocity_base.linear.prepare_bgc_catalog(args, catalog)
    selected = np.asarray(bgc["raw_idx"], dtype=np.int64)
    train = ~np.asarray(bgc["holdout"], dtype=bool)
    if not np.array_equal(selected[train], fields["train_raw_idx"]):
        raise ExactPilotError("reconstructed geometry-0 training rows changed")
    if not np.array_equal(selected[~train], fields["holdout_raw_idx"]):
        raise ExactPilotError("reconstructed geometry-0 holdout rows changed")
    direction = np.asarray(fields["mock_direction"], dtype=np.float64)[selected]
    true_distance = np.asarray(fields["mock_true_distance"], dtype=np.float64)[selected]
    positions = np.asarray(fields["mock_true_position"], dtype=np.float64)[selected]
    templates = np.column_stack((direction, -true_distance))
    variance = np.asarray(bgc["variance"], dtype=np.float64)
    q_std = np.asarray(bgc["q_std"], dtype=np.float64)
    if positions.shape != direction.shape or positions.shape[1:] != (3,):
        raise ExactPilotError("completed geometry-0 position shapes changed")
    if variance.shape != selected.shape or np.any(variance <= 0.0):
        raise ExactPilotError("completed geometry-0 variance is invalid")
    return {
        "args": args,
        "positions": positions,
        "directions": direction,
        "templates": templates,
        "variance": variance,
        "q_std": q_std,
        "train": train,
        "selected": selected,
        "train_raw_idx": fields["train_raw_idx"],
        "holdout_raw_idx": fields["holdout_raw_idx"],
    }


def _build_exact_velocity_precision(
    geometry: Mapping[str, Any], numerical: Mapping[str, Any]
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Build Q_v=I+A^T(I+B Lambda B^T)^-1 A for geometry 0."""

    import jax
    import jax.numpy as jnp

    args = geometry["args"]
    forward_all, adjoint_all, _, dtype = velocity_base.linear.build_forward(
        geometry["positions"], geometry["directions"], args
    )
    if jnp.asarray(0.0, dtype=dtype).dtype != jnp.float64:
        raise ExactPilotError("exact velocity operator requires JAX float64")
    train_index = np.flatnonzero(np.asarray(geometry["train"], dtype=bool))
    train_jax = jnp.asarray(train_index)
    A_train = jax.jit(lambda field: forward_all(field)[train_jax])

    @jax.jit
    def AT_train(values):
        expanded = jnp.zeros(geometry["selected"].size, dtype=dtype)
        return adjoint_all(expanded.at[train_jax].set(values))

    scale_np = np.sqrt(np.asarray(geometry["variance"])[train_index])
    normalized_templates_np = (
        np.asarray(geometry["templates"], dtype=np.float64)[train_index]
        / scale_np[:, None]
    )
    qvar_np = np.asarray(geometry["q_std"], dtype=np.float64) ** 2
    small_inverse_np = nuisance_woodbury_small_inverse(
        normalized_templates_np, qvar_np
    )
    scale = jnp.asarray(scale_np, dtype=dtype)
    templates = jnp.asarray(normalized_templates_np, dtype=dtype)
    small_inverse = jnp.asarray(small_inverse_np, dtype=dtype)
    An = jax.jit(lambda field: A_train(field) / scale)
    ATn = jax.jit(lambda values: AT_train(values / scale))

    @jax.jit
    def R_inverse(values):
        return values - templates @ (small_inverse @ (templates.T @ values))

    @jax.jit
    def Q_velocity(field):
        return field + ATn(R_inverse(An(field)))

    rng = np.random.default_rng(int(numerical["operator_test_seed"]))
    sx = jnp.asarray(rng.standard_normal((32, 32, 32)), dtype=dtype)
    dy = jnp.asarray(rng.standard_normal(train_index.size), dtype=dtype)
    lhs = float(jnp.vdot(An(sx), dy))
    rhs = float(jnp.vdot(sx, ATn(dy)))
    adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)
    row_test = rng.standard_normal(train_index.size)
    row_inverse = apply_nuisance_r_inverse(
        row_test, normalized_templates_np, small_inverse_np
    )
    restored = row_inverse + normalized_templates_np @ (
        qvar_np * (normalized_templates_np.T @ row_inverse)
    )
    woodbury_error = float(
        np.linalg.norm(restored - row_test) / max(np.linalg.norm(row_test), 1.0e-30)
    )
    if adjoint_error > float(numerical["adjoint_relative_error_max_inclusive"]):
        raise ExactPilotError("exact velocity A/AT adjoint gate failed")
    if woodbury_error > float(numerical["woodbury_relative_error_max_inclusive"]):
        raise ExactPilotError("exact velocity nuisance Woodbury gate failed")

    def action(field: np.ndarray) -> np.ndarray:
        value = jnp.asarray(np.asarray(field, dtype=np.float64), dtype=dtype)
        return np.asarray(Q_velocity(value), dtype=np.float64)

    report = {
        "selected_row_count": int(geometry["selected"].size),
        "training_row_count": int(train_index.size),
        "holdout_row_count": int(geometry["selected"].size - train_index.size),
        "nuisance_template_count": int(normalized_templates_np.shape[1]),
        "adjoint_relative_error": adjoint_error,
        "adjoint_relative_error_max_inclusive": float(
            numerical["adjoint_relative_error_max_inclusive"]
        ),
        "woodbury_relative_error": woodbury_error,
        "woodbury_relative_error_max_inclusive": float(
            numerical["woodbury_relative_error_max_inclusive"]
        ),
        "jax_dtype": str(jnp.asarray(0.0, dtype=dtype).dtype),
    }
    return action, report


def _load_selection(program: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(program["bindings"]["frozen_2Mpp_selection"]["path"]))
    with np.load(path, allow_pickle=False) as loaded:
        required = {"expected_counts", "population_counts", "reference_bias"}
        if not required.issubset(loaded.files):
            raise ExactPilotError("frozen 2M++ selection is incomplete")
        result = {name: np.array(loaded[name]) for name in required}
    if result["expected_counts"].shape != (6, 32, 32, 32):
        raise ExactPilotError("frozen 2M++ expected-count grid changed")
    if result["population_counts"].tolist() != [9617, 3463, 527, 15671, 6197, 1160]:
        raise ExactPilotError("frozen 2M++ population counts changed")
    if result["reference_bias"].tolist() != [1.3, 1.05, 0.85] * 2:
        raise ExactPilotError("frozen 2M++ reference bias changed")
    return result


def _load_mode_and_preconditioner(
    program: Mapping[str, Any], transfer: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    design = program["design"]
    manifest = Path(str(program["bindings"]["bin_manifest"]["path"]))
    plan, full_assignment = joint_v1._full_mode_assignment(
        velocity_base.fixed,
        manifest,
        int(design["grid_N"]),
        float(design["box_size_cMpc_h"]),
    )
    metrics_path = Path(
        str(program["bindings"]["velocity_isotropic_preconditioner_metrics"]["path"])
    )
    scenario = str(design["velocity_preconditioner_scenario"])
    with np.load(metrics_path, allow_pickle=False) as metrics:
        bin_ids = np.asarray(metrics["bin_ids"], dtype=np.int64)
        delta_bin_ids = np.asarray(metrics["delta_bin_ids"], dtype=np.int64)
        theta_bin_ids = np.asarray(metrics["theta_bin_ids"], dtype=np.int64)
        delta_trace = np.asarray(
            metrics[f"scenario_{scenario}_delta_posterior_prior_trace_fraction"],
            dtype=np.float64,
        )
        theta_trace = np.asarray(
            metrics[f"scenario_{scenario}_theta_posterior_prior_trace_fraction"],
            dtype=np.float64,
        )
    conservative_trace = np.ones(bin_ids.size, dtype=np.float64)
    for column, bin_id in enumerate(bin_ids):
        delta_match = np.flatnonzero(delta_bin_ids == bin_id)
        theta_match = np.flatnonzero(theta_bin_ids == bin_id)
        candidates = []
        if delta_match.size == 1:
            candidates.append(float(delta_trace[delta_match[0]]))
        if theta_match.size == 1:
            candidates.append(float(theta_trace[theta_match[0]]))
        if candidates:
            conservative_trace[column] = max(candidates)
    if np.any(conservative_trace <= 0.0) or not np.all(
        np.isfinite(conservative_trace)
    ):
        raise ExactPilotError("isotropic preconditioner trace is invalid")
    precision = np.ones_like(transfer, dtype=np.float64)
    for column, bin_id in enumerate(bin_ids):
        precision[full_assignment == bin_id] = 1.0 / min(
            conservative_trace[column], 1.0
        )
    metadata = {
        "bin_ids": bin_ids,
        "delta_bin_ids": delta_bin_ids,
        "theta_bin_ids": theta_bin_ids,
        "delta_trace": delta_trace,
        "theta_trace": theta_trace,
        "conservative_trace": conservative_trace,
        "precision": precision,
    }
    return plan, metadata


def _apply_transfer(field: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(
        np.fft.fftn(field, norm="ortho") * transfer, norm="ortho"
    ).real


def _density_precision_action(
    field: np.ndarray,
    transfer: np.ndarray,
    expected_counts: np.ndarray,
    bias: np.ndarray,
    marginalize_normalizations: bool,
) -> np.ndarray:
    delta = _apply_transfer(field, transfer)
    lambdas = np.asarray(expected_counts, dtype=np.float64)
    density_diagonal = np.sum(lambdas * bias[:, None, None, None] ** 2, axis=0)
    middle = density_diagonal * delta
    if marginalize_normalizations:
        normalization = lambdas.reshape(6, -1).sum(axis=1)
        for population in range(6):
            cross = lambdas[population] * bias[population]
            coefficient = float(
                np.vdot(cross, delta).real / normalization[population]
            )
            middle -= cross * coefficient
    return _apply_transfer(middle, transfer)


def _gate_mask(metrics: Mapping[str, Any], gate: Mapping[str, float]) -> np.ndarray:
    information = np.asarray(metrics["recovered_information_fraction"])
    lower = np.asarray(metrics["recovered_information_numerical_95_lower"])
    correlation = np.asarray(metrics["expected_correlation_r"])
    residual = np.asarray(metrics["expected_residual_power_ratio"])
    return (
        (information >= float(gate["information_point_min_inclusive"]))
        & (
            lower
            >= float(gate["information_numerical_95_lower_min_inclusive"])
        )
        & (
            correlation
            >= float(gate["expected_correlation_r_min_inclusive"])
        )
        & (
            residual
            <= float(gate["expected_residual_power_ratio_max_inclusive"])
        )
    )


def exact_joint_trace_spectrum(
    *,
    exact_velocity_precision: Callable[[np.ndarray], np.ndarray],
    transfer: np.ndarray,
    plan: Mapping[str, Any],
    domain_bin_ids: Mapping[str, np.ndarray],
    isotropic_preconditioner_precision: np.ndarray,
    expected_counts: np.ndarray,
    bias: np.ndarray,
    density_retention: float,
    marginalize_normalizations: bool,
    probe_count: int,
    probe_seed: int,
    operator_test_seed: int,
    cg_rtol: float,
    cg_maxiter: int,
    symmetry_max: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Estimate exact anisotropic posterior traces in delta and theta bands."""

    from scipy.sparse.linalg import LinearOperator, cg

    grid = transfer.shape[0]
    shape = (grid,) * 3
    size = grid**3
    flat = np.asarray(plan["flat_independent_field_indices"], dtype=np.int64)
    assignment = np.asarray(plan["mode_merged_bin_index"], dtype=np.int64)
    coordinates = np.column_stack(np.unravel_index(flat, shape))
    theta_keep = np.logical_and.reduce(
        [coordinates[:, axis] != grid // 2 for axis in range(3)]
    )
    domain_masks = {"delta": np.ones(flat.size, dtype=bool), "theta": theta_keep}
    transfer_flat = np.asarray(transfer, dtype=np.float64).ravel()[flat]
    density_diagonal = np.sum(
        expected_counts * bias[:, None, None, None] ** 2, axis=0
    )
    mean_density = float(np.mean(density_diagonal))

    def matvec(vector: np.ndarray) -> np.ndarray:
        field = np.asarray(vector, dtype=np.float64).reshape(shape)
        result = exact_velocity_precision(field)
        if density_retention != 0.0:
            result = result + density_retention * _density_precision_action(
                field,
                transfer,
                expected_counts,
                bias,
                marginalize_normalizations,
            )
        return np.asarray(result, dtype=np.float64).ravel()

    preconditioner_spectrum = (
        np.asarray(isotropic_preconditioner_precision, dtype=np.float64)
        + density_retention * mean_density * transfer**2
    )
    if np.any(preconditioner_spectrum <= 0.0):
        raise ExactPilotError("exact-joint preconditioner is not positive")

    def precondition(vector: np.ndarray) -> np.ndarray:
        field = np.asarray(vector, dtype=np.float64).reshape(shape)
        return np.fft.ifftn(
            np.fft.fftn(field, norm="ortho") / preconditioner_spectrum,
            norm="ortho",
        ).real.ravel()

    rng = np.random.default_rng(operator_test_seed)
    left = rng.standard_normal(size)
    right = rng.standard_normal(size)
    lhs = float(np.vdot(left, matvec(right)).real)
    rhs = float(np.vdot(matvec(left), right).real)
    symmetry_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)
    if symmetry_error > symmetry_max:
        raise ExactPilotError("exact joint precision symmetry gate failed")
    if float(np.vdot(left, matvec(left)).real) <= 0.0:
        raise ExactPilotError("exact joint precision positivity gate failed")

    operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator(
        (size, size), matvec=precondition, dtype=np.float64
    )
    trace_probes = {
        domain: np.empty((probe_count, len(domain_bin_ids[domain])), dtype=np.float64)
        for domain in DOMAINS
    }
    denominators: dict[str, np.ndarray] = {}
    for domain in DOMAINS:
        keep = domain_masks[domain]
        denominators[domain] = np.asarray(
            [
                np.sum(transfer_flat[keep & (assignment == bin_id)] ** 2)
                for bin_id in domain_bin_ids[domain]
            ],
            dtype=np.float64,
        )
        if np.any(denominators[domain] <= 0.0):
            raise ExactPilotError(f"{domain} trace denominator is not positive")
    relative_residuals = np.empty(probe_count, dtype=np.float64)
    iterations = np.zeros(probe_count, dtype=np.int64)
    for probe_index in range(probe_count):
        probe, canonical = joint_v1.canonical_probe(
            grid, flat, probe_seed + probe_index
        )

        def callback(_: np.ndarray) -> None:
            iterations[probe_index] += 1

        solution, info = cg(
            operator,
            probe.ravel(),
            M=preconditioner,
            rtol=cg_rtol,
            atol=0.0,
            maxiter=cg_maxiter,
            callback=callback,
        )
        if info != 0:
            raise ExactPilotError(
                f"exact joint CG failed for trace probe {probe_index}: info={info}"
            )
        residual = matvec(solution) - probe.ravel()
        relative_residuals[probe_index] = float(
            np.linalg.norm(residual) / max(np.linalg.norm(probe), 1.0e-30)
        )
        if relative_residuals[probe_index] > cg_rtol:
            raise ExactPilotError(
                f"trace probe {probe_index} residual exceeds frozen CG gate"
            )
        solution_k = np.fft.fftn(solution.reshape(shape), norm="ortho").ravel()[flat]
        contribution = transfer_flat**2 * np.real(
            np.conjugate(canonical) * solution_k
        )
        for domain in DOMAINS:
            keep = domain_masks[domain]
            for column, bin_id in enumerate(domain_bin_ids[domain]):
                trace_probes[domain][probe_index, column] = float(
                    np.sum(contribution[keep & (assignment == bin_id)])
                    / denominators[domain][column]
                )

    result: dict[str, Any] = {
        "operator_symmetry_relative_error": symmetry_error,
        "operator_symmetry_relative_error_max_inclusive": symmetry_max,
        "maximum_CG_relative_residual": float(np.max(relative_residuals)),
        "maximum_CG_iterations": int(np.max(iterations)),
        "all_trace_probe_CG_converged": True,
        "domains": {},
    }
    arrays: dict[str, np.ndarray] = {
        "CG_relative_residual": relative_residuals,
        "CG_iterations": iterations,
    }
    for domain in DOMAINS:
        probes = trace_probes[domain]
        trace = np.mean(probes, axis=0)
        standard_error = np.std(probes, axis=0, ddof=1) / math.sqrt(probe_count)
        information = 1.0 - trace
        lower = information - 1.96 * standard_error
        correlation = np.sqrt(np.clip(information, 0.0, 1.0))
        domain_result = {
            "bin_ids": domain_bin_ids[domain].tolist(),
            "posterior_prior_trace_fraction": trace.tolist(),
            "trace_probe_standard_error": standard_error.tolist(),
            "recovered_information_fraction": information.tolist(),
            "recovered_information_numerical_95_lower": lower.tolist(),
            "expected_response": information.tolist(),
            "expected_correlation_r": correlation.tolist(),
            "expected_residual_power_ratio": trace.tolist(),
        }
        result["domains"][domain] = domain_result
        arrays[f"{domain}_bin_ids"] = np.asarray(domain_bin_ids[domain])
        arrays[f"{domain}_trace_probes"] = probes
        arrays[f"{domain}_posterior_prior_trace_fraction"] = trace
        arrays[f"{domain}_trace_probe_standard_error"] = standard_error
        arrays[f"{domain}_recovered_information_fraction"] = information
        arrays[f"{domain}_recovered_information_numerical_95_lower"] = lower
        arrays[f"{domain}_expected_correlation_r"] = correlation
    return result, arrays


def _attach_gate_classifications(
    scenario: dict[str, Any], gates: Mapping[str, Mapping[str, float]]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    domain_masks: dict[str, dict[str, np.ndarray]] = {}
    for domain in DOMAINS:
        metrics = scenario["domains"][domain]
        domain_masks[domain] = {}
        for gate_name, gate in gates.items():
            mask = _gate_mask(metrics, gate)
            metrics[f"{gate_name}_gate"] = mask.tolist()
            domain_masks[domain][gate_name] = mask
            arrays[f"{domain}_{gate_name}_gate"] = mask
    joint_ids = np.intersect1d(
        np.asarray(scenario["domains"]["delta"]["bin_ids"], dtype=np.int64),
        np.asarray(scenario["domains"]["theta"]["bin_ids"], dtype=np.int64),
    )
    scenario["joint_delta_theta"] = {"bin_ids": joint_ids.tolist()}
    for gate_name in gates:
        delta_ids = np.asarray(scenario["domains"]["delta"]["bin_ids"])
        theta_ids = np.asarray(scenario["domains"]["theta"]["bin_ids"])
        joint = np.asarray(
            [
                domain_masks["delta"][gate_name][np.flatnonzero(delta_ids == bin_id)[0]]
                and domain_masks["theta"][gate_name][np.flatnonzero(theta_ids == bin_id)[0]]
                for bin_id in joint_ids
            ],
            dtype=bool,
        )
        prefix = 0
        for value in joint:
            if not value:
                break
            prefix += 1
        scenario["joint_delta_theta"][f"{gate_name}_gate"] = joint.tolist()
        scenario["joint_delta_theta"][f"{gate_name}_contiguous_prefix_bin_count"] = prefix
        scenario["joint_delta_theta"][f"lowest_bin_{gate_name}_pass"] = bool(
            joint.size and joint[0]
        )
        arrays[f"joint_bin_ids"] = joint_ids
        arrays[f"joint_{gate_name}_gate"] = joint
    return arrays


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_pilot(
    program_path: str | Path,
    staging_path: str | Path,
    output_path: str | Path,
    implementation_commit: str,
) -> dict[str, Any]:
    program, program_sha = load_program(program_path)
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise ExactPilotError("implementation commit must be lowercase 40-hex")
    stage = Path(staging_path)
    output = Path(output_path)
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing overwrite of {output}")
    if stage.exists() or os.path.lexists(stage):
        raise FileExistsError(f"refusing existing staging directory {stage}")
    if not output.parent.is_dir() or stage.parent.resolve() != output.parent.resolve():
        raise ExactPilotError("staging and output must share an existing parent")
    stage.mkdir(mode=0o700)

    design = program["design"]
    numerical = design["numerical_gates"]
    geometry = _load_geometry(program)
    transfer, growth_rate = velocity_base.fixed.build_density_transfer(geometry["args"])
    if transfer.shape != (32, 32, 32):
        raise ExactPilotError("frozen N32 density transfer changed")
    exact_velocity, velocity_report = _build_exact_velocity_precision(
        geometry, numerical
    )
    selection = _load_selection(program)
    plan, preconditioner = _load_mode_and_preconditioner(program, transfer)
    domain_bin_ids = {
        "delta": preconditioner["delta_bin_ids"],
        "theta": preconditioner["theta_bin_ids"],
    }
    scenarios: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "isotropic_preconditioner_bin_ids": preconditioner["bin_ids"],
        "isotropic_preconditioner_delta_trace": preconditioner["delta_trace"],
        "isotropic_preconditioner_theta_trace": preconditioner["theta_trace"],
        "population_counts": selection["population_counts"],
        "reference_bias_by_population": selection["reference_bias"],
        "selected_raw_idx": geometry["selected"],
        "train_raw_idx": geometry["train_raw_idx"],
        "holdout_raw_idx": geometry["holdout_raw_idx"],
    }
    gates = design["information_gates"]
    for scenario_name in SCENARIOS:
        specification = design["scenarios"][scenario_name]
        scenario, scenario_arrays = exact_joint_trace_spectrum(
            exact_velocity_precision=exact_velocity,
            transfer=transfer,
            plan=plan,
            domain_bin_ids=domain_bin_ids,
            isotropic_preconditioner_precision=preconditioner["precision"],
            expected_counts=selection["expected_counts"],
            bias=selection["reference_bias"],
            density_retention=float(specification["density_Fisher_retention"]),
            marginalize_normalizations=bool(
                specification["marginalize_population_normalizations"]
            ),
            probe_count=int(design["trace_probe_count"]),
            probe_seed=int(design["trace_probe_seed"]),
            operator_test_seed=int(numerical["operator_test_seed"]),
            cg_rtol=float(numerical["CG_relative_residual_max_inclusive"]),
            cg_maxiter=int(numerical["CG_max_iterations"]),
            symmetry_max=float(
                numerical["precision_symmetry_relative_error_max_inclusive"]
            ),
        )
        scenario["semantics"] = specification
        gate_arrays = _attach_gate_classifications(scenario, gates)
        scenarios[scenario_name] = scenario
        for name, value in {**scenario_arrays, **gate_arrays}.items():
            arrays[f"scenario_{scenario_name}_{name}"] = value

    known = scenarios["known_selection_reference_bias"]["joint_delta_theta"]
    marginalized = scenarios[
        "normalization_marginalized_reference_bias"
    ]["joint_delta_theta"]
    known_material = bool(known["lowest_bin_material_pass"])
    marginalized_material = bool(marginalized["lowest_bin_material_pass"])
    if marginalized_material:
        status = "PASS_EXACT_GEOMETRY0_NORMALIZATION_MARGINALIZED_MATERIAL_GATE_NO_PROMOTION"
    elif known_material:
        status = "BORDERLINE_EXACT_GEOMETRY0_KNOWN_ONLY_MATERIAL_PASS_NORMALIZATION_BOTTLENECK"
    else:
        status = "COMPLETE_EXACT_GEOMETRY0_MATERIAL_GATE_FAIL_NO_ROUTE_REJECTION"
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "program_sha256": program_sha,
        "implementation_commit": implementation_commit,
        "implementation_source_sha256": sha256_file(__file__),
        "geometry_index": 0,
        "grid_N": 32,
        "box_size_cMpc_h": 384.0,
        "cell_size_cMpc_h": 12.0,
        "growth_rate_z0": float(growth_rate),
        "velocity_operator": {
            **velocity_report,
            "precision": "I + A^T (I + B Lambda B^T)^-1 A",
            "covariance_materialized": False,
            "old_isotropic_spectrum_use": "CG_preconditioner_and_comparison_only",
        },
        "selection": {
            "artifact_path": str(
                program["bindings"]["frozen_2Mpp_selection"]["path"]
            ),
            "artifact_sha256": program["bindings"]["frozen_2Mpp_selection"][
                "sha256"
            ],
            "population_counts": selection["population_counts"].tolist(),
            "expected_count_sums": selection["expected_counts"]
            .reshape(6, -1)
            .sum(axis=1)
            .tolist(),
        },
        "information_gates": gates,
        "scenarios": scenarios,
        "decision": {
            "known_selection_lowest_joint_material_pass": known_material,
            "normalization_marginalized_lowest_joint_material_pass": marginalized_material,
            "old_0p8_stretch_classification_reported": True,
            "single_geometry_route_level_rejection_allowed": False,
            "64_geometry_production_authorized": False,
            "parent_posterior_promotion_allowed": False,
        },
        "firewall": {
            "covariance_only": True,
            "truth_field_array_generated_or_deserialized": False,
            "completed_development_truth_geometry_deserialized": True,
            "likelihood_datum_consumed_by_inference": False,
            "galaxy_positions_consumed_as_field_likelihood_datum": False,
            "observational_CF4_field_inference_executed": False,
            "untouched_256_mock_validation_executed": False,
            "present_density_posterior_created": False,
            "IC_inference_executed": False,
            "observational_resolution_0p3_cMpc_h_established": False,
        },
        "next_action_requires_new_user_approval": True,
    }
    metrics_payload = velocity_base.fixed.deterministic_npz_bytes(arrays)
    result_payload = canonical_json_bytes(result)
    _write_exclusive(stage / "metrics.npz", metrics_payload)
    _write_exclusive(stage / "result.json", result_payload)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "payloads": {
            "metrics.npz": {
                "bytes": len(metrics_payload),
                "sha256": hashlib.sha256(metrics_payload).hexdigest(),
            },
            "result.json": {
                "bytes": len(result_payload),
                "sha256": hashlib.sha256(result_payload).hexdigest(),
            },
        },
    }
    manifest_payload = canonical_json_bytes(manifest)
    _write_exclusive(stage / "manifest.json", manifest_payload)
    complete = {
        "schema": COMPLETE_SCHEMA,
        "status": status,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "COMPLETE_written_last": True,
    }
    _write_exclusive(stage / "COMPLETE", canonical_json_bytes(complete))
    directory_fd = os.open(stage, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.rename(stage, output)
    return validate_pilot(output)


def validate_pilot(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir() or {item.name for item in root.iterdir()} != EXPECTED_FILES:
        raise ExactPilotError("exact-pilot artifact file set is not exact")
    metrics_payload = (root / "metrics.npz").read_bytes()
    result_payload = (root / "result.json").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    manifest = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise ExactPilotError("exact-pilot result is not canonical JSON")
    if result.get("schema") != RESULT_SCHEMA:
        raise ExactPilotError("exact-pilot result schema changed")
    expected_payloads = {
        "metrics.npz": {
            "bytes": len(metrics_payload),
            "sha256": hashlib.sha256(metrics_payload).hexdigest(),
        },
        "result.json": {
            "bytes": len(result_payload),
            "sha256": hashlib.sha256(result_payload).hexdigest(),
        },
    }
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get(
        "payloads"
    ) != expected_payloads:
        raise ExactPilotError("exact-pilot manifest binding changed")
    if complete.get("schema") != COMPLETE_SCHEMA or complete.get(
        "manifest_sha256"
    ) != hashlib.sha256(manifest_payload).hexdigest():
        raise ExactPilotError("exact-pilot COMPLETE binding changed")
    if complete.get("COMPLETE_written_last") is not True:
        raise ExactPilotError("exact-pilot COMPLETE ordering changed")
    firewall = result.get("firewall", {})
    required_false = {
        "truth_field_array_generated_or_deserialized",
        "likelihood_datum_consumed_by_inference",
        "galaxy_positions_consumed_as_field_likelihood_datum",
        "observational_CF4_field_inference_executed",
        "untouched_256_mock_validation_executed",
        "present_density_posterior_created",
        "IC_inference_executed",
        "observational_resolution_0p3_cMpc_h_established",
    }
    if any(firewall.get(key) is not False for key in required_false):
        raise ExactPilotError("exact-pilot crossed its science firewall")
    with np.load(io.BytesIO(metrics_payload), allow_pickle=False) as metrics:
        if not metrics.files or any(
            not np.all(np.isfinite(metrics[name])) for name in metrics.files
        ):
            raise ExactPilotError("exact-pilot metrics are absent or non-finite")
    for scenario in SCENARIOS:
        item = result.get("scenarios", {}).get(scenario, {})
        if item.get("all_trace_probe_CG_converged") is not True:
            raise ExactPilotError(f"exact-pilot scenario failed numerically: {scenario}")
        for domain in DOMAINS:
            if domain not in item.get("domains", {}):
                raise ExactPilotError(f"exact-pilot {scenario} lacks {domain}")
    if result.get("decision", {}).get("parent_posterior_promotion_allowed") is not False:
        raise ExactPilotError("exact-pilot improperly promoted a parent posterior")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-pilot")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--staging", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = sub.add_parser("validate-pilot")
    validate.add_argument("--directory", required=True, type=Path)
    check = sub.add_parser("validate-program")
    check.add_argument("--program", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-pilot":
            result = run_pilot(
                args.program,
                args.staging,
                args.output,
                args.implementation_commit,
            )
            report = {"status": result["status"]}
        elif args.command == "validate-pilot":
            result = validate_pilot(args.directory)
            report = {"status": result["status"]}
        else:
            program, digest = load_program(args.program)
            report = {"status": "PASS", "schema": program["schema"], "sha256": digest}
    except (ExactPilotError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
