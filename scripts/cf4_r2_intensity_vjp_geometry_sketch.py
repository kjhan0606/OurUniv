#!/usr/bin/env python3
"""One capped training-only discovery/validation VJP geometry experiment."""

import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_conditional_map import BASE, BOX, N, N_IC
from cf4_r2_intensity_vjp_feasibility import (
    ANCHOR_SOURCE_COMMIT,
    _device_memory_snapshot,
    _save_report,
)
from cf4_r2_intensity_vjp_tail_screen import _build_intensity_operator, _load_training_rows
from cf4_r2_vjp_geometry_metrics import (
    leading_left_basis,
    project_probe,
    subspace_canonical_correlations,
    summarize_validation,
)
from cf4_r2_vjp_tail_metrics import stable_norm, normalized_repeat_error


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = BASE / "r2_intensity_vjp_geometry_sketch_20261102"
DISCOVERY_SEEDS = tuple(range(20261102, 20261118))
VALIDATION_SEEDS = tuple(range(20261202, 20261266))
RANK = 4
DISCOVERY_SPLIT = 8
VALIDATION_BATCH = 32
EXPECTED_PULLBACKS = 81
APPLICATION_CAP_SECONDS = 3.5 * 3600
SLURM_CAP_SECONDS = 4 * 3600
HOST_MEMORY_GIB = 20
ESTIMATED_HOST_PEAK_GIB = 12.5
KNOWN_VJP_PEAK_BYTES = 41133882112  # 38.31 GiB; H100 tail-screen observation.
RELATIVE_TOLERANCE = 1.0e-8
SUBSPACE_MIN_CANONICAL_CORRELATION = 0.80
MATRIX_DISAGREEMENT_LIMIT = 0.50
DIRECTIONAL_RELATIVE_HALF_WIDTH_LIMIT = 0.50
THETA_MIN_RAD = 0.10
RESIDUAL_LIMIT_NATS = 1.0
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261102


def _probe_vector(seed, positive_ids, intensities, pullback):
    rng = jax.random.PRNGKey(int(seed))
    z = jax.random.normal(rng, (len(positive_ids),), dtype=intensities.dtype)
    weights = z / jnp.sqrt(intensities[positive_ids])
    if not bool(np.asarray(jnp.all(jnp.isfinite(weights)))):
        raise FloatingPointError("Gaussian positive-row cotangent is nonfinite")
    cotangent = jnp.zeros_like(intensities).at[positive_ids].set(weights)
    del z, weights
    started = time.monotonic()
    gradient_white, gradient_tracer = pullback(cotangent)
    gradient_white.block_until_ready()
    gradient_tracer.block_until_ready()
    elapsed = time.monotonic() - started
    white = np.asarray(gradient_white, dtype=np.float64).copy()
    tracer = np.asarray(gradient_tracer, dtype=np.float64).copy()
    if not np.isfinite(white).all() or not np.isfinite(tracer).all():
        raise FloatingPointError("VJP produced a nonfinite IC or tracer gradient")
    vector = np.concatenate((white.reshape(-1), tracer.reshape(-1)))
    if vector.shape != (N_IC + 9,) or not np.isfinite(vector).all():
        raise ValueError("joint white-IC/tracer VJP has unexpected geometry")
    record = dict(
        seed=int(seed), elapsed_seconds=float(elapsed),
        white_ic_gradient_norm=stable_norm(white.reshape(-1)),
        tracer_gradient_norm=stable_norm(tracer.reshape(-1)),
        joint_gradient_norm=stable_norm(vector),
        gradient_finite=True,
    )
    del cotangent, gradient_white, gradient_tracer, white, tracer
    return vector, record


def _assert_job_contract():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    expected = os.environ["CF4_EXPECTED_COMMIT"]
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    if actual != expected:
        raise RuntimeError("source commit does not match submitted allocation")
    mode = os.environ["CF4_GPU_MODE"]
    device_kind = str(jax.devices()[0].device_kind).upper()
    if mode not in ("h200", "h100", "a100") or mode.upper() not in device_kind:
        raise RuntimeError("allocated device does not match the typed GPU request")
    configured_base = Path(os.environ["CF4_R2_OUT_BASE"])
    if configured_base != OUT_BASE:
        raise ValueError("output base differs from the frozen one-shot run")
    out = Path(os.environ["CF4_R2_OUT_DIR"])
    if out != configured_base / os.environ["SLURM_JOB_ID"]:
        raise ValueError("output path is not the unique job directory")
    out.mkdir(parents=True, exist_ok=False)
    return expected, mode, out


def _check_device_headroom(snapshot, limit_bytes):
    peak = snapshot.get("peak_bytes_in_use")
    if peak is None:
        raise MemoryError("device peak-memory measurement is unavailable")
    if 1.2 * max(KNOWN_VJP_PEAK_BYTES, int(peak)) > limit_bytes:
        raise MemoryError("observed device peak lacks the required 20 percent headroom")


def _main():
    expected, mode, out = _assert_job_contract()
    started = time.monotonic()
    report_path = out / "result.json"
    report = dict(
        status="STARTED", job_id=os.environ["SLURM_JOB_ID"],
        source_commit=expected, anchor_source_commit=ANCHOR_SOURCE_COMMIT,
        anchor=str(BASE / "r2_hmc_equil24h_20261013/hmc_retained_state.npz"),
        N=N, box_cMpc_h=BOX, field_grid_spacing_cMpc_h=BOX / N,
        target="fixed conditional-v6; training-only proposal geometry",
        operation="one rank-4 VJP discovery/independent-validation sketch",
        discovery_seeds=list(DISCOVERY_SEEDS),
        validation_seeds=list(VALIDATION_SEEDS),
        discovery_half_split=[DISCOVERY_SPLIT, len(DISCOVERY_SEEDS)-DISCOVERY_SPLIT],
        validation_batches=[VALIDATION_BATCH, VALIDATION_BATCH],
        repeat_seed=DISCOVERY_SEEDS[0], rank=RANK,
        expected_reverse_passes=EXPECTED_PULLBACKS,
        heldout_outcomes_loaded=0, heldout_scores_computed=0,
        native_truth_ids_loaded=0, full_target_correction_performed=False,
        sampler_run=False, posterior_sample=False, uncertainty_established=False,
        R2_complete=False, rows_removed_from_exact_target=0,
        intensity_floor=None, zero_anchor_rows_in_proposal=False,
        cost_caps=dict(application_seconds=APPLICATION_CAP_SECONDS,
                       slurm_seconds=SLURM_CAP_SECONDS,
                       host_memory_GiB=HOST_MEMORY_GIB,
                       estimated_host_peak_GiB=ESTIMATED_HOST_PEAK_GIB),
        resource_snapshot=dict(
            mode=mode, partition=os.environ["CF4_SUBMIT_PARTITION"],
            typed_gres=os.environ["CF4_SUBMIT_GRES"],
            h200_free_typed_gpus=int(os.environ["CF4_H200_FREE_TYPED_GPUS"]),
            h100_free_typed_gpus=int(os.environ["CF4_H100_FREE_TYPED_GPUS"]),
            a100_free_typed_gpus=int(os.environ["CF4_A100_FREE_TYPED_GPUS"]),
        ),
        gates_predeclared=dict(
            decomposition_repeat_tolerance=RELATIVE_TOLERANCE,
            basis_orthogonality_tolerance=RELATIVE_TOLERANCE,
            minimum_subspace_canonical_correlation=SUBSPACE_MIN_CANONICAL_CORRELATION,
            projected_batch_relative_frobenius_difference=MATRIX_DISAGREEMENT_LIMIT,
            directional_relative_half_width=DIRECTIONAL_RELATIVE_HALF_WIDTH_LIMIT,
            theta_min_rad=THETA_MIN_RAD,
            residual_quadratic_cost_limit_nats=RESIDUAL_LIMIT_NATS,
        ),
        LG_roles=dict(MW="ambiguous", M31="ambiguous", M33="unresolved"),
        LG_requirement=("future observables constrain those roles on the same NEW evolved LG field at <=0.3 cMpc/h; "
                        "native truth IDs are calibration/evaluation labels only"),
        Q_GOAL="test one rank-4 training-only proposal geometry toward the first conditional-v6 N256 z=0 posterior",
        Q_LEAN="one fixed 16+64 probe sketch, one repeat, no heldout read, no rank/seed ladder, no simulation",
        evaluations=[],
    )
    _save_report(report_path, report, started)

    try:
        if not jax.config.x64_enabled:
            raise RuntimeError("float64 JAX mode required by the frozen count target")
        device = jax.devices()[0]
        memory_before = _device_memory_snapshot(device)
        report.update(device_kind=str(device.device_kind), device_memory_before=memory_before)
        device_limit = memory_before.get("bytes_limit")
        if not device_limit:
            raise MemoryError("device memory limit is unavailable; 20 percent headroom cannot be verified")
        if 1.2 * KNOWN_VJP_PEAK_BYTES > device_limit:
            raise MemoryError("measured VJP peak lacks the required 20 percent device headroom")
        host_request_bytes = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2
        if host_request_bytes < HOST_MEMORY_GIB * 1024**3:
            raise MemoryError("declared 20 GiB host memory request is missing")
        if 1.2 * ESTIMATED_HOST_PEAK_GIB > HOST_MEMORY_GIB:
            raise MemoryError("estimated host workspace lacks 20 percent headroom")

        train_keys, train_rows = _load_training_rows()
        white, tracer, intensity_flat, anchor_result = _build_intensity_operator()
        forward_started = time.monotonic()
        intensities, pullback = jax.vjp(intensity_flat, white, tracer)
        intensities.block_until_ready()
        report["forward_seconds"] = time.monotonic() - forward_started
        report["anchor_checkpoint_replay_passed"] = bool(
            anchor_result["checkpoint_replay_passed"])
        if time.monotonic() - started > APPLICATION_CAP_SECONDS:
            raise TimeoutError("application cap reached before VJP probes")
        if intensities.shape != (6 * 128**3,) or intensities.dtype != jnp.float64:
            raise TypeError("active count intensity geometry/dtype changed")

        training_values = np.asarray(intensities[jnp.asarray(train_rows)], dtype=np.float64)
        key_values = np.asarray(intensities[jnp.asarray(train_keys)], dtype=np.float64)
        if (not np.isfinite(training_values).all() or np.any(training_values < 0.0)
                or not np.isfinite(key_values).all() or np.any(key_values <= 0.0)):
            raise FloatingPointError("training intensities violate the frozen anchor contract")
        positive_rows = train_rows[training_values > 0.0]
        positive_ids = jnp.asarray(positive_rows, dtype=jnp.int32)
        positive_values = training_values[training_values > 0.0]
        if len(positive_rows) != 1993850:
            raise ValueError("frozen positive training-row count changed")
        report.update(
            training_count_occurrences=47121,
            training_count_keys=int(train_keys.size),
            train_exposed_rows=int(train_rows.size),
            train_positive_rows=int(positive_rows.size),
            train_zero_intensity_rows=int(train_rows.size-positive_rows.size),
            min_positive_intensity=float(np.min(positive_values)),
            max_positive_intensity=float(np.max(positive_values)),
            max_inverse_sqrt_positive_weight=float(1.0/np.sqrt(np.min(positive_values))),
        )
        _save_report(report_path, report, started)

        sketch = np.empty((N_IC + 9, len(DISCOVERY_SEEDS)), dtype=np.float64)
        for index, seed in enumerate(DISCOVERY_SEEDS):
            if time.monotonic() - started >= APPLICATION_CAP_SECONDS:
                raise TimeoutError("application cap reached during discovery")
            vector, record = _probe_vector(seed, positive_ids, intensities, pullback)
            sketch[:, index] = vector
            record.update(tag="discovery", index=int(index))
            record["device_memory"] = _device_memory_snapshot(device)
            _check_device_headroom(record["device_memory"], device_limit)
            report["evaluations"].append(record)
            report["status"] = "DISCOVERY_RUNNING"
            _save_report(report_path, report, started)
            if 1.2 * report["host_peak_GiB"] > HOST_MEMORY_GIB:
                raise MemoryError("observed host peak lacks the requested 20 percent headroom")
            if index == 0:
                first_discovery_vector = sketch[:, index].copy()
            del vector

        basis_first, singular_first, orth_first = leading_left_basis(
            sketch[:, :DISCOVERY_SPLIT], RANK)
        basis_second, singular_second, orth_second = leading_left_basis(
            sketch[:, DISCOVERY_SPLIT:], RANK)
        basis, singular_values, orth_error = leading_left_basis(sketch, RANK)
        correlations = subspace_canonical_correlations(basis_first, basis_second)
        minimum_correlation = float(np.min(correlations))
        discovery_passed = bool(
            max(orth_first, orth_second, orth_error) <= RELATIVE_TOLERANCE
            and minimum_correlation >= SUBSPACE_MIN_CANONICAL_CORRELATION)
        report.update(
            discovery=dict(
                singular_values=singular_values.tolist(),
                first_half_singular_values=singular_first.tolist(),
                second_half_singular_values=singular_second.tolist(),
                basis_orthogonality_error=float(orth_error),
                first_half_orthogonality_error=float(orth_first),
                second_half_orthogonality_error=float(orth_second),
                canonical_correlations=correlations.tolist(),
                minimum_canonical_correlation=minimum_correlation,
                minimum_required_canonical_correlation=SUBSPACE_MIN_CANONICAL_CORRELATION,
                passed=discovery_passed,
            )
        )
        _save_report(report_path, report, started)
        if not discovery_passed:
            report["status"] = "DISCOVERY_SUBSPACE_GATE_FAILED_STOP"
            _save_report(report_path, report, started)
            raise RuntimeError(report["status"])

        repeated_vector, repeat_record = _probe_vector(
            DISCOVERY_SEEDS[0], positive_ids, intensities, pullback)
        repeat_error = normalized_repeat_error(first_discovery_vector, repeated_vector)
        report["repeat"] = dict(seed=DISCOVERY_SEEDS[0], **repeat_error,
                                passed=repeat_error["relative_error"] <= RELATIVE_TOLERANCE)
        repeat_record.update(tag="discovery_repeat", index=0,
                             device_memory=_device_memory_snapshot(device))
        _check_device_headroom(repeat_record["device_memory"], device_limit)
        report["evaluations"].append(repeat_record)
        del repeated_vector, first_discovery_vector
        _save_report(report_path, report, started)
        if 1.2 * report["host_peak_GiB"] > HOST_MEMORY_GIB:
            raise MemoryError("observed host peak lacks the requested 20 percent headroom")
        if not report["repeat"]["passed"]:
            report["status"] = "REPEAT_GATE_FAILED_STOP"
            _save_report(report_path, report, started)
            raise RuntimeError(report["status"])

        projections = np.empty((len(VALIDATION_SEEDS), RANK), dtype=np.float64)
        residual_squared = np.empty(len(VALIDATION_SEEDS), dtype=np.float64)
        full_squared = np.empty(len(VALIDATION_SEEDS), dtype=np.float64)
        for index, seed in enumerate(VALIDATION_SEEDS):
            if time.monotonic() - started >= APPLICATION_CAP_SECONDS:
                raise TimeoutError("application cap reached during validation")
            vector, record = _probe_vector(seed, positive_ids, intensities, pullback)
            p, residual_sq, full_sq = project_probe(basis, vector)
            projections[index] = p
            residual_squared[index] = residual_sq
            full_squared[index] = full_sq
            record.update(tag="validation", index=int(index),
                          projected_coordinates=p.tolist(),
                          residual_squared_norm=residual_sq,
                          full_squared_norm=full_sq,
                          device_memory=_device_memory_snapshot(device))
            _check_device_headroom(record["device_memory"], device_limit)
            report["evaluations"].append(record)
            report["status"] = "VALIDATION_RUNNING"
            _save_report(report_path, report, started)
            if 1.2 * report["host_peak_GiB"] > HOST_MEMORY_GIB:
                raise MemoryError("observed host peak lacks the requested 20 percent headroom")
            del vector

        validation = summarize_validation(
            projections, residual_squared, full_squared,
            batch_size=VALIDATION_BATCH,
            theta_min=THETA_MIN_RAD,
            residual_limit_nats=RESIDUAL_LIMIT_NATS,
            matrix_disagreement_limit=MATRIX_DISAGREEMENT_LIMIT,
            directional_relative_half_width_limit=DIRECTIONAL_RELATIVE_HALF_WIDTH_LIMIT,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        observed_peak_bytes = max(
            [int(row["device_memory"].get("peak_bytes_in_use", 0))
             for row in report["evaluations"]] or [0])
        memory_guard_passed = bool(
            1.2 * max(KNOWN_VJP_PEAK_BYTES, observed_peak_bytes) <= device_limit)
        discovery_orthogonality_passed = bool(
            report["discovery"]["basis_orthogonality_error"] <= RELATIVE_TOLERANCE
            and report["discovery"]["first_half_orthogonality_error"] <= RELATIVE_TOLERANCE
            and report["discovery"]["second_half_orthogonality_error"] <= RELATIVE_TOLERANCE)
        repeat_passed = bool(report["repeat"]["passed"])
        report.update(
            validation=validation,
            device_peak_bytes_observed=observed_peak_bytes,
            device_headroom_guard_passed=memory_guard_passed,
            discovery_subspace_passed=discovery_passed,
            discovery_orthogonality_passed=discovery_orthogonality_passed,
            repeat_passed=repeat_passed,
            final_passed=bool(
                discovery_passed and discovery_orthogonality_passed and repeat_passed
                and validation["passed"] and memory_guard_passed),
        )
        if report["final_passed"]:
            basis_path = out / "U_rank4_whiteIC_tracer.npy"
            np.save(basis_path, basis, allow_pickle=False)
            report["basis_artifact"] = dict(path=str(basis_path),
                                             shape=list(basis.shape),
                                             dtype=str(basis.dtype),
                                             bytes=int(basis_path.stat().st_size))
            report["status"] = "GEOMETRY_SKETCH_PASS_NOT_SAMPLER"
        else:
            report["status"] = "GEOMETRY_SKETCH_GATE_FAILED_STOP"
        _save_report(report_path, report, started)
        if not report["final_passed"]:
            raise RuntimeError(report["status"])
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as exc:
        if report.get("status") in ("STARTED", "DISCOVERY_RUNNING", "VALIDATION_RUNNING"):
            report["status"] = "GEOMETRY_SKETCH_FAILED_STOP"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        _save_report(report_path, report, started)
        raise


if __name__ == "__main__":
    _main()
