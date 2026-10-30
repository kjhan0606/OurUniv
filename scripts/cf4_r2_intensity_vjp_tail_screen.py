#!/usr/bin/env python3
"""Seven-pullback training-only tail/repeat screen; not an R2 sampler."""

import json
import math
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_conditional_map import BASE, BOX, N, N_IC, ROOT, KNOWN_DEVICE_PEAK_GIB
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_count_predictive import count_intensity_from_field
from cf4_r2_intensity_vjp_feasibility import (
    ANCHOR_SOURCE_COMMIT,
    _count_operator_inputs,
    _device_memory_snapshot,
    _save_report,
    _validated_anchor,
)
from cf4_r2_resolution_target import source_geometry_at_resolution
from cf4_r2_vjp_tail_metrics import (
    normalized_repeat_error,
    split_low_intensity_tail,
    stable_norm,
    summarize_vjp_split,
)


OUT_BASE = BASE / "r2_intensity_vjp_tail_screen_20261030"
SEEDS = (20261029, 20261030)
TAIL_FRACTION = 0.01
RELATIVE_TOLERANCE = 1.0e-8
APPLICATION_CAP_SECONDS = 3600
SLURM_CAP_SECONDS = 4500
HOST_MEMORY_GIB = 20
PULLBACK_ORDER = (
    "seed_20261029_full",
    "seed_20261029_tail",
    "seed_20261029_rest",
    "seed_20261029_full_repeat",
    "seed_20261030_full",
    "seed_20261030_tail",
    "seed_20261030_rest",
)


def _load_training_rows():
    with np.load(BASE / "r2_sky_closed_split_v6/split.npz", allow_pickle=False) as split:
        train_keys = np.asarray(split["train_keys"], dtype=np.int32)
        train_counts = np.asarray(split["train_counts"], dtype=np.int64)
        heldout_voxels = split["heldout_flat_voxels"]
        train_excluded = split["train_window_excluded_keys"]
        heldout_excluded = split["heldout_window_excluded_keys"]
    if int(train_counts.sum()) != 47121 or len(train_keys) != 37951:
        raise ValueError("frozen v6 training count graph changed")
    train_exposure, _ = build_population_exposure_masks(
        128, heldout_voxels, train_excluded, heldout_excluded)
    if (train_exposure.shape != (6 * 128**3,)
            or not train_exposure[train_keys].all()):
        raise ValueError("training keys are outside the frozen training exposure")
    return train_keys, np.flatnonzero(train_exposure).astype(np.int32, copy=False)


def _build_intensity_operator():
    q, anchor_result = _validated_anchor()
    white = jnp.asarray(q[:N_IC])
    tracer = jnp.asarray(q[N_IC:N_IC + 9])
    del q

    source, geometry = _count_operator_inputs()
    if geometry["grid_size"] != 128 or geometry["box_size_cMpc_h"] != BOX:
        raise ValueError("frozen count geometry changed")
    source = source_geometry_at_resolution(source, N)
    source_jax = {key: jnp.asarray(value) for key, value in source.items()}

    parent = BASE / "r2_n256_dynamics_profile_v1"
    parent_result = json.loads((parent / "result.json").read_text())
    if parent_result.get("status") != "N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR":
        raise ValueError("frozen N256 dynamics initializer is absent")
    settings = {key: parent_result["settings"][key]
                for key in ("cosmology", "a_start", "a_stop", "a_nbody_maxstep")}
    settings.update(n=N, box_cMpc_h=BOX)
    common = json.loads(
        (ROOT / "config/cf4_r2_common_cosmology_v1.json").read_text()
    )["common_cosmology"]
    if (settings["cosmology"]["h"] != 0.746
            or settings["cosmology"]["Om"] != common["Omega_m"]):
        raise ValueError("N256 dynamics cosmology differs from frozen R2 basis")
    evolve, _, conf, _, particle_mass = make_dynamics(settings)
    particle_masses = jnp.full(N_IC, particle_mass)

    @jax.jit
    def intensity_flat(white_ic, tracer_white):
        positions, velocities = evolve(white_ic)
        state = particle_grid(positions, velocities, particle_masses, conf)
        return count_intensity_from_field(
            state["rho"],
            jnp.moveaxis(state["mean_velocity_km_s"], -1, 0),
            tracer_white, source_jax, geometry,
            volume_order=2, source_chunk_size=2097152,
        ).reshape(-1)

    return white, tracer, intensity_flat, anchor_result


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    expected = os.environ["CF4_EXPECTED_COMMIT"]
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected:
        raise RuntimeError("source commit does not match submitted allocation")
    mode = os.environ["CF4_GPU_MODE"]
    if mode not in ("h200", "h100", "a100") or mode.upper() not in str(jax.devices()[0].device_kind).upper():
        raise RuntimeError("allocated device does not match the typed GPU request")

    configured_out_base = Path(os.environ["CF4_R2_OUT_BASE"])
    if configured_out_base != OUT_BASE:
        raise ValueError("output base does not match the declared one-shot run")
    out = Path(os.environ["CF4_R2_OUT_DIR"])
    if out != configured_out_base / os.environ["SLURM_JOB_ID"]:
        raise ValueError("job output directory is not the unique declared job path")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report_path = out / "result.json"
    report = dict(
        status="STARTED",
        job_id=os.environ["SLURM_JOB_ID"],
        source_commit=expected,
        anchor=str(BASE / "r2_hmc_equil24h_20261013/hmc_retained_state.npz"),
        anchor_source_commit=ANCHOR_SOURCE_COMMIT,
        N=N,
        box_cMpc_h=BOX,
        field_grid_spacing_cMpc_h=BOX / N,
        training_count_rows=47121,
        training_count_keys=37951,
        heldout_outcomes_loaded=0,
        heldout_scores_computed=0,
        full_target_sampler_run=False,
        posterior_sample=False,
        R2_complete=False,
        operation="paired training-only positive-row tail/rest VJP decomposition",
        seeds=list(SEEDS),
        pullback_order=list(PULLBACK_ORDER),
        reverse_passes=7,
        tail_fraction=TAIL_FRACTION,
        tail_rule="lowest anchor intensity among positive training-exposed rows; ceil(1 percent); ties by flat row ID",
        intensity_floor=None,
        rows_removed_from_target=0,
        metric_tolerance=RELATIVE_TOLERANCE,
        metrics_are_per_probe_not_trace_estimates=True,
        tail_share_is_not_a_failure_gate=True,
        cost_caps=dict(application_seconds=APPLICATION_CAP_SECONDS,
                       slurm_seconds=SLURM_CAP_SECONDS,
                       host_memory_GiB=HOST_MEMORY_GIB),
        resource_snapshot=dict(
            mode=mode,
            partition=os.environ["CF4_SUBMIT_PARTITION"],
            typed_gres=os.environ["CF4_SUBMIT_GRES"],
            h200_free_typed_gpus=int(os.environ["CF4_H200_FREE_TYPED_GPUS"]),
            h100_free_typed_gpus=int(os.environ["CF4_H100_FREE_TYPED_GPUS"]),
            a100_free_typed_gpus=int(os.environ["CF4_A100_FREE_TYPED_GPUS"]),
        ),
        LG_roles=dict(MW="ambiguous", M31="ambiguous", M33="unresolved"),
        LG_requirement=("future observables constrain these roles on the same NEW evolved LG field at <=0.3 cMpc/h; "
                        "native truth IDs are calibration/evaluation labels only"),
        Q_GOAL="bounded stability diagnostic for the first conditional-v6 N256 z=0 posterior route; no posterior delivered",
        Q_LEAN="one fixed low-intensity 1% partition, two fixed seeds, seven pullbacks, no floor or parameter/seed ladder",
        evaluations=[],
    )
    _save_report(report_path, report, started)

    try:
        device = jax.devices()[0]
        memory_before = _device_memory_snapshot(device)
        report.update(device_kind=str(device.device_kind), device_memory_before=memory_before)
        if not jax.config.x64_enabled:
            raise RuntimeError("float64 JAX mode is required by the frozen count target")
        device_limit = memory_before.get("bytes_limit")
        if (device_limit
                and 1.2 * KNOWN_DEVICE_PEAK_GIB * 1024**3 > device_limit):
            raise MemoryError("known R2 device peak lacks 20 percent headroom")
        host_request_bytes = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2
        if host_request_bytes < HOST_MEMORY_GIB * 1024**3:
            raise MemoryError("the declared 20 GiB host-memory request is missing")

        train_keys, train_rows = _load_training_rows()
        white, tracer, intensity_flat, anchor_result = _build_intensity_operator()
        forward_started = time.monotonic()
        intensities, pullback = jax.vjp(intensity_flat, white, tracer)
        intensities.block_until_ready()
        report["forward_seconds"] = time.monotonic() - forward_started
        if time.monotonic() - started > APPLICATION_CAP_SECONDS:
            raise TimeoutError("application cap reached before VJP screen")
        if (intensities.shape != (6 * 128**3,)
                or intensities.dtype != jnp.float64):
            raise TypeError("active count-intensity map geometry/dtype changed")

        train_values = np.asarray(intensities[jnp.asarray(train_rows)], dtype=np.float64)
        key_values = np.asarray(intensities[jnp.asarray(train_keys)], dtype=np.float64)
        if (not np.isfinite(train_values).all() or np.any(train_values < 0.0)
                or not np.isfinite(key_values).all() or np.any(key_values <= 0.0)):
            raise FloatingPointError("training count intensities violate the fixed-anchor contract")
        positive_rows = train_rows[train_values > 0.0]
        positive_values = train_values[train_values > 0.0]
        if not len(positive_rows):
            raise ValueError("no positive training-exposed rows at the fixed anchor")
        tail_positions, rest_positions = split_low_intensity_tail(
            positive_rows, positive_values, fraction=TAIL_FRACTION)
        report.update(
            anchor_checkpoint_replay_recorded=bool(anchor_result["checkpoint_replay_passed"]),
            train_exposed_rows=int(len(train_rows)),
            train_positive_rows=int(len(positive_rows)),
            train_zero_intensity_rows=int(len(train_rows) - len(positive_rows)),
            observed_training_keys=int(len(train_keys)),
            min_positive_train_intensity=float(np.min(positive_values)),
            max_positive_train_intensity=float(np.max(positive_values)),
            max_inverse_sqrt_row_weight=float(1.0 / np.sqrt(np.min(positive_values))),
            tail_rows=int(len(tail_positions)),
            rest_rows=int(len(rest_positions)),
            tail_min_intensity=float(np.min(positive_values[tail_positions])),
            tail_max_intensity=float(np.max(positive_values[tail_positions])),
            rest_min_intensity=float(np.min(positive_values[rest_positions])),
            rest_max_intensity=float(np.max(positive_values[rest_positions])),
        )
        _save_report(report_path, report, started)

        positive_ids = jnp.asarray(positive_rows, dtype=jnp.int32)
        tail_position_ids = jnp.asarray(tail_positions, dtype=jnp.int32)
        rest_position_ids = jnp.asarray(rest_positions, dtype=jnp.int32)
        result_by_tag = {}

        def run_pullback(tag, seed, positions):
            if time.monotonic() - started >= APPLICATION_CAP_SECONDS:
                raise TimeoutError("one-hour application cap reached before the next pullback")
            rng = jax.random.PRNGKey(seed)
            z = jax.random.normal(rng, (len(positive_rows),), dtype=intensities.dtype)
            weights = z / jnp.sqrt(intensities[positive_ids])
            if not bool(np.asarray(jnp.all(jnp.isfinite(weights)))):
                raise FloatingPointError("random weighted positive-row cotangent is nonfinite")
            if positions is None:
                selected_ids, selected_weights = positive_ids, weights
            else:
                selected_ids = positive_ids[positions]
                selected_weights = weights[positions]
            cotangent = jnp.zeros_like(intensities).at[selected_ids].set(selected_weights)
            del z, weights, selected_ids, selected_weights
            pull_started = time.monotonic()
            gradient_white, gradient_tracer = pullback(cotangent)
            gradient_white.block_until_ready()
            gradient_tracer.block_until_ready()
            pull_seconds = time.monotonic() - pull_started
            gw = np.asarray(gradient_white, dtype=np.float64).copy()
            gt = np.asarray(gradient_tracer, dtype=np.float64).copy()
            if not np.isfinite(gw).all() or not np.isfinite(gt).all():
                raise FloatingPointError(f"{tag} produced a nonfinite gradient")
            gw_norm, gt_norm = stable_norm(gw), stable_norm(gt)
            total_norm = math.hypot(gw_norm, gt_norm)
            if (not math.isfinite(total_norm)
                    or total_norm >= np.sqrt(np.finfo(np.float64).max)):
                raise FloatingPointError(f"{tag} rank-one outer product is not representable")
            record = dict(tag=tag, seed=seed,
                          subset="full" if positions is None else ("tail" if tag.endswith("tail") else "rest"),
                          elapsed_seconds=float(pull_seconds),
                          white_ic_gradient_norm=gw_norm,
                          tracer_gradient_norm=gt_norm,
                          joint_gradient_norm=total_norm,
                          gradient_finite=True,
                          device_memory=_device_memory_snapshot(device))
            report["evaluations"].append(record)
            _save_report(report_path, report, started)
            del cotangent, gradient_white, gradient_tracer
            return gw, gt

        seed_a, seed_b = SEEDS
        full_a_w, full_a_t = run_pullback("seed_20261029_full", seed_a, None)
        tail_a_w, tail_a_t = run_pullback("seed_20261029_tail", seed_a, tail_position_ids)
        rest_a_w, rest_a_t = run_pullback("seed_20261029_rest", seed_a, rest_position_ids)
        decomposition_a = dict(
            white_ic=summarize_vjp_split(full_a_w, tail_a_w, rest_a_w,
                                         tolerance=RELATIVE_TOLERANCE),
            tracer=summarize_vjp_split(full_a_t, tail_a_t, rest_a_t,
                                       tolerance=RELATIVE_TOLERANCE),
        )
        del tail_a_w, tail_a_t, rest_a_w, rest_a_t
        repeat_a_w, repeat_a_t = run_pullback(
            "seed_20261029_full_repeat", seed_a, None)
        repeat_a = dict(
            white_ic=normalized_repeat_error(full_a_w, repeat_a_w),
            tracer=normalized_repeat_error(full_a_t, repeat_a_t),
        )
        del full_a_w, full_a_t, repeat_a_w, repeat_a_t

        full_b_w, full_b_t = run_pullback("seed_20261030_full", seed_b, None)
        tail_b_w, tail_b_t = run_pullback("seed_20261030_tail", seed_b, tail_position_ids)
        rest_b_w, rest_b_t = run_pullback("seed_20261030_rest", seed_b, rest_position_ids)
        decomposition_b = dict(
            white_ic=summarize_vjp_split(full_b_w, tail_b_w, rest_b_w,
                                         tolerance=RELATIVE_TOLERANCE),
            tracer=summarize_vjp_split(full_b_t, tail_b_t, rest_b_t,
                                       tolerance=RELATIVE_TOLERANCE),
        )
        del tail_b_w, tail_b_t, rest_b_w, rest_b_t, full_b_w, full_b_t

        decompositions_passed = all(
            part[coord]["decomposition_passed"]
            for part in (decomposition_a, decomposition_b)
            for coord in ("white_ic", "tracer")
        )
        repeats_passed = all(
            repeat_a[coord]["relative_error"] <= RELATIVE_TOLERANCE
            for coord in ("white_ic", "tracer")
        )
        report.update(
            decomposition_by_seed={str(seed_a): decomposition_a,
                                   str(seed_b): decomposition_b},
            repeated_full_by_coordinate=repeat_a,
            decomposition_tolerance=RELATIVE_TOLERANCE,
            repeat_tolerance=RELATIVE_TOLERANCE,
            decomposition_passed=bool(decompositions_passed),
            repeat_passed=bool(repeats_passed),
            final_device_memory=_device_memory_snapshot(device),
            status=("VJP_TAIL_SPLIT_SCREEN_PASS_NOT_SAMPLER"
                    if decompositions_passed and repeats_passed
                    else "VJP_TAIL_SPLIT_NUMERICAL_STOP"),
        )
        _save_report(report_path, report, started)
        if report["status"] != "VJP_TAIL_SPLIT_SCREEN_PASS_NOT_SAMPLER":
            raise RuntimeError(report["status"])
    except Exception as exc:
        if report.get("status") == "STARTED":
            report["status"] = "VJP_TAIL_SPLIT_SCREEN_FAILED_STOP"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        _save_report(report_path, report, started)
        raise

    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
