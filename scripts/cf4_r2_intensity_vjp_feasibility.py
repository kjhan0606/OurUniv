#!/usr/bin/env python3
"""One N256 training-only count-intensity VJP feasibility check; no sampler."""

import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_conditional_map import BASE, BOX, N, N_IC, ROOT, KNOWN_DEVICE_PEAK_GIB
from cf4_r2_linked_fp_sparse_train import SOURCE
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_count_predictive import count_intensity_from_field
from cf4_r2_resolution_target import source_geometry_at_resolution


ANCHOR = BASE / "r2_hmc_equil24h_20261013/hmc_retained_state.npz"
ANCHOR_SOURCE_COMMIT = "5016282433a375ac9dd89660a10b1ae53d25a370"
OUT_BASE = BASE / "r2_intensity_vjp_feasibility_20261029"


def _save_report(path, report, started):
    report["elapsed_seconds"] = time.monotonic() - started
    report["host_peak_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def _device_memory_snapshot(device):
    stats = device.memory_stats() or {}
    return {
        key: int(stats[key]) if isinstance(stats.get(key), (int, np.integer)) else None
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    }


def _stable_norm(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.linalg.norm(values / scale))


def _count_operator_inputs():
    """Load only the frozen source geometry used by the count operator.

    In particular, do not load FP measurements, mark manifests, or held-out
    outcomes for this count-intensity-only feasibility diagnostic.
    """
    source_keys = ("positions", "angular", "radial_table", "modulus_table",
                   "redshift_table")
    with np.load(SOURCE, allow_pickle=False) as saved:
        source = {key: saved[key].copy() for key in source_keys}
    geometry = dict(
        observer=jnp.full(3, 192.),
        box_size_cMpc_h=BOX,
        hubble_km_s_Mpc=74.6,
        little_h=.746,
        radius_table_cMpc_h=jnp.asarray(source["radial_table"]),
        modulus_table_h=jnp.asarray(source["modulus_table"]),
        redshift_table=jnp.asarray(source["redshift_table"]),
        grid_size=128,
    )
    return source, geometry


def _validated_anchor():
    result_path = ANCHOR.parent / "result.json"
    result = json.loads(result_path.read_text())
    expected_orders = dict(value=2, gradient=2, matched_primal=True)
    if (result.get("status") != "CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR"
            or result.get("source_commit") != ANCHOR_SOURCE_COMMIT
            or result.get("N") != N or result.get("box_cMpc_h") != BOX
            or result.get("likelihood_changed") is not False
            or result.get("checkpoint_replay_passed") is not True
            or result.get("target_orders") != expected_orders):
        raise ValueError("anchor is not the verified unchanged conditional-v6 state")
    with np.load(ANCHOR, allow_pickle=False) as saved:
        q = np.asarray(saved["q"], dtype=np.float64).copy()
        gradient = np.asarray(saved["gradient"], dtype=np.float64)
    if (q.shape != (N_IC + 24,) or gradient.shape != q.shape
            or not np.isfinite(q).all() or not np.isfinite(gradient).all()):
        raise ValueError("verified HMC anchor coordinates/gradient are invalid")
    return q, result


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
    expected_out = configured_out_base / os.environ["SLURM_JOB_ID"]
    if out != expected_out:
        raise ValueError("job output directory is not the unique declared job path")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report_path = out / "result.json"
    report = dict(
        status="STARTED",
        job_id=os.environ["SLURM_JOB_ID"],
        source_commit=expected,
        anchor=str(ANCHOR),
        anchor_source_commit=ANCHOR_SOURCE_COMMIT,
        N=N,
        box_cMpc_h=BOX,
        field_grid_spacing_cMpc_h=BOX / N,
        training_count_rows=47121,
        training_count_keys=37951,
        heldout_outcomes_loaded=0,
        full_volume_intensities_computed=True,
        heldout_intensity_predictions_used_in_metric=0,
        heldout_scores_computed=0,
        full_target_sampler_run=False,
        posterior_sample=False,
        R2_complete=False,
        operation="one fixed-seed VJP of training-exposed positive N256 count intensities",
        vjp_seed=20261029,
        reverse_passes=1,
        dense_jacobian_materialized=False,
        source_mask="train exposure AND positive intensity at fixed anchor only",
        intensity_floor=None,
        cost_caps=dict(application_seconds=7200, slurm_seconds=9000,
                       host_memory_GiB=20),
        resource_snapshot=dict(
            mode=os.environ["CF4_GPU_MODE"],
            partition=os.environ["CF4_SUBMIT_PARTITION"],
            typed_gres=os.environ["CF4_SUBMIT_GRES"],
            h200_free_typed_gpus=int(os.environ["CF4_H200_FREE_TYPED_GPUS"]),
            h100_free_typed_gpus=int(os.environ["CF4_H100_FREE_TYPED_GPUS"]),
            a100_free_typed_gpus=int(os.environ["CF4_A100_FREE_TYPED_GPUS"]),
        ),
        evaluations=[],
    )
    _save_report(report_path, report, started)

    try:
        device = jax.devices()[0]
        before = _device_memory_snapshot(device)
        report.update(device_kind=str(device.device_kind), device_memory_before=before)
        if not jax.config.x64_enabled:
            raise RuntimeError("float64 JAX mode is required by the frozen count target")
        device_limit = before.get("bytes_limit")
        known_peak_bytes = KNOWN_DEVICE_PEAK_GIB * 1024**3
        if device_limit and 1.2 * known_peak_bytes > device_limit:
            raise MemoryError("known R2 device peak lacks 20 percent headroom")
        host_request_bytes = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2
        if host_request_bytes < 20 * 1024**3:
            raise MemoryError("the declared 20 GiB host-memory request is missing")
        q, anchor_result = _validated_anchor()
        white = jnp.asarray(q[:N_IC])
        tracer = jnp.asarray(q[N_IC:N_IC + 9])
        del q

        source, geometry = _count_operator_inputs()
        if (geometry["grid_size"] != 128
                or geometry["box_size_cMpc_h"] != BOX):
            raise ValueError("frozen count geometry changed")
        source = source_geometry_at_resolution(source, N)
        source_jax = {key: jnp.asarray(value) for key, value in source.items()}

        with np.load(BASE / "r2_sky_closed_split_v6/split.npz", allow_pickle=False) as split:
            train_keys = np.asarray(split["train_keys"], dtype=np.int32)
            train_counts = np.asarray(split["train_counts"], dtype=np.int64)
            heldout_voxels = split["heldout_flat_voxels"]
            train_excluded = split["train_window_excluded_keys"]
            heldout_excluded = split["heldout_window_excluded_keys"]
        if (int(train_counts.sum()) != 47121 or len(train_keys) != 37951):
            raise ValueError("frozen v6 training count graph changed")
        train_exposure, _ = build_population_exposure_masks(
            128, heldout_voxels, train_excluded, heldout_excluded)
        if train_exposure.shape != (6 * 128**3,) or not train_exposure[train_keys].all():
            raise ValueError("training key lies outside the frozen train exposure")
        train_rows = np.flatnonzero(train_exposure).astype(np.int32, copy=False)

        parent = BASE / "r2_n256_dynamics_profile_v1"
        parent_result = json.loads((parent / "result.json").read_text())
        if parent_result.get("status") != "N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR":
            raise ValueError("frozen N256 dynamics initializer is absent")
        settings = {key: parent_result["settings"][key]
                    for key in ("cosmology", "a_start", "a_stop", "a_nbody_maxstep")}
        settings.update(n=N, box_cMpc_h=BOX)
        common = json.loads((ROOT / "config/cf4_r2_common_cosmology_v1.json").read_text())["common_cosmology"]
        if (settings["cosmology"]["h"] != 0.746
                or settings["cosmology"]["Om"] != common["Omega_m"]):
            raise ValueError("N256 dynamics cosmology differs from the frozen R2 basis")
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

        forward_started = time.monotonic()
        intensities, pullback = jax.vjp(intensity_flat, white, tracer)
        intensities.block_until_ready()
        forward_seconds = time.monotonic() - forward_started
        if intensities.shape != (6 * 128**3,):
            raise ValueError("active count-intensity output shape changed")
        if intensities.dtype != jnp.float64:
            raise TypeError("active count-intensity output must remain float64")
        train_values = np.asarray(intensities[jnp.asarray(train_rows)], dtype=np.float64)
        key_values = np.asarray(intensities[jnp.asarray(train_keys)], dtype=np.float64)
        if (not np.isfinite(train_values).all() or np.any(train_values < 0.0)
                or not np.isfinite(key_values).all() or np.any(key_values <= 0.0)):
            raise FloatingPointError("training count intensities violate the fixed-anchor contract")
        positive_rows = train_rows[train_values > 0.0]
        if not len(positive_rows):
            raise ValueError("no positive training-exposed intensity rows at the anchor")
        positive_values = train_values[train_values > 0.0]
        min_positive = float(np.min(positive_values))
        max_positive = float(np.max(positive_values))
        max_inverse_sqrt_weight = float(1.0 / np.sqrt(min_positive))
        if not np.isfinite(max_inverse_sqrt_weight):
            raise FloatingPointError("positive-row inverse-square-root weights are nonfinite")
        observed_key_min = float(np.min(key_values))

        rng_key = jax.random.PRNGKey(report["vjp_seed"])
        z = jax.random.normal(rng_key, (len(positive_rows),), dtype=intensities.dtype)
        positive_ids = jnp.asarray(positive_rows, dtype=jnp.int32)
        weights = z / jnp.sqrt(intensities[positive_ids])
        if not bool(np.asarray(jnp.all(jnp.isfinite(weights)))):
            raise FloatingPointError("random positive-row cotangent is nonfinite")
        cotangent = jnp.zeros_like(intensities).at[positive_ids].set(weights)
        del z, weights, train_values, key_values, positive_values
        if time.monotonic() - started >= 7200:
            raise TimeoutError("application budget reached before the single VJP")

        vjp_started = time.monotonic()
        gradient_white, gradient_tracer = pullback(cotangent)
        gradient_white.block_until_ready()
        gradient_tracer.block_until_ready()
        vjp_seconds = time.monotonic() - vjp_started
        gradient_white_host = np.asarray(gradient_white, dtype=np.float64)
        gradient_tracer_host = np.asarray(gradient_tracer, dtype=np.float64)
        gradient_finite = bool(np.isfinite(gradient_white_host).all()
                               and np.isfinite(gradient_tracer_host).all())
        gradient_white_norm = _stable_norm(gradient_white_host) if gradient_finite else np.inf
        gradient_tracer_norm = _stable_norm(gradient_tracer_host) if gradient_finite else np.inf
        gradient_norm = float(np.hypot(gradient_white_norm, gradient_tracer_norm))
        gradient_finite = bool(gradient_finite and np.isfinite(gradient_norm))
        rank_one_outer_product_representable = bool(
            gradient_finite and gradient_norm < np.sqrt(np.finfo(np.float64).max))
        if not gradient_finite:
            terminal_status = "ONE_TRAINING_VJP_NONFINITE_STOP"
        elif not rank_one_outer_product_representable:
            terminal_status = "RANK_ONE_OUTER_PRODUCT_OVERFLOW_STOP"
        else:
            terminal_status = "ONE_TRAINING_VJP_FINITE_NOT_SAMPLER"
        report.update(
            status=terminal_status,
            anchor_checkpoint_replay_recorded=bool(anchor_result["checkpoint_replay_passed"]),
            train_exposed_rows=int(len(train_rows)),
            train_positive_rows=int(len(positive_rows)),
            train_zero_intensity_rows=int(len(train_rows) - len(positive_rows)),
            observed_training_keys=int(len(train_keys)),
            observed_key_min_intensity=observed_key_min,
            min_positive_train_intensity=min_positive,
            max_positive_train_intensity=max_positive,
            max_inverse_sqrt_row_weight=max_inverse_sqrt_weight,
            forward_seconds=float(forward_seconds),
            single_vjp_seconds=float(vjp_seconds),
            intensity_dtype=str(intensities.dtype),
            gradient_shapes=dict(white_ic=list(gradient_white_host.shape),
                                tracer_nuisance=list(gradient_tracer_host.shape)),
            gradient_finite=gradient_finite,
            white_ic_gradient_norm=gradient_white_norm if gradient_finite else None,
            tracer_gradient_norm=gradient_tracer_norm if gradient_finite else None,
            gradient_norm=gradient_norm if gradient_finite else None,
            gradient_rms=float(gradient_norm / np.sqrt(N_IC + 9))
            if gradient_finite else None,
            rank_one_outer_product_representable=rank_one_outer_product_representable,
            device_memory_after=_device_memory_snapshot(device),
        )
        if time.monotonic() - started > 7200:
            report["status"] = "APPLICATION_TIME_CAP_EXCEEDED_STOP"
        elif not rank_one_outer_product_representable:
            report["status"] = terminal_status
        _save_report(report_path, report, started)
        if (not rank_one_outer_product_representable
                or report["status"] == "APPLICATION_TIME_CAP_EXCEEDED_STOP"):
            raise RuntimeError(report["status"])
    except Exception as exc:
        if report.get("status") == "STARTED":
            report["status"] = "VJP_FEASIBILITY_FAILED_STOP"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        _save_report(report_path, report, started)
        raise


if __name__ == "__main__":
    main()
