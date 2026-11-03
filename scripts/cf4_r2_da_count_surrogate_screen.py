"""Fixed-anchor delayed-acceptance screen; never emits posterior draws."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_delayed_acceptance import (
    delayed_acceptance_stage2_accept,
    lift_restricted_white_field,
    restrict_white_field,
)
from cf4_r2_hmc_transport import SCALES
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_prior_split_hmc import (
    FixedSplitMetric,
    PilotBudgetStop,
    inverse_laplacian_metric_symbol,
    split_hmc_step,
)
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses
from cf4_r2_resolution_target import (
    ResolutionObservationTarget,
    linked_point_conditioning_radius,
    source_geometry_at_resolution,
)
from cf4_r2_scalar_primal import scalar_observation_terms


BASE = Path("/gpfs/kjhan/CF4/z0_density")
ROOT = Path(__file__).resolve().parents[1]
N_FINE = 256
N_COARSE = 128
BOX = 384.0
N_FINE_IC = N_FINE**3
N_COARSE_IC = N_COARSE**3
N_TRACER = 9
N_POPULATION = 15
EPSILON = 1e-30
STEP = 0.11996093316338646
TRAJECTORY_STEPS = 8
N_PROPOSALS = 8
MAX_FINE_VALUES = 1 + N_PROPOSALS
APP_SECONDS = 4 * 60 * 60
SLURM_WALL_SECONDS = 4 * 60 * 60 + 30 * 60
KNOWN_FINE_DEVICE_PEAK_GIB = 30.909375801682472
MIN_DEVICE_MEMORY_BYTES = 39_826_427_463
SUPPORT_CHUNK_CELLS = 16_384
MAX_SUPPORT_CELLS = 131_072
SEED_BASE = 20_261_103
START_CHECKPOINT = Path(os.environ.get(
    "CF4_R2_DA_START_CHECKPOINT",
    str(BASE / "r2_hmc_equil24h_20261013/hmc_retained_state.npz"),
))
ANCHOR_RESULT = Path(os.environ.get(
    "CF4_R2_DA_ANCHOR_RESULT",
    str(BASE / "r2_hmc_retained_block_jumps_20261102/447829/result.json"),
))


def _save(report: dict, path: Path, started: float) -> None:
    report["application_seconds"] = time.monotonic() - started
    report["host_peak_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    report["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def _load_anchor() -> tuple[np.ndarray, dict, dict]:
    provenance = json.loads(ANCHOR_RESULT.read_text())
    if (provenance.get("status") != "CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR"
            or provenance.get("source_commit") != "e623319d9c987099de93def954372dae3a3b6af9"
            or provenance.get("checkpoint_replay_passed") is not True
            or provenance.get("start_checkpoint") != str(START_CHECKPOINT)
            or provenance.get("target_orders") != dict(value=2, gradient=2, matched_primal=True)):
        raise ValueError("audited discarded state60 anchor provenance changed")
    anchor_row = next((row for row in provenance.get("sampler_trace", ())
                       if row.get("index") == 59), None)
    if anchor_row is None or anchor_row.get("accepted") is not True:
        raise ValueError("the fixed anchor is not the recorded retained state60 input")
    with np.load(START_CHECKPOINT, allow_pickle=False) as saved:
        q0 = np.asarray(saved["q"], dtype=np.float64).copy()
        recorded_value = float(saved["objective"])
        next_index = int(saved["next_index"])
        recorded_step = float(saved["step"])
        discarded = bool(saved["discard_for_inference"])
        posterior = bool(saved["posterior_sample"])
    if (q0.shape != (N_FINE_IC + 24,) or not np.isfinite(q0).all()
            or next_index != 60 or recorded_step != STEP
            or not discarded or posterior
            or abs(recorded_value - float(anchor_row["objective"])) > 1e-7):
        raise ValueError("state60 checkpoint does not match the recorded discarded anchor")
    return q0, anchor_row, provenance


def _count_score_with_floor(intensity, keys, counts, exposure):
    """Frozen training Poisson factor; epsilon is proposal-only at event logs."""
    flat = jnp.asarray(intensity).reshape(-1)
    event_rate = jnp.take(flat, keys)
    nvoxel = exposure.size
    if nvoxel == 0 or flat.size % nvoxel:
        raise ValueError("training exposure mask does not align with N128 intensity")
    n_population = flat.size // nvoxel
    exposure_mask = jnp.asarray(exposure, dtype=bool).reshape(-1)
    valid_keys = (keys >= 0) & (keys < flat.size)
    safe_voxels = jnp.clip(keys % nvoxel, 0, nvoxel - 1)
    valid = jnp.all(valid_keys & jnp.take(exposure_mask, safe_voxels))
    expected = jnp.sum(flat.reshape((n_population, nvoxel)) * exposure_mask[None, :])
    score = (jnp.sum(counts * jnp.log(event_rate + EPSILON)
                     - gammaln(counts + 1.)) - expected)
    finite_nonnegative = jnp.all(jnp.isfinite(flat)) & jnp.all(flat >= 0.)
    score = jnp.where(valid & finite_nonnegative, score, jnp.nan)
    n_floored = jnp.sum(event_rate <= EPSILON)
    return score, n_floored, jnp.min(event_rate), jnp.max(event_rate)


def _make_field(n, settings):
    run_settings = dict(settings, n=n, box_cMpc_h=BOX)
    evolve, _, conf, _, particle_mass = make_dynamics(run_settings)
    mass = jnp.full(n**3, particle_mass)

    @jax.jit
    def field(white):
        position, velocity = evolve(white)
        state = particle_grid(position, velocity, mass, conf)
        return state["rho"], jnp.moveaxis(state["mean_velocity_km_s"], -1, 0)

    return field


def _fine_value(q, field, target, source, observation, check_runtime_budget):
    check_runtime_budget()
    tic = time.monotonic()
    white = jnp.asarray(q[:N_FINE_IC])
    tracer = jnp.asarray(q[N_FINE_IC:N_FINE_IC + N_TRACER])
    population = jnp.asarray(q[N_FINE_IC + N_TRACER:])
    rho, velocity = field(white)
    packs, support = target.support(rho, velocity, tracer, 2)
    (score, components) = scalar_observation_terms(
        target, rho, velocity, tracer, population, packs, source, observation, 2,
        backend="direct_value_only",
    )
    score = float(score)
    components = np.asarray(components, dtype=np.float64).reshape(-1)
    if components.shape != (2,) or not np.isfinite(components).all():
        raise FloatingPointError("fine count/FP scalar is nonfinite")
    energy = 0.5 * float(np.dot(q, q)) - score
    if not np.isfinite(energy):
        raise FloatingPointError("fine target is nonfinite at a claimed finite-support state")
    return energy, dict(
        prior_NLL=0.5 * float(np.dot(q, q)),
        count_log_likelihood=float(components[0]),
        conditional_FP_log_likelihood=float(components[1]),
        support=support,
        seconds=time.monotonic() - tic,
        backend="direct_value_only",
        exact_target=True,
    )


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("this R2 feasibility screen must run on a Slurm GPU allocation")
    out = Path(os.environ["CF4_R2_OUT_DIR"])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + APP_SECONDS
    expected_commit = os.environ["CF4_EXPECTED_COMMIT"]
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if expected_commit != actual_commit:
        raise RuntimeError("current source revision differs from submitted pin")
    gpu_mode = os.environ["CF4_GPU_MODE"]
    if gpu_mode not in ("h200", "h100", "a100"):
        raise ValueError("unsupported GPU mode")
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    device_limit = stats.get("bytes_limit")
    if device.platform != "gpu" or gpu_mode.upper() not in str(device.device_kind).upper():
        raise RuntimeError("allocated device does not match the recorded typed GPU mode")
    if not device_limit or device_limit < MIN_DEVICE_MEMORY_BYTES:
        raise MemoryError("GPU allocation lacks20percent headroom over the measured exact-target peak")

    report = dict(
        status="R2_DA_COUNT_SURROGATE_RUNNING_NOT_POSTERIOR",
        job_id=os.environ["SLURM_JOB_ID"],
        source_commit=actual_commit,
        started_utc=datetime.now(timezone.utc).isoformat(),
        anchor_checkpoint=str(START_CHECKPOINT),
        anchor_result=str(ANCHOR_RESULT),
        N_fine=N_FINE,
        N_surrogate=N_COARSE,
        box_cMpc_h=BOX,
        fine_dx_cMpc_h=BOX / N_FINE,
        surrogate_dx_cMpc_h=BOX / N_COARSE,
        updated_coordinates="R q256 and nine tracer nuisances only",
        frozen_coordinates="orthogonal q256 complement and fifteen FP population nuisances",
        surrogate="N128 training-count-only Poisson with fixed event-log epsilon; full Gaussian prior on updated coordinates",
        surrogate_event_intensity_epsilon=EPSILON,
        surrogate_expected_rate_integral="unmodified; selected training exposure only",
        fine_target="unchanged N256 conditional-v6 training counts + 1,414 linked FP marks + all 24 standard-normal priors",
        fine_scalar_backend="direct_value_only; same registered N256 order-2 target",
        integrator="existing reversible fixed-metric Gaussian-prior split HMC",
        step_size=STEP,
        leapfrog_steps=TRAJECTORY_STEPS,
        independent_fixed_anchor_proposals=N_PROPOSALS,
        max_exact_scalar_values=MAX_FINE_VALUES,
        heldout_values_loaded=False,
        posterior_sample=False,
        posterior_uncertainty=False,
        R2_complete=False,
        R3_started=False,
        discard_for_inference=True,
        application_cap_seconds=APP_SECONDS,
        slurm_wall_cap_seconds=SLURM_WALL_SECONDS,
        resource_evidence=dict(
            selected_mode=gpu_mode,
            selected_partition=os.environ["CF4_SUBMIT_PARTITION"],
            selected_typed_gres=os.environ["CF4_SUBMIT_GRES"],
            H200_idle_typed_gpus_at_submit=int(os.environ["CF4_H200_IDLE_TYPED_GPUS"]),
            H200_free_typed_gpus_at_submit=int(os.environ["CF4_H200_FREE_TYPED_GPUS"]),
            H100_free_typed_gpus_at_submit=int(os.environ["CF4_H100_FREE_TYPED_GPUS"]),
            A100_free_typed_gpus_at_submit=int(os.environ["CF4_A100_FREE_TYPED_GPUS"]),
            device_limit_bytes=int(device_limit),
            required_device_memory_bytes=MIN_DEVICE_MEMORY_BYTES,
            measured_exact_GL2_peak_GiB=KNOWN_FINE_DEVICE_PEAK_GIB,
            host_request_GiB=(int(os.environ["SLURM_MEM_PER_NODE"]) / 1024
                              if os.environ.get("SLURM_MEM_PER_NODE") else None),
        ),
        coarse_evaluations=[],
        proposals=[],
        exact_fine_values=0,
        Q_GOAL=("sampler-cost feasibility only; no N256 posterior, uncertainty, heldout prediction or selection calibration. "
                "MW/M31 remain role-ambiguous and M33 unresolved; their observables must constrain those same roles "
                "on the same NEW evolved LG field at <=0.3 cMpc/h, with truth IDs calibration/evaluation only."),
        Q_LEAN="one fixed support-safe N128 count surrogate and one conditional Fourier block; no tuning, holdout, gravity run, or field promotion",
    )
    result_path = out / "result.json"

    def save():
        _save(report, result_path, started)

    def check_runtime_budget():
        if time.monotonic() >= deadline:
            report["stop_reason"] = "four-hour application cap"
            raise PilotBudgetStop()
        host_peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if host_peak_bytes >= 20 * 1024**3:
            report["stop_reason"] = "20GiB runtime host-RSS guard"
            raise PilotBudgetStop()

    save()
    try:
        q0, anchor_row, anchor_provenance = _load_anchor()
        report["anchor_validation"] = dict(
            start_index=60,
            saved_target_objective=float(anchor_row["objective"]),
            saved_target_terms=anchor_row["current_terms"],
            source_checkpoint_next_index=60,
            source_checkpoint_step=STEP,
            source_checkpoint_discarded=True,
            source_result_job_id=anchor_provenance["job_id"],
            source_result_target_orders=anchor_provenance["target_orders"],
        )
        rho128_unused, velocity128_unused, _, _, source, mix, observation, geometry = load_inputs()
        del rho128_unused, velocity128_unused
        radii = linked_point_conditioning_radius(observation)
        if (len(mix["PGC"]) != 1414 or len(radii) != 1414
                or mix.get("pre_reconciliation_rows") != 1414
                or len(mix.get("excluded_unresolved_PGCs", ())) != 0):
            raise ValueError("active conditional-v6 1,414-row linked cohort contract changed")
        with np.load(BASE / "r2_sky_closed_split_v6/split.npz", allow_pickle=False) as saved:
            train_keys_np = np.asarray(saved["train_keys"], dtype=np.int32)
            train_counts_np = np.asarray(saved["train_counts"], dtype=np.float64)
            heldout_voxels = np.asarray(saved["heldout_flat_voxels"])
            train_excluded = np.asarray(saved["train_window_excluded_keys"])
            heldout_excluded = np.asarray(saved["heldout_window_excluded_keys"])
        if (int(train_counts_np.sum()) != 47121 or len(train_keys_np) != 37951):
            raise ValueError("frozen v6 training count graph changed")
        exposure_np, _ = build_population_exposure_masks(
            128, heldout_voxels, train_excluded, heldout_excluded
        )
        exposure_np = np.asarray(exposure_np, dtype=bool)
        nvoxel = exposure_np.size
        if (not nvoxel or (6 * 128**3) % nvoxel
                or not np.asarray(exposure_np).reshape(-1)[train_keys_np % nvoxel].all()):
            raise ValueError("training counts fall outside the frozen training exposure")
        report.update(
            training_count_occurrences=int(train_counts_np.sum()),
            training_count_keys=len(train_keys_np),
            linked_FP_rows=len(mix["PGC"]),
            linked_radius_range_cMpc_h=[float(radii.min()), float(radii.max())],
            association_ledger_source_commit=mix["association_ledger_source_commit"],
            training_split="r2_sky_closed_split_v6; only heldout geometry loaded, no heldout outcomes",
            frozen_count_exposure_shape=list(exposure_np.shape),
            fine_raw_support_policy="rebuild from current N256 field for each stage-two candidate",
        )
        save()

        parent = BASE / "r2_n256_dynamics_profile_v1"
        parent_result = json.loads((parent / "result.json").read_text())
        if parent_result.get("status") != "N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR":
            raise ValueError("frozen N256 dynamics/cosmology settings are unavailable")
        settings = dict(parent_result["settings"])
        common = json.loads((ROOT / "config/cf4_r2_common_cosmology_v1.json").read_text())[
            "common_cosmology"
        ]
        if (settings["cosmology"]["h"] != 0.746
                or settings["cosmology"]["Om"] != common["Omega_m"]):
            raise ValueError("surrogate/fine dynamics differ from the frozen LCDM contract")

        source128 = source_geometry_at_resolution(source, N_COARSE)
        source256 = source_geometry_at_resolution(source, N_FINE)
        source128_jax = {key: jnp.asarray(value) for key, value in source128.items()}
        source256_jax = {key: jnp.asarray(value) for key, value in source256.items()}
        keys = jnp.asarray(train_keys_np)
        counts = jnp.asarray(train_counts_np)
        exposure = jnp.asarray(exposure_np)
        obs256 = ResolutionObservationTarget(
            N_FINE, source256, mix, observation, geometry, keys, counts, exposure,
            force_order=1, fine_order=2, max_support_cells=MAX_SUPPORT_CELLS,
            support_chunk_cells=SUPPORT_CHUNK_CELLS,
        )
        observation_jax = {key: jnp.asarray(value) for key, value in observation.items()}

        fine_field = _make_field(N_FINE, settings)
        coarse_field = _make_field(N_COARSE, settings)
        low0 = restrict_white_field(q0[:N_FINE_IC].reshape((N_FINE,) * 3), N_COARSE)
        tau0 = q0[N_FINE_IC:N_FINE_IC + N_TRACER].copy()
        population0 = q0[N_FINE_IC + N_TRACER:].copy()
        z0 = np.r_[low0.reshape(-1), tau0]

        def coarse_score_with_aux(z):
            white = z[:N_COARSE_IC].reshape((N_COARSE,) * 3)
            tracer = z[N_COARSE_IC:]
            rho, velocity = coarse_field(white)
            density, count_velocity = native_mass_momentum_to_count_cells(rho, velocity, BOX)
            count_velocity = jnp.moveaxis(count_velocity, 0, -1).reshape(-1, 3)
            intensity = predict_chunked_volume_intensity(
                source128_jax["positions"], count_velocity,
                tracer_masses(density, tracer), source128_jax["angular"],
                source_chunk_size=2_097_152, source_spacing=3.0, volume_order=2,
                **tracer_geometry(tracer, geometry), order=4, segments=8,
            )
            score, floored, minimum, maximum = _count_score_with_floor(
                intensity, keys, counts, exposure
            )
            potential = 0.5 * jnp.dot(z, z) - score
            return potential, (score, floored, minimum, maximum)

        value_and_gradient = jax.value_and_grad(coarse_score_with_aux, has_aux=True)
        coarse_compiled = jax.jit(value_and_gradient).lower(jnp.asarray(z0)).compile()
        compiled_memory = coarse_compiled.memory_analysis()
        baseline_stats = jax.devices()[0].memory_stats() or {}
        temp_bytes = int(getattr(compiled_memory, "temp_size_in_bytes", 0) or 0)
        output_bytes = int(getattr(compiled_memory, "output_size_in_bytes", 0) or 0)
        compiled_peak = int(baseline_stats.get("bytes_in_use", 0)) + temp_bytes + output_bytes
        report["surrogate_compiled_memory"] = dict(
            estimated_peak_bytes=compiled_peak,
            estimated_peak_GiB=compiled_peak / 1024**3,
            device_limit_bytes=int(device_limit),
            required_device_memory_bytes=MIN_DEVICE_MEMORY_BYTES,
            memory_analysis_temp_bytes=temp_bytes,
            memory_analysis_output_bytes=output_bytes,
        )
        if compiled_peak and 1.2 * compiled_peak > device_limit:
            raise MemoryError("compiled N128 surrogate lacks20percent GPU headroom")
        save()
        coarse_call_count = 0
        active_proposal = None

        def coarse_oracle(z):
            nonlocal coarse_call_count
            check_runtime_budget()
            tic = time.monotonic()
            (potential, aux), gradient = coarse_compiled(jnp.asarray(z))
            score, floored, minimum, maximum = aux
            potential = float(potential)
            gradient = np.asarray(gradient, dtype=np.float64)
            if (not np.isfinite(potential) or gradient.shape != np.asarray(z).shape
                    or not np.isfinite(gradient).all()):
                raise FloatingPointError("N128 floored count surrogate is nonfinite")
            coarse_call_count += 1
            row = dict(
                evaluation=coarse_call_count,
                proposal=active_proposal,
                surrogate_potential=potential,
                count_log_likelihood=float(score),
                event_rates_at_or_below_floor=int(floored),
                minimum_training_event_rate=float(minimum),
                maximum_training_event_rate=float(maximum),
                gradient_rms=float(np.sqrt(np.mean(gradient * gradient))),
                seconds=time.monotonic() - tic,
            )
            report["coarse_evaluations"].append(row)
            save()
            return potential, gradient

        initial_surrogate, initial_gradient = coarse_oracle(z0)
        initial_score = 0.5 * float(np.dot(z0, z0)) - initial_surrogate
        report["surrogate_anchor"] = dict(
            potential=initial_surrogate,
            count_log_likelihood=initial_score,
            floor_evaluation=report["coarse_evaluations"][-1],
        )
        save()

        fine_anchor, fine_anchor_terms = _fine_value(
            q0, fine_field, obs256, source256_jax, observation_jax, check_runtime_budget
        )
        report["exact_fine_anchor"] = dict(
            potential=fine_anchor,
            terms=fine_anchor_terms,
            saved_matched_gradient_potential=float(anchor_row["objective"]),
            absolute_difference_from_saved=float(abs(fine_anchor - anchor_row["objective"])),
        )
        if abs(fine_anchor - float(anchor_row["objective"])) > 1e-7:
            raise FloatingPointError("direct scalar anchor does not replay the unchanged N256 target")
        report["exact_fine_values"] = 1
        save()

        metric = FixedSplitMetric(
            inverse_laplacian_metric_symbol(N_COARSE, fundamental_mass=6000.0),
            np.diag(SCALES[:N_TRACER] ** 2),
        )
        report["fixed_metric"] = dict(
            fourier_family="exact tested restriction R C256 R*",
            fundamental_mass=6000.0,
            tracer_inverse_mass_diagonal=(SCALES[:N_TRACER] ** 2).tolist(),
            population_coordinates_updated=False,
            trajectory_steps=TRAJECTORY_STEPS,
            step_size=STEP,
            integrator="Gaussian-prior exact flow + likelihood-gradient kicks + surrogate Hamiltonian MH",
        )
        save()

        trace_rows = []
        for proposal_index in range(N_PROPOSALS):
            if time.monotonic() >= deadline:
                report["stop_reason"] = "application budget reached before next fixed-anchor proposal"
                break
            active_proposal = proposal_index
            seed = SEED_BASE + proposal_index * 104_729
            rng = np.random.default_rng(seed)
            started_proposal = time.monotonic()
            try:
                z_after_stage1, stage1_potential, stage1_gradient, stage1 = split_hmc_step(
                    coarse_oracle, metric, z0, initial_surrogate, initial_gradient, rng,
                    step=STEP, steps=TRAJECTORY_STEPS,
                )
            except PilotBudgetStop:
                report["stop_reason"] = report.get(
                    "stop_reason", "runtime guard interrupted surrogate trajectory; no partial proposal retained"
                )
                break
            row = dict(
                proposal=proposal_index,
                seed=seed,
                fixed_anchor=True,
                stage_one_accepted=bool(stage1["accepted"]),
                stage_one_log_acceptance=float(stage1["log_acceptance"]),
                stage_one_energy_error=float(stage1["energy_error"]),
                stage_one_surrogate_force_evaluations=int(stage1["force_evaluations"]),
                stage_two_attempted=False,
                fine_candidate_evaluated=False,
                retained=False,
                retained_coarse_white_rms=0.0,
                retained_tracer_rms=0.0,
                retained_fine_white_rms=0.0,
                retained_population_rms=0.0,
                seconds_stage_one=time.monotonic() - started_proposal,
            )
            if not stage1["accepted"]:
                row["retention_reason"] = "stage_one_rejection"
                trace_rows.append(row)
                report["proposals"] = trace_rows
                save()
                continue

            z_candidate = np.asarray(z_after_stage1, dtype=np.float64)
            coarse_candidate_score = 0.5 * float(np.dot(z_candidate, z_candidate)) - float(stage1_potential)
            low_candidate = z_candidate[:N_COARSE_IC].reshape((N_COARSE,) * 3)
            fine_white_candidate, lift_info = lift_restricted_white_field(
                q0[:N_FINE_IC].reshape((N_FINE,) * 3), low_candidate
            )
            tracer_candidate = z_candidate[N_COARSE_IC:]
            q_candidate = np.r_[fine_white_candidate.reshape(-1), tracer_candidate, population0]
            tau_delta = tracer_candidate - tau0
            q_delta = q_candidate - q0
            dz = low_candidate - low0
            row.update(
                stage_two_attempted=True,
                fine_candidate_evaluated=True,
                exact_fine_candidate_index=int(report["exact_fine_values"] + 1),
                lift_diagnostics=lift_info,
                proposed_coarse_white_rms=float(np.sqrt(np.mean(dz * dz))),
                proposed_tracer_rms=float(np.sqrt(np.mean(tau_delta * tau_delta))),
            )

            remaining = deadline - time.monotonic()
            if remaining < 600.0:
                row["fine_candidate_evaluated"] = False
                row["stage_two_attempted"] = False
                report["stop_reason"] = "less than600s reserved for an exact N256 scalar evaluation"
                trace_rows.append(row)
                report["proposals"] = trace_rows
                save()
                break

            try:
                fine_candidate, candidate_terms = _fine_value(
                    q_candidate, fine_field, obs256, source256_jax,
                    observation_jax, check_runtime_budget,
                )
            except PilotBudgetStop:
                row.update(stage_two_attempted=False, fine_candidate_evaluated=False,
                           retention_reason="stopped_before_exact_fine_evaluation")
                trace_rows.append(row)
                report["proposals"] = trace_rows
                report["stop_reason"] = report.get(
                    "stop_reason", "runtime guard before exact fine candidate"
                )
                save()
                break
            except ValueError as exc:
                message = str(exc)
                if ("empty candidate support at row" not in message
                        and "zero selected support at row" not in message):
                    raise
                fine_candidate = math.inf
                candidate_terms = dict(
                    zero_fine_density=True,
                    exact_support_error=message,
                    exact_target=True,
                )
            report["exact_fine_values"] += 1
            delta_z = low_candidate.reshape(-1) - low0.reshape(-1)
            delta_tau = tau_candidate = tracer_candidate - tau0
            prior_surrogate_delta = float(
                np.dot(z0, np.r_[delta_z, delta_tau])
                + 0.5 * np.dot(delta_z, delta_z)
                + 0.5 * np.dot(delta_tau, delta_tau)
            )
            prior_fine_delta = float(np.dot(q0, q_delta) + 0.5 * np.dot(q_delta, q_delta))
            if abs(prior_fine_delta - prior_surrogate_delta) > 1e-7 * max(
                    1.0, abs(prior_fine_delta), abs(prior_surrogate_delta)):
                raise FloatingPointError("fine/surrogate Gaussian prior increments disagree")
            surrogate_delta = prior_surrogate_delta - (coarse_candidate_score - initial_score)
            row.update(
                candidate_fine_potential=None if not np.isfinite(fine_candidate) else fine_candidate,
                candidate_fine_terms=candidate_terms,
                surrogate_candidate_count_log_likelihood=coarse_candidate_score,
                surrogate_count_log_likelihood_change=float(coarse_candidate_score - initial_score),
                fine_prior_NLL_change=prior_fine_delta,
                surrogate_prior_NLL_change=prior_surrogate_delta,
                prior_increment_identity_abs_error=float(abs(prior_fine_delta - prior_surrogate_delta)),
            )
            if np.isfinite(fine_candidate):
                fine_count_delta = (candidate_terms["count_log_likelihood"]
                                    - fine_anchor_terms["count_log_likelihood"])
                fine_fp_delta = (candidate_terms["conditional_FP_log_likelihood"]
                                 - fine_anchor_terms["conditional_FP_log_likelihood"])
                exact_delta = prior_fine_delta - fine_count_delta - fine_fp_delta
                discrepancy_delta = -fine_count_delta - fine_fp_delta + (
                    coarse_candidate_score - initial_score
                )
                identity_error = abs((exact_delta - surrogate_delta) - discrepancy_delta)
                if identity_error > 1e-7 * max(1.0, abs(exact_delta), abs(surrogate_delta)):
                    raise FloatingPointError("delayed-acceptance discrepancy decomposition failed")
                row["fine_count_log_likelihood_change"] = float(fine_count_delta)
                row["fine_FP_log_likelihood_change"] = float(fine_fp_delta)
                row["delta_fine_minus_surrogate"] = float(exact_delta - surrogate_delta)
                row["discrepancy_identity_abs_error"] = float(identity_error)
            else:
                exact_delta = math.inf
                row["fine_count_log_likelihood_change"] = None
                row["fine_FP_log_likelihood_change"] = None
                row["delta_fine_minus_surrogate"] = None
                row["discrepancy_identity_abs_error"] = None

            log_uniform = math.log(float(rng.uniform()))
            accepted, stage2_log_alpha = delayed_acceptance_stage2_accept(
                exact_delta, surrogate_delta, log_uniform
            )
            row.update(
                stage_two_log_acceptance=(stage2_log_alpha if np.isfinite(stage2_log_alpha) else None),
                stage_two_probability=float(math.exp(stage2_log_alpha)) if np.isfinite(stage2_log_alpha) else 0.0,
                stage_two_log_uniform=log_uniform,
                stage_two_accepted=bool(accepted),
                retained=bool(accepted),
                retention_reason="both_stages_accepted" if accepted else "stage_two_rejection",
                retained_coarse_white_rms=(float(np.sqrt(np.mean(dz * dz))) if accepted else 0.0),
                retained_tracer_rms=(float(np.sqrt(np.mean(tau_delta * tau_delta))) if accepted else 0.0),
                retained_fine_white_rms=(float(np.sqrt(np.mean(q_delta[:N_FINE_IC] ** 2))) if accepted else 0.0),
                retained_population_rms=0.0,
                seconds_total=time.monotonic() - started_proposal,
            )
            trace_rows.append(row)
            report["proposals"] = trace_rows
            report["accepted_stage_one"] = sum(r["stage_one_accepted"] for r in trace_rows)
            report["accepted_stage_two"] = sum(r.get("stage_two_accepted", False) for r in trace_rows)
            report["completed_fixed_anchor_proposals"] = len(trace_rows)
            save()

        report["proposals"] = trace_rows
        report["completed_fixed_anchor_proposals"] = len(trace_rows)
        report["accepted_stage_one"] = sum(r["stage_one_accepted"] for r in trace_rows)
        report["accepted_stage_two"] = sum(r.get("stage_two_accepted", False) for r in trace_rows)
        report["fine_scalar_values_used"] = int(report.get("exact_fine_values", 1))
        report["stop_reason"] = report.get("stop_reason", "fixed eight-proposal screen complete")
        report["status"] = ("R2_DA_COUNT_SURROGATE_COMPLETED_NOT_POSTERIOR"
                            if len(trace_rows) == N_PROPOSALS
                            else "R2_DA_COUNT_SURROGATE_PARTIAL_NOT_POSTERIOR")
        save()
    except Exception as exc:
        report.update(status="R2_DA_COUNT_SURROGATE_FAILED_NOT_POSTERIOR",
                      error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
