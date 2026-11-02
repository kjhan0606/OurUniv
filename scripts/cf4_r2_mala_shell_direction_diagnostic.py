#!/usr/bin/env python3
"""Replay one rejected R2 shell-MALA direction and check local target gradient."""

import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cf4_r2_block_elliptical_slice import (  # noqa: E402
    real_fourier_hmc_drift_band_projector,
    real_fourier_physical_mode_projectors,
)
from cf4_r2_block_mala import block_mala_log_acceptance  # noqa: E402
from cf4_r2_conditional_map import main as run_conditional_target  # noqa: E402


N = 256
N_IC = N ** 3
N_NUISANCE = 24
MAX_EVALUATIONS = 5  # checkpoint replay + 2 value/gradient + 2 scalar values
APPLICATION_SECONDS = 45 * 60
HOST_RSS_LIMIT_GIB = 20.0
KNOWN_DEVICE_PEAK_BYTES = 43_799_579_648
FD_STEP = 1.0e-4
FD_RELATIVE_LIMIT = 1.0e-2
FUNDAMENTAL_EPSILON = 0.01
SHELL_EPSILON = 0.05
FUNDAMENTAL_SEED = 20261105
SHELL_SEED = 20261106
SOURCE38_COMMIT = "308b146d9e18031cd479750f6c3079ccaaeae9eb"
STATE38 = Path(
    "/gpfs/kjhan/CF4/z0_density/r2_hmc_long_equilibrate_20261011/"
    "hmc_retained_state.npz")
REFERENCE_RESULT = Path(
    "/gpfs/kjhan/CF4/z0_density/r2_block_mala_screen_20261102/447590/"
    "result.json")
REFERENCE_SOURCE_COMMIT = "eb0776e1b70fe03f190d90621550e77744161ecb"
REFERENCE_STATUS = "BLOCK_MALA_FIXED_SCALE_ZERO_ACCEPTANCE_IN_BLOCK_NOT_POSTERIOR"


def _rss_gib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def _device_guard(device):
    stats = device.memory_stats() or {}
    limit = stats.get("bytes_limit")
    peak = stats.get("peak_bytes_in_use")
    required = max(KNOWN_DEVICE_PEAK_BYTES, int(peak or 0))
    if limit is None or 1.2 * required > int(limit):
        raise MemoryError("allocated GPU lacks 20 percent headroom over target peak")
    return dict(bytes_limit=int(limit), peak_bytes_in_use=(None if peak is None else int(peak)),
                prior_required_peak_bytes=KNOWN_DEVICE_PEAK_BYTES,
                required_headroom_factor=1.2)


def _target_terms(record):
    return {key: float(record[key]) for key in (
        "IC_prior_NLL", "tracer_nuisance_prior_NLL",
        "population_nuisance_prior_NLL", "count_log_likelihood",
        "conditional_FP_log_likelihood")}


def _clean_row(row):
    return {key: value for key, value in row.items()
            if key not in ("new_best", "optimizer_accepted")}


def _compare_reference(name, actual, expected, *, absolute_tolerance):
    error = abs(float(actual) - float(expected))
    if error > absolute_tolerance:
        raise AssertionError(
            f"reconstructed {name} differs from recorded proposal: "
            f"actual={actual!r}, expected={expected!r}, error={error:g}")
    return error


def _compare_terms(actual, expected, *, absolute_tolerance=1e-3):
    errors = {key: abs(float(actual[key]) - float(expected[key]))
              for key in actual}
    if any(error > absolute_tolerance for error in errors.values()):
        raise AssertionError(f"reconstructed target terms differ from reference: {errors}")
    return errors


def _draw_proposal(position, gradient, projector, rng, step_size):
    noise = projector(rng.standard_normal(position.shape))
    displacement = (-0.5 * step_size ** 2 * projector(gradient)
                    + step_size * noise)
    return position + displacement, displacement


def run_diagnostic(objective, q0, _out, report, save):
    started = time.monotonic()
    if (len(objective.records) != 1 or not report.get("checkpoint_replay_passed")
            or objective._last_x is None or not np.array_equal(q0, objective._last_x)):
        raise ValueError("state38 checkpoint replay must pass before reconstruction")
    if q0.shape != (N_IC + N_NUISANCE,) or _rss_gib() >= HOST_RSS_LIMIT_GIB:
        raise ValueError("invalid canonical checkpoint shape or host RSS guard reached")

    reference = json.loads(REFERENCE_RESULT.read_text())
    expected_signature = dict(
        N=N, box_cMpc_h=384.0, conditional_count_rows=47121,
        count_training_keys=37951, conditional_FP_rows=1414,
        target_orders=dict(value=2, gradient=2, matched_primal=True))
    if (reference.get("job_id") != "447590"
            or reference.get("source_commit") != REFERENCE_SOURCE_COMMIT
            or reference.get("status") != REFERENCE_STATUS
            or reference.get("exact_target_evaluations") != 10
            or reference.get("heldout_scored") is not False
            or reference.get("posterior_sample") is not False
            or any(reference.get(key) != value
                   for key, value in expected_signature.items())):
        raise ValueError("the authoritative reference result fails its frozen contract")
    reference_anchor = next((row for row in reference.get("anchor_checkpoints", [])
                             if row.get("label") == "state38"), None)
    if (reference_anchor is None or reference_anchor.get("path") != str(STATE38)
            or reference_anchor.get("source_commit") != SOURCE38_COMMIT):
        raise ValueError("state38 reference anchor path/source does not match the frozen input")
    for key, expected in expected_signature.items():
        if report.get(key) != expected:
            raise ValueError(f"current evaluator target signature changed at {key}")
    proposals = reference.get("proposals") or []
    fund_ref = next((row for row in proposals if row.get("state") == "state38"
                     and row.get("block") == "fundamental_k2_1"
                     and row.get("step_index") == 1), None)
    shell_ref = next((row for row in proposals if row.get("state") == "state38"
                      and row.get("block") == "hmc_drift_4_lt_k_le_8"
                      and row.get("step_index") == 1), None)
    if (fund_ref is None or shell_ref is None or fund_ref.get("accepted") is not True
            or shell_ref.get("accepted") is not False):
        raise ValueError("the fixed accepted-fundamental/rejected-shell pair is absent")

    device = jax.devices()[0]
    device_memory = _device_guard(device)
    fundamental, _, _, _ = real_fourier_physical_mode_projectors(
        N, nuisance_count=N_NUISANCE)
    shell = real_fourier_hmc_drift_band_projector(N, nuisance_count=N_NUISANCE)
    if fundamental.dimension != 6 or shell.dimension != 1852:
        raise AssertionError("frozen state38 Fourier block dimensions changed")

    report.update(
        estimand="one reconstructed rejected-shell direction derivative diagnostic; not posterior",
        diagnostic="conditional-v6 fixed state38 shell proposal replay and local directional FD",
        optimizer=dict(not_run=True), sampler="none; no MH move is attempted",
        longer_warm_start_authorized=False,
        posterior_sample=False, posterior_uncertainty=False,
        heldout_count_values_loaded=0, heldout_FP_marks_loaded=0,
        heldout_scored=False, R2_complete=False, R3_started=False,
        reference_job=447590, reference_result=str(REFERENCE_RESULT),
        reference_source_commit=REFERENCE_SOURCE_COMMIT,
        state38_checkpoint=str(STATE38), state38_source_commit=SOURCE38_COMMIT,
        target_replay_passed=True,
        direction_reconstruction=dict(
            accepted_fundamental_seed=FUNDAMENTAL_SEED,
            rejected_shell_seed=SHELL_SEED,
            fundamental_dimension=fundamental.dimension,
            shell_dimension=shell.dimension,
            fundamental_epsilon=FUNDAMENTAL_EPSILON,
            shell_epsilon=SHELL_EPSILON),
        finite_difference=dict(step=FD_STEP, relative_error_denominator="max(abs(AD),1)",
                               relative_tolerance=FD_RELATIVE_LIMIT,
                               central_difference=True),
        exact_target_evaluation_cap=MAX_EVALUATIONS,
        application_seconds_cap=APPLICATION_SECONDS,
        host_RSS_guard_GiB=HOST_RSS_LIMIT_GIB,
        requested_host_memory_GiB=24,
        GPU_memory_guard=device_memory,
        slurm_resource_selection=dict(
            mode=os.environ["CF4_GPU_MODE"],
            partition=os.environ["CF4_SUBMIT_PARTITION"],
            typed_gres=os.environ["CF4_SUBMIT_GRES"],
            free_typed_gpu_snapshot=dict(
                H200=int(os.environ["CF4_H200_FREE_TYPED_GPUS"]),
                H100=int(os.environ["CF4_H100_FREE_TYPED_GPUS"]),
                A100=int(os.environ["CF4_A100_FREE_TYPED_GPUS"])),
            allocated_partition=os.environ.get("SLURM_JOB_PARTITION")),
        support_policy="unchanged GL2 live-support rule rebuilt independently at each point",
        native_truth_ids_loaded=0, field_written=False,
        LG_roles=dict(MW="ambiguous", M31="ambiguous", M33="unresolved"),
        LG_requirement=("eventual observables constrain those same roles on the same NEW evolved "
                        "LG field at <=0.3 cMpc/h; native truth IDs are calibration/evaluation only"),
        Q_GOAL=("diagnose one recorded sampler transport failure on the path to the first "
                "conditional-v6 z=0 posterior; no LG or posterior delivery"),
        Q_LEAN=("one existing rejected direction, two full value/gradient evaluations and two "
                "scalar evaluations after the mandatory replay; no epsilon ladder, retry, "
                "heldout read, separate simulation, or field promotion"),
        evaluations=[_clean_row(row) for row in objective.records],
        exact_target_evaluations=len(objective.records),
        status="SHELL_DIRECTION_DIAGNOSTIC_STARTED")

    def persist(_x, _row, _best):
        guard_error = None
        report["host_peak_GiB"] = _rss_gib()
        if report["host_peak_GiB"] >= HOST_RSS_LIMIT_GIB:
            guard_error = "host RSS guard reached after target evaluation"
        try:
            report["GPU_memory_after_evaluation"] = _device_guard(device)
        except MemoryError as error:
            guard_error = str(error)
        report["evaluations"] = [_clean_row(row) for row in objective.records]
        report["exact_target_evaluations"] = len(objective.records)
        report.pop("best_evaluation", None)
        report.pop("best_objective", None)
        objective.best = None
        save()
        if guard_error is not None:
            raise MemoryError(guard_error)

    objective.on_evaluation = persist
    report.pop("best_evaluation", None)
    report.pop("best_objective", None)
    report.pop("optimizer_coordinates", None)
    save()

    try:
        u0, g0, row0 = objective._last_result
        fund_rng = np.random.default_rng(FUNDAMENTAL_SEED)
        x, d_fund = _draw_proposal(q0, g0, fundamental, fund_rng, FUNDAMENTAL_EPSILON)
        fund_uniform = max(float(fund_rng.random()), np.finfo(np.float64).tiny)
        if float(np.linalg.norm(d_fund)) == 0.0:
            raise AssertionError("reconstructed accepted fundamental move is zero")

        ux, gx = objective(x)
        row_x = objective._last_result[2]
        replay_fund_value_error = _compare_reference(
            "accepted fundamental target value", ux, fund_ref["objective_proposed"],
            absolute_tolerance=1e-5)
        replay_fund_term_errors = _compare_terms(
            _target_terms(row_x), fund_ref["proposal_terms"])
        loga_fund, logqf_fund, logqr_fund = block_mala_log_acceptance(
            q0, u0, g0, x, ux, gx, fundamental, FUNDAMENTAL_EPSILON)
        _compare_reference("accepted fundamental log-alpha", loga_fund,
                           fund_ref["log_acceptance"], absolute_tolerance=1e-7)
        if np.log(fund_uniform) >= min(0.0, loga_fund):
            raise AssertionError("replayed fundamental RNG no longer accepts as recorded")

        shell_rng = np.random.default_rng(SHELL_SEED)
        y, d_shell = _draw_proposal(x, gx, shell, shell_rng, SHELL_EPSILON)
        shell_uniform = max(float(shell_rng.random()), np.finfo(np.float64).tiny)
        d_norm = float(np.linalg.norm(d_shell))
        if d_norm == 0.0:
            raise AssertionError("reconstructed rejected shell move is zero")

        uy, gy = objective(y)
        row_y = objective._last_result[2]
        replay_shell_value_error = _compare_reference(
            "rejected shell target value", uy, shell_ref["objective_proposed"],
            absolute_tolerance=1e-5)
        replay_shell_term_errors = _compare_terms(
            _target_terms(row_y), shell_ref["proposal_terms"])
        loga_shell, logqf_shell, logqr_shell = block_mala_log_acceptance(
            x, ux, gx, y, uy, gy, shell, SHELL_EPSILON)
        _compare_reference("rejected shell log-alpha", loga_shell,
                           shell_ref["log_acceptance"], absolute_tolerance=1e-6)
        _compare_reference("rejected shell log-q forward", logqf_shell,
                           shell_ref["log_q_forward"], absolute_tolerance=1e-6)
        _compare_reference("rejected shell log-q reverse", logqr_shell,
                           shell_ref["log_q_reverse"], absolute_tolerance=1e-6)
        if np.log(shell_uniform) < min(0.0, loga_shell):
            raise AssertionError("replayed shell RNG accepts, contrary to the recorded rejection")

        direction = d_shell / d_norm
        h = FD_STEP
        score_plus = objective.score_only(x + h * direction)
        record_plus = objective.score_only.last_record
        u_plus = float(record_plus["objective"])
        score_minus = objective.score_only(x - h * direction)
        record_minus = objective.score_only.last_record
        u_minus = float(record_minus["objective"])
        if not np.isfinite([score_plus, score_minus, u_plus, u_minus]).all():
            raise FloatingPointError("central directional score evaluations are nonfinite")
        if len(objective.records) != MAX_EVALUATIONS:
            raise AssertionError("the diagnostic did not use exactly its frozen five evaluations")

        analytic_unit = float(gx @ direction)
        finite_difference = float((u_plus - u_minus) / (2.0 * h))
        fd_relative = abs(finite_difference - analytic_unit) / max(abs(analytic_unit), 1.0)
        full_step_delta = float(uy - ux)
        first_order_delta = float(gx @ d_shell)
        endpoint_gradient_change = gy - gx
        report.update(
            status=("SHELL_DIRECTION_DIAGNOSTIC_COMPLETE_FD_PASS_NOT_POSTERIOR"
                    if fd_relative <= FD_RELATIVE_LIMIT else
                    "SHELL_DIRECTION_DIAGNOSTIC_COMPLETE_FD_DISCREPANCY_NOT_POSTERIOR"),
            all_replays_passed=True,
            exact_target_evaluations=len(objective.records),
            reference_consistency=dict(
                fundamental_value_abs_error=replay_fund_value_error,
                fundamental_target_term_abs_errors=replay_fund_term_errors,
                fundamental_log_acceptance=loga_fund,
                fundamental_log_q_forward=logqf_fund,
                fundamental_log_q_reverse=logqr_fund,
                shell_value_abs_error=replay_shell_value_error,
                shell_target_term_abs_errors=replay_shell_term_errors,
                shell_log_acceptance=loga_shell,
                shell_log_q_forward=logqf_shell,
                shell_log_q_reverse=logqr_shell,
                shell_rejection_reproduced=True),
            shell_direction=dict(
                dimension=shell.dimension,
                displacement_euclidean_norm=d_norm,
                displacement_RMS=float(d_norm / np.sqrt(shell.dimension)),
                gradient_dot_displacement=first_order_delta,
                observed_full_step_delta_U=full_step_delta,
                finite_step_minus_first_order=float(full_step_delta - first_order_delta),
                endpoint_gradient_change_l2=float(np.linalg.norm(endpoint_gradient_change)),
                endpoint_gradient_change_relative_l2=float(
                    np.linalg.norm(endpoint_gradient_change)
                    / max(np.linalg.norm(gx), np.finfo(float).tiny)),
                endpoint_gradient_change_inf=float(np.max(np.abs(endpoint_gradient_change))),
                not_a_Hessian_spectrum_estimate=True),
            symmetric_directional_derivative=dict(
                direction="reconstructed rejected shell displacement, unit Euclidean norm",
                h=h, analytic=float(analytic_unit), finite_difference=finite_difference,
                absolute_error=float(abs(finite_difference - analytic_unit)),
                relative_error=float(fd_relative), relative_tolerance=FD_RELATIVE_LIMIT,
                passed=bool(fd_relative <= FD_RELATIVE_LIMIT),
                U_plus=u_plus, U_minus=u_minus,
                plus_support_seconds=record_plus.get("support_packing_seconds"),
                minus_support_seconds=record_minus.get("support_packing_seconds")),
            elapsed_seconds=time.monotonic() - started,
            host_peak_GiB=_rss_gib(),
            no_sampler_transition_attempted=True,
            no_state_or_field_promoted=True,
            interpretation=(
                "a passing small symmetric derivative checks local consistency only; the difference "
                "over the original finite shell step is not a Hessian or posterior claim"))
        save()
    except Exception as error:
        report.update(status="SHELL_DIRECTION_DIAGNOSTIC_HARD_STOP_NOT_POSTERIOR",
                      stop_reason=repr(error),
                      exact_target_evaluations=len(objective.records),
                      elapsed_seconds=time.monotonic() - started,
                      host_peak_GiB=_rss_gib(), no_state_or_field_promoted=True)
        save()
        raise


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    mode = os.environ["CF4_GPU_MODE"].upper()
    if (mode not in ("H200", "H100", "A100")
            or mode not in str(jax.devices()[0].device_kind).upper()):
        raise RuntimeError("allocated typed GPU does not match CF4_GPU_MODE")
    _device_guard(jax.devices()[0])
    run_conditional_target(
        start_checkpoint=STATE38,
        max_evaluations=MAX_EVALUATIONS,
        app_seconds=APPLICATION_SECONDS,
        support_chunk_cells=16384,
        sampling_driver=run_diagnostic,
        include_present_field_readout=False,
        save_best_parameters=False,
    )


if __name__ == "__main__":
    main()
