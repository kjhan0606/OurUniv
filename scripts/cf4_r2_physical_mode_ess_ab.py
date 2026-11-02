#!/usr/bin/env python3
"""One capped same-anchor exact-target ESS comparison; not posterior inference."""

import os
import resource
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cf4_r2_block_elliptical_slice import (  # noqa: E402
    SliceEvaluationLimit,
    elliptical_slice_step,
    real_fourier_physical_mode_projectors,
)
from cf4_r2_block_elliptical_slice_pilot import (  # noqa: E402
    CHECKPOINT,
    _check_target_decomposition,
    _score,
)
from cf4_r2_conditional_map import (  # noqa: E402
    EvaluationBudgetStop,
    main as run_target,
)
from cf4_r2_intensity_vjp_feasibility import _device_memory_snapshot  # noqa: E402


N, N_NUISANCE = 256, 24
N_IC = N ** 3
SEED_A, SEED_B = 20261103, 20261104
MAX_ARM_EVALUATIONS = 8
ARM_A_BUDGETS = (4, 4)
ARM_B_BUDGETS = (2, 2, 2, 2)
MAX_TARGET_EVALUATIONS = 2 + 2 * MAX_ARM_EVALUATIONS
APP_SECONDS = 3 * 60 * 60
HOST_RSS_LIMIT_GIB = 20.0
DEVICE_PEAK_BYTES = 43_799_579_648
MIN_SLOW_MODE_RMS = 0.10


def _all_ic_projector():
    size = N_IC + N_NUISANCE

    def project(vector):
        q = np.asarray(vector, dtype=np.float64)
        if q.shape != (size,) or not np.isfinite(q).all():
            raise ValueError("invalid canonical vector for all-IC projection")
        result = np.zeros_like(q)
        result[:N_IC] = q[:N_IC]
        return result

    def auxiliary(rng):
        result = np.zeros(size, dtype=np.float64)
        result[:N_IC] = rng.standard_normal(N_IC)
        return result

    project.dimension = N_IC
    project.block_name = "ic_all"
    project.draw_auxiliary = auxiliary
    return project


def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def _device_guard(device):
    stats = _device_memory_snapshot(device)
    limit, peak = stats.get("bytes_limit"), stats.get("peak_bytes_in_use")
    if limit is None or peak is None:
        raise MemoryError("allocated GPU memory limit/peak is unavailable")
    if 1.2 * max(DEVICE_PEAK_BYTES, int(peak)) > int(limit):
        raise MemoryError("allocated GPU lacks the required20% device headroom")
    return stats


def _mode_rms(projector, final, initial):
    delta = projector(final - initial)
    return float(np.linalg.norm(delta) / np.sqrt(projector.dimension))


def _run_arm(name, blocks, budgets, seed, initial, initial_score,
             slow_modes, evaluate, objective, save):
    started, first_eval = time.monotonic(), len(objective.records)
    q, score = initial.copy(), float(initial_score)
    results, complete = [], True
    rng = np.random.default_rng(seed)
    report["active_arm"] = name
    save()

    for block, budget in zip(blocks, budgets):
        before, block_started = len(objective.records), time.monotonic()
        try:
            update = elliptical_slice_step(
                q, score, evaluate, block, rng, max_evaluations=budget)
        except (SliceEvaluationLimit, EvaluationBudgetStop) as error:
            row = dict(name=block.block_name, dimension=block.dimension,
                       status="INCOMPLETE_NO_ACCEPTED_TRANSITION",
                       accepted_block_rms_displacement=0.0,
                       zero_displacement_is_reporting_only=True,
                       exact_target_evaluations=len(objective.records) - before,
                       elapsed_seconds=time.monotonic() - block_started,
                       stop_reason=str(error))
            if isinstance(error, SliceEvaluationLimit):
                row.update(bracket_evaluations=error.attempts,
                           bracket_contractions=error.bracket_contractions)
            results.append(row)
            complete = False
            break

        q, score = update.position, update.log_likelihood
        results.append(dict(
            name=block.block_name, dimension=block.dimension, status="ACCEPTED",
            bracket_evaluations=update.evaluations,
            bracket_contractions=update.bracket_contractions,
            accepted_angle_radians=update.angle,
            accepted_block_rms_displacement=update.block_rms_displacement,
            exact_target_evaluations=len(objective.records) - before,
            elapsed_seconds=time.monotonic() - block_started,
        ))

    moves = {}
    if complete:
        evaluations = len(objective.records) - first_eval
        elapsed = time.monotonic() - started
        for projector in slow_modes:
            rms = _mode_rms(projector, q, initial)
            moves[projector.block_name] = dict(
                prior_normalized_rms=rms,
                rms_per_target_evaluation=rms / evaluations,
                rms_per_second=rms / elapsed,
            )
    else:
        elapsed = time.monotonic() - started
        evaluations = len(objective.records) - first_eval
    return dict(
        arm=name, seed=seed, complete_fixed_sweep=complete,
        completed_blocks=[r["name"] for r in results if r["status"] == "ACCEPTED"],
        block_results=results, proposal_target_evaluations=evaluations,
        elapsed_seconds=elapsed, slow_mode_displacement=moves,
        endpoint_status="complete" if complete else "incomplete_not_comparable",
        no_state_or_field_promoted=True,
    )


def run_comparison(objective, q0, _out, report, save):
    if len(objective.records) != 1 or not report.get("checkpoint_replay_passed"):
        raise ValueError("matched anchor replay must pass before the A/B screen")
    if not np.array_equal(q0, objective._last_x) or len(q0) != N_IC + N_NUISANCE:
        raise ValueError("comparison anchor has an invalid shape or replay state")
    if _rss_gib() >= HOST_RSS_LIMIT_GIB:
        raise MemoryError("host RSS guard reached before the A/B screen")

    setup_started = time.monotonic()
    fundamental, slow, complement, nuisance = real_fourier_physical_mode_projectors(
        N, nuisance_count=N_NUISANCE)
    all_ic = _all_ic_projector()
    setup_seconds = time.monotonic() - setup_started
    device = jax.devices()[0]
    device_memory = _device_guard(device)
    score_objective = objective.score_only
    anchor_score = float(score_objective(q0))
    anchor_row = score_objective.last_record
    if (not np.array_equal(q0, score_objective.last_x)
            or abs(anchor_score - _score(objective.records[0])) > 1e-8):
        raise AssertionError("direct scalar anchor score does not match replay")
    _check_target_decomposition(anchor_row, anchor_score)

    report.update(
        diagnostic="one same-anchor, resource-capped ESS A/B screen; not posterior",
        anchor=str(CHECKPOINT), anchor_is_discarded_transport_state=True,
        target_score_backend="one fixed direct_value_only count+FP score in both arms",
        direct_minus_matched_anchor_score=anchor_score - _score(objective.records[0]),
        target_evaluations_include_replay_and_scalar_anchor=True,
        physical_mode_dimensions={p.block_name: p.dimension
                                  for p in (fundamental, slow, complement, nuisance)},
        arm_A_blocks=[(all_ic.block_name, all_ic.dimension, 4),
                      (nuisance.block_name, nuisance.dimension, 4)],
        arm_B_blocks=[(p.block_name, p.dimension, 2)
                      for p in (fundamental, slow, complement, nuisance)],
        arm_seeds={"A": SEED_A, "B": SEED_B},
        target_evaluation_cap=MAX_TARGET_EVALUATIONS,
        application_seconds_cap=APP_SECONDS,
        requested_host_memory_GiB=24.0,
        host_rss_limit_GiB=HOST_RSS_LIMIT_GIB,
        host_memory_planning_max_GiB=20.0,
        projector_setup_seconds=setup_seconds,
        device_memory_before_proposals=device_memory,
        resource_snapshot=dict(
            mode=os.environ["CF4_GPU_MODE"],
            partition=os.environ["CF4_SUBMIT_PARTITION"],
            typed_gres=os.environ["CF4_SUBMIT_GRES"],
            h200_free_typed_gpus=int(os.environ["CF4_H200_FREE_TYPED_GPUS"]),
            h100_free_typed_gpus=int(os.environ["CF4_H100_FREE_TYPED_GPUS"]),
            a100_free_typed_gpus=int(os.environ["CF4_A100_FREE_TYPED_GPUS"]),
        ),
        selection_conditioning_limit=(
            "conditional on linked-mark presence; not full-survey selection calibrated"),
        arm_results=[], no_state_or_field_promoted=True,
        heldout_count_values_loaded=0, heldout_FP_marks_loaded=0,
        native_truth_ids_loaded=0, posterior_sample=False,
        posterior_uncertainty=False, R2_complete=False, R3_started=False,
        LG_roles=dict(MW="ambiguous", M31="ambiguous", M33="unresolved"),
        LG_requirement=(
            "eventual observables constrain those roles on the same NEW evolved LG field "
            "at <=0.3 cMpc/h; truth IDs are calibration/evaluation only"),
        movement_rule=dict(minimum_prior_normalized_rms=MIN_SLOW_MODE_RMS,
                           interpretation="operational spending rule, not scientific threshold"),
        Q_GOAL=("transport diagnostic toward first conditional z=0 posterior; "
                "no LG or posterior claim"),
        Q_LEAN="one fixed A/B pair; no learned rank, ladder, heldout read or simulation",
    )
    save()

    def evaluate(q):
        if len(objective.records) >= MAX_TARGET_EVALUATIONS:
            raise EvaluationBudgetStop("predeclared A/B evaluation cap reached")
        if _rss_gib() >= HOST_RSS_LIMIT_GIB:
            raise MemoryError("host RSS guard reached before target evaluation")
        value = float(score_objective(q))
        row = score_objective.last_record
        if not np.array_equal(q, score_objective.last_x):
            raise AssertionError("evaluated state differs from the candidate")
        if not np.isclose(value, _score(row), rtol=0.0, atol=1e-10):
            raise AssertionError("recorded count+FP score mismatch")
        _check_target_decomposition(row, value)
        report["device_memory_after_last_proposal"] = _device_guard(device)
        if _rss_gib() >= HOST_RSS_LIMIT_GIB:
            report["status"] = "HOST_MEMORY_GUARD_STOP_NOT_POSTERIOR"
            save()
            raise MemoryError("host RSS guard reached after target evaluation")
        return value

    arms = (("A_all_IC", (all_ic, nuisance), ARM_A_BUDGETS, SEED_A),
            ("B_fixed_physical_modes", (fundamental, slow, complement, nuisance),
             ARM_B_BUDGETS, SEED_B))
    for name, blocks, budgets, seed in arms:
        result = _run_arm(name, blocks, budgets, seed, q0, anchor_score,
                          (fundamental, slow), evaluate, objective, save)
        report["arm_results"].append(result)
        report["exact_target_evaluations"] = len(objective.records)
        save()

    arm_a, arm_b = report["arm_results"]
    complete = arm_a["complete_fixed_sweep"] and arm_b["complete_fixed_sweep"]
    moves = arm_b["slow_mode_displacement"]
    movement_passed = complete and all(
        moves.get(p.block_name, {}).get("prior_normalized_rms", 0.0)
        >= MIN_SLOW_MODE_RMS for p in (fundamental, slow))
    ratios = {}
    if complete:
        for p in (fundamental, slow):
            key = p.block_name
            a = arm_a["slow_mode_displacement"][key]
            b = moves[key]
            ratios[key] = dict(
                rms_B_over_A=(b["prior_normalized_rms"] / a["prior_normalized_rms"]
                              if a["prior_normalized_rms"] > 0 else None),
                rms_per_evaluation_B_over_A=(b["rms_per_target_evaluation"]
                                             / a["rms_per_target_evaluation"]
                                             if a["rms_per_target_evaluation"] > 0 else None),
                rms_per_second_B_over_A=(b["rms_per_second"] / a["rms_per_second"]
                                         if a["rms_per_second"] > 0 else None),
            )
    report.update(
        status=("PHYSICAL_MODE_ESS_AB_COMPLETE_NOT_POSTERIOR" if movement_passed else
                ("PHYSICAL_MODE_ESS_AB_COMPLETE_LOW_MOVEMENT_STOP" if complete else
                 "PHYSICAL_MODE_ESS_AB_INCOMPLETE_STOP")),
        arm_comparison_complete=bool(complete),
        physical_mode_slow_displacements_passed=bool(movement_passed),
        arm_comparison_ratios=ratios,
        exact_target_evaluations=len(objective.records),
        candidate_passed=bool(movement_passed),
        interpretation=(
            "one discarded-anchor engineering diagnostic; no stationarity, mixing, ESS, "
            "posterior uncertainty, heldout prediction or R3 authorization"),
    )
    save()


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    mode = os.environ["CF4_GPU_MODE"].upper()
    if mode not in ("H200", "H100", "A100") or mode not in str(jax.devices()[0].device_kind).upper():
        raise RuntimeError("allocated device does not match the typed GPU request")
    run_target(
        start_checkpoint=CHECKPOINT,
        max_evaluations=MAX_TARGET_EVALUATIONS,
        app_seconds=APP_SECONDS,
        support_chunk_cells=16384,
        sampling_driver=run_comparison,
        include_present_field_readout=False,
        save_best_parameters=False,
        score_only_backend="direct_value_only",
    )


if __name__ == "__main__":
    main()
