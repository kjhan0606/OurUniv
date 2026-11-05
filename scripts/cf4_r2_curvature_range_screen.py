"""One capped local curvature-range falsification at the discarded R2 state 60.

This is not a posterior draw, LIS sampler, global rank estimate, or field
delivery. Every oracle call uses the unchanged conditional-v6 GL2 target.
"""

import json
import os
import resource
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_curvature_range import (  # noqa: E402
    likelihood_hessian_action,
    local_probe_schedule,
    orthonormal_random_directions,
    outside_range_energy_fraction,
    relative_action_change,
    response_range,
    support_summary_difference,
)
from cf4_r2_conditional_map import (  # noqa: E402
    EvaluationBudgetStop,
    main as run_conditional_target,
)


N = 256
N_IC = N ** 3
CHECKPOINT = Path(
    "/gpfs/kjhan/CF4/z0_density/r2_hmc_equil24h_20261013/hmc_retained_state.npz"
)
SEED = 20261021
STEP = 0.1
HALF_STEP = STEP / 2.0
SKETCH_COUNT = 4
VALIDATION_COUNT = 2
MAX_EVALUATIONS = 17
APPLICATION_SECONDS = 3 * 60 * 60
EXPECTED_SECONDS_PER_FORCE = 371.0
PAIR_SAFETY_SECONDS = 10 * 60
MAX_HOST_RSS_GIB = 38.4  # Preserve 20% headroom within the 48 GiB Slurm request.


def _record_action(objective, q0, direction, step, *, name, role, n_ic):
    plus_q = q0.copy()
    minus_q = q0.copy()
    plus_q[:n_ic] += step * direction
    minus_q[:n_ic] -= step * direction
    plus_eval = len(objective.records) + 1
    _, gradient_plus = objective(plus_q)
    minus_eval = len(objective.records) + 1
    _, gradient_minus = objective(minus_q)
    action = likelihood_hessian_action(
        gradient_plus, gradient_minus, direction, step=step, n_ic=n_ic
    )
    row = dict(
        name=name,
        role=role,
        step=float(step),
        plus_evaluation=int(plus_eval),
        minus_evaluation=int(minus_eval),
        gradient_difference_norm=float(np.linalg.norm(gradient_plus - gradient_minus)),
        likelihood_Hessian_action_IC_norm=float(np.linalg.norm(action[:n_ic])),
        likelihood_Hessian_action_nuisance_cross_norm=float(np.linalg.norm(action[n_ic:])),
        nuisance_cross_response=action[n_ic:].tolist(),
        support_summary_difference=support_summary_difference(
            json.loads(objective.records[plus_eval - 1]["support"]),
            json.loads(objective.records[minus_eval - 1]["support"]),
        ),
    )
    return action, row


def run_probe(objective, q0, _out, report, save_report):
    """Called by the existing N256 target builder after its checkpoint replay."""
    with np.load(CHECKPOINT, allow_pickle=False) as saved:
        saved_q = np.asarray(saved["q"], dtype=np.float64)
        saved_gradient = np.asarray(saved["gradient"], dtype=np.float64)
        next_index = int(saved["next_index"])
    if next_index != 60 or not np.array_equal(q0, saved_q):
        raise ValueError("curvature anchor is not the frozen terminal state 60")
    if len(objective.records) != 1 or not report.get("checkpoint_replay_passed"):
        raise ValueError("one exact checkpoint replay must pass before the probe")
    _, replay_gradient = objective(q0)  # served from the exact cache
    if not np.allclose(replay_gradient, saved_gradient, rtol=1e-4, atol=1e-4):
        raise AssertionError("saved GL2 gradient and replay differ")
    replay_error = float(np.linalg.norm(replay_gradient - saved_gradient))
    replay_scale = max(float(np.linalg.norm(saved_gradient)), 1.0)
    def noise_proxy(step):
        return max(
            replay_error / step,
            np.finfo(np.float64).eps * replay_scale / step,
        )
    report.update(
        diagnostic="single-anchor prior-whitened local likelihood-curvature range screen",
        anchor_checkpoint=str(CHECKPOINT),
        anchor_next_index=next_index,
        anchor_status="discard_for_inference; transport diagnostic only",
        exact_target="unchanged conditional v6 GL2 value/gradient; fresh 16384-cell support blocks",
        prior_coordinates="IC plus all 24 nuisance coordinates are standard normal; directions perturb IC only",
        heldout_values_loaded=0,
        posterior_sample=False,
        posterior_uncertainty=False,
        R2_complete=False,
        h=float(STEP),
        half_step=float(HALF_STEP),
        rng_seed=SEED,
        sketch_direction_count=SKETCH_COUNT,
        validation_direction_count=VALIDATION_COUNT,
        exact_target_evaluation_ceiling=MAX_EVALUATIONS,
        application_seconds_ceiling=APPLICATION_SECONDS,
        maximum_host_RSS_GiB=MAX_HOST_RSS_GIB,
        additional_host_workspace_limit_GiB=4.0,
        expected_seconds_per_force_advisory=EXPECTED_SECONDS_PER_FORCE,
        checkpoint_replay_gradient_relative_error=replay_error / replay_scale,
        finite_difference_noise_proxy_at_h=noise_proxy(STEP),
        local_screen_thresholds=dict(
            max_relative_h_step_action_change=0.05,
            max_validation_outside_squared_energy_fraction=0.10,
            min_action_to_replay_noise_proxy=10.0,
        ),
        result_interpretation=(
            "local engineering falsification only; does not estimate posterior covariance, "
            "global Hessian rank, stationarity, or authorize LIS sampling"
        ),
        MW_role="ambiguous",
        M31_role="ambiguous",
        M33_role="unresolved",
        LG_requirement=(
            "MW/M31/M33 observables must constrain these roles on the same NEW evolved "
            "LG field at <=0.3 cMpc/h; truth IDs are evaluation/calibration only"
        ),
        Q_GOAL=(
            "bounded feasibility measurement toward the conditional-v6 z=0 density/velocity "
            "posterior in the 384 cMpc/h box; not a full-survey calibrated CF4 result or field delivery"
        ),
        Q_LEAN=(
            "one anchor, 17 exact gradients maximum, existing GL2 oracle, no new sampler, "
            "no new denominator search or heldout score, no standalone PMWD or RAMSES run"
        ),
        optimizer=dict(
            method="none",
            diagnostic="matrix-free centered finite-difference likelihood-Hessian actions",
            max_exact_target_evaluations=MAX_EVALUATIONS,
            application_seconds=APPLICATION_SECONDS,
        ),
        submitted_partition=os.environ.get("CF4_SUBMIT_PARTITION"),
        submitted_typed_gres=os.environ.get("CF4_SUBMIT_GRES"),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
        slurm_job_gres=os.environ.get("SLURM_JOB_GRES"),
        curvature_actions=[],
    )
    save_report()

    directions = orthonormal_random_directions(
        N_IC, SKETCH_COUNT + VALIDATION_COUNT, seed=SEED
    )
    responses = {}
    action_rows = []
    finite_difference_rows = {}

    def pair_stop_reason():
        remaining_calls = objective.max_evaluations - len(objective.records)
        remaining_seconds = objective.deadline - time.monotonic()
        peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        report["max_host_RSS_GiB_seen"] = float(peak_rss_gib)
        if remaining_calls < 2:
            return "exact-target evaluation budget cannot cover another complete +/- pair"
        if remaining_seconds < 2 * EXPECTED_SECONDS_PER_FORCE + PAIR_SAFETY_SECONDS:
            return "application walltime guard cannot cover another complete +/- pair"
        if peak_rss_gib >= MAX_HOST_RSS_GIB:
            return "host RSS reached the 80%-of-request stop guard"
        return None

    probe_schedule = local_probe_schedule(directions, step=STEP)
    for name, role, direction, step in probe_schedule:
        stop_reason = pair_stop_reason()
        if stop_reason is not None:
            report.update(
                status="INCOMPLETE_RESOURCE_GUARD",
                local_screen_verdict="NO VERDICT; stopped before the next +/- pair",
                resource_stop_reason=stop_reason,
                exact_target_evaluations=len(objective.records),
                curvature_actions=action_rows,
            )
            save_report()
            return
        try:
            action, row = _record_action(
                objective, q0, direction, step, name=name, role=role, n_ic=N_IC
            )
        except EvaluationBudgetStop:
            report.update(
                status="INCOMPLETE_RESOURCE_GUARD",
                local_screen_verdict="NO VERDICT; target-call or walltime ceiling reached",
                exact_target_evaluations=len(objective.records),
                curvature_actions=action_rows,
            )
            save_report()
            return
        except Exception as exc:
            report.update(
                status="FAILED_CLOSED_CURVATURE_RANGE_SCREEN",
                failure=f"{type(exc).__name__}: {exc}",
                exact_target_evaluations=len(objective.records),
                local_screen_verdict="NO VERDICT; failed closed",
                curvature_actions=action_rows,
            )
            save_report()
            raise
        signal_floor = 10.0 * noise_proxy(step)
        row["finite_difference_noise_proxy"] = noise_proxy(step)
        row["action_signal_to_replay_noise_proxy"] = (
            row["likelihood_Hessian_action_IC_norm"] / signal_floor
            if signal_floor > 0 else None
        )
        row["signal_above_noise_gate"] = (
            row["likelihood_Hessian_action_IC_norm"] > signal_floor
        )
        action_rows.append(row)
        finite_difference_rows[name] = action
        if role == "range_sketch":
            responses[name] = action[:N_IC]
        elif role == "independent_random_direction_validation":
            responses[name] = action[:N_IC]
        peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        report["max_host_RSS_GiB_seen"] = float(peak_rss_gib)
        save_report()
        if peak_rss_gib >= MAX_HOST_RSS_GIB:
            report.update(
                status="INCOMPLETE_RESOURCE_GUARD",
                local_screen_verdict="NO VERDICT; host RSS reached the 80%-of-request stop guard",
                resource_stop_reason="host RSS reached 38.4 GiB after a completed +/- pair",
                exact_target_evaluations=len(objective.records),
                curvature_actions=action_rows,
            )
            save_report()
            return

    if objective.deadline - time.monotonic() < 300:
        report.update(
            status="INCOMPLETE_RESOURCE_GUARD",
            local_screen_verdict="NO VERDICT; less than five minutes remain for range checks",
            resource_stop_reason="post-evaluation linear algebra reserve was exhausted",
            exact_target_evaluations=len(objective.records),
            curvature_actions=action_rows,
        )
        save_report()
        return

    sketch_responses = [responses[f"sketch_{i + 1}"] for i in range(SKETCH_COUNT)]
    basis, singular_values = response_range(sketch_responses)
    validation_rows = []
    for i in range(VALIDATION_COUNT):
        name = f"validation_{i + 1}"
        response = responses[name]
        fraction = outside_range_energy_fraction(response, basis)
        validation_rows.append(dict(
            direction=name,
            response_norm=float(np.linalg.norm(response)),
            outside_range_squared_energy_fraction=fraction,
            passes_local_10pct_screen=(fraction is not None and fraction <= 0.10),
        ))

    stability_rows = []
    for base_name, half_name in (("sketch_1", "sketch_1_half_step"),
                                 ("validation_1", "validation_1_half_step")):
        base_action = finite_difference_rows[base_name]
        half_action = finite_difference_rows[half_name]
        change_ic = relative_action_change(base_action[:N_IC], half_action[:N_IC])
        change_joint = relative_action_change(base_action, half_action)
        stability_rows.append(dict(
            direction=base_name,
            IC_action_relative_change=change_ic,
            joint_action_relative_change=change_joint,
            passes_local_5pct_screen=(change_ic <= 0.05 and change_joint <= 0.05),
        ))

    signal_ok = all(row["signal_above_noise_gate"] for row in action_rows)
    stable = all(row["passes_local_5pct_screen"] for row in stability_rows)
    validation_ok = all(row["passes_local_10pct_screen"] for row in validation_rows)
    if not signal_ok or not stable:
        verdict = "INCONCLUSIVE_NUMERICAL_OR_STEP_INSTABILITY_STOP"
    elif not validation_ok:
        verdict = "NO_GO_LOCAL_RANK4_RANGE_FALSIFIED"
    else:
        verdict = "PASS_LOCAL_SCREEN_ONLY_REQUIRES_SEPARATE_EXACT_PROPOSAL_DESIGN"

    report.update(
        status="LOCAL_CURVATURE_RANGE_SCREEN_COMPLETE_NOT_POSTERIOR",
        exact_target_evaluations=len(objective.records),
        sketch_numerical_rank=len(basis),
        sketch_singular_values=singular_values.tolist(),
        sketch_response_norms=[float(np.linalg.norm(v)) for v in sketch_responses],
        validation=validation_rows,
        step_stability=stability_rows,
        local_screen_verdict=verdict,
        post_probe_action=(
            "no automatic sampler follow-up; even a pass permits only separate design of a "
            "full-target-corrected proposal, not inference"
        ),
        curvature_actions=action_rows,
        max_additional_host_workspace_GiB=4.0,
        maximum_host_RSS_GiB=MAX_HOST_RSS_GIB,
        dense_hessian_formed=False,
        likelihood_hessian_indefiniteness_preserved=True,
    )
    save_report()


if __name__ == "__main__":
    run_conditional_target(
        start_checkpoint=CHECKPOINT,
        max_evaluations=MAX_EVALUATIONS,
        app_seconds=APPLICATION_SECONDS,
        support_chunk_cells=16384,
        include_present_field_readout=False,
        save_best_parameters=False,
        sampling_driver=run_probe,
    )
