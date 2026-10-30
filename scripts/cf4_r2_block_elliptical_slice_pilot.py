"""One bounded three-block ESS feasibility sweep on the frozen R2 target.

This is not a posterior delivery or a claim of stationarity/mixing. Each
likelihood score comes from the existing matched GL2 value-and-gradient
primal; the transition does not use its gradients or the PMWD pullback. The
evolved field is cached for nuisance-only proposals, while support is rebuilt.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cf4_r2_block_elliptical_slice import (  # noqa: E402
    SliceEvaluationLimit,
    elliptical_slice_step,
    real_fourier_band_projectors,
)
from cf4_r2_conditional_map import (  # noqa: E402
    EvaluationBudgetStop,
    conditional_target_terms,
    main as run_conditional_target,
)
from cf4_r2_hmc_transport import mode_readout  # noqa: E402


N = 256
N_IC = N ** 3
N_NUISANCE = 24
CUTOFF = 64
CHECKPOINT = Path(
    "/gpfs/kjhan/CF4/z0_density/r2_hmc_equil24h_20261013/hmc_retained_state.npz"
)
SEED = 20261021
MAX_EVALUATIONS_PER_BLOCK = 8
MAX_NEW_PROPOSAL_EVALUATIONS = 3 * MAX_EVALUATIONS_PER_BLOCK
MAX_TOTAL_EVALUATIONS = 1 + MAX_NEW_PROPOSAL_EVALUATIONS  # Includes anchor replay.
APPLICATION_SECONDS = 3 * 60 * 60
MAX_HOST_RSS_GIB = 38.4  # Preserve >=20% headroom under --mem=48G.


def _score(record):
    """Return the complete count-plus-linked-mark log likelihood directly."""
    count = float(record["count_log_likelihood"])
    marks = float(record["conditional_FP_log_likelihood"])
    if np.isnan([count, marks]).any() or np.isposinf([count, marks]).any():
        raise FloatingPointError("nonfinite count/mark likelihood score")
    if np.isneginf([count, marks]).any():
        return -np.inf
    value = count + marks
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite count/mark likelihood score")
    return float(value)


def _check_target_decomposition(record, score):
    prior = (float(record["IC_prior_NLL"])
             + float(record["tracer_nuisance_prior_NLL"])
             + float(record["population_nuisance_prior_NLL"]))
    reconstructed = prior - score
    target = float(record["objective"])
    if np.isneginf(score):
        if not (np.isposinf(reconstructed) and np.isposinf(target)):
            raise AssertionError("zero-likelihood score does not match an infinite target")
        return
    if not np.isclose(reconstructed, target, rtol=0.0, atol=1e-8):
        raise AssertionError("direct likelihood terms do not reproduce the recorded full target")


def run_pilot(objective, q0, _out, report, save_report):
    started = time.monotonic()
    if len(objective.records) != 1 or not report.get("checkpoint_replay_passed"):
        raise ValueError("one exact saved-state replay must pass before ESS")
    if not np.array_equal(q0, objective._last_x):
        raise AssertionError("the pilot anchor is not the replayed target state")
    if len(q0) != N_IC + N_NUISANCE:
        raise ValueError("unexpected canonical N256 plus nuisance dimension")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2 >= MAX_HOST_RSS_GIB:
        raise MemoryError("host headroom guard reached before first proposal")

    low, high, nuisance = real_fourier_band_projectors(N, CUTOFF,
                                                        nuisance_count=N_NUISANCE)
    blocks = (low, high, nuisance)
    rng = np.random.default_rng(SEED)
    q = q0.copy()
    score_objective = objective.score_only
    loglike = _score(objective.records[0])
    _check_target_decomposition(objective.records[0], loglike)
    initial_mode_summary = mode_readout(q, N)
    report.update(
        estimand="conditional v6 N256 field-transition feasibility diagnostic, not posterior",
        Q_GOAL=(
            "one bounded transition toward the first conditional z=0 field posterior; no LG claim; "
            "MW/M31 remain ambiguous and M33 unresolved for the same NEW evolved LG field at <=0.3 cMpc/h"
        ),
        Q_LEAN=(
            "one fixed three-block ESS sweep; no sampler/angle/boundary ladder, no heldout read, "
            "no field promotion, no PM/RAMSES production run"
        ),
        diagnostic="one sequential exact conditional elliptical-slice sweep",
        anchor_checkpoint=str(CHECKPOINT),
        anchor_status="discarded dependent transport state; never a posterior draw",
        target="unchanged conditional-v6 N256 GL2 count plus 1414 linked raw-FP marks",
        target_score_source=(
            "direct count+FP terms from the matched order-2 value-and-gradient primal; ESS discards "
            "its gradients and skips the PMWD pullback; nuisance-only proposals reuse the evolved "
            "field but rebuild support; no separately compiled value-only path"
        ),
        target_gradient_used_by_transition=False,
        PMWD_pullback_used_by_transition=False,
        transition_sampler="sequential_block_elliptical_slice",
        optimizer=dict(not_run=True, reason="gradient-free ESS feasibility pilot"),
        observations=dict(training_count_rows=47121, training_count_keys=37951,
                          training_linked_FP_marks=1414),
        observed_count_grid_N=128,
        field_grid_N=N,
        fourier_partition=dict(
            rule=f"radial integer modes |k| <= {CUTOFF} versus complement",
            low_dimension=low.dimension,
            high_dimension=high.dimension,
            interpretation="fixed engineering partition, not a claim that high modes are unobserved",
        ),
        blocks=[dict(name=block.block_name, dimension=block.dimension)
                for block in blocks],
        low_band_summary_definition=(
            "existing cf4_r2_hmc_transport.mode_readout: three fundamental sine projections "
            "and orthonormal-white power means for 0<|k|<=2, 2<|k|<=4, 4<|k|<=8"
        ),
        initial_low_band_summary=initial_mode_summary,
        seed=SEED,
        max_evaluations_per_block=MAX_EVALUATIONS_PER_BLOCK,
        max_new_proposal_evaluations=MAX_NEW_PROPOSAL_EVALUATIONS,
        max_total_target_evaluations=MAX_TOTAL_EVALUATIONS,
        application_seconds_ceiling=APPLICATION_SECONDS,
        maximum_host_RSS_GiB=MAX_HOST_RSS_GIB,
        fresh_support_each_proposal=True,
        support_chunk_cells=16384,
        heldout_count_values_loaded=0,
        heldout_FP_marks_loaded=0,
        posterior_sample=False,
        posterior_uncertainty=False,
        R2_complete=False,
        LG_roles=dict(MW="ambiguous", M31="ambiguous", M33="unresolved"),
        LG_requirement=(
            "eventual observables constrain these same roles on the same NEW evolved LG field "
            "at <=0.3 cMpc/h; native truth IDs are evaluation/calibration only"
        ),
        blocks_completed=[],
        block_results=[],
        pilot_interpretation=(
            "one-anchor bounded transition feasibility only; no stationarity, mixing, ESS, "
            "field posterior, heldout prediction, or R3 authorization"
        ),
    )
    save_report()

    def evaluate_score(candidate):
        if len(objective.records) >= MAX_TOTAL_EVALUATIONS:
            raise EvaluationBudgetStop("predeclared total ESS likelihood-evaluation budget reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2 >= MAX_HOST_RSS_GIB:
            raise MemoryError("host headroom guard reached before target evaluation")
        score = score_objective(candidate)
        row = score_objective.last_record
        if not np.array_equal(candidate, score_objective.last_x):
            raise AssertionError("ESS candidate did not become the scalar-target evaluation cache")
        direct_score = _score(row)
        if not np.isclose(score, direct_score, rtol=0.0, atol=1e-10):
            raise AssertionError("scalar evaluator score differs from direct count+FP terms")
        score = direct_score
        _check_target_decomposition(row, score)
        save_report()
        return score

    for block in blocks:
        block_started = time.monotonic()
        used_before = len(objective.records)
        try:
            result = elliptical_slice_step(
                q, loglike, evaluate_score, block, rng,
                max_evaluations=MAX_EVALUATIONS_PER_BLOCK,
            )
        except (SliceEvaluationLimit, EvaluationBudgetStop) as error:
            incomplete = dict(name=block.block_name, dimension=block.dimension,
                              exact_target_evaluations_used=len(objective.records)
                              - used_before,
                              block_elapsed_seconds=time.monotonic() - block_started)
            if isinstance(error, SliceEvaluationLimit):
                incomplete.update(bracket_evaluations=error.attempts,
                                  bracket_contractions=error.bracket_contractions)
            report["block_results"].append(incomplete)
            report.update(
                status="ESS_PILOT_INCOMPLETE_BRACKET_NOT_TRANSITION",
                stopped_block=block.block_name,
                stop_reason=str(error),
                completed_blocks=len(report["blocks_completed"]),
                target_evaluations=len(objective.records),
                elapsed_seconds=time.monotonic() - started,
                host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                complete_three_block_sweep=False,
            )
            save_report()
            return

        q, loglike = result.position, result.log_likelihood
        row = dict(
            name=block.block_name,
            dimension=block.dimension,
            bracket_evaluations=result.evaluations,
            bracket_contractions=result.bracket_contractions,
            accepted_angle_radians=result.angle,
            prior_normalized_block_displacement_rms=result.block_rms_displacement,
            block_elapsed_seconds=time.monotonic() - block_started,
            exact_target_evaluations_used=len(objective.records) - used_before,
            log_likelihood_after=loglike,
            low_band_summary=mode_readout(q, N),
        )
        report["block_results"].append(row)
        report["blocks_completed"].append(block.block_name)
        report["target_evaluations"] = len(objective.records)
        report["host_peak_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        if report["host_peak_GiB"] >= MAX_HOST_RSS_GIB:
            report.update(status="ESS_PILOT_MEMORY_GUARD_STOP_NOT_POSTERIOR",
                          complete_three_block_sweep=False)
            save_report()
            return
        save_report()

    report.update(
        status="ESS_THREE_BLOCK_SWEEP_FEASIBILITY_COMPLETE_NOT_POSTERIOR",
        target_evaluations=len(objective.records),
        elapsed_seconds=time.monotonic() - started,
        host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
        complete_three_block_sweep=True,
        no_state_or_field_promoted=True,
    )
    save_report()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm GPU allocation required")
    run_conditional_target(
        start_checkpoint=CHECKPOINT,
        max_evaluations=MAX_TOTAL_EVALUATIONS,
        app_seconds=APPLICATION_SECONDS,
        support_chunk_cells=16384,
        sampling_driver=run_pilot,
        include_present_field_readout=False,
        save_best_parameters=False,
    )


if __name__ == "__main__":
    main()
