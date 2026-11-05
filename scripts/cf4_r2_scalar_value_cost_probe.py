#!/usr/bin/env python3
"""Bounded scalar-value primal cost/repeatability probe; never a posterior."""

import numpy as np

from cf4_r2_conditional_map import TERM_KEYS, main as run_conditional_target


CHECKPOINT = "/gpfs/kjhan/CF4/z0_density/r2_hmc_equil24h_20261013/hmc_retained_state.npz"


def run_probe(objective, q0, _out, report, save_report):
    if (len(objective.records) != 1 or objective.score_only is None
            or not hasattr(objective, "score_only_matched_primal")):
        raise RuntimeError("probe requires exactly one full value-gradient anchor replay")
    q0 = np.asarray(q0, dtype=np.float64)
    gradient_anchor = dict(objective.records[0])

    matched_score = objective.score_only_matched_primal(q0)
    matched = dict(objective.score_only_matched_primal.last_record)
    score_first = objective.score_only(q0)
    first = dict(objective.score_only.last_record)

    # A tiny change to the final population nuisance coordinate invalidates
    # the score wrapper's last-point cache while preserving the exact same
    # evolved field; it is only a deterministic repeatability control.
    q_control = q0.copy()
    q_control[-1] += 1.0e-3
    score_control = objective.score_only(q_control)
    control = dict(objective.score_only.last_record)

    score_repeat = objective.score_only(q0)
    repeat = dict(objective.score_only.last_record)
    repeat_keys = (*TERM_KEYS, "total_negative_log_target")
    repeat_exact = (score_first == score_repeat and all(
        first.get(key) == repeat.get(key) for key in repeat_keys))

    value_seconds = [
        float(first["target_evaluation_seconds"]),
        float(control["target_evaluation_seconds"]),
        float(repeat["target_evaluation_seconds"]),
    ]
    scalar_terms = (*TERM_KEYS, "objective")
    delta = {
        key: float(first[key]) - float(matched[key])
        for key in scalar_terms
        if isinstance(first.get(key), (int, float))
        and isinstance(matched.get(key), (int, float))
    }
    report["status"] = ("VALUE_ONLY_REPEAT_STABLE_COST_RECORDED_NOT_POSTERIOR"
                        if repeat_exact else "VALUE_ONLY_REPEAT_MISMATCH_NOT_POSTERIOR")
    report["score_only_probe"] = dict(
        backend="direct_value_only", score_only_evaluations=4,
        total_target_evaluations=len(objective.records),
        matched_score=float(matched_score),
        repeat_q0_exact=bool(repeat_exact),
        q0_score=float(score_first), repeat_score=float(score_repeat),
        nuisance_control_delta=float(score_control - score_first),
        nuisance_control_step=1.0e-3,
        matched_score_seconds=float(matched["target_evaluation_seconds"]),
        value_only_seconds=value_seconds,
        value_only_over_matched_time=float(
            np.mean(value_seconds) / matched["target_evaluation_seconds"]),
        direct_minus_matched_terms=delta,
        gradient_anchor_seconds=float(gradient_anchor["target_evaluation_seconds"]),
        support_packing_seconds=[float(first["support_packing_seconds"]),
                                 float(control["support_packing_seconds"]),
                                 float(repeat["support_packing_seconds"])],
        no_field_saved=True, no_heldout_values_loaded=True, no_sampler_transition=True)
    save_report()


def main():
    if not __import__("os").environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm GPU allocation required")
    run_conditional_target(
        start_checkpoint=CHECKPOINT,
        max_evaluations=5,  # one anchor replay plus four scalar evaluations
        app_seconds=7200,
        support_chunk_cells=16384,
        sampling_driver=run_probe,
        include_present_field_readout=False,
        save_best_parameters=False,
        score_only_backend="direct_value_only",
    )


if __name__ == "__main__":
    main()
