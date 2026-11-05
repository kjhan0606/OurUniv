#!/usr/bin/env python3
"""Two-state fixed-scale conditional MALA screen; not posterior inference."""

import json
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
    real_fourier_hmc_drift_band_projector,
)
from cf4_r2_block_mala import block_mala_step  # noqa: E402
from cf4_r2_conditional_map import (  # noqa: E402
    EvaluationBudgetStop,
    main as run_conditional_target,
)


N = 256
N_IC = N ** 3
N_NUISANCE = 24
MAX_EVALUATIONS = 10
APPLICATION_SECONDS = 2 * 60 * 60
HOST_RSS_LIMIT_GIB = 20.0
KNOWN_DEVICE_PEAK_BYTES = 43_799_579_648
CHECKPOINTS = (
    ("state38", Path(
        "/gpfs/kjhan/CF4/z0_density/r2_hmc_long_equilibrate_20261011/"
        "hmc_retained_state.npz"),
     "308b146d9e18031cd479750f6c3079ccaaeae9eb", 20261105, 20261106),
    ("state60", Path(
        "/gpfs/kjhan/CF4/z0_density/r2_hmc_equil24h_20261013/"
        "hmc_retained_state.npz"),
     "5016282433a375ac9dd89660a10b1ae53d25a370", 20261107, 20261108),
)
FUNDAMENTAL_EPSILON = 0.01
HMC_DRIFT_BAND_EPSILON = 0.05
MAX_PROPOSALS_PER_BLOCK_STATE = 2


def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def _device_snapshot_and_guard(device):
    stats = device.memory_stats() or {}
    limit = stats.get("bytes_limit")
    peak = stats.get("peak_bytes_in_use")
    if limit is None:
        raise MemoryError("allocated GPU memory limit is unavailable")
    required_peak = max(KNOWN_DEVICE_PEAK_BYTES, int(peak or 0))
    if 1.2 * required_peak > int(limit):
        raise MemoryError("allocated GPU lacks 20 percent headroom over recorded target peak")
    return dict(
        bytes_limit=int(limit),
        current_bytes_in_use=(None if stats.get("bytes_in_use") is None
                              else int(stats["bytes_in_use"])),
        peak_bytes_in_use=(None if peak is None else int(peak)),
        prior_required_peak_bytes=KNOWN_DEVICE_PEAK_BYTES,
        required_headroom_factor=1.2,
    )


def _target_signature(result):
    fingerprint = result.get("sampler_fingerprint", {})
    return dict(
        N=result.get("N"),
        box_cMpc_h=result.get("box_cMpc_h"),
        count_rows=result.get("conditional_count_rows"),
        count_keys=result.get("count_training_keys"),
        FP_rows=result.get("conditional_FP_rows"),
        target_orders=result.get("target_orders"),
        target_rules=result.get("target_rules"),
        source_conditioning_rule=result.get("source_conditioning_rule"),
        association_ledger_source_commit=result.get("association_ledger_source_commit"),
        split=fingerprint.get("split"),
    )


def _read_checkpoint_contract(path, expected_commit):
    result_path = path.parent / "result.json"
    if not path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"missing retained checkpoint/result: {path}")
    result = json.loads(result_path.read_text())
    if (result.get("status") != "CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR"
            or result.get("source_commit") != expected_commit
            or result.get("N") != N or result.get("box_cMpc_h") != 384.0
            or result.get("conditional_count_rows") != 47121
            or result.get("count_training_keys") != 37951
            or result.get("conditional_FP_rows") != 1414
            or result.get("target_orders") != dict(value=2, gradient=2, matched_primal=True)
            or result.get("checkpoint_replay_passed") is not True
            or result.get("heldout_count_values_loaded") != 0
            or result.get("heldout_FP_marks_loaded") != 0
            or result.get("heldout_scored") is not False):
        raise ValueError(f"retained transport state fails fixed target/heldout contract: {path}")
    trace = result.get("sampler_trace") or []
    if not trace:
        raise ValueError(f"retained transport state has no sampler trace: {path}")
    terminal = trace[-1]
    if (not np.isfinite(float(terminal["objective"]))
            or not isinstance(terminal.get("current_terms"), dict)):
        raise ValueError(f"retained state terminal trace is malformed: {path}")
    return result


def _load_checkpoint_payload(path, result, *, load_position):
    """Load one checkpoint only when its state is about to be tested."""
    trace = result["sampler_trace"][-1]
    with np.load(path, allow_pickle=False) as saved:
        q = (np.asarray(saved["q"], dtype=np.float64).copy()
             if load_position else None)
        gradient = np.asarray(saved["gradient"], dtype=np.float64).copy()
        objective = float(saved["objective"])
        terms = json.loads(str(saved["current_terms"]))
        posterior_sample = bool(saved["posterior_sample"])
        discard_for_inference = bool(saved["discard_for_inference"])
    shape_ok = gradient.shape == (N_IC + N_NUISANCE,)
    if q is not None:
        shape_ok = shape_ok and q.shape == gradient.shape and np.isfinite(q).all()
    if (not shape_ok or not np.isfinite(gradient).all() or posterior_sample
            or not discard_for_inference
            or abs(objective - float(trace["objective"])) > 1e-6
            or terms != trace["current_terms"]):
        raise ValueError(f"retained state payload/trace mismatch: {path}")
    return q, gradient, objective, terms


def _clean_evaluation(row):
    return {key: value for key, value in row.items()
            if key not in ("new_best", "optimizer_accepted")}


def _target_terms(record):
    return {key: float(record[key]) for key in (
        "IC_prior_NLL", "tracer_nuisance_prior_NLL",
        "population_nuisance_prior_NLL", "count_log_likelihood",
        "conditional_FP_log_likelihood")}


def _delta_terms(before, proposed):
    delta = dict(
        IC_prior_NLL=proposed["IC_prior_NLL"] - before["IC_prior_NLL"],
        tracer_nuisance_prior_NLL=(proposed["tracer_nuisance_prior_NLL"]
                                   - before["tracer_nuisance_prior_NLL"]),
        population_nuisance_prior_NLL=(proposed["population_nuisance_prior_NLL"]
                                       - before["population_nuisance_prior_NLL"]),
        count_NLL=-(proposed["count_log_likelihood"] - before["count_log_likelihood"]),
        conditional_FP_NLL=-(proposed["conditional_FP_log_likelihood"]
                             - before["conditional_FP_log_likelihood"]),
    )
    delta["sum"] = float(sum(delta.values()))
    return delta


def run_screen(objective, q0, _out, report, save):
    started = time.monotonic()
    if (len(objective.records) != 1 or not report.get("checkpoint_replay_passed")
            or not np.array_equal(q0, objective._last_x)):
        raise ValueError("state38 matched target replay must pass before MALA screen")
    if q0.shape != (N_IC + N_NUISANCE,):
        raise ValueError("unexpected canonical N256 plus 24 nuisance dimension")
    if _rss_gib() >= HOST_RSS_LIMIT_GIB:
        raise MemoryError("host RSS guard reached before first MALA proposal")

    device = jax.devices()[0]
    device_memory = _device_snapshot_and_guard(device)
    # The HMC-drift shell is 16 < k^2 <= 64 (4 < |k| <= 8), distinct
    # from the 66-coordinate k^2=4..8 block used by the closed ESS pilot.
    from cf4_r2_block_elliptical_slice import real_fourier_physical_mode_projectors
    fundamental, _, _, _ = real_fourier_physical_mode_projectors(
        N, nuisance_count=N_NUISANCE)
    drift_band = real_fourier_hmc_drift_band_projector(
        N, nuisance_count=N_NUISANCE)
    blocks = (("fundamental_k2_1", fundamental, FUNDAMENTAL_EPSILON),
              ("hmc_drift_4_lt_k_le_8", drift_band, HMC_DRIFT_BAND_EPSILON))
    if (fundamental.dimension != 6 or drift_band.dimension != 1852
            or fundamental.dimension + drift_band.dimension >= N_IC):
        raise AssertionError("physical block projector dimensions changed")

    state_meta = []
    for label, path, source_commit, seed_fund, seed_band in CHECKPOINTS:
        result_meta = _read_checkpoint_contract(path, source_commit)
        state_meta.append(dict(label=label, path=path, source_commit=source_commit,
                               result=result_meta, seed_fundamental=seed_fund,
                               seed_hmc_drift_band=seed_band))
    signatures = [_target_signature(item["result"]) for item in state_meta]
    if signatures[0] != signatures[1]:
        raise ValueError("the two discarded anchors do not share one frozen target signature")

    # Remove MAP boilerplate inherited from the common objective runner. Its
    # in-memory best point is neither selected nor written by this diagnostic.
    report["estimand"] = (
        "discarded-anchor block-MALA transport screen under the frozen conditional-v6 N256 target")
    report["optimizer"] = dict(not_run=True, reason="fixed-scale Metropolis MALA screen")
    report["sampler"] = "two-state fixed-block MALA; not posterior inference"
    report["diagnostic"] = "conditional-v6 full-target block-MALA engineering screen"
    report["longer_warm_start_authorized"] = False
    report.pop("optimizer_coordinates", None)
    report.pop("best_evaluation", None)
    report.pop("best_objective", None)
    report.update(
        checkpoint_replay_passed=True,
        target_signature=signatures[0],
        anchor_checkpoints=[dict(label=item["label"], path=str(item["path"]),
                                 source_commit=item["source_commit"],
                                 result_status=item["result"]["status"])
                            for item in state_meta],
        anchor_states_are_dependent=True,
        anchor_states_are_posterior_draws=False,
        transition_blocks=[dict(name=name, dimension=projector.dimension,
                                step_size=epsilon)
                           for name, projector, epsilon in blocks],
        rng_seeds={item["label"]: dict(
            fundamental=item["seed_fundamental"],
            hmc_drift_band=item["seed_hmc_drift_band"])
            for item in state_meta},
        max_proposals_per_block_state=MAX_PROPOSALS_PER_BLOCK_STATE,
        exact_target_evaluation_cap=MAX_EVALUATIONS,
        application_seconds_cap=APPLICATION_SECONDS,
        host_RSS_guard_GiB=HOST_RSS_LIMIT_GIB,
        requested_host_memory_GiB=24,
        GPU_memory_guard=device_memory,
        support_policy=("unchanged GL2 live-support rule rebuilt at every distinct candidate; "
                        "support remains an existing numerical approximation"),
        heldout_count_values_loaded=0,
        heldout_FP_marks_loaded=0,
        heldout_scored=False,
        native_truth_ids_loaded=0,
        posterior_sample=False,
        posterior_uncertainty=False,
        field_written=False,
        R2_complete=False,
        R3_started=False,
        LG_roles=dict(MW="ambiguous", M31="ambiguous", M33="unresolved"),
        LG_requirement=(
            "eventual observables constrain these same roles on the same NEW evolved LG field "
            "at <=0.3 cMpc/h; native truth IDs are calibration/evaluation only"),
        proposals=[],
        state_results=[],
        proposal_acceptance_gate=(
            "at least one accepted move in each targeted block across the two dependent states "
            "is necessary only to consider a later sampler assessment; not mixing/posterior evidence"),
        Q_GOAL=(
            "fixed transport test on route to conditional-v6 N256 z=0 posterior; no LG result"),
        Q_LEAN=(
            "two states, two physical blocks, eight proposals maximum, unchanged target/support "
            "and resource caps; no ladder/heldout/separate simulation/field promotion"),
    )

    def persist_screen_evaluation(_x, _row, _best):
        report["evaluations"] = [_clean_evaluation(r) for r in objective.records]
        report["exact_target_evaluations"] = len(objective.records)
        report.pop("best_evaluation", None)
        report.pop("best_objective", None)
        objective.best = None
        save()

    objective.on_evaluation = persist_screen_evaluation
    report["evaluations"] = [_clean_evaluation(r) for r in objective.records]
    report["exact_target_evaluations"] = len(objective.records)
    save()

    try:
        # state38 was replayed by the common target runner. Verify value, terms,
        # and cached gradient against the preserved checkpoint explicitly.
        first = state_meta[0]
        label = first["label"]
        _q_unused, g_saved, saved_u, saved_terms = _load_checkpoint_payload(
            first["path"], first["result"], load_position=False)
        value, gradient, record = objective._last_result
        if (abs(float(value) - saved_u) > 1e-8 * max(1.0, abs(saved_u))
                or max(abs(_target_terms(record)[k] - float(saved_terms[k]))
                       for k in saved_terms) > 1e-3):
            raise AssertionError("state38 replay value/terms differ from retained checkpoint")
        gradient_abs = float(np.max(np.abs(gradient - g_saved)))
        gradient_rel = float(np.linalg.norm(gradient - g_saved)
                             / max(np.linalg.norm(g_saved), np.finfo(float).tiny))
        if not np.allclose(gradient, g_saved, rtol=1e-4, atol=1e-4):
            raise AssertionError("state38 replay gradient differs from retained checkpoint")
        state_payload = [dict(label=label, checkpoint=str(CHECKPOINTS[0][1]),
                              replay_value_abs_error=abs(float(value) - saved_u),
                              replay_gradient_max_abs_error=gradient_abs,
                              replay_gradient_relative_l2_error=gradient_rel,
                              target_evaluations=1, proposals=[])]
        report["state_results"].append(state_payload[0])
        save()
        q_current = q0.copy()
        u_current = float(value)
        g_current = np.asarray(gradient, dtype=np.float64).copy()
        terms_current = _target_terms(record)

        def run_block(state_label, state_row, block_name, projector, epsilon, seed,
                      q_start, u_start, gradient_start, terms_start):
            q_block = np.asarray(q_start, dtype=np.float64).copy()
            u_block = float(u_start)
            g_block = np.asarray(gradient_start, dtype=np.float64).copy()
            t_block = dict(terms_start)
            rng = np.random.default_rng(seed)
            accepted = 0
            for step_index in range(1, MAX_PROPOSALS_PER_BLOCK_STATE + 1):
                if _rss_gib() >= HOST_RSS_LIMIT_GIB:
                    raise MemoryError("host RSS guard reached before MALA proposal")
                before_u, before_terms = u_block, dict(t_block)
                started_eval = time.monotonic()

                def evaluate(candidate):
                    if len(objective.records) >= MAX_EVALUATIONS:
                        raise EvaluationBudgetStop("fixed 10-evaluation MALA screen cap reached")
                    return objective(candidate)

                try:
                    result = block_mala_step(
                        q_block, u_block, g_block, evaluate, projector, rng,
                        step_size=epsilon)
                except EvaluationBudgetStop:
                    raise
                if (objective._last_x is None or not np.array_equal(
                        result.proposed_position, objective._last_x)):
                    raise AssertionError("MALA candidate is absent from target cache")
                proposed_value, proposed_gradient, proposed_record = objective._last_result
                proposed_terms = _target_terms(proposed_record)
                if abs(float(proposed_value) - result.proposed_potential) > 1e-10:
                    raise AssertionError("MALA candidate potential differs from target cache")
                if not np.array_equal(proposed_gradient, result.proposed_gradient):
                    raise AssertionError("MALA candidate gradient differs from target cache")
                target_eval_seconds = float(proposed_record.get(
                    "target_evaluation_seconds", time.monotonic() - started_eval))
                support_seconds = float(proposed_record.get("support_packing_seconds", 0.0))
                noise_component_rms = float(
                    np.linalg.norm(projector(result.proposed_position - q_block
                                  + 0.5 * epsilon ** 2 * projector(g_block)))
                    / (epsilon * np.sqrt(projector.dimension)))
                delta_components = _delta_terms(before_terms, proposed_terms)
                delta_components["closure_residual"] = float(
                    delta_components["sum"]
                    - (result.proposed_potential - before_u))
                proposal_row = dict(
                    state=state_label, block=block_name, step_index=step_index,
                    rng_seed=seed, step_size=epsilon, block_dimension=projector.dimension,
                    objective_before=before_u,
                    objective_proposed=float(result.proposed_potential),
                    delta_U_proposed=float(result.proposed_potential - before_u),
                    delta_U_components_proposed=delta_components,
                    log_q_forward=result.log_q_forward,
                    log_q_reverse=result.log_q_reverse,
                    log_acceptance=result.log_acceptance,
                    accepted=result.accepted,
                    proposal_displacement_rms=result.proposal_displacement_rms,
                    accepted_displacement_rms=result.accepted_displacement_rms,
                    retained_displacement_rms=result.retained_displacement_rms,
                    projected_gradient_rms=float(np.linalg.norm(projector(g_block))
                                                 / np.sqrt(projector.dimension)),
                    projected_noise_rms=noise_component_rms,
                    target_evaluation_seconds=target_eval_seconds,
                    support_packing_seconds=support_seconds,
                    proposal_terms=proposed_terms,
                    retained_terms=(proposed_terms if result.accepted else before_terms),
                    retained_objective=float(result.potential),
                    exact_target_evaluations=len(objective.records),
                )
                proposal_row["projected_noise_component_rms"] = noise_component_rms
                report["proposals"].append(proposal_row)
                state_row["proposals"].append(proposal_row)
                if result.accepted:
                    accepted += 1
                    q_block = result.position
                    u_block = result.potential
                    g_block = result.gradient
                    t_block = proposed_terms
                report["exact_target_evaluations"] = len(objective.records)
                report["host_peak_GiB"] = _rss_gib()
                report["GPU_memory_after_proposal"] = _device_snapshot_and_guard(device)
                state_row["target_evaluations"] = len(objective.records)
                state_row.setdefault("completed_proposals_by_block", {})[block_name] = step_index
                state_row.setdefault("accepted_by_block", {})[block_name] = accepted
                save()
            return q_block, u_block, g_block, t_block, accepted

        accepted_by_state_block = {}
        for block_name, projector, epsilon in blocks:
            seed = (first["seed_fundamental"] if block_name == "fundamental_k2_1"
                    else first["seed_hmc_drift_band"])
            q_current, u_current, g_current, terms_current, count = run_block(
                label, state_payload[0], block_name, projector, epsilon, seed,
                q_current, u_current, g_current, terms_current)
            accepted_by_state_block[(label, block_name)] = count
        state_payload[0]["completed"] = True
        state_payload[0]["accepted_by_block"] = {
            block: accepted_by_state_block[(label, block)] for block, _, _ in blocks}
        state_payload[0]["final_objective"] = u_current
        state_payload[0]["final_terms"] = terms_current
        state_payload[0]["target_evaluations"] = len(objective.records)
        save()

        # Release state38 vectors and objective cache before loading state60.
        del g_saved, q_current, g_current, terms_current, record
        objective._last_x = None
        objective._last_result = None
        objective.best = None

        second = state_meta[1]
        label, checkpoint = second["label"], second["path"]
        # Load state60 only after all state38 moves and arrays have been released.
        q_current, g_saved, stored_u, stored_terms = _load_checkpoint_payload(
            checkpoint, second["result"], load_position=True)
        value, gradient = objective(q_current)
        replay_record = objective._last_result[2]
        if (abs(float(value) - stored_u) > 1e-8 * max(1.0, abs(stored_u))
                or max(abs(_target_terms(replay_record)[k] - float(stored_terms[k]))
                       for k in stored_terms) > 1e-3
                or not np.allclose(gradient, g_saved, rtol=1e-4, atol=1e-4)):
            raise AssertionError("state60 replay value/terms/gradient mismatch")
        state60_row = dict(
            label=label, checkpoint=str(checkpoint),
            replay_value_abs_error=abs(float(value) - stored_u),
            replay_gradient_max_abs_error=float(np.max(np.abs(gradient - g_saved))),
            replay_gradient_relative_l2_error=float(
                np.linalg.norm(gradient - g_saved)
                / max(np.linalg.norm(g_saved), np.finfo(float).tiny)),
            target_evaluations=1, proposals=[])
        report["evaluations"] = [_clean_evaluation(r) for r in objective.records]
        report["exact_target_evaluations"] = len(objective.records)
        report["state_results"].append(state60_row)
        save()
        u_current = float(value)
        g_current = np.asarray(gradient, dtype=np.float64).copy()
        terms_current = _target_terms(replay_record)
        accepted_by_block = {}
        for block_name, projector, epsilon in blocks:
            seed = (second["seed_fundamental"] if block_name == "fundamental_k2_1"
                    else second["seed_hmc_drift_band"])
            q_current, u_current, g_current, terms_current, accepted = run_block(
                label, state60_row, block_name, projector, epsilon, seed,
                q_current, u_current, g_current, terms_current)
            accepted_by_block[block_name] = accepted
        state60_row["completed"] = True
        state60_row["accepted_by_block"] = accepted_by_block
        state60_row["final_objective"] = u_current
        state60_row["final_terms"] = terms_current
        state60_row["target_evaluations"] = len(objective.records)

        each_block_moved = all(
            sum(item.get("accepted_by_block", {}).get(name, 0)
                for item in report["state_results"]) > 0
            for name, _, _ in blocks)
        report.update(
            status=("BLOCK_MALA_SCREEN_COMPLETE_BLOCKS_MOVED_NOT_POSTERIOR"
                    if each_block_moved else
                    "BLOCK_MALA_FIXED_SCALE_ZERO_ACCEPTANCE_IN_BLOCK_NOT_POSTERIOR"),
            accepted_in_each_targeted_block_across_states=bool(each_block_moved),
            all_states_completed=True,
            exact_target_evaluations=len(objective.records),
            elapsed_seconds=time.monotonic() - started,
            host_peak_GiB=_rss_gib(),
            no_state_or_field_promoted=True,
            interpretation=(
                "two dependent discarded-state fixed-scale MALA diagnostic; no stationarity, "
                "mixing, ESS, posterior uncertainty, heldout prediction or R3 authorization"),
        )
        save()
    except EvaluationBudgetStop as error:
        report.update(
            status="BLOCK_MALA_SCREEN_BUDGET_STOP_INCOMPLETE_NOT_POSTERIOR",
            stop_reason=str(error), all_states_completed=False,
            exact_target_evaluations=len(objective.records),
            elapsed_seconds=time.monotonic() - started,
            host_peak_GiB=_rss_gib(), no_state_or_field_promoted=True,
        )
        save()
    except Exception as error:
        report.update(
            status="BLOCK_MALA_SCREEN_HARD_STOP_NOT_POSTERIOR",
            stop_reason=repr(error), all_states_completed=False,
            exact_target_evaluations=len(objective.records),
            elapsed_seconds=time.monotonic() - started,
            host_peak_GiB=_rss_gib(), no_state_or_field_promoted=True,
        )
        save()
        raise


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    mode = os.environ["CF4_GPU_MODE"].upper()
    if mode not in ("H200", "H100", "A100") or mode not in str(jax.devices()[0].device_kind).upper():
        raise RuntimeError("allocated typed GPU does not match CF4_GPU_MODE")
    stats = jax.devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    if limit is None or 1.2 * KNOWN_DEVICE_PEAK_BYTES > int(limit):
        raise MemoryError("allocated GPU is below the frozen 20 percent headroom gate")
    run_conditional_target(
        start_checkpoint=CHECKPOINTS[0][1],
        max_evaluations=MAX_EVALUATIONS,
        app_seconds=APPLICATION_SECONDS,
        support_chunk_cells=16384,
        sampling_driver=run_screen,
        include_present_field_readout=False,
        save_best_parameters=False,
    )


if __name__ == "__main__":
    main()
