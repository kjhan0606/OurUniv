"""Replay the conditional target and attribute its gradient. Not an optimization."""
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
from cf4_r2_conditional_map import (
    BOX, KNOWN_DEVICE_PEAK_GIB, MAX_SUPPORT_CELLS, N, N_IC, TERM_KEYS,
    assemble_conditional_objective, conditional_target_gradient, conditional_target_terms,
)
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import (
    ResolutionObservationTarget, linked_point_conditioning_radius, source_geometry_at_resolution,
)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
ROOT = Path(__file__).resolve().parents[1]
SAVED = BASE / 'r2_conditional_map_20261005_v3'
ABSOLUTE_TERM_LIMIT = 1e-3
RELATIVE_OBJECTIVE_LIMIT = 1e-8
FD_EPS = 1e-4
FD_RELATIVE_LIMIT = 1e-2
COMPONENT_RELATIVE_L2_LIMIT = 1e-4
COMPONENT_MAX_ABS_FACTOR = 1e-3
APP_SECONDS = 70 * 60
ADJOINT_STAGE_SECONDS = 18 * 60
VALUE_STAGE_SECONDS = 12 * 60
COMPONENTS = ((0, 'count_2mpp'), (1, 'conditional_FP'))


def reproduction_failures(saved, replayed, *, absolute_term_limit=ABSOLUTE_TERM_LIMIT,
                          relative_objective_limit=RELATIVE_OBJECTIVE_LIMIT):
    """Empty means the saved evaluation and the replay agree at the declared gate."""
    failures = []
    for key in TERM_KEYS:
        error = abs(float(replayed[key]) - float(saved[key]))
        if not np.isfinite(error) or error > absolute_term_limit:
            failures.append(dict(kind='term', term=key, absolute_error=error,
                                 limit=absolute_term_limit))
    saved_objective = float(saved['objective'])
    replay_objective = float(replayed['total_negative_log_target'])
    relative = abs(replay_objective - saved_objective) / max(abs(saved_objective), 1.0)
    if not np.isfinite(relative) or relative > relative_objective_limit:
        failures.append(dict(
            kind='objective', absolute_error=abs(replay_objective - saved_objective),
            relative_error=relative, limit=relative_objective_limit))
    return failures


def block_gradient_summary(gradient, n_ic=N_IC):
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if gradient.shape != (n_ic + 24,):
        raise ValueError('gradient is not the canonical IC plus 24 nuisance coordinates')
    spans = (('IC', 0, n_ic), ('tracer', n_ic, n_ic + 9), ('population', n_ic + 9, n_ic + 24))
    flat = int(np.argmax(np.abs(gradient)))
    summary = {}
    for name, start, stop in spans:
        block = gradient[start:stop]
        summary[name] = dict(
            size=int(block.size),
            rms=float(np.sqrt(np.mean(block * block))),
            inf=float(np.max(np.abs(block))),
        )
    if flat < n_ic:
        owner, local = 'IC', flat
    elif flat < n_ic + 9:
        owner, local = 'tracer', flat - n_ic
    else:
        owner, local = 'population', flat - (n_ic + 9)
    summary['infinity_norm_location'] = dict(
        flat_index=flat, block=owner, index_in_block=int(local), value=float(gradient[flat]))
    return summary


def finite_difference_agreement(value_at_x, value_at_step, analytic_directional_derivative,
                                eps, *, relative_limit=FD_RELATIVE_LIMIT):
    if not np.isfinite(eps) or eps == 0.0:
        raise ValueError('finite-difference step must be a nonzero finite number')
    observed = (float(value_at_step) - float(value_at_x)) / float(eps)
    analytic = float(analytic_directional_derivative)
    relative = abs(observed - analytic) / max(abs(analytic), 1.0)
    return dict(observed_directional_derivative=observed, analytic_directional_derivative=analytic,
                absolute_error=abs(observed - analytic), relative_error=relative,
                limit=relative_limit, passed=bool(np.isfinite(relative) and relative <= relative_limit))


def gradient_agreement(reconstructed, reference, *, relative_l2_limit=COMPONENT_RELATIVE_L2_LIMIT,
                       max_abs_factor=COMPONENT_MAX_ABS_FACTOR):
    difference = np.asarray(reconstructed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    relative_l2 = float(np.linalg.norm(difference) / max(float(np.linalg.norm(reference)), 1.0))
    max_abs = float(np.max(np.abs(difference)))
    max_abs_limit = max_abs_factor * max(float(np.max(np.abs(reference))), 1.0)
    passed = bool(np.isfinite(relative_l2) and np.isfinite(max_abs)
                  and relative_l2 <= relative_l2_limit and max_abs <= max_abs_limit)
    return dict(relative_l2=relative_l2, relative_l2_limit=relative_l2_limit,
                max_abs=max_abs, max_abs_limit=max_abs_limit, passed=passed)


def full_gradient_record_status(objective_ok, tracer_ok, population_ok, tracer2_ok=True):
    if objective_ok and tracer_ok and population_ok and tracer2_ok:
        return 'CONDITIONAL_FULL_GRADIENT_RECORDED'
    return 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'


def diagnostic_status(*, reproduction_passed, component_split, finite_difference,
                      budget_stopped_early):
    if not reproduction_passed:
        return 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
    if component_split == 'mismatch':
        return 'CONDITIONAL_OPTIMIZER_COMPONENT_SPLIT_MISMATCH'
    if finite_difference == 'failed':
        return 'CONDITIONAL_OPTIMIZER_FD_FAILED'
    if finite_difference != 'passed':
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET'
    if budget_stopped_early:
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET'
    if component_split == 'skipped':
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_COMPLETE_BLOCK_ONLY'
    if component_split == 'passed':
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_COMPLETE'
    raise ValueError('unknown component-split state')


def merge_component_score_gradient(field_pullback, gradients):
    if len(gradients) != 4:
        raise ValueError('component target must return four input-gradient blocks')
    rho_gradient, velocity_gradient, tracer_gradient, population_gradient = gradients
    ic_gradient, = field_pullback((rho_gradient, velocity_gradient))
    return np.concatenate([
        np.asarray(ic_gradient, dtype=np.float64).reshape(-1),
        np.asarray(tracer_gradient, dtype=np.float64).reshape(-1),
        np.asarray(population_gradient, dtype=np.float64).reshape(-1),
    ])


def _save(path, report, started):
    report['elapsed_seconds'] = time.monotonic() - started
    report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _snapshot_count(name):
    value = os.environ.get(name)
    return None if value is None else int(value)


def _stage_allowed(started, need_seconds, budget_seconds=APP_SECONDS):
    return budget_seconds - (time.monotonic() - started) >= need_seconds


def nuisance_scale(gradient, floor=1.):
    """Positive coordinate scales. The objective stays in the original parameters."""
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if gradient.shape != (24,) or not np.isfinite(gradient).all():
        raise ValueError('nuisance scale requires the finite 24-parameter gradient')
    if not np.isfinite(floor) or floor <= 0.:
        raise ValueError('nuisance scale floor must be positive')
    return 1. / np.maximum(np.abs(gradient), float(floor))


def scaled_nuisance(origin, scale, z):
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if origin.shape != (24,) or scale.shape != (24,) or z.shape != (24,):
        raise ValueError('scaled nuisance step requires three 24-vectors')
    if np.any(scale <= 0.) or not np.isfinite(scale).all():
        raise ValueError('nuisance scales must be finite and positive')
    return origin + scale * z


def pop9_newton_step(p_prev, gradient_prev, p_last, gradient_last, max_abs=1.):
    """One damped Newton step from the last two population-9 evaluations."""
    dp = float(p_last) - float(p_prev)
    dg = float(gradient_last) - float(gradient_prev)
    if dp == 0. or dg / dp <= 0.:
        raise ValueError('population-9 slope does not fall along the searched direction')
    if not np.isfinite(max_abs) or max_abs <= 0.:
        raise ValueError('Newton step cap must be positive')
    delta = -float(gradient_last) * dp / dg
    if abs(delta) > float(max_abs):
        delta = float(np.copysign(max_abs, delta))
    return float(p_last) + delta


def tracer_pair_newton_step(t0, g0, t0_prev, g0_prev, g2, g2_prev,
                           t2, g0_at_t2, g2_at_t2, t2_prev, g0_at_t2_prev, g2_at_t2_prev,
                           max_abs=0.2, cross_limit=0.1):
    """One capped Newton step in tracer 0 and tracer 2."""
    dt0 = float(t0) - float(t0_prev)
    dt2 = float(t2) - float(t2_prev)
    if dt0 == 0. or dt2 == 0.:
        raise ValueError('tracer-pair slopes need two distinct points')
    if not np.isfinite(max_abs) or max_abs <= 0.:
        raise ValueError('tracer-pair step cap must be positive')
    h00 = (float(g0) - float(g0_prev)) / dt0
    h20 = (float(g2) - float(g2_prev)) / dt0
    h02 = (float(g0_at_t2) - float(g0_at_t2_prev)) / dt2
    h22 = (float(g2_at_t2) - float(g2_at_t2_prev)) / dt2
    det = h00 * h22 - h02 * h20
    if h00 <= 0. or det <= 0.:
        raise ValueError('tracer-pair curvature is not positive definite')
    scale = max(abs(h02), abs(h20), 1.)
    if abs(h02 - h20) / scale > float(cross_limit):
        raise ValueError('tracer-pair cross derivatives do not agree')
    d0 = (-float(g0) * h22 + h02 * float(g2)) / det
    d2 = (-h00 * float(g2) + h20 * float(g0)) / det
    peak = max(abs(d0), abs(d2))
    if peak > float(max_abs):
        d0 *= float(max_abs) / peak
        d2 *= float(max_abs) / peak
    if float(g0) * d0 + float(g2) * d2 >= 0.:
        raise ValueError('tracer-pair step is not downhill')
    return float(t0) + d0, float(t2) + d2


def tracer_pair_status(initial_objective, best_objective, g0_initial, g0_best,
                       g2_initial, g2_best, likelihood_initial, likelihood_best):
    """Tenfold drop in both coupled tracers, without a worse combined likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (abs(float(g0_best)) <= abs(float(g0_initial)) / 10.
               and abs(float(g2_best)) <= abs(float(g2_initial)) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_TRACER_PAIR_REDUCED'
    if improved:
        return 'CONDITIONAL_TRACER_PAIR_IMPROVED'
    return 'CONDITIONAL_TRACER_PAIR_NO_IMPROVEMENT'


def tracer2_line_status(initial_objective, best_objective, initial_abs_gradient,
                        best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in true-K bias 1, without a worse combined likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_TRACER2_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_TRACER2_LINE_IMPROVED'
    return 'CONDITIONAL_TRACER2_LINE_NO_IMPROVEMENT'


def pop9_line_status(initial_objective, best_objective, initial_abs_gradient,
                     best_abs_gradient, fp_initial, fp_best):
    """Tenfold drop in the FP-covariance coordinate, without a worse FP term."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(fp_best) >= float(fp_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP9_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_POP9_LINE_IMPROVED'
    return 'CONDITIONAL_POP9_LINE_NO_IMPROVEMENT'


def pop3_line_status(initial_objective, best_objective, initial_abs_gradient,
                     best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in the FP mean slope, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP3_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_POP3_LINE_IMPROVED'
    return 'CONDITIONAL_POP3_LINE_NO_IMPROVEMENT'


def pop3_revisit_status(initial_objective, best_objective, initial_abs_gradient,
                        best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in the FP slope b[1, 0], without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP3_REVISIT_REDUCED'
    if improved:
        return 'CONDITIONAL_POP3_REVISIT_IMPROVED'
    return 'CONDITIONAL_POP3_REVISIT_NO_IMPROVEMENT'


def pop3_revisit2_status(initial_objective, best_objective, initial_abs_gradient,
                         best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in the FP slope b[1, 0], without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP3_REVISIT2_REDUCED'
    if improved:
        return 'CONDITIONAL_POP3_REVISIT2_IMPROVED'
    return 'CONDITIONAL_POP3_REVISIT2_NO_IMPROVEMENT'


def pop0_line_status(initial_objective, best_objective, initial_abs_gradient,
                     best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in the FP mean intercept, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP0_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_POP0_LINE_IMPROVED'
    return 'CONDITIONAL_POP0_LINE_NO_IMPROVEMENT'


def pop_pair_status(initial_objective, best_objective, initial_abs_pop0, best_abs_pop0,
                    best_abs_pop3, pop3_gate, likelihood_initial, likelihood_best):
    """Tenfold drop in population 0 while population 3 stays inside its own gate."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_pop0) <= float(initial_abs_pop0) / 10.
               and float(best_abs_pop3) <= float(pop3_gate)
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP_PAIR_REDUCED'
    if improved:
        return 'CONDITIONAL_POP_PAIR_IMPROVED'
    return 'CONDITIONAL_POP_PAIR_NO_IMPROVEMENT'


def pop2_line_status(initial_objective, best_objective, initial_abs_gradient,
                     best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in the third FP intercept, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP2_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_POP2_LINE_IMPROVED'
    return 'CONDITIONAL_POP2_LINE_NO_IMPROVEMENT'


def pop12_line_status(initial_objective, best_objective, initial_abs_gradient,
                      best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky L20, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP12_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_POP12_LINE_IMPROVED'
    return 'CONDITIONAL_POP12_LINE_NO_IMPROVEMENT'


def pop14_line_status(initial_objective, best_objective, initial_abs_gradient,
                      best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky log L22, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP14_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_POP14_LINE_IMPROVED'
    return 'CONDITIONAL_POP14_LINE_NO_IMPROVEMENT'


def pop12_revisit_status(initial_objective, best_objective, initial_abs_gradient,
                         best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky L20, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP12_REVISIT_REDUCED'
    if improved:
        return 'CONDITIONAL_POP12_REVISIT_IMPROVED'
    return 'CONDITIONAL_POP12_REVISIT_NO_IMPROVEMENT'


def pop14_revisit_status(initial_objective, best_objective, initial_abs_gradient,
                         best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky log L22, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP14_REVISIT_REDUCED'
    if improved:
        return 'CONDITIONAL_POP14_REVISIT_IMPROVED'
    return 'CONDITIONAL_POP14_REVISIT_NO_IMPROVEMENT'


def pop9_revisit_status(initial_objective, best_objective, initial_abs_gradient,
                        best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky log L00, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP9_REVISIT_REDUCED'
    if improved:
        return 'CONDITIONAL_POP9_REVISIT_IMPROVED'
    return 'CONDITIONAL_POP9_REVISIT_NO_IMPROVEMENT'


def pop14_revisit2_status(initial_objective, best_objective, initial_abs_gradient,
                          best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky log L22, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP14_REVISIT2_REDUCED'
    if improved:
        return 'CONDITIONAL_POP14_REVISIT2_IMPROVED'
    return 'CONDITIONAL_POP14_REVISIT2_NO_IMPROVEMENT'


def pop14_revisit3_status(initial_objective, best_objective, initial_abs_gradient,
                          best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky log L22, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP14_REVISIT3_REDUCED'
    if improved:
        return 'CONDITIONAL_POP14_REVISIT3_IMPROVED'
    return 'CONDITIONAL_POP14_REVISIT3_NO_IMPROVEMENT'


def pop14_revisit4_status(initial_objective, best_objective, initial_abs_gradient,
                          best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in log-Cholesky log L22, without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP14_REVISIT4_REDUCED'
    if improved:
        return 'CONDITIONAL_POP14_REVISIT4_IMPROVED'
    return 'CONDITIONAL_POP14_REVISIT4_NO_IMPROVEMENT'


def pop0_revisit_status(initial_objective, best_objective, initial_abs_gradient,
                        best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in the FP intercept b[0, 0], without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP0_REVISIT_REDUCED'
    if improved:
        return 'CONDITIONAL_POP0_REVISIT_IMPROVED'
    return 'CONDITIONAL_POP0_REVISIT_NO_IMPROVEMENT'


def pop0_revisit2_status(initial_objective, best_objective, initial_abs_gradient,
                         best_abs_gradient, likelihood_initial, likelihood_best):
    """Tenfold drop in the FP intercept b[0, 0], without a worse count-plus-FP likelihood."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(likelihood_best) >= float(likelihood_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP0_REVISIT2_REDUCED'
    if improved:
        return 'CONDITIONAL_POP0_REVISIT2_IMPROVED'
    return 'CONDITIONAL_POP0_REVISIT2_NO_IMPROVEMENT'


def coordinate_line_action(improved, sign_flipped, reduced, midpoint_used):
    """Stop on a sign change. A failed step gets one midpoint retry, not another outward step."""
    if reduced or sign_flipped:
        return 'stop'
    if improved:
        return 'continue'
    if midpoint_used:
        return 'stop'
    return 'midpoint'


def tracer0_secant(theta_pos, gradient_pos, theta_neg, gradient_neg):
    """One secant step inside a sign bracket. It does not change the objective."""
    theta_pos, theta_neg = float(theta_pos), float(theta_neg)
    gradient_pos, gradient_neg = float(gradient_pos), float(gradient_neg)
    if not (gradient_pos > 0. and gradient_neg < 0.):
        raise ValueError('secant requires opposite derivative signs')
    if theta_pos == theta_neg:
        raise ValueError('secant endpoints must differ')
    theta = theta_pos - gradient_pos * (theta_neg - theta_pos) / (gradient_neg - gradient_pos)
    low, high = sorted((theta_pos, theta_neg))
    if not low < theta < high:
        raise ValueError('secant left the sign bracket')
    return theta


def tracer0_line_status(initial_objective, best_objective, initial_abs_gradient,
                        best_abs_gradient, count_initial, count_best):
    """Classify a fixed-IC line search along the count-amplitude coordinate."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(count_best) >= float(count_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED'
    if improved:
        return 'CONDITIONAL_TRACER0_LINE_IMPROVED'
    return 'CONDITIONAL_TRACER0_LINE_NO_IMPROVEMENT'


def nuisance_block_status(initial_abs_tracer0, final_abs_tracer0, count_initial, count_final):
    """A 10x smaller amplitude gradient with a count term that does not worsen."""
    reduced = float(final_abs_tracer0) <= float(initial_abs_tracer0) / 10.
    count_not_worse = float(count_final) >= float(count_initial) - 1e-6
    if reduced and count_not_worse:
        return 'CONDITIONAL_NUISANCE_BLOCK_AMPLITUDE_REDUCED'
    return 'CONDITIONAL_NUISANCE_BLOCK_NOT_REDUCED'


def finite_difference_status(*, reproduction_passed, finite_difference):
    """Status for the best-state finite-difference completion. It does not authorize a fit."""
    if not reproduction_passed:
        return 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
    if finite_difference == 'passed':
        return 'CONDITIONAL_OPTIMIZER_FD_PASSED'
    if finite_difference == 'failed':
        return 'CONDITIONAL_OPTIMIZER_FD_FAILED'
    return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET'


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU allocation required')
    mode = os.environ['CF4_GPU_MODE']
    expected_kind = {'h200': 'H200', 'h100': 'H100', 'a100': 'A100'}[mode]
    if expected_kind not in str(jax.devices()[0].device_kind).upper():
        raise RuntimeError('allocated device does not match the selected GPU mode')
    expected = os.environ['CF4_EXPECTED_COMMIT']
    actual = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != expected:
        raise RuntimeError('source commit does not match submitted allocation')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'OWNER.txt').write_text(
        'project: CF4\nrepo: /home/kjhan/BACKUP/CF4\n'
        'purpose: conditional-target gradient check; IC is not refit in nuisance-block mode\n'
        'DO_NOT_CANCEL\n')
    fd_only = os.environ.get('CF4_R2_FD_ONLY') == '1'
    nuisance_block = os.environ.get('CF4_R2_NUISANCE_BLOCK') == '1'
    tracer0_line = os.environ.get('CF4_R2_TRACER0_LINE') == '1'
    tracer0_secant_mode = os.environ.get('CF4_R2_TRACER0_SECANT') == '1'
    pop9_line = os.environ.get('CF4_R2_POP9_LINE') == '1'
    pop9_newton = os.environ.get('CF4_R2_POP9_NEWTON') == '1'
    pop9_secant_mode = os.environ.get('CF4_R2_POP9_SECANT') == '1'
    pop9_secant2 = os.environ.get('CF4_R2_POP9_SECANT2') == '1'
    full_gradient = os.environ.get('CF4_R2_FULL_GRADIENT') == '1'
    full_gradient2 = os.environ.get('CF4_R2_FULL_GRADIENT2') == '1'
    full_gradient3 = os.environ.get('CF4_R2_FULL_GRADIENT3') == '1'
    full_gradient4 = os.environ.get('CF4_R2_FULL_GRADIENT4') == '1'
    tracer2_line = os.environ.get('CF4_R2_TRACER2_LINE') == '1'
    tracer0_revisit = os.environ.get('CF4_R2_TRACER0_REVISIT') == '1'
    tracer0_revisit_secant = os.environ.get('CF4_R2_TRACER0_REVISIT_SECANT') == '1'
    tracer_pair = os.environ.get('CF4_R2_TRACER_PAIR') == '1'
    pop3_line = os.environ.get('CF4_R2_POP3_LINE') == '1'
    pop3_secant = os.environ.get('CF4_R2_POP3_SECANT') == '1'
    full_gradient5 = os.environ.get('CF4_R2_FULL_GRADIENT5') == '1'
    pop0_line = os.environ.get('CF4_R2_POP0_LINE') == '1'
    pop_pair = os.environ.get('CF4_R2_POP_PAIR') == '1'
    full_gradient6 = os.environ.get('CF4_R2_FULL_GRADIENT6') == '1'
    pop2_line = os.environ.get('CF4_R2_POP2_LINE') == '1'
    pop2_secant = os.environ.get('CF4_R2_POP2_SECANT') == '1'
    full_gradient7 = os.environ.get('CF4_R2_FULL_GRADIENT7') == '1'
    pop12_line = os.environ.get('CF4_R2_POP12_LINE') == '1'
    full_gradient8 = os.environ.get('CF4_R2_FULL_GRADIENT8') == '1'
    pop14_line = os.environ.get('CF4_R2_POP14_LINE') == '1'
    full_gradient9 = os.environ.get('CF4_R2_FULL_GRADIENT9') == '1'
    pop12_revisit = os.environ.get('CF4_R2_POP12_REVISIT') == '1'
    full_gradient10 = os.environ.get('CF4_R2_FULL_GRADIENT10') == '1'
    pop14_revisit = os.environ.get('CF4_R2_POP14_REVISIT') == '1'
    full_gradient11 = os.environ.get('CF4_R2_FULL_GRADIENT11') == '1'
    pop9_revisit = os.environ.get('CF4_R2_POP9_REVISIT') == '1'
    full_gradient12 = os.environ.get('CF4_R2_FULL_GRADIENT12') == '1'
    pop14_revisit2 = os.environ.get('CF4_R2_POP14_REVISIT2') == '1'
    full_gradient13 = os.environ.get('CF4_R2_FULL_GRADIENT13') == '1'
    pop0_revisit = os.environ.get('CF4_R2_POP0_REVISIT') == '1'
    pop0_revisit_secant = os.environ.get('CF4_R2_POP0_REVISIT_SECANT') == '1'
    full_gradient14 = os.environ.get('CF4_R2_FULL_GRADIENT14') == '1'
    pop3_revisit = os.environ.get('CF4_R2_POP3_REVISIT') == '1'
    pop3_revisit_secant = os.environ.get('CF4_R2_POP3_REVISIT_SECANT') == '1'
    full_gradient15 = os.environ.get('CF4_R2_FULL_GRADIENT15') == '1'
    pop0_revisit2 = os.environ.get('CF4_R2_POP0_REVISIT2') == '1'
    pop0_revisit2_secant = os.environ.get('CF4_R2_POP0_REVISIT2_SECANT') == '1'
    full_gradient16 = os.environ.get('CF4_R2_FULL_GRADIENT16') == '1'
    pop14_revisit3 = os.environ.get('CF4_R2_POP14_REVISIT3') == '1'
    full_gradient17 = os.environ.get('CF4_R2_FULL_GRADIENT17') == '1'
    pop14_revisit4 = os.environ.get('CF4_R2_POP14_REVISIT4') == '1'
    full_gradient18 = os.environ.get('CF4_R2_FULL_GRADIENT18') == '1'
    pop3_revisit2 = os.environ.get('CF4_R2_POP3_REVISIT2') == '1'
    pop3_revisit2_secant = os.environ.get('CF4_R2_POP3_REVISIT2_SECANT') == '1'
    full_gradient19 = os.environ.get('CF4_R2_FULL_GRADIENT19') == '1'
    if sum((fd_only, nuisance_block, tracer0_line, tracer0_secant_mode, pop9_line,
            pop9_newton, pop9_secant_mode, pop9_secant2, full_gradient, full_gradient2,
            tracer2_line, tracer0_revisit, tracer0_revisit_secant, full_gradient3,
            tracer_pair, full_gradient4, pop3_line, pop3_secant, full_gradient5,
            pop0_line, pop_pair, full_gradient6, pop2_line, pop2_secant, full_gradient7,
            pop12_line, full_gradient8, pop14_line, full_gradient9, pop12_revisit,
            full_gradient10, pop14_revisit, full_gradient11, pop9_revisit,
            full_gradient12, pop14_revisit2, full_gradient13, pop0_revisit,
            pop0_revisit_secant, full_gradient14, pop3_revisit, pop3_revisit_secant,
            full_gradient15, pop0_revisit2, pop0_revisit2_secant, full_gradient16,
            pop14_revisit3, full_gradient17, pop14_revisit4, full_gradient18,
            pop3_revisit2, pop3_revisit2_secant, full_gradient19)) > 1:
        raise RuntimeError('conditional diagnostic modes are separate jobs')
    if (tracer0_line or tracer0_secant_mode or pop9_line or pop9_newton
            or pop9_secant_mode or pop9_secant2 or full_gradient or full_gradient2
            or tracer2_line or tracer0_revisit or tracer0_revisit_secant
            or full_gradient3 or tracer_pair or full_gradient4 or pop3_line
            or pop3_secant or full_gradient5 or pop0_line or pop_pair
            or full_gradient6 or pop2_line or pop2_secant or full_gradient7
            or pop12_line or full_gradient8 or pop14_line or full_gradient9
            or pop12_revisit or full_gradient10 or pop14_revisit or full_gradient11
            or pop9_revisit or full_gradient12 or pop14_revisit2 or full_gradient13
            or pop0_revisit or pop0_revisit_secant or full_gradient14
            or pop3_revisit or pop3_revisit_secant or full_gradient15
            or pop0_revisit2 or pop0_revisit2_secant or full_gradient16
            or pop14_revisit3 or full_gradient17 or pop14_revisit4 or full_gradient18
            or pop3_revisit2 or pop3_revisit2_secant or full_gradient19):
        budget_seconds = 70 * 60
    elif fd_only or nuisance_block:
        budget_seconds = 40 * 60
    else:
        budget_seconds = APP_SECONDS
    started = time.monotonic()
    report_path = out / 'result.json'
    report = dict(
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'], source_commit=expected,
        N=N, box_cMpc_h=BOX, optimizer_called=False, longer_warm_start_authorized=False,
        posterior_sample=False, heldout_scored=False, heldout_count_values_loaded=0,
        heldout_FP_marks_loaded=0, likelihood_changed=False,
        estimand=('same conditional v6 training target as job 413612; no new prior or likelihood term'),
        gates=dict(absolute_term_limit=ABSOLUTE_TERM_LIMIT,
                   relative_objective_limit=RELATIVE_OBJECTIVE_LIMIT,
                   finite_difference_eps=FD_EPS, finite_difference_relative_limit=FD_RELATIVE_LIMIT,
                   component_relative_l2_limit=COMPONENT_RELATIVE_L2_LIMIT,
                   component_max_abs_factor=COMPONENT_MAX_ABS_FACTOR),
        application_seconds=budget_seconds,
        finite_difference_only=fd_only,
        nuisance_block_only=nuisance_block,
        tracer0_line_only=tracer0_line,
        tracer0_secant_only=tracer0_secant_mode,
        pop9_line_only=pop9_line,
        pop9_newton_only=pop9_newton,
        pop9_secant_only=pop9_secant_mode,
        pop9_secant2_only=pop9_secant2,
        full_gradient_only=full_gradient,
        full_gradient2_only=full_gradient2,
        full_gradient3_only=full_gradient3,
        full_gradient4_only=full_gradient4,
        tracer2_line_only=tracer2_line,
        tracer0_revisit_only=tracer0_revisit,
        tracer0_revisit_secant_only=tracer0_revisit_secant,
        tracer_pair_only=tracer_pair,
        pop3_line_only=pop3_line,
        pop3_secant_only=pop3_secant,
        full_gradient5_only=full_gradient5,
        pop0_line_only=pop0_line,
        pop_pair_only=pop_pair,
        full_gradient6_only=full_gradient6,
        pop2_line_only=pop2_line,
        pop2_secant_only=pop2_secant,
        full_gradient7_only=full_gradient7,
        pop12_line_only=pop12_line,
        full_gradient8_only=full_gradient8,
        pop14_line_only=pop14_line,
        full_gradient9_only=full_gradient9,
        pop12_revisit_only=pop12_revisit,
        full_gradient10_only=full_gradient10,
        pop14_revisit_only=pop14_revisit,
        full_gradient11_only=full_gradient11,
        pop9_revisit_only=pop9_revisit,
        full_gradient12_only=full_gradient12,
        pop14_revisit2_only=pop14_revisit2,
        full_gradient13_only=full_gradient13,
        pop0_revisit_only=pop0_revisit,
        pop0_revisit_secant_only=pop0_revisit_secant,
        full_gradient14_only=full_gradient14,
        pop3_revisit_only=pop3_revisit,
        pop3_revisit_secant_only=pop3_revisit_secant,
        full_gradient15_only=full_gradient15,
        pop0_revisit2_only=pop0_revisit2,
        pop0_revisit2_secant_only=pop0_revisit2_secant,
        full_gradient16_only=full_gradient16,
        pop14_revisit3_only=pop14_revisit3,
        full_gradient17_only=full_gradient17,
        pop14_revisit4_only=pop14_revisit4,
        full_gradient18_only=full_gradient18,
        pop3_revisit2_only=pop3_revisit2,
        pop3_revisit2_secant_only=pop3_revisit2_secant,
        full_gradient19_only=full_gradient19,
        ic_coordinates_fixed=(nuisance_block or tracer0_line or tracer0_secant_mode
                              or pop9_line or pop9_newton or pop9_secant_mode
                              or pop9_secant2 or full_gradient or full_gradient2
                              or tracer2_line or tracer0_revisit or tracer0_revisit_secant
                              or full_gradient3 or tracer_pair or full_gradient4
                              or pop3_line or pop3_secant or full_gradient5 or pop0_line
                              or pop_pair or full_gradient6 or pop2_line or pop2_secant
                              or full_gradient7 or pop12_line or full_gradient8
                              or pop14_line or full_gradient9 or pop12_revisit
                              or full_gradient10 or pop14_revisit
                              or full_gradient11 or pop9_revisit
                              or full_gradient12 or pop14_revisit2
                              or full_gradient13 or pop0_revisit
                              or pop0_revisit_secant or full_gradient14
                              or pop3_revisit or pop3_revisit_secant
                              or full_gradient15 or pop0_revisit2
                              or pop0_revisit2_secant or full_gradient16
                              or pop14_revisit3 or full_gradient17
                              or pop14_revisit4 or full_gradient18
                              or pop3_revisit2 or pop3_revisit2_secant
                              or full_gradient19),
        LG_roles=dict(MW='ambiguous', M31='ambiguous', M33='unresolved'),
        Q_GOAL='reproduce and attribute the existing conditional target before any longer fit',
        Q_LEAN='two saved states, best-state component split, one directional finite difference; no sampler or heldout',
    )
    _save(report_path, report, started)
    try:
        stats = jax.devices()[0].memory_stats() or {}
        device_limit = int(stats.get('bytes_limit', 0) or 0)
        report['resource_evidence'] = dict(
            selected_mode=f'{mode} / gpu:{expected_kind}:1',
            h200_free_typed_gpus_at_submit=_snapshot_count('CF4_H200_FREE_TYPED_GPUS'),
            h100_free_typed_gpus_at_submit=_snapshot_count('CF4_H100_FREE_TYPED_GPUS'),
            a100_free_typed_gpus_at_submit=_snapshot_count('CF4_A100_FREE_TYPED_GPUS'),
            measured_prior_exact_GL2_peak_GiB=KNOWN_DEVICE_PEAK_GIB,
            current_device_limit_GiB=float(device_limit / 1024 ** 3) if device_limit else None,
            host_request_GiB=48,
        )
        if device_limit and 1.2 * KNOWN_DEVICE_PEAK_GIB * 1024 ** 3 > device_limit:
            raise MemoryError('measured exact-GL2 device peak lacks 20 percent headroom here')

        saved = json.loads((SAVED / 'result.json').read_text())
        if saved.get('status') != 'CONDITIONAL_MAP_ATTEMPT_INCOMPLETE':
            raise ValueError('frozen conditional MAP artifact is not the incomplete 413612 attempt')
        if len(saved.get('evaluations', ())) != 6:
            raise ValueError('frozen conditional MAP artifact does not have six evaluations')
        saved_initial = saved['evaluations'][0]
        saved_best = saved['evaluations'][5]
        with np.load(SAVED / 'best_parameters.npz', allow_pickle=False) as archive:
            q_best = np.concatenate([
                np.asarray(archive['white_ic'], dtype=np.float64),
                np.asarray(archive['tracer_white'], dtype=np.float64),
                np.asarray(archive['population_white'], dtype=np.float64),
            ])
            stored_objective = float(archive['objective'])
        if q_best.shape != (N_IC + 24,) or not np.isfinite(q_best).all():
            raise ValueError('saved best parameters are not a finite canonical state')
        if abs(stored_objective - float(saved_best['objective'])) > 1e-6:
            raise ValueError('best parameter objective does not match evaluation 6')

        parent = BASE / 'r2_n256_dynamics_profile_v1'
        parent_result = json.loads((parent / 'result.json').read_text())
        if parent_result.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('frozen N256 initializer is absent or has an unexpected status')
        with np.load(parent / 'initial_present_state.npz', allow_pickle=False) as archive:
            q0 = np.concatenate((
                np.asarray(archive['white_ic'], dtype=np.float64).reshape(-1),
                np.zeros(24, dtype=np.float64),
            ))
        if q0.shape != (N_IC + 24,) or not np.isfinite(q0).all():
            raise ValueError('frozen N256 initializer is not a finite canonical state')

        _, _, _, _, source, mix, observation, geometry = load_inputs()
        linked_radii = linked_point_conditioning_radius(observation)
        if (len(mix['PGC']) != 1414 or len(linked_radii) != 1414
                or mix.get('pre_reconciliation_rows') != 1414
                or len(mix.get('excluded_unresolved_PGCs', ())) != 0):
            raise ValueError('frozen v6 linked conditional cohort contract changed')
        with np.load(BASE / 'r2_sky_closed_split_v6/split.npz', allow_pickle=False) as split:
            train_keys = jnp.asarray(split['train_keys'])
            train_counts = jnp.asarray(split['train_counts'])
            heldout_voxels = split['heldout_flat_voxels']
            train_excluded = split['train_window_excluded_keys']
            heldout_excluded = split['heldout_window_excluded_keys']
        if int(np.asarray(train_counts).sum()) != 47121 or len(train_keys) != 37951:
            raise ValueError('frozen v6 training count graph changed')
        exposure, _ = build_population_exposure_masks(
            128, heldout_voxels, train_excluded, heldout_excluded)
        if not np.asarray(exposure)[np.asarray(train_keys)].all():
            raise ValueError('training count key is outside its frozen exposure')
        source = source_geometry_at_resolution(source, N)
        obs = ResolutionObservationTarget(
            N, source, mix, observation, geometry, train_keys, train_counts, jnp.asarray(exposure),
            force_order=1, fine_order=2, max_support_cells=MAX_SUPPORT_CELLS)
        settings = {key: parent_result['settings'][key]
                    for key in ('cosmology', 'a_start', 'a_stop', 'a_nbody_maxstep')}
        settings.update(n=N, box_cMpc_h=BOX)
        common = json.loads((ROOT / 'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
        if settings['cosmology']['h'] != 0.746 or settings['cosmology']['Om'] != common['Omega_m']:
            raise ValueError('N256 dynamics cosmology differs from frozen count/CF4 basis')
        evolve, _, conf, _, particle_mass = make_dynamics(settings)
        particle_masses = jnp.full(N_IC, particle_mass)

        @jax.jit
        def field(white):
            positions, velocities = evolve(white)
            state = particle_grid(positions, velocities, particle_masses, conf)
            return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

        source_jax = {key: jnp.asarray(value) for key, value in source.items()}
        obs_jax = {key: jnp.asarray(value) for key, value in observation.items()}
        if pop9_secant_mode or pop9_secant2:
            if pop9_secant_mode:
                parent_path = BASE / 'r2_conditional_pop9_newton_20261007/result.json'
                required_steps = ('verify', 'newton')
                parent_error = 'population-9 secant requires the improved Newton step'
                bracket_error = 'population-9 secant requires the Newton sign bracket'
                report_job_key = 'population9_newton_job_id'
                q_lean = (
                    'verify the population-9 Newton point, then one secant inside its sign bracket; '
                    'no sampler, heldout, or IC update')
            else:
                parent_path = BASE / 'r2_conditional_pop9_secant_20261007/result.json'
                required_steps = ('verify', 'secant')
                parent_error = 'population-9 second secant requires the improved first secant'
                bracket_error = 'population-9 second secant requires the tightened sign bracket'
                report_job_key = 'population9_secant_job_id'
                q_lean = (
                    'verify the population-9 secant point, then one secant inside the tightened bracket; '
                    'no sampler, heldout, or IC update')
            parent = json.loads(parent_path.read_text())
            if parent.get('status') != 'CONDITIONAL_POP9_LINE_IMPROVED':
                raise ValueError(parent_error)
            rows = parent.get('evaluations') or []
            if (len(rows) != 2 or rows[0].get('step') != required_steps[0]
                    or rows[1].get('step') != required_steps[1]):
                raise ValueError(bracket_error)
            if pop9_secant_mode:
                positive, negative = rows[0], rows[1]
            else:
                negative, positive = rows[0], rows[1]
            anchor = negative if pop9_secant_mode else positive
            if not (float(positive['population_coordinate_9_gradient']) > 0.
                    and float(negative['population_coordinate_9_gradient']) < 0.):
                raise ValueError('population-9 secant requires opposite derivative signs')
            tracer0 = float(anchor['tracer0'])
            if float(positive['tracer0']) != tracer0 or float(negative['tracer0']) != tracer0:
                raise ValueError('population-9 bracket changed tracer 0')
            line = json.loads(
                (BASE / 'r2_conditional_pop9_line_20261007_v2/result.json').read_text())
            gate_rows = line.get('evaluations') or []
            if (line.get('status') != 'CONDITIONAL_POP9_LINE_IMPROVED' or len(gate_rows) != 3
                    or float(gate_rows[0]['tracer0']) != tracer0):
                raise ValueError('population-9 secant requires the improved line as its tenfold gate')
            gate = gate_rows[0]
            theta = tracer0_secant(
                positive['population_coordinate_9'], positive['population_coordinate_9_gradient'],
                negative['population_coordinate_9'], negative['population_coordinate_9_gradient'])
            report[report_job_key] = parent.get('job_id')
            report['secant_population_coordinate_9'] = theta
            report['Q_LEAN'] = q_lean
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = tracer0
            origin[18] = float(anchor['population_coordinate_9'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-9 secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)
            gradient_limit = 1e-4 * max(
                abs(float(anchor['population_coordinate_9_gradient'])), 1.)
            tracer_limit = 1e-4 * max(abs(float(anchor['tracer0_gradient'])), 1.)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(anchor['objective'])) / max(
                    abs(float(anchor['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and abs(
                baseline['population_coordinate_9_gradient']
                - float(anchor['population_coordinate_9_gradient'])) <= gradient_limit
            tracer_ok = baseline is not None and abs(
                baseline['tracer0_gradient'] - float(anchor['tracer0_gradient'])) <= tracer_limit
            records = []

            def score_against_line_start(candidates):
                best = min(candidates, key=lambda row: row['objective'])
                return pop9_line_status(
                    float(gate['objective']), best['objective'],
                    abs(float(gate['population_coordinate_9_gradient'])),
                    abs(best['population_coordinate_9_gradient']),
                    float(gate['terms']['conditional_FP_log_likelihood']),
                    best['terms']['conditional_FP_log_likelihood'])

            if not (objective_ok and gradient_ok and tracer_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score_against_line_start(records)
                else:
                    proposal = origin.copy()
                    proposal[18] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score_against_line_start(records)
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        report['status'] = score_against_line_start(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if pop9_newton:
            line = json.loads(
                (BASE / 'r2_conditional_pop9_line_20261007_v2/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP9_LINE_IMPROVED':
                raise ValueError('population-9 Newton requires the improved population-9 line')
            rows = line.get('evaluations') or []
            if len(rows) != 3:
                raise ValueError('population-9 Newton requires the three recorded line evaluations')
            initial, previous, last = rows[0], rows[1], rows[2]
            tracer0 = float(last['tracer0'])
            if any(float(row['tracer0']) != tracer0 for row in rows):
                raise ValueError('population-9 line changed tracer 0')
            theta = pop9_newton_step(
                previous['population_coordinate_9'], previous['population_coordinate_9_gradient'],
                last['population_coordinate_9'], last['population_coordinate_9_gradient'])
            report['population9_line_job_id'] = line.get('job_id')
            report['newton_population_coordinate_9'] = theta
            report['newton_step'] = theta - float(last['population_coordinate_9'])
            report['Q_LEAN'] = (
                'verify population-9 evaluation 2, then one capped Newton step and at most one midpoint; '
                'no sampler, heldout, or IC update')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = tracer0
            origin[18] = float(last['population_coordinate_9'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-9 Newton produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)
            gradient_limit = 1e-4 * max(abs(float(last['population_coordinate_9_gradient'])), 1.)
            tracer_limit = 1e-4 * max(abs(float(last['tracer0_gradient'])), 1.)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(last['objective'])) / max(abs(float(last['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and abs(
                baseline['population_coordinate_9_gradient']
                - float(last['population_coordinate_9_gradient'])) <= gradient_limit
            tracer_ok = baseline is not None and abs(
                baseline['tracer0_gradient'] - float(last['tracer0_gradient'])) <= tracer_limit
            records = []

            def score_against_line_start(candidates):
                best = min(candidates, key=lambda row: row['objective'])
                return pop9_line_status(
                    float(initial['objective']), best['objective'],
                    abs(float(initial['population_coordinate_9_gradient'])),
                    abs(best['population_coordinate_9_gradient']),
                    float(initial['terms']['conditional_FP_log_likelihood']),
                    best['terms']['conditional_FP_log_likelihood'])

            if not (objective_ok and gradient_ok and tracer_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score_against_line_start(records)
                else:
                    proposal = origin.copy()
                    proposal[18] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score_against_line_start(records)
                    else:
                        row['step'] = 'newton'
                        records.append(row)
                        if row['objective'] >= baseline['objective']:
                            if budget_seconds - (time.monotonic() - started) < 60.:
                                report['budget_stop'] = True
                            else:
                                midpoint = origin.copy()
                                midpoint[18] = 0.5 * (float(last['population_coordinate_9']) + theta)
                                mid_row = evaluate_at(midpoint)
                                if mid_row is None:
                                    report['support_shape_changed'] = True
                                else:
                                    mid_row['step'] = 'midpoint'
                                    records.append(mid_row)
                        report['status'] = score_against_line_start(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if tracer0_revisit_secant:
            line = json.loads(
                (BASE / 'r2_conditional_tracer0_revisit_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_TRACER0_LINE_NO_IMPROVEMENT':
                raise ValueError('tracer-0 revisit secant requires the recorded bracket')
            rows = line.get('evaluations') or []
            if len(rows) != 2 or not (float(rows[1].get('step', 0.)) > 0.):
                raise ValueError('tracer-0 revisit secant requires the positive trial')
            negative, positive = rows[0], rows[1]
            if not (float(negative['tracer0_gradient']) < 0.
                    and float(positive['tracer0_gradient']) > 0.):
                raise ValueError('tracer-0 revisit secant requires opposite derivative signs')
            if (float(negative['tracer2']) != float(positive['tracer2'])
                    or float(negative['population_coordinate_9'])
                    != float(positive['population_coordinate_9'])):
                raise ValueError('tracer-0 revisit changed tracer 2 or population 9')
            theta = tracer0_secant(
                positive['tracer0'], positive['tracer0_gradient'],
                negative['tracer0'], negative['tracer0_gradient'])
            report['tracer0_revisit_job_id'] = line.get('job_id')
            report['secant_tracer0'] = theta
            report['Q_LEAN'] = (
                'verify the tracer-0 bracket endpoint, then one secant inside it; '
                'no outward step, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(negative['tracer0'])
            origin[2] = float(negative['tracer2'])
            origin[18] = float(negative['population_coordinate_9'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer-0 revisit secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)
            limits = (
                ('objective', None),
                ('tracer0_gradient', 1e-4 * max(abs(float(negative['tracer0_gradient'])), 1.)),
                ('tracer2_gradient', 1e-4 * max(abs(float(negative['tracer2_gradient'])), 1.)),
                ('population_coordinate_9_gradient',
                 1e-4 * max(abs(float(negative['population_coordinate_9_gradient'])), 1.)),
            )
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(negative['objective'])) / max(
                    abs(float(negative['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all(
                abs(float(baseline[name]) - float(negative[name])) <= limit
                for name, limit in limits if limit is not None)
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = tracer0_line_status(
                        baseline['objective'], baseline['objective'],
                        abs(float(negative['tracer0_gradient'])),
                        abs(baseline['tracer0_gradient']),
                        baseline['terms']['count_log_likelihood'],
                        baseline['terms']['count_log_likelihood'])
                else:
                    proposal = origin.copy()
                    proposal[0] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = tracer0_line_status(
                            baseline['objective'], baseline['objective'],
                            abs(float(negative['tracer0_gradient'])),
                            abs(baseline['tracer0_gradient']),
                            baseline['terms']['count_log_likelihood'],
                            baseline['terms']['count_log_likelihood'])
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        best = min(records, key=lambda item: item['objective'])
                        report['status'] = tracer0_line_status(
                            float(negative['objective']), best['objective'],
                            abs(float(negative['tracer0_gradient'])),
                            abs(best['tracer0_gradient']),
                            float(negative['terms']['count_log_likelihood']),
                            best['terms']['count_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if tracer0_revisit:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient2_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('tracer-0 revisit requires the second full gradient')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[18] = float(recorded['population_coordinate_9'])
            report['full_gradient2_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'verify the moved tracer-2 point, then move only tracer 0; '
                'no IC update, sampler, or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer-0 revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matched(row):
                objective_ok = abs(row['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
                limits = (
                    (row['tracer0_gradient'], recorded['tracer0_gradient']),
                    (row['tracer2_gradient'], recorded['tracer2_gradient']),
                    (row['population_coordinate_9_gradient'],
                     recorded['population_coordinate_9_gradient']),
                )
                return objective_ok and all(
                    abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)
                    for got, expected in limits)

            baseline = evaluate_at(origin)
            records = []
            if baseline is None or not matched(baseline):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['tracer0_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[0] = accepted[0] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['tracer0_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['tracer0_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['tracer0_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda row: row['objective'])
                report['status'] = tracer0_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['tracer0_gradient']),
                    baseline['terms']['count_log_likelihood'],
                    best['terms']['count_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if tracer2_line:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('tracer-2 line requires the recorded full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('tracer-2 line requires all 24 recorded nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[18] = float(recorded['population_coordinate_9'])
            report['full_gradient_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'verify the recorded gradient, then move only tracer 2 by 0.1; '
                'no IC update, sampler, or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer-2 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matched(row, reference, reference_tracer2):
                objective_ok = abs(row['objective'] - float(reference['objective'])) / max(
                    abs(float(reference['objective'])), 1.) <= 1e-8
                limits = (
                    (row['tracer0_gradient'], reference['tracer0_gradient']),
                    (row['tracer2_gradient'], reference_tracer2),
                    (row['population_coordinate_9_gradient'],
                     reference['population_coordinate_9_gradient']),
                )
                return objective_ok and all(
                    abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)
                    for got, expected in limits)

            baseline = evaluate_at(origin)
            records = []
            if baseline is None or not matched(baseline, recorded, nuisance_gradient[2]):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['tracer2_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[2] = accepted[2] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['tracer2_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['tracer2_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['tracer2_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda row: row['objective'])
                best_likelihood = (best['terms']['count_log_likelihood']
                                   + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = tracer2_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['tracer2_gradient']), initial_likelihood, best_likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if pop9_line:
            secant_result = json.loads(
                (BASE / 'r2_conditional_tracer0_secant_20261007/result.json').read_text())
            if secant_result.get('status') != 'CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED':
                raise ValueError('population-9 line requires the reduced tracer-0 secant')
            saved = secant_result['evaluations'][1]
            if saved.get('step') != 'secant':
                raise ValueError('population-9 line requires the secant evaluation')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(saved['tracer0'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-9 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)
            gradient_limit = 1e-4 * max(abs(float(saved['population_coordinate_9_gradient'])), 1.)
            tracer_limit = 1e-4 * max(abs(float(saved['tracer0_gradient'])), 1.)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(saved['objective'])) / max(abs(float(saved['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and abs(
                baseline['population_coordinate_9_gradient'] - float(saved['population_coordinate_9_gradient'])) <= gradient_limit
            tracer_ok = baseline is not None and abs(
                baseline['tracer0_gradient'] - float(saved['tracer0_gradient'])) <= tracer_limit
            records = []
            if not (objective_ok and gradient_ok and tracer_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_9_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[18] = accepted[18] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_9_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_9_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_9_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda row: row['objective'])
                report['status'] = pop9_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_9_gradient']),
                    baseline['terms']['conditional_FP_log_likelihood'],
                    best['terms']['conditional_FP_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if tracer0_secant_mode:
            line = json.loads((BASE / 'r2_conditional_tracer0_line_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_TRACER0_LINE_IMPROVED':
                raise ValueError('tracer0 secant requires the recorded improved line search')
            positive, negative = line['evaluations'][0], line['evaluations'][1]
            theta_star = tracer0_secant(
                positive['tracer0'], positive['tracer0_gradient'],
                negative['tracer0'], negative['tracer0_gradient'])
            report['secant_tracer0'] = theta_star
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            tracer = jnp.asarray(origin[:9])
            population = jnp.asarray(origin[9:])
            packs, _ = obs.support(rho, velocity, tracer, 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                tracer_i = jnp.asarray(nuisance[:9])
                population_i = jnp.asarray(nuisance[9:])
                built, support_info = obs.support(rho, velocity, tracer_i, 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, tracer_i, population_i, built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer0 secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matched(row, saved_row):
                objective_error = abs(row['objective'] - float(saved_row['objective']))
                gradient_error = abs(row['tracer0_gradient'] - float(saved_row['tracer0_gradient']))
                gradient_limit = 1e-4 * max(abs(float(saved_row['tracer0_gradient'])), 1.)
                relative_objective = objective_error / max(abs(float(saved_row['objective'])), 1.)
                return relative_objective <= 1e-8 and gradient_error <= gradient_limit

            records = []
            verify = origin.copy()
            verify[0] = float(negative['tracer0'])
            verified = evaluate_at(verify)
            if verified is None or not matched(verified, negative):
                report['verification'] = verified
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                records.append(verified)
                proposal = origin.copy()
                proposal[0] = theta_star
                secant_row = evaluate_at(proposal)
                if secant_row is None:
                    report['support_shape_changed'] = True
                    report['status'] = 'CONDITIONAL_TRACER0_LINE_IMPROVED'
                else:
                    secant_row['step'] = 'secant'
                    records.append(secant_row)
                    best = min(records, key=lambda row: row['objective'])
                    report['status'] = tracer0_line_status(
                        float(positive['objective']), best['objective'],
                        abs(float(positive['tracer0_gradient'])), abs(best['tracer0_gradient']),
                        float(positive['terms']['count_log_likelihood']),
                        best['terms']['count_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if tracer0_line:
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            report['fixed_ic_evolution_seconds'] = time.monotonic() - started
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            def evaluate_at(nuisance, compiled, base_shapes):
                tic = time.monotonic()
                tracer = jnp.asarray(nuisance[:9])
                population = jnp.asarray(nuisance[9:])
                packs, support_info = obs.support(rho, velocity, tracer, 2)
                shapes = shapes_of(packs)
                if shapes != base_shapes:
                    return None, shapes, time.monotonic() - tic
                (score, components), grads = compiled(
                    rho, velocity, tracer, population, packs, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer0 line search produced a nonfinite target')
                return dict(
                    objective=value, execute_seconds=time.monotonic() - tic,
                    terms={key: detail[key] for key in TERM_KEYS},
                    tracer0=float(nuisance[0]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                ), shapes, time.monotonic() - tic

            tracer = jnp.asarray(origin[:9])
            population = jnp.asarray(origin[9:])
            support_tic = time.monotonic()
            packs, _ = obs.support(rho, velocity, tracer, 2)
            report['baseline_support_seconds'] = time.monotonic() - support_tic
            base_shapes = shapes_of(packs)
            compile_tic = time.monotonic()
            compiled = obs.derivative.lower(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2).compile()
            report['derivative_compile_seconds'] = time.monotonic() - compile_tic
            baseline, _, _ = evaluate_at(origin, compiled, base_shapes)
            if baseline is None:
                raise RuntimeError('baseline support changed before its own derivative')
            failures = reproduction_failures(saved_best, baseline['terms'] | {
                'total_negative_log_target': baseline['objective']})
            report['baseline'] = baseline
            report['baseline_reproduction_failures'] = failures
            records = [baseline]
            if failures:
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                step = -0.1
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[0] = accepted[0] + step
                    row, shapes, _ = evaluate_at(proposal, compiled, base_shapes)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    if row['objective'] < accepted_objective:
                        accepted = proposal
                        accepted_objective = row['objective']
                        step = -0.1
                        if abs(row['tracer0_gradient']) <= abs(baseline['tracer0_gradient']) / 10.:
                            break
                    elif step == -0.1:
                        step = -0.05
                    else:
                        break
                best = min(records, key=lambda row: row['objective'])
                report['status'] = tracer0_line_status(
                    baseline['objective'], best['objective'], abs(baseline['tracer0_gradient']),
                    abs(best['tracer0_gradient']), baseline['terms']['count_log_likelihood'],
                    best['terms']['count_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if nuisance_block:
            from scipy.optimize import minimize
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            report['fixed_ic_evolution_seconds'] = time.monotonic() - started
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            records = []

            def nuisance_value_gradient(nuisance):
                nuisance = np.asarray(nuisance, dtype=np.float64)
                q = np.concatenate((q_best[:N_IC], nuisance))
                tracer = jnp.asarray(nuisance[:9])
                population = jnp.asarray(nuisance[9:])
                tic = time.monotonic()
                packs, support_info = obs.support(rho, velocity, tracer, 2)
                (score, components), grads = obs.derivative(
                    rho, velocity, tracer, population, packs, source_jax, obs_jax, 2)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('nuisance block produced a nonfinite target')
                records.append(dict(
                    objective=value,
                    seconds=time.monotonic() - tic,
                    terms={key: detail[key] for key in TERM_KEYS},
                    tracer0=float(nuisance[0]),
                    population9=float(nuisance[9]),
                    tracer0_gradient=float(gradient[0]),
                    population9_gradient=float(gradient[9]),
                    gradient_rms=float(np.sqrt(np.mean(gradient * gradient))),
                    gradient_inf=float(np.max(np.abs(gradient))),
                ))
                return value, gradient, detail

            value0, gradient0, _ = nuisance_value_gradient(origin)
            report['initial_nuisance_evaluation'] = records[-1]
            if records[-1]['seconds'] > 480.:
                report['status'] = 'CONDITIONAL_NUISANCE_BLOCK_TIMING_STOP'
                report['longer_warm_start_authorized'] = False
                _save(report_path, report, started)
                print(json.dumps(dict(status=report['status'],
                                      longer_warm_start_authorized=False), allow_nan=False), flush=True)
                return
            scale = nuisance_scale(gradient0)
            report['nuisance_scale'] = scale.tolist()
            cache = dict(x=origin.copy(), value=value0, gradient=gradient0.copy())

            def scaled_objective(z):
                if budget_seconds - (time.monotonic() - started) < 60.:
                    raise TimeoutError('nuisance-block budget cannot hold another evaluation')
                nuisance = scaled_nuisance(origin, scale, z)
                if np.array_equal(nuisance, cache['x']):
                    return cache['value'], scale * cache['gradient']
                value, gradient, _ = nuisance_value_gradient(nuisance)
                cache.update(x=nuisance.copy(), value=value, gradient=gradient.copy())
                return value, scale * gradient

            try:
                solver = minimize(
                    scaled_objective, np.zeros(24), method='L-BFGS-B', jac=True,
                    options=dict(maxiter=6, maxfun=10, maxls=5, ftol=1e-12, gtol=1e-8))
                report['optimizer'] = dict(
                    success=bool(solver.success), message=str(solver.message),
                    iterations=int(solver.nit), evaluations=int(solver.nfev))
            except TimeoutError as error:
                report['budget_stop'] = repr(error)
            final = records[-1]
            report['evaluations'] = records
            report['ic_max_abs_change'] = 0.
            report['status'] = nuisance_block_status(
                abs(records[0]['tracer0_gradient']), abs(final['tracer0_gradient']),
                records[0]['terms']['count_log_likelihood'],
                final['terms']['count_log_likelihood'])
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        def evaluate_full(q):
            white = jnp.asarray(q[:N_IC])
            tracer = jnp.asarray(q[N_IC:N_IC + 9])
            population = jnp.asarray(q[N_IC + 9:])
            (rho, velocity), pullback = jax.vjp(field, white)
            packs, support_info = obs.support(rho, velocity, tracer, 2)
            (score, components), grads = obs.derivative(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2)
            ic_gradient, = pullback((grads[0], grads[1]))
            value, gradient, detail = assemble_conditional_objective(
                q, score, components, ic_gradient, grads[2], grads[3], support_info)
            return dict(value=value, gradient=gradient, detail=detail, rho=rho, velocity=velocity,
                        tracer=tracer, population=population, packs=packs, pullback=pullback, q=q)

        if full_gradient:
            secant = json.loads(
                (BASE / 'r2_conditional_pop9_secant2_20261007/result.json').read_text())
            if secant.get('status') != 'CONDITIONAL_POP9_LINE_REDUCED':
                raise ValueError('full gradient requires the reduced population-9 secant')
            rows = secant.get('evaluations') or []
            if len(rows) != 2 or rows[1].get('step') != 'secant':
                raise ValueError('full gradient requires the reduced secant evaluation')
            saved = rows[1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            report['population9_secant2_job_id'] = secant.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced population-9 point, including the IC pullback '
                'and all 24 nuisance components; no IC update, sampler, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8
            tracer_limit = 1e-4 * max(abs(float(saved['tracer0_gradient'])), 1.)
            population_limit = 1e-4 * max(
                abs(float(saved['population_coordinate_9_gradient'])), 1.)
            tracer_ok = abs(float(nuisance[0]) - float(saved['tracer0_gradient'])) <= tracer_limit
            population_ok = abs(
                float(nuisance[18]) - float(saved['population_coordinate_9_gradient'])) <= population_limit
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['tracer0_gradient'] = float(nuisance[0])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(objective_ok, tracer_ok, population_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient2:
            line = json.loads(
                (BASE / 'r2_conditional_tracer2_line_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_TRACER2_LINE_IMPROVED':
                raise ValueError('second full gradient requires the improved tracer-2 line')
            rows = line.get('evaluations') or []
            if len(rows) < 2:
                raise ValueError('second full gradient requires the tracer-2 trials')
            saved = min(rows, key=lambda row: float(row['objective']))
            if saved is rows[0]:
                raise ValueError('tracer-2 line did not improve on its verified point')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            report['tracer2_line_job_id'] = line.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the improved tracer-2 point, including the IC pullback '
                'and all 24 nuisance components; no further tracer-2 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8
            tracer_limit = 1e-4 * max(abs(float(saved['tracer0_gradient'])), 1.)
            tracer2_limit = 1e-4 * max(abs(float(saved['tracer2_gradient'])), 1.)
            population_limit = 1e-4 * max(
                abs(float(saved['population_coordinate_9_gradient'])), 1.)
            tracer_ok = abs(float(nuisance[0]) - float(saved['tracer0_gradient'])) <= tracer_limit
            tracer2_ok = abs(float(nuisance[2]) - float(saved['tracer2_gradient'])) <= tracer2_limit
            population_ok = abs(
                float(nuisance[18]) - float(saved['population_coordinate_9_gradient'])) <= population_limit
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient3:
            secant = json.loads(
                (BASE / 'r2_conditional_tracer0_revisit_secant_20261007/result.json').read_text())
            if secant.get('status') != 'CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED':
                raise ValueError('third full gradient requires the reduced tracer-0 secant')
            rows = secant.get('evaluations') or []
            if len(rows) != 2 or rows[1].get('step') != 'secant':
                raise ValueError('third full gradient requires the secant evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('tracer-0 secant did not improve on its verified point')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            report['tracer0_revisit_secant_job_id'] = secant.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced tracer-0 secant, including the IC pullback '
                'and all 24 nuisance components; no further tracer step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8
            tracer_limit = 1e-4 * max(abs(float(saved['tracer0_gradient'])), 1.)
            tracer2_limit = 1e-4 * max(abs(float(saved['tracer2_gradient'])), 1.)
            population_limit = 1e-4 * max(
                abs(float(saved['population_coordinate_9_gradient'])), 1.)
            tracer_ok = abs(float(nuisance[0]) - float(saved['tracer0_gradient'])) <= tracer_limit
            tracer2_ok = abs(float(nuisance[2]) - float(saved['tracer2_gradient'])) <= tracer2_limit
            population_ok = abs(
                float(nuisance[18]) - float(saved['population_coordinate_9_gradient'])) <= population_limit
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if tracer_pair:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient3_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('tracer pair requires the third full gradient')
            secant = json.loads(
                (BASE / 'r2_conditional_tracer0_revisit_secant_20261007/result.json').read_text())
            if secant.get('status') != 'CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED':
                raise ValueError('tracer pair requires the reduced tracer-0 secant')
            secant_rows = secant.get('evaluations') or []
            if len(secant_rows) != 2 or secant_rows[1].get('step') != 'secant':
                raise ValueError('tracer pair requires the tracer-0 secant column')
            earlier, current = secant_rows[0], secant_rows[1]
            line = json.loads(
                (BASE / 'r2_conditional_tracer2_line_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_TRACER2_LINE_IMPROVED':
                raise ValueError('tracer pair requires the improved tracer-2 line')
            line_rows = line.get('evaluations') or []
            if len(line_rows) < 2:
                raise ValueError('tracer pair requires two tracer-2 evaluations')
            column_prev, column = line_rows[-2], line_rows[-1]
            same_tracer2 = (
                float(earlier['tracer2']) == float(current['tracer2'])
                == float(column['tracer2']) == float(recorded['tracer2']))
            same_column_tracer0 = (
                float(column_prev['tracer0']) == float(column['tracer0'])
                == float(earlier['tracer0']))
            same_population = (
                float(recorded['population_coordinate_9'])
                == float(current['population_coordinate_9'])
                == float(earlier['population_coordinate_9']))
            if not (same_tracer2 and same_column_tracer0 and same_population):
                raise ValueError('tracer-pair columns do not share the recorded coordinates')
            if float(recorded['tracer0']) != float(current['tracer0']):
                raise ValueError('full gradient is not at the tracer-0 secant')

            def archived(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            def objective_matches(got, expected):
                return abs(float(got) - float(expected)) / max(abs(float(expected)), 1.) <= 1e-8

            current_matches = (
                objective_matches(recorded['objective'], current['objective'])
                and archived(recorded['tracer0_gradient'], current['tracer0_gradient'])
                and archived(recorded['tracer2_gradient'], current['tracer2_gradient'])
                and archived(recorded['population_coordinate_9_gradient'],
                             current['population_coordinate_9_gradient']))
            column_matches = (
                objective_matches(column['objective'], earlier['objective'])
                and archived(column['tracer0_gradient'], earlier['tracer0_gradient'])
                and archived(column['tracer2_gradient'], earlier['tracer2_gradient']))
            if not (current_matches and column_matches):
                raise ValueError('tracer-pair archives disagree at the shared points')
            proposed0, proposed2 = tracer_pair_newton_step(
                current['tracer0'], current['tracer0_gradient'],
                earlier['tracer0'], earlier['tracer0_gradient'],
                current['tracer2_gradient'], earlier['tracer2_gradient'],
                column['tracer2'], column['tracer0_gradient'], column['tracer2_gradient'],
                column_prev['tracer2'], column_prev['tracer0_gradient'],
                column_prev['tracer2_gradient'])
            report['full_gradient3_job_id'] = recorded.get('job_id')
            report['tracer0_revisit_secant_job_id'] = secant.get('job_id')
            report['tracer2_line_job_id'] = line.get('job_id')
            report['proposed_tracer0'] = proposed0
            report['proposed_tracer2'] = proposed2
            report['Q_LEAN'] = (
                'one capped Newton step in tracer 0 and tracer 2 from the recorded columns; '
                'one midpoint if that objective is worse; no second Newton, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[18] = float(recorded['population_coordinate_9'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer pair produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)
            objective_ok = baseline is not None and objective_matches(
                baseline['objective'], recorded['objective'])
            gradient_ok = baseline is not None and all(
                archived(baseline[name], recorded[name]) for name in (
                    'tracer0_gradient', 'tracer2_gradient',
                    'population_coordinate_9_gradient'))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])

                def pair_status(rows):
                    best = min(rows, key=lambda item: item['objective'])
                    likelihood = (best['terms']['count_log_likelihood']
                                  + best['terms']['conditional_FP_log_likelihood'])
                    return tracer_pair_status(
                        baseline['objective'], best['objective'],
                        baseline['tracer0_gradient'], best['tracer0_gradient'],
                        baseline['tracer2_gradient'], best['tracer2_gradient'],
                        initial_likelihood, likelihood)

                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = pair_status(records)
                else:
                    proposal = origin.copy()
                    proposal[0] = proposed0
                    proposal[2] = proposed2
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = pair_status(records)
                    else:
                        row['step'] = 'newton'
                        records.append(row)
                        if row['objective'] > baseline['objective']:
                            if budget_seconds - (time.monotonic() - started) < 60.:
                                report['budget_stop'] = True
                            else:
                                midpoint = origin.copy()
                                midpoint[0] = 0.5 * (origin[0] + proposed0)
                                midpoint[2] = 0.5 * (origin[2] + proposed2)
                                mid_row = evaluate_at(midpoint)
                                if mid_row is None:
                                    report['support_shape_changed'] = True
                                else:
                                    mid_row['step'] = 'midpoint'
                                    records.append(mid_row)
                        report['status'] = pair_status(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient4:
            pair = json.loads(
                (BASE / 'r2_conditional_tracer_pair_20261007/result.json').read_text())
            if pair.get('status') != 'CONDITIONAL_TRACER_PAIR_IMPROVED':
                raise ValueError('fourth full gradient requires the improved tracer pair')
            rows = pair.get('evaluations') or []
            if len(rows) < 2 or rows[0].get('step') != 'verify' or rows[-1].get('step') != 'newton':
                raise ValueError('fourth full gradient requires the Newton evaluation')
            saved = rows[-1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('tracer pair did not improve on its verified point')
            if (float(saved['tracer0']) != float(pair['proposed_tracer0'])
                    or float(saved['tracer2']) != float(pair['proposed_tracer2'])):
                raise ValueError('Newton evaluation is not the proposed tracer pair')
            if float(saved['population_coordinate_9']) != float(rows[0]['population_coordinate_9']):
                raise ValueError('tracer pair moved population coordinate 9')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            report['tracer_pair_job_id'] = pair.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the improved tracer-pair point, including the IC pullback '
                'and all 24 nuisance components; no further tracer step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8
            tracer_limit = 1e-4 * max(abs(float(saved['tracer0_gradient'])), 1.)
            tracer2_limit = 1e-4 * max(abs(float(saved['tracer2_gradient'])), 1.)
            population_limit = 1e-4 * max(
                abs(float(saved['population_coordinate_9_gradient'])), 1.)
            tracer_ok = abs(float(nuisance[0]) - float(saved['tracer0_gradient'])) <= tracer_limit
            tracer2_ok = abs(float(nuisance[2]) - float(saved['tracer2_gradient'])) <= tracer2_limit
            population_ok = abs(
                float(nuisance[18]) - float(saved['population_coordinate_9_gradient'])) <= population_limit
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop3_line:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient4_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-3 line requires the fourth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-3 line requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[18] = float(recorded['population_coordinate_9'])
            report['full_gradient4_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one line in population coordinate 3, the FP mean slope b[1, 0]; '
                'verify the full gradient, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-3 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_3_gradient'], nuisance_gradient[12]),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_3_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[12] = accepted[12] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_3_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_3_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_3_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop3_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_3_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop3_secant:
            line = json.loads(
                (BASE / 'r2_conditional_pop3_line_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP3_LINE_IMPROVED':
                raise ValueError('population-3 secant requires the improved line')
            rows = line.get('evaluations') or []
            if len(rows) != 4:
                raise ValueError('population-3 secant requires the four line evaluations')
            positive, negative = rows[2], rows[3]
            if not (float(positive['population_coordinate_3_gradient']) > 0.
                    and float(negative['population_coordinate_3_gradient']) < 0.):
                raise ValueError('population-3 secant requires the sign change on the last step')
            if float(negative['objective']) >= float(positive['objective']):
                raise ValueError('population-3 sign change did not improve the objective')
            shared = ('tracer0', 'tracer2', 'population_coordinate_9')
            if any(float(row[key]) != float(rows[0][key]) for row in rows for key in shared):
                raise ValueError('population-3 line moved tracer 0, tracer 2, or population 9')
            theta = tracer0_secant(
                positive['population_coordinate_3'], positive['population_coordinate_3_gradient'],
                negative['population_coordinate_3'], negative['population_coordinate_3_gradient'])
            report['population3_line_job_id'] = line.get('job_id')
            report['secant_population_coordinate_3'] = theta
            report['Q_LEAN'] = (
                'verify the negative-derivative end of the population-3 bracket, then one secant; '
                'no second step, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(negative['tracer0'])
            origin[2] = float(negative['tracer2'])
            origin[12] = float(negative['population_coordinate_3'])
            origin[18] = float(negative['population_coordinate_9'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-3 secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(negative['objective'])) / max(
                    abs(float(negative['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['population_coordinate_3_gradient'],
                        negative['population_coordinate_3_gradient']),
                matches(baseline['tracer0_gradient'], negative['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], negative['tracer2_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        negative['population_coordinate_9_gradient']),
            ))
            records = []
            start = rows[0]
            start_likelihood = (start['terms']['count_log_likelihood']
                                + start['terms']['conditional_FP_log_likelihood'])

            def score(candidates):
                best = min(candidates, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                return pop3_line_status(
                    start['objective'], best['objective'],
                    abs(float(start['population_coordinate_3_gradient'])),
                    abs(best['population_coordinate_3_gradient']),
                    start_likelihood, likelihood)

            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score(records)
                else:
                    proposal = origin.copy()
                    proposal[12] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score(records)
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        report['status'] = score(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient5:
            secant = json.loads(
                (BASE / 'r2_conditional_pop3_secant_20261007/result.json').read_text())
            if secant.get('status') != 'CONDITIONAL_POP3_LINE_REDUCED':
                raise ValueError('fifth full gradient requires the reduced population-3 secant')
            rows = secant.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != 'secant':
                raise ValueError('fifth full gradient requires the population-3 secant evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('population-3 secant did not improve on its verified point')
            if float(saved['population_coordinate_3']) != float(secant['secant_population_coordinate_3']):
                raise ValueError('secant evaluation is not the proposed population-3 coordinate')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            report['population3_secant_job_id'] = secant.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced population-3 point, including the IC pullback '
                'and all 24 nuisance components; no further population-3 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and population3_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok, population_ok and population3_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop0_line:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient5_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-0 line requires the fifth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-0 line requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            report['full_gradient5_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one line in population coordinate 0, the FP mean intercept b[0, 0]; '
                'verify the full gradient, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-0 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_0_gradient'], nuisance_gradient[9]),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_0_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[9] = accepted[9] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_0_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_0_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_0_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop0_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_0_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop_pair:
            line = json.loads(
                (BASE / 'r2_conditional_pop0_line_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP0_LINE_NO_IMPROVEMENT':
                raise ValueError('population pair requires the unimproved population-0 line')
            rows = line.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != -0.1:
                raise ValueError('population pair requires the rejected population-0 step')
            current, failed = rows[0], rows[1]
            if float(failed['objective']) <= float(current['objective']):
                raise ValueError('population-0 step was not worse than the verified point')
            if not (float(current['population_coordinate_0_gradient']) > 0.
                    and float(failed['population_coordinate_0_gradient']) < 0.):
                raise ValueError('population pair requires the population-0 sign change')
            shared = ('tracer0', 'tracer2', 'population_coordinate_3', 'population_coordinate_9')
            if any(float(current[key]) != float(failed[key]) for key in shared):
                raise ValueError('population-0 line moved another recorded coordinate')
            gradient4 = json.loads(
                (BASE / 'r2_conditional_full_gradient4_20261007/result.json').read_text())
            gradient5 = json.loads(
                (BASE / 'r2_conditional_full_gradient5_20261007/result.json').read_text())
            if gradient4.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population pair requires the fourth full gradient')
            if gradient5.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population pair requires the fifth full gradient')
            earlier = gradient4.get('nuisance_gradient') or []
            latest = gradient5.get('nuisance_gradient') or []
            if len(earlier) != 24 or len(latest) != 24:
                raise ValueError('population pair requires both nuisance gradients')
            pop3_line = json.loads(
                (BASE / 'r2_conditional_pop3_line_20261007/result.json').read_text())
            pop3_rows = pop3_line.get('evaluations') or []
            if not pop3_rows:
                raise ValueError('population pair requires the population-3 line start')
            earlier_pop3 = float(pop3_rows[0]['population_coordinate_3'])
            if earlier_pop3 != float(np.asarray(q_best[N_IC + 12])):
                raise ValueError('population-3 line did not start at the saved coordinate')
            if earlier_pop3 == float(current['population_coordinate_3']):
                raise ValueError('population-3 column needs two distinct coordinates')

            def archived(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            def objective_matches(got, expected):
                return abs(float(got) - float(expected)) / max(abs(float(expected)), 1.) <= 1e-8

            current_matches = (
                objective_matches(gradient5['objective'], current['objective'])
                and archived(latest[9], current['population_coordinate_0_gradient'])
                and archived(latest[12], current['population_coordinate_3_gradient'])
                and archived(gradient5['tracer0_gradient'], current['tracer0_gradient'])
                and archived(gradient5['tracer2_gradient'], current['tracer2_gradient'])
                and archived(gradient5['population_coordinate_9_gradient'],
                             current['population_coordinate_9_gradient'])
                and archived(earlier[12], pop3_rows[0]['population_coordinate_3_gradient']))
            if not current_matches:
                raise ValueError('population-pair archives disagree at the shared points')
            proposed0, proposed3 = tracer_pair_newton_step(
                current['population_coordinate_0'], current['population_coordinate_0_gradient'],
                failed['population_coordinate_0'], failed['population_coordinate_0_gradient'],
                current['population_coordinate_3_gradient'],
                failed['population_coordinate_3_gradient'],
                current['population_coordinate_3'], latest[9], latest[12],
                earlier_pop3, earlier[9], earlier[12])
            report['population0_line_job_id'] = line.get('job_id')
            report['full_gradient4_job_id'] = gradient4.get('job_id')
            report['full_gradient5_job_id'] = gradient5.get('job_id')
            report['proposed_population_coordinate_0'] = proposed0
            report['proposed_population_coordinate_3'] = proposed3
            report['Q_LEAN'] = (
                'one capped Newton step in population 0 and population 3 from the recorded columns; '
                'one midpoint if that objective is worse; no second Newton, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(current['tracer0'])
            origin[2] = float(current['tracer2'])
            origin[9] = float(current['population_coordinate_0'])
            origin[12] = float(current['population_coordinate_3'])
            origin[18] = float(current['population_coordinate_9'])
            pop3_gate = abs(float(earlier[12])) / 10.

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population pair produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)
            objective_ok = baseline is not None and objective_matches(
                baseline['objective'], current['objective'])
            gradient_ok = baseline is not None and all(
                archived(baseline[name], current[name]) for name in (
                    'population_coordinate_0_gradient', 'population_coordinate_3_gradient',
                    'population_coordinate_9_gradient', 'tracer0_gradient', 'tracer2_gradient'))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])

                def pair_status(candidates):
                    best = min(candidates, key=lambda item: item['objective'])
                    likelihood = (best['terms']['count_log_likelihood']
                                  + best['terms']['conditional_FP_log_likelihood'])
                    return pop_pair_status(
                        baseline['objective'], best['objective'],
                        abs(baseline['population_coordinate_0_gradient']),
                        abs(best['population_coordinate_0_gradient']),
                        abs(best['population_coordinate_3_gradient']), pop3_gate,
                        initial_likelihood, likelihood)

                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = pair_status(records)
                else:
                    proposal = origin.copy()
                    proposal[9] = proposed0
                    proposal[12] = proposed3
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = pair_status(records)
                    else:
                        row['step'] = 'newton'
                        records.append(row)
                        if row['objective'] > baseline['objective']:
                            if budget_seconds - (time.monotonic() - started) < 60.:
                                report['budget_stop'] = True
                            else:
                                midpoint = origin.copy()
                                midpoint[9] = 0.5 * (origin[9] + proposed0)
                                midpoint[12] = 0.5 * (origin[12] + proposed3)
                                mid_row = evaluate_at(midpoint)
                                if mid_row is None:
                                    report['support_shape_changed'] = True
                                else:
                                    mid_row['step'] = 'midpoint'
                                    records.append(mid_row)
                        report['status'] = pair_status(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient6:
            pair = json.loads(
                (BASE / 'r2_conditional_pop_pair_20261007/result.json').read_text())
            if pair.get('status') != 'CONDITIONAL_POP_PAIR_REDUCED':
                raise ValueError('sixth full gradient requires the reduced population pair')
            rows = pair.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != 'newton':
                raise ValueError('sixth full gradient requires the population-pair Newton evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('population pair did not improve on its verified point')
            if (float(saved['population_coordinate_0']) != float(pair['proposed_population_coordinate_0'])
                    or float(saved['population_coordinate_3']) != float(pair['proposed_population_coordinate_3'])):
                raise ValueError('Newton evaluation is not the proposed population pair')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            report['population_pair_job_id'] = pair.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced population-pair point, including the IC pullback '
                'and all 24 nuisance components; no further population step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            matched = objective_ok and tracer_ok and tracer2_ok and population0_ok and population3_ok and population_ok
            if matched:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok, population_ok and population0_ok and population3_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop2_line:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient6_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-2 line requires the sixth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-2 line requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            report['full_gradient6_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one line in population coordinate 2, the FP mean intercept b[0, 2]; '
                'verify the full gradient, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-2 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_2_gradient'], nuisance_gradient[11]),
                matches(baseline['population_coordinate_12_gradient'], nuisance_gradient[21]),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_2_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[11] = accepted[11] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_2_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_2_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_2_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop2_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_2_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop2_secant:
            line = json.loads(
                (BASE / 'r2_conditional_pop2_line_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP2_LINE_NO_IMPROVEMENT':
                raise ValueError('population-2 secant requires the unimproved line')
            rows = line.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or abs(float(rows[1].get('step')) - 0.1) > 1e-12:
                raise ValueError('population-2 secant requires the rejected positive step')
            current, failed = rows[0], rows[1]
            if float(failed['objective']) <= float(current['objective']):
                raise ValueError('population-2 step was not worse than the verified point')
            if not (float(current['population_coordinate_2_gradient']) < 0.
                    and float(failed['population_coordinate_2_gradient']) > 0.):
                raise ValueError('population-2 secant requires the sign change')
            shared = ('tracer0', 'tracer2', 'population_coordinate_0',
                      'population_coordinate_9', 'population_coordinate_12')
            if any(float(current[key]) != float(failed[key]) for key in shared):
                raise ValueError('population-2 line moved another recorded coordinate')
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient6_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-2 secant requires the sixth full gradient')
            theta = tracer0_secant(
                failed['population_coordinate_2'], failed['population_coordinate_2_gradient'],
                current['population_coordinate_2'], current['population_coordinate_2_gradient'])
            report['population2_line_job_id'] = line.get('job_id')
            report['full_gradient6_job_id'] = recorded.get('job_id')
            report['secant_population_coordinate_2'] = theta
            report['Q_LEAN'] = (
                'verify the accepted end of the population-2 bracket, then one secant; '
                'no second step, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(current['tracer0'])
            origin[2] = float(current['tracer2'])
            origin[9] = float(current['population_coordinate_0'])
            origin[11] = float(current['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(current['population_coordinate_9'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-2 secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            baseline = evaluate_at(origin)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(current['objective'])) / max(
                    abs(float(current['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all(
                matches(baseline[name], current[name]) for name in (
                    'population_coordinate_2_gradient', 'population_coordinate_0_gradient',
                    'population_coordinate_9_gradient', 'population_coordinate_12_gradient',
                    'tracer0_gradient', 'tracer2_gradient'))
            records = []
            start_likelihood = (current['terms']['count_log_likelihood']
                                + current['terms']['conditional_FP_log_likelihood'])

            def score(candidates):
                best = min(candidates, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                return pop2_line_status(
                    current['objective'], best['objective'],
                    abs(float(current['population_coordinate_2_gradient'])),
                    abs(best['population_coordinate_2_gradient']),
                    start_likelihood, likelihood)

            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score(records)
                else:
                    proposal = origin.copy()
                    proposal[11] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score(records)
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        report['status'] = score(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient7:
            secant = json.loads(
                (BASE / 'r2_conditional_pop2_secant_20261007/result.json').read_text())
            if secant.get('status') != 'CONDITIONAL_POP2_LINE_REDUCED':
                raise ValueError('seventh full gradient requires the reduced population-2 secant')
            rows = secant.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != 'secant':
                raise ValueError('seventh full gradient requires the population-2 secant evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('population-2 secant did not improve on its verified point')
            if float(saved['population_coordinate_2']) != float(secant['secant_population_coordinate_2']):
                raise ValueError('secant evaluation is not the proposed population-2 coordinate')
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient6_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('seventh full gradient requires the sixth full gradient')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(recorded['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            report['population2_secant_job_id'] = secant.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced population-2 point, including the IC pullback '
                'and all 24 nuisance components; no further population-2 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            population_ok = population0_ok and population2_ok and population9_ok and population12_ok
            if objective_ok and tracer_ok and tracer2_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop12_line:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient7_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-12 line requires the seventh full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-12 line requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            report['full_gradient7_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one line in population coordinate 12, the log-Cholesky entry L20; '
                'verify the full gradient, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-12 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], nuisance_gradient[4]),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'], nuisance_gradient[12]),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'], nuisance_gradient[23]),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_12_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[21] = accepted[21] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_12_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_12_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_12_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop12_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_12_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient8:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop12_line_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP12_LINE_IMPROVED':
                raise ValueError('eighth full gradient requires the improved population-12 line')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('eighth full gradient requires the three population-12 steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-12 line did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-12 line did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_12'])
                   - (float(rows[0]['population_coordinate_12']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-12 point is not the third step')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population12_line_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the improved population-12 point, including the IC pullback '
                'and all 24 nuisance components; no further population-12 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop14_line:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient8_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-14 line requires the eighth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-14 line requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient8_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one line in population coordinate 14, the log-Cholesky entry log L22; '
                'verify the full gradient, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-14 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_14_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[23] = accepted[23] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_14_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_14_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_14_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop14_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_14_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient9:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop14_line_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP14_LINE_IMPROVED':
                raise ValueError('ninth full gradient requires the improved population-14 line')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('ninth full gradient requires the three population-14 steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-14 line did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-14 line did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_14'])
                   - (float(rows[0]['population_coordinate_14']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-14 point is not the third step')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population14_line_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the improved population-14 point, including the IC pullback '
                'and all 24 nuisance components; no further population-14 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop12_revisit:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient9_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-12 revisit requires the ninth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-12 revisit requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient9_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 12, the log-Cholesky entry L20, '
                'at the ninth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-12 revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_12_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[21] = accepted[21] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_12_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_12_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_12_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop12_revisit_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_12_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient10:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop12_revisit_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP12_REVISIT_IMPROVED':
                raise ValueError('tenth full gradient requires the improved population-12 revisit')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('tenth full gradient requires the three population-12 revisit steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-12 revisit did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-12 revisit did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_12'])
                   - (float(rows[0]['population_coordinate_12']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-12 point is not the third step')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population12_revisit_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the revisited population-12 point, including the IC pullback '
                'and all 24 nuisance components; no further population-12 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop14_revisit:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient10_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-14 revisit requires the tenth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-14 revisit requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient10_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 14, the log-Cholesky entry log L22, '
                'at the tenth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-14 revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_14_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[23] = accepted[23] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_14_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_14_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_14_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop14_revisit_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_14_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient11:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop14_revisit_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP14_REVISIT_IMPROVED':
                raise ValueError('eleventh full gradient requires the improved population-14 revisit')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('eleventh full gradient requires the three population-14 revisit steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-14 revisit did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-14 revisit did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_14'])
                   - (float(rows[0]['population_coordinate_14']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-14 point is not the third step')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population14_revisit_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the revisited population-14 point, including the IC pullback '
                'and all 24 nuisance components; no further population-14 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop9_revisit:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient11_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-9 revisit requires the eleventh full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-9 revisit requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient11_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 9, the log-Cholesky entry log L00, '
                'at the eleventh-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-9 revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_9_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[18] = accepted[18] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_9_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_9_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_9_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop9_revisit_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_9_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient12:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop9_revisit_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP9_REVISIT_REDUCED':
                raise ValueError('twelfth full gradient requires the reduced population-9 revisit')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('twelfth full gradient requires the three population-9 revisit steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-9 revisit did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-9 revisit did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_9'])
                   - (float(rows[0]['population_coordinate_9']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-9 point is not the third step')
            if not (float(rows[2]['population_coordinate_9_gradient']) > 0.
                    and float(rows[3]['population_coordinate_9_gradient']) < 0.):
                raise ValueError('population-9 revisit did not change sign on the third step')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population9_revisit_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced population-9 point, including the IC pullback '
                'and all 24 nuisance components; no further population-9 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop14_revisit2:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient12_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-14 second revisit requires the twelfth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-14 second revisit requires all 24 nuisance derivatives')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient12_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 14, the log-Cholesky entry log L22, '
                'at the twelfth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-14 second revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_14_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[23] = accepted[23] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_14_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_14_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_14_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop14_revisit2_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_14_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient13:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop14_revisit2_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP14_REVISIT2_IMPROVED':
                raise ValueError('thirteenth full gradient requires the improved population-14 revisit')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('thirteenth full gradient requires the three population-14 steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-14 revisit did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-14 revisit did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_14'])
                   - (float(rows[0]['population_coordinate_14']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-14 point is not the third step')
            if any(float(row['population_coordinate_14_gradient']) <= 0. for row in rows):
                raise ValueError('population-14 revisit changed the derivative sign')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population14_revisit2_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the second population-14 revisit, including the IC pullback '
                'and all 24 nuisance components; no further population-14 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop0_revisit:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient13_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-0 revisit requires the thirteenth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-0 revisit requires all 24 nuisance derivatives')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 0
                    or int(location.get('flat_index', -1)) != N_IC + 9):
                raise ValueError(
                    'population-0 revisit requires the joint infinity norm at population coordinate 0')
            if abs(float(location.get('value'))
                   - float(recorded['population_coordinate_0_gradient'])) > 1e-6:
                raise ValueError('joint infinity norm does not match the population-0 derivative')
            if float(recorded['population_coordinate_0_gradient']) >= 0.:
                raise ValueError('population-0 revisit requires a negative derivative')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient13_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 0, the FP mean intercept b[0, 0], '
                'at the thirteenth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-0 revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_0_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[9] = accepted[9] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_0_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_0_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_0_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop0_revisit_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_0_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop0_revisit_secant:
            line = json.loads(
                (BASE / 'r2_conditional_pop0_revisit_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP0_REVISIT_NO_IMPROVEMENT':
                raise ValueError('population-0 secant requires the unimproved revisit')
            rows = line.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or abs(float(rows[1].get('step')) - 0.1) > 1e-12:
                raise ValueError('population-0 secant requires the rejected positive step')
            current, failed = rows[0], rows[1]
            if float(failed['objective']) <= float(current['objective']):
                raise ValueError('population-0 step was not worse than the verified point')
            if not (float(current['population_coordinate_0_gradient']) < 0.
                    and float(failed['population_coordinate_0_gradient']) > 0.):
                raise ValueError('population-0 secant requires the sign change')
            shared = ('tracer0', 'tracer2', 'tracer4', 'population_coordinate_2',
                      'population_coordinate_3', 'population_coordinate_9',
                      'population_coordinate_12', 'population_coordinate_14')
            if any(float(current[key]) != float(failed[key]) for key in shared):
                raise ValueError('population-0 revisit moved another recorded coordinate')
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient13_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-0 secant requires the thirteenth full gradient')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 0
                    or int(location.get('flat_index', -1)) != N_IC + 9):
                raise ValueError('population-0 secant requires the joint infinity norm at population 0')
            if float(recorded['population_coordinate_0']) != float(current['population_coordinate_0']):
                raise ValueError('population-0 revisit did not start at the recorded coordinate')
            if abs(float(recorded['objective']) - float(current['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) > 1e-8:
                raise ValueError('population-0 revisit did not reproduce the recorded objective')
            theta = tracer0_secant(
                failed['population_coordinate_0'], failed['population_coordinate_0_gradient'],
                current['population_coordinate_0'], current['population_coordinate_0_gradient'])
            report['population0_revisit_job_id'] = line.get('job_id')
            report['full_gradient13_job_id'] = recorded.get('job_id')
            report['secant_population_coordinate_0'] = theta
            report['Q_LEAN'] = (
                'verify the accepted end of the population-0 bracket, then one secant; '
                'no second step, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(current['tracer0'])
            origin[2] = float(current['tracer2'])
            origin[4] = float(current['tracer4'])
            origin[9] = float(current['population_coordinate_0'])
            origin[11] = float(current['population_coordinate_2'])
            origin[12] = float(current['population_coordinate_3'])
            origin[18] = float(current['population_coordinate_9'])
            origin[21] = float(current['population_coordinate_12'])
            origin[23] = float(current['population_coordinate_14'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-0 secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            baseline = evaluate_at(origin)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(current['objective'])) / max(
                    abs(float(current['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all(
                matches(baseline[name], current[name]) for name in (
                    'population_coordinate_0_gradient', 'population_coordinate_2_gradient',
                    'population_coordinate_3_gradient', 'population_coordinate_9_gradient',
                    'population_coordinate_12_gradient', 'population_coordinate_14_gradient',
                    'tracer0_gradient', 'tracer2_gradient', 'tracer4_gradient'))
            records = []
            start_likelihood = (current['terms']['count_log_likelihood']
                                + current['terms']['conditional_FP_log_likelihood'])

            def score(candidates):
                best = min(candidates, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                return pop0_revisit_status(
                    current['objective'], best['objective'],
                    abs(float(current['population_coordinate_0_gradient'])),
                    abs(best['population_coordinate_0_gradient']),
                    start_likelihood, likelihood)

            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score(records)
                else:
                    proposal = origin.copy()
                    proposal[9] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score(records)
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        report['status'] = score(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient14:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop0_revisit_secant_20261007/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP0_REVISIT_REDUCED':
                raise ValueError('fourteenth full gradient requires the reduced population-0 secant')
            rows = recorded.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != 'secant':
                raise ValueError('fourteenth full gradient requires the population-0 secant evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('population-0 secant did not improve on its verified point')
            if float(saved['population_coordinate_0']) != float(recorded['secant_population_coordinate_0']):
                raise ValueError('secant evaluation is not the proposed population-0 coordinate')
            if abs(float(saved['population_coordinate_0_gradient'])) > abs(
                    float(rows[0]['population_coordinate_0_gradient'])) / 10.:
                raise ValueError('population-0 secant did not meet its tenfold gate')
            initial_likelihood = (rows[0]['terms']['count_log_likelihood']
                                  + rows[0]['terms']['conditional_FP_log_likelihood'])
            final_likelihood = (saved['terms']['count_log_likelihood']
                                + saved['terms']['conditional_FP_log_likelihood'])
            if float(final_likelihood) < float(initial_likelihood) - 1e-6:
                raise ValueError('population-0 secant lowered the count plus conditional-FP likelihood')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population0_revisit_secant_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced population-0 secant, including the IC pullback '
                'and all 24 nuisance components; no further population-0 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop3_revisit:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient14_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-3 revisit requires the fourteenth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-3 revisit requires all 24 nuisance derivatives')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 3
                    or int(location.get('flat_index', -1)) != N_IC + 12):
                raise ValueError(
                    'population-3 revisit requires the joint infinity norm at population coordinate 3')
            if abs(float(location.get('value'))
                   - float(recorded['population_coordinate_3_gradient'])) > 1e-6:
                raise ValueError('joint infinity norm does not match the population-3 derivative')
            if float(recorded['population_coordinate_3_gradient']) >= 0.:
                raise ValueError('population-3 revisit requires a negative derivative')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient14_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 3, the FP mean slope b[1, 0], '
                'at the fourteenth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-3 revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_3_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[12] = accepted[12] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_3_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_3_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_3_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop3_revisit_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_3_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop3_revisit_secant:
            line = json.loads(
                (BASE / 'r2_conditional_pop3_revisit_20261008/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP3_REVISIT_NO_IMPROVEMENT':
                raise ValueError('population-3 secant requires the unimproved revisit')
            rows = line.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or abs(float(rows[1].get('step')) - 0.1) > 1e-12:
                raise ValueError('population-3 secant requires the rejected positive step')
            current, failed = rows[0], rows[1]
            if float(failed['objective']) <= float(current['objective']):
                raise ValueError('population-3 step was not worse than the verified point')
            if not (float(current['population_coordinate_3_gradient']) < 0.
                    and float(failed['population_coordinate_3_gradient']) > 0.):
                raise ValueError('population-3 secant requires the sign change')
            shared = ('tracer0', 'tracer2', 'tracer4', 'population_coordinate_0',
                      'population_coordinate_2', 'population_coordinate_9',
                      'population_coordinate_12', 'population_coordinate_14')
            if any(float(current[key]) != float(failed[key]) for key in shared):
                raise ValueError('population-3 revisit moved another recorded coordinate')
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient14_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-3 secant requires the fourteenth full gradient')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 3
                    or int(location.get('flat_index', -1)) != N_IC + 12):
                raise ValueError('population-3 secant requires the joint infinity norm at population 3')
            if float(recorded['population_coordinate_3']) != float(current['population_coordinate_3']):
                raise ValueError('population-3 revisit did not start at the recorded coordinate')
            if abs(float(recorded['objective']) - float(current['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) > 1e-8:
                raise ValueError('population-3 revisit did not reproduce the recorded objective')
            theta = tracer0_secant(
                failed['population_coordinate_3'], failed['population_coordinate_3_gradient'],
                current['population_coordinate_3'], current['population_coordinate_3_gradient'])
            report['population3_revisit_job_id'] = line.get('job_id')
            report['full_gradient14_job_id'] = recorded.get('job_id')
            report['secant_population_coordinate_3'] = theta
            report['Q_LEAN'] = (
                'verify the accepted end of the population-3 bracket, then one secant; '
                'no second step, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(current['tracer0'])
            origin[2] = float(current['tracer2'])
            origin[4] = float(current['tracer4'])
            origin[9] = float(current['population_coordinate_0'])
            origin[11] = float(current['population_coordinate_2'])
            origin[12] = float(current['population_coordinate_3'])
            origin[18] = float(current['population_coordinate_9'])
            origin[21] = float(current['population_coordinate_12'])
            origin[23] = float(current['population_coordinate_14'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-3 secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            baseline = evaluate_at(origin)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(current['objective'])) / max(
                    abs(float(current['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all(
                matches(baseline[name], current[name]) for name in (
                    'population_coordinate_0_gradient', 'population_coordinate_2_gradient',
                    'population_coordinate_3_gradient', 'population_coordinate_9_gradient',
                    'population_coordinate_12_gradient', 'population_coordinate_14_gradient',
                    'tracer0_gradient', 'tracer2_gradient', 'tracer4_gradient'))
            records = []
            start_likelihood = (current['terms']['count_log_likelihood']
                                + current['terms']['conditional_FP_log_likelihood'])

            def score(candidates):
                best = min(candidates, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                return pop3_revisit_status(
                    current['objective'], best['objective'],
                    abs(float(current['population_coordinate_3_gradient'])),
                    abs(best['population_coordinate_3_gradient']),
                    start_likelihood, likelihood)

            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score(records)
                else:
                    proposal = origin.copy()
                    proposal[12] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score(records)
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        report['status'] = score(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient15:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop3_revisit_secant_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP3_REVISIT_REDUCED':
                raise ValueError('fifteenth full gradient requires the reduced population-3 secant')
            rows = recorded.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != 'secant':
                raise ValueError('fifteenth full gradient requires the population-3 secant evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('population-3 secant did not improve on its verified point')
            if float(saved['population_coordinate_3']) != float(recorded['secant_population_coordinate_3']):
                raise ValueError('secant evaluation is not the proposed population-3 coordinate')
            if abs(float(saved['population_coordinate_3_gradient'])) > abs(
                    float(rows[0]['population_coordinate_3_gradient'])) / 10.:
                raise ValueError('population-3 secant did not meet its tenfold gate')
            initial_likelihood = (rows[0]['terms']['count_log_likelihood']
                                  + rows[0]['terms']['conditional_FP_log_likelihood'])
            final_likelihood = (saved['terms']['count_log_likelihood']
                                + saved['terms']['conditional_FP_log_likelihood'])
            if float(final_likelihood) < float(initial_likelihood) - 1e-6:
                raise ValueError('population-3 secant lowered the count plus conditional-FP likelihood')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population3_revisit_secant_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced population-3 secant, including the IC pullback '
                'and all 24 nuisance components; no further population-3 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop0_revisit2:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient15_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-0 second revisit requires the fifteenth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-0 second revisit requires all 24 nuisance derivatives')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 0
                    or int(location.get('flat_index', -1)) != N_IC + 9):
                raise ValueError(
                    'population-0 second revisit requires the joint infinity norm at population coordinate 0')
            if abs(float(location.get('value'))
                   - float(recorded['population_coordinate_0_gradient'])) > 1e-6:
                raise ValueError('joint infinity norm does not match the population-0 derivative')
            if float(recorded['population_coordinate_0_gradient']) >= 0.:
                raise ValueError('population-0 second revisit requires a negative derivative')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient15_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 0, the FP mean intercept b[0, 0], '
                'at the fifteenth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-0 second revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_0_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[9] = accepted[9] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_0_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_0_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_0_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop0_revisit2_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_0_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop0_revisit2_secant:
            line = json.loads(
                (BASE / 'r2_conditional_pop0_revisit2_20261008/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP0_REVISIT2_NO_IMPROVEMENT':
                raise ValueError('population-0 second secant requires the unimproved revisit')
            rows = line.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or abs(float(rows[1].get('step')) - 0.1) > 1e-12:
                raise ValueError('population-0 second secant requires the rejected positive step')
            current, failed = rows[0], rows[1]
            if float(failed['objective']) <= float(current['objective']):
                raise ValueError('population-0 step was not worse than the verified point')
            if not (float(current['population_coordinate_0_gradient']) < 0.
                    and float(failed['population_coordinate_0_gradient']) > 0.):
                raise ValueError('population-0 second secant requires the sign change')
            shared = ('tracer0', 'tracer2', 'tracer4', 'population_coordinate_2',
                      'population_coordinate_3', 'population_coordinate_9',
                      'population_coordinate_12', 'population_coordinate_14')
            if any(float(current[key]) != float(failed[key]) for key in shared):
                raise ValueError('population-0 revisit moved another recorded coordinate')
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient15_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-0 second secant requires the fifteenth full gradient')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 0
                    or int(location.get('flat_index', -1)) != N_IC + 9):
                raise ValueError('population-0 second secant requires the joint infinity norm at population 0')
            if float(recorded['population_coordinate_0']) != float(current['population_coordinate_0']):
                raise ValueError('population-0 revisit did not start at the recorded coordinate')
            if abs(float(recorded['objective']) - float(current['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) > 1e-8:
                raise ValueError('population-0 revisit did not reproduce the recorded objective')
            theta = tracer0_secant(
                failed['population_coordinate_0'], failed['population_coordinate_0_gradient'],
                current['population_coordinate_0'], current['population_coordinate_0_gradient'])
            report['population0_revisit2_job_id'] = line.get('job_id')
            report['full_gradient15_job_id'] = recorded.get('job_id')
            report['secant_population_coordinate_0'] = theta
            report['Q_LEAN'] = (
                'verify the accepted end of the second population-0 bracket, then one secant; '
                'no second step, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(current['tracer0'])
            origin[2] = float(current['tracer2'])
            origin[4] = float(current['tracer4'])
            origin[9] = float(current['population_coordinate_0'])
            origin[11] = float(current['population_coordinate_2'])
            origin[12] = float(current['population_coordinate_3'])
            origin[18] = float(current['population_coordinate_9'])
            origin[21] = float(current['population_coordinate_12'])
            origin[23] = float(current['population_coordinate_14'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-0 second secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            baseline = evaluate_at(origin)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(current['objective'])) / max(
                    abs(float(current['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all(
                matches(baseline[name], current[name]) for name in (
                    'population_coordinate_0_gradient', 'population_coordinate_2_gradient',
                    'population_coordinate_3_gradient', 'population_coordinate_9_gradient',
                    'population_coordinate_12_gradient', 'population_coordinate_14_gradient',
                    'tracer0_gradient', 'tracer2_gradient', 'tracer4_gradient'))
            records = []
            start_likelihood = (current['terms']['count_log_likelihood']
                                + current['terms']['conditional_FP_log_likelihood'])

            def score(candidates):
                best = min(candidates, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                return pop0_revisit2_status(
                    current['objective'], best['objective'],
                    abs(float(current['population_coordinate_0_gradient'])),
                    abs(best['population_coordinate_0_gradient']),
                    start_likelihood, likelihood)

            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score(records)
                else:
                    proposal = origin.copy()
                    proposal[9] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score(records)
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        report['status'] = score(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient16:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop0_revisit2_secant_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP0_REVISIT2_REDUCED':
                raise ValueError('sixteenth full gradient requires the reduced population-0 secant')
            rows = recorded.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != 'secant':
                raise ValueError('sixteenth full gradient requires the population-0 secant evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('population-0 secant did not improve on its verified point')
            if float(saved['population_coordinate_0']) != float(recorded['secant_population_coordinate_0']):
                raise ValueError('secant evaluation is not the proposed population-0 coordinate')
            if abs(float(saved['population_coordinate_0_gradient'])) > abs(
                    float(rows[0]['population_coordinate_0_gradient'])) / 10.:
                raise ValueError('population-0 secant did not meet its tenfold gate')
            initial_likelihood = (rows[0]['terms']['count_log_likelihood']
                                  + rows[0]['terms']['conditional_FP_log_likelihood'])
            final_likelihood = (saved['terms']['count_log_likelihood']
                                + saved['terms']['conditional_FP_log_likelihood'])
            if float(final_likelihood) < float(initial_likelihood) - 1e-6:
                raise ValueError('population-0 secant lowered the count plus conditional-FP likelihood')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population0_revisit2_secant_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the second reduced population-0 secant, including the IC pullback '
                'and all 24 nuisance components; no further population-0 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop14_revisit3:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient16_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-14 third revisit requires the sixteenth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-14 third revisit requires all 24 nuisance derivatives')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 14
                    or int(location.get('flat_index', -1)) != N_IC + 23):
                raise ValueError(
                    'population-14 third revisit requires the joint infinity norm at population coordinate 14')
            if abs(float(location.get('value'))
                   - float(recorded['population_coordinate_14_gradient'])) > 1e-6:
                raise ValueError('joint infinity norm does not match the population-14 derivative')
            if float(recorded['population_coordinate_14_gradient']) <= 0.:
                raise ValueError('population-14 third revisit requires a positive derivative')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient16_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 14, the log-Cholesky entry log L22, '
                'at the sixteenth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-14 third revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_14_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[23] = accepted[23] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_14_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_14_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_14_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop14_revisit3_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_14_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient17:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop14_revisit3_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP14_REVISIT3_IMPROVED':
                raise ValueError('seventeenth full gradient requires the improved population-14 revisit')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('seventeenth full gradient requires the three population-14 steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-14 revisit did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-14 revisit did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_14'])
                   - (float(rows[0]['population_coordinate_14']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-14 point is not the third step')
            if any(float(row['population_coordinate_14_gradient']) <= 0. for row in rows):
                raise ValueError('population-14 revisit changed the derivative sign')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population14_revisit3_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the third population-14 revisit, including the IC pullback '
                'and all 24 nuisance components; no further population-14 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop14_revisit4:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient17_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-14 fourth revisit requires the seventeenth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-14 fourth revisit requires all 24 nuisance derivatives')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 14
                    or int(location.get('flat_index', -1)) != N_IC + 23):
                raise ValueError(
                    'population-14 fourth revisit requires the joint infinity norm at population coordinate 14')
            if abs(float(location.get('value'))
                   - float(recorded['population_coordinate_14_gradient'])) > 1e-6:
                raise ValueError('joint infinity norm does not match the population-14 derivative')
            if float(recorded['population_coordinate_14_gradient']) <= 0.:
                raise ValueError('population-14 fourth revisit requires a positive derivative')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient17_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 14, the log-Cholesky entry log L22, '
                'at the seventeenth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-14 fourth revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_14_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[23] = accepted[23] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_14_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_14_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_14_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop14_revisit4_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_14_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient18:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop14_revisit4_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP14_REVISIT4_IMPROVED':
                raise ValueError('eighteenth full gradient requires the improved population-14 revisit')
            rows = recorded.get('evaluations') or []
            if len(rows) != 4 or rows[0].get('step') != 'verify':
                raise ValueError('eighteenth full gradient requires the three population-14 steps')
            if any(float(row['step']) >= 0. for row in rows[1:]):
                raise ValueError('population-14 fourth revisit did not step against a positive derivative')
            if any(float(rows[i]['objective']) >= float(rows[i - 1]['objective']) for i in range(1, 4)):
                raise ValueError('population-14 fourth revisit did not improve at every step')
            if abs(float(rows[-1]['population_coordinate_14'])
                   - (float(rows[0]['population_coordinate_14']) - 0.3)) > 1e-12:
                raise ValueError('accepted population-14 point is not three steps of -0.1')
            if any(float(row['population_coordinate_14_gradient']) <= 0. for row in rows):
                raise ValueError('population-14 fourth revisit changed the derivative sign')
            saved = rows[-1]
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population14_revisit4_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the fourth population-14 revisit, including the IC pullback '
                'and all 24 nuisance components; no further population-14 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop3_revisit2:
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient18_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-3 second revisit requires the eighteenth full gradient')
            nuisance_gradient = recorded.get('nuisance_gradient') or []
            if len(nuisance_gradient) != 24:
                raise ValueError('population-3 second revisit requires all 24 nuisance derivatives')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 3
                    or int(location.get('flat_index', -1)) != N_IC + 12):
                raise ValueError(
                    'population-3 second revisit requires the joint infinity norm at population coordinate 3')
            if abs(float(location.get('value'))
                   - float(recorded['population_coordinate_3_gradient'])) > 1e-6:
                raise ValueError('joint infinity norm does not match the population-3 derivative')
            if float(recorded['population_coordinate_3_gradient']) >= 0.:
                raise ValueError('population-3 second revisit requires a negative derivative')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(recorded['tracer0'])
            origin[2] = float(recorded['tracer2'])
            origin[4] = float(recorded['tracer4'])
            origin[9] = float(recorded['population_coordinate_0'])
            origin[11] = float(recorded['population_coordinate_2'])
            origin[12] = float(recorded['population_coordinate_3'])
            origin[18] = float(recorded['population_coordinate_9'])
            origin[21] = float(recorded['population_coordinate_12'])
            origin[23] = float(recorded['population_coordinate_14'])
            report['full_gradient18_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one new line in population coordinate 3, the FP mean slope b[1, 0], '
                'at the eighteenth-gradient point; verify, then at most three steps; no IC update or heldout')

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-3 second revisit produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(recorded['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all((
                matches(baseline['tracer0_gradient'], recorded['tracer0_gradient']),
                matches(baseline['tracer2_gradient'], recorded['tracer2_gradient']),
                matches(baseline['tracer4_gradient'], recorded['tracer4_gradient']),
                matches(baseline['population_coordinate_0_gradient'],
                        recorded['population_coordinate_0_gradient']),
                matches(baseline['population_coordinate_2_gradient'],
                        recorded['population_coordinate_2_gradient']),
                matches(baseline['population_coordinate_3_gradient'],
                        recorded['population_coordinate_3_gradient']),
                matches(baseline['population_coordinate_9_gradient'],
                        recorded['population_coordinate_9_gradient']),
                matches(baseline['population_coordinate_12_gradient'],
                        recorded['population_coordinate_12_gradient']),
                matches(baseline['population_coordinate_14_gradient'],
                        recorded['population_coordinate_14_gradient']),
            ))
            records = []
            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_3_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                initial_likelihood = (baseline['terms']['count_log_likelihood']
                                      + baseline['terms']['conditional_FP_log_likelihood'])
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[12] = accepted[12] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_3_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_3_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_3_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                report['status'] = pop3_revisit2_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_3_gradient']),
                    initial_likelihood, likelihood)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if pop3_revisit2_secant:
            line = json.loads(
                (BASE / 'r2_conditional_pop3_revisit2_20261008/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_POP3_REVISIT2_NO_IMPROVEMENT':
                raise ValueError('population-3 second secant requires the unimproved revisit')
            rows = line.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or abs(float(rows[1].get('step')) - 0.1) > 1e-12:
                raise ValueError('population-3 second secant requires the rejected positive step')
            current, failed = rows[0], rows[1]
            if float(failed['objective']) <= float(current['objective']):
                raise ValueError('population-3 second step was not worse than the verified point')
            if not (float(current['population_coordinate_3_gradient']) < 0.
                    and float(failed['population_coordinate_3_gradient']) > 0.):
                raise ValueError('population-3 second secant requires the sign change')
            shared = ('tracer0', 'tracer2', 'tracer4', 'population_coordinate_0',
                      'population_coordinate_2', 'population_coordinate_9',
                      'population_coordinate_12', 'population_coordinate_14')
            if any(float(current[key]) != float(failed[key]) for key in shared):
                raise ValueError('population-3 second revisit moved another recorded coordinate')
            recorded = json.loads(
                (BASE / 'r2_conditional_full_gradient18_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED':
                raise ValueError('population-3 second secant requires the eighteenth full gradient')
            location = (recorded.get('joint_gradient') or {}).get('infinity_norm_location') or {}
            if (location.get('block') != 'population'
                    or int(location.get('index_in_block', -1)) != 3
                    or int(location.get('flat_index', -1)) != N_IC + 12):
                raise ValueError('population-3 second secant requires the joint infinity norm at population 3')
            if float(recorded['population_coordinate_3']) != float(current['population_coordinate_3']):
                raise ValueError('population-3 second revisit did not start at the recorded coordinate')
            if abs(float(recorded['objective']) - float(current['objective'])) / max(
                    abs(float(recorded['objective'])), 1.) > 1e-8:
                raise ValueError('population-3 second revisit did not reproduce the recorded objective')
            theta = tracer0_secant(
                failed['population_coordinate_3'], failed['population_coordinate_3_gradient'],
                current['population_coordinate_3'], current['population_coordinate_3_gradient'])
            report['population3_revisit2_job_id'] = line.get('job_id')
            report['full_gradient18_job_id'] = recorded.get('job_id')
            report['secant_population_coordinate_3'] = theta
            report['Q_LEAN'] = (
                'verify the accepted end of the second population-3 bracket, then one secant; '
                'no second step, IC update, or heldout')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(current['tracer0'])
            origin[2] = float(current['tracer2'])
            origin[4] = float(current['tracer4'])
            origin[9] = float(current['population_coordinate_0'])
            origin[11] = float(current['population_coordinate_2'])
            origin[12] = float(current['population_coordinate_3'])
            origin[18] = float(current['population_coordinate_9'])
            origin[21] = float(current['population_coordinate_12'])
            origin[23] = float(current['population_coordinate_14'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(rho, velocity, jnp.asarray(nuisance[:9]), 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-3 second secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]), tracer2=float(nuisance[2]),
                    tracer4=float(nuisance[4]),
                    population_coordinate_0=float(nuisance[9]),
                    population_coordinate_2=float(nuisance[11]),
                    population_coordinate_3=float(nuisance[12]),
                    population_coordinate_9=float(nuisance[18]),
                    population_coordinate_12=float(nuisance[21]),
                    population_coordinate_14=float(nuisance[23]),
                    tracer0_gradient=float(gradient[0]), tracer2_gradient=float(gradient[2]),
                    tracer4_gradient=float(gradient[4]),
                    population_coordinate_0_gradient=float(gradient[9]),
                    population_coordinate_2_gradient=float(gradient[11]),
                    population_coordinate_3_gradient=float(gradient[12]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    population_coordinate_12_gradient=float(gradient[21]),
                    population_coordinate_14_gradient=float(gradient[23]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matches(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            baseline = evaluate_at(origin)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(current['objective'])) / max(
                    abs(float(current['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and all(
                matches(baseline[name], current[name]) for name in (
                    'population_coordinate_0_gradient', 'population_coordinate_2_gradient',
                    'population_coordinate_3_gradient', 'population_coordinate_9_gradient',
                    'population_coordinate_12_gradient', 'population_coordinate_14_gradient',
                    'tracer0_gradient', 'tracer2_gradient', 'tracer4_gradient'))
            records = []
            start_likelihood = (current['terms']['count_log_likelihood']
                                + current['terms']['conditional_FP_log_likelihood'])

            def score(candidates):
                best = min(candidates, key=lambda item: item['objective'])
                likelihood = (best['terms']['count_log_likelihood']
                              + best['terms']['conditional_FP_log_likelihood'])
                return pop3_revisit2_status(
                    current['objective'], best['objective'],
                    abs(float(current['population_coordinate_3_gradient'])),
                    abs(best['population_coordinate_3_gradient']),
                    start_likelihood, likelihood)

            if not (objective_ok and gradient_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                baseline['step'] = 'verify'
                records.append(baseline)
                if budget_seconds - (time.monotonic() - started) < 60.:
                    report['budget_stop'] = True
                    report['status'] = score(records)
                else:
                    proposal = origin.copy()
                    proposal[12] = theta
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        report['status'] = score(records)
                    else:
                        row['step'] = 'secant'
                        records.append(row)
                        report['status'] = score(records)
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        if full_gradient19:
            recorded = json.loads(
                (BASE / 'r2_conditional_pop3_revisit2_secant_20261008/result.json').read_text())
            if recorded.get('status') != 'CONDITIONAL_POP3_REVISIT2_REDUCED':
                raise ValueError('nineteenth full gradient requires the reduced population-3 secant')
            rows = recorded.get('evaluations') or []
            if len(rows) != 2 or rows[0].get('step') != 'verify' or rows[1].get('step') != 'secant':
                raise ValueError('nineteenth full gradient requires the population-3 secant evaluation')
            saved = rows[1]
            if float(saved['objective']) >= float(rows[0]['objective']):
                raise ValueError('population-3 secant did not improve on its verified point')
            if float(saved['population_coordinate_3']) != float(recorded['secant_population_coordinate_3']):
                raise ValueError('secant evaluation is not the proposed population-3 coordinate')
            if abs(float(saved['population_coordinate_3_gradient'])) > abs(
                    float(rows[0]['population_coordinate_3_gradient'])) / 10.:
                raise ValueError('population-3 secant did not meet its tenfold gate')
            initial_likelihood = (rows[0]['terms']['count_log_likelihood']
                                  + rows[0]['terms']['conditional_FP_log_likelihood'])
            final_likelihood = (saved['terms']['count_log_likelihood']
                                + saved['terms']['conditional_FP_log_likelihood'])
            if float(final_likelihood) < float(initial_likelihood) - 1e-6:
                raise ValueError('population-3 secant lowered the count plus conditional-FP likelihood')
            q = np.array(q_best, dtype=np.float64, copy=True)
            q[N_IC] = float(saved['tracer0'])
            q[N_IC + 2] = float(saved['tracer2'])
            q[N_IC + 4] = float(saved['tracer4'])
            q[N_IC + 9] = float(saved['population_coordinate_0'])
            q[N_IC + 11] = float(saved['population_coordinate_2'])
            q[N_IC + 12] = float(saved['population_coordinate_3'])
            q[N_IC + 18] = float(saved['population_coordinate_9'])
            q[N_IC + 21] = float(saved['population_coordinate_12'])
            q[N_IC + 23] = float(saved['population_coordinate_14'])
            report['population3_revisit2_secant_job_id'] = recorded.get('job_id')
            report['Q_LEAN'] = (
                'one full gradient at the reduced second population-3 secant, including the IC pullback '
                'and all 24 nuisance components; no further population-3 step, IC update, or heldout')
            state = evaluate_full(q)
            nuisance = np.asarray(state['gradient'][N_IC:], dtype=np.float64)
            objective_ok = abs(float(state['value']) - float(saved['objective'])) / max(
                abs(float(saved['objective'])), 1.) <= 1e-8

            def component_ok(got, expected):
                return abs(float(got) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1.)

            tracer_ok = component_ok(nuisance[0], saved['tracer0_gradient'])
            tracer2_ok = component_ok(nuisance[2], saved['tracer2_gradient'])
            tracer4_ok = component_ok(nuisance[4], saved['tracer4_gradient'])
            population0_ok = component_ok(nuisance[9], saved['population_coordinate_0_gradient'])
            population2_ok = component_ok(nuisance[11], saved['population_coordinate_2_gradient'])
            population3_ok = component_ok(nuisance[12], saved['population_coordinate_3_gradient'])
            population9_ok = component_ok(nuisance[18], saved['population_coordinate_9_gradient'])
            population12_ok = component_ok(nuisance[21], saved['population_coordinate_12_gradient'])
            population14_ok = component_ok(nuisance[23], saved['population_coordinate_14_gradient'])
            population_ok = (population0_ok and population2_ok and population3_ok
                             and population9_ok and population12_ok and population14_ok)
            report['objective'] = float(state['value'])
            report['tracer0'] = float(q[N_IC])
            report['tracer2'] = float(q[N_IC + 2])
            report['tracer4'] = float(q[N_IC + 4])
            report['population_coordinate_0'] = float(q[N_IC + 9])
            report['population_coordinate_2'] = float(q[N_IC + 11])
            report['population_coordinate_3'] = float(q[N_IC + 12])
            report['population_coordinate_9'] = float(q[N_IC + 18])
            report['population_coordinate_12'] = float(q[N_IC + 21])
            report['population_coordinate_14'] = float(q[N_IC + 23])
            report['tracer0_gradient'] = float(nuisance[0])
            report['tracer2_gradient'] = float(nuisance[2])
            report['tracer4_gradient'] = float(nuisance[4])
            report['population_coordinate_0_gradient'] = float(nuisance[9])
            report['population_coordinate_2_gradient'] = float(nuisance[11])
            report['population_coordinate_3_gradient'] = float(nuisance[12])
            report['population_coordinate_9_gradient'] = float(nuisance[18])
            report['population_coordinate_12_gradient'] = float(nuisance[21])
            report['population_coordinate_14_gradient'] = float(nuisance[23])
            report['terms'] = {key: state['detail'][key] for key in TERM_KEYS}
            report['nuisance_gradient'] = nuisance.tolist()
            if objective_ok and tracer_ok and tracer2_ok and tracer4_ok and population_ok:
                report['joint_gradient'] = block_gradient_summary(state['gradient'])
                np.savez(out / 'gradient.npz',
                         gradient=np.asarray(state['gradient'], dtype=np.float64),
                         q=np.asarray(q, dtype=np.float64))
            report['status'] = full_gradient_record_status(
                objective_ok, tracer_ok and tracer4_ok, population_ok, tracer2_ok)
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'],
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

        def evaluate_value(q):
            white = jnp.asarray(q[:N_IC])
            tracer = jnp.asarray(q[N_IC:N_IC + 9])
            population = jnp.asarray(q[N_IC + 9:])
            rho, velocity = field(white)
            packs, support_info = obs.support(rho, velocity, tracer, 2)
            score, components = obs.value(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2)
            value, detail = conditional_target_terms(q, score, components, support_info)
            return value, detail

        def compare(label, state, saved_row):
            failures = reproduction_failures(saved_row, state['detail'])
            return dict(label=label, objective=state['value'],
                        total_negative_log_target=state['detail']['total_negative_log_target'],
                        terms={key: state['detail'][key] for key in TERM_KEYS},
                        failures=failures, passed=not failures)

        if fd_only:
            report['initializer'] = dict(
                skipped=True,
                reason='job 414485 already reproduced the initializer inside the declared gate')
        else:
            if not _stage_allowed(started, ADJOINT_STAGE_SECONDS, budget_seconds):
                raise TimeoutError('application budget cannot hold the first replay')
            initial = evaluate_full(q0)
            report['initializer'] = compare('initializer', initial, saved_initial)
            _save(report_path, report, started)
            del initial
        if not _stage_allowed(started, ADJOINT_STAGE_SECONDS, budget_seconds):
            raise TimeoutError('application budget cannot hold the best-state replay')
        best = evaluate_full(q_best)
        report['best_state'] = compare('best', best, saved_best)
        reproduction_passed = report['best_state']['passed'] and (
            fd_only or report['initializer']['passed'])
        report['reproduction_passed'] = reproduction_passed
        _save(report_path, report, started)

        component_split = None
        finite_difference = None
        budget_stopped_early = False
        if reproduction_passed and fd_only:
            report['best_joint_gradient'] = block_gradient_summary(best['gradient'])
            finite_difference = None
            if _stage_allowed(started, VALUE_STAGE_SECONDS, budget_seconds):
                gradient = best['gradient']
                norm = float(np.linalg.norm(gradient))
                if not np.isfinite(norm) or norm == 0.0:
                    raise FloatingPointError('best-state gradient has no finite direction')
                direction = gradient / norm
                best.pop('pullback', None)
                best.pop('packs', None)
                best.pop('rho', None)
                best.pop('velocity', None)
                stepped_value, _ = evaluate_value(q_best + FD_EPS * direction)
                agreement = finite_difference_agreement(
                    best['value'], stepped_value, float(np.dot(gradient, direction)), FD_EPS)
                report['finite_difference'] = agreement
                finite_difference = 'passed' if agreement['passed'] else 'failed'
            report['status'] = finite_difference_status(
                reproduction_passed=True, finite_difference=finite_difference)
            report['longer_warm_start_authorized'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], reproduction_passed=True,
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if reproduction_passed:
            report['best_joint_gradient'] = block_gradient_summary(best['gradient'])
            _save(report_path, report, started)
            if not _stage_allowed(started, ADJOINT_STAGE_SECONDS, budget_seconds):
                budget_stopped_early = True
            else:
                component_gradients = {}
                component_split = 'passed'
                for index, name in COMPONENTS:
                    if not _stage_allowed(started, ADJOINT_STAGE_SECONDS):
                        budget_stopped_early = True
                        component_split = None
                        break
                    compiled = obs.component_derivative.lower(
                        best['rho'], best['velocity'], best['tracer'], best['population'],
                        best['packs'], source_jax, obs_jax, 2, index).compile()
                    analysis = compiled.memory_analysis()
                    current = jax.devices()[0].memory_stats() or {}
                    peak = (current.get('bytes_in_use', 0) + analysis.temp_size_in_bytes
                            + analysis.output_size_in_bytes)
                    limit = int(current.get('bytes_limit', 0) or 0)
                    margin = dict(component=name, estimated_peak_GiB=peak / 1024 ** 3,
                                  limit_GiB=limit / 1024 ** 3 if limit else None,
                                  has_20_percent_margin=bool(limit and 1.2 * peak <= limit))
                    report.setdefault('component_memory', []).append(margin)
                    _save(report_path, report, started)
                    if not margin['has_20_percent_margin']:
                        component_split = 'skipped'
                        del compiled
                        break
                    component_score, gradients = compiled(
                        best['rho'], best['velocity'], best['tracer'], best['population'],
                        best['packs'], source_jax, obs_jax)
                    jax.block_until_ready((component_score, gradients))
                    score_gradient = merge_component_score_gradient(best['pullback'], gradients)
                    if score_gradient.shape != q_best.shape or not np.isfinite(score_gradient).all():
                        raise FloatingPointError(f'{name} component gradient is invalid')
                    component_gradients[name] = score_gradient
                    report.setdefault('components', {})[name] = dict(
                        score=float(component_score), **block_gradient_summary(score_gradient))
                    _save(report_path, report, started)
                    del compiled, gradients, score_gradient
                if component_split == 'passed':
                    reconstructed = (q_best - component_gradients['count_2mpp']
                                     - component_gradients['conditional_FP'])
                    agreement = gradient_agreement(reconstructed, best['gradient'])
                    report['component_gradient_agreement'] = agreement
                    if not agreement['passed']:
                        component_split = 'mismatch'
                    del reconstructed
                component_gradients.clear()
            best.pop('pullback', None)
            best.pop('packs', None)
            best.pop('rho', None)
            best.pop('velocity', None)
            best.pop('tracer', None)
            best.pop('population', None)
            if component_split != 'mismatch' and not budget_stopped_early and _stage_allowed(started, VALUE_STAGE_SECONDS):
                gradient = best['gradient']
                norm = float(np.linalg.norm(gradient))
                if not np.isfinite(norm) or norm == 0.0:
                    raise FloatingPointError('best-state gradient has no finite direction')
                direction = gradient / norm
                stepped_value, _ = evaluate_value(q_best + FD_EPS * direction)
                agreement = finite_difference_agreement(
                    best['value'], stepped_value, float(np.dot(gradient, direction)), FD_EPS)
                report['finite_difference'] = agreement
                finite_difference = 'passed' if agreement['passed'] else 'failed'
            elif finite_difference is None and component_split != 'mismatch':
                budget_stopped_early = True
        report['status'] = diagnostic_status(
            reproduction_passed=reproduction_passed, component_split=component_split,
            finite_difference=finite_difference, budget_stopped_early=budget_stopped_early)
        report['longer_warm_start_authorized'] = False
        _save(report_path, report, started)
        print(json.dumps(dict(status=report['status'], reproduction_passed=reproduction_passed,
                              longer_warm_start_authorized=False), allow_nan=False), flush=True)
    except Exception as error:
        report.update(status='FAILED_CONDITIONAL_OPTIMIZER_DIAGNOSTIC', error=repr(error),
                      longer_warm_start_authorized=False)
        _save(report_path, report, started)
        raise


if __name__ == '__main__':
    main()
