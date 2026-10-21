"""Common-state metric control; never returns a selected chain endpoint."""
import copy
import json
import os
import time
import numpy as np
from scipy.fft import fftn
from cf4_r2_prior_split_hmc import split_hmc_step


def paired_control(oracle, metrics, q, value, gradient, rng, *, step, steps,
                   on_arm=None):
    """Same standard Gaussian and acceptance uniform; caller owns budget.

    Each arm starts from q and clones the supplied diagnostic RNG. The caller's
    RNG advances once, as one transition. The original chain RNG is not passed.
    """
    if set(metrics) != {'A','B'}:
        raise ValueError('two predeclared metric arms A/B required')
    n = metrics['A'].c.shape[0]
    if (metrics['B'].c.shape != metrics['A'].c.shape
            or not np.array_equal(metrics['A'].b, metrics['B'].b)):
        raise ValueError('aligned IC grids and unchanged nuisance metric required')
    k = np.fft.fftfreq(n)*n
    k2 = k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2
    mask = (k2>16)&(k2<=64)
    def displacement(proposed, before):
        delta = (proposed-before)[:n**3].reshape((n,)*3)
        spectral = fftn(delta, norm='ortho', workers=1)
        return {'band_squared_jump':float(np.sum(np.abs(spectral[mask])**2))}
    original = copy.deepcopy(rng.bit_generator.state)
    rows = {}
    terminal_rng = None
    for arm in ('A','B'):
        clone = copy.deepcopy(rng)
        clone.bit_generator.state = copy.deepcopy(original)
        retained, _, _, info = split_hmc_step(
            oracle, metrics[arm], q, value, gradient, clone,
            step=step, steps=steps, proposal_readout=displacement)
        proposed = info['proposal_readout']['band_squared_jump']
        alpha = float(np.exp(info['log_acceptance']))
        rows[arm] = dict(**info, acceptance_probability=alpha,
            proposed_band_squared_jump_per_force=proposed/steps,
            weighted_band_squared_jump_per_force=alpha*proposed/steps,
            retained_band_squared_jump_per_force=
                displacement(retained,q)['band_squared_jump']/steps,
            discard_for_inference=True)
        # In-support arms consume identical counts of Gaussian and uniform draws.
        if terminal_rng is not None and clone.bit_generator.state != terminal_rng:
            raise ValueError('paired arms consumed inconsistent random streams')
        terminal_rng = copy.deepcopy(clone.bit_generator.state)
        if on_arm is not None:
            on_arm(arm, rows[arm])
    rng.bit_generator.state = terminal_rng
    return rows


def replay_diagnostic_rng_pairs(rng, dimension, completed_pairs):
    """Advance a diagnostic RNG over completed pairs and verify their MH draws.

    Each paired transition consumes one standard-normal momentum vector and
    one shared acceptance uniform: both metric arms clone the same RNG state.
    This reconstructs the stream without repeating any target evaluations.
    """
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError('positive momentum dimension required')
    uniforms = []
    for expected_pair, row in enumerate(completed_pairs):
        if row.get('pair') != expected_pair or row.get('complete') is not True:
            raise ValueError('prior pairs must be a complete ordered prefix')
        arms = row.get('arms', {})
        if set(arms) != {'A', 'B'}:
            raise ValueError('each prior pair must contain both arms')
        rng.standard_normal(dimension)
        uniform = float(rng.uniform())
        for arm, result in arms.items():
            accepted = bool(np.log(uniform) < float(result['log_acceptance']))
            if accepted != bool(result['accepted']):
                raise ValueError(
                    f'diagnostic RNG replay disagrees with pair {expected_pair} arm {arm}')
        uniforms.append(uniform)
    return uniforms


def run(objective, initial, out, report, save):
    from cf4_r2_conditional_map import N
    from cf4_r2_hmc_transport import SCALES
    from cf4_r2_prior_split_hmc import (
        FixedSplitMetric, inverse_laplacian_metric_symbol, band_scaled_inverse_mass,
    )
    c = inverse_laplacian_metric_symbol(N)
    metrics = {name:FixedSplitMetric(symbol, np.diag(SCALES**2))
               for name,symbol in [('A',c),('B',band_scaled_inverse_mass(c))]}
    value, gradient = objective(initial)
    initial_field = objective.records[-1].get('present_field_readout')
    seed = 2026102001
    rng = np.random.default_rng(seed)
    start_pair = int(os.environ.get('CF4_BAND_CONTROL_START_PAIR', '0'))
    expected_pair_uniform = None
    if start_pair not in (0, 3):
        raise ValueError('only a fresh run or the predeclared pair-3 continuation is supported')
    report.update(status='CONDITIONAL_BAND_CONTROL_RUNNING_NOT_POSTERIOR',
        optimizer=None, paired_results=[], diagnostic_seed=seed,
        original_chain_rng_consumed=False, selected_endpoint=False,
        discard_for_inference=True, posterior_sample=False,
        posterior_uncertainty=False, control_steps=8,
        control_step=.11996093316338646,
        control_metric='B=4*A only for16<integer_k_squared<=64',
        engineering_rule='weighted pooled B>=2*A and B>A in at least3/4 complete pairs')
    if start_pair == 3:
        prior_path = os.environ.get('CF4_BAND_CONTROL_PRIOR_RESULT')
        if not prior_path:
            raise ValueError('pair-3 continuation requires the prior result artifact')
        with open(prior_path, encoding='utf-8') as stream:
            prior = json.load(stream)
        prior_pairs = prior.get('paired_results', [])
        expected_checkpoint = report.get('start_checkpoint')
        if not (prior.get('source_commit') == '666edbdff5702f7d525efec6461b7fe1cf3049a7'
                and prior.get('status') == 'CONDITIONAL_BAND_CONTROL_DIAGNOSTIC_NOT_POSTERIOR'
                and prior.get('stop_reason') == 'insufficient complete-pair budget'
                and prior.get('complete_pairs') == 3
                and [row.get('pair') for row in prior_pairs] == [0, 1, 2]
                and all(row.get('complete') is True for row in prior_pairs)
                and prior.get('diagnostic_seed') == seed
                and prior.get('N') == N
                and prior.get('box_cMpc_h') == 384.0
                and prior.get('control_metric') == report['control_metric']
                and prior.get('control_step') == .11996093316338646
                and prior.get('control_steps') == 8
                and prior.get('start_checkpoint') == expected_checkpoint
                and prior.get('posterior_sample') is False
                and prior.get('discard_for_inference') is True):
            raise ValueError('prior artifact does not match the frozen three-pair control')
        uniforms = replay_diagnostic_rng_pairs(rng, metrics['A'].size, prior_pairs)
        probe_rng = copy.deepcopy(rng)
        probe_rng.standard_normal(metrics['A'].size)
        expected_pair_uniform = float(probe_rng.uniform())
        report['paired_results'] = copy.deepcopy(prior_pairs)
        report['pair3_continuation'] = dict(
            prior_result_path=prior_path,
            prior_source_commit=prior['source_commit'],
            prior_exact_target_evaluations=prior['exact_target_evaluations'],
            prior_pair_ids=[0, 1, 2],
            reconstructed_acceptance_uniforms=uniforms,
            expected_pair3_acceptance_uniform=expected_pair_uniform,
            all_prior_arm_decisions_verified=True,
            repeated_prior_target_evaluations=0,
        )
        report['Q_LEAN'] = (
            'complete only the one predeclared missing matched pair; no factor, '
            'step-size or trajectory-length sweep')
    elif os.environ.get('CF4_BAND_CONTROL_PRIOR_RESULT'):
        raise ValueError('prior result path is only valid for pair-3 continuation')
    save()
    for pair in range(start_pair, 4):
        if (objective.max_evaluations-len(objective.records) < 16
                or objective.deadline-time.monotonic() < 16*500+120):
            report['stop_reason'] = 'insufficient complete-pair budget'
            break
        row = dict(pair=pair, arms={}, complete=False)
        report['paired_results'].append(row)
        def on_arm(arm, info):
            info = dict(info)
            info['proposed_present_field_readout'] = objective.records[-1].get('present_field_readout')
            info['retained_present_field_readout'] = (
                info['proposed_present_field_readout'] if info['accepted'] else initial_field)
            row['arms'][arm] = info
            save()
        paired_control(objective,metrics,initial,value,gradient,rng,
                       step=.11996093316338646,steps=8,on_arm=on_arm)
        if pair == 3 and expected_pair_uniform is not None:
            for arm, info in row['arms'].items():
                accepted = bool(np.log(expected_pair_uniform)
                                < float(info['log_acceptance']))
                if accepted != bool(info['accepted']):
                    raise ValueError(f'pair-3 RNG replay disagrees with arm {arm}')
            report['pair3_continuation']['pair3_arm_decisions_verified'] = True
        row['complete'] = True
        save()
    completed = [r for r in report['paired_results'] if r['complete']]
    report['complete_pairs'] = len(completed)
    if len(completed)==4:
        scores = {a:sum(r['arms'][a]['weighted_band_squared_jump_per_force']
                        for r in completed) for a in ('A','B')}
        wins = sum(r['arms']['B']['weighted_band_squared_jump_per_force'] >
                   r['arms']['A']['weighted_band_squared_jump_per_force'] for r in completed)
        report['weighted_scores'] = scores
        report['B_pair_wins'] = wins
        report['engineering_gain_rule_passed'] = (
            scores['A']>0 and scores['B']>=2*scores['A'] and wins>=3)
        if 'pair3_continuation' in report:
            report['pair3_continuation']['cumulative_exact_target_evaluations_across_jobs'] = (
                report['pair3_continuation']['prior_exact_target_evaluations']
                + len(objective.records))
    report['status'] = 'CONDITIONAL_BAND_CONTROL_DIAGNOSTIC_NOT_POSTERIOR'
    save()


if __name__ == '__main__':
    from cf4_r2_conditional_map import BASE, main
    pair3 = os.environ.get('CF4_BAND_CONTROL_START_PAIR') == '3'
    main(start_checkpoint=BASE/'r2_hmc_equil24h_20261013/hmc_retained_state.npz',
         max_evaluations=17 if pair3 else 65,
         app_seconds=9900 if pair3 else 27000, support_chunk_cells=16384,
         sampling_driver=run)
