"""One bounded sequential transport bundle; not a posterior delivery."""
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.fft import fftn
from cf4_r2_conditional_map import BASE, N, TERM_KEYS, EvaluationBudgetStop, main
from cf4_r2_conditional_hmc_pilot import movement
from cf4_r2_prior_split_hmc import FixedSplitMetric, inverse_laplacian_metric_symbol, split_hmc_step

START = Path(os.environ.get('CF4_HMC_START',
    str(BASE/'r2_conditional_hmc_step010_20261011/hmc_retained_state.npz')))
SCALES = np.array([.002,.005,.005,.005,.010,.005,.030,.010,.010,
    .0005,.003,.001,.001,.003,.002,.003,.003,.003,.010,.010,.020,.003,.003,.020])


def mode_readout(q, n):
    cube = q[:n**3].reshape((n,)*3)
    wave = np.sin(2*np.pi*np.arange(n)/n)
    norm = np.sqrt(n**2*np.sum(wave**2))
    sine = [float(cube.sum(axis=tuple(j for j in range(3) if j!=i)) @ wave/norm)
            for i in range(3)]
    k = np.fft.fftfreq(n)*n
    k2 = k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2
    power = np.abs(fftn(cube, norm='ortho', workers=1))**2
    bands = {}
    for lo, hi in ((0,2),(2,4),(4,8)):
        mask = (k2>lo**2)&(k2<=hi**2)
        bands[f'{lo}<integer_k<={hi}'] = float(power[mask].mean()) if mask.any() else None
    return dict(fundamental_sine=sine, white_IC_band_power=bands)


def run(objective, initial, out, report, save):
    resume_metadata = None
    with np.load(START, allow_pickle=False) as saved:
        if not np.array_equal(initial, saved['q']):
            raise ValueError('resume coordinates differ from retained terminal state')
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(saved['rng_state']))
        if 'fingerprint' in saved.files:
            resume_metadata = dict(fingerprint=json.loads(str(saved['fingerprint'])),
                next_index=int(saved['next_index']), step=float(saved['step']),
                phase=str(saved['phase']))
    q = initial.copy()
    value, gradient = objective(q)
    current_terms = {k: objective.records[-1][k] for k in TERM_KEYS}
    metric = FixedSplitMetric(inverse_laplacian_metric_symbol(N), np.diag(SCALES**2))
    fingerprint = dict(N=N, box=384., target_rules=report['target_rules'],
        target_orders=report['target_orders'], split=report['split_source'],
        association=report['association_ledger_source_commit'],
        metric=dict(fundamental_mass=6000., nuisance_inverse_mass=SCALES**2))
    fingerprint['metric']['nuisance_inverse_mass'] = (SCALES**2).tolist()
    step = .1
    trace = []
    next_index = 0
    if resume_metadata is not None:
        if resume_metadata['fingerprint'] != fingerprint:
            raise ValueError('transport target/metric fingerprint changed')
        next_index = resume_metadata['next_index']
        step = resume_metadata['step']
        previous = json.loads((START.parent/'result.json').read_text())
        trace = previous['sampler_trace'].copy()
        if (next_index != len(trace) or not np.isfinite(step) or not .01 <= step <= .15
                or resume_metadata['phase'] != ('warmup' if next_index<6 else 'fixed_diagnostic')):
            raise ValueError('transport resume schedule/phase is inconsistent')
    report.update(status='CONDITIONAL_HMC_TRANSPORT_RUNNING_NOT_POSTERIOR',
        optimizer=None, sampler_trace=trace, posterior_sample=False,
        posterior_uncertainty=False, discard_for_inference=True, sampler_fingerprint=fingerprint,
        Q_LEAN='one sequential bounded warmup/transport bundle; no holdout values or separate simulation')
    def checkpoint(index):
        np.savez(out/'hmc_retained_state.npz', q=q, gradient=gradient, objective=value,
            rng_state=json.dumps(rng.bit_generator.state), next_index=index, step=step,
            fingerprint=json.dumps(fingerprint), current_terms=json.dumps(current_terms),
            phase='warmup' if index<6 else 'fixed_diagnostic', posterior_sample=False,
            discard_for_inference=True)
        report['current_state_terms'] = current_terms.copy()
        save()
    checkpoint(next_index)
    end_index = int(os.environ.get('CF4_HMC_END_INDEX', '10'))
    if end_index < next_index:
        raise ValueError('transport end index precedes retained state')
    for index in range(next_index, end_index):
        before_rng = rng.bit_generator.state
        lengths = [2,2,4,4,6,6]
        steps = lengths[index] if index<6 else int(rng.choice([4,6]))
        # Reserve complete-trajectory time using conservative measured-force cost.
        if (objective.max_evaluations-len(objective.records) < steps
                or objective.deadline-time.monotonic() < steps*500.+120.):
            rng.bit_generator.state = before_rng
            report['stop_reason'] = 'insufficient complete-trajectory budget'
            break
        before = q.copy()
        started = time.monotonic()
        try:
            new_q, new_value, new_gradient, info = split_hmc_step(
                objective, metric, q, value, gradient, rng, step=step, steps=steps)
        except EvaluationBudgetStop:
            rng.bit_generator.state = before_rng
            report['stop_reason'] = 'budget interrupted trajectory; endpoint discarded'
            break
        q, value, gradient = new_q, new_value, new_gradient
        if info['accepted']:
            current_terms = {k: objective.records[-1][k] for k in TERM_KEYS}
        delta = q-before
        spectral_delta = fftn(delta[:N**3].reshape((N,)*3), norm='ortho', workers=1)
        proposal_distance = (np.sum(np.abs(spectral_delta)**2/metric.c)
                             + np.sum((delta[N**3:]/SCALES)**2))
        row = dict(index=index, warmup=index<6, discard_for_inference=True, steps=steps, step=step, **info,
            **movement(q,before,N), **mode_readout(q,N), objective=value,
            current_terms=current_terms.copy(), acceptance_probability=float(np.exp(info['log_acceptance'])),
            seconds=time.monotonic()-started,
            canonical_jump_rms=float(np.sqrt(np.mean(delta*delta))),
            proposal_metric_jump_rms=float(np.sqrt(proposal_distance/(N**3+24))))
        trace.append(row)
        if index<6:
            step=float(np.clip(step*np.exp(.25*(np.exp(info['log_acceptance'])-.8)
                                          /np.sqrt(index+1)),.01,.15))
        checkpoint(index+1)
    report.update(status='CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR',
        warmup_completed=sum(r['warmup'] for r in trace),
        diagnostic_retained=sum(not r['warmup'] for r in trace))
    checkpoint(len(trace))


if __name__ == '__main__':
    main(start_checkpoint=START,
         max_evaluations=int(os.environ.get('CF4_HMC_MAX_EVALUATIONS', '49')),
         app_seconds=int(os.environ.get('CF4_HMC_APP_SECONDS', '19800')),
         support_chunk_cells=16384, sampling_driver=run)
