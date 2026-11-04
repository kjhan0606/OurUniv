"""One bounded sequential transport bundle; not a posterior delivery."""
import json
import os
import time
from functools import lru_cache
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


@lru_cache(maxsize=4)
def hmc_drift_mode_indices(n):
    """Full-lattice coordinates for the six fundamental and fixed HMC shell modes."""
    if not isinstance(n, int) or n < 18:
        raise ValueError('the HMC drift shell requires n>=18 so |k|<=8 is below Nyquist')
    fundamental = np.asarray(
        [(kx % n, ky % n, kz % n)
         for kx, ky, kz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))],
        dtype=np.intp)
    drift = np.asarray(
        [(kx % n, ky % n, kz % n)
         for kx in range(-8, 9)
         for ky in range(-8, 9)
         for kz in range(-8, 9)
         if 16 < kx*kx + ky*ky + kz*kz <= 64],
        dtype=np.intp)
    if len(fundamental) != 6 or len(drift) != 1852:
        raise AssertionError('the fixed fundamental/HMC-shell real-mode counts changed')
    return fundamental, drift


def retained_block_jump_readout(delta, n):
    """Measure post-Metropolis retained motion in fixed canonical-coordinate blocks."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (n**3 + len(SCALES),) or not np.isfinite(delta).all():
        raise ValueError('finite retained displacement must contain N^3 IC and 24 nuisance values')
    fundamental, drift_shell = hmc_drift_mode_indices(n)
    spectrum = fftn(delta[:n**3].reshape((n,)*3), norm='ortho', workers=1)
    fundamental_energy = float(np.sum(
        np.abs(spectrum[fundamental[:,0], fundamental[:,1], fundamental[:,2]])**2,
        dtype=np.float64))
    drift_energy = float(np.sum(
        np.abs(spectrum[drift_shell[:,0], drift_shell[:,1], drift_shell[:,2]])**2,
        dtype=np.float64))
    total_ic_energy = float(np.dot(delta[:n**3], delta[:n**3]))
    remaining_energy = total_ic_energy - fundamental_energy - drift_energy
    if remaining_energy < -1e-10 * max(total_ic_energy, 1.0):
        raise FloatingPointError('Fourier block energies exceed the retained IC displacement')
    remaining_energy = max(remaining_energy, 0.0)
    remaining_modes = n**3 - len(fundamental) - len(drift_shell)
    nuisance = delta[n**3:]
    return dict(
        fundamental_mode_count=len(fundamental),
        fundamental_jump_rms=float(np.sqrt(fundamental_energy / len(fundamental))),
        hmc_drift_shell_mode_count=len(drift_shell),
        hmc_drift_shell_jump_rms=float(np.sqrt(drift_energy / len(drift_shell))),
        remaining_ic_mode_count=remaining_modes,
        remaining_ic_jump_rms=float(np.sqrt(remaining_energy / remaining_modes)),
        canonical_nuisance_mode_count=len(nuisance),
        canonical_nuisance_jump_rms=float(np.sqrt(np.mean(nuisance*nuisance))),
    ), spectrum


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
    current_field = objective.records[-1].get('present_field_readout')
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
        retained_block_jump_readout='post-Metropolis q_new-q_old; fundamental k^2=1, exact HMC shell 16<k^2<=64, remaining IC, and 24 nuisance coordinates',
        transition_evaluation_count_check='each row must match objective-record growth to force_evaluations',
        Q_LEAN='one sequential bounded warmup/transport bundle; no holdout values or separate simulation')
    def checkpoint(index):
        np.savez(out/'hmc_retained_state.npz', q=q, gradient=gradient, objective=value,
            rng_state=json.dumps(rng.bit_generator.state), next_index=index, step=step,
            fingerprint=json.dumps(fingerprint), current_terms=json.dumps(current_terms),
            phase='warmup' if index<6 else 'fixed_diagnostic', posterior_sample=False,
            discard_for_inference=True)
        report['current_state_terms'] = current_terms.copy()
        report['current_present_field_readout'] = current_field
        save()
    checkpoint(next_index)
    end_index = int(os.environ.get('CF4_HMC_END_INDEX', '10'))
    lengths_text = os.environ.get('CF4_HMC_LENGTHS', '4,6')
    fixed_lengths = tuple(int(item) for item in lengths_text.split(','))
    if (not fixed_lengths or any(item < 1 or item > 12 for item in fixed_lengths)
            or len(set(fixed_lengths)) != len(fixed_lengths)):
        raise ValueError('distinct trajectory lengths in 1..12 required')
    report['fixed_trajectory_lengths'] = list(fixed_lengths)
    report['retained_block_jump_coverage_start_index'] = next_index
    report['retained_block_jump_coverage_end_index_exclusive'] = end_index
    report['trajectory_segment'] = dict(start_index=next_index,
        requested_end_index=end_index, start_checkpoint=str(START),
        fixed_lengths=list(fixed_lengths), policy='state-independent uniform mixture')
    if end_index < next_index:
        raise ValueError('transport end index precedes retained state')
    for index in range(next_index, end_index):
        before_rng = rng.bit_generator.state
        lengths = [2,2,4,4,6,6]
        steps = lengths[index] if index<6 else int(rng.choice(fixed_lengths))
        # Reserve complete-trajectory time using conservative measured-force cost.
        if (objective.max_evaluations-len(objective.records) < steps
                or objective.deadline-time.monotonic() < steps*500.+120.):
            rng.bit_generator.state = before_rng
            report['stop_reason'] = 'insufficient complete-trajectory budget'
            break
        before = q.copy()
        evaluation_count_before = len(objective.records)
        started = time.monotonic()
        try:
            new_q, new_value, new_gradient, info = split_hmc_step(
                objective, metric, q, value, gradient, rng, step=step, steps=steps)
        except EvaluationBudgetStop:
            rng.bit_generator.state = before_rng
            report['stop_reason'] = 'budget interrupted trajectory; endpoint discarded'
            break
        q, value, gradient = new_q, new_value, new_gradient
        transition_evaluations = len(objective.records) - evaluation_count_before
        if transition_evaluations != int(info['force_evaluations']):
            raise RuntimeError('retained HMC transition target calls disagree with force_evaluations')
        if info['accepted']:
            current_terms = {k: objective.records[-1][k] for k in TERM_KEYS}
            current_field = objective.records[-1].get('present_field_readout')
        delta = q-before
        block_jumps, spectral_delta = retained_block_jump_readout(delta, N)
        proposal_distance = (np.sum(np.abs(spectral_delta)**2/metric.c)
                             + np.sum((delta[N**3:]/SCALES)**2))
        row = dict(index=index, warmup=index<6, discard_for_inference=True, steps=steps, step=step, **info,
            **movement(q,before,N), **mode_readout(q,N), **block_jumps,
            target_evaluations=transition_evaluations,
            target_evaluation_count_matches_transition=True, objective=value,
            current_terms=current_terms.copy(), acceptance_probability=float(np.exp(info['log_acceptance'])),
            present_field_readout=current_field,
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
         support_chunk_cells=int(os.environ.get('CF4_HMC_SUPPORT_CHUNK_CELLS', '16384')),
         sampling_driver=run)
