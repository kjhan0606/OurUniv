"""Connect the bounded split-HMC mechanics to the unchanged live CF4 oracle.

This is an algorithm-feasibility trace, not stationary posterior samples.
No likelihood tempering, frozen candidate support, or heldout scoring.
"""
import json
from pathlib import Path
import time

import numpy as np

from cf4_r2_prior_split_hmc import (
    FixedSplitMetric, bounded_split_pilot, canonical_from_optimizer_oracle,
    inverse_laplacian_metric_symbol)


def same_target_reference(restart, report):
    parent = json.loads((Path(restart).parent/'result.json').read_text())
    sampling=parent['status']=='SPLIT_HMC_FEASIBILITY_STOP_NOT_POSTERIOR'
    if (parent['status'] not in ('PARTIAL_MAP_OPTIMIZER_STOP_NOT_POSTERIOR',
                                'SPLIT_HMC_FEASIBILITY_STOP_NOT_POSTERIOR')
            or parent['training_singletons'] != 1414
            or parent['training_counts'] != 47121
            or parent['N'] != report['N']
            or parent['box_cMpc_h'] != report['box_cMpc_h']
            or parent['count_integration'] != report['count_integration']
            or not parent['sampler_trace' if sampling else 'trace']):
        raise ValueError('sampling restart requires completed matching1414/count working state')
    reference=dict(parts=parent['final_parts'] if sampling else parent['trace'][-1]['parts'],
        objective=parent['final_objective'],
        reference_report=str(Path(restart).parent/'result.json'),sampling_parent=sampling)
    if sampling:
        last=parent['sampler_trace'][-1]
        if last['warmup'] or not 1e-6<=last['step_size']<=.3:
            raise ValueError('sampling continuation needs a finite frozen step')
        reference['frozen_step']=last['step_size']
    return reference


def run_pilot(*, initial, value, gradient, objective, score_only, terminal_field,
              metric_report, report, save_report, out, started, cap, n,
              long_trajectory=False):
    n_ic=n**3
    old=json.loads(Path(metric_report).read_text())
    hx=np.asarray(old['nuisance_optimizer_metric']['matrix'],dtype=float)
    if hx.shape != (10,10) or old['N'] != n:
        raise ValueError('saved optimizer proposal metric must have ten aligned nuisances')
    scale=np.r_[np.full(9,100.),1.]
    b=hx/scale[:,None]/scale[None,:]
    metric=FixedSplitMetric(inverse_laplacian_metric_symbol(n),b)
    step,warmup,retained,steps,seed=.1,16,16,2,2026092903
    if long_trajectory:
        reference=report['same_target_sampling_reference']
        if not reference['sampling_parent']:
            raise ValueError('long-path pilot must start from the completed split pilot')
        step,warmup,retained,steps,seed=reference['frozen_step'],0,8,8,2026092904
        with np.load(Path(reference['reference_report']).parent/'fixed_proposal_metric.npz',
                     allow_pickle=False) as previous:
            if (not np.array_equal(metric.c,previous['IC_inverse_mass']) or
                    not np.array_equal(metric.b,previous['canonical_nuisance_inverse_mass'])):
                raise ValueError('long-path comparison must retain the same proposal metric')
    np.savez(out/'fixed_proposal_metric.npz',IC_inverse_mass=metric.c,
             canonical_nuisance_inverse_mass=metric.b)
    q=initial.copy(); q[n_ic:n_ic+9]/=100.
    g=gradient.copy(); g[n_ic:n_ic+9]*=100.
    oracle=canonical_from_optimizer_oracle(objective,n_ic)
    rng=np.random.default_rng(seed)

    def optimizer_position(position):
        x=position.copy(); x[n_ic:n_ic+9]*=100.
        return x

    def endpoint_value(position):
        return score_only(optimizer_position(position))[0]

    checked=endpoint_value(q)
    if not np.isclose(checked,value,rtol=0.,atol=1e-7):
        raise AssertionError('canonical restart changed the independent full target')
    report.update(classification='PRIOR_SPLIT_HMC_FEASIBILITY_NOT_POSTERIOR',
        sampler_trace=[],sampler=dict(warmup=warmup,retained=retained,integration_steps=steps,
            initial_step=step,seed=seed,tempered=False,
            metric_source=str(metric_report),metric='fixed proposal guess, NOT covariance',
            coordinate_seam_value_difference=float(checked-value)),
        initial_IC_mean_square=float(np.mean(q[:n_ic]**2)),
        posterior_uncertainty=False,stationarity_claimed=False)

    def checkpoint(position,potential,derivative,row):
        # A rejected proposal MUST retain the old position/gradient. The live
        # oracle's latest diagnostic may belong to the rejected proposal;
        # never copy its parts into the accepted-state trace.
        np.savez(out/'accepted_checkpoint.npz',parameters=optimizer_position(position),
                 canonical_gradient=derivative,potential=potential)
        clean={k:(str(v) if isinstance(v,float) and not np.isfinite(v) else v)
               for k,v in row.items()}
        report['sampler_trace'].append(clean)
        report['rng_state']=rng.bit_generator.state
        save_report()
        print(json.dumps(dict(sampler=clean),allow_nan=False),flush=True)

    # Preserve the initial accepted state even if the first proposal fails.
    np.savez(out/'accepted_checkpoint.npz',parameters=initial,
             canonical_gradient=g,potential=value)
    save_report()
    q,value,g,trace,message=bounded_split_pilot(oracle,metric,q,value,g,rng,
        step=step,warmup=warmup,retained=retained,steps=steps,
        seconds_left=lambda:cap-(time.monotonic()-started)-180.,
        callback=checkpoint,endpoint_value=endpoint_value)
    solution=optimizer_position(q)
    final_value,parts,_,width=score_only(solution)
    if not np.isclose(final_value,value,rtol=0.,atol=1e-7):
        raise AssertionError('accepted terminal sampling state changed its target')
    rho,vel,variance,valid=terminal_field(q[:n_ic])
    np.savez(out/'final_state.npz',white_ic=q[:n_ic],tracer=q[n_ic:n_ic+9],
        white_fp_zero=q[-1],rho=np.asarray(rho),velocity_km_s=np.asarray(vel),
        physical_velocity_variance_km2_s2=np.asarray(variance),velocity_valid=np.asarray(valid))
    np.save(out/'final_canonical_gradient.npy',g)
    retained=[row for row in trace if not row['warmup']]
    report.update(status='SPLIT_HMC_FEASIBILITY_STOP_NOT_POSTERIOR',
        sampler_message=message,proposals=len(trace),
        warmup_proposals=sum(row['warmup'] for row in trace),retained_proposals=len(retained),
        retained_accepted=sum(row['accepted'] for row in retained),
        final_objective=float(value),final_parts=parts.tolist(),max_neighbors=width,
        final_canonical_gradient_inf=float(np.max(np.abs(g))),
        physical_dispersion_saved=True,posterior_velocity_uncertainty=False)
    save_report()
