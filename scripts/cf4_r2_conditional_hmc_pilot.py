"""Two fixed-step current-v6 transitions; explicitly not posterior UQ."""
import json
import numpy as np
from cf4_r2_conditional_map import BASE, N, main
from cf4_r2_prior_split_hmc import (
    FixedSplitMetric, inverse_laplacian_metric_symbol, split_hmc_step,
)


def run(objective, initial, out, report, save):
    scales = np.array([.002,.005,.005,.005,.010,.005,.030,.010,.010,
        .0005,.003,.001,.001,.003,.002,.003,.003,.003,.010,.010,.020,.003,.003,.020])
    metric = FixedSplitMetric(inverse_laplacian_metric_symbol(N), np.diag(scales**2))
    rng = np.random.default_rng(20261011)
    q = initial.copy()
    value, gradient = objective(q)
    report.update(status='CONDITIONAL_HMC_PILOT_RUNNING_NOT_POSTERIOR', optimizer=None,
        estimand='conditional v6 exact-target HMC transition diagnostic; not posterior UQ',
        Q_LEAN='two one-step matched-GL2 transitions; no optimizer, holdout access or separate simulation',
        sampler_trace=[], sampler=dict(step=.02, steps=1, proposals=2,
        metric_fixed=True, adaptation=False, acceptance_target='current v6 matched GL2'),
        posterior_sample=False, posterior_uncertainty=False)
    save()
    for index in range(2):
        before = q.copy()
        q, value, gradient, info = split_hmc_step(
            objective, metric, q, value, gradient, rng, step=.02, steps=1)
        report['sampler_trace'].append(dict(proposal=index+1, **info,
            canonical_jump_rms=float(np.sqrt(np.mean((q-before)**2))),
            objective=float(value)))
        np.savez(out/'hmc_retained_state.npz', q=q, gradient=gradient,
            objective=value, rng_state=json.dumps(rng.bit_generator.state),
            posterior_sample=False)
        save()
    report.update(status='CONDITIONAL_HMC_TRANSITION_PILOT_NOT_POSTERIOR',
                  accepted_proposals=sum(row['accepted'] for row in report['sampler_trace']))
    save()


if __name__ == '__main__':
    main(start_checkpoint=BASE/'r2_conditional_full_gradient78_20261010/gradient.npz',
         max_evaluations=3, app_seconds=3600, support_chunk_cells=16384,
         sampling_driver=run)
