"""Train-only selected-FP radial-prior sensitivity on one saved PM state.

This diagnoses whether an overlap-free singleton mark subset can reduce the
unjustified group-selection dependence. It is not a fit or heldout prediction.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_fp_distance import fp_log_likelihood_ratio
from cf4_r2_live_field_marks import bind_state_geometry

BASE = Path('/gpfs/kjhan/CF4/z0_density')
SPLIT = BASE/'r2_sky_closed_split_v5/split.npz'
GEOMETRY = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
STATE = BASE/'r2_live_field_pilot_v1/initial_chain0.npz'
OUT = BASE/'r2_fp_singleton_prior_sensitivity_v1'


def summaries(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(total=float(values.sum()), median=float(np.median(values)),
                p90=float(np.percentile(values,90)),
                p99=float(np.percentile(values,99)))


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(SPLIT, allow_pickle=False) as f:
        labels = f['fp_source_group'].copy()
        roles = f['fp_role'].copy()
    with np.load(GEOMETRY, allow_pickle=False) as f:
        np.testing.assert_array_equal(labels, f['group_labels'])
        row_group = f['row_group'].copy()
        row_count = np.bincount(row_group, minlength=len(labels))
        anchor_count = np.bincount(f['anchor_group'], minlength=len(labels))
        no_count_member = f['twompp_member_count'] == 0
        selected = ((roles == 0) & no_count_member
                    & (row_count == 1) & (anchor_count == 0))
        groups = np.flatnonzero(selected)
        if len(groups) != 3535:
            raise ValueError('frozen overlap-free FP train support changed')
        # Every selected group has precisely one FP source row. No anchor or
        # counted member is included in this diagnostic factor.
        group_to_row = np.full(len(labels), -1, dtype=np.int64)
        group_to_row[row_group] = np.arange(len(row_group))
        rows = group_to_row[groups]
        if np.any(rows < 0) or np.any(row_group[rows] != groups):
            raise ValueError('singleton row mapping changed')
        geometry = {k:jnp.asarray(f[k][groups]) for k in
                    ('distance','quadrature_weight','directions','zcos',
                     'redshift_sufficient')}
        dz = jnp.asarray(f['dz_row'][rows])
        mean = jnp.asarray(f['eta_mean'][rows])
        std = jnp.asarray(f['eta_std'][rows])
        alpha = jnp.asarray(f['eta_alpha'][rows])
    with np.load(STATE, allow_pickle=False) as f:
        if float(f['PM_mesh_origin_fraction']) != 0. or float(f['box_cMpc_h']) != 384.:
            raise ValueError('saved PM coordinate contract changed')
        rho = jnp.asarray(f['rho'])
        velocity = jnp.asarray(f['mean_velocity_km_s'])

    def scores(rho_now, velocity_now):
        bound = bind_state_geometry(rho_now, velocity_now, geometry,
                                    box=384., selected_bias=1.)
        d = geometry['distance']
        logq = jnp.log(geometry['quadrature_weight'])
        logz = bound['redshift_logkernel']
        mark = fp_log_likelihood_ratio(
            jnp.log10(dz[:,None]/d), 0.,
            mean[:,None], std[:,None], alpha[:,None])
        bases = (bound['log_distance_weight']+logz,
                 logq+2.*jnp.log(d)+logz,
                 logq-jnp.log(d)+logz)
        return jnp.stack([logsumexp(base+mark,axis=1)
                          -logsumexp(base,axis=1) for base in bases])

    result = np.asarray(jax.jit(scores)(rho,velocity))
    if result.shape != (3,len(groups)) or not np.isfinite(result).all():
        raise FloatingPointError('training singleton prior sensitivity not finite')
    model_names = ('d2_rho','d2_only','log_distance_flat')
    report = dict(classification='R2_TRAIN_ONLY_FP_SINGLETON_PRIOR_SENSITIVITY',
                  status='COMPLETED_NOT_CALIBRATED', job_id=os.environ['SLURM_JOB_ID'],
                  graph_closed_split='v5', training_groups=int(len(groups)),
                  heldout_marks_read=False, count_likelihood_used=False,
                  source_mock_used=False, IC_fitted=False, sampler_run=False,
                  field_state='archived unconditional prior IC; PM native origin0',
                  radial_models=list(model_names),
                  score={name:summaries(result[i]) for i,name in enumerate(model_names)},
                  absolute_difference={
                      name:summaries(np.abs(result[0]-result[i]))
                      for i,name in enumerate(model_names[1:],start=1)},
                  signed_total_difference={
                      name:float(np.sum(result[0]-result[i]))
                      for i,name in enumerate(model_names[1:],start=1)},
                  interpretation='One-state training-only sensitivity of the conditional '
                                 'FP mark factor. Neither a Bayes factor, selected-group '
                                 'calibration, heldout prediction nor a posterior.',
                  MW_M31_M33='Latent roles from each NEW field; M33 may remain '
                             'unresolved; no native truth IDs used.',
                  R2_posterior=False, N256=False,
                  host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (SPLIT,GEOMETRY,STATE)})
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
