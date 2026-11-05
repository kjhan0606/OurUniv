"""One prior IC: source-selected sky counts + FP marks on the same PM state.

This is a bounded compilation/gradient and actual-data support control, not
an IC fit, heldout prediction, calibrated observation model, or posterior.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_source_sky_joint import source_sky_count_fp_parts

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_source_sky_joint_control_v2'
N, BOX = 128, 384.


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    split_path = BASE/'r2_sky_closed_split_v4/split.npz'
    fp_path = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
    source_path = BASE/'r2_marked_source_geometry_v1/geometry.npz'
    with np.load(split_path, allow_pickle=False) as f:
        split = {k:f[k].copy() for k in f.files}
    with np.load(fp_path, allow_pickle=False) as f:
        if list(f['method_names']) != ['sbf','snII','snIa','tf']:
            raise ValueError('FP method ordering changed')
        labels = f['group_labels'].copy()
        fp = {k:f[k].copy() for k in f.files
              if k not in ('method_names','group_labels')}
    np.testing.assert_array_equal(labels, split['fp_source_group'])
    roles = split['fp_role']
    train_groups = np.flatnonzero(roles == 0)
    if (len(train_groups) == 0 or np.count_nonzero(roles == 1) == 0
            or np.any(~np.isin(roles, [0,1,2]))):
        raise ValueError('FP sky split does not provide train/heldout roles')
    fp['group_holdout'] = roles != 0
    fp['train_group_index'] = train_groups
    fp = {k:jnp.asarray(v) for k,v in fp.items()}
    with np.load(source_path, allow_pickle=False) as f:
        source = {k:jnp.asarray(f[k]) for k in f.files}
    mask = np.zeros(N**3, dtype=bool)
    mask[split['heldout_flat_voxels']] = True
    if (len(split['point_recno']) != 57238
            or split['train_counts'].sum()+split['heldout_counts'].sum() != 57238
            or split['heldout_counts'].sum() != 9696
            or mask.sum() != N**3//8):
        raise ValueError('frozen R2 sky/count geometry changed')
    count = {k:jnp.asarray(split[k]) for k in
             ('train_keys','train_counts','heldout_keys','heldout_counts')}
    mask = jnp.asarray(mask)
    common = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
    base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {k:base[k] for k in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
    settings.update(n=N, box_cMpc_h=BOX)
    if (settings['cosmology']['h'] != common['h']
            or settings['cosmology']['Om'] != common['Omega_m']):
        raise ValueError('PM/observation cosmology mismatch')
    evolve, _initial, conf, _cosmo, particle_mass = make_dynamics(settings)
    mass = jnp.full((N**3,), particle_mass)
    hyper = jnp.zeros(8, dtype=jnp.float64)
    group = jnp.zeros(len(train_groups), dtype=jnp.float64)
    tracer = jnp.zeros(9, dtype=jnp.float64)
    calibration_sd = jnp.asarray([.004,1.,1.,1.,1.,2.])
    ic = jnp.asarray(np.random.default_rng(2026092702).standard_normal(N**3))

    def parts_at(x):
        pos, vel = evolve(x)
        field = particle_grid(pos, vel, mass, conf)
        parts, intensity = source_sky_count_fp_parts(
            field['rho'], jnp.moveaxis(field['mean_velocity_km_s'],-1,0),
            x, hyper, group, tracer, fp, source, calibration_sd,
            count['train_keys'], count['train_counts'],
            count['heldout_keys'], count['heldout_counts'], mask,
            box=BOX, hubble=common['H0_km_s_Mpc'], h=common['h'], n=N,
            rate_parameterization='all_faint_historical')
        minimum = jnp.min(intensity.reshape(-1)[count['train_keys']])
        held_minimum = jnp.min(intensity.reshape(-1)[count['heldout_keys']])
        return parts, jnp.stack((minimum, held_minimum))

    def target_with_aux(x):
        parts, minima = parts_at(x)
        return jnp.sum(parts[:3]), (parts, minima)

    report = dict(classification='R2_SOURCE_SKY_COUNT_FP_ONE_PRIOR_IC_CONTROL',
                  status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
                  N=N, box_cMpc_h=BOX, train_points=int(split['train_counts'].sum()),
                  heldout_points=int(split['heldout_counts'].sum()),
                  FP_train_groups=int(len(train_groups)),
                  FP_sky_heldout_groups=int(np.count_nonzero(roles == 1)),
                  FP_buffer_groups=int(np.count_nonzero(roles == 2)),
                  IC_seed=2026092702, IC_is_unconditional_prior_draw=True,
                  tracer_white_coordinates_all_zero=True,
                  group_offsets_all_zero=True,
                  LF_and_group_selection_calibrated=False,
                  actual_CF4_conditioned=False,
                  heldout_prediction=False, sampler_run=False,
                  R2_posterior=False, N256=False,
                  mw_m31_m33='Latent NEW-state roles; unresolved M33 remains allowed; '
                              'no role or truth ID used in this control.',
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (split_path,fp_path,source_path)})
    OUT.mkdir(parents=True)
    def write():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write()
    try:
        t0 = time.monotonic()
        (value,(parts,minima)), gradient = jax.jit(
            jax.value_and_grad(target_with_aux,has_aux=True))(ic)
        gradient.block_until_ready()
        report['compiled_value_gradient_seconds'] = time.monotonic()-t0
        observed = np.asarray(parts)
        grad = np.asarray(gradient)
        if not np.isfinite(observed).all() or not np.isfinite(grad).all():
            raise FloatingPointError('nonfinite train/heldout factors or IC gradient')
        if np.any(np.asarray(minima) <= 0):
            raise FloatingPointError('observed sky count support failure')
        direction = np.random.default_rng(2026092703).standard_normal(N**3)
        direction /= np.linalg.norm(direction)
        reverse = float(np.dot(grad,direction))
        epsilon = 2e-5
        score = jax.jit(lambda x: target_with_aux(x)[0])
        finite = float((score(ic+epsilon*direction)
                        -score(ic-epsilon*direction))/(2*epsilon))
        relative = abs(reverse-finite)/max(1.,abs(reverse),abs(finite))
        report.update(status='GRADIENT_AND_SUPPORT_PASS_NOT_CALIBRATED'
                      if relative < .02 else 'GRADIENT_MISMATCH_NOT_CALIBRATED',
                      factors=dict(zip(('train_count','train_FP','white_prior',
                                        'untrained_heldout_count_readout'),
                                       map(float,observed))),
                      train_and_heldout_occupied_min_intensity=list(map(float,np.asarray(minima))),
                      IC_directional_reverse=reverse, IC_directional_finite_difference=finite,
                      IC_directional_relative_discrepancy=relative,
                      white_IC_gradient_l2=float(np.linalg.norm(grad)))
        if relative >= .02:
            raise AssertionError('source-sky joint IC gradient discrepancy')
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        write()
        raise
    write()
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
