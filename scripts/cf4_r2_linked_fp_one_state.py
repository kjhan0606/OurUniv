"""One predeclared train-only linked FP mark on an archived N128 PM state.

This is an actual-source numeric ownership control, NOT a group-selection
calibration, heldout prediction, field fit, or R2 posterior.
"""

import csv
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
from cf4_r2_fp_distance import fp_log_likelihood_ratio
from cf4_r2_marked_tracer_jax import (
    conditional_single_link_logfactor, intrinsic_biased_source_masses,
    intrinsic_lf_bin_fractions, predict_source_marked_intensity,
    predict_source_marked_key_contributions,
    predict_source_marked_radial_key_density,
)
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_linked_fp_one_state_v1'
SPLIT = BASE/'r2_sky_closed_split_v5/split.npz'
POINTS = BASE/'r2_point_mark_manifest_v1/points.npz'
FP = BASE/'r2_source_observation_assembly_v1/observations.npz'
GROUP = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
SOURCE = BASE/'r2_marked_source_geometry_v1/geometry.npz'
STATE = BASE/'r2_pm128_unconditional_v1/state.npz'
CROSSMATCH = ROOT/'data/cf4_2mpp_crossmatch_v1.csv'
N, BOX = 128, 384.


def selected_train_link():
    with np.load(SPLIT, allow_pickle=False) as f:
        recno = f['point_recno'].copy()
        labels = f['fp_source_group'].copy()
        roles = f['fp_role'].copy()
    with np.load(POINTS, allow_pickle=False) as f:
        point = {k:f[k].copy() for k in ('population','flat_cell',
                                         'radius_cMpc_h')}
    with np.load(FP, allow_pickle=False) as f:
        pgc = f['PGC'].copy()
        source_group = f['source_group'].copy()
    with np.load(GROUP, allow_pickle=False) as f:
        np.testing.assert_array_equal(labels, f['group_labels'])
        row_group = f['row_group'].copy()
        rows = np.bincount(row_group, minlength=len(labels))
        anchors = np.bincount(f['anchor_group'], minlength=len(labels))
        fp = {k:f[k].copy() for k in ('dz_row','eta_mean','eta_std',
                                      'eta_alpha')}
    rec_index = {int(v):i for i,v in enumerate(recno)}
    label_index = {str(v):i for i,v in enumerate(labels)}
    pgc_group = {int(p):label_index[str(s)] for p,s in zip(pgc,source_group)}
    links = {}
    with CROSSMATCH.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['match_class'] != 'secure_joint_mark' or not row['twompp_recno']:
                continue
            i = rec_index.get(int(row['twompp_recno']))
            g = pgc_group.get(int(row['PGC']))
            if i is not None and g is not None:
                links.setdefault(g,set()).add(i)
    options = sorted((str(labels[g]),g,next(iter(indices)))
                     for g,indices in links.items()
                     if roles[g] == 0 and len(indices) == 1 and rows[g] == 1
                     and anchors[g] == 0
                     and 30 <= point['radius_cMpc_h'][next(iter(indices))] <= 120)
    if len(options) != 1002 or options[0] != ('P1085367',0,26507):
        raise ValueError('frozen geometry-only one-link choice changed')
    label,g,i = options[0]
    row = int(np.flatnonzero(row_group == g)[0])
    return label,g,i,row,point,fp


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    label,g,i,row,point,fp = selected_train_link()
    with np.load(STATE,allow_pickle=False) as f:
        rho = jnp.asarray(f['rho'],dtype=jnp.float64)
        vel = jnp.asarray(f['velocity_km_s'],dtype=jnp.float64)
    rho_cell,vel_cell = native_mass_momentum_to_count_cells(rho,vel,BOX)
    with np.load(SOURCE,allow_pickle=False) as f:
        source = {k:jnp.asarray(f[k]) for k in f.files}
    positions = source['positions']
    angular = source['angular']
    source_velocity = jnp.moveaxis(vel_cell,0,-1).reshape(-1,3)
    fraction = jnp.sum(intrinsic_lf_bin_fractions()[1:4])
    intrinsic = intrinsic_biased_source_masses(
        rho_cell,jnp.log(fraction),jnp.ones(5),
        reference_interval=(-25.,-21.))
    pop = int(point['population'][i])
    voxel = tuple(int(v) for v in np.unravel_index(
        int(point['flat_cell'][i]),(N,)*3))
    radius = float(point['radius_cMpc_h'][i])
    cosmology = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
    args = dict(observer=jnp.full(3,BOX/2.),box_size_cMpc_h=BOX,
                hubble_km_s_Mpc=cosmology['H0_km_s_Mpc'],
                little_h=cosmology['h'],
                radius_table_cMpc_h=source['radial_table'],
                modulus_table_h=source['modulus_table'],
                redshift_table=source['redshift_table'],grid_size=N,
                sigma_los_km_s=100.,radial_min_cMpc_h=5.,
                radial_max_cMpc_h=180.)
    report = dict(classification='R2_TRAIN_ONLY_ONE_LINKED_FP_SOURCE_CONTROL',
                  status='STARTED',job_id=os.environ['SLURM_JOB_ID'],
                  source_group=label,group_index=g,point_index=i,FP_row=row,
                  count_population=pop,count_voxel=list(voxel),
                  point_observed_radius_cMpc_h=radius,
                  actual_CF4_conditioned=False,association_calibrated=False,
                  group_selection_calibrated=False,FP_group_covariance_calibrated=False,
                  heldout_marks_used=False,field_fit=False,sampler=False,
                  R2_posterior=False,N256=False,
                  MW_M31_M33='NEW-field latent roles unresolved; no truth identity selected',
                  input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (SPLIT,POINTS,FP,GROUP,SOURCE,STATE,CROSSMATCH)})
    OUT.mkdir(parents=True)
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        t=time.monotonic()
        intensity = jax.jit(predict_source_marked_intensity,
            static_argnames=('grid_size','quadrature_order',
                             'radial_min_cMpc_h','radial_max_cMpc_h'))(
            positions,source_velocity,intrinsic,angular,**args)
        count_mean = float(intensity[(pop,)+voxel])
        key_mass = jax.jit(predict_source_marked_key_contributions,
            static_argnames=('population','grid_size','quadrature_order',
                             'radial_min_cMpc_h','radial_max_cMpc_h'))(
            positions,source_velocity,intrinsic,angular,pop,
            jnp.asarray(voxel),**args)
        count_parts = float(jnp.sum(key_mass))
        report['count_and_key_seconds']=time.monotonic()-t
        if count_mean <= 0 or abs(count_parts-count_mean)>1e-8*count_mean:
            raise AssertionError('source-key decomposition disagrees with full count')
        relative = (positions-jnp.full(3,BOX/2.)+BOX/2.) % BOX-BOX/2.
        true_distance = jnp.linalg.norm(relative,axis=1)
        eta = jnp.log10(float(fp['dz_row'][row])/true_distance)
        log_fp = fp_log_likelihood_ratio(
            eta,0.,float(fp['eta_mean'][row]),
            float(fp['eta_std'][row]),float(fp['eta_alpha'][row]))
        fp_by_source = jnp.broadcast_to(log_fp[None,:],intrinsic.shape)
        def score_at_velocity_scale(scale):
            density = predict_source_marked_radial_key_density(
                positions,scale*source_velocity,intrinsic,angular,pop,
                jnp.asarray(voxel),
                radius,**args)
            return (conditional_single_link_logfactor(
                density,jnp.zeros_like(density),fp_by_source),
                jnp.sum(density))
        t=time.monotonic()
        score, derivative = jax.jit(jax.value_and_grad(
            lambda scale: score_at_velocity_scale(scale)[0]))(jnp.asarray(1.))
        plain = jax.jit(score_at_velocity_scale)
        score_plain, radial_density = plain(jnp.asarray(1.))
        eps=1e-4
        finite = float((plain(jnp.asarray(1.+eps))[0]
                        -plain(jnp.asarray(1.-eps))[0])/(2*eps))
        relative_error = abs(float(derivative)-finite)/max(1.,abs(finite),
                                                           abs(float(derivative)))
        report['continuous_mark_seconds']=time.monotonic()-t
        report.update(count_key_mean=count_mean,count_key_source_sum=count_parts,
                      observed_radius_key_density=float(radial_density),
                      conditional_FP_logfactor_at_unit_velocity=float(score),
                      conditional_FP_logfactor_plain=float(score_plain),
                      velocity_scale_derivative=float(derivative),
                      velocity_scale_finite_difference=finite,
                      derivative_relative_discrepancy=relative_error,
                      status='ONE_LINKED_SOURCE_MECHANICS_PASS_NOT_CALIBRATED')
        if (not np.isfinite(float(score)) or float(radial_density)<=0
                or relative_error > .01
                or abs(float(score)-float(score_plain)) > 1e-8):
            raise FloatingPointError('linked selected-source score or support failed')
    except Exception as exc:
        report['status']='FAILED'
        report['error']=f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
