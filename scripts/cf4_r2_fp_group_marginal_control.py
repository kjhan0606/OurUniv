"""Actual-source conditional group integration; provisional covariance sensitivity."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_fp_group_marginal import (redshift_sufficient, joint_redshift_logkernel,
    conditional_group_scores, shared_zero_logfactor)
from cf4_z0_physical_field import read_centred

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_fp_group_marginal_v1'
SOURCE = Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic()
    with np.load(BASE/'r2_sdss_fp_source_link_v1/source_link.npz') as f:
        data = {k: f[k].copy() for k in f.files}
    if hashlib.md5(SOURCE.read_bytes()).hexdigest() != 'b5b6e31caf7ea469c2ac2cb775fa8d14':
        raise ValueError('source checksum mismatch')
    with SOURCE.open() as f:
        cols = f.readline().lstrip('#').split()
        source = {int(r['PGC']): r for r in (dict(zip(cols, l.split(), strict=True)) for l in f if l.strip())}
    with (ROOT/'data/2mpp_catalog.csv').open() as f:
        members = {int(r['recno']): r for r in csv.DictReader(f)}
    with np.load(BASE/'r2_point_mark_manifest_v1/points.npz') as f:
        eligible = set(map(int, f['recno']))
    group_labels, row_group = np.unique(data['source_group'], return_inverse=True)
    ng = len(group_labels)
    nrows = np.bincount(row_group)
    dirs = np.stack([np.bincount(row_group, weights=data['directions'][:,k])/nrows for k in range(3)],axis=1)
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    zgroup = np.bincount(row_group, weights=data['zgroup'])/nrows
    if np.max(np.abs(zgroup[row_group]-data['zgroup'])) > 1e-10:
        raise ValueError('source group has inconsistent reported redshifts')
    hold = np.bincount(row_group,weights=data['holdout'])
    if np.any((hold != 0) & (hold != nrows)):
        raise ValueError('split leaks source group')
    group_hold = hold > 0
    pgc_to_group = dict(zip(map(int,data['PGC']),map(int,row_group)))
    rec_groups = defaultdict(set)
    with (ROOT/'data/cf4_2mpp_crossmatch_v1.csv').open() as f:
        for r in csv.DictReader(f):
            if r['match_class'] != 'secure_joint_mark' or not r['twompp_recno']:
                continue
            p, rec = int(r['PGC']), int(r['twompp_recno'])
            if p in pgc_to_group and rec in eligible:
                rec_groups[rec].add(pgc_to_group[p])
    grouped_records = defaultdict(list)
    ambiguous = []
    for rec, groups in rec_groups.items():
        if len(groups) == 1:
            grouped_records[next(iter(groups))].append(rec)
        else:
            ambiguous.append(rec)
    richness = np.ones(ng, dtype=int)
    for p, g in pgc_to_group.items():
        richness[g] = max(richness[g], int(source[p]['NgroupT17']))
    # Trial law: shared unresolved COM scatter100, independent member scatter
    #150/300, group mean covariance sigma^2/N; extra group-catalogue scatter50.
    # These are declared sensitivity values, NOT source-calibrated errors.
    def sufficient(sigma):
        output = []
        for g in range(ng):
            recs = sorted(grouped_records[g])
            if len(recs) > richness[g]:
                raise ValueError('more matched velocity members than source group richness')
            n = 1+len(recs)
            covariance = np.ones((n,n))*100.**2
            covariance[0,:] += sigma**2/richness[g]
            covariance[1:,0] += sigma**2/richness[g]
            covariance[0,0] += 50.**2
            for k, rec in enumerate(recs, 1):
                error = float(members[rec]['e_HV'] or 0.)/(1+zgroup[g])
                covariance[k,k] += sigma**2+error**2
            y = [299792.458*zgroup[g]]+[float(members[r]['Vcmb']) for r in recs]
            output.append(redshift_sufficient(y,covariance,[f'SDSS:{group_labels[g]}']+[f'2mpp:{r}' for r in recs]))
        return jnp.asarray(np.array(output))
    suff = {s:sufficient(s) for s in (150.,300.)}
    ztab = np.linspace(0,.2,20001)
    dtab = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+ztab)**3+.69),ztab,initial=0)
    dzgroup = np.interp(zgroup,ztab,dtab)
    dzrow = jnp.asarray(np.interp(data['zgroup'],ztab,dtab))
    with np.load(BASE/'r2_pm128_unconditional_v1/state.npz') as f:
        velocity = jnp.asarray(f['velocity_km_s'],dtype=jnp.float64)
    gid = jnp.asarray(row_group)
    moments = [jnp.asarray(data[k]) for k in ('eta_mean','eta_std','eta_alpha')]
    train = jnp.asarray(~group_hold)
    allgroups = jnp.ones(ng,dtype=bool)
    trials = []
    saved_scores = {}
    for nq in (257,513):
        low = np.maximum(1., dzgroup-40.)
        high = np.minimum(191.99/np.max(np.abs(dirs),axis=1),dzgroup+40.)
        dist = low[:,None]+(high-low)[:,None]*np.linspace(0,1,nq)
        dd = (high-low)/(nq-1)
        weight = np.broadcast_to(dd[:,None],dist.shape).copy()
        weight[:,[0,-1]] *= .5
        zcos = jnp.asarray(np.interp(dist,dtab,ztab))
        positions = jnp.asarray(192.+dirs[:,None,:]*dist[:,:,None])
        radial = sum(read_centred(velocity[k],positions,384.)*jnp.asarray(dirs[:,k,None]) for k in range(3))
        d = jnp.asarray(dist)
        for sigma in (150.,300.):
            for zpstd in (.004, float(np.hypot(.004,.0116))):
                # A global calibration can be much narrower than its prior
                # after thousands of rows. Use a dense grid, not sparse
                # prior-centred GH; refine distance AND calibration quadrature.
                nz = 129 if nq == 257 else 257
                zero = jnp.asarray(np.linspace(-8*zpstd,8*zpstd,nz))
                logzw_np = -.5*(np.asarray(zero)/zpstd)**2
                logzw_np[[0,-1]] += np.log(.5)
                logzw = jnp.asarray(logzw_np)
                logdw = jnp.asarray(np.log(weight)+2*np.log(dist))
                def scores(a):
                    prediction = 299792.458*zcos+(1+zcos)*a*radial
                    kernel = joint_redshift_logkernel(prediction,zcos,suff[sigma])
                    return conditional_group_scores(d,logdw,kernel,gid,dzrow,*moments,zero)
                calculate = jax.jit(scores)
                state_scores = calculate(1.)
                zero_scores = calculate(0.)
                state_scores.block_until_ready()
                get = lambda s, mask: float(shared_zero_logfactor(s,logzw,mask))
                trial = dict(distance_nodes=nq, zero_nodes=nz, sigma_member_km_s=sigma, zero_std_dex=zpstd,
                    train_logratio_to_zero_velocity=get(state_scores,train)-get(zero_scores,train),
                    all_logratio_to_zero_velocity=get(state_scores,allgroups)-get(zero_scores,allgroups))
                # Heldout conditional on training updates the SAME zero point.
                trial['holdout_conditional_logratio'] = trial['all_logratio_to_zero_velocity']-trial['train_logratio_to_zero_velocity']
                prediction = 299792.458*zcos+(1+zcos)*radial
                base = logdw+joint_redshift_logkernel(prediction,zcos,suff[sigma])
                base_np = np.asarray(base)
                edge = np.exp(logsumexp(base_np[:,[0,-1]],axis=1)-logsumexp(base_np,axis=1))
                trial['max_endpoint_weight'] = float(edge.max())
                if nq == 513 and sigma == 150. and zpstd == .004:
                    f = lambda a: shared_zero_logfactor(scores(a),logzw,train)
                    derivative = float(jax.jit(jax.grad(f))(1.))
                    fd = (float(f(1.0001))-float(f(.9999)))/.0002
                    trial.update(amplitude_derivative=derivative,finite_difference=fd,
                        gradient_agreement=bool(np.isclose(derivative,fd,rtol=2e-3,atol=1e-4)))
                    saved_scores = dict(group_scores=np.asarray(state_scores),zero_nodes=np.asarray(zero),
                        log_zero_weights=np.asarray(logzw),group_labels=group_labels,group_holdout=group_hold)
                if not np.isfinite(list(trial.values())).all():
                    raise FloatingPointError('nonfinite marginal calculation')
                trials.append(trial)
                print(json.dumps(trial),flush=True)
    differences = [abs(trials[i]['train_logratio_to_zero_velocity']-trials[i+4]['train_logratio_to_zero_velocity']) for i in range(4)]
    result = dict(classification='CONDITIONAL_GROUP_DISTANCE_AND_SHARED_ZERO_IMPLEMENTATION_CONTROL',
        job_id=os.environ['SLURM_JOB_ID'],rows=len(row_group),source_groups=ng,
        train_groups=int((~group_hold).sum()),holdout_groups=int(group_hold.sum()),
        conditioned_unique_2mpp_members=sum(map(len,grouped_records.values())),
        groups_with_2mpp_members=sum(bool(v) for v in grouped_records.values()),
        ambiguous_2mpp_covariates_omitted=ambiguous,
        trials=trials,max_train_ratio_quadrature_difference=max(differences),
        normalized_conditional_redshift_denominator=True,shared_zero_point_integrated_once=True,
        group_covariance_calibrated=False,selected_group_distance_prior='provisional r^2 dd in recorded finite radial window',
        within_count_cell_point_law_implemented=False,complete_joint_likelihood=False,
        independent_holdout_validation=False,R2_posterior=False,new_gravity_runs=0,
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__),ROOT/'src/cf4_r2_fp_group_marginal.py')},
        source_sha256={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                       for name in ('2mpp_catalog.csv','cf4_2mpp_crossmatch_v1.csv')},
        runtime_seconds=time.monotonic()-start)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'group_factors.npz',**saved_scores)
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
