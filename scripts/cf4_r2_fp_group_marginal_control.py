"""Actual-source conditional group integration; provisional covariance sensitivity."""
import csv
import argparse
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
    conditional_group_scores, shared_zero_logfactor, selected_group_logweights,
    conditional_latent_group_scores, nonfp_modulus_logmarks)
from cf4_z0_physical_field import read_centred

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_fp_group_marginal_v1'
SOURCE = Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--same-field-density', action='store_true')
    parser.add_argument('--latent-group', action='store_true')
    parser.add_argument('--cross-method-anchors', action='store_true')
    parser.add_argument('--joint-calibration-cache', action='store_true')
    args = parser.parse_args()
    if sum((args.latent_group, args.cross_method_anchors, args.joint_calibration_cache)) > 1:
        parser.error('do not combine uncalibrated latent-role and anchor controls')
    args.same_field_density = (args.same_field_density or args.latent_group
                              or args.cross_method_anchors or args.joint_calibration_cache)
    out = BASE/'r2_fp_same_field_radial_v1' if args.same_field_density else OUT
    if args.latent_group:
        out = BASE/'r2_fp_latent_group_v1'
    if args.cross_method_anchors:
        out = BASE/'r2_cross_method_same_field_v1'
    if args.joint_calibration_cache:
        out = BASE/'r2_joint_calibration_cache_v1'
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    input_path = (BASE/'r2_source_observation_assembly_v1/observations.npz'
                  if args.same_field_density else BASE/'r2_sdss_fp_source_link_v1/source_link.npz')
    with np.load(input_path) as f:
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
    # Exchangeable observed-subset roles, solely an UNCALIBRATED mechanics
    # fixture. No native/mock truth role labels enter these probabilities.
    log_none = jnp.full(ng, np.log(.5))
    log_central = jnp.asarray(np.log(.5/nrows[row_group]))
    dirs = np.stack([np.bincount(row_group, weights=data['directions'][:,k])/nrows for k in range(3)],axis=1)
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    zgroup = np.bincount(row_group, weights=data['zgroup'])/nrows
    if np.max(np.abs(zgroup[row_group]-data['zgroup'])) > 1e-10:
        raise ValueError('source group has inconsistent reported redshifts')
    hold = np.bincount(row_group,weights=data['holdout'])
    if np.any((hold != 0) & (hold != nrows)):
        raise ValueError('split leaks source group')
    group_hold = hold > 0
    anchor_data = None
    if args.cross_method_anchors or args.joint_calibration_cache:
        anchor_base = BASE/'r2_cross_method_anchors_v1'
        with np.load(anchor_base/'anchors.npz') as f:
            anchor_data = {k:f[k].copy() for k in f.files}
        with np.load(anchor_base/'fp_group_moments.npz') as f:
            np.testing.assert_array_equal(f['group_labels'], group_labels)
            group_hold = f['holdout'].copy()
        method_names = sorted(set(anchor_data['method']))
        method_map = {m:i for i,m in enumerate(method_names)}
        anchor_method = jnp.asarray([method_map[m] for m in anchor_data['method']])
        if args.cross_method_anchors:
            binding = json.loads((anchor_base/'result.json').read_text())
            fit = binding['conditional_diagnostic_fit']
            assert fit['methods'] == method_names
            anchor_offset = jnp.asarray(fit['offset_mag'])
        np.testing.assert_array_equal(group_labels[anchor_data['group_index']],anchor_data['source_group'])
        np.testing.assert_array_equal(group_hold[anchor_data['group_index']],anchor_data['holdout'])
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
    sigmas = (150.,) if args.same_field_density else (150.,300.)
    zero_stds = (.004,) if args.same_field_density else (.004, float(np.hypot(.004,.0116)))
    suff = {s:sufficient(s) for s in sigmas}
    ztab = np.linspace(0,.2,20001)
    dtab = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+ztab)**3+.69),ztab,initial=0)
    dzgroup = np.interp(zgroup,ztab,dtab)
    dzrow = jnp.asarray(np.interp(data['zgroup'],ztab,dtab))
    with np.load(BASE/'r2_pm128_unconditional_v1/state.npz') as f:
        velocity = jnp.asarray(f['velocity_km_s'],dtype=jnp.float64)
        density = jnp.asarray(f['rho'],dtype=jnp.float64) if args.same_field_density else None
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
        rho = read_centred(density,positions,384.) if args.same_field_density else None
        d = jnp.asarray(dist)
        if args.joint_calibration_cache:
            # Fixed-state speed cache only. No prior/initialization from the
            # preceding fit; only original marks and the closed split enter.
            kernel = joint_redshift_logkernel(299792.458*zcos+(1+zcos)*radial,zcos,suff[150.])
            cache = dict(distance=dist,
                log_distance_weight=np.asarray(selected_group_logweights(
                    d,jnp.asarray(weight),rho,1.,jnp.zeros_like(d))),
                redshift_logkernel=np.asarray(kernel),row_group=row_group,
                dz_row=np.asarray(dzrow),group_holdout=group_hold,
                group_labels=group_labels,method_names=np.asarray(method_names),
                predicted_modulus=5*np.log10((1+zgroup[:,None])*dist/.746)+25,
                anchor_group=anchor_data['group_index'],anchor_modulus=anchor_data['modulus'],
                anchor_error=anchor_data['error'],anchor_method=np.asarray(anchor_method),
                **{k:data[k] for k in ('eta_mean','eta_std','eta_alpha')})
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'geometry_q{nq}.npz',**cache)
            print(f'joint calibration geometry Q={nq} saved; no fitted offsets read',flush=True)
            continue
        extra_marks = None
        if args.cross_method_anchors:
            # Frozen observed-z luminosity convention, matching the conditional
            # calibration diagnostic; not exact relativistic distance modelling.
            mu_prediction = 5*jnp.log10((1+jnp.asarray(zgroup[:,None]))*d/.746)+25
            extra_marks = nonfp_modulus_logmarks(mu_prediction,
                jnp.asarray(anchor_data['group_index']),jnp.asarray(anchor_data['modulus']),
                jnp.asarray(anchor_data['error']),anchor_method,anchor_offset)
        for sigma in sigmas:
            for zpstd in zero_stds:
                # A global calibration can be much narrower than its prior
                # after thousands of rows. Use a dense grid, not sparse
                # prior-centred GH; refine distance AND calibration quadrature.
                nz = 129 if nq == 257 else 257
                zero = jnp.asarray(np.linspace(-8*zpstd,8*zpstd,nz))
                logzw_np = -.5*(np.asarray(zero)/zpstd)**2
                logzw_np[[0,-1]] += np.log(.5)
                logzw = jnp.asarray(logzw_np)
                uniform_logdw = jnp.asarray(np.log(weight)+2*np.log(dist))
                # b=1, group inclusion=1 are an explicit mechanics fixture,
                # NOT measured bias or a calibrated selected-group prior.
                logdw = (selected_group_logweights(d,jnp.asarray(weight),rho,1.,jnp.zeros_like(d))
                         if args.same_field_density else uniform_logdw)
                nu = 5 if nq == 257 else 9
                ux, uw = np.polynomial.hermite.hermgauss(nu)
                group_offset = jnp.asarray(np.sqrt(2.)*.005*ux)
                log_group_weight = jnp.asarray(np.log(uw))
                def scores(a, bias=1.):
                    prediction = 299792.458*zcos+(1+zcos)*a*radial
                    kernel = joint_redshift_logkernel(prediction,zcos,suff[sigma])
                    w = (selected_group_logweights(d,jnp.asarray(weight),rho,bias,jnp.zeros_like(d))
                         if args.same_field_density else logdw)
                    if args.latent_group:
                        return conditional_latent_group_scores(d,w,kernel,gid,dzrow,*moments,zero,
                            log_none,log_central,-.005,.005,group_offset,log_group_weight)
                    return conditional_group_scores(d,w,kernel,gid,dzrow,*moments,zero,
                                                     extra_log_marks=extra_marks)
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
                if args.same_field_density:
                    kernel = joint_redshift_logkernel(299792.458*zcos+(1+zcos)*radial,zcos,suff[sigma])
                    if args.latent_group:
                        baseline_scores = jax.jit(conditional_group_scores)(d,logdw,kernel,gid,dzrow,*moments,zero)
                        trial.update(group_offset_nodes=nu,group_offset_std_dex=.005,
                            central_offset_dex=-.005,satellite_offset_dex=.005,
                            observed_central_probability=.5,
                            train_latent_minus_baseline=get(state_scores,train)-get(baseline_scores,train),
                            all_latent_minus_baseline=get(state_scores,allgroups)-get(baseline_scores,allgroups))
                    elif args.cross_method_anchors:
                        fp_only = jax.jit(conditional_group_scores)(d,logdw,kernel,gid,dzrow,*moments,zero)
                        trial['train_anchor_on_minus_off'] = get(state_scores,train)-get(fp_only,train)
                        trial['all_anchor_on_minus_off'] = get(state_scores,allgroups)-get(fp_only,allgroups)
                    else:
                        uniform_scores = conditional_group_scores(d,uniform_logdw,kernel,gid,dzrow,*moments,zero)
                        trial['train_density_weight_minus_uniform'] = get(state_scores,train)-get(uniform_scores,train)
                    trial['zero_density_quadrature_nodes'] = int(np.sum(np.asarray(rho)<=0))
                prediction = 299792.458*zcos+(1+zcos)*radial
                base = logdw+joint_redshift_logkernel(prediction,zcos,suff[sigma])
                base_np = np.asarray(base)
                edge = np.exp(logsumexp(base_np[:,[0,-1]],axis=1)-logsumexp(base_np,axis=1))
                trial['max_endpoint_weight'] = float(edge.max())
                if nq == 513 and sigma == 150. and zpstd == .004 and not (args.latent_group or args.cross_method_anchors):
                    f = lambda a: shared_zero_logfactor(scores(a),logzw,train)
                    derivative = float(jax.jit(jax.grad(f))(1.))
                    fd = (float(f(1.0001))-float(f(.9999)))/.0002
                    trial.update(amplitude_derivative=derivative,finite_difference=fd,
                        gradient_agreement=bool(np.isclose(derivative,fd,rtol=2e-3,atol=1e-4)))
                    if args.same_field_density:
                        fb = lambda b: shared_zero_logfactor(scores(1.,b),logzw,train)
                        db = float(jax.jit(jax.grad(fb))(1.))
                        fdb = (float(fb(1.0001))-float(fb(.9999)))/.0002
                        trial.update(radial_bias_derivative=db,radial_bias_finite_difference=fdb,
                            radial_bias_gradient_agreement=bool(np.isclose(db,fdb,rtol=2e-3,atol=1e-4)))
                    saved_scores = dict(group_scores=np.asarray(state_scores),zero_nodes=np.asarray(zero),
                        log_zero_weights=np.asarray(logzw),group_labels=group_labels,group_holdout=group_hold)
                if args.latent_group and nq == 513:
                    saved_scores = dict(group_scores=np.asarray(state_scores),zero_nodes=np.asarray(zero),
                        log_zero_weights=np.asarray(logzw),group_labels=group_labels,group_holdout=group_hold,
                        group_offset_nodes=np.asarray(group_offset),log_group_offset_weights=np.asarray(log_group_weight),
                        row_group=row_group,log_none_weight=np.asarray(log_none),log_central_weight=np.asarray(log_central))
                if args.cross_method_anchors and nq == 513:
                    saved_scores = dict(group_scores=np.asarray(state_scores),zero_nodes=np.asarray(zero),
                        log_zero_weights=np.asarray(logzw),group_labels=group_labels,group_holdout=group_hold,
                        method_names=np.asarray(fit['methods']),method_offsets=np.asarray(anchor_offset))
                if not np.isfinite(list(trial.values())).all():
                    raise FloatingPointError('nonfinite marginal calculation')
                trials.append(trial)
                print(json.dumps(trial),flush=True)
    if args.joint_calibration_cache:
        inputs = [input_path,anchor_base/'anchors.npz',anchor_base/'fp_group_moments.npz',
                  BASE/'r2_pm128_unconditional_v1/state.npz']
        manifest = dict(classification='FIXED_STATE_CONDITIONAL_CALIBRATION_GEOMETRY',
            job_id=os.environ['SLURM_JOB_ID'],train_groups=int((~group_hold).sum()),
            holdout_groups=int(group_hold.sum()),FP_rows=len(row_group),
            nonFP_rows=len(anchor_data['PGC']),method_names=method_names,
            fitted_offsets_used=False,new_gravity_runs=0,R2_posterior=False,
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__),ROOT/'src/cf4_r2_fp_group_marginal.py')},
            runtime_seconds=time.monotonic()-start)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(manifest),flush=True)
        return
    half = len(trials)//2
    differences = [abs(trials[i]['train_logratio_to_zero_velocity']-trials[i+half]['train_logratio_to_zero_velocity']) for i in range(half)]
    result = dict(classification='CONDITIONAL_GROUP_DISTANCE_AND_SHARED_ZERO_IMPLEMENTATION_CONTROL',
        job_id=os.environ['SLURM_JOB_ID'],rows=len(row_group),source_groups=ng,
        train_groups=int((~group_hold).sum()),holdout_groups=int(group_hold.sum()),
        conditioned_unique_2mpp_members=sum(map(len,grouped_records.values())),
        groups_with_2mpp_members=sum(bool(v) for v in grouped_records.values()),
        ambiguous_2mpp_covariates_omitted=ambiguous,
        trials=trials,max_train_ratio_quadrature_difference=max(differences),
        normalized_conditional_redshift_denominator=True,shared_zero_point_integrated_once=True,
        group_covariance_calibrated=False,
        same_state_density_radial_measure=args.same_field_density,
        selected_group_distance_prior=('r^2 rho dd; bias1/inclusion1 mechanics fixture, NOT calibrated'
                                      if args.same_field_density else 'provisional r^2 dd in recorded finite radial window'),
        within_count_cell_point_law_implemented=False,complete_joint_likelihood=False,
        independent_holdout_validation=False,R2_posterior=False,new_gravity_runs=0,
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__),ROOT/'src/cf4_r2_fp_group_marginal.py')},
        source_sha256={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                       for name in ('2mpp_catalog.csv','cf4_2mpp_crossmatch_v1.csv')},
        runtime_seconds=time.monotonic()-start)
    if args.latent_group:
        result.update(classification='UNCALIBRATED_LATENT_ROLE_AND_SHARED_GROUP_MARK_CONTROL',
            roles_and_group_scatter_calibrated=False,role_dependent_redshift_kernel=False,
            source_fit_covariance_resolved=False,nuisance_fits=0,
            max_latent_baseline_quadrature_difference=max(abs(trials[0][k]-trials[1][k])
                for k in ('train_latent_minus_baseline','all_latent_minus_baseline')),
            full_adjoint_cost_measured=False)
    if args.cross_method_anchors:
        result.update(classification='ACTUAL_NONFP_SAME_FIELD_CONDITIONAL_MARK_CONNECTION',
            nonfp_rows=len(anchor_data['PGC']),nonfp_groups=len(set(anchor_data['group_index'])),
            method_offsets_conditioned_from_training_diagnostic=True,
            fitted_offsets_used_as_independent_prior=False,
            relative_excess_scatter_used_as_FP_group_error=False,
            exact_luminosity_Doppler_model=False,
            anchor_source_result_sha256=hashlib.sha256((anchor_base/'result.json').read_bytes()).hexdigest(),
            max_anchor_factor_quadrature_difference=max(abs(trials[0][k]-trials[1][k])
                for k in ('train_anchor_on_minus_off','all_anchor_on_minus_off')))
    out.mkdir(parents=True)
    np.savez_compressed(out/'group_factors.npz',**saved_scores)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
