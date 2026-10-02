"""Fixed-state periodic observed-ray FP mechanics on every training link."""
import json
import os
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_linked_fp_sparse_train import (
    FP, load_train_singletons, select_training_single_mark_links,
)
from cf4_r2_native_to_count_cells import native_moments_to_count_cells
from cf4_r2_observed_ray import observed_ray_components, extend_flat_distance_tables
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_raw_live_mark import chunk_log_terms, POPULATION_ORIGIN, POPULATION_SCALE
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses


BASE=Path('/gpfs/kjhan/CF4/z0_density')
STATE=BASE/'r2_raw_joint_pilot_v1/accepted_present_state.npz'
SPLIT=BASE/'r2_sky_closed_split_v6/split.npz'
BOUND=BASE/'r2_observed_ray_cohort_bound_20261002_v2/result.json'
BOX=384.


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(exist_ok=False)
    started=time.monotonic()

    _,_,_,_,source,mix,o,g=load_inputs()
    options,point,_=load_train_singletons(SPLIT)
    with np.load(FP,allow_pickle=False) as f:
        options=select_training_single_mark_links(
            options,f['membership_state'].astype(str),include_grouped=True)
        chosen=[item for p in range(6) for item in options
                if point['population'][item[2]]==p]
        pgc=np.asarray([f['PGC'][item[3]] for item in chosen],dtype=np.int64)
        direction=np.asarray([f['directions'][item[3]] for item in chosen],dtype=np.float64)
    np.testing.assert_array_equal(mix['PGC'],pgc)
    flat=point['flat_cell'];pop=point['population']
    linked_points=np.asarray([item[2] for item in chosen],dtype=np.int64)
    linked_voxels=np.stack(np.unravel_index(flat[linked_points],(128,)*3),axis=-1)
    np.testing.assert_array_equal(linked_voxels,np.asarray(o['voxel']))
    with np.load(SPLIT,allow_pickle=False) as f:
        training=np.isin(pop.astype(np.int64)*128**3+flat,f['train_keys'])
        train_keys=len(f['train_keys'])
    if int(training.sum())!=47121 or len(chosen)!=1414:
        raise ValueError('frozen training cohort ownership changed')
    if direction.shape!=(len(chosen),3) or not np.isfinite(direction).all():
        raise ValueError('observed-ray registration failed')
    direction_error=float(np.max(np.abs(np.linalg.norm(direction,axis=1)-1.)))
    if direction_error>2e-12:
        raise ValueError('FP source direction is not unit normalized')

    with np.load(STATE,allow_pickle=False) as f:
        rho,velocity,variance=map(jnp.asarray,
            (f['rho'],f['mean_velocity_km_s'],f['physical_velocity_variance_km2_s2']))
        old=f['tracer']
        tracer=jnp.asarray(np.r_[old[:6],0.,.12*np.exp(.5*old[7]),old[8]])
        params=jnp.asarray(POPULATION_ORIGIN)+jnp.asarray(POPULATION_SCALE)*jnp.asarray(f['population_white'])
    rho,velocity,variance=jax.jit(native_moments_to_count_cells,static_argnums=3)(
        rho,velocity,variance,BOX)
    velocity=jnp.moveaxis(velocity,0,-1).reshape(-1,3)
    variance=jnp.moveaxis(variance,0,-1).reshape(-1,3)
    g=tracer_geometry(tracer,g)
    if not np.allclose(np.asarray(g['observer']),np.full(3,BOX/2),rtol=0.,atol=1e-12):
        raise ValueError('directional face formula assumes the registered box-centre observer')
    masses=tracer_masses(rho,tracer)
    sky=jnp.asarray(source['angular'])

    vmax=float(jnp.max(jnp.linalg.norm(velocity,axis=1)))
    sigma_bound=float(jnp.sqrt(30.**2+.5**2*jnp.max(variance)))
    displacement_bound=.01*(vmax+8*sigma_bound)
    max_support=float(np.max(mix['observed_radius'])+displacement_bound)
    if not max_support<BOX:
        raise ValueError('one-full-box ray interval does not cover the conservative radial support')
    g,table_extension=extend_flat_distance_tables(g,max_support+1.)

    with open(BOUND,encoding='utf-8') as stream:
        prior_bound=json.load(stream)
    face=BOX/2/np.max(np.abs(direction),axis=1)
    radius=np.asarray(mix['observed_radius'],dtype=np.float64)
    margin=face-radius-displacement_bound
    uncertified=np.flatnonzero(margin<=0.)
    safe=np.flatnonzero(margin>0.)
    if (len(safe)!=int(prior_bound['positive_margin_rows'])
            or len(uncertified)!=int(prior_bound['nonpositive_margin_rows'])):
        raise ValueError('saved-state first-face cohort differs from the recorded certificate')
    representative_safe=(safe[np.linspace(0,len(safe)-1,min(40,len(safe)),dtype=int)]
                         if len(safe) else np.zeros(0,dtype=np.int64))
    convergence_rows=np.asarray(sorted(set(map(int,uncertified))|
        set(map(int,representative_safe))),dtype=np.int64)

    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],
        source_commit=os.environ['CF4_EXPECTED_COMMIT'],R2_complete=False,
        actual_state='saved N128 accepted pilot; nonstationary; fixed nuisance state',
        training_points=int(training.sum()),training_keys=int(train_keys),
        registered_training_FP_links=len(chosen),heldout_scored=False,
        PM_evolutions=0,count_backend_changed=False,posterior_promoted=False,
        FP_geometry='forward q>=0; wrapped periodic field positions; unwrapped physical q passed into FP modulus, redshift, eta and K-correction',
        angular_selection='omitted only as a positive direction-constant scalar that cancels in this per-direction conditional FP numerator/denominator',
        count_law_semantics=dict(observed_radial_shell_cMpc_h=[5.,180.],
            box_half_cMpc_h=BOX/2,uses_27_periodic_RSD_shell_images=True,
            count_FP_image_fraction_reconciliation='NOT DONE; no target wiring in this bundle'),
        FP_first_face_margin_cMpc_h=dict(positive=int(len(safe)),nonpositive=int(len(uncertified))),
        previous_bound_job=int(prior_bound['job_id']),
        bound=dict(maximum_grid_speed_km_s=vmax,maximum_diagonal_variance_km2_s2=float(jnp.max(variance)),
            eight_sigma_plus_coherent_shift_cMpc_h=displacement_bound,
            maximum_supported_unwrapped_radius_cMpc_h=max_support,
            minimum_full_box_horizon_cMpc_h=BOX),
        distance_table_extension=table_extension,rows=[],
        order8_completed=0,order4_sample_completed=0,AD_FD=[],
        limitations=['fixed-state in-sample conditional factor mechanics only',
            'no count/FP image-semantics reconciliation or target wiring',
            'no calibration, heldout, PM/gravity, posterior, or production claim',
            'MW/M31 remain role-ambiguous and M33 unresolved; later evidence must constrain the same NEW field'])

    def save():
        report['seconds']=time.monotonic()-started
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        tmp=out/'result.json.tmp'
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        tmp.replace(out/'result.json')

    def observation_at(i):
        return {k:jnp.asarray(value[i]) for k,value in o.items()}

    def make_eval(population,order):
        def evaluate(logscale,angle,obs):
            closure=dict(core_sigma_km_s=30.,dispersion_scale=.5*jnp.exp(logscale),broad_fraction=.5)
            positions,vel,intrinsic,selection,q,weights=observed_ray_components(
                angle,obs['radius'],velocity,variance,masses,sky,population,g,closure,
                source_grid=128,order=order)
            face_radius=BOX/2/jnp.max(jnp.abs(angle))
            near=weights*(q[None,:]<=face_radius)
            far=weights*(q[None,:]>face_radius)
            common=dict(population=population,geometry=g,cut_order=64,
                radial_source_mass=weights,source_radius_cMpc_h=q)
            full=chunk_log_terms(params,positions,vel,intrinsic,selection,obs,**common)
            near_terms=chunk_log_terms(params,positions,vel,intrinsic,selection,obs,
                **dict(common,radial_source_mass=near))
            far_terms=chunk_log_terms(params,positions,vel,intrinsic,selection,obs,
                **dict(common,radial_source_mass=far))
            radial_total=jnp.sum(weights)
            radial_image=jnp.sum(far)
            frac_num=jnp.where(jnp.isfinite(far_terms[0]),
                jnp.exp(far_terms[0]-full[0]),0.)
            frac_den=jnp.where(jnp.isfinite(far_terms[1]),
                jnp.exp(far_terms[1]-full[1]),0.)
            return jnp.stack((full[0]-full[1],full[0],full[1],
                near_terms[0],near_terms[1],far_terms[0],far_terms[1],
                radial_total,radial_image,frac_num,frac_den))
        return jax.jit(evaluate)

    order8={p:make_eval(p,8) for p in range(6)}
    order4={p:make_eval(p,4) for p in range(6)}
    by_population=[np.flatnonzero(np.asarray(mix['population'])==p) for p in range(6)]
    eval_order=[int(by_population[p][j]) for j in range(max(map(len,by_population)))
        for p in range(6) if j<len(by_population[p])]
    row8={};compiled_order8_populations=set()
    report['phase']='ORDER8_ALL_TRAINING_LINKS';save()
    for count,i in enumerate(eval_order,1):
        p=int(mix['population'][i]);obs=observation_at(i);angle=jnp.asarray(direction[i])
        first_population_compile=p not in compiled_order8_populations
        compiled_order8_populations.add(p)
        tic=time.monotonic();values=order8[p](jnp.asarray(0.),angle,obs)
        values=np.asarray(jax.tree_util.tree_map(lambda x:x.block_until_ready(),values),dtype=np.float64)
        elapsed=time.monotonic()-tic
        nll,num,den,near_num,near_den,far_num,far_den,radial,image,frac_num,frac_den=values.tolist()
        pieces=np.asarray([nll,num,den,radial,image,frac_num,frac_den])
        finite=bool(np.isfinite(pieces).all() and radial>0.)
        if np.isfinite(num) and np.isfinite(den):
            num_join=float(np.logaddexp(near_num,far_num))
            den_join=float(np.logaddexp(near_den,far_den))
            decomposition_error=max(abs(num_join-num),abs(den_join-den))
        else:
            decomposition_error=None
        record=dict(index=int(i),PGC=int(pgc[i]),population=p,
            first_face_cMpc_h=float(face[i]),no_wrap_margin_cMpc_h=float(margin[i]),
            logpdf=float(nll) if np.isfinite(nll) else None,
            log_numerator=float(num) if np.isfinite(num) else None,
            log_denominator=float(den) if np.isfinite(den) else None,
            radial_selected_weight=float(radial),periodic_image_radial_mass_fraction=float(image/radial) if radial>0 else None,
            periodic_image_numerator_fraction=float(frac_num) if np.isfinite(frac_num) else None,
            periodic_image_denominator_fraction=float(frac_den) if np.isfinite(frac_den) else None,
            numerator_denominator_split_error=decomposition_error,
            status='FINITE' if finite else 'NONFINITE_OR_ZERO_SUPPORT',seconds=elapsed,
            first_population_compile=first_population_compile)
        row8[i]=record;report['rows'].append(record);report['order8_completed']=count
        if count==20:
            first20=time.monotonic()-started
            compile_seconds=[r['seconds'] for r in report['rows'] if r['first_population_compile']]
            warm=[r['seconds'] for r in report['rows'] if not r['first_population_compile']]
            if len(compile_seconds)!=6 or len(warm)<5:
                report.update(status='STOPPED_RUNTIME_ESTIMATE_UNRESOLVED',phase='STOPPED_AFTER_FIRST20')
                save();return
            warm_rate=float(np.mean(warm))
            # All six order-8 populations have now compiled. Forecast only the
            # warm per-row run, then reserve six order-4 and two AD compilations.
            projected=(first20+(len(chosen)-count+len(convergence_rows))*warm_rate
                +8*max(compile_seconds))
            report['runtime_forecast_after_20_rows']=dict(
                elapsed_first20_seconds=first20,order8_population_compile_seconds=float(sum(compile_seconds)),
                warmed_rows=len(warm),warm_seconds_per_row=warm_rate,
                remaining_order8_rows=len(chosen)-count,order4_convergence_rows=len(convergence_rows),
                reserved_additional_compilations=8,reserved_compile_seconds=8*max(compile_seconds),
                total_projected_seconds=projected)
            report['runtime_projection_after_20_rows_seconds']=projected
            if projected>1200.:
                report.update(status='STOPPED_RUNTIME_PROJECTION_GT20MIN',phase='STOPPED_AFTER_FIRST20')
                save();return
        if count%50==0:save()
    save()

    nonfinite_rows=[r['PGC'] for r in report['rows'] if r['status']!='FINITE']
    if nonfinite_rows:
        report.update(status='STOPPED_AFTER_FULL_CENSUS_NONFINITE_ROWS',
            phase='REVIEW_REQUIRED',nonfinite_or_zero_support_PGC=nonfinite_rows)
        save();return

    report['phase']='ORDER4_CONVERGENCE_SAMPLE';save()
    convergence=[]
    for i in convergence_rows:
        p=int(mix['population'][i]);obs=observation_at(int(i));angle=jnp.asarray(direction[i])
        values=np.asarray(jax.tree_util.tree_map(lambda x:x.block_until_ready(),
            order4[p](jnp.asarray(0.),angle,obs)),dtype=np.float64)
        logpdf4=float(values[0]);logpdf8=row8[int(i)]['logpdf']
        delta=abs(logpdf8-logpdf4) if logpdf8 is not None and np.isfinite(logpdf4) else None
        convergence.append(dict(index=int(i),PGC=int(pgc[i]),logpdf_order4=logpdf4 if np.isfinite(logpdf4) else None,
            logpdf_order8=logpdf8,absolute_delta_nat=delta,
            status='FINITE' if delta is not None else 'NONFINITE'))
        report['order4_sample_completed']=len(convergence)
        if len(convergence)%20==0:save()
    deltas=[r['absolute_delta_nat'] for r in convergence if r['absolute_delta_nat'] is not None]
    report['convergence']=dict(sample_size=len(convergence),uncertified_rows_sampled=int(len(uncertified)),
        representative_safe_rows_sampled=int(len(representative_safe)),
        maximum_absolute_order4_order8_delta_nat=max(deltas) if deltas else None,
        sum_absolute_order4_order8_delta_nat=float(np.sum(deltas)) if deltas else None,
        per_row_tolerance_nat=1e-3,total_absolute_tolerance_nat=.1,
        pass_=bool(len(deltas)==len(convergence) and max(deltas,default=np.inf)<=1e-3
            and sum(deltas)<=.1))
    save()

    report['phase']='AD_FD_CHECKS';save()
    interior=int(np.flatnonzero(pgc==26124)[0])
    image_candidates=[i for i,r in row8.items()
        if r['periodic_image_radial_mass_fraction'] is not None
        and r['periodic_image_radial_mass_fraction']>0.]
    if not image_candidates:
        report.update(status='STOPPED_NO_FINITE_IMAGE_MASS_FOR_AD_FD',phase='REVIEW_REQUIRED')
        save();return
    wrapped=max(image_candidates,key=lambda i:row8[i]['periodic_image_radial_mass_fraction'])
    anchor=[interior,wrapped]
    for i in anchor:
        p=int(mix['population'][i]);obs=observation_at(i);angle=jnp.asarray(direction[i]);fn=order8[p]
        scalar=lambda s:fn(s,angle,obs)[0]
        derivative=jax.jit(jax.value_and_grad(scalar))
        tic=time.monotonic();value,ad=derivative(jnp.asarray(0.));value,ad=map(float,
            (value.block_until_ready(),ad.block_until_ready()))
        eps=1e-4;plus=float(scalar(jnp.asarray(eps)).block_until_ready())
        minus=float(scalar(jnp.asarray(-eps)).block_until_ready())
        fd=(plus-minus)/(2*eps);error=abs(ad-fd)/max(1.,abs(ad),abs(fd))
        report['AD_FD'].append(dict(index=i,PGC=int(pgc[i]),case=('interior' if margin[i]>0 else 'periodic-image'),
            value=value,AD=ad,FD=fd,relative_error=error,seconds=time.monotonic()-tic,
            pass_=bool(np.isfinite([value,ad,fd,error]).all() and error<=2e-5)))
        save()

    finite_rows=sum(r['status']=='FINITE' for r in row8.values())
    bad_rows=[r['PGC'] for r in report['rows'] if r['status']!='FINITE']
    image_num=[r['periodic_image_numerator_fraction'] for r in report['rows']
        if r['periodic_image_numerator_fraction'] is not None]
    image_den=[r['periodic_image_denominator_fraction'] for r in report['rows']
        if r['periodic_image_denominator_fraction'] is not None]
    report['summary']=dict(order8_finite_rows=finite_rows,nonfinite_or_zero_support_PGC=bad_rows,
        maximum_image_numerator_fraction=max(image_num) if image_num else None,
        maximum_image_denominator_fraction=max(image_den) if image_den else None,
        rows_over_one_percent_image_numerator=sum(x>.01 for x in image_num),
        rows_over_one_percent_image_denominator=sum(x>.01 for x in image_den),
        split_decomposition_max_abs_log_error=max((r['numerator_denominator_split_error'] or 0.
            for r in report['rows']),default=0.))
    status_ok=(finite_rows==len(chosen) and report['convergence']['pass_']
        and all(v['pass_'] for v in report['AD_FD'])
        and report['summary']['split_decomposition_max_abs_log_error']<1e-9)
    report.update(status=('PERIODIC_TRAINING_FP_MECHANICS_CHECKED_NOT_POSTERIOR' if status_ok
        else 'PERIODIC_TRAINING_FP_MECHANICS_INCOMPLETE_NOT_POSTERIOR'),phase='COMPLETE_NOT_PROMOTED')
    save()


if __name__=='__main__':main()
