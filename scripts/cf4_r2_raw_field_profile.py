"""Full native-field raw adjoint with fresh support at each finite-difference state."""
import json
import os
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_linked_fp_sparse_train import (load_train_singletons,
    select_training_single_mark_links,linked_point_conditioning_radii,FP,SOURCE)
from cf4_r2_v6_active1414_association_reconciliation import (
    clean_conditional_row_indices,reconcile_selected_rows,
    read_verified_group_ledger)
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_raw_live_mark import POPULATION_ORIGIN,POPULATION_SCALE
from cf4_r2_raw_volume_target import FreshRawSupport,raw_field_logpdf

BASE=Path('/gpfs/kjhan/CF4/z0_density');BOX=384.;N=128


def load_inputs():
    with np.load(BASE/'r2_prior_split_long_v1/final_state.npz',allow_pickle=False) as f:
        rho,velocity,tracer=map(jnp.asarray,(f['rho'],f['velocity_km_s'],f['tracer']))
    with np.load(SOURCE,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    with np.load(BASE/'r2_raw_source_mixture_v1/raw_source_mixture.npz',allow_pickle=False) as f:
        mix={k:f[k].copy() for k in ('PGC','population','dz_row','observed_radius','observed_ksmag')}
    with np.load(BASE/'r2_raw_population_fit_v1/final_fit.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['PGC'],mix['PGC'])
        pop=jnp.asarray((f['parameters']-POPULATION_ORIGIN)/POPULATION_SCALE)
        centre=float(f['richness_center'])
    options,point,_=load_train_singletons(BASE/'r2_sky_closed_split_v6/split.npz',
        include_fp_parameters=False)
    with np.load(FP,allow_pickle=False) as f:
        options=select_training_single_mark_links(options,f['membership_state'].astype(str),include_grouped=True)
        chosen=[o for p in range(6) for o in options if point['population'][o[2]]==p]
        fp_pgcs=f['PGC'].astype(np.int64,copy=True)
        np.testing.assert_array_equal(mix['PGC'],[fp_pgcs[o[3]] for o in chosen])
    group_rows,edge_rows,ledger_result=read_verified_group_ledger()
    ownership=reconcile_selected_rows(chosen,fp_pgcs,mix['PGC'],group_rows,edge_rows)
    clean_indices=np.asarray(clean_conditional_row_indices(ownership),dtype=np.int64)
    if not len(clean_indices):
        raise ValueError('association ledger leaves no admissible conditional FP rows')
    excluded_indices=np.setdiff1d(np.arange(len(ownership)),clean_indices,assume_unique=True)
    linked_radii=linked_point_conditioning_radii(chosen,point,fp_pgcs,mix['PGC'])
    prefilter_rows=len(mix['PGC'])
    mix={key:value[clean_indices] for key,value in mix.items()}
    chosen=[chosen[i] for i in clean_indices]
    linked_radii=linked_radii[clean_indices]
    mix['pre_reconciliation_rows']=prefilter_rows
    mix['excluded_unresolved_PGCs']=np.asarray([ownership[i]['fp_pgc'] for i in excluded_indices],dtype=np.int64)
    mix['source_conditioning_rule']='secure linked 2M++ point redshift; ambiguous/unresolved groups excluded from FP factor'
    mix['association_ledger_source_commit']=ledger_result['source_commit']
    with np.load(BASE/'r2_raw_selected_component_v1/training_cut_geometry.npz',allow_pickle=False) as f:
        index={int(p):i for i,p in enumerate(f['PGC'])};ids=[index[int(p)] for p in mix['PGC']]
        optical={k:f[k][ids] for k in ('x','optical_error_covariance','cut_lower','cut_upper')}
    with np.load(BASE/'r2_raw_fp_inputs_v1/training_photometry.npz',allow_pickle=False) as f:
        index={int(p):i for i,p in enumerate(f['selected_PGC'])};ids=[index[int(p)] for p in mix['PGC']]
        richness=np.log1p(f['selected_photometry'][ids,list(f['columns']).index('NgroupT17')])-centre
    voxels=np.array([np.unravel_index(point['flat_cell'][o[2]],(N,)*3) for o in chosen])
    observation={k:jnp.asarray(v) for k,v in dict(voxel=voxels,radius=mix['observed_radius'],
        dz=mix['dz_row'],ksmag=mix['observed_ksmag'],x=optical['x'],
        error_covariance=optical['optical_error_covariance'],richness=richness,
        cut_lower=optical['cut_lower'],cut_upper=optical['cut_upper'],
        source_conditioning_radius_cMpc_h=linked_radii).items()}
    geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=BOX,hubble_km_s_Mpc=74.6,
        little_h=.746,radius_table_cMpc_h=jnp.asarray(source['radial_table']),
        modulus_table_h=jnp.asarray(source['modulus_table']),redshift_table=jnp.asarray(source['redshift_table']),grid_size=N)
    return rho,velocity,tracer,pop,source,mix,observation,geometry


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        R2_complete=False,PM_evolutions=0,heldout_scored=False,optimizer_steps=0,
        limits='N128/3 development; unresolved associations excluded from conditional FP factor; no incidence/absolute-scale calibration; MW/M31 ambiguous,M33 unresolved')
    def save():
        report.update(seconds=time.monotonic()-started,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        benchmark=json.loads((BASE/'r2_cut_benchmark_v1/result.json').read_text())
        if benchmark['status']!='CUT_SHORTCUT_CHECKED_NOT_POSTERIOR':raise ValueError('actual cut comparison must finish first')
        rho,velocity,tracer,pop,source,mix,o,g=load_inputs()
        report.update(conditional_FP_rows=len(mix['PGC']),
            pre_reconciliation_conditional_FP_rows=int(mix['pre_reconciliation_rows']),
            excluded_unresolved_FP_rows=len(mix['excluded_unresolved_PGCs']),
            source_conditioning_rule=mix['source_conditioning_rule'],
            association_ledger_source_commit=mix['association_ledger_source_commit'],
            same_state_reference='old v1 CF4-radius readout is not applicable to this linked-point conditional target')
        save()
        builder=FreshRawSupport(source['positions'],source['angular'],mix['population'],o,g,source_spacing=3.)
        source={k:jnp.asarray(v) for k,v in source.items()}
        centred=jax.jit(native_mass_momentum_to_count_cells,static_argnums=2)
        def support(r,v,t):
            _,cv=centred(r,v,BOX)
            return builder.build(jnp.moveaxis(cv,0,-1).reshape(-1,3),t)
        packs,packing=support(rho,velocity,tracer)
        report.update(phase='LEGACY_FULL_NATIVE_GRADIENT',packing=packing);save()
        def target(r,v,t,p,pack,source,o,fast):
            density,cv=native_mass_momentum_to_count_cells(r,v,BOX)
            values=raw_field_logpdf(density,jnp.moveaxis(cv,0,-1).reshape(-1,3),t,p,pack,source,o,g,
                source_spacing=3.,cut_order=256 if fast else 64,cut_integration_axis=1 if fast else 0,
                cut_marginal_tolerance=1e-12 if fast else 0.,
                source_conditioning_radius_cMpc_h=o['source_conditioning_radius_cMpc_h'])
            return values.sum()-.5*(jnp.vdot(t,t)+jnp.vdot(p,p)),values
        derivative=jax.jit(jax.value_and_grad(target,argnums=(0,1,2,3),has_aux=True),static_argnums=7)
        values_only=jax.jit(target,static_argnums=7)
        gradients={};readouts={}
        for fast in (False,True):
            label='fast' if fast else 'legacy';report['phase']=label;save()
            compiled=derivative.lower(rho,velocity,tracer,pop,packs,source,o,fast).compile()
            memory=compiled.memory_analysis().temp_size_in_bytes/1024**3
            if memory>60:raise MemoryError('raw native temporary memory exceeds60GiB')
            tic=time.monotonic();(value,values),grad=compiled(rho,velocity,tracer,pop,packs,source,o)
            value=float(value);values=np.asarray(values);grad=tuple(map(np.asarray,grad))
            if not np.isfinite(values).all() or not all(np.isfinite(x).all() for x in grad):raise FloatingPointError('nonfinite raw native gradient')
            gradients[label]=grad;readouts[label]=values
            report[label]=dict(value=value,raw_logpdf_sum=float(values.sum()),
                value_gradient_seconds=time.monotonic()-tic,device_temporary_GiB=memory)
            if not fast:
                report['legacy_same_state_reference']='not compared: historical v1 used CF4 group radius and all1414 rows'
            save();print(json.dumps(report[label]),flush=True)
            del compiled
        report['legacy_fast_raw_max_abs']=float(np.max(np.abs(readouts['fast']-readouts['legacy'])))
        # Full native density, velocity and all24 nuisances varied together.
        axis=jnp.arange(N)*2*jnp.pi/N
        dr=rho*jnp.cos(axis)[:,None,None]*.03;dv=velocity*.03
        rng=np.random.default_rng(2026092908);dn=rng.normal(size=24);dn/=np.linalg.norm(dn)
        dt,dp=map(jnp.asarray,(dn[:9],dn[9:]));eps=2e-5
        fd_values=[];report['phase']='FRESH_SUPPORT_FINITE_DIFFERENCES';save()
        for sign in (1,-1):
            if time.monotonic()-started>4800:raise TimeoutError('bounded native raw profile')
            r,v,t,p=rho+sign*eps*dr,velocity+sign*eps*dv,tracer+sign*eps*dt,pop+sign*eps*dp
            fresh,info=support(r,v,t)
            value=float(values_only(r,v,t,p,fresh,source,o,True)[0]);fd_values.append(value)
            report.setdefault('finite_difference_support',[]).append(info);save()
        fd=(fd_values[0]-fd_values[1])/(2*eps)
        ad=sum(float(np.sum(a*np.asarray(b))) for a,b in zip(gradients['fast'],(dr,dv,dt,dp)))
        error=abs(fd-ad)/max(1.,abs(fd),abs(ad))
        report.update(native_direction_AD=ad,native_direction_FD=fd,native_direction_relative_error=error)
        save()
        if not np.isfinite(fd) or error>2e-5:raise AssertionError('fresh support native raw finite difference failed')
        np.savez_compressed(out/'native_raw_gradient.npz',rho=gradients['fast'][0],velocity_km_s=gradients['fast'][1],
            tracer=gradients['fast'][2],population_white=gradients['fast'][3],PGC=mix['PGC'],raw_logpdf=readouts['fast'])
        report['status']='NATIVE_RAW_TARGET_CHECKED_NOT_POSTERIOR';save()
    except Exception as error:
        report.update(status='FAILED_NATIVE_RAW_PROFILE',error=repr(error));save();raise


if __name__=='__main__':main()
