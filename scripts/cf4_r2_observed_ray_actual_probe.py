"""Registration and two-row observed-ray FP prototype; no count replacement."""
import json
import os
from pathlib import Path
import time
import resource
import jax
import jax.numpy as jnp
import numpy as np
from cf4_actual_selection import base
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_linked_fp_sparse_train import load_train_singletons,select_training_single_mark_links,FP,POINTS
from cf4_r2_native_to_count_cells import native_moments_to_count_cells
from cf4_r2_raw_volume_target import tracer_geometry,tracer_masses
from cf4_r2_observed_ray import observed_ray_components
from cf4_r2_raw_live_mark import chunk_log_terms,POPULATION_ORIGIN,POPULATION_SCALE

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic()
    _,_,_,_,source,mix,o,g=load_inputs()
    options,point,_=load_train_singletons(BASE/'r2_sky_closed_split_v6/split.npz')
    with np.load(FP,allow_pickle=False) as f:
        options=select_training_single_mark_links(options,f['membership_state'].astype(str),include_grouped=True)
        chosen=[a for p in range(6) for a in options if point['population'][a[2]]==p]
        np.testing.assert_array_equal(mix['PGC'],[f['PGC'][a[3]] for a in chosen])
    with np.load(POINTS,allow_pickle=False) as f:
        direction=base.supergalactic_unit_vectors(f['ra_deg'],f['dec_deg'])
        rpoint=f['radius_cMpc_h'];flat=f['flat_cell'];pop=f['population']
    rebuilt=np.floor((192.+rpoint[:,None]*direction)/3.).astype(int)
    rebuilt_flat=np.ravel_multi_index(rebuilt.T,(128,)*3)
    with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
        training=np.isin(pop.astype(np.int64)*128**3+flat,f['train_keys'])
    if training.sum()!=47121:raise ValueError('training point ownership changed')
    failures=np.flatnonzero(training&(rebuilt_flat!=flat))
    with np.load(BASE/'r2_raw_joint_pilot_v1/accepted_present_state.npz',allow_pickle=False) as f:
        rho,v,var=map(jnp.asarray,(f['rho'],f['mean_velocity_km_s'],f['physical_velocity_variance_km2_s2']))
        old=f['tracer'];tracer=jnp.asarray(np.r_[old[:6],0.,.12*np.exp(.5*old[7]),old[8]])
        params=jnp.asarray(POPULATION_ORIGIN)+jnp.asarray(POPULATION_SCALE)*jnp.asarray(f['population_white'])
    rho,v,var=jax.jit(native_moments_to_count_cells,static_argnums=3)(rho,v,var,384.)
    v=jnp.moveaxis(v,0,-1).reshape(-1,3);var=jnp.moveaxis(var,0,-1).reshape(-1,3)
    mass=tracer_masses(rho,tracer);sky=jnp.asarray(source['angular']);geometry=tracer_geometry(tracer,g)
    vmax=float(jnp.max(jnp.linalg.norm(v,axis=1)))
    sigma_bound=float(jnp.sqrt(30.**2+.5**2*jnp.max(var)))
    report=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        R2_complete=False,PM_evolutions=0,heldout_scored=False,training_points=int(training.sum()),
        coordinate_mismatches=failures.tolist(),rows=[],
        count_backend_changed=False,within_voxel_angular_density_scored=False,
        limitations='Fixed-direction conditional optical FP/K prototype. No extra observed-redshift likelihood. No full periodic production or calibrated closure prior.')
    def save():
        report.update(seconds=time.monotonic()-started,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    if len(failures):raise AssertionError('actual training voxel/direction registration mismatch')
    for row in (2,76):
        angle=jnp.asarray(direction[chosen[row][2]]);obs={k:val[row] for k,val in o.items()}
        np.testing.assert_array_equal(rebuilt[chosen[row][2]],np.asarray(obs['voxel']))
        result=dict(row=row,PGC=int(mix['PGC'][row]),rules=[],
            no_wrap_sufficient_margin_cMpc_h=192.-float(obs['radius'])-.01*(vmax+8*sigma_bound))
        for order in (4,8):
            def score(logscale,vel,variance,masses,angular):
                closure=dict(core_sigma_km_s=30.,dispersion_scale=.5*jnp.exp(logscale),broad_fraction=.5)
                positions,velocity,intrinsic,selection,weights=observed_ray_components(angle,obs['radius'],
                    vel,variance,masses,angular,int(mix['population'][row]),geometry,closure,source_grid=128,order=order)
                a,b=chunk_log_terms(params,positions,velocity,intrinsic,selection,obs,
                    population=int(mix['population'][row]),geometry=geometry,cut_order=64,
                    radial_source_mass=weights)
                return a-b,weights.sum()
            derivative=jax.jit(jax.value_and_grad(score,argnums=0,has_aux=True))
            tic=time.monotonic();(value,weight),grad=derivative(0.,v,var,mass,sky)
            value,weight,grad=map(float,(value,weight,grad))
            if not np.isfinite([value,weight,grad]).all() or weight<=0:raise ValueError('nonfinite ray prototype')
            eps=1e-4;forward=jax.jit(score)
            fd=float((forward(eps,v,var,mass,sky)[0]-forward(-eps,v,var,mass,sky)[0])/(2*eps))
            err=abs(grad-fd)/max(1.,abs(grad),abs(fd))
            if err>2e-5:raise AssertionError('ray optical FP derivative mismatch')
            result['rules'].append(dict(source_radius_order=order,raw_FP_logpdf=value,
                radial_selected_weight=weight,logscale_AD=grad,logscale_FD=fd,
                normalized_derivative_error=err,seconds=time.monotonic()-tic))
        report['rows'].append(result);save();print(json.dumps(result),flush=True)
    report['status']='ACTUAL_OBSERVED_RAY_FP_PROTOTYPE_NOT_POSTERIOR';save()


if __name__=='__main__':main()
