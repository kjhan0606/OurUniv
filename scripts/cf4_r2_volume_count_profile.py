"""Actual full-training count-volume target and native-field gradient cost."""
import json
import os
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_linked_fp_sparse_train import SOURCE
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_marked_tracer_jax import (intrinsic_biased_source_masses,
    intrinsic_lf_bin_fractions,sparse_marked_poisson_log_likelihood)
from cf4_r2_shell_cdf_count import predict_source_volume_intensity

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        PM_evolutions=0,optimizer_steps=0,heldout_scored=False,R2_complete=False,rules=[])
    def save():
        report.update(seconds=time.monotonic()-started,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        with np.load(BASE/'r2_prior_split_long_v1/final_state.npz',allow_pickle=False) as f:
            rho,velocity,tracer=map(jnp.asarray,(f['rho'],f['velocity_km_s'],f['tracer']))
        with np.load(SOURCE,allow_pickle=False) as f:source={k:jnp.asarray(f[k]) for k in f.files}
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
            keys,counts=map(jnp.asarray,(f['train_keys'],f['train_counts']))
            exposure,_=build_population_exposure_masks(128,f['heldout_flat_voxels'],
                f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        exposure=jnp.asarray(exposure)
        def target(rho,velocity,tr,volume_order):
            density,v=native_mass_momentum_to_count_cells(rho,velocity,384.)
            mstar=-23.28+.2*tr[8];alpha=-1+.06*jnp.exp(.5*tr[7])
            masses=intrinsic_biased_source_masses(density,
                jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tr[0],jnp.exp(.5*tr[1:6]),
                mstar=mstar,alpha=alpha,reference_interval=(-25.,-21.))
            intensity=predict_source_volume_intensity(source['positions'],
                jnp.moveaxis(v,0,-1).reshape(-1,3),masses,source['angular'],
                source_spacing=3.,volume_order=volume_order,observer=jnp.full(3,192.),
                box_size_cMpc_h=384.,hubble_km_s_Mpc=74.6,little_h=.746,
                radius_table_cMpc_h=source['radial_table'],modulus_table_h=source['modulus_table'],
                redshift_table=source['redshift_table'],grid_size=128,
                sigma_los_km_s=100*jnp.exp(.5*tr[6]),mstar=mstar,alpha=alpha,order=4,segments=8)
            value=sparse_marked_poisson_log_likelihood(intensity,keys,counts,selected_voxel_mask=exposure)
            return value,jnp.sum(intensity.reshape(-1)*exposure)
        evaluate=jax.jit(target,static_argnums=3)
        derivative=jax.jit(jax.value_and_grad(target,argnums=(0,1,2),has_aux=True),static_argnums=3)
        for order in (2,4):
            if time.monotonic()-started>2200:raise TimeoutError('bounded full-count profiling')
            compiled=derivative.lower(rho,velocity,tracer,order).compile()
            memory=compiled.memory_analysis().temp_size_in_bytes/1024**3
            entry=dict(source_nodes_per_axis=order,LOS_order=4,LOS_segments=8,device_temporary_GiB=memory)
            report['rules'].append(entry);save()
            if memory>60:raise MemoryError('full volume adjoint temporary>60GiB')
            tic=time.monotonic()
            (score,expected),(gr,gv,gt)=compiled(rho,velocity,tracer)
            score=float(score);expected=float(expected)
            gr,gv,gt=map(np.asarray,(gr,gv,gt))
            entry.update(score=score,expected_training_count=expected,full_gradient_seconds=time.monotonic()-tic)
            if not np.isfinite(score) or not all(np.isfinite(g).all() for g in (gr,gv,gt)):
                raise FloatingPointError('nonfinite full count score/gradient')
            # One physical field direction; native velocity, not only readout velocity.
            analytic=float(np.sum(gv*np.asarray(velocity)));eps=1e-5
            tic=time.monotonic()
            plus=float(evaluate(rho,(1+eps)*velocity,tracer,order)[0])
            minus=float(evaluate(rho,(1-eps)*velocity,tracer,order)[0])
            fd=(plus-minus)/(2*eps)
            entry.update(two_forward_including_compile_seconds=time.monotonic()-tic,
                velocity_direction_analytic=analytic,velocity_direction_finite_difference=fd,
                velocity_direction_relative_error=abs(fd-analytic)/max(1.,abs(fd),abs(analytic)),
                tracer_gradient=gt.tolist())
            save();print(json.dumps(entry),flush=True)
            if entry['velocity_direction_relative_error']>2e-5:
                raise AssertionError('native-field count derivative mismatch')
            if order==4:
                np.savez_compressed(out/'fine_native_gradients.npz',rho=gr,velocity_km_s=gv,tracer=gt)
        report['fine_minus_coarse_score']=report['rules'][1]['score']-report['rules'][0]['score']
        report['status']='FULL_COUNT_VOLUME_PROFILE_NOT_POSTERIOR';save()
    except Exception as error:
        report.update(status='FAILED_FULL_COUNT_VOLUME_PROFILE',error=repr(error));save();raise


if __name__=='__main__':main()
