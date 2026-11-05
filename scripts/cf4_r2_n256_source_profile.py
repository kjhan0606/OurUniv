"""N256 source-workspace profile of a lifted N128 field, NOT new resolution."""
import json
import os
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_raw_volume_target import tracer_masses,tracer_geometry
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_count_statistics import count_statistic_layout,poisson_from_statistics
from cf4_r2_count_statistic_integral import volume_count_statistics

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic()
    compact=os.environ.get('CF4_R2_COUNT_STATS')=='1'
    # Compressed output leaves room for larger source batches. Avoid paying
    # hundreds of nested shell scans merely to minimize already-small memory.
    # The compiled20% device margin below remains mandatory before execution.
    chunk=int(os.environ.get('CF4_R2_SOURCE_CHUNK',1048576 if compact else 32768))
    if chunk<1:raise ValueError('positive explicit source batch required')
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        R2_complete=False,PM_evolutions=0,heldout_scored=False,field_information_resolution_cMpc_h=3.,
        source_integration_grid=256,observed_count_grid=128,source_chunk=chunk,compressed_readout=compact,
        source_cell_mass_factor=.125,
        limits='same N128 piecewise-constant field replicated, not a new N256 dynamical state or posterior; no LG identities')
    def save():
        report.update(seconds=time.monotonic()-started,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if compact and json.loads((BASE/'r2_count_statistic_profile_v1/result.json').read_text())['status']!='COMPRESSED_COUNT_NATIVE_REFERENCE_MATCHED_NOT_POSTERIOR':
            raise ValueError('full native reference comparison required before compressed N256 profile')
        r,v,tr,_,old,_,_,geometry=load_inputs()
        coarse_coords=(np.indices((128,)*3,dtype=np.int16).reshape(3,-1).T.astype(np.float32)+.5)*3.
        np.testing.assert_array_equal(old['positions'],coarse_coords)
        del coarse_coords
        density,velocity=jax.jit(native_mass_momentum_to_count_cells,static_argnums=2)(r,v,384.)
        def repeat(a,axes):
            for axis in axes:a=jnp.repeat(a,2,axis=axis)
            return a
        rho=repeat(density,(0,1,2));vel=jnp.moveaxis(repeat(velocity,(1,2,3)),0,-1).reshape(-1,3)
        sky=repeat(jnp.asarray(old['angular']).reshape(2,128,128,128),(1,2,3)).reshape(2,-1)
        coords=np.indices((256,)*3,dtype=np.int16).reshape(3,-1).T.astype(np.float32)
        positions=jnp.asarray((coords+.5)*1.5);del coords
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
            keys,counts=map(jnp.asarray,(f['train_keys'],f['train_counts']))
            exposure,_=build_population_exposure_masks(128,f['heldout_flat_voxels'],
                f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        exposure=jnp.asarray(exposure)
        layout=count_statistic_layout(128,np.asarray(keys),np.asarray(exposure)) if compact else None
        # Global source count increased8x: retain the SAME physical rate prior.
        old_total=float(tracer_masses(density,tr).sum())
        new_total=float((tracer_masses(rho,tr)*.125).sum())
        report['intrinsic_mass_ratio']=new_total/old_total;save()
        if abs(new_total/old_total-1)>1e-10:raise AssertionError('source refinement changed physical rate')
        def target(rho,vel,tr,pos,sky):
            masses=tracer_masses(rho,tr)*.125
            if compact:
                statistics=volume_count_statistics(pos,vel,masses,sky,layout,len(keys)+1,source_chunk_size=chunk,
                    source_spacing=1.5,volume_order=2,**tracer_geometry(tr,geometry),order=4,segments=8)
                return poisson_from_statistics(statistics,counts),statistics[-1]
            intensity=predict_chunked_volume_intensity(pos,vel,masses,sky,source_chunk_size=chunk,
                source_spacing=1.5,volume_order=2,**tracer_geometry(tr,geometry),order=4,segments=8)
            score=sparse_marked_poisson_log_likelihood(intensity,keys,counts,selected_voxel_mask=exposure)
            return score,jnp.sum(intensity.reshape(-1)*exposure)
        report['phase']='COMPILE_FULL_SOURCE_ADJOINT';save()
        derivative=jax.jit(jax.value_and_grad(target,argnums=(0,1,2),has_aux=True))
        compiled=derivative.lower(rho,vel,tr,positions,sky).compile()
        m=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
        peak=stats.get('bytes_in_use',0)+m.temp_size_in_bytes+m.output_size_in_bytes
        report.update(device_temporary_GiB=m.temp_size_in_bytes/1024**3,estimated_peak_GiB=peak/1024**3,
            phase='FULL_SOURCE_ADJOINT');save()
        if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:raise MemoryError('source adjoint lacks20percent device margin')
        tic=time.monotonic();(score,expected),(gr,gv,gt)=compiled(rho,vel,tr,positions,sky)
        score=float(score);expected=float(expected)
        finite=bool(jnp.isfinite(gr).all()&jnp.isfinite(gv).all()&jnp.isfinite(gt).all())
        report.update(score=score,expected_training_count=expected,full_gradient_seconds=time.monotonic()-tic,
            rate_gradient=float(gt[0]),analytic_rate_gradient=2*(float(counts.sum())-expected),
            density_normalization_direction=float(jnp.sum(gr*rho)))
        save()
        if not finite or not np.isfinite(score):raise FloatingPointError('nonfinite source profile')
        if abs(report['rate_gradient']-report['analytic_rate_gradient'])>1e-6:
            raise AssertionError('global rate derivative changed through source chunks')
        if abs(report['density_normalization_direction'])>1e-6:
            raise AssertionError('chunked target lost global density normalization')
        # No full gradient archive: this is resource evidence, not an IC state.
        report['status']='N256_SOURCE_WORKSPACE_PROFILE_NOT_N256_FIELD';save()
    except Exception as error:
        report.update(status='FAILED_N256_SOURCE_PROFILE',error=repr(error));save();raise


if __name__=='__main__':main()
