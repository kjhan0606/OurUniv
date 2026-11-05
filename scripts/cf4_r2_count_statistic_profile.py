"""Same-state full native-gradient comparison of compressed Poisson readout."""
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
from cf4_r2_count_statistics import count_statistic_layout,poisson_from_statistics
from cf4_r2_count_statistic_integral import volume_count_statistics
from cf4_r2_count_exposure import build_population_exposure_masks

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        R2_complete=False,PM_evolutions=0,heldout_scored=False,volume_order=4,source_chunk=131072,
        limits='exact output projection of the existing numerical law, not new posterior or calibration')
    def save():
        report.update(seconds=time.monotonic()-started,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        rho,vel,tr,_,source,_,_,geometry=load_inputs()
        source={k:jnp.asarray(v) for k,v in source.items()}
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
            keys=f['train_keys'];counts=jnp.asarray(f['train_counts'])
            exposure,_=build_population_exposure_masks(128,f['heldout_flat_voxels'],f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        layout=count_statistic_layout(128,keys,exposure);size=len(keys)+1
        report.update(occupied_keys=len(keys),output_compression_ratio=6*128**3/size);save()
        def target(r,v,t,pos,sky,layout):
            density,cv=native_mass_momentum_to_count_cells(r,v,384.)
            statistics=volume_count_statistics(pos,jnp.moveaxis(cv,0,-1).reshape(-1,3),tracer_masses(density,t),
                sky,layout,size,source_spacing=3.,volume_order=4,source_chunk_size=131072,
                **tracer_geometry(t,geometry),order=4,segments=8)
            return poisson_from_statistics(statistics,counts),statistics[-1]
        f=jax.jit(jax.value_and_grad(target,argnums=(0,1,2),has_aux=True))
        report['phase']='COMPILING';save()
        compiled=f.lower(rho,vel,tr,source['positions'],source['angular'],layout).compile()
        memory=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
        peak=stats.get('bytes_in_use',0)+memory.temp_size_in_bytes+memory.output_size_in_bytes
        report.update(phase='FULL_NATIVE_GRADIENT',device_temporary_GiB=memory.temp_size_in_bytes/1024**3,
            estimated_peak_GiB=peak/1024**3);save()
        if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:raise MemoryError('20percent device margin required')
        tic=time.monotonic();(value,expected),grad=compiled(rho,vel,tr,source['positions'],source['angular'],layout)
        value=float(value);expected=float(expected);grad=tuple(map(np.asarray,grad))
        report.update(score=value,expected_training_count=expected,full_gradient_seconds=time.monotonic()-tic)
        reference=json.loads((BASE/'r2_volume_count_profile_v1/result.json').read_text())
        if reference['status']!='FULL_COUNT_VOLUME_PROFILE_NOT_POSTERIOR':raise ValueError('completed full-grid reference required')
        ref=reference['rules'][1]
        with np.load(BASE/'r2_volume_count_profile_v1/fine_native_gradients.npz',allow_pickle=False) as f:
            errors={k:float(np.max(np.abs(a-f[k]))) for k,a in zip(('rho','velocity_km_s','tracer'),grad)}
        report.update(score_abs_error=abs(value-ref['score']),expected_count_abs_error=abs(expected-ref['expected_training_count']),
            gradient_max_abs_errors=errors)
        save()
        if not np.isfinite(np.r_[value,expected,*errors.values()]).all() or max(
                report['score_abs_error'],report['expected_count_abs_error'],*errors.values())>1e-7:
            raise AssertionError('compressed count law differs from full-grid native reference')
        report['status']='COMPRESSED_COUNT_NATIVE_REFERENCE_MATCHED_NOT_POSTERIOR';save()
    except Exception as error:
        report.update(status='FAILED_COMPRESSED_COUNT_PROFILE',error=repr(error));save();raise


if __name__=='__main__':main()
