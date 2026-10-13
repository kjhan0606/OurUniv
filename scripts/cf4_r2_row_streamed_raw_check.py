"""Actual1414-row N256 GL2 streaming equivalence; no new dynamics or fit."""
import json
import os
from pathlib import Path
import resource
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import source_geometry_at_resolution
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_row_streamed_raw import RowStreamedRawReadout

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],R2_complete=False,
        purpose='row-streamed numerical sensitivity preparation; no posterior promotion',
        source_volume_order=2,heldout_scored=False,PM_evolutions=0)
    def save():(out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        parent=BASE/'r2_n256_affine_pilot_v1'
        reference=json.loads((parent/'result.json').read_text())
        if reference['status']!='N256_RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR' or not reference['trace'][-1]['accepted']:
            raise ValueError('same-state terminal fine readout required')
        expected=reference['evaluations'][-1]['parts'][1]
        _,_,_,_,source,mix,o,g=load_inputs();source=source_geometry_at_resolution(source,256)
        with np.load(parent/'accepted_present_state.npz',allow_pickle=False) as f:
            rho,velocity,tracer,pop=map(jnp.asarray,(f['rho'],f['mean_velocity_km_s'],f['tracer'],f['population_white']))
        density,velocity=jax.jit(native_mass_momentum_to_count_cells,static_argnums=2)(rho,velocity,384.)
        velocity=jnp.moveaxis(velocity,0,-1).reshape(-1,3)
        readout=RowStreamedRawReadout(source,mix['population'],o,g,source_spacing=1.5,
            volume_order=2,rows_per_batch=64)
        def progress(row):
            report['completed_rows']=row['stop'];save();print(json.dumps(row),flush=True)
        values,info=readout.evaluate(density,velocity,tracer,pop,progress=progress)
        actual=float(values.sum());error=abs(actual-expected)
        report.update(actual_raw_score=actual,reference_raw_score=expected,absolute_error=error,
            rows=len(values),**info,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        if len(values)!=1414 or error>1e-7:raise AssertionError('row-streamed target differs from full-cohort target')
        np.savez(out/'readout.npz',PGC=mix['PGC'],raw_logpdf=values)
        report['status']='N256_ROW_STREAMING_VALUE_CHECKED_NOT_GL4_SENSITIVITY';save()
    except Exception as error:
        report.update(status='FAILED_ROW_STREAMING_CHECK',error=repr(error));save();raise


if __name__=='__main__':main()
