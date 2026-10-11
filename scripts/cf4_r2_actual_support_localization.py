"""Two failed training rows, ALL source cells; no fit, prior or support floor."""
import json
import os
from pathlib import Path
import time
import resource
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_native_to_count_cells import native_moments_to_count_cells
from cf4_r2_raw_volume_target import tracer_geometry,tracer_masses,volume_rule
from cf4_r2_marked_tracer_jax import predict_source_marked_radial_key_density

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);start=time.monotonic()
    _,_,_,_,source,mix,o,g=load_inputs()
    with np.load(BASE/'r2_raw_joint_pilot_v1/accepted_present_state.npz',allow_pickle=False) as f:
        rho,v,var=map(jnp.asarray,(f['rho'],f['mean_velocity_km_s'],f['physical_velocity_variance_km2_s2']))
        old=f['tracer'];tracer=jnp.asarray(np.r_[old[:6],0.,.12*np.exp(.5*old[7]),old[8]])
    rho,v,var=jax.jit(native_moments_to_count_cells,static_argnums=3)(rho,v,var,384.)
    v=jnp.moveaxis(v,0,-1).reshape(-1,3);var=jnp.moveaxis(var,0,-1).reshape(-1,3)
    masses=tracer_masses(rho,tracer);pos=jnp.asarray(source['positions']);sky=jnp.asarray(source['angular'])
    report=dict(job_id=os.environ['SLURM_JOB_ID'],status='RUNNING',R2_complete=False,
        source_commit=os.environ['CF4_EXPECTED_COMMIT'],PM_evolutions=0,heldout_scored=False,
        diagnostic_only=True,all_source_cells=len(pos),rows=[])
    def save():
        report.update(seconds=time.monotonic()-start,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    for row in (2,76):
        population=int(mix['population'][row]);voxel=o['voxel'][row];radius=o['radius'][row]
        def read(offset,vel,variance,mass,angular):
            position=(pos+offset)%384.
            base=dict(tracer_geometry(tracer,g),sigma_los_km_s=30.)
            outputs=[]
            for scale in (0.,.5):
                physical=dict(base,source_velocity_variances_km2_s2=variance,dispersion_scale=scale)
                for tail in (8.,float('inf')):
                    value=predict_source_marked_radial_key_density(position,vel,mass,angular,population,
                        voxel,radius,**physical,deposition='voxel_cdf',radial_tail_sigma=tail)
                    outputs.append(jnp.array([value.sum(),jnp.sum(value.sum(axis=0)>0)]))
            legacy=predict_source_marked_radial_key_density(position,vel,mass,angular,population,
                voxel,radius,**base,deposition='tsc')
            outputs.append(jnp.array([legacy.sum(),jnp.sum(legacy.sum(axis=0)>0)]))
            return jnp.stack(outputs)
        read=jax.jit(read)
        result=dict(row=row,PGC=int(mix['PGC'][row]),population=population,
            observed_radius=float(radius),voxel=np.asarray(voxel).tolist(),rules=[])
        # Two existing rules only, not a quadrature escalation ladder.
        for order in (1,2):
            offsets,weights=volume_rule(3.,order);total=np.zeros((5,2));tic=time.monotonic()
            for offset,weight in zip(offsets,weights):
                value=np.asarray(read(jnp.asarray(offset),v,var,masses,sky))
                total[:,0]+=weight*value[:,0];total[:,1]+=value[:,1]
            result['rules'].append(dict(volume_order=order,
                labels=['core_8sigma','core_untruncated','broad_8sigma','broad_untruncated','legacy_TSC_core30'],
                source_radial_weight_and_positive_nodes=total.tolist(),seconds=time.monotonic()-tic))
            save()
        report['rows'].append(result);save();print(json.dumps(result),flush=True)
    report['status']='ACTUAL_ALL_SOURCE_SUPPORT_LOCALIZED_NOT_POSTERIOR';save()


if __name__=='__main__':main()
