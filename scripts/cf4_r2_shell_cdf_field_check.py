"""Saved-field count-only integration check; no PM evolution/fit/heldout."""
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
from cf4_r2_marked_tracer_jax import (
    intrinsic_biased_source_masses,intrinsic_lf_bin_fractions,
    sparse_marked_poisson_log_likelihood,predict_source_marked_intensity_los_node)
from cf4_2mpp_joint_likelihood_jax import _gaussian_hermite_rule
from cf4_r2_shell_cdf_count import predict_shell_cdf_intensity

BASE=Path('/gpfs/kjhan/CF4/z0_density')
OUT=Path(os.environ.get('CF4_R2_OUT_DIR',str(BASE/'r2_shell_cdf_field_check_v1')))
REFERENCE_STATE=BASE/'r2_v6_partial_map_v2/final_state.npz'


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    OUT.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(status='STARTED',source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        job_id=os.environ['SLURM_JOB_ID'],R2_complete=False,heldout_scored=False,
        PM_evolutions=0,optimizer_steps=0,
        state=str(Path(os.environ.get('CF4_R2_CHECK_STATE',str(REFERENCE_STATE))).resolve()),
        comparisons={})
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,
                      host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        with np.load(BASE/report['state']) as f:
            rho,vel,tracer=map(jnp.asarray,(f['rho'],f['velocity_km_s'],f['tracer']))
        with np.load(SOURCE) as f:
            source={k:jnp.asarray(f[k]) for k in f.files}
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz') as f:
            keys,counts=map(jnp.asarray,(f['train_keys'],f['train_counts']))
            exposure,_=build_population_exposure_masks(128,f['heldout_flat_voxels'],
                f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        exposure=jnp.asarray(exposure)
        density,vel=native_mass_momentum_to_count_cells(rho,vel,384.)
        vel=jnp.moveaxis(vel,0,-1).reshape(-1,3)
        sigma=100*jnp.exp(.5*tracer[6])
        if not 0<8*.01*float(sigma)<192:
            raise ValueError('outside bounded 27-image support, no clipping')
        intrinsic=intrinsic_biased_source_masses(density,
            jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tracer[0],
            jnp.exp(.5*tracer[1:6]),alpha=-1+.06*jnp.exp(.5*tracer[7]),
            mstar=-23.28+.2*tracer[8],reference_interval=(-25.,-21.))
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=source['radial_table'],
            modulus_table_h=source['modulus_table'],redshift_table=source['redshift_table'],
            grid_size=128,sigma_los_km_s=sigma,radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,
            mstar=-23.28+.2*tracer[8],alpha=-1+.06*jnp.exp(.5*tracer[7]))
        def score(intensity):
            return sparse_marked_poisson_log_likelihood(intensity,keys,counts,
                                                       selected_voxel_mask=exposure)
        @jax.jit
        def gh15(velocity,masses):
            nodes,weights=_gaussian_hermite_rule(15)
            def add(total,nw):
                node,weight=nw
                return total+predict_source_marked_intensity_los_node(source['positions'],
                    velocity,masses,source['angular'],node,weight,**geometry),None
            return jax.lax.scan(add,jnp.zeros((6,128,128,128)),
                                (jnp.asarray(nodes),jnp.asarray(weights)))[0]
        def cdf(scale,velocity,masses,order,segments):
            return predict_shell_cdf_intensity(source['positions'],scale*velocity,masses,
                source['angular'],order=order,segments=segments,**geometry)
        evaluate=jax.jit(cdf,static_argnums=(3,4))
        stratified=os.environ.get('CF4_R2_CDF_STRATIFIED')=='1'
        rules=(('CDF4x16',4,16),('CDF4x32',4,32)) if stratified else (
               ('CDF16',16,1),('CDF32',32,1))
        if os.environ.get('CF4_R2_CDF_FINE')=='1':
            rules=(('CDF4x32',4,32),('CDF8x32',8,32))
        low_name,low_order,low_segments=rules[0]
        high_name,_,_=rules[1]
        fields={}
        calls=[('GH15',lambda:gh15(vel,intrinsic))]+[
            (name,lambda order=order,segments=segments:evaluate(1.,vel,intrinsic,order,segments))
            for name,order,segments in rules]
        for name,call in calls:
            tic=time.monotonic()
            fields[name]=call(); fields[name].block_until_ready()
            value=float(score(fields[name]))
            if not np.isfinite(value):
                raise FloatingPointError(f'{name} count score not finite')
            row=dict(score=value,seconds=time.monotonic()-tic,
                     expected_training_count=float(jnp.sum(fields[name].reshape(-1)*exposure)))
            report['comparisons'][name]=row
            save(); print(json.dumps({name:row}),flush=True)
        for left,right in [('GH15',high_name),(low_name,high_name)]:
            a,b=fields[left].reshape(-1),fields[right].reshape(-1)
            report[f'{left}_vs_{right}']=dict(
                exposure_L1_relative=float(jnp.sum(jnp.abs(a-b)*exposure)/jnp.sum(b*exposure)),
                max_occupied_log_difference=float(jnp.max(jnp.abs(jnp.log(a[keys])-jnp.log(b[keys])))),
                count_score_difference=report['comparisons'][left]['score']-report['comparisons'][right]['score'])
        if (os.environ.get('CF4_R2_CDF_FINE')=='1'
                and Path(report['state'])==REFERENCE_STATE.resolve()):
            previous=json.loads((BASE/'r2_shell_cdf_field_check_v4/result.json').read_text())
            deltas={name:report['comparisons'][name]['score']-
                    previous['comparisons'][name]['score'] for name in ('GH15','CDF4x32')}
            report['LF_reuse_previous_score_differences']=deltas
            save()
            if any(abs(d)>1e-7 for d in deltas.values()):
                raise AssertionError('LF algebra reuse changed the saved-field score')
        save()
        derivative=jax.jit(jax.grad(lambda scale,v,m:score(cdf(scale,v,m,low_order,low_segments))))
        compiled=derivative.lower(1.,vel,intrinsic).compile()
        analysis=compiled.memory_analysis()
        device_stats=jax.devices()[0].memory_stats() or {}
        if analysis is not None:
            report['derivative_memory']=dict(temporary_GiB=analysis.temp_size_in_bytes/1024**3,
                arguments_GiB=analysis.argument_size_in_bytes/1024**3,
                outputs_GiB=analysis.output_size_in_bytes/1024**3,
                current_device_GiB=device_stats.get('bytes_in_use',0)/1024**3,
                device_limit_GiB=device_stats.get('bytes_limit',0)/1024**3)
            save()
            peak=device_stats.get('bytes_in_use',0)+analysis.temp_size_in_bytes+analysis.output_size_in_bytes
            if device_stats.get('bytes_limit') and 1.2*peak>device_stats['bytes_limit']:
                raise MemoryError('compiled derivative lacks 20 percent device-memory margin')
        reverse=float(compiled(1.,vel,intrinsic))
        epsilon=1e-6
        finite=float((score(evaluate(1.+epsilon,vel,intrinsic,low_order,low_segments))-
                      score(evaluate(1.-epsilon,vel,intrinsic,low_order,low_segments)))/(2*epsilon))
        error=abs(reverse-finite)/max(1.,abs(reverse),abs(finite))
        report['velocity_scale_adjoint']=dict(rule=low_name,epsilon=epsilon,reverse=reverse,
                                             finite_difference=finite,relative_discrepancy=error)
        report['status']='SAVED_FIELD_COMPARISON_COMPLETE_NOT_POSTERIOR'
        save(); print(json.dumps(report,allow_nan=False),flush=True)
    except Exception as exc:
        report.update(status='STOPPED_NOT_POSTERIOR',error=f'{type(exc).__name__}: {exc}')
        save(); raise


if __name__=='__main__':
    main()
