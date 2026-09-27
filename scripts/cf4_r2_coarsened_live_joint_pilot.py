"""One actual-data same-IC count + conditional-mark gradient control."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cf4_r1_particle_forward import make_dynamics,particle_grid
from cf4_r2_coarsened_live_joint import count_and_mark_parts

BASE=Path('/gpfs/kjhan/CF4/z0_density')
OUT=BASE/'r2_coarsened_live_joint_v1'
N,BOX=128,384.


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    start=time.monotonic()
    geometry_path=BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
    count_path=BASE/'r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz'
    point_path=BASE/'r2_point_mark_manifest_v1/points.npz'
    selection_path=BASE/'r2_common_selection_128_v1/selection_3.h5'
    with np.load(geometry_path,allow_pickle=False) as f:
        g={k:jnp.asarray(f[k]) for k in f.files if k not in ('method_names','group_labels')}
    with np.load(count_path,allow_pickle=False) as f:
        keys=np.asarray(f['parent_keys'],dtype=np.int32)
        counts=np.asarray(f['parent_counts'],dtype=np.int32)
    with np.load(point_path,allow_pickle=False) as f:
        point_keys=f['population'].astype(np.int64)*N**3+f['flat_cell']
        point_zero=int(f['map_zero'].sum())
        point_count=len(point_keys)
    projected_key,projected_count=np.unique(point_keys,return_counts=True)
    np.testing.assert_array_equal(projected_key,keys)
    np.testing.assert_array_equal(projected_count,counts)
    if len(keys)!=45776 or point_count!=57238 or point_zero!=1:
        raise ValueError('inclusive point/count ownership changed')
    with h5py.File(selection_path,'r') as f:
        if f.attrs['status']!='INTEGRATION_COMPLETE_NOT_CALIBRATED' or f['selection_shells'].shape!=(6,6,N,N,N):
            raise ValueError('selection geometry/status mismatch')
        exposure=np.empty((6,N,N,N),dtype=np.float64)
        for lo in range(0,N,4):
            exposure[:,lo:lo+4]=f['selection_shells'][:,:,lo:lo+4].sum(axis=1,dtype=np.float64)
    if not np.isfinite(exposure).all() or np.any(exposure<0) or np.any(exposure.reshape(-1)[keys]<=0):
        raise ValueError('invalid observed count support in integrated selection')
    cfg=json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    common,published=cfg['common_cosmology'],cfg['published_prior']
    nbar=np.asarray(published['original_mean_count_per_cell_bright_first'],float)*(
        BOX/N/published['original_cell_cMpc_h'])**3
    bias=jnp.asarray(published['linear_regime_bias_bright_first'])
    rate_cfg=json.loads((ROOT/'config/cf4_r2_rate_nuisance_v1.json').read_text())
    shape=jnp.full(6,rate_cfg['rate_shape'])
    frozen=json.loads((ROOT/'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())['inference_model']
    fog=jnp.asarray(frozen['FoG_prior_median_km_s'])
    redshift=jnp.asarray(frozen['fixed_redshift_error_km_s'])
    sd=jnp.asarray([.004,1.,1.,1.,1.,2.])
    settings0=json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings={k:settings0[k] for k in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
    settings.update(n=N,box_cMpc_h=BOX)
    if settings['cosmology']['h']!=common['h'] or settings['cosmology']['Om']!=common['Omega_m']:
        raise ValueError('PM/observation cosmology mismatch')
    size=N**3
    nh=len(sd)+2
    ng=len(g['train_group_index'])
    e=jnp.asarray(exposure)
    k=jnp.asarray(keys)
    c=jnp.asarray(counts)
    rate_mean=jnp.asarray(nbar)
    sources=[geometry_path,count_path,point_path,selection_path,
             ROOT/'config/cf4_r2_common_cosmology_v1.json',
             ROOT/'config/cf4_r2_rate_nuisance_v1.json']
    report=dict(classification='R2_COARSENED_COUNTS_AND_CONDITIONAL_CF4_MARKS_ONE_LIVE_IC',
        job_id=os.environ['SLURM_JOB_ID'],status='STARTED',N=N,box_cMpc_h=BOX,
        native_PM_origin_fraction=0.,observed_count_voxel_origin_fraction=.5,
        observed_2mpp_points=point_count,occupied_population_cells=len(keys),
        literal_pointwise_map_zero=point_zero,source_groups=int(g['distance'].shape[0]),
        training_source_groups=ng,source_FP_rows=len(g['row_group']),
        nonFP_rows=len(g['anchor_group']),joint_redshift_member_count=int(jnp.sum(g['twompp_member_count'])),
        inclusion_and_association_law_calibrated=False,within_cell_law_field_independent_assumption=True,
        historical_survival_or_pointwise_map_applied=False,observed_count_rate_renormalization=False,
        full_joint_survey_likelihood=False,posterior_sample=False,N256_or_LG_delivered=False,
        count_rate_prior=dict(family='Gamma',shape=float(shape[0]),prior_mean=nbar.tolist(),calibrated=False),
        source_paths=[str(p) for p in sources],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources if p!=selection_path},
        selection_status='INTEGRATION_COMPLETE_NOT_CALIBRATED',
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'src/cf4_r2_coarsened_live_joint.py',
                ROOT/'src/cf4_r2_native_to_count_cells.py',
                ROOT/'src/cf4_r2_hierarchical_marks.py',ROOT/'src/cf4_r1_particle_forward.py')})
    OUT.mkdir(parents=True)
    def write():
        report['runtime_seconds']=time.monotonic()-start
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write()
    try:
        evolve,_initial,conf,_cosmo,mass_one=make_dynamics(settings)
        mass=jnp.full((size,),mass_one)
        def parts_and_unit(x):
            pos,vel=evolve(x[:size])
            field=particle_grid(pos,vel,mass,conf)
            vector=jnp.moveaxis(field['mean_velocity_km_s'],-1,0)
            return count_and_mark_parts(field['rho'],vector,x[:size],x[size:size+nh],
                x[size+nh:],g,sd,e,k,c,rate_mean,shape,bias,fog,redshift,
                box=BOX,hubble=common['H0_km_s_Mpc'],h=common['h'])
        def scalar_and_parts(x):
            parts,_=parts_and_unit(x)
            return parts.sum(),parts
        seed=2026092801
        ic=np.random.default_rng(seed).standard_normal(size)
        u=np.random.default_rng(2026092802).standard_normal(ng)
        first=jnp.asarray(np.r_[ic,np.zeros(nh),u])
        del ic,u
        t=time.monotonic()
        (total,part),grad=jax.jit(jax.value_and_grad(scalar_and_parts,has_aux=True))(first)
        grad.block_until_ready()
        report['first_joint_value_gradient_seconds']=time.monotonic()-t
        observed=np.asarray(part)
        gradient=np.asarray(grad)
        if not np.isfinite(observed).all() or not np.isfinite(gradient).all():
            raise FloatingPointError('nonfinite joint score/gradient')
        report['initial_seed']=seed
        report['initial_logfactors']={k:float(v) for k,v in zip(('count','conditional_mark','white_prior'),observed)}
        report['initial_total_logtarget']=float(total)
        report['initial_joint_gradient_RMS']=float(np.sqrt(np.mean(gradient*gradient)))
        report['initial_IC_gradient_RMS']=float(np.sqrt(np.mean(gradient[:size]**2)))
        write()
        # One forward support readout from the SAME initial white IC/field.
        t=time.monotonic()
        _,unit=jax.jit(parts_and_unit)(first)
        unit_np=np.asarray(unit)
        occ=unit_np.reshape(-1)[keys]
        if not np.isfinite(unit_np).all() or np.any(unit_np<0) or np.any(occ<=0):
            raise ValueError('occupied count intensity support failure')
        report['initial_rate_one_expected_total']=float(unit_np.sum())
        report['occupied_zero_unit_intensity']=int(np.count_nonzero(occ<=0))
        report['support_check_seconds']=time.monotonic()-t
        del unit_np,occ,unit
        write()
        direction=np.random.default_rng(2026092803).standard_normal(len(first))
        direction[:size]/=np.sqrt(np.mean(direction[:size]**2))
        direction=jnp.asarray(direction)
        @jax.jit
        def parts_only(x):
            return parts_and_unit(x)[0]
        tangent=np.asarray(jax.jvp(parts_only,(first,),(direction,))[1])
        prior_tangent=-float(jnp.vdot(first,direction))
        if abs(tangent[2]-prior_tangent)>1e-6:
            raise FloatingPointError('analytic white prior tangent mismatch')
        finite=[]
        for epsilon in (2e-5,1e-5):
            plus=np.asarray(parts_only(first+epsilon*direction))
            minus=np.asarray(parts_only(first-epsilon*direction))
            derivative=(plus-minus)/(2*epsilon)
            finite.append(dict(epsilon=epsilon,count=float(derivative[0]),
                conditional_mark=float(derivative[1]),
                count_scaled_error=float(abs(derivative[0]-tangent[0])/max(1.,abs(derivative[0]),abs(tangent[0]))),
                mark_scaled_error=float(abs(derivative[1]-tangent[1])/max(1.,abs(derivative[1]),abs(tangent[1])))))
        report['directional_derivative']=dict(autodiff_count=float(tangent[0]),
            autodiff_mark=float(tangent[1]),analytic_prior=prior_tangent,
            total_autodiff=float(jnp.vdot(grad,direction)),finite_differences=finite)
        report['directional_derivative']['total_vs_parts_difference']=float(
            jnp.vdot(grad,direction)-sum(tangent))
        report['status']='PASS_NUMERICAL_PARTIAL_TARGET_NOT_CALIBRATED' if (
            finite[-1]['count_scaled_error']<.02 and finite[-1]['mark_scaled_error']<.02
            and abs(report['directional_derivative']['total_vs_parts_difference'])<1e-5
        ) else 'NO_GO_DIRECTIONAL_DERIVATIVE'
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write()
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
