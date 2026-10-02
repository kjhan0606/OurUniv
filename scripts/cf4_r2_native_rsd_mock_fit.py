"""One native galaxy RSD/count calibration mock with a fixed matter field.

Known source-window geometry, native K proxy, one universe. No CF4 outcome,
gravity evolution or posterior inference over a cosmological field.
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from cf4_r2_native_mock import (distance_tables,place_native_halves,observe,
    exposure_and_radial_bins,aggregate)
from cf4_r2_raw_volume_target import tracer_masses,tracer_geometry
from cf4_r2_shell_cdf_count import predict_source_volume_intensity
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood

jax.config.update('jax_enable_x64',True)
BASE=Path('/gpfs/kjhan/CF4/z0_density')
GALAXIES=BASE/'r2_tng_native_k_response_20261002_v3/native_k_galaxies.npz'
MATTER=BASE/'bundle_c_v1/total_matter_v1/matter_moments.h5'
OUT=Path(os.environ.get('CF4_R2_OUT_DIR',str(BASE/'r2_native_rsd_mock_fit_20261002_v1')))
MAX_EVALUATIONS=int(os.environ.get('CF4_R2_MAX_EVALUATIONS','48'))
MAX_ITERATIONS=int(os.environ.get('CF4_R2_MAX_ITERATIONS','24'))
RESTART=os.environ.get('CF4_R2_RESTART_RESULT')
APPLICATION_SECONDS=75*60


class BudgetStop(Exception):pass


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('submit via Slurm with a GPU')
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        classification='ONE_NATIVE_K_SOURCE_WINDOW_COUNT_MOCK_NOT_R2_CALIBRATION',
        source_galaxies=str(GALAXIES),source_matter=str(MATTER),
        source_spacing_cMpc_h=1.5,count_spacing_cMpc_h=3.,
        near_origin=[154.5,154.5,154.5],far_translation_x_cMpc_h=84.,
        native_split_x_cMpc_h=37.5,observer=[192.,192.,192.],
        actual_CF4_2Mpp_outcomes_read=False,gravity_evolutions=0,
        maximum_fit_evaluations=MAX_EVALUATIONS,maximum_fit_iterations=MAX_ITERATIONS,
        application_seconds=APPLICATION_SECONDS,
        priors='same nine standard-normal tracer development coordinates; one penalty',
        source_volume_order=2,LOS_order=4,LOS_segments=8,trace=[],
        limits=['Native IR K Vega is a Ks proxy; dust/passband/aperture mapping remains open.',
            'Two translated disjoint halves define known source windows, not a full physical384 box.',
            'Each native galaxy/cell appears once; spatial tests still share one hydro75 box.',
            'Native coarse NGP moments versus TSC count response and COM versus stellar velocities.',
            'Earlier source diagnostics consumed this universe; this is development prediction, not pristine independent validation.',
            'No IC, R2 prior injection, actual heldout outcome or LG role assignment.'],
        Q_GOAL='test actual luminosity-defined galaxy/RSD response under the active finite count law upstream of R2 present-field delivery',
        Q_LEAN='one preserved native field and catalogue,9 nuisance coordinates, bounded fit and one numerical endpoint comparison',
        MW_M31='ambiguous on the NEW R2 field; observables must constrain that same state',
        M33='unresolved on the NEW R2 field; native IDs are calibration labels only')

    def save():
        report['elapsed_seconds']=time.monotonic()-start
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    save()
    with np.load(GALAXIES,allow_pickle=False) as f:
        galaxies={k:f[k] for k in f.files}
    with h5py.File(MATTER,'r') as f:
        moments=f['coarse'][:];h=float(f.attrs['h']);omega=float(f.attrs['Omega_m'])
    if moments.shape!=(7,50,50,50) or np.any(moments[0]<=0):
        raise ValueError('complete positive native matter field required')
    if not np.isfinite(moments).all():raise ValueError('nonfinite native moments')
    if not np.allclose(galaxies['K_h_proxy'],galaxies['K_physical']-5*np.log10(h),rtol=0,atol=1e-10):
        raise ValueError('native magnitude/h convention mismatch')
    if len(np.unique(galaxies['native_id']))!=len(galaxies['native_id']):
        raise ValueError('native calibration catalogue repeats objects')
    rho=jnp.asarray(moments[0]/moments[0].mean())
    velocity=np.moveaxis(moments[1:4]/moments[0],0,-1).reshape(-1,3)
    axis=(np.arange(50)+.5)*1.5
    native_pos=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    source_pos=jnp.asarray(place_native_halves(native_pos))
    source_vel=jnp.asarray(velocity)
    angular=jnp.ones((2,len(native_pos)))
    rtable,ztab,mtab=distance_tables(omega)
    geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
        hubble_km_s_Mpc=100*h,little_h=h,grid_size=128,
        radius_table_cMpc_h=jnp.asarray(rtable),redshift_table=jnp.asarray(ztab),
        modulus_table_h=jnp.asarray(mtab),radial_min_cMpc_h=5.,radial_max_cMpc_h=180.)
    mock=observe(place_native_halves(galaxies['position']),galaxies['velocity'],
        galaxies['K_h_proxy'],rtable,ztab,mtab)
    keys,counts=mock['keys'],mock['counts']
    train,test,rbin=exposure_and_radial_bins()
    train_rows=train[keys%128**3]
    train_keys,train_counts=keys[train_rows],counts[train_rows]
    if len(train_keys)==0:raise ValueError('empty training mock')
    selected=mock['selected']
    if int(counts.sum())!=int(selected.sum()):raise ValueError('mock counts not conserved')
    migration=np.zeros((5,6),dtype=int)
    np.add.at(migration,(mock['true_bin'][selected],mock['population'][selected]),1)
    observed_train=aggregate(keys,counts,train,rbin)
    observed_test=aggregate(keys,counts,test,rbin)
    report.update(native_h=h,native_Omega_m=omega,source_cells=len(native_pos),
        selected_native_galaxies=int(selected.sum()),train_count=int(train_counts.sum()),
        test_count=int(counts[test[keys%128**3]].sum()),
        buffered_count=int(counts[(~train&~test)[keys%128**3]].sum()),
        population_counts=np.bincount(mock['population'][selected],minlength=6).tolist(),
        true_to_observed_population=migration.tolist(),
        selected_stellar_resolution={str(n):int(np.sum(selected&(galaxies['star_count']>=n))) for n in (1,100,300)},
        observed_training_table=observed_train.tolist(),observed_test_table=observed_test.tolist())
    np.savez_compressed(OUT/'mock_observations.npz',keys=keys,counts=counts,
        native_selected_ids=galaxies['native_id'][selected],
        selected_star_count=galaxies['star_count'][selected],
        train_exposure=train,test_exposure=test)
    save()
    key_j,count_j,train_j=jnp.asarray(train_keys),jnp.asarray(train_counts),jnp.asarray(train)

    def prediction(tracer,volume_order=2,los_segments=8,deposition='tsc'):
        # Rates are per3-cMpc/h cell; each1.5 source cell carries1/8 volume.
        return predict_source_volume_intensity(source_pos,source_vel,
            tracer_masses(rho,tracer)/8.,angular,source_spacing=1.5,
            volume_order=volume_order,order=4,segments=los_segments,
            deposition=deposition,**tracer_geometry(tracer,geometry))

    def target(tracer):
        intensity=prediction(tracer)
        score=sparse_marked_poisson_log_likelihood(intensity,key_j,count_j,selected_voxel_mask=train_j)
        return .5*jnp.vdot(tracer,tracer)-score,(score,intensity)

    vg=jax.jit(jax.value_and_grad(target,has_aux=True))
    scalar=jax.jit(lambda q:target(q)[0])
    q=np.zeros(9);q[6]=2*np.log(3.) # predeclared300km/s start; not native-truth fit.
    if RESTART:
        previous=json.loads(Path(RESTART).read_text())
        if (previous['source_galaxies']!=str(GALAXIES)
            or previous['source_matter']!=str(MATTER)
            or previous['classification']!=report['classification']
            or previous['source_volume_order']!=2 or previous['LOS_segments']!=8):
            raise ValueError('restart source/model contract mismatch')
        q=np.asarray(previous['final_coordinates'],dtype=float)
        if q.shape!=(9,) or not np.isfinite(q).all():
            raise ValueError('invalid restart coordinates')
        report['restart']=dict(result=RESTART,coordinates=q.tolist(),
            optimizer_history='fresh L-BFGS history; training-only rate reprofiling',
            original_job=previous['job_id'])
    tic=time.monotonic();(value,(score,initial)),grad=vg(jnp.asarray(q))
    jax.block_until_ready((value,score,initial,grad))
    report['first_gradient_compile_seconds']=time.monotonic()-tic
    initial_host=np.asarray(initial)
    if not np.isfinite(float(value)) or not np.isfinite(np.asarray(grad)).all():
        report.update(status='INITIAL_TARGET_SUPPORT_OR_GRADIENT_FAILURE',
            occupied_zero_intensity=int(np.sum(initial_host.reshape(-1)[train_keys]<=0)))
        save();raise ValueError(report['status'])
    mean=float(np.sum(initial_host.reshape(6,-1)[:,train]))
    expected_rate_grad=q[0]-2*(int(train_counts.sum())-mean)
    rate_error=abs(float(grad[0])-expected_rate_grad)/max(1.,abs(expected_rate_grad))
    if rate_error>1e-8:raise ValueError('analytic rate-gradient identity failed')
    scale=int(train_counts.sum())/mean
    q[0]+=.5*np.log(scale) # training-only common-rate initialization.
    (value,(score,initial)),grad=vg(jnp.asarray(q))
    jax.block_until_ready((value,score,initial,grad))
    initial_host=np.asarray(initial)
    direction=np.random.default_rng(20261002).normal(size=9)
    direction/=np.linalg.norm(direction)
    eps=1e-4
    fd=float((scalar(jnp.asarray(q+eps*direction))-scalar(jnp.asarray(q-eps*direction)))/(2*eps))
    reverse=float(np.dot(np.asarray(grad),direction))
    fd_error=abs(fd-reverse)/max(1.,abs(fd),abs(reverse))
    report['startup_checks']=dict(rate_gradient_relative_error=rate_error,
        rate_profiled_expected_count=float(np.sum(initial_host.reshape(6,-1)[:,train])),
        direct_vs_derivative_primal_error=abs(float(scalar(jnp.asarray(q)))-float(value)),
        directional_reverse=reverse,directional_finite_difference=fd,
        directional_relative_error=fd_error)
    if (fd_error>2e-4 or report['startup_checks']['direct_vs_derivative_primal_error']>1e-7
        or abs(report['startup_checks']['rate_profiled_expected_count']-int(train_counts.sum()))>1e-7):
        save();raise ValueError('native count target value/derivative/rate check failed')
    initial_q=q.copy();accepted=q.copy();evaluations=0
    save()

    def objective(x):
        nonlocal evaluations
        if evaluations>=MAX_EVALUATIONS or time.monotonic()-start>=APPLICATION_SECONDS:
            raise BudgetStop()
        if not np.isfinite(x).all() or not 0<8*np.exp(.5*x[6])<192.:
            raise ValueError('optimizer trial outside existing27-image domain')
        tic=time.monotonic();(val,(sc,lam)),gr=vg(jnp.asarray(x))
        jax.block_until_ready((val,sc,lam,gr));evaluations+=1
        v,g=float(val),np.asarray(gr)
        if not np.isfinite(v) or not np.isfinite(g).all():
            raise ValueError('nonfinite native mock optimizer trial')
        report['trace'].append(dict(evaluation=evaluations,objective=v,
            count_logscore=float(sc),gradient_inf=float(np.max(np.abs(g))),
            coordinates=np.asarray(x).tolist(),seconds=time.monotonic()-tic))
        save()
        return v,g

    def callback(x):
        nonlocal accepted
        accepted=np.asarray(x).copy()

    try:
        fit=minimize(objective,q,jac=True,method='L-BFGS-B',callback=callback,
            options=dict(maxiter=MAX_ITERATIONS,maxfun=MAX_EVALUATIONS-8,
                         maxls=8,ftol=1e-12,gtol=1e-3))
        accepted=fit.x.copy()
        report['optimizer']=dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit))
    except BudgetStop:
        report['optimizer']=dict(success=False,message='predeclared evaluation/application cap')
    (value,(score,final)),grad=vg(jnp.asarray(accepted))
    jax.block_until_ready((value,score,final,grad));final_host=np.asarray(final)
    report.update(status='NATIVE_RSD_MOCK_FIT_DEVELOPMENT_COMPLETE',
        evaluations=evaluations,initial_coordinates=initial_q.tolist(),
        final_coordinates=accepted.tolist(),final_gradient_inf=float(jnp.max(jnp.abs(grad))),
        final_count_logscore=float(score),final_objective=float(value),
        physical_beta=np.exp(.5*accepted[1:6]).tolist(),
        physical_sigma_los_km_s=float(100*np.exp(.5*accepted[6])),
        physical_alpha=float(-1+.5*accepted[7]),physical_Mstar=float(-23.28+.2*accepted[8]))
    report['stationarity_verified']=bool(report['optimizer']['success']
        and report['final_gradient_inf']<=1e-3)
    save()

    def metrics(lam,mask):
        flat=lam.reshape(-1);keep=mask[keys%128**3]
        kk,nn=keys[keep],counts[keep]
        positive=flat[kk]>0
        if not positive.all():return dict(occupied_zero_intensity=int(np.sum(~positive)))
        expected=lam.reshape(6,-1)[:,mask].sum()
        table=np.stack([np.bincount(rbin[mask],weights=row[mask],minlength=16)
                        for row in lam.reshape(6,-1)])
        obs=aggregate(keys,counts,mask,rbin);use=table>=5.
        return dict(logscore=float(np.sum(nn*np.log(flat[kk])-gammaln(nn+1))-expected),
            expected_count=float(expected),observed_count=int(nn.sum()),
            expected_table=table.tolist(),population_totals=table.sum(axis=1).tolist(),
            aggregate_L1=float(np.sum(np.abs(obs-table))),
            aggregate_Pearson_ge5=float(np.sum((obs[use]-table[use])**2/table[use])),
            aggregate_bins_ge5=int(use.sum()))

    report['prediction']={name:{split:metrics(lam,mask) for split,mask in (('train',train),('test',test))}
                          for name,lam in (('initial',initial_host),('fitted',final_host))}
    np.savez_compressed(OUT/'predicted_counts.npz',initial=initial_host,fitted=final_host)
    save()
    # Two existing-law numerical controls at this fixed fitted nuisance point.
    for name,kwargs in (('source_GL4',dict(volume_order=4)),
                        ('LOS_segments16',dict(los_segments=16)),
                        ('diagnostic_NGP',dict(deposition='ngp'))):
        if time.monotonic()-start>=APPLICATION_SECONDS:
            report['numerical_controls_incomplete']=True;break
        tic=time.monotonic()
        f=jax.jit(lambda t:prediction(t,**kwargs))
        alternative=f(jnp.asarray(accepted));jax.block_until_ready(alternative)
        host=np.asarray(alternative)
        if not np.isfinite(host).all():raise ValueError('nonfinite endpoint control')
        report[name]=dict(seconds=time.monotonic()-tic,
            train=metrics(host,train),test=metrics(host,test),
            relative_L1_to_GL2=float(np.sum(np.abs(host-final_host))/np.sum(final_host)))
        save()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    final_train=np.asarray(report['prediction']['fitted']['train']['expected_table'])
    final_test=np.asarray(report['prediction']['fitted']['test']['expected_table'])
    vmax=max(np.log1p(a).max() for a in (observed_train,observed_test,final_train,final_test))
    for ax,title,table in zip(axs.ravel(),('Native training counts','Fitted training means',
        'Native test counts','Predicted test means'),(observed_train,final_train,observed_test,final_test)):
        im=ax.imshow(np.log1p(table),origin='lower',aspect='auto',vmin=0,vmax=vmax,
            extent=(0,192,-.5,5.5));ax.set_title(title);ax.set_ylabel('observed K population')
        ax.set_xlabel('cell-centre radius (cMpc/h); final bin >=180')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='log(1 + count)',shrink=.8)
    fig.suptitle('One native-K source-window RSD mock: development comparison')
    fig.savefig(OUT/'population_radius_prediction.png',dpi=150);plt.close(fig)
    save();print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
