"""Saved-state FP consistency or training-distance readout; no PM/fit/heldout."""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from cf4_r2_linked_fp_sparse_train import load_train_singletons, FP, SOURCE
from cf4_r2_linked_singleton_target import partial_v6_count_singleton_parts
from cf4_r2_linked_singleton_jax import linked_singleton_logfactors_for_population
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_marked_tracer_jax import intrinsic_biased_source_masses, intrinsic_lf_bin_fractions
from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax

BASE = Path('/gpfs/kjhan/CF4/z0_density')
N, BOX, WIDTH = 128, 384., 8192


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
                  source_commit=os.environ['CF4_EXPECTED_COMMIT'], device=str(jax.devices()),
                  PM_evolutions=0, optimizer_steps=0, heldout_scored=False, R2_complete=False)
    def save():
        report['seconds'] = time.monotonic()-started
        (out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    combined_check = os.environ.get('CF4_R2_FP_COMBINED_CHECK') == '1'
    compact_check = os.environ.get('CF4_R2_FP_COMPACT_CHECK') == '1'
    flat_check = os.environ.get('CF4_R2_FP_FLAT_CHECK') == '1'
    science_state = os.environ.get('CF4_R2_FP_SCIENCE_STATE')
    if science_state and (combined_check or compact_check or flat_check):
        raise ValueError('science readout must not launch compiler diagnostics')
    paths = ([Path(science_state)] if science_state else
             [BASE/'r2_v6_fixed_field_nuisance_v1/final_state.npz',
              BASE/'r2_v6_joint_secant_map_v2/initial_state.npz'])
    states = []
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            states.append({k:f[k].copy() for k in
                ('white_ic','rho','velocity_km_s','tracer','white_fp_zero')})
    if len(states)==2:
        report['state_max_abs_differences'] = {
            k:float(np.max(np.abs(states[0][k]-states[1][k]))) for k in states[0]}
    options, point, fp = load_train_singletons(BASE/'r2_sky_closed_split_v6/split.npz')
    with np.load(FP, allow_pickle=False) as f:
        membership = f['membership_state'].astype(str)
    options = [o for o in options if membership[o[3]] in {
        'source_ungrouped_catalogue_present','source_ungrouped_catalogue_absent'}]
    assert len(options) == 429
    with np.load(SOURCE, allow_pickle=False) as f:
        source = {k:jnp.asarray(f[k]) for k in f.files}
    selected = {p:[o for o in options if point['population'][o[2]] == p] for p in range(6)}
    metadata, centers = {}, {}
    for p, batch in selected.items():
        voxels = np.array([np.unravel_index(int(point['flat_cell'][o[2]]),(N,)*3)
                           for o in batch], dtype=np.int32).reshape(-1,3)
        centers[p] = (voxels+.5)*(BOX/N)
        fi = np.array([o[4] for o in batch],dtype=np.int32)
        metadata[p] = dict(voxel_ijk=jnp.asarray(voxels),
            observed_radius_cMpc_h=jnp.array([point['radius_cMpc_h'][o[2]] for o in batch]),
            **{k:jnp.asarray(fp[k][fi]) for k in ('dz_row','eta_mean','eta_std','eta_alpha')})

    @jax.jit
    def shifted(rho, vel):
        _, v = native_mass_momentum_to_count_cells(rho, vel, BOX)
        return observer_centred_spherical_rsd_jax(source['positions'],
            jnp.moveaxis(v,0,-1).reshape(-1,3),jnp.full(3,192.),BOX,74.6,
            little_h=.746,scale_factor=1.)[0]

    @jax.jit
    def mark(rho, vel, tracer, zero, links):
        return partial_v6_count_singleton_parts(rho,vel,jnp.zeros(0),tracer,source,links,
            jnp.zeros(0,dtype=jnp.int32),jnp.zeros(0),jnp.zeros(6*N**3,dtype=bool),
            white_fp_zero=zero,count_integration='shell_cdf')[0][1]

    def per_row(rho, vel, tracer, zero, links):
        density, v = native_mass_momentum_to_count_cells(rho,vel,BOX)
        intrinsic = intrinsic_biased_source_masses(density,
            jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tracer[0],
            jnp.exp(.5*tracer[1:6]),mstar=-23.28+.2*tracer[8],
            alpha=-1+.06*jnp.exp(.5*tracer[7]),reference_interval=(-25.,-21.))
        geometry = dict(observer=jnp.full(3,192.),box_size_cMpc_h=BOX,hubble_km_s_Mpc=74.6,
            little_h=.746,radius_table_cMpc_h=source['radial_table'],
            modulus_table_h=source['modulus_table'],redshift_table=source['redshift_table'],
            grid_size=N,radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,
            mstar=-23.28+.2*tracer[8],alpha=-1+.06*jnp.exp(.5*tracer[7]))
        batches = [linked_singleton_logfactors_for_population(
            source['positions'],jnp.moveaxis(v,0,-1).reshape(-1,3),intrinsic,source['angular'],
            **links[p],population=p,sigma_los_km_s=100*jnp.exp(.5*tracer[6]),
            radial_geometry=geometry,fp_zero_dex=.004*zero,return_eta_moments=bool(science_state))
            for p in range(6) if len(selected[p])]
        if science_state:
            return tuple(jnp.concatenate([batch[k] for batch in batches]) for k in (0,2,3))
        return jnp.concatenate([batch[0] for batch in batches])
    rows_compiled = jax.jit(per_row)
    report['states'] = []
    for index, state in enumerate(states):
        args = [jnp.asarray(state[k]) for k in ('rho','velocity_km_s','tracer','white_fp_zero')]
        radius = 8*.01*100*np.exp(.5*state['tracer'][6])+np.sqrt(3)*1.5*BOX/N
        tree = cKDTree(np.asarray(shifted(*args[:2])) % BOX,boxsize=BOX)
        links = {}
        for p, batch in selected.items():
            neighborhoods = tree.query_ball_point(centers[p],radius,workers=1)
            ids = np.zeros((len(batch),WIDTH),dtype=np.int32)
            active = np.zeros_like(ids,dtype=bool)
            for i, neighbors in enumerate(neighborhoods):
                assert 0 < len(neighbors) <= WIDTH
                ids[i,:] = neighbors[0]; ids[i,:len(neighbors)] = neighbors
                active[i,:len(neighbors)] = True
            links[p] = dict(metadata[p],candidate_source_ids=jnp.asarray(ids),
                candidate_mask=jnp.asarray(active),association_logprob=jnp.zeros((len(batch),5,WIDTH)))
        value = float(mark(*args,links))
        evaluated = rows_compiled(*args,links)
        rows = np.asarray(evaluated[0] if science_state else evaluated)
        np.savez(out/f'rows_{index}.npz',factors=rows,
                 labels=np.array([o[0] for p in range(6) for o in selected[p]]))
        report['states'].append(dict(path=str(paths[index]),FP=value,per_row_sum=float(rows.sum())))
        save(); print(json.dumps(report['states'][-1]),flush=True)
        if science_state:
            mean,var = map(np.asarray,evaluated[1:])
            observed = np.concatenate([np.asarray(metadata[p]['eta_mean']) for p in range(6)])
            std = np.concatenate([np.asarray(metadata[p]['eta_std']) for p in range(6)])
            zero = .004*float(state['white_fp_zero'])
            predicted = mean+zero
            previous=json.loads((paths[index].parent/'result.json').read_text())
            reference=previous['trace'][-1]['parts'][1]
            if (not np.isfinite(np.r_[rows,mean,var,observed,std]).all()
                    or np.any(var<0.) or np.any(std<=0.)
                    or not np.isclose(value,reference,rtol=0.,atol=1e-7)
                    or not np.isclose(rows.sum(),value,rtol=0.,atol=1e-10)):
                raise AssertionError('training readout is invalid or does not reproduce fitted FP factor')
            np.savez(out/'training_distance_prediction.npz',
                labels=np.array([o[0] for p in range(6) for o in selected[p]]),
                eta_source_mean=observed,eta_source_std=std,
                eta_prediction_before_zero=mean,eta_prediction_with_zero=predicted,
                count_conditioned_eta_variance=var,FP_log_factors=rows)
            report['training_distance_readout']=dict(rows=len(rows),white_fp_zero=float(state['white_fp_zero']),
                zero_dex=zero,mean_residual_before_zero=float(np.mean(observed-mean)),
                mean_residual_after_zero=float(np.mean(observed-predicted)),
                RMS_residual_before_zero=float(np.sqrt(np.mean((observed-mean)**2))),
                RMS_residual_after_zero=float(np.sqrt(np.mean((observed-predicted)**2))),
                correlation=(float(np.corrcoef(observed,predicted)[0,1])
                             if np.std(observed)>0 and np.std(predicted)>0 else None),
                mean_source_reported_std=float(std.mean()),
                standardized_residual_mean=float(np.mean((observed-predicted)/std)),
                standardized_residual_std=float(np.std((observed-predicted)/std)),
                classification='TRAINING_PLUG_IN_DISTANCE_READOUT_NOT_POSTERIOR_PREDICTIVE',
                interpretation='moments conditioned on count key/redshift/association at the fitted field; '
                    'not reweighted by this FP mark; not field posterior variance or independent validation',
                limits='published source PDF moments are not a calibrated Gaussian residual law; '
                    'shared zero was fitted to these same training rows; no chi-square significance claim',
                MW_M31_M33='Not identified here; MW/M31 ambiguous and M33 unresolved on same NEW field; no truth IDs')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
            ax=axes[0]
            ax.errorbar(predicted,observed,xerr=np.sqrt(var),yerr=std,
                        fmt='.',markersize=2,alpha=.2,elinewidth=.4,color='tab:blue')
            limits=(float(min(observed.min(),predicted.min())),float(max(observed.max(),predicted.max())))
            ax.plot(limits,limits,'k--',lw=1,label='equal values')
            ax.set(xlabel='Predicted eta (fitted field + fitted shared zero)',
                   ylabel='Published FP eta mean',title='429 training FP links: direct distance comparison')
            ax.legend()
            extent=max(4.,float(np.ceil(np.max(np.abs(np.r_[
                (observed-mean)/std,(observed-predicted)/std])))))
            bins=np.linspace(-extent,extent,41)  # Include every row, not only central residuals.
            for label,residual in [('before shared zero',(observed-mean)/std),
                                   ('after fitted shared zero',(observed-predicted)/std)]:
                axes[1].hist(residual,bins=bins,histtype='step',label=label)
            axes[1].axvline(0.,color='black',lw=1)
            axes[1].set(xlabel='(source mean - prediction) / source reported std',
                        ylabel='Training rows',title='Descriptive residuals; NOT a calibrated Gaussian test')
            axes[1].legend()
            fig.suptitle('eta = log10(redshift distance / true distance)\nTraining plug-in readout, NOT posterior prediction or independent validation')
            fig.savefig(out/'training_distance_prediction.png',dpi=150)
            plt.close(fig)
            save()
        if flat_check and index == 1:
            # Same mathematical FP function/inputs as the nested-jit discrepancy;
            # differentiate the underlying function before the sole outer jit.
            flat = jax.jit(jax.value_and_grad(mark.__wrapped__,argnums=(0,1,2,3)))
            flat_value,flat_grad = flat(*args,links)
            report['flat_vg'] = dict(FP=float(flat_value),
                tracer_gradient=np.asarray(flat_grad[2]).tolist(),zero_gradient=float(flat_grad[3]))
            save(); print(json.dumps(report['flat_vg']),flush=True)
            if not np.isclose(float(flat_value),value,rtol=0.,atol=1e-10):
                raise AssertionError('flat FP autodiff still changes primal value')
        if compact_check and index == 1:
            density,velocity = native_mass_momentum_to_count_cells(*args[:2],BOX)
            tracer,zero = args[2:]
            intrinsic = intrinsic_biased_source_masses(density,
                jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tracer[0],
                jnp.exp(.5*tracer[1:6]),mstar=-23.28+.2*tracer[8],
                alpha=-1+.06*jnp.exp(.5*tracer[7]),reference_interval=(-25.,-21.))
            velocity = jnp.moveaxis(velocity,0,-1).reshape(-1,3)
            compact_rows, compact_vg_rows = [], []
            for p in range(6):
                if not len(selected[p]):
                    continue
                link = links[p]
                width = int(np.max(np.asarray(link['candidate_mask']).sum(axis=1)))
                ids = link['candidate_source_ids'][:,:width]
                positions = source['positions'][ids]
                velocities = velocity[ids]
                masses = jnp.moveaxis(intrinsic[:,ids],0,1)*link['candidate_mask'][:,:width,None].transpose(0,2,1)
                sky = jnp.moveaxis(source['angular'][:,ids],0,1)
                def one(pos,vel,mass,angular,voxel,radius,dz,mean,std,alpha,zero,tracer):
                    geometry = dict(observer=jnp.full(3,192.),box_size_cMpc_h=BOX,
                        hubble_km_s_Mpc=74.6,little_h=.746,
                        radius_table_cMpc_h=source['radial_table'],modulus_table_h=source['modulus_table'],
                        redshift_table=source['redshift_table'],grid_size=N,
                        radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,
                        mstar=-23.28+.2*tracer[8],alpha=-1+.06*jnp.exp(.5*tracer[7]))
                    return linked_singleton_logfactors_for_population(pos,vel,mass,angular,
                        jnp.arange(width)[None],jnp.ones((1,width),dtype=bool),jnp.zeros((1,5,width)),
                        voxel[None],radius[None],dz[None],mean[None],std[None],alpha[None],
                        population=p,sigma_los_km_s=100*jnp.exp(.5*tracer[6]),
                        radial_geometry=geometry,fp_zero_dex=.004*zero)[0][0]
                batched = jax.vmap(one,in_axes=(0,0,0,0,0,0,0,0,0,0,None,None))
                inputs = (positions,velocities,masses,sky,*[link[k] for k in (
                    'voxel_ijk','observed_radius_cMpc_h','dz_row','eta_mean','eta_std','eta_alpha')],zero,tracer)
                plain = np.asarray(jax.jit(batched)(*inputs))
                def summed(*values):
                    factors = batched(*values)
                    return factors.sum(),factors
                (total, factors),grad = jax.jit(jax.value_and_grad(summed,
                    argnums=(1,2,10,11),has_aux=True))(*inputs)
                compact_rows.extend(plain.tolist()); compact_vg_rows.extend(np.asarray(factors).tolist())
                report.setdefault('compact_populations',[]).append(dict(population=p,width=width,
                    plain=float(plain.sum()),vg=float(total),
                    max_row_difference=float(np.max(np.abs(plain-np.asarray(factors)))),
                    zero_gradient=float(grad[2])))
                save(); print(json.dumps(report['compact_populations'][-1]),flush=True)
            np.savez(out/'compact_rows.npz',plain=compact_rows,vg=compact_vg_rows,
                     labels=np.array([o[0] for p in range(6) for o in selected[p]]))
        if combined_check and index == 1:
            isolated_value, isolated_grad = jax.jit(jax.value_and_grad(
                mark,argnums=(0,1,2,3)))(*args,links)
            report['isolated_vg'] = dict(FP=float(isolated_value),
                tracer_gradient=np.asarray(isolated_grad[2]).tolist(),zero_gradient=float(isolated_grad[3]))
            save(); print(json.dumps(report['isolated_vg']),flush=True)
            with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
                keys,counts = jnp.asarray(f['train_keys']),jnp.asarray(f['train_counts'])
                exposure,_ = build_population_exposure_masks(N,f['heldout_flat_voxels'],
                    f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
            exposure = jnp.asarray(exposure)
            def combined(rho,vel,tracer,zero,links):
                parts,_ = partial_v6_count_singleton_parts(rho,vel,jnp.zeros(0),tracer,
                    source,links,keys,counts,exposure,white_fp_zero=zero,
                    count_integration='shell_cdf')
                return jnp.sum(parts[:3]),parts[:3]
            (value,parts),grad = jax.jit(jax.value_and_grad(combined,
                argnums=(0,1,2,3),has_aux=True))(*args,links)
            report['combined_vg'] = dict(parts=np.asarray(parts).tolist(),value=float(value),
                tracer_gradient=np.asarray(grad[2]).tolist(),zero_gradient=float(grad[3]))
            save(); print(json.dumps(report['combined_vg']),flush=True)
    report['status'] = ('TRAINING_FP_DISTANCE_READOUT_COMPLETE_NOT_POSTERIOR' if science_state
                        else 'SAVED_STATE_FP_CHECK_COMPLETE_NOT_POSTERIOR')
    save()


if __name__ == '__main__':
    main()
