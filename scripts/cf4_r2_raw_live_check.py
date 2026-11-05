"""Bounded live raw-mark value/gradient check; no fitting or PM evolution."""
import json
import os
from pathlib import Path
import resource
import time
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from cf4_r2_linked_fp_sparse_train import load_train_singletons,select_training_single_mark_links,FP,SOURCE
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_marked_tracer_jax import (intrinsic_biased_source_masses,intrinsic_lf_bin_fractions,
    predict_source_marked_radial_key_density)
from cf4_r2_raw_live_mark import streaming_raw_mark,POPULATION_ORIGIN,POPULATION_SCALE,logadd_nonempty

BASE=Path('/gpfs/kjhan/CF4/z0_density'); BOX=384.;N=128;CHUNK=64


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],
        source_commit=os.environ['CF4_EXPECTED_COMMIT'],R2_complete=False,
        PM_evolutions=0,heldout_scored=False,optimizer_steps=0,rows=[],
        limits='observation adapter only; no count target replacement or calibrated field posterior; '
               '4^3 volume approximation with measured residual; MW/M31 ambiguous,M33 unresolved')
    def save():
        report['seconds']=time.monotonic()-started
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        packed=os.environ.get('CF4_R2_RAW_PACKED')=='1'
        report['sparse_components']=packed
        if packed:
            with np.load(BASE/'r2_raw_live_check_v1/live_derivatives.npz',allow_pickle=False) as f:
                old_gradients=dict(zip(f['cohort_rows'],f['gradient']))
        predecessor=json.loads((BASE/'r2_source_cell_quadrature_v2/result.json').read_text())
        rows=predecessor['order8_rows']
        reference={r['row']:r for r in predecessor['rows'] if r['nodes_per_axis']==4}
        with np.load(BASE/'r2_prior_split_long_v1/final_state.npz',allow_pickle=False) as f:
            density,vel=jax.jit(native_mass_momentum_to_count_cells,static_argnums=2)(
                jnp.asarray(f['rho']),jnp.asarray(f['velocity_km_s']),BOX)
            tracer=f['tracer'].copy()
        velocity=jnp.moveaxis(vel,0,-1).reshape(-1,3)
        with np.load(SOURCE,allow_pickle=False) as f: source={k:f[k].copy() for k in f.files}
        with np.load(BASE/'r2_raw_source_mixture_v1/raw_source_mixture.npz',allow_pickle=False) as f:
            mix={k:f[k].copy() for k in ('PGC','population','dz_row','observed_radius','observed_ksmag')}
        with np.load(BASE/'r2_raw_population_fit_v1/final_fit.npz',allow_pickle=False) as f:
            params=f['parameters'].copy(); richness_center=float(f['richness_center'])
            np.testing.assert_array_equal(f['PGC'],mix['PGC'])
        with np.load(BASE/'r2_raw_selected_component_v1/training_cut_geometry.npz',allow_pickle=False) as f:
            order={int(p):i for i,p in enumerate(f['PGC'])}
            indices=[order[int(p)] for p in mix['PGC']]
            optical={k:f[k][indices] for k in ('x','optical_error_covariance','cut_lower','cut_upper')}
        with np.load(BASE/'r2_raw_fp_inputs_v1/training_photometry.npz',allow_pickle=False) as f:
            order={int(p):i for i,p in enumerate(f['selected_PGC'])}
            indices=[order[int(p)] for p in mix['PGC']]
            richness=np.log1p(f['selected_photometry'][indices,list(f['columns']).index('NgroupT17')])-richness_center
        options,point,_=load_train_singletons(BASE/'r2_sky_closed_split_v6/split.npz')
        with np.load(FP,allow_pickle=False) as f:
            options=select_training_single_mark_links(options,f['membership_state'].astype(str),include_grouped=True)
            chosen=[o for p in range(6) for o in options if point['population'][o[2]]==p]
            np.testing.assert_array_equal(mix['PGC'],[f['PGC'][o[3]] for o in chosen])
        voxels=np.array([np.unravel_index(point['flat_cell'][chosen[i][2]],(N,)*3) for i in rows])
        tree=cKDTree(source['positions']%BOX,boxsize=BOX)
        # Larger than the previous proven bound, retaining exact current/trial
        # support for this bounded tiny-direction check. Not reusable HMC cache.
        radius=float(predecessor['support_radius'])+2.
        candidates=tree.query_ball_point((voxels+.5)*BOX/N,radius,workers=1)
        width=((max(map(len,candidates))+CHUNK-1)//CHUNK)*CHUNK
        report.update(cohort_rows=rows,support_radius=radius,cells_per_row=width,
            population_prior='proper weak model prior, NOT external calibration',
            field_directions=['global velocity scaling','positive normalized first-x-mode density modulation'])
        t,w=np.polynomial.legendre.leggauss(4)
        triples=np.array(list(product(range(4),repeat=3)))
        offsets=jnp.asarray(t[triples]*1.5); volumes=jnp.asarray(np.prod(w[triples]/2,axis=1))
        z=np.r_[tracer,(params-POPULATION_ORIGIN)/POPULATION_SCALE,0.,0.]
        direction=np.random.default_rng(2026092908).normal(size=26)
        direction/=np.linalg.norm(direction); direction[24:]*=.1
        delta=2e-5
        mode=jnp.asarray(np.cos(2*np.pi*source['positions'][:,0]/BOX).reshape(density.shape),dtype=density.dtype)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=BOX,hubble_km_s_Mpc=74.6,
            little_h=.746,radius_table_cMpc_h=jnp.asarray(source['radial_table']),
            modulus_table_h=jnp.asarray(source['modulus_table']),redshift_table=jnp.asarray(source['redshift_table']),
            grid_size=N)
        if os.environ.get('CF4_R2_CUT_BENCHMARK')=='1':
            from cf4_r2_cut_benchmark import run
            run(out,report,save,started,source,mix,optical,richness,density,velocity,
                chosen,point,z,direction,geometry,radius)
            return
        if os.environ.get('CF4_R2_RAW_COHORT')=='1':
            from cf4_r2_raw_cohort_check import run
            run(out,report,save,started,source,mix,optical,richness,density,velocity,
                chosen,point,z,direction,geometry,radius)
            return
        def objective(q,pack,o,density,velocity,mode,population):
            tr=q[:9]; parameters=jnp.asarray(POPULATION_ORIGIN)+jnp.asarray(POPULATION_SCALE)*q[9:24]
            rho=density*jnp.exp(q[25]*mode);rho/=jnp.mean(rho)
            mstar=-23.28+.2*tr[8]; alpha=-1+.06*jnp.exp(.5*tr[7])
            mass=intrinsic_biased_source_masses(rho,
                jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tr[0],
                jnp.exp(.5*tr[1:6]),mstar=mstar,alpha=alpha,reference_interval=(-25.,-21.))
            ids=pack['ids']
            if packed:
                nc=len(ids)//4096
                pos=pack['positions'].reshape(nc,4096,3)
                v=((1+q[24])*velocity[ids]).reshape(nc,4096,3)
                m=mass[:,ids]*pack['mask'][None]*pack['volume'][None]
                m=jnp.moveaxis(m.reshape(5,nc,4096),0,1)
                sky=jnp.moveaxis(pack['angular'].reshape(2,nc,4096),0,1)
                bins=pack['bin'].reshape(nc,4096)
            else:
                nc=ids.shape[0]//CHUNK
                pos=((pack['positions'][:,None,:]+offsets[None])%BOX).reshape(nc,CHUNK*64,3)
                v=jnp.broadcast_to(((1+q[24])*velocity[ids])[:,None,:],(len(ids),64,3)).reshape(nc,CHUNK*64,3)
                m=(mass[:,ids,None]*pack['mask'][None,:,None]*volumes[None,None,:])
                m=jnp.moveaxis(m.reshape(5,nc,CHUNK*64),0,1)
                sky=jnp.moveaxis(jnp.repeat(pack['angular'],64,axis=1).reshape(2,nc,CHUNK*64),0,1)
                bins=None
            score=streaming_raw_mark(parameters,pos,v,m,sky,o,population=population,
                geometry=dict(geometry,mstar=mstar,alpha=alpha,sigma_los_km_s=100*jnp.exp(.5*tr[6])),
                component_bins=bins)
            return score-.5*jnp.vdot(q[:24],q[:24]),score
        evaluate=jax.jit(jax.value_and_grad(objective,has_aux=True),static_argnums=6)
        # The explicit all-empty chunk accumulator must have a finite zero derivative.
        empty=jax.grad(lambda u:jnp.where(jnp.isfinite(logadd_nonempty(-jnp.inf,-jnp.inf)),u,0.))(1.)
        if float(empty)!=0.:raise AssertionError('empty accumulator derivative')
        save();print(json.dumps(report),flush=True)
        gradients=[]
        for row,voxel,ids in zip(rows,voxels,candidates):
            if time.monotonic()-started>1000:raise TimeoutError('bounded live-mark readout')
            ids=np.asarray(sorted(ids),dtype=np.int32); used=len(ids)
            ids=np.pad(ids,(0,width-used),constant_values=ids[0])
            packing_seconds=0.
            if packed:
                packing_start=time.monotonic()
                positions=(source['positions'][ids,None,:]+np.asarray(offsets)[None])%BOX
                positions=positions.reshape(-1,3)
                expanded_ids=np.repeat(ids,64)
                sky=source['angular'][:,expanded_ids]
                # REFRESH union at all three local test states. This is not a
                # permanent candidate cache and must be rebuilt at new field states.
                def weights(q):
                    return predict_source_marked_radial_key_density(jnp.asarray(positions),
                        (1+q[24])*velocity[expanded_ids],jnp.ones((5,len(expanded_ids))),
                        jnp.asarray(sky),int(mix['population'][row]),jnp.asarray(voxel),
                        float(mix['observed_radius'][row]),**geometry,
                        mstar=-23.28+.2*q[8],alpha=-1+.06*jnp.exp(.5*q[7]),
                        sigma_los_km_s=100*jnp.exp(.5*q[6]))
                read_weights=jax.jit(weights)
                positive=np.zeros((5,len(expanded_ids)),dtype=bool)
                for q in (z,z+delta*direction,z-delta*direction):
                    positive|=np.asarray(read_weights(jnp.asarray(q)))>0
                positive[:,used*64:]=False
                bins,points=np.nonzero(positive)
                count=len(points); pad=(-count)%4096
                packed_ids=np.pad(expanded_ids[points],(0,pad),constant_values=expanded_ids[points[0]])
                packed_positions=np.concatenate([positions[points],np.repeat(positions[points[:1]],pad,axis=0)])
                pack=dict(ids=packed_ids,positions=packed_positions,mask=np.arange(count+pad)<count,
                    angular=source['angular'][:,packed_ids],bin=np.pad(bins,(0,pad)).astype(np.int32),
                    volume=np.pad(np.tile(np.asarray(volumes),width)[points],(0,pad)))
                report['support_policy']='refreshed union for current and two finite-difference states; NOT a frozen HMC cache'
                packing_seconds=time.monotonic()-packing_start
            else:
                pack=dict(ids=ids,positions=source['positions'][ids].astype(float),
                    mask=np.arange(width)<used,angular=source['angular'][:,ids])
            pack={k:jnp.asarray(v) for k,v in pack.items()}
            o={k:jnp.asarray(v) for k,v in dict(voxel=voxel,radius=mix['observed_radius'][row],
                dz=mix['dz_row'][row],ksmag=mix['observed_ksmag'][row],x=optical['x'][row],
                error_covariance=optical['optical_error_covariance'][row],richness=richness[row],
                cut_lower=optical['cut_lower'][row],cut_upper=optical['cut_upper'][row]).items()}
            arguments=(pack,o,density,velocity,mode,int(mix['population'][row]))
            compiled=evaluate.lower(jnp.asarray(z),*arguments).compile()
            memory=compiled.memory_analysis().temp_size_in_bytes/1024**3
            if memory>60:raise RuntimeError('live raw-mark temporary memory exceeds bounded pilot')
            start=time.monotonic()
            (value,score),gradient=compiled(jnp.asarray(z),pack,o,density,velocity,mode)
            score=float(score);gradient=np.asarray(gradient);value=float(value)
            seconds=time.monotonic()-start
            plus=float(compiled(jnp.asarray(z+delta*direction),pack,o,density,velocity,mode)[0][0])
            minus=float(compiled(jnp.asarray(z-delta*direction),pack,o,density,velocity,mode)[0][0])
            fd=(plus-minus)/(2*delta);ad=float(gradient@direction)
            entry=dict(row=int(row),PGC=int(mix['PGC'][row]),raw_logpdf=score,
                reference_error=score-reference[row]['raw_logpdf'],
                velocity_derivative=float(gradient[24]),
                velocity_derivative_reference_error=float(gradient[24]-reference[row]['velocity_derivative']),
                joint_direction_analytic=ad,joint_direction_finite_difference=fd,
                joint_direction_relative_error=abs(fd-ad)/max(1.,abs(fd),abs(ad)),
                first_value_gradient_seconds=seconds,device_temporary_GiB=memory,
                packing_seconds=packing_seconds)
            if packed:
                entry.update(positive_components=count,
                    dense_gradient_max_abs_difference=float(np.max(np.abs(gradient-old_gradients[row]))))
            if (not np.isfinite(np.r_[score,gradient,fd]).all() or abs(entry['reference_error'])>1e-7
                or abs(entry['velocity_derivative_reference_error'])>1e-7
                or entry['joint_direction_relative_error']>2e-5):
                report['rows'].append(entry);save();raise AssertionError('live mark value/derivative comparison')
            if packed and entry['dense_gradient_max_abs_difference']>1e-7:
                report['rows'].append(entry);save();raise AssertionError('dense/sparse26-coordinate gradient mismatch')
            report['rows'].append(entry);gradients.append(gradient);save();print(json.dumps(entry),flush=True)
        np.savez(out/'live_derivatives.npz',cohort_rows=rows,gradient=np.array(gradients),
            canonical=z,joint_direction=direction,population_origin=POPULATION_ORIGIN,population_scale=POPULATION_SCALE)
        report['status']='LIVE_RAW_MARK_ADAPTER_CHECKED_NOT_POSTERIOR';save()
    except Exception as error:
        report.update(status='FAILED_LIVE_RAW_MARK_CHECK',error=repr(error));save();raise


if __name__=='__main__':main()
