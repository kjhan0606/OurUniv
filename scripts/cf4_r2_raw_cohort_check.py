"""Whole training cohort observation readout; no PM, fitting or posterior claim."""
from itertools import product
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from cf4_r2_marked_tracer_jax import (predict_source_marked_radial_key_density,
    intrinsic_biased_source_masses,intrinsic_lf_bin_fractions)
from cf4_r2_raw_live_mark import (streaming_raw_mark,logadd_nonempty,
    POPULATION_ORIGIN,POPULATION_SCALE)

BASE=Path('/gpfs/kjhan/CF4/z0_density'); BOX=384.;N=128;BLOCK=4096


def run(out,report,save,started,source,mix,optical,richness,density,velocity,
        chosen,point,z,direction,geometry,radius):
    n=len(mix['PGC']);eps=2e-5
    if n!=1414:raise ValueError('frozen1414-row cohort required')
    report.update(status='PREPARING_WHOLE_TRAINING_COHORT',cohort_rows=n,
        support_policy='fresh union at the current and two finite-difference states; not reusable HMC cache',
        finite_difference_step=eps,prepared_rows=0)
    voxels=np.array([np.unravel_index(point['flat_cell'][o[2]],(N,)*3) for o in chosen])
    tree=cKDTree(source['positions']%BOX,boxsize=BOX)
    candidates=tree.query_ball_point((voxels+.5)*BOX/N,radius,workers=1)
    width=((max(map(len,candidates))+63)//64)*64
    t,w=np.polynomial.legendre.leggauss(4)
    nodes=np.array(list(product(range(4),repeat=3)))
    offsets=t[nodes]*1.5;volumes=np.prod(w[nodes]/2,axis=1)
    def weight(q,positions,vel,sky,voxel,r,population):
        return predict_source_marked_radial_key_density(positions,(1+q[24])*vel,
            jnp.ones((5,len(positions))),sky,population,voxel,r,**geometry,
            mstar=-23.28+.2*q[8],alpha=-1+.06*jnp.exp(.5*q[7]),
            sigma_los_km_s=100*jnp.exp(.5*q[6]))
    # Dynamic positions/velocities/metadata; six population compilations are
    # reused, rather than compiling a new closure per observed galaxy.
    weight=jax.jit(weight,static_argnums=6)
    batches=[{k:[] for k in ('ids','node','bin','row')} for _ in range(6)]
    components=0;packing_start=time.monotonic();save()
    for row,ids in enumerate(candidates):
        if time.monotonic()-started>2200:raise TimeoutError('bounded cohort preparation/readout')
        ids=np.asarray(sorted(ids),dtype=np.int32);used=len(ids)
        ids=np.pad(ids,(0,width-used),constant_values=ids[0])
        expanded=np.repeat(ids,64)
        pos=(source['positions'][ids,None,:]+offsets[None])%BOX
        arguments=(jnp.asarray(pos.reshape(-1,3)),velocity[expanded],
            jnp.asarray(source['angular'][:,expanded]),jnp.asarray(voxels[row]),
            jnp.asarray(mix['observed_radius'][row]),int(mix['population'][row]))
        positive=np.zeros((5,len(expanded)),dtype=bool)
        for q in (z,z+eps*direction,z-eps*direction):
            positive|=np.asarray(weight(jnp.asarray(q),*arguments))>0
        positive[:,used*64:]=False
        bins,indices=np.nonzero(positive)
        if not len(indices):raise ValueError(f'empty raw source support at row{row}')
        b=batches[int(mix['population'][row])]
        b['ids'].append(expanded[indices]);b['node'].append((indices%64).astype(np.uint8))
        b['bin'].append(bins.astype(np.int8));b['row'].append(np.full(len(indices),row,dtype=np.int32))
        components+=len(indices)
        if components>40_000_000:raise MemoryError('bounded40-million component payload exceeded')
        if (row+1)%100==0 or row+1==n:
            report.update(prepared_rows=row+1,positive_components=components,
                packing_seconds=time.monotonic()-packing_start);save()
            print(json.dumps({k:report[k] for k in ('prepared_rows','positive_components','packing_seconds')}),flush=True)
    del candidates,tree,positive,arguments
    packed=[];archive={}
    for p,b in enumerate(batches):
        count=sum(map(len,b['ids']));pad=(-count)%BLOCK
        payload={k:np.concatenate(v) for k,v in b.items()}
        for k,v in payload.items():archive[f'p{p}_{k}']=v
        payload={k:np.pad(v,(0,pad),constant_values=v[0]) for k,v in payload.items()}
        payload['mask']=np.arange(count+pad)<count
        packed.append({k:jnp.asarray(v) for k,v in payload.items()})
    np.savez_compressed(out/'current_state_component_ids.npz',**archive,
        PGC=mix['PGC'],canonical=z,finite_difference_direction=direction,
        finite_difference_step=eps,quadrature_nodes_per_axis=4)
    del archive,batches,payload
    observations={k:jnp.asarray(v) for k,v in dict(voxel=voxels,radius=mix['observed_radius'],
        dz=mix['dz_row'],ksmag=mix['observed_ksmag'],x=optical['x'],
        error_covariance=optical['optical_error_covariance'],richness=richness,
        cut_lower=optical['cut_lower'],cut_upper=optical['cut_upper']).items()}
    source_positions=jnp.asarray(source['positions']);source_angular=jnp.asarray(source['angular'])
    modulation=jnp.cos(2*jnp.pi*source_positions[:,0]/BOX).reshape(density.shape)
    # Match the already checked direction, which was computed in archive
    # precision with NumPy then promoted. Do not change it at the comparison.
    modulation=jnp.asarray(np.cos(2*np.pi*source['positions'][:,0]/BOX).reshape(density.shape),dtype=density.dtype)
    def objective(q,selected,packed,density,velocity,positions,angular,modulation,o):
        tr=q[:9];params=jnp.asarray(POPULATION_ORIGIN)+jnp.asarray(POPULATION_SCALE)*q[9:24]
        rho=density*jnp.exp(q[25]*modulation);rho/=jnp.mean(rho)
        mstar=-23.28+.2*tr[8];alpha=-1+.06*jnp.exp(.5*tr[7])
        masses=intrinsic_biased_source_masses(rho,
            jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tr[0],jnp.exp(.5*tr[1:6]),
            mstar=mstar,alpha=alpha,reference_interval=(-25.,-21.))
        g=dict(geometry,mstar=mstar,alpha=alpha,sigma_los_km_s=100*jnp.exp(.5*tr[6]))
        numerator=jnp.full(n,-jnp.inf);denominator=jnp.full(n,-jnp.inf)
        for p,pack in enumerate(packed):
            ids=pack['ids'];node=pack['node'].astype(jnp.int32);nc=len(ids)//BLOCK
            pos=((positions[ids]+jnp.asarray(offsets)[node])%BOX).reshape(nc,BLOCK,3)
            vel=((1+q[24])*velocity[ids]).reshape(nc,BLOCK,3)
            mass=masses[:,ids]*jnp.asarray(volumes)[node][None]*pack['mask'][None]
            mass=jnp.moveaxis(mass.reshape(5,nc,BLOCK),0,1)
            sky=jnp.moveaxis(angular[:,ids].reshape(2,nc,BLOCK),0,1)
            a,b=streaming_raw_mark(params,pos,vel,mass,sky,o,population=p,geometry=g,
                component_bins=pack['bin'].reshape(nc,BLOCK),
                component_rows=pack['row'].reshape(nc,BLOCK),return_log_terms=True)
            numerator=logadd_nonempty(numerator,a);denominator=logadd_nonempty(denominator,b)
        values=numerator-denominator
        return jnp.sum(jnp.where(selected,values,0.))-.5*jnp.vdot(q[:24],q[:24]),values
    report.update(status='COMPILING_WHOLE_COHORT_READOUT',positive_components=components);save()
    evaluate=jax.jit(jax.value_and_grad(objective,has_aux=True))
    all_rows=jnp.ones(n,dtype=bool)
    args=(tuple(packed),density,velocity,source_positions,source_angular,modulation,observations)
    compiled=evaluate.lower(jnp.asarray(z),all_rows,*args).compile()
    memory=compiled.memory_analysis().temp_size_in_bytes/1024**3
    report['device_temporary_GiB']=memory;save()
    if memory>60:raise MemoryError('cohort temporary GPU memory exceeds60GiB bound')
    with np.load(BASE/'r2_raw_live_check_v1/live_derivatives.npz',allow_pickle=False) as f:
        reference_rows=f['cohort_rows'].copy();reference_gradient=f['gradient'].sum(axis=0)
    reference_gradient[:24]+=(len(reference_rows)-1)*z[:24] # prior ONCE, not once per row
    mask=np.zeros(n,dtype=bool);mask[reference_rows]=True
    (_,values),gradient=compiled(jnp.asarray(z),jnp.asarray(mask),*args)
    gradient=np.asarray(gradient);values=np.asarray(values)
    discrepancy=float(np.max(np.abs(gradient-reference_gradient)))
    report['seven_row_dense_gradient_max_abs_difference']=discrepancy;save()
    if not np.isfinite(values).all() or discrepancy>1e-7:
        raise AssertionError('whole-cohort adapter fails dense-reference subgroup comparison')
    evaluated_at=time.monotonic()
    (value,values),gradient=compiled(jnp.asarray(z),all_rows,*args)
    value=float(value);values=np.asarray(values);gradient=np.asarray(gradient)
    report['whole_cohort_value_gradient_seconds']=time.monotonic()-evaluated_at
    plus=float(compiled(jnp.asarray(z+eps*direction),all_rows,*args)[0][0])
    minus=float(compiled(jnp.asarray(z-eps*direction),all_rows,*args)[0][0])
    fd=(plus-minus)/(2*eps);ad=float(gradient@direction)
    error=abs(fd-ad)/max(1.,abs(fd),abs(ad))
    report.update(raw_logpdf_sum=float(values.sum()),joint_direction_analytic=ad,
        joint_direction_finite_difference=fd,joint_direction_relative_error=error)
    save()
    if not np.isfinite(np.r_[value,gradient,values]).all() or error>2e-5:
        raise AssertionError('whole-cohort directional derivative failed')
    np.savez(out/'whole_cohort_readout.npz',PGC=mix['PGC'],raw_logpdf=values,
        canonical=z,gradient=gradient,joint_direction=direction)
    report['status']='WHOLE_COHORT_LIVE_RAW_MARK_CHECKED_NOT_POSTERIOR';save()
    print(json.dumps(report),flush=True)
