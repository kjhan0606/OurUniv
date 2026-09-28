"""Bounded same-state cut quadrature comparison; never a reusable HMC cache."""
from itertools import product
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_marked_tracer_jax import intrinsic_biased_source_masses,intrinsic_lf_bin_fractions
from cf4_r2_raw_live_mark import streaming_raw_mark,logadd_nonempty,POPULATION_ORIGIN,POPULATION_SCALE

BASE=Path('/gpfs/kjhan/CF4/z0_density');BLOCK=4096;BOX=384.;N=128


def run(out,report,save,started,source,mix,optical,richness,density,velocity,
        chosen,point,z,direction,geometry,radius):
    previous=json.loads((BASE/'r2_source_cell_quadrature_v2/result.json').read_text())
    projection=optical['x']@np.array([[2.,0.,1.],[.04,1.,0.]]).T
    edge=np.r_[np.argmin(projection-optical['cut_lower'],axis=0),
               np.argmin(optical['cut_upper']-projection,axis=0)]
    rows=np.unique(np.r_[previous['order8_rows'],edge]).astype(int)
    n=len(rows);eps=2e-5
    remap=np.full(len(mix['PGC']),-1,dtype=np.int32);remap[rows]=np.arange(n)
    t,w=np.polynomial.legendre.leggauss(4);nodes=np.array(list(product(range(4),repeat=3)))
    offsets=jnp.asarray(t[nodes]*1.5);volumes=jnp.asarray(np.prod(w[nodes]/2,axis=1))
    packs=[]
    with np.load(BASE/'r2_raw_live_cohort_v1/current_state_component_ids.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['PGC'],mix['PGC'])
        np.testing.assert_array_equal(f['canonical'],z)
        np.testing.assert_array_equal(f['finite_difference_direction'],direction)
        assert float(f['finite_difference_step'])==eps and int(f['quadrature_nodes_per_axis'])==4
        for p in range(6):
            oldrow=f[f'p{p}_row'];keep=remap[oldrow]>=0
            payload={k:f[f'p{p}_{k}'][keep] for k in ('ids','node','bin')}
            payload['row']=remap[oldrow[keep]]
            count=len(payload['row'])
            if not count:continue
            pad=(-count)%BLOCK
            payload={k:np.pad(v,(0,pad),constant_values=v[0]) for k,v in payload.items()}
            payload['mask']=np.arange(count+pad)<count
            packs.append((p,{k:jnp.asarray(v) for k,v in payload.items()}))
    voxels=np.array([np.unravel_index(point['flat_cell'][chosen[i][2]],(N,)*3) for i in rows])
    observation={k:jnp.asarray(v) for k,v in dict(voxel=voxels,radius=mix['observed_radius'][rows],
        dz=mix['dz_row'][rows],ksmag=mix['observed_ksmag'][rows],x=optical['x'][rows],
        error_covariance=optical['optical_error_covariance'][rows],richness=richness[rows],
        cut_lower=optical['cut_lower'][rows],cut_upper=optical['cut_upper'][rows]).items()}
    positions=jnp.asarray(source['positions']);angular=jnp.asarray(source['angular'])
    mode=jnp.asarray(np.cos(2*np.pi*source['positions'][:,0]/BOX).reshape(density.shape),dtype=density.dtype)
    report.update(status='CUT_QUADRATURE_BENCHMARK',cohort_rows=rows.tolist(),
        PGC=mix['PGC'][rows].tolist(),edge_rows=edge.tolist(),
        support_policy='reuse only exact saved field and identical two finite-difference states',
        numerical_scope='axis1 order256 full correlated rule versus certified marginal at1e-12; original axis0/64 reported separately')
    save()
    def objective(q,cut_order,axis,tolerance):
        tr=q[:9];params=jnp.asarray(POPULATION_ORIGIN)+jnp.asarray(POPULATION_SCALE)*q[9:24]
        rho=density*jnp.exp(q[25]*mode);rho/=jnp.mean(rho)
        mstar=-23.28+.2*tr[8];alpha=-1+.06*jnp.exp(.5*tr[7])
        masses=intrinsic_biased_source_masses(rho,
            jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tr[0],jnp.exp(.5*tr[1:6]),
            mstar=mstar,alpha=alpha,reference_interval=(-25.,-21.))
        g=dict(geometry,mstar=mstar,alpha=alpha,sigma_los_km_s=100*jnp.exp(.5*tr[6]))
        numerator=jnp.full(n,-jnp.inf);denominator=jnp.full(n,-jnp.inf)
        for p,pack in packs:
            ids=pack['ids'];node=pack['node'].astype(jnp.int32);nc=len(ids)//BLOCK
            pos=((positions[ids]+offsets[node])%BOX).reshape(nc,BLOCK,3)
            vel=((1+q[24])*velocity[ids]).reshape(nc,BLOCK,3)
            mass=masses[:,ids]*volumes[node][None]*pack['mask'][None]
            mass=jnp.moveaxis(mass.reshape(5,nc,BLOCK),0,1)
            sky=jnp.moveaxis(angular[:,ids].reshape(2,nc,BLOCK),0,1)
            a,b=streaming_raw_mark(params,pos,vel,mass,sky,observation,population=p,geometry=g,
                component_bins=pack['bin'].reshape(nc,BLOCK),component_rows=pack['row'].reshape(nc,BLOCK),
                return_log_terms=True,cut_order=cut_order,cut_integration_axis=axis,cut_marginal_tolerance=tolerance)
            numerator=logadd_nonempty(numerator,a);denominator=logadd_nonempty(denominator,b)
        values=numerator-denominator
        return values.sum()-.5*jnp.vdot(q[:24],q[:24]),values
    results={};arrays={}
    for label,order,axis,tol in [('reference',256,1,0.),('fast',256,1,1e-12),('legacy',64,0,0.)]:
        if time.monotonic()-started>2100:raise TimeoutError('bounded cut benchmark')
        report['phase']=label;save()
        func=lambda q:objective(q,order,axis,tol)
        compiled=jax.jit(jax.value_and_grad(func,has_aux=True)).lower(jnp.asarray(z)).compile()
        memory=compiled.memory_analysis().temp_size_in_bytes/1024**3
        if memory>60:raise MemoryError('cut benchmark temporary memory bound')
        begin=time.monotonic();(value,values),gradient=compiled(jnp.asarray(z))
        value=float(value);values=np.asarray(values);gradient=np.asarray(gradient)
        seconds=time.monotonic()-begin
        if not np.isfinite(np.r_[value,values,gradient]).all():raise AssertionError('nonfinite cut readout')
        results[label]=dict(value=value,value_gradient_seconds=seconds,device_temporary_GiB=memory)
        arrays[label+'_values']=values;arrays[label+'_gradient']=gradient
        if label=='fast':
            plus=float(compiled(jnp.asarray(z+eps*direction))[0][0])
            minus=float(compiled(jnp.asarray(z-eps*direction))[0][0])
            fd=(plus-minus)/(2*eps);ad=float(gradient@direction)
            results[label].update(directional_AD=ad,directional_FD=fd,
                directional_relative_error=abs(fd-ad)/max(1.,abs(fd),abs(ad)))
        report['results']=results;save();print(json.dumps(results[label]),flush=True)
        del compiled
        jax.clear_caches()
    value_error=float(np.max(np.abs(arrays['fast_values']-arrays['reference_values'])))
    grad_error=float(np.max(np.abs(arrays['fast_gradient']-arrays['reference_gradient'])))
    report.update(fast_reference_value_max_abs=value_error,fast_reference_gradient_max_abs=grad_error,
        legacy_reference_value_max_abs=float(np.max(np.abs(arrays['legacy_values']-arrays['reference_values']))))
    np.savez(out/'readouts.npz',cohort_rows=rows,PGC=mix['PGC'][rows],canonical=z,**arrays)
    save()
    if value_error>1e-7 or grad_error>1e-7 or results['fast']['directional_relative_error']>2e-5:
        raise AssertionError('certified marginal empirical value/gradient controls failed')
    report['status']='CUT_SHORTCUT_CHECKED_NOT_POSTERIOR';save()
