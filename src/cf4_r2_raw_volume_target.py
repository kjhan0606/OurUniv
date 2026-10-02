"""One field's volume-integrated count and conditional raw-FP observation terms.

Proper priors belong to the caller, once. These terms do not claim selection
calibration or a posterior. Source support must be refreshed at every state;
the state-dependent eight-sigma neighborhood is a numerical approximation.

The LF-shape coordinates in the caller are centred on a 2M++-derived
Lavaux-Hudson reference, while this target also scores 2M++ counts. They are
development regularizers, not independent LF calibration. In particular,
the alpha > -1 transform is a consequence of the current unbounded faint bin,
not a physically established cutoff; revise the tail and selection law
together before future posterior sampling.
"""
from itertools import product
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from cf4_r2_marked_tracer_jax import (intrinsic_biased_source_masses,
    intrinsic_lf_bin_fractions,predict_source_marked_radial_key_density,
    sparse_marked_poisson_log_likelihood)
from cf4_r2_raw_live_mark import (streaming_raw_mark,logadd_nonempty,
    POPULATION_ORIGIN,POPULATION_SCALE)
from cf4_r2_shell_cdf_count import predict_source_volume_intensity


def volume_rule(spacing,order):
    if spacing<=0 or order<1:raise ValueError('positive source spacing and rule required')
    t,w=np.polynomial.legendre.leggauss(order)
    nodes=np.array(list(product(range(order),repeat=3)))
    return t[nodes]*spacing/2,np.prod(w[nodes]/2,axis=1)


def tracer_geometry(tracer,geometry):
    return dict(geometry,mstar=-23.28+.2*tracer[8],
        alpha=-1+.06*jnp.exp(.5*tracer[7]),sigma_los_km_s=100*jnp.exp(.5*tracer[6]))


def tracer_masses(density,tracer):
    return intrinsic_biased_source_masses(density,
        jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tracer[0],
        jnp.exp(.5*tracer[1:6]),mstar=-23.28+.2*tracer[8],
        alpha=-1+.06*jnp.exp(.5*tracer[7]),reference_interval=(-25.,-21.))


def raw_field_logpdf(density,velocity,tracer,population_white,packs,source,observation,
                     geometry,*,source_spacing,volume_order=4,block=4096,
                     cut_order=64,cut_integration_axis=0,cut_marginal_tolerance=0.):
    """All conditional raw marks, no prior and no repeated count occurrence.

    density is cell-centred; velocity is(source,3). Packs contain padded
    source-cell,node,bin,observation IDs and an explicit zero-padding mask.
    Gradients pass through every cell's mass/velocity and all24 nuisances.
    """
    offsets,weights=map(jnp.asarray,volume_rule(source_spacing,volume_order))
    masses=tracer_masses(density,tracer);g=tracer_geometry(tracer,geometry)
    parameters=jnp.asarray(POPULATION_ORIGIN)+jnp.asarray(POPULATION_SCALE)*population_white
    n=observation['x'].shape[0];num=jnp.full(n,-jnp.inf);den=jnp.full(n,-jnp.inf)
    for p,pack in enumerate(packs):
        ids=pack['ids'];node=pack['node'].astype(jnp.int32)
        if len(ids)==0:continue
        if len(ids)%block:raise ValueError('raw support padding is not block-aligned')
        nc=len(ids)//block
        pos=((source['positions'][ids]+offsets[node])%geometry['box_size_cMpc_h']).reshape(nc,block,3)
        vel=velocity[ids].reshape(nc,block,3)
        mass=masses[:,ids]*weights[node][None]*pack['mask'][None]
        mass=jnp.moveaxis(mass.reshape(5,nc,block),0,1)
        sky=jnp.moveaxis(source['angular'][:,ids].reshape(2,nc,block),0,1)
        a,b=streaming_raw_mark(parameters,pos,vel,mass,sky,observation,population=p,geometry=g,
            component_bins=pack['bin'].reshape(nc,block),component_rows=pack['row'].reshape(nc,block),
            return_log_terms=True,cut_order=cut_order,cut_integration_axis=cut_integration_axis,
            cut_marginal_tolerance=cut_marginal_tolerance)
        num=logadd_nonempty(num,a);den=logadd_nonempty(den,b)
    return num-den


def count_field_loglike(density,velocity,tracer,source,geometry,keys,counts,exposure,
                        *,source_spacing,volume_order=4,los_order=4,los_segments=8):
    intensity=predict_source_volume_intensity(source['positions'],velocity,tracer_masses(density,tracer),
        source['angular'],source_spacing=source_spacing,volume_order=volume_order,
        **tracer_geometry(tracer,geometry),order=los_order,segments=los_segments)
    return sparse_marked_poisson_log_likelihood(intensity,keys,counts,selected_voxel_mask=exposure)


class FreshRawSupport:
    """Deterministic state-local support; static tree is NOT a frozen field.

    The radius bounds cell displacement+LOS tail+TSC+cell subnode extent.
    Every positive bin/subnode inside it is retained. No density/mark score
    selects the support. Rebuild on each state, including Metropolis energies.
    Eight-sigma omission remains a numerical limitation, not an exact tail
    proof for normalized marks. Host/device ceilings stop, never truncate.
    """
    def __init__(self,positions,angular,population,observation,geometry,*,source_spacing,
                 volume_order=4,block=4096,max_components=40_000_000):
        self.positions=np.asarray(positions);self.angular=np.asarray(angular)
        self.population=np.asarray(population,dtype=int);self.o=observation;self.g=geometry
        self.box=float(geometry['box_size_cMpc_h']);self.spacing=source_spacing
        self.offsets,_=volume_rule(source_spacing,volume_order);self.block=block
        self.max_components=max_components;self.tree=cKDTree(self.positions%self.box,boxsize=self.box)
        self.nsub=len(self.offsets)
        def weight(vel,tr,pos,sky,voxel,radius,pop):
            return predict_source_marked_radial_key_density(pos,vel,jnp.ones((5,len(pos))),sky,
                pop,voxel,radius,**tracer_geometry(tr,geometry))
        self.weight=jax.jit(weight,static_argnums=6)

    def build(self,velocity,tracer):
        tic=time.monotonic();v=np.asarray(velocity);tr=np.asarray(tracer)
        if not np.isfinite(v).all() or not np.isfinite(tr).all():raise ValueError('finite support state required')
        sigma=100*np.exp(.5*tr[6]);conversion=float(self.g['little_h']/self.g['hubble_km_s_Mpc'])
        if not 0<8*conversion*sigma<self.box/2:raise ValueError('LOS width outside existing27-image domain')
        speed=np.linalg.norm(v,axis=1)
        spatial=np.sqrt(3.)*(1.5*self.box/self.g['grid_size']+.5*self.spacing)
        radius=conversion*(speed.max()+8*sigma)+spatial
        centres=(np.asarray(self.o['voxel'])+.5)*self.box/self.g['grid_size']
        candidates=self.tree.query_ball_point(centres,radius,workers=1)
        for row,ids in enumerate(candidates):
            ids=np.asarray(sorted(ids),dtype=np.int32)
            delta=(self.positions[ids]-centres[row]+self.box/2)%self.box-self.box/2
            candidates[row]=ids[np.linalg.norm(delta,axis=1)<=conversion*(speed[ids]+8*sigma)+spatial]
        width=((max(map(len,candidates))+63)//64)*64
        if width>32768:raise MemoryError('support exceeds bounded source-cell workspace')
        batches=[{k:[] for k in ('ids','node','bin','row')} for _ in range(6)];total=0
        for row,ids in enumerate(candidates):
            if not len(ids):raise ValueError(f'empty candidate support at row{row}')
            used=len(ids);ids=np.pad(ids,(0,width-used),constant_values=ids[0])
            expanded=np.repeat(ids,self.nsub)
            pos=((self.positions[ids,None]+self.offsets[None])%self.box).reshape(-1,3)
            positive=np.asarray(self.weight(jnp.asarray(v[expanded]),jnp.asarray(tr),jnp.asarray(pos),
                jnp.asarray(self.angular[:,expanded]),jnp.asarray(self.o['voxel'][row]),
                jnp.asarray(self.o['radius'][row]),int(self.population[row])))>0
            positive[:,used*self.nsub:]=False
            bins,indices=np.nonzero(positive)
            if not len(indices):raise ValueError(f'zero selected support at row{row}')
            total+=len(indices)
            if total>self.max_components:raise MemoryError('bounded raw component workspace exceeded')
            b=batches[self.population[row]]
            b['ids'].append(expanded[indices]);b['node'].append((indices%self.nsub).astype(np.int32))
            b['bin'].append(bins.astype(np.int32));b['row'].append(np.full(len(indices),row,dtype=np.int32))
        packs=[]
        for b in batches:
            payload={k:np.concatenate(vals) if vals else np.empty(0,dtype=np.int32) for k,vals in b.items()}
            count=len(payload['ids']);pad=(-count)%self.block
            payload={k:np.pad(v,(0,pad),constant_values=v[0] if count else 0) for k,v in payload.items()}
            payload['mask']=np.arange(count+pad)<count
            packs.append({k:jnp.asarray(v) for k,v in payload.items()})
        return tuple(packs),dict(components=total,max_cells=width,radius=radius,
            packing_seconds=time.monotonic()-tic,source_volume_order=round(self.nsub**(1/3)))
