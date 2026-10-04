"""One field's volume-integrated count and conditional raw-FP observation terms.

Proper priors belong to the caller, once. These terms do not claim selection
calibration or a posterior. Source support must be refreshed at every state;
the state-dependent eight-sigma neighborhood is a numerical approximation.

The LF-shape coordinates in the caller are centred on a 2M++-derived
Lavaux-Hudson reference, while this target also scores 2M++ counts. They are
development regularizers, not independent LF calibration. The active target
integrates only the finite observed K-selection windows; it does not define
the divergent all-faint galaxy total. Its broad alpha coordinate may cross
-1. This is observation-law plumbing only, not an approval to sample before
selection, association and independent/joint calibration are defensible.
"""
from itertools import product
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from cf4_r2_marked_tracer_jax import (intrinsic_biased_source_reference_rates,
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
        alpha=-1+.5*tracer[7],sigma_los_km_s=100*jnp.exp(.5*tracer[6]),
        finite_reference_interval=(-25.,-21.))


def tracer_masses(density,tracer):
    return intrinsic_biased_source_reference_rates(density,
        jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tracer[0],
        jnp.exp(.5*tracer[1:6]))


def raw_field_logpdf(density,velocity,tracer,population_white,packs,source,observation,
                     geometry,*,source_spacing,volume_order=4,block=4096,
                     cut_order=64,cut_integration_axis=0,cut_marginal_tolerance=0.,
                     velocity_variances=None,velocity_closure=None,
                     source_conditioning_radius_cMpc_h=None):
    """All conditional raw marks, no prior and no repeated count occurrence.

    density is cell-centred; velocity is(source,3). Packs contain padded
    source-cell,node,bin,observation IDs and an explicit zero-padding mask.
    ``source_conditioning_radius_cMpc_h`` may separate a linked catalogue
    point's redshift kernel from the CF4 row's mark redshift. Gradients pass
    through every cell's mass/velocity and all24 nuisances.
    """
    if source_conditioning_radius_cMpc_h is None:
        source_conditioning_radius_cMpc_h = observation.get(
            'source_conditioning_radius_cMpc_h')
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
            cut_marginal_tolerance=cut_marginal_tolerance,
            source_velocity_variances_km2_s2=None if velocity_variances is None else velocity_variances[ids].reshape(nc,block,3),
            velocity_closure=velocity_closure,
            source_conditioning_radius_cMpc_h=source_conditioning_radius_cMpc_h)
        num=logadd_nonempty(num,a);den=logadd_nonempty(den,b)
    return num-den


def count_field_loglike(density,velocity,tracer,source,geometry,keys,counts,exposure,
                        *,source_spacing,volume_order=4,los_order=4,los_segments=8,
                        velocity_variances=None,velocity_closure=None):
    mass=tracer_masses(density,tracer);g=tracer_geometry(tracer,geometry)
    def predict(extra):
        return predict_source_volume_intensity(source['positions'],velocity,mass,
            source['angular'],source_spacing=source_spacing,volume_order=volume_order,
            **dict(g,**extra),order=los_order,segments=los_segments)
    if velocity_closure is None:
        intensity=predict({})
    else:
        if velocity_variances is None:raise ValueError('mixture counts require same-field variances')
        from cf4_r2_velocity_closure import mixture_components
        intensity=0.
        for weight,width,scale in mixture_components(velocity_closure):
            intensity+=weight*predict(dict(sigma_los_km_s=width,deposition='voxel_cdf_grid',
                source_velocity_variances_km2_s2=velocity_variances,dispersion_scale=scale))
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
                 volume_order=4,block=4096,max_components=40_000_000,
                 source_conditioning_radius_cMpc_h=None):
        self.positions=np.asarray(positions);self.angular=np.asarray(angular)
        self.population=np.asarray(population,dtype=int);self.o=observation;self.g=geometry
        if source_conditioning_radius_cMpc_h is None:
            source_conditioning_radius_cMpc_h = observation.get(
                'source_conditioning_radius_cMpc_h', observation['radius'])
        self.source_conditioning_radius=np.asarray(
            source_conditioning_radius_cMpc_h,dtype=np.float64)
        if (self.source_conditioning_radius.shape!=(len(self.population),)
                or not np.isfinite(self.source_conditioning_radius).all()
                or np.any(self.source_conditioning_radius<=0)):
            raise ValueError('one finite positive source-conditioning radius per observation required')
        self.box=float(geometry['box_size_cMpc_h']);self.spacing=source_spacing
        self.offsets,_=volume_rule(source_spacing,volume_order);self.block=block
        self.max_components=max_components;self.tree=cKDTree(self.positions%self.box,boxsize=self.box)
        self.nsub=len(self.offsets)
        def weight(vel,tr,pos,sky,voxel,radius,pop):
            return predict_source_marked_radial_key_density(pos,vel,jnp.ones((5,len(pos))),sky,
                pop,voxel,radius,**tracer_geometry(tr,geometry))
        self.weight=jax.jit(weight,static_argnums=6)
        def mixed_weight(vel,pos,sky,voxel,radius,pop,var,core,scale,fraction,tr):
            total=jnp.zeros((5,len(pos)))
            for w,s in ((1-fraction,0.),(fraction,scale)):
                g=dict(tracer_geometry(tr,geometry),sigma_los_km_s=core,
                    deposition='voxel_cdf',source_velocity_variances_km2_s2=var,dispersion_scale=s)
                total+=w*predict_source_marked_radial_key_density(pos,vel,jnp.ones((5,len(pos))),
                    sky,pop,voxel,radius,**g)
            return total
        self.mixed_weight=jax.jit(mixed_weight,static_argnums=5)

    def build(self,velocity,tracer,*,velocity_variances=None,velocity_closure=None):
        tic=time.monotonic();v=np.asarray(velocity);tr=np.asarray(tracer)
        if not np.isfinite(v).all() or not np.isfinite(tr).all():raise ValueError('finite support state required')
        sigma=100*np.exp(.5*tr[6]);conversion=float(self.g['little_h']/self.g['hubble_km_s_Mpc'])
        if velocity_closure is not None:
            var=np.asarray(velocity_variances)
            core=float(velocity_closure['core_sigma_km_s'])
            scale=float(velocity_closure['dispersion_scale'])
            fraction=float(velocity_closure['broad_fraction'])
            if (var.shape!=v.shape or not np.isfinite(var).all() or np.any(var<0)
                    or not np.isfinite([core,scale,fraction]).all()
                    or core<=0 or scale<0 or not 0<fraction<1):
                raise ValueError('finite nonnegative physical variance and valid mixture required')
            # Conservative over ALL subnode ray directions, not a centre-LOS
            # width that can omit a rotated source-volume contribution.
            sigma=np.sqrt(core**2+scale**2*np.max(var,axis=1))
        if not np.all((8*conversion*sigma>0)&(8*conversion*sigma<self.box/2)):
            raise ValueError('LOS width outside existing27-image domain')
        speed=np.linalg.norm(v,axis=1)
        spatial=np.sqrt(3.)*(1.5*self.box/self.g['grid_size']+.5*self.spacing)
        radius=conversion*np.max(speed+8*sigma)+spatial
        centres=(np.asarray(self.o['voxel'])+.5)*self.box/self.g['grid_size']
        candidates=self.tree.query_ball_point(centres,radius,workers=1)
        for row,ids in enumerate(candidates):
            ids=np.asarray(sorted(ids),dtype=np.int32)
            delta=(self.positions[ids]-centres[row]+self.box/2)%self.box-self.box/2
            local_sigma=sigma if np.ndim(sigma)==0 else sigma[ids]
            candidates[row]=ids[np.linalg.norm(delta,axis=1)<=conversion*(speed[ids]+8*local_sigma)+spatial]
        width=((max(map(len,candidates))+63)//64)*64
        if width>32768:raise MemoryError('support exceeds bounded source-cell workspace')
        batches=[{k:[] for k in ('ids','node','bin','row')} for _ in range(6)];total=0
        for row,ids in enumerate(candidates):
            if not len(ids):raise ValueError(f'empty candidate support at row{row}')
            used=len(ids);ids=np.pad(ids,(0,width-used),constant_values=ids[0])
            expanded=np.repeat(ids,self.nsub)
            pos=((self.positions[ids,None]+self.offsets[None])%self.box).reshape(-1,3)
            if velocity_closure is None:
                value=self.weight(jnp.asarray(v[expanded]),jnp.asarray(tr),jnp.asarray(pos),
                    jnp.asarray(self.angular[:,expanded]),jnp.asarray(self.o['voxel'][row]),
                    jnp.asarray(self.source_conditioning_radius[row]),int(self.population[row]))
            else:
                value=self.mixed_weight(jnp.asarray(v[expanded]),jnp.asarray(pos),
                    jnp.asarray(self.angular[:,expanded]),jnp.asarray(self.o['voxel'][row]),
                    jnp.asarray(self.source_conditioning_radius[row]),int(self.population[row]),
                    jnp.asarray(var[expanded]),core,scale,fraction,jnp.asarray(tr))
            positive=np.asarray(value)>0
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
