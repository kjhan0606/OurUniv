"""Resolution-aware wiring of the existing raw/count observation law.

Source-field resolution may change; observed keys/exposure and the physical
rate prior must not. No priors, identity labels or heldout scores are added.
"""
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_volume_target import (FreshRawSupport,raw_field_logpdf,
    tracer_masses,tracer_geometry)
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood


def linked_point_conditioning_radius(observation):
    """Validate the per-row 2M++ radius used by the conditional raw-mark law."""
    if 'source_conditioning_radius_cMpc_h' not in observation:
        raise ValueError('linked-point source-conditioning radii are required')
    radius = np.asarray(observation['source_conditioning_radius_cMpc_h'], dtype=np.float64)
    observed_radius = np.asarray(observation['radius'], dtype=np.float64)
    if (radius.shape != observed_radius.shape or radius.ndim != 1
            or not np.isfinite(radius).all() or np.any(radius <= 0.)):
        raise ValueError('linked-point source-conditioning radii must align and be positive')
    return radius


def source_geometry_at_resolution(source,n,box=384.):
    """Refine the declared piecewise-constant angular selection, not data.

    Angular completeness remains the original128-grid selection model.
    It is neither a new high-resolution survey mask nor a density constraint.
    Positions are physical source-cell centres; observed keys remain128-grid.
    """
    if n not in (128,256) or box!=384.:
        raise ValueError('only the declared128/256 source geometry is supported')
    positions=np.asarray(source['positions'])
    angular=np.asarray(source['angular'])
    if positions.shape!=(128**3,3) or angular.shape!=(2,128**3):
        raise ValueError('original128-grid source geometry required')
    expected=(np.indices((128,)*3,dtype=np.int16).reshape(3,-1).T.astype(np.float32)+.5)*3.
    if not np.array_equal(positions,expected):
        raise ValueError('original source ordering/origin mismatch')
    if n==128:return dict(positions=jnp.asarray(positions),angular=jnp.asarray(angular))
    sky=angular.reshape(2,128,128,128)
    for axis in (1,2,3):sky=np.repeat(sky,2,axis=axis)
    coords=(np.indices((n,)*3,dtype=np.int16).reshape(3,-1).T.astype(np.float32)+.5)*(box/n)
    return dict(positions=jnp.asarray(coords),angular=jnp.asarray(sky.reshape(2,-1)))


class ResolutionObservationTarget:
    """Same numerical law at an explicitly declared source quadrature.

    N128 rules2/4 and N256 rules1/2 have comparable source-node counts, NOT
    identical target values. Fine numerical sensitivity remains a science
    requirement. Coarse forces never define the Metropolis target.
    """
    def __init__(self,n,source,mix,observation,geometry,keys,counts,exposure,*,
                 force_order,fine_order,source_chunk=2097152,max_support_cells=32768):
        if n not in (128,256) or geometry['grid_size']!=128:
            raise ValueError('native128/256 fields and frozen observed128 grid required')
        if not 1<=force_order<fine_order or source_chunk<1:
            raise ValueError('declared force/fine rules and positive batch required')
        if source['positions'].shape!=(n**3,3) or source['angular'].shape!=(2,n**3):
            raise ValueError('aligned source geometry required')
        linked_point_conditioning_radius(observation)
        self.n=n;self.box=float(geometry['box_size_cMpc_h']);self.spacing=self.box/n
        self.source=source;self.observation=observation
        self.rate_volume_factor=(self.spacing/3.)**3
        self.builders={order:FreshRawSupport(np.asarray(source['positions']),np.asarray(source['angular']),
            mix['population'],observation,geometry,source_spacing=self.spacing,volume_order=order,
            max_support_cells=max_support_cells)
            for order in (force_order,fine_order)}
        self.centred=jax.jit(native_mass_momentum_to_count_cells,static_argnums=2)
        def data(r,v,t,p,packs,source,o,order):
            density,cv=native_mass_momentum_to_count_cells(r,v,self.box)
            cv=jnp.moveaxis(cv,0,-1).reshape(-1,3)
            intensity=predict_chunked_volume_intensity(source['positions'],cv,
                tracer_masses(density,t)*self.rate_volume_factor,source['angular'],
                source_chunk_size=source_chunk,source_spacing=self.spacing,volume_order=order,
                **tracer_geometry(t,geometry),order=4,segments=8)
            count=sparse_marked_poisson_log_likelihood(intensity,keys,counts,selected_voxel_mask=exposure)
            # The constant cell-volume/rate factor cancels between numerator
            # and denominator of each CONDITIONAL raw mark, not in counts.
            raw=raw_field_logpdf(density,cv,t,p,packs,source,o,geometry,
                source_spacing=self.spacing,volume_order=order,cut_order=256,
                cut_integration_axis=1,cut_marginal_tolerance=1e-12,
                source_conditioning_radius_cMpc_h=o['source_conditioning_radius_cMpc_h']).sum()
            return count+raw,jnp.array([count,raw])
        self.value=jax.jit(data,static_argnums=7)
        self.derivative=jax.jit(jax.value_and_grad(data,argnums=(0,1,2,3),has_aux=True),static_argnums=7)
        def component(r,v,t,p,packs,source,o,order,component_index):
            return data(r,v,t,p,packs,source,o,order)[1][component_index]
        self.component_derivative=jax.jit(
            jax.value_and_grad(component,argnums=(0,1,2,3)),static_argnums=(7,8))

    def support(self,r,v,t,order):
        _,cv=self.centred(r,v,self.box)
        packs,info=self.builders[order].build(jnp.moveaxis(cv,0,-1).reshape(-1,3),t)
        quantum=262144 if order==max(self.builders) else 65536
        result=[]
        for pack in packs:
            padding=(-len(pack['ids']))%quantum
            result.append({k:jnp.pad(x,(0,padding),constant_values=False if k=='mask' else 0)
                           for k,x in pack.items()})
        return tuple(result),info
