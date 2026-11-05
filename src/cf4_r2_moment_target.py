"""Moment-aware joint count/raw-FP target. Priors belong to the caller once.

This changes the development observation law, not the existing sampler.
Diagonal particle dispersion is not mean-field posterior uncertainty, and
the closure still requires population/passband/solver calibration.
"""
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_velocity_closure import closure_from_coordinates,active_tracer_coordinates,mixture_components
from cf4_r2_native_to_count_cells import native_moments_to_count_cells
from cf4_r2_raw_volume_target import FreshRawSupport,raw_field_logpdf,tracer_masses,tracer_geometry
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood


class MomentObservationTarget:
    def __init__(self,n,source,population,observation,geometry,keys,counts,exposure,*,
                 volume_orders=(1,2),source_chunk=32768,raw_block=256,cut_order=64):
        self.box=float(geometry['box_size_cMpc_h']);self.spacing=self.box/n
        if source['positions'].shape!=(n**3,3):raise ValueError('native source geometry required')
        self.builders={order:FreshRawSupport(np.asarray(source['positions']),np.asarray(source['angular']),
            population,observation,geometry,source_spacing=self.spacing,volume_order=order,block=raw_block)
            for order in volume_orders}
        self.centred=jax.jit(native_moments_to_count_cells,static_argnums=3)
        # Preserve the old physical rate convention (per 3cMpc/h source cell).
        rate_factor=(self.spacing/3.)**3
        def data(r,v,variance,t8,p,c,packs,source,o,order):
            density,cv,vs=native_moments_to_count_cells(r,v,variance,self.box)
            cv=jnp.moveaxis(cv,0,-1).reshape(-1,3);vs=jnp.moveaxis(vs,0,-1).reshape(-1,3)
            tracer=active_tracer_coordinates(t8);closure=closure_from_coordinates(c)
            masses=tracer_masses(density,tracer)*rate_factor
            intensity=0.
            for weight,width,scale in mixture_components(closure):
                g=dict(tracer_geometry(tracer,geometry),sigma_los_km_s=width,
                    deposition='voxel_cdf_grid',source_velocity_variances_km2_s2=vs,dispersion_scale=scale)
                intensity+=weight*predict_chunked_volume_intensity(source['positions'],cv,masses,source['angular'],
                    source_chunk_size=source_chunk,source_spacing=self.spacing,volume_order=order,**g,
                    order=4,segments=8)
            count=sparse_marked_poisson_log_likelihood(intensity,keys,counts,selected_voxel_mask=exposure)
            raw=raw_field_logpdf(density,cv,tracer,p,packs,source,o,geometry,
                source_spacing=self.spacing,volume_order=order,block=raw_block,cut_order=cut_order,
                velocity_variances=vs,velocity_closure=closure).sum()
            return count+raw,jnp.array([count,raw])
        self.value=jax.jit(data,static_argnums=9)
        self.derivative=jax.jit(jax.value_and_grad(data,argnums=(0,1,2,3,4,5),has_aux=True),static_argnums=9)

    def support(self,r,v,variance,t8,c,order):
        _,cv,vs=self.centred(r,v,variance,self.box)
        return self.builders[order].build(jnp.moveaxis(cv,0,-1).reshape(-1,3),
            active_tracer_coordinates(t8),velocity_variances=jnp.moveaxis(vs,0,-1).reshape(-1,3),
            velocity_closure=closure_from_coordinates(c))
