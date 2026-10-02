"""Observed-direction source integral prototype, NOT periodic production.

Piecewise-constant source-cell rates/mean/diagonal variance. Exact cell
crossings and Gaussian-CDF source-radius integration remove discrete angular
rays. Caller must PROVE boundary-wrap contributions cannot reach the datum;
otherwise this omits terms and is only a labelled diagnostic. No likelihood
floor, native identities or observed-redshift likelihood is introduced.
"""
import jax.numpy as jnp
import numpy as np
from cf4_r2_shell_cdf_count import _probability_nodes
from cf4_r2_velocity_closure import mixture_components,conditional_los_sigma
from cf4_r2_marked_tracer_jax import source_mark_transfer


def source_ray_intervals(direction,observer,n,box):
    """All physical minimum-image cube crossings along signed source radius.

    Domain length may exceed box on diagonal rays, but EACH axis spans<=box,
    so n+2 planes per axis cover all crossings without a radius budget.
    """
    direction=jnp.asarray(direction)
    exit_radius=box/2/jnp.max(jnp.abs(direction))
    lo=-exit_radius;hi=exit_radius;dx=box/n
    parallel=direction==0;den=jnp.where(parallel,1.,direction)
    first=jnp.floor((observer+lo*direction)/dx)
    index=jnp.arange(n+2)[None,:]
    planes=first[:,None]+jnp.where(direction[:,None]>0,1+index,-index)
    crossing=(planes*dx-observer[:,None])/den[:,None]
    crossing=jnp.where(parallel[:,None],hi,crossing)
    events=jnp.sort(jnp.concatenate((jnp.array([lo,hi]),jnp.clip(crossing,lo,hi).ravel())))
    lower,upper=events[:-1],events[1:]
    mid=(lower+upper)/2
    pos=(observer+mid[:,None]*direction)%box
    cell=jnp.floor(pos/dx).astype(jnp.int32)%n
    flat=cell[:,0]*n*n+cell[:,1]*n+cell[:,2]
    return lower,upper,flat


def observed_ray_components(direction,observed_radius,velocity,variance,masses,angular,
                            population,geometry,closure,*,source_grid,order=4):
    """Quadrature source positions and UNNORMALIZED selected radial masses.

    One same integrand can later define observed-volume counts via f/r².
    This function is not authorization to replace the active count backend.
    """
    box=geometry['box_size_cMpc_h'];observer=geometry['observer']
    lo,hi,ids=source_ray_intervals(direction,observer,source_grid,box)
    vel=velocity[ids];var=variance[ids]
    conversion=geometry['little_h']/geometry['hubble_km_s_Mpc']
    centre=observed_radius-conversion*jnp.sum(vel*direction[None],axis=1)
    nodes=[];weights=[];cell_ids=[]
    for fraction,core,scale in mixture_components(closure):
        sigma=conversion*conditional_los_sigma(jnp.broadcast_to(direction,var.shape),core,var,scale)
        lower=jnp.maximum(lo,centre-8*sigma);upper=jnp.minimum(hi,centre+8*sigma)
        active=upper>lower
        q,w=_probability_nodes(centre,sigma,lower[None],upper[None],active[None],order)
        q=q[0].reshape(-1);w=w[0].reshape(-1)*fraction
        # Zero padding is finite through the optical/eta kernel, not0*NaN.
        q=jnp.where((w>0)&(q!=0),q,observed_radius)
        nodes.append(q);weights.append(w);cell_ids.append(jnp.tile(ids,order))
    q=jnp.concatenate(nodes);weight=jnp.concatenate(weights);ids=jnp.concatenate(cell_ids)
    radius=jnp.abs(q);table=geometry['radius_table_cMpc_h']
    transfer=source_mark_transfer(jnp.interp(radius,table,geometry['modulus_table_h']),
        jnp.interp(observed_radius,table,geometry['modulus_table_h']),
        jnp.interp(radius,table,geometry['redshift_table']),
        jnp.interp(observed_radius,table,geometry['redshift_table']),
        mstar=geometry['mstar'],alpha=geometry['alpha'],
        finite_reference_interval=geometry.get('finite_reference_interval'))[population]
    radial=transfer*masses[:,ids]*angular[population//3,ids][None]*weight[None]*radius[None]**2/(box/source_grid)**3
    positions=(observer+q[:,None]*direction)%box
    return positions,velocity[ids],masses[:,ids],angular[:,ids],radial
