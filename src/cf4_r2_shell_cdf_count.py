"""Boundary-fitted count quadrature; development, not calibrated inference.

Integrate the Gaussian LOS law over periodic observer-shell intersections.
No GH atom is switched at a survey cut. The explicit eight-sigma tail
truncation is a numerical approximation (omitted Gaussian mass <1.3e-15),
not a nuisance-domain prior. The caller must enforce 8*sigma_radius < box/2;
under that existing live-support bound, 27 observer images suffice.
"""
from itertools import product
import numbers

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
import numpy as np

from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax, tsc_deposit_jax
from cf4_r2_marked_tracer_jax import source_mark_transfer


def shell_cdf_nodes(mu, direction, sigma, image_center, *, radial_min=5.,
                    radial_max=180., order=16, tail_sigma=8.):
    """Signed ray coordinates/weights: (two shell segments, nodes, sources).

    ``mu`` and ``direction`` describe coherent positions relative to the
    observer in its minimum-image cell. ``image_center`` is another observer
    image relative to that observer. Weights are *unconditional* Gaussian
    probabilities, never normalized to the selected region.
    """
    if order<1 or not 0<radial_min<radial_max or tail_sigma<=0:
        raise ValueError('invalid shell quadrature')
    if isinstance(sigma,numbers.Real) and sigma<=0:
        raise ValueError('positive LOS dispersion required')
    mu,direction=jnp.asarray(mu),jnp.asarray(direction)
    center=jnp.asarray(image_center)
    middle=jnp.sum(direction*center,axis=-1)
    transverse=jnp.maximum(jnp.sum(center*center)-middle*middle,0.)
    outer_ok=transverse<radial_max**2
    inner_ok=transverse<radial_min**2
    # Mask before sqrt, not after: inactive tangencies must have finite VJPs.
    outer=jnp.sqrt(jnp.where(outer_ok,radial_max**2-transverse,1.))
    inner=jnp.where(inner_ok,jnp.sqrt(jnp.where(
        inner_ok,radial_min**2-transverse,1.)),0.)
    lower=jnp.stack((middle-outer,middle+inner))
    upper=jnp.stack((middle-inner,middle+outer))
    lower=jnp.maximum(lower,mu-tail_sigma*sigma)
    upper=jnp.minimum(upper,mu+tail_sigma*sigma)
    active=outer_ok[None]&(upper>lower)&(jnp.sum(direction**2,axis=1)>0)[None]
    za=jnp.where(active,(lower-mu)/sigma,0.)
    zb=jnp.where(active,(upper-mu)/sigma,0.)
    # Positive-tail intervals use survival probabilities to retain precision.
    positive=za>0
    p0=jnp.where(positive,ndtr(-zb),ndtr(za))
    p1=jnp.where(positive,ndtr(-za),ndtr(zb))
    probability=jnp.where(active,p1-p0,0.)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    u=p0[:,None,:]+jnp.asarray((nodes+1)/2)[None,:,None]*probability[:,None,:]
    u=jnp.where(active[:,None,:],u,.5)
    q=mu[None,None,:]+sigma*jnp.where(positive[:,None,:],-ndtri(u),ndtri(u))
    weight=probability[:,None,:]*jnp.asarray(weights/2)[None,:,None]
    return q,weight


def predict_shell_cdf_intensity(source_positions,source_velocities_km_s,
    intrinsic_bin_masses,angular_completeness,*,observer,box_size_cMpc_h,
    hubble_km_s_Mpc,little_h,radius_table_cMpc_h,modulus_table_h,redshift_table,
    grid_size,sigma_los_km_s,radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,
    mstar=-23.28,alpha=-.94,order=16):
    """Same source-selected K/TSC count integrand, boundary-fitted LOS law.

    Caller owns current-width support checks and calibration. This does not
    replace the production target automatically, nor the continuous FP mark.
    """
    pos=jnp.asarray(source_positions)
    intrinsic=jnp.asarray(intrinsic_bin_masses)
    angular=jnp.asarray(angular_completeness)
    if (pos.ndim!=2 or pos.shape[1]!=3 or intrinsic.shape!=(5,pos.shape[0])
            or angular.shape!=(2,pos.shape[0]) or radial_max_cMpc_h>=box_size_cMpc_h/2):
        raise ValueError('invalid source geometry or overlapping observer shells')
    sigma=little_h*sigma_los_km_s/hubble_km_s_Mpc
    if isinstance(sigma,numbers.Real) and not 0<8*sigma<box_size_cMpc_h/2:
        raise ValueError('27-image quadrature requires 0<8*sigma_radius<box/2')
    shifted,_,direction=observer_centred_spherical_rsd_jax(pos,source_velocities_km_s,
        observer,box_size_cMpc_h,hubble_km_s_Mpc,little_h=little_h,scale_factor=1.)
    relative=lambda x:(x-observer+box_size_cMpc_h/2)%box_size_cMpc_h-box_size_cMpc_h/2
    def radius_of(x):
        r2=jnp.sum(relative(x)**2,axis=1)
        return jnp.where(r2>0,jnp.sqrt(jnp.where(r2>0,r2,1.)),0.)
    mu=radius_of(shifted)
    true_radius=radius_of(pos)
    true_modulus=jnp.interp(true_radius,radius_table_cMpc_h,modulus_table_h)
    true_redshift=jnp.interp(true_radius,radius_table_cMpc_h,redshift_table)
    images=jnp.asarray(list(product((-1,0,1),repeat=3)))*box_size_cMpc_h

    @jax.checkpoint
    def add_node(total,qw):
        q,weight=qw
        observed=(observer+q[:,None]*direction)%box_size_cMpc_h
        radius=radius_of(observed)
        transfer=source_mark_transfer(true_modulus,
            jnp.interp(radius,radius_table_cMpc_h,modulus_table_h),true_redshift,
            jnp.interp(radius,radius_table_cMpc_h,redshift_table),mstar=mstar,alpha=alpha)
        contribution=jnp.stack([tsc_deposit_jax(observed,
            weight*angular[p//3]*jnp.sum(transfer[p]*intrinsic,axis=0),
            grid_size,box_size_cMpc_h) for p in range(6)])
        return total+contribution,None

    @jax.checkpoint
    def add_image(total,center):
        q,w=shell_cdf_nodes(mu,direction,sigma,center,radial_min=radial_min_cMpc_h,
                            radial_max=radial_max_cMpc_h,order=order)
        q,w=q.reshape(-1,pos.shape[0]),w.reshape(-1,pos.shape[0])
        total=jax.lax.cond(jnp.any(w>0),
            lambda t:jax.lax.scan(add_node,t,(q,w))[0],lambda t:t,total)
        return total,None
    dtype=jnp.result_type(pos,source_velocities_km_s,intrinsic,angular,observer,sigma)
    return jax.lax.scan(add_image,jnp.zeros((6,grid_size,grid_size,grid_size),
                                          dtype=dtype),images)[0]
