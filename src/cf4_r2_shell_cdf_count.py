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
from cf4_r2_marked_tracer_jax import source_mark_transfer, tsc_weight_at_voxel


def cell_averaged_tsc_weight(positions,voxel,grid_size,box):
    """TSC convolved with one cell top-hat = separable cubic B-spline.

    Exact for translation of the deposit kernel ONLY. Holding source LF,
    angular completeness and coherent radial mapping at the cell centre is
    an additional approximation; this is NOT the full source-volume integral.
    """
    if grid_size<4:raise ValueError('cell-averaged kernel requires grid>=4')
    cell=(positions%box)/(box/grid_size)-.5
    d=jnp.abs((jnp.asarray(voxel)[None]-cell+grid_size/2)%grid_size-grid_size/2)
    w=jnp.where(d<1.,(4.-6*d*d+3*d**3)/6.,jnp.where(d<2.,(2.-d)**3/6.,0.))
    return jnp.prod(w,axis=1)


def _shell_intervals(mu,direction,sigma,image_center,radial_min,radial_max,tail_sigma):
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
    return lower,upper,active


def _probability_nodes(mu,sigma,lower,upper,active,order):
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


def shell_cdf_nodes(mu, direction, sigma, image_center, *, radial_min=5.,
                    radial_max=180., order=16, tail_sigma=8., segments=1):
    """Reference node materialization (2*segments, nodes, sources).

    Production-size contractions below stream segments instead. Weights are
    unconditional Gaussian probabilities, never selected-region normalized.
    """
    if order<1 or segments<1 or not 0<radial_min<radial_max or tail_sigma<=0:
        raise ValueError('invalid shell quadrature')
    if isinstance(sigma,numbers.Real) and sigma<=0:
        raise ValueError('positive LOS dispersion required')
    mu,direction=jnp.asarray(mu),jnp.asarray(direction)
    lower,upper,active=_shell_intervals(mu,direction,sigma,image_center,
                                      radial_min,radial_max,tail_sigma)
    width=jnp.where(active,upper-lower,0.)
    fraction=jnp.arange(segments,dtype=mu.dtype)/segments
    starts=lower[:,None,:]+fraction[None,:,None]*width[:,None,:]
    ends=starts+width[:,None,:]/segments
    lower,upper=starts.reshape(-1,mu.size),ends.reshape(-1,mu.size)
    active=jnp.broadcast_to(active[:,None,:],starts.shape).reshape(-1,mu.size)
    return _probability_nodes(mu,sigma,lower,upper,active,order)


def predict_shell_cdf_intensity(source_positions,source_velocities_km_s,
    intrinsic_bin_masses,angular_completeness,*,observer,box_size_cMpc_h,
    hubble_km_s_Mpc,little_h,radius_table_cMpc_h,modulus_table_h,redshift_table,
    grid_size,sigma_los_km_s,radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,
    mstar=-23.28,alpha=-.94,order=16,segments=1,
    target_population=None,target_voxel=None,source_cell_average=False):
    """Same source-selected K/TSC count integrand, boundary-fitted LOS law.

    Caller owns current-width support checks and calibration. This does not
    replace the production target automatically, nor the continuous FP mark.
    An optional population/voxel pair reads one scalar with the transpose of
    the identical deposit kernel; it does not change the integration rule.
    """
    pos=jnp.asarray(source_positions)
    intrinsic=jnp.asarray(intrinsic_bin_masses)
    angular=jnp.asarray(angular_completeness)
    scalar = target_population is not None
    if scalar != (target_voxel is not None):
        raise ValueError('population and voxel must be supplied together')
    if source_cell_average and not scalar:
        raise ValueError('approximate cell-averaged kernel is a scalar diagnostic only')
    if scalar and (not isinstance(target_population,int) or not 0<=target_population<6
                   or jnp.asarray(target_voxel).shape!=(3,) or grid_size<3):
        raise ValueError('invalid scalar count key')
    if (pos.ndim!=2 or pos.shape[1]!=3 or intrinsic.shape!=(5,pos.shape[0])
            or angular.shape!=(2,pos.shape[0]) or radial_max_cMpc_h>=box_size_cMpc_h/2
            or not 0<radial_min_cMpc_h<radial_max_cMpc_h or min(order,segments)<1):
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
        if scalar:
            p=target_population
            kernel=cell_averaged_tsc_weight if source_cell_average else tsc_weight_at_voxel
            contribution=jnp.sum(weight*angular[p//3]*jnp.sum(transfer[p]*intrinsic,axis=0)
                *kernel(observed,target_voxel,grid_size,box_size_cMpc_h))
        else:
            contribution=jnp.stack([tsc_deposit_jax(observed,
                weight*angular[p//3]*jnp.sum(transfer[p]*intrinsic,axis=0),
                grid_size,box_size_cMpc_h) for p in range(6)])
        return total+contribution,None

    @jax.checkpoint
    def add_image(total,center):
        lower,upper,active=_shell_intervals(mu,direction,sigma,center,
            radial_min_cMpc_h,radial_max_cMpc_h,8.)
        width=jnp.where(active,upper-lower,0.)
        # Do not materialize (images,segments,nodes,sources) on the reverse
        # tape. Each physical segment regenerates its own few quadrature nodes.
        @jax.checkpoint
        def add_segment(current,index):
            branch=index//segments
            fraction=(index%segments).astype(mu.dtype)/segments
            start=lower[branch]+fraction*width[branch]
            end=start+width[branch]/segments
            def integrate(value):
                q,w=_probability_nodes(mu,sigma,start[None],end[None],active[branch][None],order)
                return jax.lax.scan(add_node,value,(q[0],w[0]))[0]
            return jax.lax.cond(jnp.any(active[branch]),integrate,lambda v:v,current),None
        total=jax.lax.cond(jnp.any(active),
            lambda t:jax.lax.scan(add_segment,t,jnp.arange(2*segments))[0],lambda t:t,total)
        return total,None
    dtype=jnp.result_type(pos,source_velocities_km_s,intrinsic,angular,observer,sigma)
    shape=() if scalar else (6,grid_size,grid_size,grid_size)
    return jax.lax.scan(add_image,jnp.zeros(shape,dtype=dtype),images)[0]


def predict_source_volume_intensity(positions,velocities,intrinsic,angular,*,
                                    source_spacing,volume_order=4,**geometry):
    """Stream cell-volume GL nodes through the UNCHANGED source count law.

    Mass/velocity/angular completeness are cell-constant. No approximate
    cell-averaged TSC kernel; LF, radial mapping and cuts are re-evaluated at
    every subnode. Source spacing need not equal the observed count grid.
    """
    if volume_order<1 or source_spacing<=0 or geometry.get('source_cell_average',False):
        raise ValueError('positive source cell/rule; no second smoothing kernel')
    nodes,weights=np.polynomial.legendre.leggauss(volume_order)
    indices=np.array(list(product(range(volume_order),repeat=3)))
    offsets=jnp.asarray(nodes[indices]*source_spacing/2)
    volume=jnp.asarray(np.prod(weights[indices]/2,axis=1))
    @jax.checkpoint
    def add(total,item):
        offset,weight=item
        field=predict_shell_cdf_intensity((positions+offset)%geometry['box_size_cMpc_h'],
            velocities,intrinsic*weight,angular,**geometry)
        return total+field,None
    n=geometry['grid_size']
    shape=() if geometry.get('target_population') is not None else (6,n,n,n)
    dtype=jnp.result_type(positions,velocities,intrinsic,angular,offsets)
    return jax.lax.scan(add,jnp.zeros(shape,dtype=dtype),(offsets,volume))[0]
