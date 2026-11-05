"""Periodic observed-direction source integral prototype, not a production target.

Piecewise-constant source-cell rates/mean/diagonal variance are sampled from
the wrapped field, while the physical observer-centred radius q stays
unwrapped. The ray is oriented and forward-only. This is still a fixed-state
conditional FP mechanics diagnostic, not a calibrated posterior.
"""
import numpy as np
import jax.numpy as jnp
from cf4_r2_shell_cdf_count import _probability_nodes
from cf4_r2_velocity_closure import mixture_components,conditional_los_sigma
from cf4_r2_marked_tracer_jax import source_mark_transfer


def extend_flat_distance_tables(geometry,max_radius_cMpc_h,*,fit_span_cMpc_h=12.):
    """Extend the saved flat-cosmology lookup smoothly beyond its old edge.

    The saved table ends at the box half-size (192 cMpc/h), but a periodic
    observed ray can have physical q beyond that edge. Extrapolate z(q) with a
    local quadratic fit to the final table segment, then use the flat-FLRW
    identity D_L/h=(1+z)q. This is a bounded numerical adapter, not a new
    cosmology; it rejects a large edge discontinuity or nonmonotone extension.
    """
    radius=np.asarray(geometry['radius_table_cMpc_h'],dtype=np.float64)
    modulus=np.asarray(geometry['modulus_table_h'],dtype=np.float64)
    redshift=np.asarray(geometry['redshift_table'],dtype=np.float64)
    maximum=float(max_radius_cMpc_h)
    if (radius.ndim!=1 or modulus.shape!=radius.shape or redshift.shape!=radius.shape
            or not np.isfinite(radius).all() or not np.isfinite(modulus).all()
            or not np.isfinite(redshift).all() or np.any(np.diff(radius)<=0)
            or np.any(np.diff(redshift)<=0) or not radius[-1]>0
            or maximum<=radius[-1] or fit_span_cMpc_h<=0):
        raise ValueError('invalid saved distance table or requested extension')
    edge=float(radius[-1]);select=radius>=edge-fit_span_cMpc_h
    x=radius[select]-edge
    coeff=np.polynomial.polynomial.polyfit(x,redshift[select],2)
    step=min(float(np.min(np.diff(radius))),.01)
    extension=np.r_[np.arange(edge+step,maximum,step),maximum]
    extension=extension[extension>edge]
    zext=np.polynomial.polynomial.polyval(extension-edge,coeff)
    mu_ext=5*np.log10(extension*(1.+zext))+25.
    edge_mu=5*np.log10(edge*(1.+redshift[-1]))+25.
    if (not np.isfinite(zext).all() or not np.isfinite(mu_ext).all()
            or np.any(np.diff(zext)<=0) or zext[0]<=redshift[-1]
            or abs(edge_mu-modulus[-1])>2e-4):
        raise ValueError('distance-table extension is not smooth/monotone')
    extended=dict(geometry)
    extended['radius_table_cMpc_h']=jnp.asarray(np.r_[radius,extension])
    extended['redshift_table']=jnp.asarray(np.r_[redshift,zext])
    extended['modulus_table_h']=jnp.asarray(np.r_[modulus,mu_ext])
    diagnostics=dict(original_max_cMpc_h=edge,extended_max_cMpc_h=float(extension[-1]),
        local_quadratic_fit_span_cMpc_h=float(fit_span_cMpc_h),
        local_fit_max_abs_residual=float(np.max(np.abs(
            np.polynomial.polynomial.polyval(x,coeff)-redshift[select]))),
        luminosity_distance_edge_mismatch_mag=float(edge_mu-modulus[-1]))
    return extended,diagnostics


def source_ray_intervals(direction,observer,n,box):
    """Periodic-cell crossings on one full box-length forward ray q>=0.

    A sky direction is oriented: the antipodal half-line is a different
    observation and must not enter this radial integral. The endpoint is
    ``box/max(abs(direction))``: each Cartesian axis traverses at most one
    box length, so n+2 planes per axis cover all crossings. Callers must
    verify that their finite radial support lies inside this horizon.
    """
    direction=jnp.asarray(direction)
    ray_horizon=box/jnp.max(jnp.abs(direction))
    lo=jnp.asarray(0.,dtype=direction.dtype);hi=ray_horizon;dx=box/n
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
                            population,geometry,closure,*,source_grid,order=4,segments=1):
    """Periodic-ray nodes and UNNORMALIZED conditional optical-FP source masses.

    The fixed angular-selection probability is common to numerator and
    denominator of this per-direction conditional FP factor and cancels;
    therefore the cell-averaged sky map is intentionally not sampled along
    wrapped cells. ``segments`` equally subdivides each ray/cell q interval
    before Gaussian-CDF quadrature; its default of1 preserves the original
    node layout. The five-bin mass retains the q²/source-cell-volume
    Jacobian, LF transfer, Gaussian mixture and source-field values.

    ``source_radius`` returned below is the unwrapped physical q. A caller
    must pass it to the raw-mark normalizer; minimum-image radius is wrong for
    wrapped ray nodes. The active count backend is unchanged.
    """
    if segments<1:raise ValueError('ray quadrature needs at least one segment per cell interval')
    box=geometry['box_size_cMpc_h'];observer=geometry['observer']
    lo,hi,ids=source_ray_intervals(direction,observer,source_grid,box)
    interval_width=hi-lo
    segment_fraction=jnp.arange(segments,dtype=lo.dtype)/segments
    starts=lo[:,None]+segment_fraction[None,:]*interval_width[:,None]
    ends=starts+interval_width[:,None]/segments
    lo=starts.reshape(-1);hi=ends.reshape(-1);ids=jnp.repeat(ids,segments)
    vel=velocity[ids];var=variance[ids]
    conversion=geometry['little_h']/geometry['hubble_km_s_Mpc']
    centre=observed_radius-conversion*jnp.sum(vel*direction[None],axis=1)
    nodes=[];weights=[];cell_ids=[]
    for mixture_fraction,core,scale in mixture_components(closure):
        sigma=conversion*conditional_los_sigma(jnp.broadcast_to(direction,var.shape),core,var,scale)
        centre_cell=centre
        lower=jnp.maximum(lo,centre_cell-8*sigma);upper=jnp.minimum(hi,centre_cell+8*sigma)
        active=upper>lower
        q,w=_probability_nodes(centre_cell,sigma,lower[None],upper[None],active[None],order)
        q=q[0].reshape(-1);w=w[0].reshape(-1)*mixture_fraction
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
    # A direction-only completeness scalar cancels exactly from this
    # conditional likelihood. Do not substitute the wrapped cell's sky mask.
    radial=transfer*masses[:,ids]*weight[None]*radius[None]**2/(box/source_grid)**3
    positions=(observer+q[:,None]*direction)%box
    return positions,velocity[ids],masses[:,ids],jnp.ones_like(angular[:,ids]),q,radial
