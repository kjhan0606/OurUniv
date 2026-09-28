"""Streaming differentiable raw marks on a supplied live source-cell field.

This is an observation adapter, not a calibrated posterior or PM evolution.
Caller owns refreshed source support, matched count-volume integration and
the single-mark association assumption. No old eta likelihood or free zero.
"""
import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_marked_tracer_jax import predict_source_marked_radial_key_density,TRUE_EDGES,OBS_EDGES
from cf4_r2_raw_selected_fp_jax import row_logpdf

# Proper weak modelling priors, NOT empirical calibration measurements.
POPULATION_ORIGIN=np.r_[ [.3,2.2,2.7],np.zeros(6),
                         np.log(.3),0.,np.log(.3),0.,0.,np.log(.3)]
POPULATION_SCALE=np.r_[ [1.,.5,1.],np.full(6,.5),1.,.3,1.,.3,.3,1.]


def logadd_nonempty(a,b):
    both_empty=jnp.isneginf(a)&jnp.isneginf(b)
    value=jnp.logaddexp(jnp.where(both_empty,0.,a),jnp.where(both_empty,0.,b))
    return jnp.where(both_empty,-jnp.inf,value)


def chunk_log_terms(parameters,positions,velocities,intrinsic,angular,observation,
                    *,population,geometry,magnitude_order=24,cut_order=64,
                    component_bin=None,component_row=None,
                    cut_integration_axis=0,cut_marginal_tolerance=0.):
    """UNNORMALIZED raw numerator/selection denominator on one source chunk."""
    o=observation
    if component_row is None:
        mass=predict_source_marked_radial_key_density(positions,velocities,intrinsic,angular,
            population,o['voxel'],o['radius'],**geometry)
        geometric=o
    else:
        if component_bin is None:raise ValueError('multirow stream requires packed bin IDs')
        geometric={k:o[k][component_row] for k in ('voxel','radius','dz','ksmag')}
        def one(pos,vel,mass,sky,voxel,radius):
            return predict_source_marked_radial_key_density(pos[None],vel[None],mass[:,None],sky[:,None],
                population,voxel,radius,**geometry)[:,0]
        mass=jax.vmap(one,in_axes=(0,0,1,1,0,0),out_axes=1)(positions,velocities,intrinsic,angular,
            geometric['voxel'],geometric['radius'])
    relative=(positions-geometry['observer']+geometry['box_size_cMpc_h']/2)%geometry['box_size_cMpc_h']-geometry['box_size_cMpc_h']/2
    rt=jnp.linalg.norm(relative,axis=1)
    table=geometry['radius_table_cMpc_h']
    mt=jnp.interp(rt,table,geometry['modulus_table_h'])
    mo=jnp.interp(geometric['radius'],table,geometry['modulus_table_h'])
    zt=jnp.interp(rt,table,geometry['redshift_table'])
    zo=jnp.interp(geometric['radius'],table,geometry['redshift_table'])
    correction=1.16*2.9*(zo-zt)-1.6*jnp.log10((1+zo)/(1+zt))
    shift=mo-mt-correction
    lo=jnp.maximum(jnp.maximum(jnp.asarray(TRUE_EDGES[:-1])[:,None],
        (-jnp.inf if population//3==0 else 11.5)-mt-correction),OBS_EDGES[population%3]+shift)
    hi=jnp.minimum(jnp.minimum(jnp.asarray(TRUE_EDGES[1:])[:,None],
        (11.5 if population//3==0 else 12.5)-mt-correction),OBS_EDGES[population%3+1]+shift)
    if component_bin is not None:
        # Sparse entries are individual (bin,subnode) components. A source
        # repeated for distinct bins is intentional; each bin appears once.
        index=jnp.arange(positions.shape[0])
        mass=mass[component_bin,index][None]
        lo=lo[component_bin,index][None];hi=hi[component_bin,index][None]
    valid=(hi>lo)&(mass>0)
    # Inactive geometry has zero mass and finite dummy LF integrals so its
    # derivative is zero, not0*NaN. No floor is applied to any active density.
    lo=jnp.where(valid,lo,-24.).reshape(-1)
    hi=jnp.where(valid,hi,-23.).reshape(-1)
    eta=jnp.broadcast_to(jnp.log10(geometric['dz']/rt)[None],mass.shape).reshape(-1)
    observed_m=jnp.broadcast_to((geometric['ksmag']-mt-correction)[None],mass.shape).reshape(-1)
    safe_mass=jnp.where(valid,mass,1.).reshape(-1)
    logw=jnp.where(valid.reshape(-1),jnp.log(safe_mass),-jnp.inf)
    t,w=np.polynomial.legendre.leggauss(magnitude_order)
    magnitude=lo[:,None]+(hi-lo)[:,None]*jnp.asarray((t+1)/2)
    def loglf(m):
        power=.4*jnp.log(10.)*(geometry['mstar']-m)
        return (geometry['alpha']+1)*power-jnp.exp(power)
    logq=loglf(magnitude)+jnp.log(jnp.asarray(w/2))
    norm=jax.scipy.special.logsumexp(logq,axis=1)
    logmd=jnp.where((observed_m>=lo)&(observed_m<=hi),
        loglf(observed_m)-jnp.log(hi-lo)-norm,-jnp.inf)
    data=dict(x=o['x'][None],row=jnp.zeros(eta.shape,dtype=jnp.int32),eta=eta,
        error_covariance=o['error_covariance'][None],richness=o['richness'][None],
        observed_M=observed_m,magnitude=magnitude,logq=logq-norm[:,None],
        log_weight=logw,log_M_density=logmd,cut_lower=o['cut_lower'][None],
        cut_upper=o['cut_upper'][None])
    if component_row is not None:
        data.update(row=component_row,**{k:o[k] for k in
            ('x','error_covariance','richness','cut_lower','cut_upper')})
    t,w=np.polynomial.legendre.leggauss(cut_order)
    a,b=row_logpdf(parameters,data,jnp.asarray((t+1)/2),jnp.asarray(w/2),return_log_terms=True,
        cut_integration_axis=cut_integration_axis,cut_marginal_tolerance=cut_marginal_tolerance)
    return (a[0],b[0]) if component_row is None else (a,b)


def streaming_raw_mark(parameters,positions,velocities,intrinsic,angular,observation,
                       *,population,geometry,component_bins=None,component_rows=None,
                       return_log_terms=False,cut_order=64,
                       cut_integration_axis=0,cut_marginal_tolerance=0.):
    """Chunk-major geometry; intrinsic(chunk,5,source), angular(chunk,2,source)."""
    @jax.checkpoint
    def step(acc,parts):
        a,b=chunk_log_terms(parameters,*parts[:4],observation,population=population,geometry=geometry,
            component_bin=None if component_bins is None else parts[4],
            component_row=None if component_rows is None else parts[5],cut_order=cut_order,
            cut_integration_axis=cut_integration_axis,cut_marginal_tolerance=cut_marginal_tolerance)
        return (logadd_nonempty(acc[0],a),logadd_nonempty(acc[1],b)),None
    parts=(positions,velocities,intrinsic,angular)
    if component_bins is not None:parts+= (component_bins,)
    if component_rows is not None:
        if component_bins is None:raise ValueError('multirow stream requires packed bins')
        parts+=(component_rows,)
    initial=-jnp.inf if component_rows is None else jnp.full(observation['x'].shape[0],-jnp.inf)
    (numerator,denominator),_=jax.lax.scan(step,(initial,initial),
        parts)
    if return_log_terms:return numerator,denominator
    return numerator-denominator
