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
                    *,population,geometry,magnitude_order=24,cut_order=64):
    """UNNORMALIZED raw numerator/selection denominator on one source chunk."""
    o=observation
    mass=predict_source_marked_radial_key_density(positions,velocities,intrinsic,angular,
        population,o['voxel'],o['radius'],**geometry)
    relative=(positions-geometry['observer']+geometry['box_size_cMpc_h']/2)%geometry['box_size_cMpc_h']-geometry['box_size_cMpc_h']/2
    rt=jnp.linalg.norm(relative,axis=1)
    table=geometry['radius_table_cMpc_h']
    mt=jnp.interp(rt,table,geometry['modulus_table_h'])
    mo=jnp.interp(o['radius'],table,geometry['modulus_table_h'])
    zt=jnp.interp(rt,table,geometry['redshift_table'])
    zo=jnp.interp(o['radius'],table,geometry['redshift_table'])
    correction=1.16*2.9*(zo-zt)-1.6*jnp.log10((1+zo)/(1+zt))
    shift=mo-mt-correction
    lo=jnp.maximum(jnp.maximum(jnp.asarray(TRUE_EDGES[:-1])[:,None],
        (-jnp.inf if population//3==0 else 11.5)-mt-correction),OBS_EDGES[population%3]+shift)
    hi=jnp.minimum(jnp.minimum(jnp.asarray(TRUE_EDGES[1:])[:,None],
        (11.5 if population//3==0 else 12.5)-mt-correction),OBS_EDGES[population%3+1]+shift)
    valid=(hi>lo)&(mass>0)
    # Inactive geometry has zero mass and finite dummy LF integrals so its
    # derivative is zero, not0*NaN. No floor is applied to any active density.
    lo=jnp.where(valid,lo,-24.).reshape(-1)
    hi=jnp.where(valid,hi,-23.).reshape(-1)
    eta=jnp.broadcast_to(jnp.log10(o['dz']/rt)[None],mass.shape).reshape(-1)
    observed_m=jnp.broadcast_to((o['ksmag']-mt-correction)[None],mass.shape).reshape(-1)
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
    t,w=np.polynomial.legendre.leggauss(cut_order)
    a,b=row_logpdf(parameters,data,jnp.asarray((t+1)/2),jnp.asarray(w/2),return_log_terms=True)
    return a[0],b[0]


def streaming_raw_mark(parameters,positions,velocities,intrinsic,angular,observation,
                       *,population,geometry):
    """Chunk-major geometry; intrinsic(chunk,5,source), angular(chunk,2,source)."""
    @jax.checkpoint
    def step(acc,parts):
        a,b=chunk_log_terms(parameters,*parts,observation,population=population,geometry=geometry)
        return (logadd_nonempty(acc[0],a),logadd_nonempty(acc[1],b)),None
    (numerator,denominator),_=jax.lax.scan(step,(-jnp.inf,-jnp.inf),
        (positions,velocities,intrinsic,angular))
    return numerator-denominator
