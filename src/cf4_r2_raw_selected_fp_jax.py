"""Differentiable counterpart of the CPU raw selected-mark reference.

Fixed saved source weights only. No field evolution, prior, incidence fit or
physical-calibration claim. Fifteen shared population parameters, not row fits.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtri


def log_interval(lo, hi):
    positive = lo > 0
    a, b = jnp.where(positive, -hi, lo), jnp.where(positive, -lo, hi)
    la, lb = log_ndtr(a), log_ndtr(b)
    return lb+jnp.log(-jnp.expm1(la-lb))


def inverse_log_cdf(logp):
    """Avoid exp(logp) underflow without clipping Gaussian tail support."""
    tail = logp < -30.
    target = jnp.where(tail, logp, -30.)
    initial = -jnp.sqrt(-2*target)
    def step(_, x):
        logcdf = log_ndtr(x)
        derivative = jnp.exp(-.5*x*x-.5*jnp.log(2*jnp.pi)-logcdf)
        return x-(logcdf-target)/derivative
    solved = jax.lax.fori_loop(0, 5, step, initial)
    central = ndtri(jnp.exp(jnp.where(tail, -30., logp)))
    return jnp.where(tail, solved, central)


def unpack(parameters):
    mean = parameters[:9].reshape(3,3)
    p = parameters[9:]
    lower = jnp.array([[jnp.exp(p[0]),0.,0.],
                       [p[1],jnp.exp(p[2]),0.],
                       [p[3],p[4],jnp.exp(p[5])]])
    return mean, lower@lower.T


def row_logpdf(parameters, data, cut_t, cut_w, *, return_log_terms=False):
    """Exact saved-candidate mixture; LF quadrature supplied as normalized q."""
    b, intrinsic = unpack(parameters)
    x, row = data['x'], data['row']
    n = x.shape[0]
    covariance = intrinsic[None]+data['error_covariance']
    inverse = jnp.linalg.inv(covariance)
    logdet = jnp.linalg.slogdet(covariance)[1]
    base = b[0]+data['richness'][:,None]*b[2]
    eta_vector = jnp.stack((data['eta'],jnp.zeros_like(data['eta']),
                           jnp.zeros_like(data['eta'])),axis=-1)
    mean = base[row]+b[1]*(data['observed_M'][:,None]+23.)+eta_vector
    residual = x[row]-mean
    log_g = -.5*(jnp.einsum('ni,nij,nj->n',residual,inverse[row],residual)
                  +logdet[row]+3*jnp.log(2*jnp.pi))
    matrix = jnp.array([[2.,0.,1.],[.04,1.,0.]])
    cut_cov = jnp.einsum('ai,nij,bj->nab',matrix,covariance,matrix)
    sd = jnp.sqrt(jnp.diagonal(cut_cov,axis1=-2,axis2=-1))[row]
    rho = (cut_cov[:,0,1]/jnp.sqrt(cut_cov[:,0,0]*cut_cov[:,1,1]))[row]
    cut_base = (base[row]+eta_vector)@matrix.T
    cut_slope = b[1]@matrix.T
    def selected_at_magnitude(acc, inputs):
        magnitude, logq = inputs
        cm = cut_base+(magnitude[:,None]+23.)*cut_slope
        lo, hi = (data['cut_lower'][row]-cm)/sd, (data['cut_upper'][row]-cm)/sd
        flip = lo[:,0]>0
        a, bb = jnp.where(flip,-hi[:,0],lo[:,0]), jnp.where(flip,-lo[:,0],hi[:,0])
        corr = jnp.where(flip,-rho,rho)
        mass = log_interval(a,bb)
        logu = jnp.logaddexp(log_ndtr(a)[:,None],mass[:,None]+jnp.log(cut_t))
        z = inverse_log_cdf(logu)
        csd = jnp.sqrt(1-corr*corr)[:,None]
        shift = corr[:,None]*z
        logconditional = log_interval((lo[:,1,None]-shift)/csd,(hi[:,1,None]-shift)/csd)
        logz = mass+jax.scipy.special.logsumexp(jnp.log(cut_w)+logconditional,axis=-1)
        return jnp.logaddexp(acc,logq+logz), None
    logz,_ = jax.lax.scan(jax.checkpoint(selected_at_magnitude),
        jnp.full_like(data['eta'],-jnp.inf),(data['magnitude'].T,data['logq'].T))
    def segment_lse(values):
        maximum = jax.lax.stop_gradient(jax.ops.segment_max(values,row,num_segments=n))
        safe_maximum=jnp.where(jnp.isfinite(maximum),maximum,0.)
        total = jax.ops.segment_sum(jnp.exp(values-safe_maximum[row]),row,num_segments=n)
        return jnp.where(total>0,safe_maximum+jnp.log(jnp.where(total>0,total,1.)),-jnp.inf)
    lognumerator = segment_lse(data['log_weight']+data['log_M_density']+log_g)
    logdenominator = segment_lse(data['log_weight']+logz)
    if return_log_terms:
        return lognumerator,logdenominator
    return lognumerator-logdenominator
