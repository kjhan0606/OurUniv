"""Selected-population response and excess group-error model on a live field.

This is a declared conditional development model, not calibrated inclusion or
the missing joint count/point likelihood. Excess FP offsets are group-shared,
not independent per member, not measured physical depth, and not an estimate
of the original FP fit covariance. Source PDFs and raw non-FP marks enter once.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from cf4_r2_fp_distance import fp_log_likelihood_ratio
from cf4_r2_fp_group_marginal import nonfp_modulus_logmarks
from cf4_r2_live_field_marks import bind_state_geometry


def decode_hyper(white, calibration_sd, *, bias_log_sd=.5,
                 tau_median=.02, tau_log_sd=.7):
    """Proper DEVELOPMENT priors; none comes from a fit of the same data.

    Six previous calibration/radial parameters + log selected-density slope
    + log excess FP shared SD. The density slope cannot separate physical
    tracer bias from environment-dependent inclusion. The lognormal tau is
    explicit extra discrepancy regularization, NOT source-error calibration.
    """
    count = calibration_sd.shape[0]
    return (white[:count]*calibration_sd,
            jnp.exp(bias_log_sd*white[count]),
            tau_median*jnp.exp(tau_log_sd*white[count+1]))


def shifted_group_scores(parameters, group_offset, bound):
    """One FP offset per group INSIDE the common distance integral.

    Non-FP distance indicators share that distance, but NOT the FP-specific
    excess offset. Reference eta=0 is fixed independently of all offsets.
    """
    d, gid = bound['distance'], bound['row_group']
    base = (bound['log_distance_weight'] + bound['redshift_logkernel']
            + parameters[-1]*jnp.log(d/100.))
    eta = (jnp.log10(bound['dz_row'][:,None]/d[gid]) + parameters[0]
           + group_offset[gid,None])
    member = fp_log_likelihood_ratio(eta,0.,bound['eta_mean'][:,None],
                                    bound['eta_std'][:,None],bound['eta_alpha'][:,None])
    marks = jax.ops.segment_sum(member,gid,num_segments=d.shape[0])
    marks += nonfp_modulus_logmarks(bound['predicted_modulus'],bound['anchor_group'],
        bound['anchor_modulus'],bound['anchor_error'],bound['anchor_method'],parameters[1:-1])
    return logsumexp(base+marks,axis=-1)-logsumexp(base,axis=-1)


def hierarchical_group_scores(rho, velocity, white_hyper, white_group,
                              geometry, calibration_sd, *, box=384.):
    """Noncentred offsets; heldout offsets are not learned from heldout marks.

    train_group_index contains only whole, closed training groups. A caller
    computing heldout predictions must integrate NEW offsets with
    marginal_group_scores, not score the zeros populated here for heldout.
    """
    parameters,bias,tau = decode_hyper(white_hyper,calibration_sd)
    offset = jnp.zeros(geometry['distance'].shape[0]).at[
        geometry['train_group_index']].set(tau*white_group)
    bound = bind_state_geometry(rho,velocity,geometry,box=box,selected_bias=bias)
    return shifted_group_scores(parameters,offset,bound)


def joint_logdensity(white_ic, white_hyper, white_group, rho, velocity,
                     geometry, calibration_sd, *, box=384.):
    scores = hierarchical_group_scores(rho,velocity,white_hyper,white_group,
                                       geometry,calibration_sd,box=box)
    return (jnp.where(geometry['group_holdout'],0.,scores).sum()
            - .5*(jnp.vdot(white_ic,white_ic)+jnp.vdot(white_hyper,white_hyper)
                  +jnp.vdot(white_group,white_group)))


def marginal_group_scores(rho, velocity, white_hyper, geometry, calibration_sd,
                          normal_nodes, log_normal_weights, *, box=384.):
    """Integrate independent new group offsets, each shared by its members.

    This is a per-state conditional factor, NOT posterior predictive averaging
    over field/hyperparameter uncertainty. Normal nodes/weights are supplied
    Gaussian quadrature; compare orders before interpreting the diagnostic.
    """
    parameters,bias,tau = decode_hyper(white_hyper,calibration_sd)
    bound = bind_state_geometry(rho,velocity,geometry,box=box,selected_bias=bias)
    at_node = lambda node: shifted_group_scores(parameters,
        jnp.full(geometry['distance'].shape[0],tau*node),bound)
    scores = jax.lax.map(at_node,normal_nodes)
    weights = log_normal_weights-logsumexp(log_normal_weights)
    return logsumexp(scores+weights[:,None],axis=0)
