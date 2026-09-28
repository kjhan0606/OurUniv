"""Export sparse SAME-state source-bin weights, not new observations/priors."""
import numpy as np
from scipy.special import logsumexp

from cf4_r2_fp_distance import fp_log_likelihood_ratio
from cf4_r2_marked_tracer_jax import TRUE_EDGES, OBS_EDGES


def export_raw_source_mixture(out, evaluated, selected, metadata, source,
                              source_pgc, point, point_ksmag, state, rows):
    eta = np.asarray(evaluated[3])
    bin_logw = np.asarray(evaluated[5])
    if bin_logw.shape != (len(rows), 5, eta.shape[1]):
        raise ValueError('raw source-bin geometry changed')
    np.testing.assert_allclose(logsumexp(bin_logw,axis=(1,2)),0.,atol=1e-12)
    chosen = [o for p in range(6) for o in selected[p]]
    mask = np.isfinite(bin_logw)
    row_id, bin_id, candidate_id = np.nonzero(mask)
    logw = bin_logw[mask]
    eta_flat = eta[row_id,candidate_id]
    ptr = np.r_[0,np.cumsum(mask.sum(axis=(1,2)))]
    if np.any(np.diff(ptr)==0):
        raise ValueError('empty source row')
    dz = np.concatenate([np.asarray(metadata[p]['dz_row']) for p in range(6)])
    source_radius = dz[row_id]/10**eta_flat
    observed_radius = np.array([point['radius_cMpc_h'][o[2]] for o in chosen])
    ksmag = np.array([point_ksmag[o[2]] for o in chosen])
    population = np.array([point['population'][o[2]] for o in chosen])
    table, modulus, redshift = map(np.asarray,(source['radial_table'],
        source['modulus_table'],source['redshift_table']))
    mt = np.interp(source_radius,table,modulus)
    mo = np.interp(observed_radius,table,modulus)[row_id]
    zt = np.interp(source_radius,table,redshift)
    zo = np.interp(observed_radius,table,redshift)[row_id]
    correction = 1.16*2.9*(zo-zt)-1.6*np.log10((1+zo)/(1+zt))
    app = population[row_id]//3
    obin = population[row_id]%3
    lower = np.maximum.reduce([np.array(TRUE_EDGES)[bin_id],
        np.where(app==0,-np.inf,11.5)-mt-correction,
        np.array(OBS_EDGES)[obin]+mo-mt-correction])
    upper = np.minimum.reduce([np.array(TRUE_EDGES)[bin_id+1],
        np.where(app==0,11.5,12.5)-mt-correction,
        np.array(OBS_EDGES)[obin+1]+mo-mt-correction])
    observed_M = ksmag[row_id]-mt-correction
    if not np.isfinite(np.r_[lower,upper,eta_flat]).all() or np.any(upper<=lower):
        raise ValueError('positive count weight has invalid LF interval')
    inside = (observed_M>=lower)&(observed_M<=upper)
    if np.any(np.add.reduceat(inside.astype(int),ptr[:-1])==0):
        raise ValueError('observed K has no count-source support')
    means,std,alpha = [np.concatenate([np.asarray(metadata[p][key]) for p in range(6)])
        for key in ('eta_mean','eta_std','eta_alpha')]
    marks = np.asarray(fp_log_likelihood_ratio(eta_flat+.004*float(state['white_fp_zero']),
        0.,means[row_id],std[row_id],alpha[row_id]))
    reproduced = np.array([logsumexp((logw+marks)[ptr[i]:ptr[i+1]]) for i in range(len(rows))])
    np.testing.assert_allclose(reproduced,rows,rtol=0.,atol=1e-7)
    mstar = float(-23.28+.2*state['tracer'][8])
    lf_alpha = float(-1+.06*np.exp(.5*state['tracer'][7]))
    np.savez_compressed(out/'raw_source_mixture.npz',row_ptr=ptr,
        PGC=np.array([source_pgc[o[3]] for o in chosen]),
        source_group=np.array([o[0] for o in chosen]),
        eta=eta_flat,true_k_bin=bin_id.astype(np.int8),log_weight=logw,
        magnitude_lower=lower,magnitude_upper=upper,observed_M=observed_M,
        observed_ksmag=ksmag,population=population,dz_row=dz,
        observed_radius=observed_radius,mstar=mstar,lf_alpha=lf_alpha)
    return dict(rows=len(rows),positive_source_bin_components=len(logw),
        maximum_components_per_row=int(np.max(np.diff(ptr))),
        old_FP_max_abs_discrepancy=float(np.max(np.abs(reproduced-rows))),
        mstar=mstar,lf_alpha=lf_alpha,FP_applied_to_weights=False,
        target_changed=False,selection_calibrated=False,
        definition='count-key/redshift conditioned; K bins NOT collapsed; no FP reweighting',
        limits='constant association; old radial/TSC/periodic-alias approximation; no posterior ensemble')
