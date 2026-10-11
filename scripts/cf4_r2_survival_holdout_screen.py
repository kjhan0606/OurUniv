"""Independent survival-mark sky check and saved-state count holdout screen.

No PM evolution, IC inference, bias/FoG tuning or heldout-based model choice.
The saved unconditional N128 state is a fixed mechanics fixture, not a fit.
"""

import json
import os
from pathlib import Path
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_continuous_tracer import predict_continuous_intensity

CATALOGUE = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1')
SELECTION = Path('/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1')
PM_STATE = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1/state.npz')
PREVIOUS = Path('/gpfs/kjhan/CF4/z0_density/r2_rate_nuisance_n128_screen_v2/result.json')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_survival_holdout_screen_v1')
N = 128
DRAW_COUNT = 2048
NULL_COUNT = 1024


def sky_mark_test(rows, yes, no):
    """Conditional permutation test of population/shell exchangeability by octant."""
    mask = rows['calibration']
    offset = rows['position_cMpc_h'][mask]-192.
    octant = ((offset[:,0]>=0).astype(np.int8)*4
               +(offset[:,1]>=0).astype(np.int8)*2
               +(offset[:,2]>=0).astype(np.int8))
    group = rows['population'][mask].astype(np.int32)*6+rows['shell'][mask].astype(np.int32)
    n = np.bincount(group*8+octant,minlength=36*8).reshape(36,8)
    y = np.bincount((group*8+octant)[rows['survives'][mask]],
                    minlength=36*8).reshape(36,8)
    np.testing.assert_array_equal(y.sum(axis=1),yes.reshape(-1))
    np.testing.assert_array_equal((n-y).sum(axis=1),no.reshape(-1))

    def statistic(nk, yk, total_yes):
        p = total_yes/nk.sum()
        variance = nk*p*(1-p)
        active = variance > 0
        return float(np.sum((yk[active]-nk[active]*p)**2/variance[active]))

    active = [(index,nk,int(yk.sum()),yk) for index,(nk,yk) in enumerate(zip(n,y))
              if nk.sum()>0 and 0<yk.sum()<nk.sum()]
    observed = sum(statistic(nk,yk,total) for _,nk,total,yk in active)
    rng = np.random.default_rng(2026092603)
    null = np.zeros(NULL_COUNT)
    for _,nk,total,_ in active:
        for index in range(NULL_COUNT):
            simulated = rng.multivariate_hypergeometric(nk,total)
            null[index] += statistic(nk,simulated,total)
    per_group = [(index//6,index%6,statistic(nk,yk,total),int(nk.sum()))
                 for index,nk,total,yk in active]
    return dict(calibration_marks=int(n.sum()),surviving_marks=int(y.sum()),
                active_population_shells=len(active),octants=8,
                conditional_pearson_statistic=observed,
                conditional_null_median=float(np.median(null)),
                conditional_null_p95=float(np.quantile(null,.95)),
                conditional_permutation_p=(1+int(np.count_nonzero(null>=observed)))/(NULL_COUNT+1),
                conditional_null_draws=NULL_COUNT,
                largest_group_statistics=sorted(per_group,key=lambda entry:entry[2],reverse=True)[:5])


def log_marginal(survival, design_train, design_hold, integrals,
                 train, hold, prior_mean, alpha):
    """Six Gamma-integrated Poisson factors, train and train+hold at fixed S."""
    beta = alpha/prior_mean
    log_constant = alpha*np.log(beta)-gammaln(alpha)
    result = []
    for include_hold in (False,True):
        total = 0.
        for p in range(6):
            train_keys, train_counts = train[p]
            hold_keys, hold_counts = hold[p]
            unit_train = .6*(design_train[p]@survival[p])
            if np.any(unit_train<=0):
                return -np.inf,-np.inf
            weighted = np.dot(train_counts,np.log(unit_train))
            factorial = np.sum(gammaln(train_counts+1.))
            count = np.sum(train_counts)
            exposure = .6*np.dot(integrals[p],survival[p])
            if include_hold:
                unit_hold = .2*(design_hold[p]@survival[p])
                if np.any(unit_hold<=0):
                    return -np.inf,-np.inf
                weighted += np.dot(hold_counts,np.log(unit_hold))
                factorial += np.sum(gammaln(hold_counts+1.))
                count += np.sum(hold_counts)
                exposure += .2*np.dot(integrals[p],survival[p])
            total += (log_constant[p]+gammaln(alpha+count)
                      -(alpha+count)*np.log(beta[p]+exposure)+weighted-factorial)
        result.append(total)
    return tuple(result)


def effective_samples(log_weight):
    weight = np.exp(log_weight-np.max(log_weight))
    return float(np.sum(weight)**2/np.sum(weight**2))


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.monotonic()
    contract = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    cosmology, prior = contract['common_cosmology'],contract['published_prior']
    with np.load(CATALOGUE/'rows.npz',allow_pickle=False) as f:
        rows = {key:f[key] for key in ('position_cMpc_h','population','shell',
                                       'calibration','survives')}
    with np.load(CATALOGUE/'survival_marks.npz',allow_pickle=False) as f:
        yes,no = f['yes'],f['no']
    sky = sky_mark_test(rows,yes,no)
    del rows
    with np.load(CATALOGUE/'counts_3_sparse.npz',allow_pickle=False) as f:
        sparse = {key:f[key] for key in ('train_keys','train_counts',
                                        'holdout_keys','holdout_counts')}
    with np.load(PM_STATE,allow_pickle=False) as f:
        rho = jnp.asarray(f['rho'],dtype=jnp.float64)
        velocity = jnp.asarray(f['velocity_km_s'],dtype=jnp.float64)
    fixed = json.loads((ROOT/'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())['inference_model']
    bias = jnp.asarray(prior['linear_regime_bias_bright_first'])
    def tracer_response(density, mean_velocity):
        return predict_continuous_intensity(density,mean_velocity,
            jnp.ones((6,N,N,N)),jnp.ones(6),bias,jnp.zeros(6),
            box=384.,observer=jnp.asarray([192.]*3),
            hubble=cosmology['H0_km_s_Mpc'],little_h=cosmology['h'],
            sigma_fog=jnp.asarray(fixed['FoG_prior_median_km_s']),
            sigma_redshift=jnp.asarray(fixed['fixed_redshift_error_km_s']))
    raw = jax.jit(tracer_response)(rho,velocity)
    raw = np.asarray(raw)
    if not np.isfinite(raw).all() or np.any(raw<0):
        raise FloatingPointError('nonfinite or negative raw tracer response')
    del rho,velocity
    train = []
    hold = []
    design_train = []
    design_hold = []
    integrals = np.empty((6,6))
    with h5py.File(SELECTION/'selection_3.h5','r') as f:
        shells = f['selection_shells']
        for p in range(6):
            shell = np.asarray(shells[p],dtype=np.float64).reshape(6,-1)
            raw_pop = raw[p].reshape(-1)
            integrals[p] = shell@raw_pop
            for prefix,target,design in (('train',train,design_train),
                                          ('holdout',hold,design_hold)):
                keys = sparse[prefix+'_keys']
                subset = keys//N**3==p
                cell = keys[subset]%N**3
                counts = sparse[prefix+'_counts'][subset].astype(np.float64)
                target.append((cell,counts))
                design.append((shell[:,cell].T*raw_pop[cell,None]).copy())
    del raw
    mean = (yes+1)/(yes+no+2)
    prior_mean = np.asarray(prior['original_mean_count_per_cell_bright_first'])*(
        3./prior['original_cell_cMpc_h'])**3
    alpha = 1.
    mean_train,mean_joint = log_marginal(mean,design_train,design_hold,
                                        integrals,train,hold,prior_mean,alpha)
    previous = json.loads(PREVIOUS.read_text())
    if abs(mean_train-previous['saved_state_rate_log_likelihood_shape_sensitivity']['1.0'])>1e-3:
        raise AssertionError('compressed saved-state count likelihood mismatch')
    rng = np.random.default_rng(2026092604)
    log_train = np.empty(DRAW_COUNT)
    log_joint = np.empty(DRAW_COUNT)
    for i in range(DRAW_COUNT):
        survival = rng.beta(yes+1,no+1)
        log_train[i],log_joint[i] = log_marginal(survival,design_train,
            design_hold,integrals,train,hold,prior_mean,alpha)
    if not np.isfinite(log_train).all() or not np.isfinite(log_joint).all():
        raise FloatingPointError('Beta draw likelihood support failure')
    integrated = float(logsumexp(log_joint)-logsumexp(log_train))
    half_predictive = [float(logsumexp(log_joint[sl])-logsumexp(log_train[sl]))
                       for sl in (slice(None,DRAW_COUNT//2),slice(DRAW_COUNT//2,None))]
    report = dict(classification='R2_SAVED_STATE_SURVIVAL_HOLDOUT_MODEL_STRESS',
        source_commit=os.environ['EXPECTED_COMMIT'],sky_mark_test=sky,
        galaxy_training_count=int(sum(np.sum(c) for _,c in train)),
        galaxy_holdout_count=int(sum(np.sum(c) for _,c in hold)),
        saved_state_rate_train_log_likelihood=float(mean_train),
        saved_state_rate_joint_log_likelihood=float(mean_joint),
        saved_state_mean_survival_holdout_log_predictive=float(mean_joint-mean_train),
        saved_state_Beta_integrated_holdout_log_predictive=integrated,
        Beta_draws=DRAW_COUNT,half_draw_integrated_predictive=half_predictive,
        train_weight_ESS=effective_samples(log_train),
        joint_weight_ESS=effective_samples(log_joint),
        survival_mark_model='independent Beta(yes+1,no+1) by population/shell',
        rate_model='six Gamma(shape=1, published-mean) integrated analytically',
        population_bias_and_FoG_fixed=True,
        unconditional_PM_state_not_posterior=True,
        heldout_used_to_choose_model=False,
        elapsed_seconds=time.monotonic()-started)
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
