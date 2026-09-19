"""Read-only diagnosis of saved actual-data chains; no repair or new fit."""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_actual_data_preview import ROOT, PLAN, load_data
from cf4_z10_exact_radial import gaussian_block, energy
from cf4_datum_bearing_z0_phasec_pilot import chain_diagnostics


def main():
    start = time.perf_counter()
    plan = json.loads(PLAN.read_text())
    source = Path(plan["output_root"])/"task_0"
    out = Path("/gpfs/kjhan/CF4/z0_density/actual_preview_diagnostic_v1")
    out.mkdir(parents=True, exist_ok=False)
    model, design, _, _, _, counts, heldcounts, radial, _ = load_data(0, plan)
    result = json.loads((source/"result.json").read_text())
    A = np.asarray(model.B)*np.asarray(model.qstd)[None,:]
    variance, train = design["variance"], ~design["holdout"]
    _, covariance, precision = gaussian_block(A, np.zeros_like(radial), variance, train)
    factor = np.linalg.cholesky(precision)
    operator = covariance @ (A[train].T/variance[train])
    radial_base = jax.jit(lambda vector: model.radial_prediction(model.fields(vector)[2], jnp.zeros(24)))
    objective = jax.jit(model.nlp)
    field_fn = jax.jit(model.fields)
    qs, means, whitened, trace_rows, shifts, normalizers = [], [], [], [], [], []
    for chain in range(4):
        with np.load(source/f"chain_{chain}.npz", allow_pickle=False) as saved:
            projection = saved["projections"]
            names = saved["projection_names"].tolist()
            trace_rows.append(projection[:,:24].copy())
            # Predetermined stride: 64 saved field evaluations per chain, no outcome selection.
            draws = saved["retained_vectors"][::4].copy()
        cm, cq, cw, cn = [], [], [], []
        for i, vector in enumerate(draws):
            q = vector[model.field_size+20:model.field_size+24]
            prediction = np.asarray(radial_base(jnp.asarray(vector)))
            mean = operator @ (radial[train]-prediction[train])
            cm.append(mean); cq.append(q); cw.append((q-mean)@factor)
            _, rho, _ = (np.asarray(a) for a in field_fn(jnp.asarray(vector)))
            nuisance = vector[model.field_size:]
            bias = np.asarray(model.bias)*np.exp(model.settings["bias_log_sigma"]*nuisance[6:12])
            log_norm = np.array([float(jax.scipy.special.logsumexp(b*np.log(rho))-np.log(rho.size)) for b in bias])
            cn.append([*log_norm, *(model.settings["alpha_log_sigma"]*nuisance[:6]-log_norm)])
            if i == 0:
                altered = vector.copy(); altered[model.field_size+20:model.field_size+24] += np.array([.1,-.2,.1,.15])
                q2 = altered[model.field_size+20:model.field_size+24]
                np.testing.assert_allclose(float(objective(altered, counts, radial)-objective(vector, counts, radial)),
                    energy(q2,mean,precision)-energy(q,mean,precision), rtol=1e-9, atol=1e-7)
                alt_radial = radial.copy(); alt_radial[~train] += 1e6
                np.testing.assert_array_equal(operator @ (alt_radial[train]-prediction[train]), mean)
        qs.append(cq); means.append(cm); whitened.append(cw); normalizers.append(cn)
        shifts.append(dict(chain=chain, first_half_mean=projection[:512,:24].mean(axis=0).tolist(),
                           second_half_mean=projection[512:,:24].mean(axis=0).tolist()))
        print(f"saved-chain diagnostic {chain+1}/4 complete", flush=True)
    qs, means, whitened, normalizers = map(np.asarray, (qs,means,whitened,normalizers))
    nuisance = np.asarray(trace_rows)
    radial_report = dict(parameter_order=["bulk_x", "bulk_y", "bulk_z", "H0_offset"],
        physical_units=["km/s", "km/s", "km/s", "km/s/Mpc"],
        conditional_SD_physical=(np.sqrt(np.diag(covariance))*design["q_std"]).tolist(),
        chain_q_means_physical=(qs.mean(axis=1)*design["q_std"]).tolist(),
        chain_conditional_means_physical=(means.mean(axis=1)*design["q_std"]).tolist(),
        q_conditional_mean_correlations=[float(np.corrcoef(qs[...,i].ravel(),means[...,i].ravel())[0,1]) for i in range(4)],
        conditional_mean_variance_over_conditional_noise=(means.reshape(-1,4).var(axis=0)/np.diag(covariance)).tolist(),
        conditional_whitened_offset_mean=whitened.reshape(-1,4).mean(axis=0).tolist(),
        conditional_whitened_offset_SD=whitened.reshape(-1,4).std(axis=0).tolist(),
        conditional_mean_diagnostics=chain_diagnostics(means, [f"conditional_mean_{i}" for i in range(4)]),
        same_objective_conditional_identity_checked=True,
        interpretation="Conditional Gaussian block depends on the inferred field. Resampling q alone cannot certify or fix mixing of the field marginal. No q or field was changed.")
    with np.load(source/"observation_predictions.npz", allow_pickle=False) as p:
        predicted = .2*p["intensity_mean"]
        z = p["count_holdout_standardized_residual"]
        exposure = p["selection_exposure"]
        axis = (np.arange(32)+.5)*12-192
        radius = np.sqrt(sum(a*a for a in np.meshgrid(axis,axis,axis,indexing="ij")))
        bins = np.array([0,30,60,90,120,150,200])
        rows = []
        for pop in range(6):
            for lower, upper in zip(bins[:-1], bins[1:]):
                mask = (exposure[pop]>0)&(radius>=lower)&(radius<upper)
                if not mask.any(): continue
                rows.append(dict(population=pop, radius_cMpc_h=[int(lower),int(upper)], cells=int(mask.sum()),
                    heldout_observed=int(heldcounts[pop][mask].sum()), heldout_expected=float(predicted[pop][mask].sum()),
                    residual_mean=float(z[pop][mask].mean()), residual_SD=float(z[pop][mask].std())))
    selected = nuisance[:,:,[0,3,6,9,18,20,22,23]].reshape(-1,8)
    report = dict(status="DIAGNOSTIC_COMPLETE_NO_FIT_OR_REPAIR", source_status=result["status"],
        commit=os.environ["EXPECTED_COMMIT"], job=os.environ["SLURM_JOB_ID"],
        radial=radial_report, chain_half_means=shifts,
        nuisance_correlation_indices=[0,3,6,9,18,20,22,23], nuisance_correlation=np.corrcoef(selected,rowvar=False).tolist(),
        tracer_log_normalizer_quantiles=np.quantile(normalizers[...,:6], [.025,.5,.975],axis=(0,1)).tolist(),
        effective_unnormalized_log_rate_shift_quantiles=np.quantile(normalizers[...,6:], [.025,.5,.975],axis=(0,1)).tolist(),
        count_radial_diagnostics=rows,
        limits="Descriptive statistics of unconverged chains, 64 retained fields per chain. Binned residuals are correlated and not significance tests; physical truth is unknown. No heldout retuning, posterior promotion, sampler change or new simulation.",
        elapsed_s=time.perf_counter()-start)
    (out/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "conditional_SD_physical": radial_report["conditional_SD_physical"],
        "q_conditional_mean_correlations": radial_report["q_conditional_mean_correlations"]}), flush=True)


if __name__ == "__main__": main()
