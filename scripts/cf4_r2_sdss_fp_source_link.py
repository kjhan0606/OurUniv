"""One bounded source bridge + differentiable FP pullback. Not posterior fit."""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_fp_distance import predicted_eta, fp_log_likelihood_ratio

SOURCE = Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')
NATIVE = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz')
STATE = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1/state.npz')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_sdss_fp_source_link_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    if hashlib.md5(SOURCE.read_bytes()).hexdigest() != 'b5b6e31caf7ea469c2ac2cb775fa8d14':
        raise ValueError('published SDSS v1.1 checksum mismatch')
    if hashlib.sha256((ROOT/'data/cf4_galaxies.csv').read_bytes()).hexdigest() != '28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709':
        raise ValueError('CF4 source changed')
    with SOURCE.open() as f:
        columns = f.readline().lstrip('#').split()
        rows = [dict(zip(columns, line.split(), strict=True)) for line in f if line.strip()]
    with (ROOT/'data/cf4_galaxies.csv').open() as f:
        cf4rows = list(csv.DictReader(f))
    counts = Counter(int(r['PGC']) for r in cf4rows)
    cf4 = {int(r['PGC']): r for r in cf4rows if counts[int(r['PGC'])] == 1}
    source_counts = Counter(int(r['PGC']) for r in rows)
    with np.load(NATIVE) as f:
        native = dict(zip(f['CF4_pgc'].astype(int), f['CF4_holdout'].astype(bool), strict=True))
    matched = [r for r in rows if int(r['PGC']) > 0 and source_counts[int(r['PGC'])] == 1
               and int(r['PGC']) in cf4 and int(cf4[int(r['PGC'])]['1PGC']) in native
               and cf4[int(r['PGC'])]['DMfp']]
    if not matched:
        raise RuntimeError('no unique native CF4 FP matches')
    get = lambda name: np.array([float(r[name]) for r in matched])
    pgc = get('PGC').astype(np.int64)
    cfgroup = np.array([int(cf4[p]['1PGC']) for p in pgc])
    # Preserve BOTH observed groupings: close holdout over their bipartite graph.
    tgroup = [f"T{r['IDgroupT17']}" if int(r['IDgroupT17']) > 0 else f"P{r['PGC']}" for r in matched]
    held_cf = {int(g) for g in cfgroup if native[int(g)]}
    held_t = set()
    while True:
        before = len(held_cf) + len(held_t)
        held_t.update(t for g, t in zip(cfgroup, tgroup) if g in held_cf)
        held_cf.update(int(g) for g, t in zip(cfgroup, tgroup) if t in held_t)
        if before == len(held_cf) + len(held_t):
            break
    held = np.array([g in held_cf for g in cfgroup])
    # Source provides Galactic l,b. Same de Vaucouleurs rotation as the
    # existing cf4_lg_bulkvel.gal_to_sg fallback; no new runtime dependency.
    def unit(l, b):
        l, b = np.radians(l), np.radians(b)
        return np.stack([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)], axis=-1)
    pole = unit(47.37, 6.32)
    xaxis = unit(137.37, 0.)
    yaxis = np.cross(pole, xaxis)
    directions = unit(get('l'), get('b')) @ np.stack([xaxis, yaxis, pole]).T
    ztable = np.linspace(0., .2, 20001)
    dtable = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+ztable)**3+.69), ztable, initial=0.)
    zgroup = get('zcmb_group')
    distz = np.interp(zgroup, ztable, dtable)
    # Freeze geometry from observed redshift, not from eta or the trial field.
    eligible = (distz > 15.) & (distz < 180.) & (get('logdist_corr_err') > 0)
    for name in ('logdist_corr', 'logdist_corr_err', 'logdist_corr_alpha'):
        eligible &= np.isfinite(get(name))
    args = [jnp.asarray(x[eligible]) for x in (directions, zgroup)]
    mean, std, alpha = [jnp.asarray(get(n)[eligible]) for n in ('logdist_corr', 'logdist_corr_err', 'logdist_corr_alpha')]
    with np.load(STATE) as f:
        velocity = jnp.asarray(f['velocity_km_s'], dtype=jnp.float64)
    def predict(a):
        return predicted_eta(velocity, *args, jnp.asarray(ztable), jnp.asarray(dtable), amplitude=a)
    evaluate = jax.jit(predict)
    eta, distance, residual = map(np.asarray, evaluate(1.))
    inside = np.all(np.abs(np.asarray(args[0])*distance[:, None]) < 192., axis=1)
    if not np.all(inside & np.isfinite(eta) & (np.abs(residual)*299792.458 < .01)):
        detail = dict(rows=len(eta), outside=int((~inside).sum()),
                      nonfinite=int((~np.isfinite(eta)).sum()),
                      unconverged=int((np.abs(residual)*299792.458 >= .01).sum()),
                      max_residual_km_s=float(np.nanmax(np.abs(residual))*299792.458))
        raise RuntimeError(f'cold-root convergence/support failure; no rows dropped: {detail}')
    selected_held = held[eligible]
    # No fitting to either split. This score tests source-factor wiring only.
    train = jnp.asarray(~selected_held)
    def score(a):
        predicted = predict(a)[0]
        return jnp.sum(jnp.where(train, fp_log_likelihood_ratio(predicted, 0., mean, std, alpha), 0.))
    score_grad = jax.jit(jax.value_and_grad(score))
    value, gradient = map(float, score_grad(1.))
    eps = 1e-4
    finite_difference = float((score(1+eps)-score(1-eps))/(2*eps))
    if not np.isclose(gradient, finite_difference, rtol=2e-3, atol=1e-4):
        raise RuntimeError('field-amplitude derivative mismatch')
    ratios = np.asarray(fp_log_likelihood_ratio(jnp.asarray(eta), 0., mean, std, alpha))
    summary = dict(classification='SOURCE_FP_LIKELIHOOD_PULLBACK_COLD_CONTROL_NOT_POSTERIOR',
        code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), ROOT/'src/cf4_r2_fp_distance.py', ROOT/'tests/test_cf4_r2_fp_distance.py')},
        job_id=os.environ['SLURM_JOB_ID'], source_doi='10.5281/zenodo.6824749',
        source_md5=hashlib.md5(SOURCE.read_bytes()).hexdigest(),
        source_rows=len(rows), unique_CF4_native_FP_matches=len(matched), eligible_rows=int(eligible.sum()),
        eligible_CF4_groups=int(len(set(cfgroup[eligible]))),
        train_rows=int((~selected_held).sum()), holdout_rows=int(selected_held.sum()),
        holdout_promotions_from_group_closure=int(sum(held & ~np.array([native[int(g)] for g in cfgroup]))),
        eta_predicted_rms=float(np.sqrt(np.mean(eta**2))),
        root_max_residual_km_s=float(np.max(np.abs(residual))*299792.458),
        train_log_ratio_to_zero_velocity=value, train_amplitude_derivative=gradient,
        derivative_finite_difference=finite_difference,
        holdout_log_ratio_to_zero_velocity=float(ratios[selected_held].sum()),
        same_state=str(STATE), fitted_parameters=0, new_gravity_runs=0,
        selection_correction_already_in_source=True, source_flat_eta_prior=True,
        source_uses_group_redshift=True, shared_zero_point_marginalized=False,
        FoG_group_kernel_calibrated=False, full_CF4_methods_included=False,
        overlap_joint_likelihood=False, R2_posterior=False)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'source_link.npz', PGC=pgc[eligible], CF4_group=cfgroup[eligible],
        source_group=np.array(tgroup)[eligible], holdout=selected_held, directions=directions[eligible],
        zgroup=zgroup[eligible], eta_mean=np.asarray(mean), eta_std=np.asarray(std),
        eta_alpha=np.asarray(alpha), eta_predicted=eta, distance_cMpc_h=distance)
    (OUT/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
