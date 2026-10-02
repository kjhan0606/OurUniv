"""External native-K response evidence using existing TNG, no CF4 outcomes.

The TNG K band is a proxy for observational Ks. No inferred coefficients are
injected into R2. Matter uses the preserved NGP total-matter moments, whereas
the live PM source uses its own scatter; that correspondence remains required.
"""
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

RAW = Path('/scratch/kjhan/IllustrisTNG/TNG100-1/output/groups_099')
CATALOGUE = Path('/gpfs/kjhan/CF4/z0_density/r2_tng_native_k_catalog_20261002_v1.h5')
MATTER = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/total_matter_v1/matter_moments.h5')
OUT = Path(os.environ.get('CF4_R2_OUT_DIR',
    '/gpfs/kjhan/CF4/z0_density/r2_tng_native_k_response_20261002_v2'))
EDGES = np.array([-np.inf, -25., -23.-2./3., -22.-1./3., -21., np.inf])


def response(counts, density, train, test):
    """Poisson beta fit including empty cells; rate estimated on train only."""
    log_rho = np.log(density)
    ntrain = int(counts[train].sum())
    ntest = int(counts[test].sum())
    if ntrain == 0:
        return dict(status='NO_TRAINING_SUPPORT', train_count=0, test_count=ntest)
    weighted = float(np.sum(counts[train]*log_rho[train]))

    def negative_profile(beta):
        return ntrain*logsumexp(beta*log_rho[train])-beta*weighted

    fit = minimize_scalar(negative_profile, bounds=(.05, 10.), method='bounded',
                          options=dict(xatol=1e-9))
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError('native response optimization failed')
    beta = float(fit.x)
    # Full-box normalization matches the definition of the R2 bias response.
    logr = beta*log_rho-logsumexp(beta*log_rho.ravel())+np.log(density.size)
    rate = ntrain/np.exp(logr[train]).sum()
    logmu = np.log(rate)+logr[test]
    mu = np.exp(logmu)
    score = float(np.sum(counts[test]*logmu-mu))
    null_mean = ntrain/int(train.sum())
    null_score = float(np.sum(counts[test]*np.log(null_mean)-null_mean))
    selected = mu >= 5.
    return dict(status='NATIVE_K_PROXY_RESPONSE_ONLY', beta=beta,
        optimizer_boundary=bool(beta < .051 or beta > 9.999),
        rate_per_cell=float(rate), train_count=ntrain, test_count=ntest,
        predicted_test_count=float(mu.sum()),
        test_logscore_gain_over_train_rate_uniform=score-null_score,
        test_pearson_expected_ge5=float(np.sum((counts[test][selected]-mu[selected])**2/mu[selected])),
        test_bins_expected_ge5=int(selected.sum()),
        test_empty_cells=int(np.sum(counts[test] == 0)))


def main():
    start = time.monotonic()
    # Known independent Poisson response, including train-only rate profiling.
    control = response(np.array([5,10,20,40]), np.array([.5,1.,2.,4.]),
                       np.array([True,True,True,False]),
                       np.array([False,False,False,True]))
    if abs(control['beta']-1.) > 1e-6 or abs(control['predicted_test_count']-40.) > 1e-4:
        raise ValueError('known-response/rate control failed')
    OUT.mkdir(parents=True, exist_ok=False)
    with h5py.File(CATALOGUE, 'r') as f:
        if f.attrs['status'] != 'COMPLETE_NATIVE_FIELD_COPY_NO_SELECTION':
            raise ValueError('incomplete staged catalogue')
        header = f['chunks/0/Header'].attrs
        h = float(header['HubbleParam'])
        box = float(header['BoxSize'])/1000.
        files = int(header['NumFiles'])
        expected = int(header['Nsubgroups_Total'])
    pieces = {k: [] for k in ('position', 'velocity', 'K_physical', 'star_count', 'native_id')}
    seen = 0
    invalid_stellar_photometry = 0
    with h5py.File(CATALOGUE, 'r') as catalogue:
        for index in range(files):
            f = catalogue[f'chunks/{index}']
            header = f['Header'].attrs
            if float(header['HubbleParam']) != h or float(header['BoxSize'])/1000. != box:
                raise ValueError('inconsistent native catalogue header')
            n = int(header['Nsubgroups_ThisFile'])
            if n:
                sub = f['Subhalo']
                stars = sub['star_count'][:]
                flag = sub['SubhaloFlag'][:].astype(bool)
                mag = sub['K_physical'][:]
                stellar = flag & (stars > 0)
                valid = stellar & np.isfinite(mag) & (np.abs(mag) < 90.)
                invalid_stellar_photometry += int(np.sum(stellar & ~valid))
                pieces['position'].append(sub['SubhaloPos'][:][valid]/1000.)
                pieces['velocity'].append(sub['SubhaloVel'][:][valid])
                pieces['K_physical'].append(mag[valid])
                pieces['star_count'].append(stars[valid])
                pieces['native_id'].append(np.arange(seen, seen+n, dtype=np.int64)[valid])
            seen += n
            if (index+1) % 64 == 0:
                print(f'catalogue {index+1}/{files}, native rows {seen}', flush=True)
    if seen != expected:
        raise ValueError('incomplete native catalogue')
    data = {k: np.concatenate(v) for k, v in pieces.items()}
    if not all(np.isfinite(data[k]).all() for k in ('position', 'velocity', 'K_physical')):
        raise ValueError('nonfinite selected native coordinates')
    data['K_h_proxy'] = data['K_physical'].astype(float)-5*np.log10(h)
    with h5py.File(MATTER, 'r') as f:
        if abs(float(f.attrs['h'])-h) > 1e-12:
            raise ValueError('catalogue/matter cosmology mismatch')
        moments = f['coarse'][:]
        dx = float(f['coarse'].attrs['dx_cMpc_h'])
        omega = float(f.attrs['Omega_m'])
    mass = moments[0]
    n = mass.shape[0]
    if mass.shape != (n,n,n) or abs(n*dx-box) > 1e-12 or np.any(mass <= 0):
        raise ValueError('positive complete native matter field required')
    rho = mass/mass.mean()
    velocity = np.moveaxis(moments[1:4]/mass, 0, -1)
    cells = np.floor((data['position'] % box)/dx).astype(int)
    ix = tuple(cells.T)
    residual = data['velocity']-velocity[ix]
    population = np.searchsorted(EDGES[1:-1], data['K_h_proxy'], side='right')
    x = np.indices(mass.shape)[0]
    train, test = x < 30, x >= 35
    profiles = []
    for p in range(5):
        use = population == p
        keys = np.ravel_multi_index(tuple(cells[use].T), mass.shape)
        counts = np.bincount(keys, minlength=mass.size).reshape(mass.shape)
        report = response(counts, rho, train, test)
        rr = residual[use]
        report.update(population=p, native_count=int(use.sum()),
            stellar_resolution_counts={str(floor): int(np.sum(use & (data['star_count'] >= floor)))
                                       for floor in (1,100,300)},
            residual_mean_xyz_km_s=rr.mean(axis=0).tolist() if len(rr) else None,
            residual_sigma_xyz_km_s=rr.std(axis=0).tolist() if len(rr) else None,
            residual_rms_1d_km_s=float(np.sqrt(np.mean(rr**2))) if len(rr) else None)
        profiles.append(report)
    # IDs label this external calibration source only, never generated candidates.
    np.savez_compressed(OUT/'native_k_galaxies.npz', **data)
    payload = dict(status='EXTERNAL_NATIVE_K_PROXY_RESPONSE_NOT_R2_CALIBRATION',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), catalogue=str(RAW),
        staged_catalogue=str(CATALOGUE),
        matter=str(MATTER), files=files, native_rows=seen,
        selected_galaxies=len(population), invalid_stellar_photometry=invalid_stellar_photometry,
        h=h, Omega_m=omega, box_cMpc_h=box, spacing_cMpc_h=dx,
        magnitude_transform='M_h_proxy = native_K_physical - 5 log10(native_h)',
        finite_true_K_edges=EDGES[1:-1].tolist(),
        split='x<45 train; 45<=x<52.5 buffer; x>=52.5 test (cMpc/h)',
        known_response_control=control, profiles=profiles, elapsed_seconds=time.monotonic()-start,
        limits=['Native K is not calibrated observational 2MASS Ks; no passband/dust crosswalk.',
            'One 75 cMpc/h hydro box, different cosmology from R2; correlated spatial test.',
            'Native NGP total-matter cells differ from the PM source scatter operator.',
            'Subhalo all-matter COM velocities are a proxy for observed stellar velocities.',
            'Poisson power-law adequacy and a single Gaussian LOS width are hypotheses.',
            'No CF4/2M++ observation read, no R2 prior change or field/posterior promotion.',
            'Native IDs label calibration only; MW/M31 ambiguous and M33 unresolved on NEW R2 field.'])
    (OUT/'result.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
