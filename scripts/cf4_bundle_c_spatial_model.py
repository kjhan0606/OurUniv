"""One bounded spatial conditional-prior fit, native-data holdout and field draws.

This isolates the remainder law using native coarse moments and FIXED native
member components. A coarse-field prior and stochastic halo-field coupling
are NOT yet supplied, so no actual LG/CF4 posterior is asserted by a pass.
"""
from itertools import combinations
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.ndimage import label

from cf4_spatial_copula import Marginals, detail_channels, conditional_moments, spectral_draw
from cf4_continuous_matter import restrict, check_realizable

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
SOURCE = ROOT / 'spatial_calibration_v1'
OUT = ROOT / 'spatial_conditional_v1'


def overlaps(a, b):
    # Both native periodic cubes have side24<75/2 and exact grid-aligned faces.
    d = abs((np.asarray(a)-b+37.5) % 75-37.5)
    return bool(np.all(d < 24))


def choose_split(patches):
    train = [i for i, p in enumerate(patches) if not p['heldout']]
    held = [i for i, p in enumerate(patches) if p['heldout']]
    best = None
    for subset in combinations(held, 3):
        if any(overlaps(patches[i]['patch_lower_cMpc_h'], patches[j]['patch_lower_cMpc_h']) for i, j in combinations(subset, 2)):
            continue
        keep = [i for i in train if all(not overlaps(patches[i]['patch_lower_cMpc_h'], patches[j]['patch_lower_cMpc_h']) for j in subset)]
        if best is None or len(keep) > len(best[0]):
            best = (keep, list(subset))
    if best is None or len(best[0]) < 8:
        raise ValueError('cannot get>=8 training cubes and3 disjoint heldout cubes; no relaxed split')
    return best


def halo_field(f, record):
    value = np.zeros((7, 128, 128, 128))
    origin = np.rint(np.array(record['patch_lower_cMpc_h'])/.1875).astype(int)
    for sid in record['ids']:
        group = f[f'halos/{sid}']
        cells = (group['global_cell'][:]-origin) % 400
        key = np.ravel_multi_index(cells.T, (128,)*3)
        value.reshape(7, -1)[:, key] += group['moments'][:]
    return value


def mass_statistics(mass):
    contrast = mass/mass.mean()-1
    fft = np.fft.rfftn(contrast, norm='ortho')
    axis = np.fft.fftfreq(128, d=.1875)*2*np.pi
    last = np.fft.rfftfreq(128, d=.1875)*2*np.pi
    wave = np.sqrt(axis[:, None, None]**2+axis[None, :, None]**2+last[None, None, :]**2)
    edges = np.geomspace(2*np.pi/24, np.pi/.1875, 9)
    power = np.abs(fft)**2
    # Account for the implicit negative-z half of the real FFT.
    multiplicity = np.ones_like(power)*2
    multiplicity[:, :, [0, -1]] = 1
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (wave >= lo) & (wave < hi)
        bands.append(float(np.sum(power[mask]*multiplicity[mask])/multiplicity[mask].sum()))
    threshold = np.quantile(mass, .99)
    hot = mass > threshold
    components, _ = label(hot)  # Same nonperiodic6-neighbor statistic for both cubes.
    sizes = np.bincount(components.ravel())[1:]
    return dict(power=bands, k_edges_h_cMpc=edges.tolist(),
        top1pct_mass_fraction=float(mass[hot].sum()/mass.sum()),
        largest_hot_component_fraction=float(sizes.max(initial=0)/max(hot.sum(), 1)),
        density_quantiles_01_16_50_84_99=np.quantile(mass/mass.mean(), [.01, .16, .5, .84, .99]).tolist())


def profile_regression(source, train, held):
    profiles = {p['sid']: p for p in source['profiles']}
    result = {}
    for primary in (True, False):
        def matrix(indices):
            ids = sorted({sid for i in indices for sid in source['patches'][i]['ids'] if profiles[sid]['primary'] == primary})
            x, y = [], []
            for sid in ids:
                p = profiles[sid]
                mass = p['host_M200c_Msun'] if primary else p['mass_Msun']
                increments = np.diff(np.r_[0., p['cumulative_radius_kpc']])
                if np.any(increments <= 0):
                    raise ValueError('nonpositive cumulative-radius increment; no profile floor')
                sigma = np.sqrt(np.trace(p['physical_velocity_covariance_km2_s2'])/3)
                x.append([1., np.log(mass/1e12)])
                y.append(np.r_[np.log(p['mass_Msun']/mass)] if primary else np.empty(0))
                y[-1] = np.r_[y[-1], np.log(increments), np.log(sigma)]
            return ids, np.array(x), np.array(y)
        tids, x, y = matrix(train)
        vids, vx, vy = matrix(held)
        coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y-x@coefficient
        covariance = residual.T@residual/(len(y)-2)
        covariance = .9*covariance+.1*np.diag(np.diag(covariance))
        np.linalg.cholesky(covariance)
        result['primary' if primary else 'satellite'] = dict(training_ids=tids, heldout_ids=vids,
            coefficients=coefficient.tolist(), residual_covariance=covariance.tolist(),
            heldout_standardized_residuals=((vy-vx@coefficient)/np.sqrt(np.diag(covariance))).tolist(),
            interpretation='Gaussian log-radius increments yield positive monotone mass-quantile profiles; include log bound/host ratio for primaries and log scalar dispersion. Not yet rendered/sphericalized or coupled to new matter fields. Small correlated calibration set; hyperparameter uncertainty omitted.')
    return result


def morphology_comparison(stats, reference):
    power = np.array(stats['power'])/reference['power']
    concentration = stats['top1pct_mass_fraction']/reference['top1pct_mass_fraction']
    connectivity = stats['largest_hot_component_fraction']/reference['largest_hot_component_fraction']
    return dict(power_ratio=power.tolist(), top1pct_mass_fraction_ratio=concentration,
        largest_hot_component_ratio=connectivity,
        development_morphology_pass=bool(np.all((power[4:] >= .5) & (power[4:] <= 2))
            and .5 <= concentration <= 2 and .5 <= connectivity <= 2),
        native_statistics=reference, generated_statistics=stats)


def build():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('numerical model fitting requires Slurm')
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    source = json.loads((SOURCE / 'result.json').read_text())
    train, held = choose_split(source['patches'])
    print(f'Geometry-only split: train={train}, held={held}; no shared train/held voxels', flush=True)
    rng = np.random.default_rng(14092)
    samples = []
    with h5py.File(SOURCE / 'spatial_components.h5', 'r') as f:
        if f.attrs['status'] != 'NATIVE_JOINT_PROFILES_AND_REMAINDER_NOT_CF4_LG_POSTERIOR':
            raise ValueError('complete native spatial calibration required')
        for i in train:
            channels, _ = detail_channels(f[f'patches/{i}/remainder'][:])
            # Fixed uniform cell sample; no observational or heldout selection.
            sample = rng.choice(channels[0].size, size=32768, replace=False)
            samples.append(channels.reshape(5, -1)[:, sample])
        quantiles = np.quantile(np.concatenate(samples, axis=1), np.linspace(0, 1, 1025), axis=1).T
        del samples, channels
        marginals = Marginals(quantiles)
        factors, variance = [], np.zeros(5)
        for i in train:
            channels, _ = detail_channels(f[f'patches/{i}/remainder'][:])
            latent = marginals.gaussianize(channels, rng)
            latent -= latent.mean(axis=(1, 2, 3), keepdims=True)
            variance += np.mean(latent**2, axis=(1, 2, 3))/len(train)
            factors.append(np.fft.rfftn(latent, axes=(-3, -2, -1), norm='ortho').astype(np.complex64))
            print(f'Fitted spatial spectral factor {i}', flush=True)
        del channels, latent
        # Gaussian copula coordinates require unit variance. This is a latent
        # covariance convention; never rescale an observed density spectrum.
        normalization = 1/np.sqrt(variance)
        for factor in factors:
            factor *= normalization[:, None, None, None]
        diagonal = sum(np.abs(factor)**2/len(factors) for factor in factors)
        profiles = profile_regression(source, train, held)
        with h5py.File(OUT / 'model.h5', 'x') as model:
            model.attrs.update(status='EMPIRICAL_CONDITIONAL_REMAINDER_LAW_NOT_FULL_JOINT_PRIOR',
                source_commit=os.environ['EXPECTED_COMMIT'], train_indices=train, heldout_indices=held,
                periodic_copula_patch=True, limits='Native coarse7 moments and halo components held fixed for evaluation. No coarse-field prior or stochastic halo-field coupling. Higher-order phase dependence absent.')
            model.create_dataset('marginal_quantiles', data=quantiles)
            model.create_dataset('latent_variance_before_unit_normalization', data=variance)
            model.create_dataset('diagonal_power', data=diagonal, compression='gzip', compression_opts=1)
            for i, factor in enumerate(factors):
                model.create_dataset(f'factors/{i}', data=factor, compression='gzip', compression_opts=1)
            model.attrs['profile_regression_json'] = json.dumps(profiles)
        reports = []
        with h5py.File(OUT / 'conditional_fields.h5', 'x') as output:
            output.attrs.update(status='INCOMPLETE', dx_cMpc_h=.1875,
                limits='Conditional native-coarse oracle comparison, not CF4/LG reconstruction. Fixed source member components are known inputs, not newly found/resolved halos. Four draws do not calibrate posterior uncertainty.')
            for i in held:
                native = f[f'patches/{i}/remainder'][:]
                coarse = restrict(native, 8)
                members = halo_field(f, source['patches'][i])
                reference = mass_statistics(native[0]+members[0])
                remainder_reference = mass_statistics(native[0])
                g = output.create_group(f'patch_{i}')
                g.attrs['native_record_json'] = json.dumps(source['patches'][i])
                g.create_dataset('native_total', data=native+members, compression='gzip', compression_opts=1)
                density_sum, density_sq = np.zeros_like(native[0]), np.zeros_like(native[0])
                for draw in range(4):
                    latent = spectral_draw(factors, diagonal, rng, (128,)*3)
                    channels = marginals.invert(latent)
                    value = conditional_moments(channels, coarse)
                    check_realizable(value)
                    error = float(np.max(abs(restrict(value, 8)-coarse)/np.maximum(np.max(abs(coarse), axis=(1, 2, 3))[:, None, None, None], 1)))
                    if error > 1e-8:
                        raise ValueError('conditional coarse moment conservation failed')
                    total = value+members
                    stats = mass_statistics(total[0])
                    comparison = morphology_comparison(stats, reference)
                    remainder_comparison = morphology_comparison(mass_statistics(value[0]), remainder_reference)
                    # Predeclared broad diagnostics on individual draws, not
                    # mean-map smoothing. No after-the-fact power correction.
                    accepted = comparison['development_morphology_pass'] and remainder_comparison['development_morphology_pass']
                    reports.append(dict(patch=i, draw=draw, conservative_relative_error=error,
                        total_comparison=comparison, remainder_comparison=remainder_comparison,
                        development_morphology_pass=accepted))
                    g.create_dataset(f'draw_{draw}', data=total, compression='gzip', compression_opts=1)
                    density_sum += total[0]
                    density_sq += total[0]**2
                    print(f'Conditional patch={i}, draw={draw}, morphology_pass={accepted}', flush=True)
                    del value, total, latent, channels
                g.create_dataset('four_draw_mean_mass', data=density_sum/4, compression='gzip')
                g.create_dataset('four_draw_mass_variance', data=np.maximum(density_sq/4-(density_sum/4)**2, 0), compression='gzip')
            output.attrs['status'] = 'CONDITIONAL_DRAW_DIAGNOSTICS_COMPLETE_NOT_OBSERVED_FIELD'
    accepted = all(r['development_morphology_pass'] for r in reports)
    report = dict(status='CONDITIONAL_REMAINDER_DIAGNOSTIC_PASS_NOT_FULL_JOINT_PRIOR' if accepted else 'NO_GO_CONDITIONAL_REMAINDER_MORPHOLOGY',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'],
        training_patches=train, heldout_patches=held, train_heldout_native_voxel_overlap=False,
        heldout_mutual_voxel_overlap=False, comparisons=reports, profiles=profiles,
        elapsed_seconds=time.monotonic()-started,
        limits=['Geometry-only split; same box long modes and training overlaps remain, not independent universes.',
                'Gaussian copula is a candidate for conditional remainder, not an exact cosmological field law; higher-order phases are missing.',
                'Empirical bounded quantile marginals, randomized atoms, fixed5% spectral diagonal shrinkage, stationary periodic latent patch; none tuned on heldout.',
                'All seven native coarse moments and unchanged native member components supplied as oracle conditions. Not CF4 or actual LG inference.',
                'Mass-radius profile probability model fitted separately, not yet rendered or jointly coupled. Coarse prior, halo feasibility normalization and actual observation likelihood remain outstanding.',
                'Mass variance over four conditional draws is a diagnostic, not calibrated posterior uncertainty. Physical sigma derives from moments separately.',
                'A failure must not trigger amplitude repair, wider kernels, best-draw choice, more seeds or further Gaussian-copula variants automatically.'])
    with (OUT / 'result.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons', 'profiles')}), flush=True)


if __name__ == '__main__':
    build()
