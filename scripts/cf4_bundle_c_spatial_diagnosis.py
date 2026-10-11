"""One saved-case ablation, no new model fit, draw selection or amplitude repair."""
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

from cf4_continuous_matter import restrict, check_realizable
from cf4_spatial_copula import (Marginals, detail_channels, conditional_moments,
                               moments_state, expand, block_sum)
from cf4_bundle_c_spatial_model import ROOT, SOURCE, mass_statistics, morphology_comparison

FIT = ROOT / 'spatial_conditional_v1'
OUT = ROOT / 'spatial_diagnosis_v1'


def moment_errors(value, native, coarse):
    mean, variance = moments_state(value)
    nm, nv = moments_state(native)
    weights = native[0]/native[0].sum()
    scale = np.maximum(np.max(abs(coarse), axis=(1, 2, 3)), 1)
    return dict(realizability_relative_deficit=check_realizable(value),
        coarse_channel_relative_error=(np.max(abs(restrict(value, 8)-coarse), axis=(1, 2, 3))/scale).tolist(),
        mass_weighted_velocity_RMSE_km_s=np.sqrt(np.sum(weights*(mean-nm)**2, axis=(1, 2, 3))).tolist(),
        mass_weighted_physical_sigma_RMSE_km_s=np.sqrt(np.sum(weights*(np.sqrt(variance)-np.sqrt(nv))**2, axis=(1, 2, 3))).tolist())


def mass_report(mass, native, coarse, reference):
    comparison = morphology_comparison(mass_statistics(mass), reference)
    comparison.update(total_mass_ratio=float(mass.sum()/native.sum()),
        mass_L1_over_native_mass=float(np.sum(abs(mass-native))/native.sum()),
        coarse_mass_relative_error=float(np.max(abs(block_sum(mass)-coarse[0]))/coarse[0].max()))
    return comparison


def build():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('run this numerical diagnosis via Slurm')
    OUT.mkdir(exist_ok=False)
    start = time.monotonic()
    previous = json.loads((FIT / 'result.json').read_text())
    records, previews = [], []
    rng = np.random.default_rng(44019)  # Randomized CDF atoms only; no new field draws.
    with h5py.File(FIT / 'model.h5', 'r') as model:
        marginal = Marginals(model['marginal_quantiles'][:])
    with h5py.File(SOURCE / 'spatial_components.h5', 'r') as source, h5py.File(FIT / 'conditional_fields.h5', 'r') as saved:
        for i in previous['heldout_patches']:
            native = source[f'patches/{i}/remainder'][:]
            channels, coarse = detail_channels(native)
            reference = mass_statistics(native[0])
            direct = conditional_moments(channels, coarse)
            direct_report = mass_report(direct[0], native[0], coarse, reference)
            direct_report['moments'] = moment_errors(direct, native, coarse)
            if direct_report['mass_L1_over_native_mass'] > 1e-10:
                raise ValueError('native density representation roundtrip is broken')
            preview = [native[0, :, :, 60:68].mean(axis=2), direct[0, :, :, 60:68].mean(axis=2)]
            del direct
            decoded = marginal.invert(marginal.gaussianize(channels, rng))
            before = decoded[0]*expand(coarse[0])/512
            before_report = mass_report(before, native[0], coarse, reference)
            tail = []
            for k in range(5):
                low = channels[k] < marginal.quantiles[k, 0]
                high = channels[k] > marginal.quantiles[k, -1]
                tail.append(dict(channel=k, outside_lower_cell_fraction=float(low.mean()),
                    outside_upper_cell_fraction=float(high.mean()),
                    mass_in_outside_cells_fraction=float(native[0][low | high].sum()/native[0].sum())))
            del channels, before
            reconstructed = conditional_moments(decoded, coarse)
            after_report = mass_report(reconstructed[0], native[0], coarse, reference)
            after_report['moments'] = moment_errors(reconstructed, native, coarse)
            preview.append(reconstructed[0, :, :, 60:68].mean(axis=2))
            # Show fixed draw0, not best/worst draw. Saved product is TOTAL;
            # subtract its unchanged native members to compare remainders.
            fixed_members = saved[f'patch_{i}/native_total'][0]-native[0]
            generated_mass = saved[f'patch_{i}/draw_0'][0]-fixed_members
            preview.append(generated_mass[:, :, 60:68].mean(axis=2))
            previews.append(np.array(preview)/native[0].mean())
            records.append(dict(patch=i, native_statistics=reference,
                direct_representation_roundtrip=direct_report,
                quantile_roundtrip_before_conservation=before_report,
                quantile_roundtrip_after_conservation=after_report,
                marginal_tail_report=tail,
                existing_generated_comparisons=[r['remainder_comparison'] for r in previous['comparisons'] if r['patch'] == i]))
            print(json.dumps(dict(patch=i, direct_L1=direct_report['mass_L1_over_native_mass'],
                quantile_power_before=before_report['power_ratio'],
                quantile_power_after=after_report['power_ratio'])), flush=True)
            del native, decoded, reconstructed, fixed_members, generated_mass
    report = dict(status='SAVED_CASE_CAUSE_SEPARATION_COMPLETE_NOT_NEW_MODEL',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'],
        source_fit_commit=previous['source_commit'], patches=records,
        elapsed_seconds=time.monotonic()-start,
        limits=['Same three now-used development cases, not fresh validation.',
                'No new stochastic draws or fit. Existing12 generated results reused unchanged.',
                'Quantile roundtrip preserves spatial ordering except atoms/clipping; not an exact phase-preserving Fourier experiment.',
                'The residual generated-field gap mixes marginal sampling, missing environment dependence, spectral/cross-mode structure and nonlinear normalization; not uniquely phase causation.',
                'Scalar fine dispersion loses directional information. Its direct velocity roundtrip is not mathematically required to be exact, unlike mass.',
                'All native parent moments supplied; neither observation reconstruction nor a replacement model.'])
    # Single small visual accompanies the numerical ablation; no new full-grid output.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout='constrained')
    titles = ['Native remainder', 'Direct representation', 'Quantile + conservation', 'Existing draw 0']
    for row, (patch, fields) in enumerate(zip(previous['heldout_patches'], previews)):
        for col, field in enumerate(fields):
            im = axes[row, col].imshow(np.log10(1+np.maximum(field, 0)), origin='lower', extent=(0, 24, 0, 24), vmin=0, vmax=2.5, cmap='magma')
            axes[row, col].set_title(f'Patch {patch}: {titles[col]}', fontsize=9)
    fig.colorbar(im, ax=axes, label='log10(1 + slab mass / native mean cell mass)')
    fig.suptitle('Native TNG remainder; central 1.5 cMpc/h slab, 0.1875 grid. NOT the observed LG.')
    fig.savefig(OUT / 'comparison.png', dpi=140)
    plt.close(fig)
    with (OUT / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(report['status'], flush=True)


if __name__ == '__main__':
    build()
