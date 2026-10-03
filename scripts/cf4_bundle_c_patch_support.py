"""One bounded dense-anchor/isotropy test; never count duplicates as new physics."""
from itertools import product
import json
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.special import logsumexp

from cf4_patch_symmetry import (cube_rotations, rotate_features, coarse_features,
    source_groups, grouped_support, read_rotated_patch)

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
SOURCE = ROOT / 'total_matter_v1/matter_moments.h5'


def main():
    start = time.monotonic()
    if json.loads((ROOT / 'total_matter_v1/result.json').read_text())['status'] != 'TOTAL_MATTER_PRIOR_SOURCE_NOT_CF4_POSTERIOR':
        raise ValueError('complete total-matter input required')
    out = ROOT / 'patch_support_v1'
    out.mkdir(exist_ok=False)
    with h5py.File(SOURCE, 'r') as f:
        coarse = f['coarse'][:]
        original_origins = f['patch_origins_coarse'][:]
        original_features = f['patch_coarse_features'][:]
        train = f['patch_train'][:]
        bandwidth = f['prior_bandwidth'][:]
    # Exact original9 targets and prior bandwidth; no changes to the condition.
    np.testing.assert_allclose(coarse_features(coarse, original_origins), original_features, rtol=1e-10, atol=1e-5)
    targets = original_features[~train]
    origins = np.array(list(product(range(19), range(35), range(35))), dtype=np.int64)
    # Heldout cubes start at native x=34; training ends at or before that face.
    if not np.all(origins[:, 0] + 16 <= 34) or not np.all(original_origins[~train, 0] == 34):
        raise ValueError('training/heldout spatial overlap')
    native = coarse_features(coarse, origins)
    rotations = cube_rotations()
    enriched = np.concatenate([rotate_features(native, *rotation) for rotation in rotations])
    baseline = original_features[train]
    rotated = np.concatenate([rotate_features(baseline, *rotation) for rotation in rotations])
    banks = dict(baseline=baseline, rotations_only=rotated, dense_native=native, dense_rotated=enriched)
    groups = dict(baseline=source_groups(original_origins[train]),
        rotations_only=np.tile(source_groups(original_origins[train]), 24),
        dense_native=source_groups(origins), dense_rotated=np.tile(source_groups(origins), 24))
    results = {}
    with h5py.File(out / 'conditionals.h5', 'x') as f:
        f.create_dataset('native_origins_coarse', data=origins)
        f.create_dataset('permutations', data=np.array([r[0] for r in rotations]))
        f.create_dataset('signs', data=np.array([r[1] for r in rotations]))
        f.create_dataset('frozen_bandwidth', data=bandwidth)
        for label, features in banks.items():
            rows = []
            saved = f.create_dataset(label, shape=(9, len(features)), dtype='f8', compression='gzip', compression_opts=1)
            for i, target in enumerate(targets):
                difference = (features - target) / bandwidth
                distance2 = np.sum(difference**2, axis=1)
                log_weight = -.5 * distance2
                log_weight -= logsumexp(log_weight)
                weights = np.exp(log_weight)
                grouped = grouped_support(weights, groups[label])
                residual = weights @ difference
                raw_ess = float(1 / np.dot(weights, weights))
                if label == 'baseline':
                    anchor = weights
                elif label == 'rotations_only':
                    anchor = weights.reshape(24, len(baseline)).sum(axis=0)
                elif label == 'dense_native':
                    anchor = weights
                else:
                    anchor = weights.reshape(24, len(origins)).sum(axis=0)
                rows.append(dict(raw_component_ESS=raw_ess, anchor_ESS=float(1 / np.dot(anchor, anchor)),
                    spatial_group_ESS=grouped['ess'], max_spatial_group_weight=grouped['max_weight'],
                    nearest_standardized_feature_RMS=float(np.sqrt(distance2.min() / len(target))),
                    mean_standardized_feature_RMS=float(np.sqrt(np.mean(residual**2)))))
                saved[i] = weights
            results[label] = rows
            print(label, json.dumps(rows), flush=True)
            if label == 'baseline':
                old = json.loads((ROOT / 'total_matter_v1/prior_support.json').read_text())
                np.testing.assert_allclose([r['raw_component_ESS'] for r in rows], old['heldout_coarse_condition_ESS'], rtol=1e-10)
        f.attrs.update(status='COARSE_SUMMARY_SUPPORT_ONLY_NOT_CF4_LG_POSTERIOR', source=str(SOURCE),
            group_limit='Source tiles are NOT independent; overlap and common long modes remain.')
    # Exercise a rotated, shifted actual fine field, not only summary vectors.
    anchor_index, rotation_index = len(origins) // 2, 7
    sample = read_rotated_patch(SOURCE, origins[anchor_index], rotation_index)
    moments = sample['integrals']
    parent = np.concatenate([moments['mass'][None], moments['momentum'], moments['second_moment']])
    parent = parent.reshape(7, 2, 64, 2, 64, 2, 64).sum(axis=(2, 4, 6))
    actual = np.r_[np.log(parent[0]).ravel(), (parent[1:4] / parent[0]).ravel()]
    np.testing.assert_allclose(actual, enriched[rotation_index * len(origins) + anchor_index], rtol=1e-9, atol=1e-5)
    enriched_rows = results['dense_rotated']
    sufficient = all(r['spatial_group_ESS'] >= 4 and r['max_spatial_group_weight'] <= .5 for r in enriched_rows)
    report = dict(status='ENRICHED_FINITE_SUPPORT_ONLY_NOT_LG_GO' if sufficient else 'NO_GO_FINITE_BANK_SUPPORT_CLOSE_THIS_REPAIR',
        native_anchors=len(origins), rotations=24, component_hypotheses=len(enriched),
        conditioning_dimensions=32, frozen_original_bandwidth=True, heldout_targets=9,
        source_overlap_across_train_holdout=False, actual_fine_transform_check=True,
        independent_universe_count=1, results=results, elapsed_seconds=time.monotonic()-start,
        limits='Translations/rotations are not new independent realizations. Spatial group ESS is only a concentration diagnostic, not an independence certificate. Whole native fields and catalogues require the same rigid transform. No likelihood broadening, exact parent-cell enforcement, actual CF4/LG scoring, continuous prior or phase-recovery claim.',
        next_policy='If insufficient, stop expanding this finite bank; design a continuous jointly changing matter/halo model rather than widen the kernel, duplicate samples or narrow the science target.')
    with (out / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('status', 'component_hypotheses', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
