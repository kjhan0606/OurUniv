"""Choose an R2 heldout octant from survey footprint alone, before fitting.

No observed velocities, distances, magnitudes, counts residuals, model state,
likelihood, or prior score enters the choice. Exclude octant 3 because the
earlier luminosity-function diagnostic used it as a validation region.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_r2_sky_closed_split import octant_from_xyz

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE / 'r2_sky_footprint_choice_v1'


def complete_group_counts(octants, labels):
    _, inverse = np.unique(labels, return_inverse=True)
    size = int(inverse.max())+1
    counts = np.zeros(8, dtype=np.int64)
    for octant in range(8):
        inside = np.bincount(inverse, weights=(octants == octant), minlength=size)
        total = np.bincount(inverse, minlength=size)
        counts[octant] = np.count_nonzero(inside == total)
    return counts


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(BASE/'r2_point_mark_manifest_v1/points.npz', allow_pickle=False) as f:
        cells = np.stack(np.unravel_index(f['flat_cell'], (128,)*3), axis=1)
        point_oct = octant_from_xyz(cells-64)
    with np.load(BASE/'bundle_c_v1/native_data_v2/CF4_native.npz', allow_pickle=False) as f:
        cf4_oct = octant_from_xyz(f['CF4_pos']-192.)
    with np.load(BASE/'r2_source_observation_assembly_v1/observations.npz',
                 allow_pickle=False) as f:
        fp_oct = octant_from_xyz(f['directions'])
        fp_groups = f['source_group']
    with np.load(BASE/'r2_tf_matched_point_bridge_v1/tf_groups_linked.npz',
                 allow_pickle=False) as f:
        matched = f['point_population'] >= 0
        tf_oct = octant_from_xyz(f['directions'][matched])
    point = np.bincount(point_oct, minlength=8)
    cf4 = np.bincount(cf4_oct, minlength=8)
    fp = complete_group_counts(fp_oct, fp_groups)
    tf = np.bincount(tf_oct, minlength=8)
    # Predeclared coverage criterion, no mark or field values. Each term is
    # coverage relative to a uniform eight-octant share of that sample.
    eligible = ((point >= 3000) & (cf4 >= 1000) & (fp >= 100)
                & (tf >= 100))
    eligible[3] = False
    if not np.any(eligible):
        raise ValueError('no octant supports all four validation modalities')
    coverage = np.stack((point*8/point.sum(), cf4*8/cf4.sum(),
                         fp*8/len(np.unique(fp_groups)), tf*8/matched.sum()))
    quality = np.min(coverage, axis=0)
    quality[~eligible] = -1.
    chosen = int(np.argmax(quality))
    report = dict(classification='R2_GEOMETRY_ONLY_HOLDOUT_CHOICE',
                  job_id=os.environ['SLURM_JOB_ID'],
                  previous_development_octant=5,
                  excluded_lf_diagnostic_octant=3,
                  point_octant_counts=point.tolist(),
                  native_cf4_octant_counts=cf4.tolist(),
                  complete_fp_source_group_octant_counts=fp.tolist(),
                  matched_tf_octant_counts=tf.tolist(),
                  eligible_octants=np.flatnonzero(eligible).tolist(),
                  coverage_minimum=quality.tolist(), chosen_octant=chosen,
                  choice_rule='maximize minimum observed-footprint coverage '
                              'relative to an equal one-eighth share, subject '
                              'to point>=3000, CF4>=1000, complete FP>=100, '
                              'matched TF>=100 and octant!=3; lowest index breaks ties',
                  no_observation_values_or_model_scores_read=True,
                  no_fit_or_posterior=True)
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
