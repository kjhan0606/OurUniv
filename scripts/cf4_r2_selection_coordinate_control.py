"""Small fixed-distance comparison of the six-bin source selection laws.

The result is a geometry/model sensitivity control, not a fitted survey
likelihood or an observed-universe velocity estimate.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_r2_observed_magnitude_transfer import (
    _schechter_interval_probability, observed_magnitude_transfer)
from cf4_twompp_joint_information_budget_pilot_v1 import (
    _cosmology_distance_table, schechter_fraction)

OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_selection_coordinate_control_v1.json')
TRUE_EDGES = (-np.inf, -25., -23.6666666666667,
              -22.3333333333333, -21., np.inf)
OBS_EDGES = TRUE_EDGES[1:-1]


def one_case(r_true, r_observed, cosmology):
    dl_h = _cosmology_distance_table(np.array([r_true, r_observed]), cosmology) * cosmology['h']
    mu_r, mu_s = 5*np.log10(dl_h) + 25
    transfer = observed_magnitude_transfer(mu_r, mu_s)
    intrinsic_lf = np.array([
        _schechter_interval_probability(TRUE_EDGES[j], TRUE_EDGES[j+1],
                                        mstar=-23.28, alpha=-.94)
        for j in range(5)])
    source_consistent = transfer @ intrinsic_lf
    old_observed_voxel = np.array([
        _schechter_interval_probability(OBS_EDGES[p%3], OBS_EDGES[p%3+1],
                                        mstar=-23.28, alpha=-.94)
        * schechter_fraction(np.array([dl_h[1]]),
            None if p//3 == 0 else 11.5, 11.5 if p//3 == 0 else 12.5,
            OBS_EDGES[p%3], OBS_EDGES[p%3+1], -23.28, -.94)[0]
        for p in range(6)])
    joint = transfer * intrinsic_lf[None, :]
    total = float(joint.sum())
    tail = float(joint[:, [0, 4]].sum()) / total
    migrated_core = sum(float(joint[p, j]) for p in range(6)
                        for j in (1, 2, 3) if p%3 != j-1) / total
    difference = float(np.abs(source_consistent-old_observed_voxel).sum()) / total
    return dict(true_radius_cMpc_h=r_true, observed_radius_cMpc_h=r_observed,
                modulus_shift_mag=float(mu_s-mu_r), selected_lf_measure=total,
                bright_faint_intrinsic_tail_fraction=tail,
                core_absolute_bin_migration_fraction=migrated_core,
                post_RSD_exposure_L1_difference_over_selected=difference)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit numerical comparison through Slurm')
    if OUT.exists():
        raise FileExistsError(OUT)
    cosmology = json.loads((ROOT / 'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
    cases = [one_case(float(r), float(r+shift), cosmology)
             for r in (10., 30., 90., 175.) for shift in (-3., 0., 3.)]
    zero = [x for x in cases if x['true_radius_cMpc_h'] == x['observed_radius_cMpc_h']]
    if max(x['post_RSD_exposure_L1_difference_over_selected'] for x in zero) > 1e-12:
        raise AssertionError('zero-displacement source and voxel laws differ')
    report = dict(status='SELECTION_COORDINATE_DIAGNOSTIC_NOT_CALIBRATED',
                  model='Schechter LF only; no bias, angular mask, grouped marks or actual velocities',
                  displacement_cMpc_h=[-3., 0., 3.],
                  true_radii_cMpc_h=[10., 30., 90., 175.],
                  redshift_radial_cut_not_applied=True,
                  intrinsically_out_of_range_bins_included=True,
                  cases=cases)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
