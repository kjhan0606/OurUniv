"""Compare the frozen 2M++ datum with the CF4/PM fiducial cosmology.

This is a read-only scientific sensitivity audit of the catalogue. It does not
reclassify the likelihood data or certify a new selection function.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_actual_selection import base, magnitude_h

OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_cosmology_contract_audit_v1')
NATIVE = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2')


def labels(catalog, cosmology, design):
    radius, magnitude = base.distance_and_absolute_magnitude(
        catalog['Vcmb'], catalog['Ksmag'], cosmology)
    eligible, _, apparent, absolute = base.classify_disjoint_tracer(
        catalog, set(), radius, magnitude_h(magnitude, cosmology['h']), design)
    return radius, eligible, 3*apparent + absolute


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue calculation must run through Slurm')
    if OUT.exists():
        raise FileExistsError(OUT)
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    old = tracer['cosmology']
    new = dict(old, source='CF4 and R1 PM fiducial', h=.746,
               H0_km_s_Mpc=74.6, Omega_m=.31, Omega_b=.05)
    catalog = base.load_catalog(tracer['inputs']['twompp_catalog']['path'])
    ro, eo, po = labels(catalog, old, tracer['tracer_design'])
    rn, en, pn = labels(catalog, new, tracer['tracer_design'])
    with np.load(NATIVE/'galaxy_native.npz', allow_pickle=False) as frozen:
        recno = frozen['recno']
        frozen_pop = frozen['population']
        frozen_radius = frozen['radius_cMpc_h']
        used = frozen['survives'] & ~frozen['calibration']
        order = np.argsort(catalog['recno'])
        idx = order[np.searchsorted(catalog['recno'][order], recno)]
        np.testing.assert_array_equal(catalog['recno'][idx], recno)
        np.testing.assert_array_equal(po[idx], frozen_pop)
        np.testing.assert_allclose(ro[idx], frozen_radius, rtol=1e-12, atol=1e-10)
        if not np.all(eo[idx]):
            raise AssertionError('frozen row eligibility changed at source fiducial')
        box = 384.
        direction = frozen['direction']
        oldpos = ro[idx, None]*direction+box/2
        newpos = rn[idx, None]*direction+box/2
        cell_o = np.floor(oldpos/3).astype(int)
        cell_n = np.floor(newpos/3).astype(int)
        bin_change = (po[idx] != pn[idx])
        eligible_change = ~en[idx]
        cell_change = np.any(cell_o != cell_n, axis=1)
        radial_delta = rn[idx]-ro[idx]
    with np.load(NATIVE/'CF4_native.npz', allow_pickle=False) as cf4:
        cf4_radius = np.linalg.norm(cf4['CF4_pos']-192., axis=1)
        # Reinterpreting fixed physical CF4 distances at h=.6711 would move
        # their coordinates by this factor; velocities cannot be rescaled so.
        cf4_coord_change = cf4_radius*(old['h']/new['h']-1)
    summary = dict(classification='CATALOGUE_COSMOLOGY_SENSITIVITY_ONLY',
                   old_cosmology=old, candidate_common_cosmology=new,
                   catalog_rows=int(len(ro)), source_eligible=int(eo.sum()),
                   candidate_eligible=int(en.sum()),
                   all_eligibility_changed=int(np.count_nonzero(eo != en)),
                   frozen_rows=int(len(recno)), frozen_used=int(used.sum()),
                   frozen_rows_lost_eligibility=int(eligible_change.sum()),
                   frozen_used_lost_eligibility=int(np.count_nonzero(eligible_change & used)),
                   frozen_population_changed=int(bin_change.sum()),
                   frozen_used_population_changed=int(np.count_nonzero(bin_change & used)),
                   frozen_N128_cell_changed=int(cell_change.sum()),
                   frozen_used_N128_cell_changed=int(np.count_nonzero(cell_change & used)),
                   frozen_radial_delta_cMpc_h_quantiles=np.quantile(radial_delta,
                       [0, .01, .5, .99, 1]).tolist(),
                   CF4_rows=int(len(cf4_radius)),
                   CF4_coordinate_delta_if_rebased_cMpc_h_quantiles=np.quantile(
                       cf4_coord_change, [0, .5, .95, 1]).tolist(),
                   quantitative_joint_likelihood_ready=False,
                   selection_integral_recomputed=False,
                   CF4_velocity_rederived=False)
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
