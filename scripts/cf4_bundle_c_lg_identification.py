"""One cached native-field identification report, not an observed LG posterior."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cf4_bundle_c_continuous import read_periodic_patch
from cf4_lg_field_candidates import identify, match_for_evaluation, aperture_offsets

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT / 'lg_identification_v1'
DX = .1875
ROLES = ('MW', 'M31', 'M33')


def overlap(cells, matched, radius):
    sets = [None if i < 0 else set(map(tuple, cells[i]+aperture_offsets(radius))) for i in matched]
    return {f'{ROLES[i]}-{ROLES[j]}': None if sets[i] is None or sets[j] is None
            else len(sets[i] & sets[j])/min(len(sets[i]), len(sets[j]))
            for i, j in ((0, 1), (0, 2), (1, 2))}


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('EXPECTED_COMMIT'):
        raise RuntimeError('Slurm and frozen source commit required')
    request = json.loads((ROOT/'spatial_calibration_v1/request.json').read_text())
    native = json.loads((ROOT/'spatial_calibration_v1/result.json').read_text())
    h = request['h']
    profiles = {p['sid']: p for p in native['profiles']}
    patches = native['patches']
    if len(patches) != 32 or len(request['choices']) != 32:
        raise ValueError('expected frozen 32-fixture population')
    OUT.mkdir(exist_ok=False)
    conventions = dict(
        status='FROZEN_BEFORE_CANDIDATE_EXTRACTION',
        source_commit=os.environ['EXPECTED_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
        dx_cMpc_h=DX, h=h, observer_cell_cMpc_h=[[12]*3, [13.5]*3],
        population='32 selected TNG patches, potentially overlapping; NOT independent universes',
        support_source='scripts/cf4_bundle_c_lg_population.py: native population upper distance supports',
        MW_M31_ceiling_physical_Mpc=3., M31_M33_ceiling_physical_Mpc=1.5,
        ceiling_margin_cMpc_h=float(np.sqrt(3)*DX), boundary_excluded_cells=2,
        matching_radius_cMpc_h=DX,
        matching='minimum-total-distance with dummy cost 1.001*radius; not forced maximum completeness',
        velocity_frame='BOX-frame peculiar km/s; NO Hubble or Solar subtraction',
        residual_order=['ln(native_bound_mass/aperture_mass)',
                        'native_catalog_center_minus_peak_x_kpc',
                        'native_catalog_center_minus_peak_y_kpc',
                        'native_catalog_center_minus_peak_z_kpc',
                        'native_member_COM_minus_aperture_vx_km_s',
                        'native_member_COM_minus_aperture_vy_km_s',
                        'native_member_COM_minus_aperture_vz_km_s'],
        primary_aperture_radius_cells=1, descriptive_aperture_radius_cells=2,
        proxy_warning='Total-matter apertures are correlated proxies, NOT bound masses or M200c',
        truth_access='Candidate catalogue is written before per-fixture truth matching',
        plan_audit='Fable5 CONDITIONAL GO: four disclosure conditions implemented in this report')
    (OUT/'conventions.json').write_text(json.dumps(conventions, indent=2)+'\n')
    records = []
    samples = {(split, role): {} for split in (0, 1) for role in ROLES}
    occurrences = {(split, role): 0 for split in (0, 1) for role in ROLES}
    plotted_splits = set()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    with h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as source, \
            h5py.File(OUT/'candidates.h5', 'x') as catalog:
        if source.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE':
            raise ValueError('native source status mismatch')
        if not np.isclose(source.attrs['h'], h) or source['fine'].shape != (7, 400, 400, 400):
            raise ValueError('native source cosmology/shape mismatch')
        catalog.attrs['status'] = 'FIELD_ONLY_CANDIDATES_NOT_IDENTIFIED_HALOS'
        catalog.attrs['source_commit'] = os.environ['EXPECTED_COMMIT']
        for row, patch in enumerate(patches):
            choice = request['choices'][row]
            if patch['ids'] != choice['ids'] or patch['source_row'] != choice['source_row']:
                raise ValueError('frozen fixture ordering mismatch')
            lower = np.array(patch['patch_lower_cMpc_h'])
            start = np.rint(lower/DX).astype(int) % 400
            field = read_periodic_patch(source['fine'], start, 128)
            found = identify(field, DX, conventions['observer_cell_cMpc_h'], h=h)
            group = catalog.create_group(str(row))
            for key in ('cells', 'positions', 'moments', 'mean_velocity', 'physical_sigma',
                        'mw_candidates', 'mw_m31_edges', 'm31_m33_edges'):
                group.create_dataset(key, data=found[key])
            group.attrs['triplet_count'] = found['triplet_count']
            catalog.flush()
            # Native identities enter the evaluation only AFTER frozen extraction.
            truth = [profiles[sid] for sid in patch['ids']]
            positions = np.array([(np.array(p['position_ckpc_h'])/1000-lower) % 75 for p in truth])
            if np.any((positions < 0) | (positions >= 24)):
                raise ValueError('truth outside source patch')
            match = match_for_evaluation(found['positions'], positions, DX)
            all_match = match_for_evaluation(found['all_interior_peak_positions'], positions, DX)
            matched = match['matched']
            pair_set = set(map(tuple, found['mw_m31_edges']))
            third_set = set(map(tuple, found['m31_m33_edges']))
            eligible = (all(i >= 0 for i in matched) and matched[0] in found['mw_candidates']
                        and tuple(matched[:2]) in pair_set and tuple(matched[1:]) in third_set)
            boundary = np.any((positions < 2*DX) | (positions >= 126*DX), axis=1)
            split = int(patch['heldout'])
            record = dict(fixture=row, source_row=patch['source_row'], heldout=split, ids=patch['ids'],
                selected_candidates=len(found['cells']), mw_candidates=len(found['mw_candidates']),
                triplet_count=found['triplet_count'], raw_peak_count=found['raw_peak_count'],
                boundary_excluded_peak_count=found['boundary_excluded_peak_count'],
                tied_peak_group_count=found['tied_peak_group_count'],
                truth_in_excluded_shell=boundary.tolist(), truth_in_excluded_shell_count=int(boundary.sum()),
                match=match, all_interior_peaks_match=all_match,
                all_roles_matched=all(i >= 0 for i in matched), matched_triplet_role_eligible=bool(eligible),
                aperture_overlap={str(r): overlap(found['cells'], matched, r) for r in (1, 2)},
                residuals={})
            for r, (role, index) in enumerate(zip(ROLES, matched)):
                if index < 0:
                    continue
                p = truth[r]
                residual = np.r_[np.log(p['mass_Msun']/found['moments'][0, 0, index]),
                    (positions[r]-found['positions'][index])*1000/h,
                    np.array(p['COM_velocity_km_s'])-found['mean_velocity'][0, :, index]]
                record['residuals'][role] = residual.tolist()
                occurrences[split, role] += 1
                samples[split, role].setdefault(p['sid'], residual)
            records.append(record)
            if split not in plotted_splits:
                plotted_splits.add(split)
                ax = axes[split]
                # Fixed observer slab and full xy footprint: no best-case selection.
                slab = field[0, :, :, 64:72].sum(axis=2)
                ax.imshow(np.log10(np.maximum(slab, 1)).T, origin='lower', extent=(0, 24, 0, 24))
                points = found['positions']
                in_slab = (points[:, 2] >= 12) & (points[:, 2] < 13.5)
                ax.scatter(points[in_slab, 0], points[in_slab, 1], s=12, facecolors='none',
                           edgecolors='white', label='field-only peaks in observer z slab')
                for role, pos in zip(ROLES, positions):
                    ax.scatter(*pos[:2], marker='x', color='red')
                    ax.annotate(role, pos[:2], color='red', fontsize=8)
                ax.set_title(f'First {"heldout" if split else "train"} fixture; truth overlaid AFTER detection')
                ax.set_xlabel('x [cMpc/h]'); ax.set_ylabel('y [cMpc/h]')
            print(json.dumps({k: record[k] for k in ('fixture', 'heldout', 'selected_candidates',
                  'triplet_count', 'all_roles_matched', 'matched_triplet_role_eligible')}), flush=True)
    calibration = {}
    for role in ROLES:
        train = np.array(list(samples[0, role].values())).reshape(-1, 7)
        held = samples[1, role]
        if set(samples[0, role]) & set(held):
            raise ValueError('train-heldout identity leakage within role')
        mean = train.mean(0) if len(train) else None
        cov = np.cov(train, rowvar=False) if len(train) >= 8 else None
        calibration[role] = dict(train_unique=len(train), heldout_unique=len(held),
            train_occurrences=occurrences[0, role], heldout_occurrences=occurrences[1, role],
            train_mean=None if mean is None else mean.tolist(),
            train_covariance=None if cov is None else cov.tolist(),
            covariance_rank=None if cov is None else int(np.linalg.matrix_rank(cov)),
            limitation='Matched-only, first occurrence per ID/role/split; no cross-role covariance or precision',
            heldout_centered_residuals={str(sid): (v-mean).tolist() for sid, v in held.items()} if mean is not None else {})
    summary = {}
    for split in (0, 1):
        subset = [r for r in records if r['heldout'] == split]
        summary['heldout' if split else 'train'] = dict(fixtures=len(subset),
            role_matched_counts={role: sum(r['match']['matched'][i] >= 0 for r in subset) for i, role in enumerate(ROLES)},
            all_roles_matched=sum(r['all_roles_matched'] for r in subset),
            matched_role_eligible_triplets=sum(r['matched_triplet_role_eligible'] for r in subset),
            shared_nearest_peak_fixtures=sum(bool(r['match']['shared_nearest_pairs']) for r in subset),
            truth_boundary_excluded=sum(r['truth_in_excluded_shell_count'] for r in subset))
    result = dict(status='NATIVE_FIELD_IDENTIFICATION_DIAGNOSTIC_COMPLETE_NOT_OBSERVED_LG_POSTERIOR',
        conventions=conventions, summary=summary, calibration=calibration, fixtures=records,
        next_scope='No automatic learning or likelihood fit; report assignment/missingness before next plan')
    (OUT/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    fig.tight_layout()
    fig.savefig(OUT/'first_fixture_maps.png', dpi=140)
    plt.close(fig)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
