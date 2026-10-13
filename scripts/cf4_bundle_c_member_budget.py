"""One cached native/mock union-budget report; no fitting or actual LG scoring."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from cf4_member_budget import support_keys, evaluate, ROLES, PASS, FAIL, UNAVAILABLE, TOL

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT/'member_budget_v1'
CASES = [16, 17, 20]
DX, N = .1875, 128


def mock_records(row, radius, metadata, sparse):
    """Native memberships used ONLY here to construct explicitly supplied mock data."""
    patch = metadata['patches'][row]
    origin = np.rint(np.array(patch['patch_lower_cMpc_h'])/DX).astype(int)
    members, annotations, supports = {}, {}, {}
    for role, sid in zip(ROLES, patch['ids']):
        group = sparse[f'halos/{sid}']
        profile = json.loads(group.attrs['profile_json'])
        position = (np.array(profile['position_ckpc_h'])/1000-origin*DX) % 75
        keys = support_keys(position, radius, dx=DX, n=N)
        supports[role] = keys
        cells = (group['global_cell'][:]-origin) % 400
        inside = np.all((cells >= 0) & (cells < N), axis=1)
        member_keys = np.ravel_multi_index(cells[inside].T, (N,)*3)
        moments = group['moments'][:, inside]
        keep = np.zeros(len(member_keys), bool) if keys is None else np.isin(member_keys, keys)
        total = moments[:, keep].sum(1)
        mass = float(total[0])
        velocity = total[1:4]/mass if mass > 0 else np.zeros(3)
        sigma = np.sqrt(np.maximum(total[4:7]/mass-velocity**2, 0)) if mass > 0 else np.zeros(3)
        members[role] = dict(position_cMpc_h=position.tolist(), radius_cMpc_h=radius,
            mass_Msun=mass, velocity_km_s=velocity.tolist())
        annotations[role] = dict(source_id_for_mock_construction_only=sid,
            retained_fraction_of_whole_bound_mass=mass/profile['mass_Msun'],
            support_cell_count=None if keys is None else len(keys),
            selected_member_cell_count=int(keep.sum()), physical_member_sigma_km_s=sigma.tolist(),
            dispersion_use='Reported only; demanded Q uses bulk lower envelope, not zero physical sigma')
    return members, annotations, supports


def field_cells(dataset, keys, origin):
    """Read small cached total-field selections, never alternative-field halo labels."""
    local = np.array(np.unravel_index(keys, (N,)*3)).T
    xyz = (local+origin) % 400
    values = np.empty((7, len(keys)))
    for xy in np.unique(xyz[:, :2], axis=0):
        indices = np.flatnonzero(np.all(xyz[:, :2] == xy, axis=1))
        order = np.argsort(xyz[indices, 2])
        indices = indices[order]
        values[:, indices] = dataset[:, int(xy[0]), int(xy[1]), xyz[indices, 2]]
    return values


def conditional_summary(records, mode):
    alternatives = [r for r in records if r['target_fixture'] != r['field_fixture']]
    valid = [r for r in alternatives if r['summary']['hosts_'+mode] != UNAVAILABLE
             and r['summary']['all_three_'+mode] != UNAVAILABLE]
    survivors = [r for r in valid if r['summary']['hosts_'+mode] == PASS]
    extra = [r for r in survivors if r['summary']['all_three_'+mode] == FAIL]
    status = ('INCONCLUSIVE_NO_HOST_COMPATIBLE_ALTERNATIVES' if not survivors else
              'M33_ADDS_NECESSARY_EXCLUSIONS' if extra else 'NO_ADDED_EXCLUSION_IN_AVAILABLE_CONTEXTS')
    return dict(status=status, alternative_contexts=len(alternatives), available_alternatives=len(valid),
        host_compatible_alternatives=len(survivors), added_M33_exclusions=len(extra),
        added_exclusion_cases=[[r['target_fixture'], r['field_fixture']] for r in extra])


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('EXPECTED_COMMIT'):
        raise RuntimeError('Slurm and frozen source commit required')
    metadata = json.loads((ROOT/'spatial_calibration_v1/result.json').read_text())
    split = json.loads((ROOT/'spatial_conditional_v1/result.json').read_text())
    if split['heldout_patches'] != CASES or split['heldout_mutual_voxel_overlap']:
        raise ValueError('fixed geometry-disjoint source cases required')
    OUT.mkdir(exist_ok=False)
    report = dict(status='STARTED_NOT_COMPLETE', source_commit=os.environ['EXPECTED_COMMIT'],
        job_id=os.environ['SLURM_JOB_ID'], cases=CASES, dx_cMpc_h=DX, tolerance=TOL,
        mass_unit='physical Msun', velocity_frame='BOX peculiar km/s; common supplied MW COM centering',
        radii={}, limits=[
            'Noiseless mock supplied positions, enclosed member masses and absolute BOX COMs; stronger than real observations.',
            'No inference inputs are native IDs or member masks; same target positions used in each alternative field.',
            'Own-field agreement is a construction control, not prediction or astronomical validation.',
            'Necessary conditions only: no spatial allocation, binding, likelihood, posterior or phase recovery.',
            'rQ includes free member internal dispersion; it is not a unique zero-mass remainder.',
            'Unavailable contexts are not negative evidence; no host survivors means inconclusive M33 increment.',
            'No actual mass prior, q_F repair, population selection law, new learner or simulation.'])

    def save():
        (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    save()
    read_cells = 0
    with h5py.File(ROOT/'spatial_calibration_v1/spatial_components.h5', 'r') as sparse, \
            h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as total:
        if (sparse.attrs['status'] != 'NATIVE_JOINT_PROFILES_AND_REMAINDER_NOT_CF4_LG_POSTERIOR'
                or total.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE'
                or not np.isclose(sparse.attrs['h'], total.attrs['h'])
                or not np.isclose(sparse.attrs['fine_dx_cMpc_h'], DX)):
            raise ValueError('cached moment source convention mismatch')
        report['h'] = float(total.attrs['h'])
        for radius_cells in (1, 2):
            result = dict(mock_inputs={}, comparisons=[])
            for row in CASES:
                members, annotations, supports = mock_records(row, radius_cells*DX, metadata, sparse)
                result['mock_inputs'][str(row)] = dict(members=members, annotations=annotations)
                valid_keys = [keys for keys in supports.values() if keys is not None]
                keys = np.unique(np.concatenate(valid_keys)) if valid_keys else np.array([], dtype=int)
                for field in CASES:
                    # Only patch origin is used, not alternative-field member positions or identities.
                    origin = np.rint(np.array(metadata['patches'][field]['patch_lower_cMpc_h'])/DX).astype(int)
                    values = field_cells(total['fine'], keys, origin)
                    read_cells += len(keys)
                    evaluation = evaluate(keys, values, members, dx=DX, n=N)
                    result['comparisons'].append(dict(target_fixture=row, field_fixture=field, **evaluation))
            result['mass_summary'] = conditional_summary(result['comparisons'], 'mass')
            result['moment_summary'] = conditional_summary(result['comparisons'], 'moment')
            report['radii'][str(radius_cells)] = result
            save()
            print(json.dumps(dict(radius_cells=radius_cells, mass=result['mass_summary'], moment=result['moment_summary'])), flush=True)
    own_failures = [dict(radius=r, target=c['target_fixture'], summary=c['summary'])
        for r, value in report['radii'].items() for c in value['comparisons']
        if c['target_fixture'] == c['field_fixture'] and FAIL in c['summary'].values()]
    report['own_field_control_failures'] = own_failures
    report['cached_cells_read_including_repeated_contexts'] = read_cells
    report['status'] = 'NATIVE_SELF_CONTROL_FAILED' if own_failures else 'NATIVE_MEMBER_BUDGET_DIAGNOSTIC_COMPLETE_NOT_LG_POSTERIOR'
    cmap = ListedColormap(['#d95f59', '#c5c5c5', '#5b9bc5'])
    norm = BoundaryNorm([-1.5, -.5, .5, 1.5], cmap.N)
    numbers, labels = {FAIL: -1, UNAVAILABLE: 0, PASS: 1}, {FAIL: 'excluded', UNAVAILABLE: 'unavailable', PASS: 'not excluded'}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, (radius, result) in enumerate(report['radii'].items()):
        for j, mode in enumerate(('hosts_mass', 'all_three_mass', 'hosts_moment', 'all_three_moment')):
            states = [c['summary'][mode] for c in result['comparisons']]
            matrix = np.array([numbers[s] for s in states]).reshape(3, 3)
            ax = axes[i, j]
            ax.imshow(matrix, cmap=cmap, norm=norm)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, labels[states[3*a+b]], ha='center', fontsize=8)
            ax.set_xticks(range(3), CASES); ax.set_yticks(range(3), CASES)
            ax.set_xlabel('Alternative field'); ax.set_ylabel('Fixed mock target')
            ax.set_title(f'R={radius}dx; {mode}')
    fig.suptitle('Necessary budget tests ONLY; not excluded does not mean a reconstructed halo')
    fig.tight_layout(); fig.savefig(OUT/'member_budget_tables.png', dpi=140); plt.close(fig)
    save()
    if own_failures:
        raise RuntimeError('native positive control failed; preserve results and inspect implementation')


if __name__ == '__main__':
    main()
