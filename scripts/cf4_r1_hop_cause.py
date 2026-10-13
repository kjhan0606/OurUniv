"""Attribute archived host differences to membership and fixed-member motion."""
import json
import os
from pathlib import Path
import numpy as np
from cf4_zoom_z0_gate import _record, _skip_header
from cf4_r1_particle_entry import write

SOURCE = Path('/gpfs/kjhan/CF4/r1_hop/job_361459')


def read_state(label):
    with (SOURCE/label/'grp.tag').open('rb') as f:
        count, groups = _record(f, '<i4')
        tag = _record(f, '<i4')
    with (SOURCE/label/'particles00001').open('rb') as f:
        _, n = _skip_header(f)
        x = np.stack([_record(f, '<f8') for _ in range(3)], axis=1)*12
        v = np.stack([_record(f, '<f8') for _ in range(3)], axis=1)
        mass = _record(f, '<f8')
    with (SOURCE/label/'hop.den').open('rb') as f:
        nd = np.fromfile(f, '<i4', 1)[0]
        density = np.fromfile(f, '<f4')
    assert n == count == nd == len(tag) == len(density)
    assert np.all((tag >= -1) & (tag < groups))
    np.testing.assert_array_equal(mass, np.full(n, 1/n))
    return tag, x, v, density


def decompose(aA, bA, aB, bB):
    dynamics = .5*((aA-bA)+(aB-bB))
    membership = .5*((aA-aB)+(bA-bB))
    np.testing.assert_allclose(dynamics+membership, aA-bB, atol=1e-10)
    return dynamics, membership


def summary(rows):
    pairs = [r for r in rows if r.get('baseline_reported_pair')]
    unstable = [r for r in pairs if abs(r['relative_mass_difference']) > .2]
    def stats(items):
        return dict(n=len(items),
            median_abs_mass_difference=float(np.median([abs(r['relative_mass_difference']) for r in items])) if items else None,
            exchange_with_ungrouped=sum(r['lost_ungrouped']+r['gained_ungrouped'] for r in items),
            exchange_with_other_groups=sum(r['lost_other_groups']+r['gained_other_groups'] for r in items),
            density_crossing_exchanges=sum(r['density_crossing_exchanges'] for r in items),
            topology_flag_count=sum(r['topology_flag'] for r in items),
            median_velocity_dynamics=float(np.median([r['velocity_dynamics_norm'] for r in items])) if items else None,
            median_velocity_membership=float(np.median([r['velocity_membership_norm'] for r in items])) if items else None)
    return dict(all_sources=len(rows), unmatched=sum('target_group' not in r for r in rows),
        paired=stats(pairs), unstable=stats(unstable),
        by_source_particle_count={f'{lo}-{hi}': dict(all=stats([r for r in pairs if lo <= r['source_count'] < hi]),
            unstable=stats([r for r in unstable if lo <= r['source_count'] < hi]))
            for lo, hi in [(100,300),(300,1000),(1000,10000),(10000,10000000)]})


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    # Exact attribution regression: pure motion and pure membership controls.
    np.testing.assert_allclose(decompose(np.array([3.]), np.array([1.]), np.array([3.]), np.array([1.]))[1], 0)
    np.testing.assert_allclose(decompose(np.array([3.]), np.array([3.]), np.array([1.]), np.array([1.]))[0], 0)
    old = json.loads((SOURCE/'result.json').read_text())
    assert old['status'] == 'COMPLETE_DRIVER_JUDGMENT_REQUIRED'
    out = Path('/gpfs/kjhan/CF4/r1_hop_cause')/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', source_commit=os.environ['EXPECTED_COMMIT'], comparisons={},
        limits='One interpolated-IC seed. Attribution is between membership and dynamics, not a proof of unique integrator/force error. HOP masses are not M200c; MW/M31/M33 and bound M33 unidentified.')
    ta, xa, va, da = read_state('amr9')
    try:
        for label in ['cic', 'tsc', 'amr8']:
            tb, xb, vb, db = read_state(label)
            rows = []
            for oldrow in old['matches']['amr9-'+label]:
                g = oldrow['source_group']
                A = ta == g
                row = dict(source_group=g, source_count=int(A.sum()),
                           baseline_reported_pair=bool(oldrow['reciprocal'] and 'relative_mass_difference' in oldrow))
                if not oldrow['choices']:
                    rows.append(row)
                    continue
                h = oldrow['choices'][0]['group_id']
                B = tb == h
                shared, lost, gained = A & B, A & ~B, B & ~A
                na, nb = int(A.sum()), int(B.sum())
                assert na-nb == int(lost.sum())-int(gained.sum())
                lu, lo = int(np.sum(lost & (tb < 0))), int(np.sum(lost & (tb >= 0)))
                gu, go = int(np.sum(gained & (ta < 0))), int(np.sum(gained & (ta >= 0)))
                assert na-nb == lu+lo-gu-go
                # Decompose exact vector difference using both fixed member sets.
                dynamics, membership = decompose(va[A].mean(0), vb[A].mean(0), va[B].mean(0), vb[B].mean(0))
                ids_a, counts_a = np.unique(tb[A & (tb >= 0)], return_counts=True)
                ids_b, counts_b = np.unique(ta[B & (ta >= 0)], return_counts=True)
                secondary_a = int(np.max(counts_a[ids_a != h], initial=0))
                secondary_b = int(np.max(counts_b[ids_b != g], initial=0))
                # Flag is descriptive, not a pass/fail cut or a halo identification.
                row.update(target_group=h, target_count=nb, shared=int(shared.sum()),
                    reciprocal=oldrow['reciprocal'], relative_mass_difference=na/nb-1,
                    lost_ungrouped=lu, lost_other_groups=lo, gained_ungrouped=gu, gained_other_groups=go,
                    net_ungrouped_mass_counts=lu-gu, net_other_group_mass_counts=lo-go,
                    density_crossing_exchanges=int(np.sum(lost & (da>=80) & (db<80))+np.sum(gained & (db>=80) & (da<80))),
                    secondary_source_fraction=secondary_a/na, secondary_target_fraction=secondary_b/nb,
                    topology_flag=bool(secondary_a/na>=.2 or secondary_b/nb>=.2 or not oldrow['reciprocal']),
                    velocity_dynamics_vector=dynamics.tolist(), velocity_membership_vector=membership.tolist(),
                    velocity_dynamics_norm=float(np.linalg.norm(dynamics)), velocity_membership_norm=float(np.linalg.norm(membership)),
                    fixed_source_particle_velocity_RMS=float(np.sqrt(np.mean(np.sum((va[A]-vb[A])**2, axis=1)))),
                    fixed_source_sigma_a=np.std(va[A], axis=0).tolist(), fixed_source_sigma_b=np.std(vb[A], axis=0).tolist())
                if row['baseline_reported_pair']:
                    np.testing.assert_allclose(row['relative_mass_difference'], oldrow['relative_mass_difference'], atol=1e-12)
                rows.append(row)
            report['comparisons'][label] = dict(summary=summary(rows), rows=rows)
            write(out/'result.json', report)
            print(label+' '+json.dumps(report['comparisons'][label]['summary']), flush=True)
            del tb, xb, vb, db
        report['status'] = 'COMPLETE_DRIVER_JUDGMENT_REQUIRED'
    except Exception as exc:
        report.update(status='FAILED', error=str(exc))
        raise
    finally:
        write(out/'result.json', report)


if __name__ == '__main__':
    main()
