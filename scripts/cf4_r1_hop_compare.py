"""Common HOP host readout and particle-overlap matching; no LG labeling."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from grafic_io import _write_record
from cf4_zoom_z0_gate import _record, _skip_header
from cf4_r1_ramses_reference import load_snapshot
from cf4_r1_particle_entry import ROOT, write


def export_hop(path, x, v):
    # Minimal DMO particle stream for ReadRamses, NOT a restart snapshot.
    n = len(x)
    with path.open('xb') as f:
        for value, dtype in [([1], '<i4'), ([3], '<i4'), ([n], '<i4'),
                             ([0]*4, '<i4'), ([0], '<i8'), ([0], '<f8'),
                             ([0], '<f8'), ([0], '<i4')]:
            _write_record(f, np.asarray(value, dtype=dtype).tobytes())
        for array in [*(x/12.).T, *v.T, np.full(n, 1/n)]:
            _write_record(f, np.asarray(array, dtype='<f8').tobytes())
    with path.open('rb') as f:
        assert _skip_header(f) == (1, n)
        for axis in range(3):
            np.testing.assert_array_equal(_record(f, '<f8'), x[:, axis]/12.)
        for axis in range(3):
            np.testing.assert_array_equal(_record(f, '<f8'), v[:, axis])
        np.testing.assert_array_equal(_record(f, '<f8'), np.full(n, 1/n))


def catalog(tags, x, v, mp, groups):
    counts = np.bincount(tags[tags >= 0], minlength=groups)
    rows = []
    for g in np.flatnonzero(counts >= 100):
        keep = tags == g
        angle = x[keep]*(2*np.pi/12.)
        center = np.mod(np.arctan2(np.sin(angle).mean(0), np.cos(angle).mean(0)), 2*np.pi)*12/(2*np.pi)
        rows.append(dict(group_id=int(g), count=int(counts[g]), mass_Msun_h=float(counts[g]*mp),
                         center_cMpc_h=center.tolist(), velocity_km_s=v[keep].mean(0).tolist()))
    return rows


def match(left, right, a, b):
    # Both particle arrays use the same initial-ID order. Match by overlap,
    # not a truth halo catalogue or chosen LG analogue.
    rows = []
    for source in a:
        gid = source['group_id']
        other = right[left == gid]
        ids, counts = np.unique(other[other >= 0], return_counts=True)
        order = np.argsort(-counts)
        choices = [dict(group_id=int(ids[k]), shared=int(counts[k])) for k in order[:2]]
        row = dict(source_group=gid, source_count=source['count'], choices=choices, reciprocal=False)
        if choices:
            target_id = choices[0]['group_id']
            backward = left[right == target_id]
            bi, bc = np.unique(backward[backward >= 0], return_counts=True)
            row['reciprocal'] = bool(len(bi) and bi[np.argmax(bc)] == gid)
            row['source_shared_fraction'] = choices[0]['shared']/source['count']
            row['target_shared_fraction'] = choices[0]['shared']/int(np.sum(right == target_id))
            target = next((item for item in b if item['group_id'] == target_id), None)
            if target:
                row['relative_mass_difference'] = source['mass_Msun_h']/target['mass_Msun_h']-1
                dx = (np.asarray(source['center_cMpc_h'])-target['center_cMpc_h']+6)%12-6
                row['center_distance_cMpc_h'] = float(np.linalg.norm(dx))
                row['velocity_difference_km_s'] = float(np.linalg.norm(np.asarray(source['velocity_km_s'])-target['velocity_km_s']))
        rows.append(row)
    return rows


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    cfg = json.loads((ROOT/'config/cf4_r1_ramses_reference_v7.json').read_text())
    # Small matching regression: same membership despite permuted group labels.
    test_a = np.asarray([0, 0, 1, 1, -1])
    test_b = np.asarray([1, 1, 0, 0, -1])
    matched = match(test_a, test_b, [dict(group_id=0, count=2)], [])
    assert matched[0]['reciprocal'] and matched[0]['choices'][0]['group_id'] == 1
    assert matched[0]['source_shared_fraction'] == matched[0]['target_shared_fraction'] == 1.
    out = Path('/gpfs/kjhan/CF4/r1_hop')/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    bins = Path('/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses')
    report = dict(status='RUNNING', source_commit=os.environ['EXPECTED_COMMIT'], catalogs={}, matches={},
        binary_sha256={s: hashlib.sha256((bins/s).read_bytes()).hexdigest() for s in ['hop', 'regroup']},
        limits='HOP hosts with >=100 particles, not M200c or bound subhalos; no MW/M31/M33 identification. Shared IDs only match generated structures across solvers.',
        observed_posterior=False, resolved_LG=False, R1_complete=False)
    tags = {}
    try:
        for label in ['cic', 'tsc', 'amr8', 'amr9']:
            work = out/label
            work.mkdir()
            with np.load(Path(cfg['reference_directory'])/f"reference_particles_seed{cfg['seed']}.npz") as data:
                mp = float(data['particle_mass_Msun_h'])
                if label == 'cic':
                    x, v = data['position_cMpc_h'], data['velocity_km_s']
            if label == 'tsc':
                with np.load(Path(cfg['tsc_directory'])/f"tsc_final_seed{cfg['seed']}.npz") as data:
                    x, v = data['position_cMpc_h'], data['velocity_km_s']
            if label.startswith('amr'):
                x, v, info = load_snapshot(Path('/gpfs/kjhan/CF4/r1_ref_v7/job_360338')/label/'output_00002', 12., 128)
                np.testing.assert_allclose(info['aexp'], 1., atol=2e-6, rtol=0)
            export_hop(work/'particles00001', x, v)
            commands = [([str(bins/'hop'), '-in', 'particles', '-p', '1.', '-nd', '64', '-nh', '64', '-nm', '4', '-o', 'hop'], 'hop.log'),
                        ([str(bins/'regroup'), '-root', 'hop', '-douter', '80.', '-dsaddle', '200.', '-dpeak', '240.', '-f77', '-o', 'grp'], 'regroup.log')]
            for command, logfile in commands:
                with (work/logfile).open('x') as f:
                    subprocess.run(command, cwd=work, stdout=f, stderr=subprocess.STDOUT, check=True,
                                   timeout=1200 if logfile == 'hop.log' else 60)
            with (work/'grp.tag').open('rb') as f:
                count, groups = _record(f, '<i4')
                tag = _record(f, '<i4')
            if count != len(x) or len(tag) != count or np.any(tag < -1) or np.any(tag >= groups):
                raise RuntimeError('invalid HOP tag count/range')
            tags[label] = tag
            report['catalogs'][label] = catalog(tag, x, v, mp, int(groups))
            write(out/'result.json', report)
            print(label+': '+str(len(report['catalogs'][label]))+' hosts >=100 particles', flush=True)
            del x, v
        for other in ['cic', 'tsc', 'amr8']:
            report['matches']['amr9-'+other] = match(tags['amr9'], tags[other], report['catalogs']['amr9'], report['catalogs'][other])
        report['status'] = 'COMPLETE_DRIVER_JUDGMENT_REQUIRED'
    except Exception as exc:
        report.update(status='FAILED', error=str(exc))
        raise
    finally:
        write(out/'result.json', report)


if __name__ == '__main__':
    main()
