"""Native joint profiles/remainder calibration for the C-spatial bundle.

build: Slurm numerical work. serve: syntax I/O only for explicitly requested
member ranges. No observations scored, halo painting or new field prior yet.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import h5py
import numpy as np

from cf4_bundle_c_tng_operator import load_catalog
from cf4_bundle_c_tng_stage import RAW
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_continuous_matter import ContinuousMatter, restrict

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT / 'spatial_calibration_v1'


def request_members():
    rng = np.random.default_rng(83019)
    with h5py.File(ROOT / 'lg_population_v1/population_and_marks.h5', 'r') as f:
        ids, split, branch = f['identities'][:], f['heldout'][:], f['branch'][:]
    choices = []
    for subset in (0, 1):
        available = np.flatnonzero((split == subset) & (branch == 0))
        for mw in rng.choice(np.unique(ids[available, 0]), size=16, replace=False):
            rows = available[ids[available, 0] == mw]
            m31 = rng.choice(np.unique(ids[rows, 1]))
            row = int(rng.choice(rows[ids[rows, 1] == m31]))
            choices.append(dict(source_row=row, heldout=subset, ids=ids[row].tolist()))
    train_ids = {i for c in choices if c['heldout'] == 0 for i in c['ids']}
    held_ids = {i for c in choices if c['heldout'] == 1 for i in c['ids']}
    if train_ids & held_ids:
        raise ValueError('native training/heldout object overlap')
    unique = sorted(train_ids | held_ids)
    cat, header, snapshot, counts = load_catalog()
    g, s = cat['Group'], cat['Subhalo']
    objects = {sid: dict(sid=sid, group_id=int(s['SubhaloGrNr'][sid]),
        position_ckpc_h=s['SubhaloPos'][sid].tolist(), velocity_km_s=s['SubhaloVel'][sid].tolist(),
        bound_mass_native=float(s['SubhaloMass'][sid]), types=[]) for sid in unique}
    for sid, obj in objects.items():
        group = obj['group_id']
        obj['primary'] = bool(g['GroupFirstSub'][group] == sid)
        obj['host_M200c_native'] = float(g['Group_M_Crit200'][group]) if obj['primary'] else None
    for p in (0, 1, 4, 5):
        gp = np.r_[0, np.cumsum(g['GroupLenType'][:, p], dtype=np.int64)]
        sp = np.r_[0, np.cumsum(s['SubhaloLenType'][:, p], dtype=np.int64)]
        fp = np.r_[0, np.cumsum(counts[:, p], dtype=np.int64)]
        for sid, obj in objects.items():
            length = int(s['SubhaloLenType'][sid, p])
            if not length:
                continue
            group = obj['group_id']
            start = int(gp[group]+sp[sid]-sp[g['GroupFirstSub'][group]])
            pieces = []
            for i in range(448):
                lo, hi = max(start, int(fp[i])), min(start+length, int(fp[i+1]))
                if hi > lo:
                    pieces.append(dict(file=i, start=lo-int(fp[i]), stop=hi-int(fp[i])))
            if sum(part['stop']-part['start'] for part in pieces) != length:
                raise ValueError('incomplete native member range')
            obj['types'].append(dict(type=p, count=length, pieces=pieces))
    for choice in choices:
        mw, m31, m33 = [objects[sid] for sid in choice['ids']]
        if not (mw['primary'] and m31['primary'] and not m33['primary'] and
                mw['group_id'] != m31['group_id'] == m33['group_id']):
            raise ValueError('two-primary + M31-satellite source identity mismatch')
    total_rows = sum(t['count'] for obj in objects.values() for t in obj['types'])
    if total_rows > 50000000:
        raise ValueError('selected native read exceeds50m row budget; do not expand silently')
    return dict(status='EXPLICIT_NATIVE_MEMBER_REQUEST_NOT_LG_POSTERIOR_SELECTION', choices=choices,
        objects=list(objects.values()), total_rows=total_rows, h=float(header['HubbleParam']),
        a=float(header['Time']), DM_mass_native=float(snapshot['MassTable'][1]),
        selection='16 uniform distinct MW-role observers per native split, then uniform M31 and M33. Companions may recur; unique subhalos streamed once.')


def serve(path):
    request = json.loads(Path(path).read_text())
    stream = sys.stdout.buffer
    for obj in request['objects']:
        pickle.dump(('object', obj['sid']), stream, protocol=5)
        for item in obj['types']:
            p = item['type']
            for piece in item['pieces']:
                with h5py.File(RAW / f"snapdir_099/snap_099.{piece['file']}.hdf5", 'r') as f:
                    g = f[f'PartType{p}']
                    for lo in range(piece['start'], piece['stop'], 131072):
                        hi = min(lo+131072, piece['stop'])
                        pickle.dump(('rows', p, g['Coordinates'][lo:hi], g['Velocities'][lo:hi],
                            None if p == 1 else g['Masses'][lo:hi]), stream, protocol=5)
        pickle.dump(('end_object', obj['sid']), stream, protocol=5)
        stream.flush()
    pickle.dump(('end',), stream, protocol=5)
    stream.flush()


def measure_object(obj, stream, request):
    if pickle.load(stream) != ('object', obj['sid']):
        raise ValueError('unexpected object in source stream')
    h, a = request['h'], request['a']
    positions, velocities, masses = [], [], []
    counts = np.zeros(6, dtype=np.int64)
    while True:
        item = pickle.load(stream)
        if item[0] == 'end_object':
            if item[1] != obj['sid']:
                raise ValueError('source identity mismatch')
            break
        _, p, x, v, m = item
        counts[p] += len(x)
        positions.append(x.astype(float)/1000)
        velocities.append(v.astype(float)*np.sqrt(a))
        masses.append((np.full(len(x), request['DM_mass_native']) if p == 1 else m.astype(float))*1e10/h)
    expected = np.zeros(6, dtype=np.int64)
    for item in obj['types']:
        expected[item['type']] = item['count']
    np.testing.assert_array_equal(counts, expected)
    x, v, m = map(np.concatenate, (positions, velocities, masses))
    if not all(np.isfinite(y).all() for y in (x, v, m)) or np.any(m < 0) or m.sum() <= 0:
        raise ValueError('invalid native massive members')
    positive = m > 0
    x, v, m = x[positive], v[positive], m[positive]
    # Global NGP keys are identical to the full400^3 native total-matter source.
    cell = np.floor((x % 75)/.1875).astype(int)
    key, inverse = np.unique(np.ravel_multi_index(cell.T, (400,)*3), return_inverse=True)
    value = np.array([np.bincount(inverse, weights=w, minlength=len(key)) for w in
        [m, *(m*v[:, i] for i in range(3)), *(m*v[:, i]**2 for i in range(3))]])
    totals = value.sum(axis=1)
    mass_error = float(totals[0]/(obj['bound_mass_native']*1e10/h)-1)
    com = totals[1:4]/totals[0]
    com_error = float(np.max(abs(com-obj['velocity_km_s'])))
    if abs(mass_error) > 2e-4 or com_error > .2:
        raise ValueError('native particle mass/COM does not match source catalogue')
    dr = ((x-np.array(obj['position_ckpc_h'])/1000+37.5) % 75-37.5)*1000/h
    radius = np.linalg.norm(dr, axis=1)
    order = np.argsort(radius)
    fraction = np.cumsum(m[order])/m.sum()
    probabilities = np.array([.01, .05, .1, .25, .5, .75, .9, .95, .99, 1.])
    radii = np.interp(probabilities, fraction, radius[order])
    dv = v-com
    covariance = (dv*m[:, None]).T @ dv/m.sum()
    report = dict(sid=obj['sid'], group_id=obj['group_id'], primary=obj['primary'],
        mass_Msun=float(m.sum()), host_M200c_Msun=None if obj['host_M200c_native'] is None else obj['host_M200c_native']*1e10/h,
        position_ckpc_h=obj['position_ckpc_h'], COM_velocity_km_s=com.tolist(), counts=counts.tolist(),
        mass_relative_error=mass_error, COM_max_error_km_s=com_error,
        cumulative_mass_fractions=probabilities.tolist(), cumulative_radius_kpc=radii.tolist(),
        physical_velocity_covariance_km2_s2=covariance.tolist())
    return dict(cell=np.array(np.unravel_index(key, (400,)*3)).T, moments=value, report=report)


def build():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('numerical calibration requires Slurm')
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    request = request_members()
    path = OUT / 'request.json'
    with path.open('x') as f:
        json.dump(request, f, indent=2)
    print(f"Request {len(request['objects'])} distinct halos, {request['total_rows']} massive rows", flush=True)
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'syntax', 'env',
        f'PYTHONPATH={REPO / "src"}:{REPO / "scripts"}', '/home/kjhan/miniconda3/bin/python3.13',
        str(Path(__file__).resolve()), 'serve', '--request', str(path)]
    source = subprocess.Popen(command, stdout=subprocess.PIPE)
    components = {}
    try:
        for obj in request['objects']:
            components[obj['sid']] = measure_object(obj, source.stdout, request)
        if pickle.load(source.stdout) != ('end',) or source.wait(timeout=30) != 0:
            raise ValueError('native source did not finish correctly')
    finally:
        if source.poll() is None:
            source.terminate()
            try:
                source.wait(timeout=10)
            except subprocess.TimeoutExpired:
                source.kill()
                source.wait()
        source.stdout.close()
    print('Native member profiles measured; decomposing whole matter patches', flush=True)
    reports = []
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as native, h5py.File(OUT / 'spatial_components.h5', 'x') as out:
        if native.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE' or not np.isclose(native.attrs['h'], request['h']):
            raise ValueError('native total matter/cosmology mismatch')
        out.attrs.update(status='INCOMPLETE', h=request['h'], fine_dx_cMpc_h=.1875,
            axes='Unrotated native TNG axes; native400 periodic grid with faces0..75 cMpc/h',
            fields='mass, momentum_xyz, diagonal_second_xyz; physical Msun and peculiar km/s',
            limits='Calibration sources only. Shared patches/cosmology, not independent realizations or a finite prior bank. Native catalogue valid only for unchanged source components.')
        for sid, component in components.items():
            group = out.create_group(f'halos/{sid}')
            group.create_dataset('global_cell', data=component['cell'])
            group.create_dataset('moments', data=component['moments'])
            group.attrs['profile_json'] = json.dumps(component['report'])
        for index, choice in enumerate(request['choices']):
            mw = components[choice['ids'][0]]['report']
            origin = (np.floor(np.array(mw['position_ckpc_h'])/1000/1.5).astype(int)-8) % 50
            total = read_periodic_patch(native['fine'], origin*8, 128)
            local = []
            for sid in choice['ids']:
                comp = components[sid]
                cell = (comp['cell']-origin*8) % 400
                if np.any(cell >= 128):
                    raise ValueError('complete native halo members exit source patch; no silent clipping')
                local.append(dict(subhalo_id=sid, cell=cell, moments=comp['moments'],
                    position_cMpc_h=(np.array(comp['report']['position_ckpc_h'])/1000-origin*1.5) % 75))
            model = ContinuousMatter(total, local, dx=.1875)
            # Direct reconstruction, without changing/rescaling any halo or
            # applying the old algebraic remainder compensation.
            restored = model.remainder.copy()
            for comp in local:
                keys = np.ravel_multi_index(comp['cell'].T, (128,)*3)
                restored.reshape(7, -1)[:, keys] += comp['moments']
            norm = np.maximum(np.max(abs(total), axis=(1, 2, 3)), 1.)
            error = np.max(abs(restored-total), axis=(1, 2, 3))/norm
            if error.max() > 1e-8:
                raise ValueError('native component+remainder reconstruction fails')
            total_coarse, rem_coarse = restrict(total, 8), restrict(model.remainder, 8)
            check = rem_coarse.copy()
            for comp in local:
                key = np.ravel_multi_index((comp['cell']//8).T, (16,)*3)
                for k in range(7):
                    np.add.at(check.reshape(7, -1)[k], key, comp['moments'][k])
            coarse_error = float(np.max(abs(check-total_coarse)/np.maximum(np.max(abs(total_coarse), axis=(1, 2, 3))[:, None, None, None], 1)))
            if coarse_error > 1e-8:
                raise ValueError('coarse component sums do not reconstruct native total')
            record = dict(**choice, patch_lower_cMpc_h=(origin*1.5).tolist(),
                total_mass_Msun=float(total[0].sum()), member_mass_fraction=float(sum(c['moments'][0].sum() for c in local)/total[0].sum()),
                reconstruction_relative_error=error.tolist(), coarse_reconstruction_error=coarse_error,
                roundoff_removed=model.roundoff_removed.tolist(), remainder_variance_deficit=model.remainder_variance_deficit)
            reports.append(record)
            group = out.create_group(f'patches/{index}')
            group.attrs['record_json'] = json.dumps(record)
            group.create_dataset('remainder', data=model.remainder, chunks=(1, 8, 32, 32), compression='gzip', compression_opts=1)
            group.create_dataset('total_coarse', data=total_coarse)
            print(f'Spatial patch {index+1}/32 done', flush=True)
            del total, restored, model
        out.attrs['status'] = 'NATIVE_JOINT_PROFILES_AND_REMAINDER_NOT_CF4_LG_POSTERIOR'
    report = dict(status='NATIVE_SPATIAL_CALIBRATION_INPUT_READY_NOT_BUNDLE_COMPLETION',
        source_commit=os.environ['EXPECTED_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
        distinct_native_halos=len(components), patch_count=len(reports), massive_rows_read=request['total_rows'],
        profiles=[c['report'] for c in components.values()], patches=reports, elapsed_seconds=time.monotonic()-started,
        limits='Actual native spatial profiles and nonnegative remaining total matter; not a generative distribution, observed LG field, or32 independent simulations. No observation likelihood, covariance fit, new simulation, halo transport, profile sphericalization or source full pass.')
    with (OUT / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('profiles', 'patches')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'serve'])
    parser.add_argument('--request')
    args = parser.parse_args()
    (build() if args.action == 'build' else serve(args.request))
