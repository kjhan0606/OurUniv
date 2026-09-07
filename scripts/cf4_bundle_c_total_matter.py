"""Stream existing total-matter particles into a physical z0 prior source.

serve is I/O only on syntax; build runs exclusively under Slurm. One SSH
source process, no process polling, full-snapshot copy or filesystem probes.
"""
import argparse
from itertools import product
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np

from cf4_bundle_c_tng_stage import RAW, OUT as CATALOG_ROOT

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/total_matter_v1')


def serve(max_files):
    """Copy native arrays/metadata only; no scientific computation on login node."""
    stream = sys.stdout.buffer
    with h5py.File(RAW / 'snapdir_099/snap_099.0.hdf5', 'r') as f:
        pickle.dump(('header', dict(f['Header'].attrs)), stream, protocol=5)
    for index in range(max_files):
        with h5py.File(RAW / f'snapdir_099/snap_099.{index}.hdf5', 'r') as f:
            for p in (0, 1, 4, 5):
                name = f'PartType{p}'
                count = int(f['Header'].attrs['NumPart_ThisFile'][p])
                if not count:
                    continue
                g = f[name]
                for lo in range(0, count, 131072):
                    hi = min(lo + 131072, count)
                    pickle.dump(('particles', p, g['Coordinates'][lo:hi], g['Velocities'][lo:hi],
                        None if p == 1 else g['Masses'][lo:hi]), stream, protocol=5)
        pickle.dump(('file_done', index), stream, protocol=5)
        stream.flush()
    pickle.dump(('end', max_files), stream, protocol=5)
    stream.flush()


def deposit(acc, coordinates, velocity, mass, box):
    """Accumulate mass, momentum and diagonal second moments on a periodic box."""
    if not all(np.isfinite(x).all() for x in (coordinates, velocity, mass)) or np.any(mass < 0):
        raise ValueError('nonfinite particles or negative mass')
    n = acc.shape[-1]
    cell = np.floor(np.mod(coordinates, box) / (box / n)).astype(np.int64)
    key = np.ravel_multi_index(cell.T, (n,) * 3)
    # Sparse reduction avoids clearing a 400^3 temporary for every source block.
    unique, inverse = np.unique(key, return_inverse=True)
    flat = acc.reshape(7, -1)
    flat[0, unique] += np.bincount(inverse, weights=mass, minlength=len(unique))
    for axis in range(3):
        flat[1 + axis, unique] += np.bincount(inverse, weights=mass * velocity[:, axis], minlength=len(unique))
        flat[4 + axis, unique] += np.bincount(inverse, weights=mass * velocity[:, axis]**2, minlength=len(unique))


def build(max_files):
    if 'SLURM_JOB_ID' not in __import__('os').environ:
        raise RuntimeError('numerical builder requires Slurm')
    out = OUTPUT if max_files == 448 else OUTPUT.with_name(f'total_matter_benchmark_{max_files}_v1')
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'syntax',
        '/home/kjhan/miniconda3/bin/python3.13', str(Path(__file__).resolve()), 'serve', '--max-files', str(max_files)]
    source = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        marker, header = pickle.load(source.stdout)
        if marker != 'header' or abs(float(header['Time']) - 1) > 1e-8:
            raise ValueError('complete z0 stream header required')
        box = float(header['BoxSize']) / 1000
        if not np.isclose(box, 75.):
            raise ValueError('this source grid is specified for the native 75 cMpc/h box')
        h, a = float(header['HubbleParam']), float(header['Time'])
        acc = np.zeros((7, 400, 400, 400))
        direct = np.zeros((7, 50, 50, 50))
        counts = np.zeros(6, dtype=np.int64)
        source_totals = np.zeros(7)
        files = 0
        while True:
            item = pickle.load(source.stdout)
            if item[0] == 'end':
                if item[1] != max_files or files != max_files:
                    raise ValueError('source stream ended early')
                break
            if item[0] == 'file_done':
                if item[1] != files:
                    raise ValueError('source chunks out of order')
                files += 1
                if files % 8 == 0 or files == max_files:
                    print(f'full matter files={files}/{max_files}, particles={int(counts.sum())}, seconds={time.monotonic()-start:.1f}', flush=True)
                continue
            _, p, x, v, m = item
            counts[p] += len(x)
            x = x.astype(float) / 1000
            v = v.astype(float) * np.sqrt(a)
            m = (np.full(len(x), header['MassTable'][1]) if p == 1 else m.astype(float)) * 1e10 / h
            source_totals[0] += m.sum()
            source_totals[1:4] += (m[:, None] * v).sum(axis=0)
            source_totals[4:7] += (m[:, None] * v**2).sum(axis=0)
            deposit(acc, x, v, m, box)
            deposit(direct, x, v, m, box)
        if source.wait(timeout=30) != 0:
            raise RuntimeError('native I/O source failed')
    finally:
        if source.poll() is None:
            source.terminate()
            try:
                source.wait(timeout=10)
            except subprocess.TimeoutExpired:
                source.kill()
                source.wait()
        if source.stdout:
            source.stdout.close()
    coarse = acc.reshape(7, 50, 8, 50, 8, 50, 8).sum(axis=(2, 4, 6))
    # Normalize global checks by absolute physical totals, not cancelling momenta.
    component_scale = np.maximum(np.sum(np.abs(acc), axis=(1, 2, 3)), 1)
    global_error = np.abs(acc.sum(axis=(1, 2, 3)) - source_totals) / component_scale
    restriction_error = np.max(np.abs(coarse - direct), axis=(1, 2, 3)) / np.maximum(np.max(np.abs(direct), axis=(1, 2, 3)), 1)
    if np.max(global_error) > 1e-9 or np.max(restriction_error) > 1e-9:
        raise ValueError('mass/momentum/second-moment conservation failed')
    complete = max_files == 448
    if complete:
        expected = np.asarray(header['NumPart_Total'], np.uint64) + (np.asarray(header['NumPart_Total_HighWord'], np.uint64) << np.uint64(32))
        np.testing.assert_array_equal(counts[[0, 1, 4, 5]], expected[[0, 1, 4, 5]])
        # rho_crit = 2.77536627e11 h^2 Msun/Mpc^3; physical volume=(L/h)^3.
        # Check the absolute total-matter normalization; never rescale to pass.
        expected_mass = 2.77536627e11 * float(header['Omega0']) * box**3 / h
        cosmological_mass_error = float(source_totals[0] / expected_mass - 1)
        if abs(cosmological_mass_error) > 1e-3:
            raise ValueError('full-box mass disagrees with native Omega_m by >0.1%; do not renormalize')
        write_prior_source(out, acc, coarse, h, float(header['Omega0']))
    report = dict(status='TOTAL_MATTER_PRIOR_SOURCE_NOT_CF4_POSTERIOR' if complete else 'TIMING_ONLY_INCOMPLETE_SPATIAL_SAMPLE',
        files=files, particle_counts=counts.tolist(), fine_dx_cMpc_h=.1875, coarse_dx_cMpc_h=1.5,
        global_moment_relative_error=global_error.tolist(), restriction_error=restriction_error.tolist(),
        native_cosmological_mass_relative_error=cosmological_mass_error if complete else None,
        elapsed_seconds=time.monotonic()-start, peak_grid_memory_bytes=int(acc.nbytes + direct.nbytes),
        limits='Includes gas, DM, stars/winds, BH dynamical mass, both FoF and unbound matter. Partial chunks are NOT spatially representative and never train the prior. Full run supplies one TNG cosmology/box only; no CF4/LG conditioning or validated continuous fine-field prior.')
    with (out / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


def write_prior_source(out, fine, coarse, h, omega):
    """Keep full moments plus linked patch/catalogue identities; never paste halos."""
    with h5py.File(out / 'matter_moments.h5', 'x') as f:
        f.attrs.update(status='NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE', h=h, Omega_m=omega,
            field_order='mass_Msun, momentum_xyz_Msun_km_s, diagonal_second_xyz_Msun_km2_s2',
            source_catalogue=str(CATALOG_ROOT / 'native_catalog.h5'))
        for name, value, dx in (('fine', fine, .1875), ('coarse', coarse, 1.5)):
            ds = f.create_dataset(name, shape=value.shape, dtype='f8', chunks=(1, 8, 50, 50) if name == 'fine' else (1, 10, 50, 50), compression='gzip', compression_opts=1)
            ds.attrs['dx_cMpc_h'] = dx
            for low in range(0, value.shape[1], 8):
                ds[:, low:low + 8] = value[:, low:low + 8]
        # 27 disjoint 24-Mpc/h patches: a finite-support baseline, not 27
        # independent universes. Gaps remain between patches, no overlap split.
        origins = np.array(list(product((0, 17, 34), repeat=3)), dtype=np.int64)
        features = []
        for origin in origins:
            x, y, z = origin
            patch = coarse[:, x:x + 16, y:y + 16, z:z + 16]
            parent = patch.reshape(7, 2, 8, 2, 8, 2, 8).sum(axis=(2, 4, 6))
            mass = parent[0]
            if np.any(mass <= 0):
                raise ValueError('empty 12-cMpc/h parent in full matter source')
            # 8 log masses and 24 mass-weighted bulk velocities; no LF/count map.
            features.append(np.concatenate([np.log(mass).ravel(), (parent[1:4] / mass).ravel()]))
        features = np.array(features)
        train = origins[:, 0] < 34
        bandwidth = features[train].std(axis=0, ddof=1)
        if np.any(bandwidth <= 0):
            raise ValueError('degenerate prior summary bandwidth')
        f.create_dataset('patch_origins_coarse', data=origins)
        f.create_dataset('patch_coarse_features', data=features)
        f.create_dataset('prior_bandwidth', data=bandwidth)
        f.create_dataset('patch_train', data=train)
        f.attrs['prior_limits'] = 'Equal component weights with diagonal Gaussian coarse-summary kernel; bandwidth=train SD. 18 train and9 nonoverlap check patches share one box. Condition summaries approximately, NEVER enforce exact parent cells by rescaling fields or reusing stale halo catalogues.'
    from cf4_empirical_field_prior import conditional_weights, read_component
    first = read_component(out / 'matter_moments.h5', 0)
    if first['integrals']['mass'].shape != (128,) * 3:
        raise ValueError('whole-patch reader geometry mismatch')
    support = [conditional_weights(features[train], value, bandwidth)['ess'] for value in features[~train]]
    with (out / 'prior_support.json').open('x') as f:
        json.dump(dict(status='FINITE_PATCH_PRIOR_SUPPORT_DIAGNOSTIC_ONLY', training_components=18,
            heldout_components=9, heldout_coarse_condition_ESS=support,
            actual_LG_conditioning=False, actual_CF4_conditioning=False,
            limits='A discrete development prior, not full-field exact coarse conditioning or continuous LG posterior support.'), f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['serve', 'build'])
    parser.add_argument('--max-files', type=int, default=448)
    args = parser.parse_args()
    if not 1 <= args.max_files <= 448:
        parser.error('max-files must lie in1..448')
    (serve if args.action == 'serve' else build)(args.max_files)
