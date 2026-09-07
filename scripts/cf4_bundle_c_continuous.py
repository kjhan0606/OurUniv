"""Native total-matter transport + ONE noiseless inverse control, Slurm only.

This is a kinematic parameterization experiment, NOT a posterior or a
calibration of physically allowed changes. No new raw-snapshot pass.
"""
from itertools import product
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.optimize import least_squares

from cf4_continuous_matter import ContinuousMatter, check_realizable, restrict
from cf4_bundle_c_tng_stage import OUT as NATIVE

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
SOURCE = ROOT / 'total_matter_v1/matter_moments.h5'
OUTPUT = ROOT / 'continuous_transport_v1'


def native_rows(ids):
    fields = ('SubhaloPos', 'SubhaloVel', 'SubhaloMass', 'SubhaloLenType')
    rows = {sid: {} for sid in ids}
    offset = 0
    with h5py.File(NATIVE / 'native_catalog.h5', 'r') as f:
        for i in range(448):
            g = f[f'chunks/{i}']
            length = int(g['Header'].attrs['Nsubgroups_ThisFile'])
            for sid in ids:
                if offset <= sid < offset + length:
                    rows[sid] = {name: g[f'Subhalo/{name}'][sid-offset] for name in fields}
            offset += length
    if any(len(row) != len(fields) for row in rows.values()):
        raise ValueError('requested native catalogue rows missing')
    return rows


def read_periodic_patch(dataset, start, n):
    side = dataset.shape[-1]
    segments = []
    for s in start:
        count = min(n, side - s)
        entries = [(slice(s, s+count), slice(0, count))]
        if count < n:
            entries.append((slice(0, n-count), slice(count, n)))
        segments.append(entries)
    out = np.empty((7, n, n, n))
    for parts in product(*segments):
        out[(slice(None),) + tuple(p[1] for p in parts)] = dataset[(slice(None),) + tuple(p[0] for p in parts)]
    return out


def prepare():
    request = json.loads((NATIVE / 'particle_request.json').read_text())
    ids, h, dx, n = request['subhalo_ids'], request['h'], .1875, 128
    rows = native_rows(ids)
    # Same native periodic axes/faces as the full source. Integer alignment
    # is essential for subtracting disjoint member moments without smoothing.
    origin = (np.floor(rows[ids[0]]['SubhaloPos'].astype(float)/1000/1.5).astype(int) - 8) % 50
    lower = origin * 1.5
    with h5py.File(SOURCE, 'r') as f:
        if f.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE':
            raise ValueError('complete total-matter source required')
        if not np.isclose(f.attrs['h'], h):
            raise ValueError('source cosmology mismatch')
        total = read_periodic_patch(f['fine'], origin*8, n)
    parts = {sid: [[], [], []] for sid in ids}
    with h5py.File(NATIVE / 'native_particles.h5', 'r') as f:
        if f.attrs['status'] != 'COMPLETE_NATIVE_FOF_COPY_NOT_SPATIAL_CUTOUT':
            raise ValueError('complete native fixture required')
        for item in request['particle_types']:
            p, offset = item['type'], 0
            g = f[f'PartType{p}']
            for sid in ids:
                length = int(rows[sid]['SubhaloLenType'][p])
                sl = slice(offset, offset+length)
                x = (g['Coordinates'][sl].astype(float)/1000 - lower) % 75
                v = g['Velocities'][sl].astype(float) * np.sqrt(request['a'])
                m = (np.full(length, request['DM_mass_native']) if p == 1 else g['Masses'][sl].astype(float))*1e10/h
                for dest, value in zip(parts[sid], (x, v, m)):
                    dest.append(value)
                offset += length
    components = []
    for sid in ids:
        x, v, m = map(np.concatenate, parts[sid])
        cell = np.floor(x/dx).astype(int)
        # This is a complete member component, not a silently clipped profile.
        if np.any(cell < 2) or np.any(cell >= n-2):
            raise ValueError('native halo profile lacks two-cell transport buffer')
        key, inverse = np.unique(np.ravel_multi_index(cell.T, (n,)*3), return_inverse=True)
        weights = np.concatenate([m[None], (m[:, None]*v).T, (m[:, None]*v*v).T])
        value = np.array([np.bincount(inverse, weights=w, minlength=len(key)) for w in weights])
        sums = value.sum(axis=1)
        np.testing.assert_allclose(sums[0], rows[sid]['SubhaloMass']*1e10/h, rtol=2e-4)
        np.testing.assert_allclose(sums[1:4]/sums[0], rows[sid]['SubhaloVel'], atol=.2, rtol=0)
        components.append(dict(subhalo_id=sid, cell=np.array(np.unravel_index(key, (n,)*3)).T,
            moments=value, position_cMpc_h=(rows[sid]['SubhaloPos'].astype(float)/1000-lower)%75))
    model = ContinuousMatter(total, components, dx=dx)
    return model, total, dict(h=h, dx_cMpc_h=dx, patch_lower_native_cMpc_h=lower.tolist(),
        source=str(SOURCE), native_group_id=request['group_id'], subhalo_ids=ids,
        identities='Native engineering fixture only; NOT MW/M31/M33 or an LG analogue')


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def run():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('scientific calculation requires Slurm')
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    model, baseline, metadata = prepare()
    print('Native total-matter decomposition ready', flush=True)
    zero = np.zeros((3, 7))
    control, _, _ = model.evaluate(zero)
    scale = np.maximum(np.max(abs(baseline), axis=(1, 2, 3)), 1)
    identity_error = np.max(abs(control-baseline), axis=(1, 2, 3))/scale
    if np.max(identity_error) > 1e-8:
        raise ValueError('zero transport does not reproduce native total moments')
    del control

    # Dimensionless injected values and box bounds are engineering stress
    # amplitudes, NOT LG measurements, observational errors or priors.
    units = np.array([.1875]*3 + [.2] + [30.]*3)
    injected = np.array([[.35, -.25, .2, .6, .4, -.3, .2],
                         [-.3, .4, -.2, -.4, -.3, .5, -.4],
                         [.2, .3, -.35, .5, .2, .3, -.5]])
    support = []
    for comp in model.components:
        for offset in product(range(-2, 3), repeat=3):
            cells = comp['cell'] + offset
            support.append(np.ravel_multi_index(cells.T, (model.n,)*3))
    keys = np.unique(np.concatenate(support))
    coordinates = np.array(np.unravel_index(keys, (model.n,)*3))
    train = coordinates.sum(axis=0) % 2 == 0
    training_keys, heldout_keys = keys[train], keys[~train]
    target = model.selected(injected*units, training_keys)
    initial = model.selected(zero, training_keys)
    # Fixed conditioning of an exact inverse problem, NOT measurement sigma.
    # Only mass and momentum are fitted; second moments are prediction checks.
    weights = np.maximum(np.abs(initial[:4]), np.max(abs(initial[:4]), axis=1)[:, None]*.005)
    def residual(params):
        prediction = model.selected(params.reshape(3, 7)*units, training_keys)
        return ((prediction[:4]-target[:4])/weights).ravel()
    fit = least_squares(residual, zero.ravel(), bounds=(-1, 1), max_nfev=120,
                        ftol=1e-11, xtol=1e-11, gtol=1e-11, diff_step=1e-5)
    recovered = fit.x.reshape(3, 7)
    truth, truth_marks, truth_remainder = model.evaluate(injected*units)
    prediction, recovered_marks, recovered_remainder = model.evaluate(recovered*units)
    before = np.linalg.norm(residual(zero.ravel()))
    after = np.linalg.norm(fit.fun)
    parameter_error = float(np.max(abs(recovered-injected)))
    heldtruth = truth.reshape(7, -1)[:, heldout_keys]
    heldprediction = prediction.reshape(7, -1)[:, heldout_keys]
    heldbaseline = baseline.reshape(7, -1)[:, heldout_keys]
    heldout_change_error = np.linalg.norm(heldprediction-heldtruth, axis=1) / np.maximum(np.linalg.norm(heldtruth-heldbaseline, axis=1), 1.)
    total_error = []
    restriction_error = []
    for theta, value in ((injected*units, truth), (recovered*units, prediction)):
        check_realizable(value)
        integrals = value.sum(axis=(1, 2, 3))
        absolute = np.maximum(np.sum(abs(baseline), axis=(1, 2, 3)), 1)
        total_error.append((abs(integrals[:4]-model.total[:4])/absolute[:4]).tolist())
        direct, _, _ = model.evaluate(theta, ratio=8)
        down = restrict(value, 8)
        restriction_error.append(float(np.max(abs(direct-down)/np.maximum(np.max(abs(direct), axis=(1, 2, 3))[:, None, None, None], 1))))
    if np.max(total_error) > 1e-8 or max(restriction_error) > 1e-8:
        raise ValueError('conservation/restriction failed')
    passed = bool(fit.success and parameter_error < .02 and np.max(heldout_change_error) < .02)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    report = dict(status='KINEMATIC_CONTROL_PASS_NOT_PHYSICAL_PRIOR_OR_LG_POSTERIOR' if passed else 'NO_GO_KINEMATIC_INVERSE_CONTROL',
        metadata=metadata, source_commit=os.environ.get('EXPECTED_COMMIT'), job_id=os.environ['SLURM_JOB_ID'],
        native_identity_relative_error=identity_error.tolist(),
        subtraction_roundoff_removed_moments=model.roundoff_removed.tolist(),
        remainder_max_relative_variance_deficit=model.remainder_variance_deficit,
        mass_momentum_conservation_relative_error=total_error, fine_to_coarse_error=restriction_error,
        fitted_parameters=21, training_cells=len(training_keys), heldout_cells=len(heldout_keys),
        injected_dimensionless=injected.tolist(), recovered_dimensionless=recovered.tolist(),
        parameter_units=units.tolist(), maximum_dimensionless_parameter_error=parameter_error,
        optimizer_success=bool(fit.success), optimizer_message=fit.message, optimizer_nfev=fit.nfev,
        training_initial_residual_norm=float(before), training_final_residual_norm=float(after),
        heldout_error_relative_to_injected_change=heldout_change_error.tolist(),
        jacobian_smallest_singular_value=float(singular[-1]),
        truth_marks=truth_marks, recovered_marks=recovered_marks,
        truth_remainder=truth_remainder, recovered_remainder=recovered_remainder,
        elapsed_seconds=time.monotonic()-started,
        limits=['Same-generator noiseless inverse control; NOT independent validation or a posterior.',
            'One FoF fixture, not LG. Native memberships/shapes fixed; new boundness/host M200c not established.',
            'Remainder includes other halos and diffuse matter; global mass rescaling/boost is algebraic compensation, not calibrated physical response.',
            'Finite-volume transport at .1875 adds fractional-cell smoothing; does not resolve halo forces or recover high-k phases.',
            'Mass and momentum conserved for whole patch, not frozen in each parent cell. Kinetic energy need not be conserved by state changes.',
            'No prior widths, LG likelihood, LCDM/dynamic compatibility, mean-velocity posterior uncertainty or nine-target prior-support success claimed.',
            'Need a calibrated physical response/prior and COM/profile discrepancy before real LG conditioning; do not repeat toy controls as its substitute.'])
    with h5py.File(OUTPUT / 'fields.h5', 'x') as f:
        f.attrs.update(status=report['status'], dx_cMpc_h=.1875, h=metadata['h'],
            patch_lower_native_cMpc_h=metadata['patch_lower_native_cMpc_h'],
            field_order='mass, momentum_xyz, diagonal_second_xyz; physical Msun and peculiar km/s',
            uncertainty='No posterior; diagonal seconds encode physical velocity dispersion, not uncertainty of mean.')
        for name, value in (('native', baseline), ('injected', truth), ('recovered', prediction)):
            f.create_dataset(name, data=value, compression='gzip', compression_opts=1)
    with (OUTPUT / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, default=encode, allow_nan=False)
    print(json.dumps(report, default=encode, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    run()
