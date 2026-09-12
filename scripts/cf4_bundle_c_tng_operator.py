"""One native-particle validation of the resolved z0 operator; NOT an LG fit."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from cf4_resolved_moments import particle_moments, aggregate, derived, periodic_delta, resolved_catalog
from cf4_lg_observation_contract import solar_reference, basis, predict
from cf4_bundle_c_tng_stage import OUT, FIELDS

ROOT = Path(__file__).resolve().parents[1]


def load_catalog():
    pieces = {g: {k: [] for k in fields} for g, fields in FIELDS.items()}
    snapshot_counts = []
    with h5py.File(OUT / 'native_catalog.h5', 'r') as f:
        if f.attrs['status'] != 'COMPLETE_NATIVE_FIELD_COPY_NO_SELECTION':
            raise ValueError('native staging incomplete')
        head = dict(f['chunks/0/Header'].attrs)
        snapshot = dict(f['chunks/0/SnapshotHeader'].attrs)
        for i in range(448):
            chunk = f[f'chunks/{i}']
            for g, fields in pieces.items():
                for k in fields:
                    if k in chunk[g]:
                        pieces[g][k].append(chunk[g][k][:])
            snapshot_counts.append(chunk['SnapshotHeader'].attrs['NumPart_ThisFile'])
    cat = {g: {k: np.concatenate(v) for k, v in fields.items()} for g, fields in pieces.items()}
    if len(cat['Subhalo']['SubhaloPos']) != head['Nsubgroups_Total'] or len(cat['Group']['GroupNsubs']) != head['Ngroups_Total']:
        raise ValueError('incomplete native catalogue')
    if abs(head['Time'] - 1) > 1e-8:
        raise ValueError('z0 input required')
    return cat, head, snapshot, np.array(snapshot_counts, dtype=np.int64)


def prepare():
    cat, head, snapshot, counts = load_catalog()
    g, s = cat['Group'], cat['Subhalo']
    mass = g['Group_M_Crit200'].astype(float) * 1e10 / head['HubbleParam']
    # A fixed, bounded operator fixture, not an observational mass likelihood,
    # isolation cut, optimized analogue, or empirical conditional prior.
    eligible = np.flatnonzero((mass >= 5e11) & (mass <= 3e12) & (g['GroupNsubs'] >= 3)
        & (g['GroupLenType'].astype(np.int64).sum(axis=1) <= 1000000))
    chosen = None
    for index in eligible:
        first = int(g['GroupFirstSub'][index])
        ids = np.arange(first, first + 3)
        if np.all(s['SubhaloFlag'][ids]) and np.all(s['SubhaloLenType'][ids, 1] >= 100):
            chosen = int(index)
            break
    if chosen is None:
        raise ValueError('no bounded three-resolved-object fixture; do not relax silently')
    group_start = g['GroupLenType'][:chosen].astype(np.int64).sum(axis=0)
    group_length = g['GroupLenType'][chosen].astype(np.int64)
    file_start = np.vstack([np.zeros(6, dtype=np.int64), np.cumsum(counts, axis=0)])
    types = []
    for p in (0, 1, 4, 5):
        length = int(group_length[p])
        if not length:
            continue
        start, stop = int(group_start[p]), int(group_start[p] + length)
        pieces = []
        for fi in range(len(counts)):
            lo, hi = max(start, int(file_start[fi, p])), min(stop, int(file_start[fi + 1, p]))
            if hi > lo:
                pieces.append(dict(file=fi, start=lo - int(file_start[fi, p]), stop=hi - int(file_start[fi, p]),
                    destination_start=lo - start, destination_stop=hi - start))
        if sum(x['stop'] - x['start'] for x in pieces) != length:
            raise ValueError('FoF particle range not covered by snapshot headers')
        types.append(dict(type=p, count=length, fields=['Coordinates', 'Velocities'] + ([] if p == 1 else ['Masses']), pieces=pieces))
    request = dict(status='NATIVE_FOF_PARTICLE_REQUEST_NOT_LG_SELECTION', group_id=chosen,
        subhalo_ids=ids.tolist(), particle_types=types, h=float(head['HubbleParam']), a=float(head['Time']),
        box_ckpc_h=float(head['BoxSize']), DM_mass_native=float(snapshot['MassTable'][1]),
        source='TNG100-1 snapshot99; existing local raw data, no download',
        fixture_rule='First native group ID with M200c 5e11..3e12 physical Msun, <=1e6 particles, >=3 valid subhalos each >=100 DM particles. Engineering fixture only, NOT an LG prior/cut.',
        limits='Three objects in ONE FoF group are not asserted to be MW/M31/M33 analogues. No CF4 data or observed LG likelihood used.')
    with (OUT / 'particle_request.json').open('x') as f:
        json.dump(request, f, indent=2)
    print(json.dumps(request), flush=True)


def measure():
    start = time.monotonic()
    request = json.loads((OUT / 'particle_request.json').read_text())
    cat, head, _, _ = load_catalog()
    g, s = cat['Group'], cat['Subhalo']
    ids = request['subhalo_ids']
    h, a = request['h'], request['a']
    center = s['SubhaloPos'][ids[0]].astype(float)
    positions, velocities, masses = [], [], []
    sub_mass = np.zeros(3)
    sub_momentum = np.zeros((3, 3))
    with h5py.File(OUT / 'native_particles.h5', 'r') as f:
        if f.attrs['status'] != 'COMPLETE_NATIVE_FOF_COPY_NOT_SPATIAL_CUTOUT':
            raise ValueError('particle staging incomplete')
        for item in request['particle_types']:
            p = item['type']
            group = f[f'PartType{p}']
            position = group['Coordinates'][:]
            velocity = group['Velocities'][:].astype(float) * np.sqrt(a)
            mass = (np.full(len(position), request['DM_mass_native']) if p == 1 else group['Masses'][:].astype(float)) * 1e10 / h
            positions.append(periodic_delta(position, center, request['box_ckpc_h']) / 1000)
            velocities.append(velocity)
            masses.append(mass)
            offset = 0
            for j, sid in enumerate(ids):
                length = int(s['SubhaloLenType'][sid, p])
                sub_mass[j] += mass[offset:offset + length].sum()
                sub_momentum[j] += (mass[offset:offset + length, None] * velocity[offset:offset + length]).sum(axis=0)
                offset += length
    position, velocity, mass = map(np.concatenate, (positions, velocities, masses))
    reference_mass = s['SubhaloMass'][ids].astype(float) * 1e10 / h
    reference_velocity = s['SubhaloVel'][ids].astype(float)
    np.testing.assert_allclose(sub_mass, reference_mass, rtol=2e-4)
    com_difference = sub_momentum / sub_mass[:, None] - reference_velocity
    if np.max(np.abs(com_difference)) > .2:
        raise ValueError('particle COM and catalogue velocity differ by >0.2 km/s; investigate rather than score LG')
    fine = particle_moments(position, velocity, mass, lower=[-12.] * 3, dx=.1875, n=128)
    coarse = aggregate(fine, 8)
    direct = particle_moments(position, velocity, mass, lower=[-12.] * 3, dx=1.5, n=16)
    for name in ('mass', 'momentum', 'second_moment'):
        np.testing.assert_allclose(coarse[name], direct[name], rtol=1e-8, atol=1e-2)
    np.testing.assert_array_equal(coarse['count'], direct['count'])
    with h5py.File(OUT / 'fof_component_moments.h5', 'x') as f:
        f.attrs.update(status='RESOLVED_FOF_COMPONENT_NOT_TOTAL_MATTER_MAP_OR_CF4_POSTERIOR',
            group_id=request['group_id'], cosmology_h=h, axes='Native TNG axes, MW-role potential minimum at origin',
            limits='Bound/FoF matter only; no external diffuse matter. Physical velocity dispersion is NOT posterior uncertainty.')
        for name, integrals, dx in (('fine', fine, .1875), ('coarse', coarse, 1.5)):
            group = f.create_group(name)
            group.attrs.update(dx_cMpc_h=dx, lower_cMpc_h=[-12.] * 3)
            for key, value in {**{k: integrals[k] for k in ('mass', 'momentum', 'second_moment', 'count')}, **derived(integrals, dx)}.items():
                group.create_dataset(key, data=value, compression='gzip', compression_opts=1)
    contract = json.loads((ROOT / 'config/cf4_lg_observation_contract_v1.json').read_text())
    catalog = resolved_catalog(s['SubhaloPos'], s['SubhaloVel'], s['SubhaloMass'], s['SubhaloGrNr'],
        g['GroupFirstSub'], g['Group_M_Crit200'], dict(zip(('MW', 'M31', 'M33'), ids)),
        h=h, a=a, box_ckpc_h=request['box_ckpc_h'], rotation=np.eye(3), frame=contract['frame'])
    solar_pos, solar_vel = solar_reference(contract)
    # Mock sky only for exercising projection of the real simulated objects.
    # The original actual-data contract is never modified on disk or scored.
    for name in ('M31', 'M33'):
        x = catalog[name]['position_kpc'] - solar_pos
        contract['measurements'][name]['ra_deg'] = float(np.rad2deg(np.arctan2(x[1], x[0])) % 360)
        contract['measurements'][name]['dec_deg'] = float(np.rad2deg(np.arcsin(x[2] / np.linalg.norm(x))))
    prediction = predict(catalog, contract, h=h, solar_position_kpc=solar_pos, solar_velocity_km_s=solar_vel)
    report = dict(status='NATIVE_RESOLVED_OPERATOR_AND_MOMENTS_PASS_NOT_LG_CONDITIONING',
        group_id=request['group_id'], subhalo_ids=ids, cosmology=dict(h=h, Omega_m=float(head['Omega0'])),
        particle_count=len(mass), particle_mass_relative_error=(sub_mass / reference_mass - 1).tolist(),
        particle_COM_minus_catalog_km_s=com_difference.tolist(),
        captured_mass_fraction=float(mass[fine['inside']].sum() / mass.sum()),
        fine_dx_cMpc_h=.1875, coarse_dx_cMpc_h=1.5, exact_integer_count_restriction=True,
        conservative_mass_momentum_second_moment=True, mock_sky_observables=prediction.tolist(),
        explicit_identity_catalog=catalog, elapsed_seconds=time.monotonic()-start,
        limits='One real TNG FoF fixture, NOT an LG analogue ensemble or field prior. Mass/profile and velocity moments are physical simulation data; catalogue COM is not stellar-disk COM. No actual LG likelihood, CF4 conditioning, full matter map, or fine posterior sampling. TNG cosmology is not silently rescaled to the CF4 parent.')
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    with (OUT / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, default=encode, allow_nan=False)
    print(json.dumps(report, default=encode, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'measure'])
    (prepare if parser.parse_args().action == 'prepare' else measure)()
