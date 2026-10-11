"""Prepare fixed N128 angular and distance tables for the marked GPU screen."""

import json
import os
from pathlib import Path
import sys

import healpy as hp
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_actual_selection import base

OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_marked_source_geometry_v1')
N, BOX = 128, 384.


def angular_cell_average(map_files, positions, *, observer, dx, block=65536):
    """Eight fixed subcell sightlines; angular-only coarsened approximation."""
    maps = [base.load_completeness_map(path, 512) for path in map_files]
    rotation = SkyCoord(CartesianRepresentation(np.eye(3)),
                        frame='supergalactic').icrs.cartesian.xyz.value
    result = np.zeros((2, len(positions)), dtype=np.float64)
    shifts = np.array([[i, j, k] for i in (-.25, .25)
                       for j in (-.25, .25) for k in (-.25, .25)])*dx
    for lo in range(0, len(positions), block):
        hi = min(lo+block, len(positions))
        relative = positions[lo:hi]-observer
        for shift in shifts:
            rays = relative+shift
            pixels = hp.vec2pix(512, *(rotation @ rays.T), nest=False)
            result[:, lo:hi] += np.stack([m[pixels] for m in maps])/8.
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit through Slurm')
    if OUT.exists():
        raise FileExistsError(OUT)
    cfg = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    cosmology = cfg['common_cosmology']
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    map_files = [tracer['inputs'][name]['path'] for name in
                 ('completeness_11_5', 'completeness_12_5')]
    dx = BOX/N
    axis = (np.arange(N)+.5)*dx
    positions = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'),
                         axis=-1).reshape(-1, 3)
    observer = np.full(3, BOX/2.)
    angular = angular_cell_average(map_files, positions, observer=observer,
                                   dx=dx)
    if not np.isfinite(angular).all() or np.any((angular < 0) | (angular > 1)):
        raise ValueError('invalid angular completeness')
    model = FlatLambdaCDM(H0=cosmology['H0_km_s_Mpc']*u.km/u.s/u.Mpc,
                          Om0=cosmology['Omega_m'], Ob0=cosmology['Omega_b'],
                          Tcmb0=cosmology['Tcmb_K']*u.K)
    z_grid = np.linspace(0, .2, 20001)
    r_grid = model.comoving_distance(z_grid).to_value(u.Mpc)*cosmology['h']
    radial_table = np.linspace(.001, 192., 20001)
    z_table = np.interp(radial_table, r_grid, z_grid)
    dl_h = model.luminosity_distance(z_table).to_value(u.Mpc)*cosmology['h']
    modulus_table = 5*np.log10(dl_h)+25.
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'geometry.npz', positions=positions.astype('f4'),
                        angular=angular.astype('f4'), radial_table=radial_table,
                        redshift_table=z_table, modulus_table=modulus_table)
    (OUT/'result.json').write_text(json.dumps(dict(
        classification='SOURCE_MARKED_N128_FIXED_GEOMETRY_NOT_CALIBRATED',
        angular_map_paths=map_files, angular_subcell_order=2,
        sources=len(positions), zero_angular_both=int(np.count_nonzero(
            np.all(angular == 0, axis=0))), cosmology=cosmology,
        source_or_observation_model_calibrated=False),
        indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
