"""Direct six-population/shell selection integral for the common N128 fiducial.

No reweighting or interpolation of the old h=.6711 selection is used.
The angular footprint and luminosity function remain development models.
"""

from itertools import product
import json
import os
from pathlib import Path
import sys
import time

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy import units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_actual_selection import base
from cf4_twompp_joint_information_budget_pilot_v1 import (
    _cosmology_distance_table, schechter_fraction)

DATA = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('selection integral must run through Slurm')
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic()
    contract = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    c = contract['common_cosmology']
    cosmology = dict(h=c['h'], H0_km_s_Mpc=c['H0_km_s_Mpc'],
                     Omega_m=c['Omega_m'], Omega_b=c['Omega_b'], Tcmb_K=c['Tcmb_K'])
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    maps = [base.load_completeness_map(tracer['inputs'][name]['path'], 512)
            for name in ('completeness_11_5', 'completeness_12_5')]
    rotation = SkyCoord(CartesianRepresentation(np.eye(3)),
                        frame='supergalactic').icrs.cartesian.xyz.value
    edges = np.array([5., 30., 60., 90., 120., 150., 180.])
    rtab = np.linspace(edges[0], edges[-1], 200001)
    dl = _cosmology_distance_table(rtab, cosmology)*cosmology['h']
    absolute = tracer['tracer_design']['absolute_K_edges']
    def fraction(distance, population):
        app, ab = divmod(population, 3)
        return schechter_fraction(distance, None if app == 0 else 11.5,
            11.5 if app == 0 else 12.5, absolute[ab], absolute[ab+1], -23.28, -.94)
    radial = np.array([fraction(dl, p) for p in range(6)])
    check_radius = np.random.default_rng(2026092502).uniform(5., 180., 1024)
    check_dl = _cosmology_distance_table(check_radius, cosmology)*cosmology['h']
    table_error = max(float(np.max(np.abs(np.interp(check_radius, rtab, radial[p])
                                - fraction(check_dl, p)))) for p in range(6))
    if table_error > 1e-6:
        raise RuntimeError('radial luminosity table interpolation inaccurate')
    n, dx, order, slab_width = 128, 3., 6, 4
    axis = (np.arange(n)+.5)*dx-192.
    nodes, weights = np.polynomial.legendre.leggauss(order)
    with np.load(DATA/'counts_3_sparse.npz', allow_pickle=False) as f:
        keys = f['all_keys']
    occupied_pop = keys//n**3
    occupied_cell = np.array(np.unravel_index(keys%n**3, (n,)*3)).T
    total = np.zeros((6, 6), dtype=np.float64)
    unsupported = []
    OUT.mkdir(parents=True)
    with h5py.File(OUT/'selection_3.h5', 'x') as h:
        output = h.create_dataset('selection_shells', shape=(6,6,n,n,n), dtype='f4',
                                  chunks=(1,1,slab_width,64,64), compression='gzip',
                                  compression_opts=1)
        h.attrs.update(status='INCOMPLETE', box_cMpc_h=384., dx_cMpc_h=dx,
                       quadrature_order=order, h=cosmology['h'])
        for lower in range(0, n, slab_width):
            centers = np.array(np.meshgrid(axis[lower:lower+slab_width], axis,
                                           axis, indexing='ij')).reshape(3,-1)
            block = np.zeros((6,6,centers.shape[1]), dtype=np.float64)
            for a,b,cidx in product(range(order), repeat=3):
                xyz = centers + dx/2*nodes[[a,b,cidx],None]
                radius = np.linalg.norm(xyz, axis=0)
                active = (radius >= edges[0]) & (radius <= edges[-1])
                ids = np.flatnonzero(active)
                if ids.size == 0:
                    continue
                r = radius[ids]
                pixels = hp.vec2pix(512, *(rotation@xyz[:,ids]), nest=False)
                shell = np.clip(np.searchsorted(edges,r,side='right')-1,0,5)
                weight = weights[a]*weights[b]*weights[cidx]/8
                for p in range(6):
                    block[p,shell,ids] += weight*maps[p//3][pixels]*np.interp(r,rtab,radial[p])
            total += block.sum(axis=2)*dx**3
            selection = block.reshape(6,6,slab_width,n,n).astype(np.float32)
            output[:,:,lower:lower+slab_width] = selection
            take = np.flatnonzero((occupied_cell[:,0]>=lower)
                                  & (occupied_cell[:,0]<lower+slab_width))
            collapsed = selection.sum(axis=1)
            value = collapsed[occupied_pop[take], occupied_cell[take,0]-lower,
                              occupied_cell[take,1], occupied_cell[take,2]]
            unsupported.extend(keys[take[value <= 0]].tolist())
            print(f'common selection x={lower+slab_width}/{n}; elapsed={time.monotonic()-start:.1f}s',
                  flush=True)
        h.attrs['status'] = ('INTEGRATION_COMPLETE_NOT_CALIBRATED' if not unsupported
                             else 'UNRESOLVED_OBSERVED_SUPPORT')
    report = dict(classification='DIRECT_COMMON_COSMOLOGY_N128_SELECTION',
                  cosmology=cosmology, N=n, dx_cMpc_h=dx, quadrature_order=order,
                  LF_table_max_absolute_error=table_error,
                  effective_volume_cMpc_h3=total.tolist(),
                  observed_cells=int(len(keys)),
                  occupied_zero_selection_keys=unsupported,
                  elapsed_seconds=time.monotonic()-start,
                  source_maps='official 2M++ HEALPix maps in tracer program',
                  old_selection_reused=False, angular_quadrature_certified=False,
                  survival_bias_FoG_calibrated=False,
                  quantitative_joint_likelihood_ready=False)
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('classification','observed_cells',
        'occupied_zero_selection_keys','elapsed_seconds')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
