"""Exact row alignment/transforms for post-fit selected-mark holdout scoring.

This pure transformer is not an extraction command and must only receive
measurement rows after the posterior-predictive embargo is released. It uses
the existing optical covariance/cut transforms and keeps the CF4 FP redshift
reference separate from the linked 2M++ point/source-conditioning radius.
"""

import math

import numpy as np

from cf4_r2_raw_selected_fp import optical_cut_geometry, optical_error_covariance


N_GRID = 128
FP_FIELDS = ('r', 'er', 's', 'es', 'i', 'ei', 'deVMag_r', 'NgroupT17')
CUT_FIELDS = ('SIGMA_STARS',)
POINT_FIELDS = ('population', 'flat_cell', 'radius_cMpc_h', 'ksmag')


def prepare_postfit_mark_observations(manifest, *, fp_by_pgc, cut_by_pgc,
                                      point_by_recno, group_by_label,
                                      richness_center):
    """Align a certified identity manifest with raw marks and exact transforms.

    Inputs are keyed by the manifest's PGC, linked point recno and
    source-group label. Missing/duplicate/invalid rows raise; none are silently
    removed. This helper must not be called before valid post-fit draws exist.
    """
    if not manifest:
        raise ValueError('certified post-fit mark manifest is empty')
    centre = float(richness_center)
    if not math.isfinite(centre):
        raise ValueError('training-only richness centre must be finite')
    keys = ('PGC', 'source_group', 'point_recno')
    for key in keys:
        values = [row[key] for row in manifest]
        if len(values) != len(set(values)):
            raise ValueError(f'manifest repeats {key}')

    output = []
    for identity in manifest:
        pgc = int(identity['PGC'])
        recno = int(identity['point_recno'])
        label = str(identity['source_group'])
        try:
            fp = fp_by_pgc[pgc]
            cut = cut_by_pgc[pgc]
            point = point_by_recno[recno]
            group = group_by_label[label]
        except KeyError as exc:
            raise ValueError(f'missing post-fit measurement/geometry row for {exc.args[0]}') from exc
        if any(name not in fp for name in FP_FIELDS):
            raise ValueError(f'FP row {pgc} lacks a required raw-mark field')
        if any(name not in cut for name in CUT_FIELDS):
            raise ValueError(f'FP row {pgc} lacks a required cut field')
        if any(name not in point for name in POINT_FIELDS):
            raise ValueError(f'linked point {recno} lacks a required point field')
        if 'dz_row' not in group:
            raise ValueError(f'CF4 source group {label} lacks its frozen dz_row reference')

        x = np.asarray([fp['r'], fp['s'], fp['i']], dtype=np.float64)
        errors = np.asarray([fp['er'], fp['es'], fp['ei']], dtype=np.float64)
        sigma = float(cut['SIGMA_STARS'])
        magnitude_r = float(fp['deVMag_r'])
        population = int(point['population'])
        flat_cell = int(point['flat_cell'])
        point_radius = float(point['radius_cMpc_h'])
        k_magnitude = float(point['ksmag'])
        dz_row = float(group['dz_row'])
        richness = float(fp['NgroupT17'])
        if (not np.isfinite(np.r_[x, errors, sigma, magnitude_r, point_radius,
                                  k_magnitude, dz_row, richness]).all()
                or np.any(errors <= 0.) or sigma <= 0. or point_radius <= 0.
                or dz_row <= 0. or richness < 0.):
            raise ValueError(f'nonfinite or invalid post-fit input for PGC {pgc}')
        if population < 0 or population >= 6:
            raise ValueError(f'unknown 2M++ population for linked point {recno}')
        if flat_cell < 0 or flat_cell >= N_GRID**3:
            raise ValueError(f'linked point {recno} lies outside the frozen N128 grid')

        error_covariance = optical_error_covariance(*errors)
        cut_matrix, cut_lower, cut_upper = optical_cut_geometry(x, magnitude_r, sigma)
        output.append(dict(
            PGC=pgc,
            source_group=label,
            point_recno=recno,
            population=population,
            flat_cell=flat_cell,
            # Keep these separate even where this single-link cohort gives
            # equal numeric values: they have different operator ownership.
            linked_point_radius_cMpc_h=point_radius,
            source_conditioning_radius_cMpc_h=point_radius,
            cf4_fp_dz_row=dz_row,
            observed_ksmag=k_magnitude,
            x=x,
            optical_error_covariance=error_covariance,
            cut_matrix=cut_matrix,
            cut_lower=cut_lower,
            cut_upper=cut_upper,
            richness=np.log1p(richness)-centre,
        ))
    return output
