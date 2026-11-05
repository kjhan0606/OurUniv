"""Synthetic transform/alignment tests; no real held-out rows are loaded."""
import unittest

import numpy as np

from cf4_r2_raw_mark_holdout_inputs import prepare_postfit_mark_observations
from cf4_r2_raw_selected_fp import optical_cut_geometry, optical_error_covariance


def inputs():
    manifest = [dict(PGC=101, source_group='g1', point_recno=501)]
    fp = {101: dict(r=14.2, er=.03, s=2.1, es=.04, i=13.7, ei=.02,
                    deVMag_r=14.0, NgroupT17=2)}
    cut = {101: dict(SIGMA_STARS=130.)}
    point = {501: dict(population=3, flat_cell=12345,
                       radius_cMpc_h=42.5, ksmag=11.2)}
    group = {'g1': dict(dz_row=6200.)}
    return manifest, fp, cut, point, group


class RawMarkHoldoutInputTests(unittest.TestCase):
    def test_aligns_exact_transforms_and_keeps_distance_owners_separate(self):
        manifest, fp, cut, point, group = inputs()
        rows = prepare_postfit_mark_observations(
            manifest, fp_by_pgc=fp, cut_by_pgc=cut, point_by_recno=point,
            group_by_label=group, richness_center=.5)
        row = rows[0]
        np.testing.assert_allclose(row['x'], [14.2, 2.1, 13.7])
        np.testing.assert_allclose(row['optical_error_covariance'],
                                   optical_error_covariance(.03, .04, .02))
        matrix, lower, upper = optical_cut_geometry(
            [14.2, 2.1, 13.7], 14.0, 130.)
        np.testing.assert_allclose(row['cut_matrix'], matrix)
        np.testing.assert_allclose(row['cut_lower'], lower)
        np.testing.assert_allclose(row['cut_upper'], upper)
        self.assertEqual(row['linked_point_radius_cMpc_h'], 42.5)
        self.assertEqual(row['source_conditioning_radius_cMpc_h'], 42.5)
        self.assertEqual(row['cf4_fp_dz_row'], 6200.)
        self.assertAlmostEqual(row['richness'], np.log1p(2)-.5)

    def test_missing_or_invalid_rows_are_not_silently_dropped(self):
        manifest, fp, cut, point, group = inputs()
        with self.assertRaisesRegex(ValueError, 'missing post-fit'):
            prepare_postfit_mark_observations(
                manifest, fp_by_pgc={}, cut_by_pgc=cut, point_by_recno=point,
                group_by_label=group, richness_center=.5)
        fp[101]['er'] = 0.
        with self.assertRaisesRegex(ValueError, 'invalid post-fit input'):
            prepare_postfit_mark_observations(
                manifest, fp_by_pgc=fp, cut_by_pgc=cut, point_by_recno=point,
                group_by_label=group, richness_center=.5)

    def test_rejects_duplicate_manifest_identity(self):
        manifest, fp, cut, point, group = inputs()
        manifest.append(dict(PGC=101, source_group='g2', point_recno=502))
        with self.assertRaisesRegex(ValueError, 'manifest repeats PGC'):
            prepare_postfit_mark_observations(
                manifest, fp_by_pgc=fp, cut_by_pgc=cut, point_by_recno=point,
                group_by_label=group, richness_center=.5)


if __name__ == '__main__':
    unittest.main()
