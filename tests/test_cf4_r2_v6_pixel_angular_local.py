import unittest

import healpy as hp
import numpy as np

from cf4_r2_v6_pixel_angular_geometry import (
    OBSERVER,
    flatten_cell_indices,
    pixel_completeness,
    source_volume_rule,
)
from cf4_r2_ray_cost_profile import ray_box_intervals
from cf4_r2_v6_angular_ray_reference import source_cell_ray_volume_rule


class PixelAngularLocalTests(unittest.TestCase):
    def assert_ray_weights_match_exact_box_volume(self, sample, ijk, nside, order):
        cell_size = 384.0 / 256
        relative = np.asarray(sample["positions"]) - OBSERVER[None, :]
        radii = np.linalg.norm(relative, axis=1)
        directions = relative / radii[:, None]
        directions = directions.reshape(-1, order, 3)
        self.assertTrue(np.allclose(directions, directions[:, :1, :], rtol=0., atol=2e-14))
        directions = directions[:, 0, :]
        low = np.asarray(ijk, dtype=np.float64) * cell_size - 384.0 / 2.0
        high = low + cell_size
        enter, leave, intersects = ray_box_intervals(directions.T, low, high)
        self.assertTrue(np.all(intersects))
        exact_ray_weights = (
            hp.nside2pixarea(nside) * (leave**3 - enter**3)
            / (3.0 * cell_size**3)
        )
        integrated_ray_weights = np.asarray(sample["weights"]).reshape(-1, order).sum(axis=1)
        np.testing.assert_allclose(integrated_ray_weights, exact_ray_weights,
                                   rtol=2e-12, atol=2e-15)

    def test_active_gl2_source_volume_rule_is_normalized_and_cell_local(self):
        offsets, weights = source_volume_rule()
        self.assertEqual(offsets.shape, (8, 3))
        self.assertEqual(weights.shape, (8,))
        self.assertTrue(np.isclose(weights.sum(), 1.0, rtol=0.0, atol=2e-15))
        self.assertTrue(np.all(np.abs(offsets) < 0.75))
        self.assertTrue(np.allclose(offsets.mean(axis=0), 0.0, rtol=0.0, atol=1e-15))

    def test_pixel_completeness_uses_same_rings_pixel_for_both_maps(self):
        nside = 1
        pixels = hp.nside2npix(nside)
        maps = [np.linspace(0.0, 1.0, pixels), np.linspace(1.0, 0.0, pixels)]
        points = np.asarray([OBSERVER + [1.0, 0.0, 0.0],
                             OBSERVER + [0.0, -1.0, 0.0]])
        actual = pixel_completeness(points, maps, np.eye(3), nside=nside)
        pix = hp.vec2pix(nside, [1.0, 0.0], [0.0, -1.0], [0.0, 0.0], nest=False)
        expected = np.stack([m[pix] for m in maps])
        self.assertTrue(np.array_equal(actual, expected))

    def test_pixel_completeness_rejects_observer_coincident_direction(self):
        maps = [np.ones(12), np.ones(12)]
        with self.assertRaisesRegex(ValueError, "observer-coincident"):
            pixel_completeness(np.asarray([OBSERVER]), maps, np.eye(3), nside=1)

    def test_flattened_ids_index_their_declared_cells_not_candidate_ranks(self):
        ijk = np.asarray([[2, 1, 4], [0, 0, 1]], dtype=np.int32)
        flat_ids = flatten_cell_indices(ijk, n=5)
        index_coded_field = np.arange(5**3, dtype=np.int64).reshape(5, 5, 5)
        self.assertTrue(np.array_equal(flat_ids, np.asarray([59, 1])))
        self.assertTrue(np.array_equal(
            index_coded_field.reshape(-1)[flat_ids],
            index_coded_field[ijk[:, 0], ijk[:, 1], ijk[:, 2]]))

    def test_known_n256_control_has_the_full_grid_flattened_id(self):
        ijk = np.asarray([[126, 124, 139]], dtype=np.int32)
        self.assertEqual(flatten_cell_indices(ijk, n=256).tolist(), [8_289_419])

    def test_flattened_ids_reject_out_of_grid_cells(self):
        with self.assertRaisesRegex(ValueError, "outside the grid"):
            flatten_cell_indices(np.asarray([[256, 0, 0]], dtype=np.int32), n=256)

    def test_exact_ray_box_intersection_and_source_volume_weights(self):
        diagonal = np.asarray([[1.], [1.], [1.]]) / np.sqrt(3.)
        enter, leave, valid = ray_box_intervals(
            diagonal, np.asarray([1., 1., 1.]), np.asarray([2., 2., 2.]))
        self.assertEqual(valid.tolist(), [True])
        self.assertTrue(np.isclose(enter[0], np.sqrt(3.)))
        self.assertTrue(np.isclose(leave[0], 2. * np.sqrt(3.)))

        # Pixel-centre ray weights remain physical dOmega r^2 dr / Vcell
        # and are not forced to sum to one.
        ijk = np.asarray([168, 128, 128], dtype=np.int32)
        coarse = source_cell_ray_volume_rule(ijk, 128, 2, np.eye(3))
        coarse_r4 = source_cell_ray_volume_rule(ijk, 128, 4, np.eye(3))
        fine = source_cell_ray_volume_rule(ijk, 256, 2, np.eye(3))
        self.assertEqual(coarse["positions"].shape[0], 2 * coarse["intersecting_rays"])
        self.assertTrue(np.all(coarse["weights"] > 0.))
        self.assert_ray_weights_match_exact_box_volume(coarse, ijk, 128, 2)
        self.assertGreater(fine["intersecting_rays"], coarse["intersecting_rays"])
        self.assert_ray_weights_match_exact_box_volume(fine, ijk, 256, 2)
        self.assertTrue(np.isclose(coarse["volume_weight_sum"], coarse_r4["volume_weight_sum"],
                                   rtol=0., atol=2e-14))
