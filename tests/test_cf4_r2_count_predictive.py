"""Forward-map wiring regression; no survey or held-out values are loaded."""
import unittest
from unittest import mock

import numpy as np

import cf4_r2_count_predictive as predictive


class CountPredictiveForwardTests(unittest.TestCase):
    def test_matches_frozen_resolution_target_count_wiring(self):
        n = 4
        source = {
            'positions': np.arange(n ** 3 * 3, dtype=np.float32).reshape(n ** 3, 3),
            'angular': np.ones((2, n ** 3), dtype=np.float32),
        }
        geometry = {'box_size_cMpc_h': 16.0, 'grid_size': 4}
        density = np.ones((n, n, n), dtype=np.float32)
        velocity = np.zeros((3, n, n, n), dtype=np.float32)
        tracer = np.zeros(9, dtype=np.float32)
        mass_density = np.arange(n ** 3, dtype=np.float32)
        cell_velocity = np.arange(3 * n ** 3, dtype=np.float32).reshape(3, n, n, n)
        marker = mock.Mock()
        marker.shape = (6, 4, 4, 4)

        with (mock.patch.object(predictive, 'native_mass_momentum_to_count_cells',
                                return_value=(mass_density, cell_velocity)) as deposit,
              mock.patch.object(predictive, 'tracer_masses',
                                return_value=np.full((5, n ** 3), 3.0)) as masses,
              mock.patch.object(predictive, 'tracer_geometry',
                                return_value={'grid_size': 4, 'sentinel': 17}) as tracer_geom,
              mock.patch.object(predictive, 'predict_chunked_volume_intensity',
                                return_value=marker) as forward):
            result = predictive.count_intensity_from_field(
                density, velocity, tracer, source, geometry, volume_order=2,
                source_chunk_size=11)

        self.assertIs(result, marker)
        deposit.assert_called_once()
        masses.assert_called_once()
        tracer_geom.assert_called_once()
        args, kwargs = forward.call_args
        np.testing.assert_array_equal(np.asarray(args[0]), source['positions'])
        expected_cv = np.moveaxis(cell_velocity, 0, -1).reshape(-1, 3)
        np.testing.assert_array_equal(np.asarray(args[1]), expected_cv)
        np.testing.assert_allclose(np.asarray(args[2]), 3.0 * (4.0 / 3.0) ** 3)
        np.testing.assert_array_equal(np.asarray(args[3]), source['angular'])
        self.assertEqual(kwargs, dict(source_chunk_size=11, source_spacing=4.0,
                                      volume_order=2, grid_size=4, sentinel=17,
                                      order=4, segments=8))

    def test_rejects_target_subpopulation(self):
        n = 2
        source = {'positions': np.zeros((n ** 3, 3)), 'angular': np.ones((2, n ** 3))}
        density = np.ones((n, n, n))
        velocity = np.zeros((3, n, n, n))
        tracer = np.zeros(9)
        partial_geometry = {'box_size_cMpc_h': 16.0, 'grid_size': 4,
                            'target_population': 1}
        with self.assertRaisesRegex(ValueError, 'complete six-population'):
            predictive.count_intensity_from_field(
                density, velocity, tracer, source, partial_geometry, volume_order=2)

    def test_rejects_incomplete_forward_grid(self):
        n = 2
        source = {'positions': np.zeros((n ** 3, 3)), 'angular': np.ones((2, n ** 3))}
        geometry = {'box_size_cMpc_h': 16.0, 'grid_size': 4}
        marker = mock.Mock()
        marker.shape = (6, 3, 4, 4)
        with (mock.patch.object(predictive, 'native_mass_momentum_to_count_cells',
                                return_value=(np.ones(n ** 3), np.zeros((3, n, n, n)))),
              mock.patch.object(predictive, 'tracer_masses',
                                return_value=np.ones((5, n ** 3))),
              mock.patch.object(predictive, 'tracer_geometry',
                                return_value={'grid_size': 4}),
              mock.patch.object(predictive, 'predict_chunked_volume_intensity',
                                return_value=marker)):
            with self.assertRaisesRegex(ValueError, 'complete six-population grid'):
                predictive.count_intensity_from_field(
                    np.ones((n, n, n)), np.zeros((3, n, n, n)), np.zeros(9),
                    source, geometry, volume_order=2)

    def test_rejects_misaligned_field_and_source_before_forward_call(self):
        source = {'positions': np.zeros((63, 3)), 'angular': np.ones((2, 63))}
        geometry = {'box_size_cMpc_h': 16.0, 'grid_size': 4}
        with mock.patch.object(predictive, 'predict_chunked_volume_intensity') as forward:
            with self.assertRaisesRegex(ValueError, 'source geometry'):
                predictive.count_intensity_from_field(
                    np.ones((4, 4, 4)), np.zeros((3, 4, 4, 4)), np.zeros(9),
                    source, geometry, volume_order=2)
            forward.assert_not_called()

    def test_rejects_nonfinite_or_invalid_box(self):
        n = 2
        source = {'positions': np.zeros((n ** 3, 3)), 'angular': np.ones((2, n ** 3))}
        with self.assertRaisesRegex(ValueError, 'box size'):
            predictive.count_intensity_from_field(
                np.ones((n, n, n)), np.zeros((3, n, n, n)), np.zeros(9),
                source, {'box_size_cMpc_h': float('nan'), 'grid_size': 4},
                volume_order=2)


if __name__ == '__main__':
    unittest.main()
