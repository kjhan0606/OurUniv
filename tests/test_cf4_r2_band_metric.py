import unittest
import numpy as np
from cf4_r2_prior_split_hmc import (
    band_scaled_inverse_mass, inverse_laplacian_metric_symbol, FixedSplitMetric,
)


class BandMetricTests(unittest.TestCase):
    def test_only_declared_band_changes_and_input_is_preserved(self):
        c = inverse_laplacian_metric_symbol(20)
        before = c.copy()
        new = band_scaled_inverse_mass(c)
        k = np.fft.fftfreq(20)*20
        k2 = k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2
        mask = (k2 > 16) & (k2 <= 64)
        np.testing.assert_array_equal(c, before)
        np.testing.assert_array_equal(new[~mask], c[~mask])
        np.testing.assert_array_equal(new[mask], 4*c[mask])
        FixedSplitMetric(new, np.eye(2))  # Includes real-field symmetry check.
        self.assertEqual(new[4,0,0], c[4,0,0])
        self.assertEqual(new[8,0,0], 4*c[8,0,0])

    def test_invalid_configuration(self):
        for kwargs in ({'factor':0}, {'factor':np.inf}, {'lower':8,'upper':4}):
            with self.assertRaises(ValueError):
                band_scaled_inverse_mass(np.ones((4,4,4)), **kwargs)

