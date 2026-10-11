import unittest
import numpy as np
from cf4_r2_field_readout import present_field_readout


class FieldReadoutTests(unittest.TestCase):
    def test_mass_weighted_velocity_and_empty_cells(self):
        rho = np.zeros((2,2,2))
        rho[0,0,0], rho[1,1,1] = 1., 3.
        velocity = np.full((3,2,2,2), 1000.)
        velocity[:,0,0,0] = 0.
        velocity[:,1,1,1] = 4.
        result = present_field_readout(rho, velocity)
        self.assertEqual(result['density_mean'], .5)
        self.assertEqual(result['occupied_fraction'], .25)
        np.testing.assert_allclose(result['mass_weighted_bulk_velocity_km_s'], 3.)
        np.testing.assert_allclose(result['mass_weighted_spatial_velocity_rms_km_s'], np.sqrt(3.))
        self.assertFalse(result['posterior_uncertainty'])

    def test_invalid_fields(self):
        for rho,velocity in [(np.zeros((2,2,2)),np.zeros((3,2,2,2))),
                             (-np.ones((2,2,2)),np.zeros((3,2,2,2))),
                             (np.ones((2,2,2)),np.zeros((2,2,2,3)))]:
            with self.assertRaises(ValueError):
                present_field_readout(rho,velocity)
