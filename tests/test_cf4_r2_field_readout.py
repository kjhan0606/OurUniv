import unittest
import numpy as np
from cf4_r2_field_readout import present_field_readout


class FieldReadoutTests(unittest.TestCase):
    def test_fixed_observer_spheres(self):
        rho = np.ones((8,8,8))
        velocity = np.full((3,8,8,8), 7.)
        rho[3,3,3] = 3.
        velocity[:,3,3,3] = 11.
        result = present_field_readout(rho, velocity, box=384.)
        axis = (np.arange(8)+.5)*48.-192.
        r2 = axis[:,None,None]**2+axis[None,:,None]**2+axis[None,None,:]**2
        for radius in (50,100):
            count = np.count_nonzero(r2 <= radius**2)
            sphere = result['observer_spheres_cMpc_h'][str(radius)]
            self.assertAlmostEqual(sphere['density_mean'], (count+2)/count)
            np.testing.assert_allclose(sphere['mass_weighted_velocity_km_s'],
                                       (7*(count-1)+33)/(count+2))

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
