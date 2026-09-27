import numpy as np
import unittest

from cf4_r2_observed_magnitude_transfer import observed_magnitude_transfer
from cf4_twompp_joint_information_budget_pilot_v1 import schechter_fraction


class ObservedMagnitudeTransferTests(unittest.TestCase):
    def test_no_rsd_matches_archived_six_population_radial_fraction(self):
        edges = (-25., -23.6666666666667, -22.3333333333333, -21.)
        distance_h = np.array([15., 30., 90., 180.])
        mu = 5*np.log10(distance_h) + 25
        transfer = observed_magnitude_transfer(mu, mu)
        self.assertEqual(transfer.shape, (6, 5, 4))
        self.assertTrue(np.all(transfer[:, 0] == 0))
        self.assertTrue(np.all(transfer[:, 4] == 0))
        for a in range(2):
            for b in range(3):
                p = 3*a+b
                expect = schechter_fraction(distance_h,
                    None if a == 0 else 11.5, 11.5 if a == 0 else 12.5,
                    edges[b], edges[b+1], -23.28, -.94)
                np.testing.assert_allclose(transfer[p, b+1], expect, rtol=1e-12, atol=1e-14)
                other = [j for j in range(5) if j != b+1]
                self.assertTrue(np.all(transfer[p, other] == 0))


    def test_rsd_migrates_luminosity_labels_and_retains_probability_mass(self):
        true_h, observed_h = 30., 33.
        transfer = observed_magnitude_transfer(
            5*np.log10(true_h)+25, 5*np.log10(observed_h)+25)
        self.assertEqual(transfer.shape, (6, 5))
        self.assertTrue(np.all((transfer >= 0) & (transfer <= 1)))
        self.assertTrue(np.all(transfer.sum(axis=0) <= 1+1e-14))
        # For s>r the inferred absolute magnitude is brighter: faint-side
        # intrinsic galaxies can enter the observed -25<M_h<-21 range.
        self.assertGreater(transfer[:, 4].sum(), 0)
        # A true middle-bin galaxy can cross an observed absolute-magnitude edge.
        self.assertGreater(transfer[:, 2].sum(), transfer[1, 2] + transfer[4, 2])


    def test_nonfinite_modulus_rejected(self):
        with self.assertRaises(ValueError):
            observed_magnitude_transfer(np.nan, 32.)
