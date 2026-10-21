import unittest
import numpy as np
from cf4_r2_affine_force import AffineCorrectedForce


class AffineForceTests(unittest.TestCase):
    def test_anchor_gradient_and_potential_derivative(self):
        h=np.diag([2.,3.]);b=np.array([.4,-.2]);anchor=np.array([.3,.6])
        coarse=lambda q:(.5*q@h@q+b@q,h@q+b)
        fine_gradient=np.array([-.7,1.1]);d=fine_gradient-coarse(anchor)[1]
        force=AffineCorrectedForce(coarse,anchor,d)
        np.testing.assert_allclose(force(anchor)[1],fine_gradient)
        self.assertEqual(force(anchor)[0],coarse(anchor)[0])
        q=np.array([-.8,.2]);direction=np.array([.7,-.9]);eps=1e-5
        fd=(force(q+eps*direction)[0]-force(q-eps*direction)[0])/(2*eps)
        self.assertAlmostEqual(fd,float(force(q)[1]@direction),places=9)
        anchor[:]=0.;d[:]=0.
        np.testing.assert_allclose(force.anchor,[.3,.6])
        self.assertFalse(force.correction.flags.writeable)

    def test_equal_curvature_proxy_matches_fine_force_globally_not_just_anchor(self):
        coarse=lambda q:(.5*float(q@q)+float(np.sum(q)),q+1.)
        fine=lambda q:(.5*float(q@q)-2*float(np.sum(q)),q-2.)
        anchor=np.array([.5,-.3]);force=AffineCorrectedForce(coarse,anchor,fine(anchor)[1]-coarse(anchor)[1])
        constant=force(anchor)[0]-fine(anchor)[0]
        for q in (np.array([2.,-1.]),np.zeros(2)):
            np.testing.assert_allclose(force(q)[1],fine(q)[1])
            self.assertAlmostEqual(force(q)[0]-fine(q)[0],constant)
        with self.assertRaises(ValueError):force(np.zeros(3))


if __name__=='__main__':unittest.main()
