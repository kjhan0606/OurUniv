import unittest
import numpy as np
from cf4_r2_native_mock import distance_tables,place_native_halves,observe,aggregate


class NativeMockTests(unittest.TestCase):
    def test_native_halves_are_disjoint_and_translate_internal_separations(self):
        x=np.array([[0.,1.,2.],[1.,1.,2.],[37.5,1.,2.],[38.5,1.,2.]])
        y=place_native_halves(x)
        np.testing.assert_allclose(y[1]-y[0],x[1]-x[0])
        np.testing.assert_allclose(y[3]-y[2],x[3]-x[2])
        self.assertEqual(y[2,0]-y[0,0],121.5)

    def test_zero_velocity_recovers_magnitude_and_counts_once(self):
        r,z,m=distance_tables(.3089)
        p=np.array([[172.,192.,192.],[292.,192.,192.]])
        out=observe(p,np.zeros_like(p),np.array([-23.1,-23.1]),r,z,m)
        np.testing.assert_allclose(out['observed_absolute_K'],[-23.1,-23.1],atol=1e-12)
        np.testing.assert_allclose(out['positions'],p)
        self.assertEqual(int(out['counts'].sum()),int(out['selected'].sum()))
        self.assertEqual(out['population'][0],1)

    def test_velocity_shift_and_redshift_K_correction(self):
        r,z,m=distance_tables(.3089)
        p=np.array([[292.,192.,192.]])
        v=np.array([[300.,0.,0.]])
        out=observe(p,v,np.array([-23.1]),r,z,m)
        self.assertAlmostEqual(out['observed_radius'][0],103.)
        z0,z1=np.interp([100.,103.],r,z)
        m0,m1=np.interp([100.,103.],r,m)
        correction=1.16*2.9*(z1-z0)-1.6*np.log10((1+z1)/(1+z0))
        self.assertAlmostEqual(out['observed_absolute_K'][0],-23.1+m0+correction-m1)

    def test_population_specific_aggregation_and_exposure(self):
        n=128**3
        exposure=np.zeros(n,dtype=bool);exposure[2]=True
        rbin=np.zeros(n,dtype=int);rbin[2]=7
        table=aggregate(np.array([2,n+2,n+3]),np.array([3,4,9]),exposure,rbin)
        self.assertEqual(table[0,7],3.)
        self.assertEqual(table[1,7],4.)
        self.assertEqual(table.sum(),7.)


if __name__=='__main__':unittest.main()
