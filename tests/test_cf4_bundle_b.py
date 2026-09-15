import json
from pathlib import Path
import unittest
import numpy as np
from cf4_lg_observation_contract import basis,solar_reference,predict,log_likelihood,approximate_covariance
from cf4_bundle_b_environment import aperture

ROOT=Path(__file__).resolve().parents[1]


class BundleBTest(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/"config/cf4_lg_observation_contract_v1.json").read_text())
        self.sun,self.vsun=solar_reference(self.c)
        self.cat={"frame":self.c["frame"],"resolved_halos":True,
                  "MW":{"position_kpc":[0,0,0],"peculiar_velocity_km_s":[0,0,0]}}
        for name in self.c["galaxy_order"]:
            x=self.c["measurements"][name]
            modulus,vlos,pmra,pmdec=x["value"]
            distance=10**((modulus-10)/5)
            rotation=basis(x["ra_deg"],x["dec_deg"])
            pos=self.sun+distance*rotation[0]
            vel=self.vsun+rotation.T@np.array([vlos,pmra*4.74047*distance/1000,pmdec*4.74047*distance/1000])-.0746*pos
            self.cat[name]={"position_kpc":pos,"peculiar_velocity_km_s":vel}

    def forward(self):
        return predict(self.cat,self.c,h=.746,solar_position_kpc=self.sun,solar_velocity_km_s=self.vsun)

    def test_roundtrip_units_frame_hubble(self):
        expected=np.concatenate([self.c["measurements"][g]["value"] for g in self.c["galaxy_order"]])
        np.testing.assert_allclose(self.forward(),expected,atol=1e-10)
        self.cat["M31"]["peculiar_velocity_km_s"]+=10*basis(10.68,41.27)[0]
        self.assertAlmostEqual(self.forward()[1]-expected[1],10.,places=9)

    def test_required_identity_and_resolution(self):
        self.cat["resolved_halos"]=False
        with self.assertRaises(ValueError): self.forward()
        self.cat["resolved_halos"]=True
        del self.cat["M33"]
        with self.assertRaises(KeyError): self.forward()

    def test_covariance(self):
        pred=self.forward()
        with self.assertRaises(ValueError): log_likelihood(pred,self.c)
        covariance=approximate_covariance(self.c)
        self.assertGreater(covariance[0,4],0)
        base=log_likelihood(pred,self.c,observation_covariance=covariance)
        shifted=pred+np.linalg.cholesky(covariance)[:,2]
        self.assertAlmostEqual(log_likelihood(shifted,self.c,observation_covariance=covariance)-base,-.5,places=9)
        bad=covariance.copy(); bad[0,0]=-1
        with self.assertRaises(np.linalg.LinAlgError): log_likelihood(pred,self.c,observation_covariance=bad)

    def test_native_aperture_constant_and_boundary(self):
        weight,coverage=aperture(np.array([150.,192.,192.]),24,np.ones((32,)*3,bool))
        self.assertAlmostEqual(weight@np.ones(32**3),1.)
        self.assertGreater(coverage["support_fraction"],.99)
        _,edge=aperture(np.array([347.,192.,192.]),31,np.ones((32,)*3,bool))
        self.assertLess(edge["radial_box_coverage"],1.)
        _,empty=aperture(np.array([150.,192.,192.]),24,np.zeros((32,)*3,bool))
        self.assertEqual(empty["support_fraction"],0.)


if __name__=="__main__": unittest.main()
