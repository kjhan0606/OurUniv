import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_volume_target import FreshRawSupport,volume_rule


class RawVolumeTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_volume_rule_normalization_and_moments(self):
        offsets,weights=volume_rule(3.,4)
        self.assertAlmostEqual(float(weights.sum()),1.,places=14)
        np.testing.assert_allclose(weights@offsets,0.,atol=1e-14)
        np.testing.assert_allclose(weights@(offsets**2),np.full(3,3.**2/12),atol=1e-14)

    def test_support_is_rebuilt_when_cell_velocity_changes(self):
        radius=jnp.linspace(.001,400.,4001)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,grid_size=128)
        positions=np.array([[292.,192.,192.],[352.,192.,192.]])
        obs=dict(voxel=np.array([[97,64,64]]),radius=np.array([100.]))
        support=FreshRawSupport(positions,np.ones((2,2)),[0],obs,geometry,
            source_spacing=3.,volume_order=2,block=64)
        def ids(v):
            packs,_=support.build(v,np.zeros(9))
            return set(np.asarray(packs[0]['ids'])[np.asarray(packs[0]['mask'])].tolist())
        v=np.zeros((2,3));self.assertEqual(ids(v),{0})
        v[1,0]=-6000.;self.assertEqual(ids(v),{0,1})
        v[1,0]=0.;self.assertEqual(ids(v),{0})


if __name__=='__main__':unittest.main()
