import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_volume_target import (
    FreshRawSupport,tracer_geometry,tracer_masses,volume_rule,
)
from cf4_r2_resolution_target import linked_point_conditioning_radius


class RawVolumeTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_chunking_retains_candidates_beyond_workspace_ceiling(self):
        geometry=dict(box_size_cMpc_h=384.,grid_size=128,
                      little_h=.746,hubble_km_s_Mpc=74.6)
        positions=np.tile([292.,192.,192.],(65,1))
        obs=dict(voxel=np.array([[97,64,64]]),radius=np.array([100.]))
        support=FreshRawSupport(positions,np.ones((2,65)),[0],obs,geometry,
            source_spacing=3.,volume_order=1,block=64,max_support_cells=64)
        # Isolate packing from the mark law: every bin/node has positive weight.
        support.weight=lambda vel,tr,pos,sky,voxel,radius,pop:np.ones((5,len(pos)))
        with self.assertRaises(MemoryError):
            support.build(np.zeros((65,3)),np.zeros(9))
        support.support_chunk_cells=32
        packs,info=support.build(np.zeros((65,3)),np.zeros(9))
        selected=np.asarray(packs[0]['ids'])[np.asarray(packs[0]['mask'])]
        np.testing.assert_array_equal(np.bincount(selected,minlength=65),np.full(65,5))
        self.assertEqual(info['largest_candidate_cells'],65)
        self.assertEqual(info['components'],325)
        self.assertEqual(info['max_cells'],64)

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
        native_tree=support.tree
        class RowOnlyTree:
            def query_ball_point(self,centre,*args,**kwargs):
                # A batched query would recreate the host-memory regression.
                np.testing.assert_equal(np.asarray(centre).shape,(3,))
                return native_tree.query_ball_point(centre,*args,**kwargs)
        support.tree=RowOnlyTree()
        def ids(v):
            support.support_chunk_cells=None
            packs,_=support.build(v,np.zeros(9))
            reference=packs[0]
            support.support_chunk_cells=1
            chunked,info=support.build(v,np.zeros(9))
            def entries(pack):
                mask=np.asarray(pack['mask'])
                return sorted(zip(*(np.asarray(pack[k])[mask].tolist()
                                    for k in ('ids','node','bin','row'))))
            self.assertEqual(entries(reference),entries(chunked[0]))
            self.assertEqual(info['support_chunk_cells'],1)
            return set(np.asarray(packs[0]['ids'])[np.asarray(packs[0]['mask'])].tolist())
        v=np.zeros((2,3));self.assertEqual(ids(v),{0})
        v[1,0]=-6000.;self.assertEqual(ids(v),{0,1})
        v[1,0]=0.;self.assertEqual(ids(v),{0})

    def test_support_uses_linked_point_radius_from_observation(self):
        radius=jnp.linspace(.001,400.,4001)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,grid_size=128)
        observation=dict(voxel=np.array([[97,64,64]]),radius=np.array([100.]),
            source_conditioning_radius_cMpc_h=np.array([50.]))
        support=FreshRawSupport(np.array([[292.,192.,192.]]),np.ones((2,1)),[0],
            observation,geometry,source_spacing=3.,volume_order=2,block=64)
        np.testing.assert_array_equal(support.source_conditioning_radius,[50.])

    def test_resolution_target_requires_aligned_linked_point_radii(self):
        observation=dict(radius=np.array([100.,120.]),
            source_conditioning_radius_cMpc_h=np.array([50.,60.]))
        np.testing.assert_array_equal(linked_point_conditioning_radius(observation),[50.,60.])
        with self.assertRaisesRegex(ValueError,'are required'):
            linked_point_conditioning_radius(dict(radius=np.array([100.])))
        with self.assertRaisesRegex(ValueError,'align and be positive'):
            linked_point_conditioning_radius(dict(radius=np.array([100.,120.]),
                source_conditioning_radius_cMpc_h=np.array([50.])))
        with self.assertRaisesRegex(ValueError,'align and be positive'):
            linked_point_conditioning_radius(dict(radius=np.array([100.]),
                source_conditioning_radius_cMpc_h=np.array([0.])))

    def test_active_lf_coordinate_crosses_minus_one_with_finite_selected_rate(self):
        tracer=jnp.zeros(9).at[7].set(-1.)
        geometry=tracer_geometry(tracer,{})
        self.assertAlmostEqual(float(geometry['alpha']),-1.5,places=12)
        self.assertEqual(geometry['finite_reference_interval'],(-25.,-21.))
        density=jnp.array([.4,.8,1.2,1.6])
        masses=tracer_masses(density,tracer)
        self.assertTrue(np.isfinite(np.asarray(masses)).all())
        self.assertEqual(masses.shape,(5,4))


if __name__=='__main__':unittest.main()
