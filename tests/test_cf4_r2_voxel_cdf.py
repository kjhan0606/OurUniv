import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr
from cf4_r2_shell_cdf_count import ray_voxel_interval,conditional_los_sigma

jax.config.update('jax_enable_x64',True)


class VoxelRayTests(unittest.TestCase):
    def test_conditional_width_projection_and_zero_variance_identity(self):
        d=jnp.array([[1.,0.,0.],[0.,1.,0.]])
        variance=jnp.array([[9.,16.,25.],[9.,16.,25.]])
        np.testing.assert_allclose(conditional_los_sigma(d,2.,variance,.5),
            np.sqrt([4.+.25*9.,4.+.25*16.]))
        np.testing.assert_allclose(conditional_los_sigma(d,2.,jnp.zeros_like(d),1.),[2.,2.])

    def test_conditional_scale_derivative(self):
        d=jnp.array([[1.,0.,0.]])
        variance=jnp.array([[9.,16.,25.]])
        value=lambda s:conditional_los_sigma(d,2.,variance,s)[0]
        eps=1e-5;x=.5
        self.assertAlmostEqual(float(jax.grad(value)(x)),float((value(x+eps)-value(x-eps))/(2*eps)),places=8)

    def test_zero_direction_has_finite_inactive_interval(self):
        lo,hi,active=ray_voxel_interval(jnp.zeros((1,3)),jnp.ones(3),
            (1,1,1),4,4.,jnp.zeros(3))
        self.assertFalse(active[0]);self.assertTrue(np.isfinite([lo[0],hi[0]]).all())

    def test_parallel_ray_and_half_open_face(self):
        d=jnp.array([[1.,0.,0.]])
        lo,hi,active=ray_voxel_interval(d,jnp.array([0.,1.,1.]),(1,1,1),4,4.,jnp.zeros(3))
        np.testing.assert_allclose([lo[0],hi[0]],[1.,2.]);self.assertTrue(active[0])
        _,_,outside=ray_voxel_interval(d,jnp.array([0.,1.,1.]),(1,0,1),4,4.,jnp.zeros(3))
        self.assertFalse(outside[0])

    def test_negative_direction_and_periodic_image(self):
        lo,hi,active=ray_voxel_interval(jnp.array([[-1.,0.,0.]]),
            jnp.array([2.,1.5,1.5]),(3,1,1),4,4.,jnp.array([-4.,0.,0.]))
        np.testing.assert_allclose([lo[0],hi[0]],[2.,3.]);self.assertTrue(active[0])

    def test_cell_partition_conserves_gaussian_measure(self):
        total=0.
        for i in range(4):
            lo,hi,active=ray_voxel_interval(jnp.array([[1.,0.,0.]]),
                jnp.array([0.,1.5,1.5]),(i,1,1),4,4.,jnp.zeros(3))
            self.assertTrue(active[0]);total+=ndtr((float(hi[0])-2)/.3)-ndtr((float(lo[0])-2)/.3)
        self.assertAlmostEqual(total,ndtr(2/.3)-ndtr(-2/.3),places=14)

    def test_direction_derivative_matches_finite_difference(self):
        def value(x):
            d=jnp.array([[x,.1,.2]])
            lo,hi,_=ray_voxel_interval(d,jnp.array([0.,1.,1.]),(2,1,1),4,4.,jnp.zeros(3))
            return hi[0]-lo[0]
        x=1.;eps=1e-5
        self.assertAlmostEqual(float(jax.grad(value)(x)),float((value(x+eps)-value(x-eps))/(2*eps)),places=8)


if __name__=='__main__':unittest.main()
