import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_observed_ray import source_ray_intervals,observed_ray_components
from cf4_r2_marked_tracer_jax import source_mark_transfer


class ObservedRayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_forward_ray_cube_partition(self):
        for direction in ([1.,0.,0.],[1.,-1.,1.],[-.2,.7,-.4]):
            direction=jnp.asarray(direction);direction/=jnp.linalg.norm(direction)
            lo,hi,ids=source_ray_intervals(direction,jnp.full(3,24.),4,48.)
            qexit=24/float(jnp.max(jnp.abs(direction)))
            self.assertAlmostEqual(float(jnp.sum(hi-lo)),qexit,places=12)
            self.assertGreaterEqual(float(jnp.min(lo)),0.)
            self.assertTrue(np.all(np.asarray(hi>=lo)))
            self.assertTrue(np.all(np.asarray((ids>=0)&(ids<64))))

    def test_uniform_ray_gaussian_second_moment(self):
        radius=11.;direction=jnp.array([1.,1.,1.])/jnp.sqrt(3.)
        table=jnp.array([.001,100.])
        g=dict(observer=jnp.full(3,24.),box_size_cMpc_h=48.,hubble_km_s_Mpc=74.6,little_h=.746,
            radius_table_cMpc_h=table,modulus_table_h=jnp.array([32.,32.]),redshift_table=jnp.zeros(2),
            mstar=-23.28,alpha=-.94,finite_reference_interval=(-25.,-21.))
        velocity=jnp.zeros((64,3));variance=jnp.full((64,3),10000.)
        masses=jnp.ones((5,64));angular=jnp.ones((2,64))
        closure=dict(core_sigma_km_s=20.,dispersion_scale=.3,broad_fraction=.5)
        *_,weights=observed_ray_components(direction,radius,velocity,variance,masses,angular,1,g,closure,
            source_grid=4,order=32)
        transfer=source_mark_transfer(jnp.array([32.]),jnp.array([32.]),jnp.array([0.]),jnp.array([0.]),
            mstar=-23.28,alpha=-.94,finite_reference_interval=(-25.,-21.))[1].sum()
        mean_square=radius**2+.01**2*(20.**2+.5*.3**2*10000.)
        np.testing.assert_allclose(weights.sum(),transfer*mean_square/12.**3,rtol=3e-5)


if __name__=='__main__':unittest.main()
