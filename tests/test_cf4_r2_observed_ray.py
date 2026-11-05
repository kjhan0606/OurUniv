import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_observed_ray import (source_ray_intervals,observed_ray_components,
    extend_flat_distance_tables)
from cf4_r2_marked_tracer_jax import source_mark_transfer
from cf4_r2_raw_live_mark import chunk_log_terms,POPULATION_ORIGIN


class ObservedRayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_forward_ray_cube_partition(self):
        for direction in ([1.,0.,0.],[1.,-1.,1.],[-.2,.7,-.4]):
            direction=jnp.asarray(direction);direction/=jnp.linalg.norm(direction)
            lo,hi,ids=source_ray_intervals(direction,jnp.full(3,24.),4,48.)
            horizon=48/float(jnp.max(jnp.abs(direction)))
            self.assertAlmostEqual(float(jnp.sum(hi-lo)),horizon,places=12)
            self.assertGreaterEqual(float(jnp.min(lo)),0.)
            self.assertTrue(np.all(np.asarray(hi>=lo)))
            self.assertTrue(np.all(np.asarray((ids>=0)&(ids<64))))
        lo,hi,ids=source_ray_intervals(jnp.array([1.,0.,0.]),jnp.full(3,24.),4,48.)
        self.assertTrue(np.any(np.asarray(lo)<24.)&np.any(np.asarray(hi)>24.))
        active=np.asarray(hi)>np.asarray(lo)
        np.testing.assert_array_equal(np.asarray(ids)[active],np.array([42,58,10,26]))

    def test_flat_distance_extension_preserves_edge_and_monotonicity(self):
        radius=np.linspace(.001,192.,20001)
        redshift=.00032*radius+7e-8*radius**2
        modulus=5*np.log10(radius*(1.+redshift))+25.
        g=dict(radius_table_cMpc_h=jnp.asarray(radius),
            redshift_table=jnp.asarray(redshift),modulus_table_h=jnp.asarray(modulus))
        extended,info=extend_flat_distance_tables(g,225.)
        new_radius=np.asarray(extended['radius_table_cMpc_h'])
        new_z=np.asarray(extended['redshift_table'])
        new_mu=np.asarray(extended['modulus_table_h'])
        self.assertGreaterEqual(new_radius[-1],225.)
        self.assertTrue(np.all(np.diff(new_z)>0))
        np.testing.assert_allclose(new_mu[-1],
            5*np.log10(new_radius[-1]*(1.+new_z[-1]))+25.,atol=2e-12)
        self.assertLess(info['luminosity_distance_edge_mismatch_mag'],2e-4)

    def test_uniform_ray_gaussian_second_moment(self):
        radius=11.;direction=jnp.array([1.,1.,1.])/jnp.sqrt(3.)
        table=jnp.array([.001,100.])
        g=dict(observer=jnp.full(3,24.),box_size_cMpc_h=48.,hubble_km_s_Mpc=74.6,little_h=.746,
            radius_table_cMpc_h=table,modulus_table_h=jnp.array([32.,32.]),redshift_table=jnp.zeros(2),
            mstar=-23.28,alpha=-.94,finite_reference_interval=(-25.,-21.))
        velocity=jnp.zeros((64,3));variance=jnp.ones((64,3))
        masses=jnp.ones((5,64));angular=jnp.ones((2,64))
        closure=dict(core_sigma_km_s=.1,dispersion_scale=.01,broad_fraction=.5)
        *_,weights=observed_ray_components(direction,radius,velocity,variance,masses,angular,1,g,closure,
            source_grid=4,order=32)
        *_,split_weights=observed_ray_components(direction,radius,velocity,variance,masses,angular,1,g,closure,
            source_grid=4,order=16,segments=2)
        transfer=source_mark_transfer(jnp.array([32.]),jnp.array([32.]),jnp.array([0.]),jnp.array([0.]),
            mstar=-23.28,alpha=-.94,finite_reference_interval=(-25.,-21.))[1].sum()
        mean_square=radius**2+.01**2*(.1**2+.5*.01**2)
        np.testing.assert_allclose(weights.sum(),transfer*mean_square/12.**3,rtol=3e-5)
        np.testing.assert_allclose(split_weights.sum(),transfer*mean_square/12.**3,rtol=3e-5)

    def test_periodic_uniform_fp_ray_is_direction_invariant_and_keeps_q(self):
        radius=26.;n=4;box=48.
        table=jnp.linspace(.001,100.,4001)
        g=dict(observer=jnp.full(3,24.),box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,
            radius_table_cMpc_h=table,modulus_table_h=5*jnp.log10(table)+25.,
            redshift_table=table/3000.,grid_size=n,mstar=-23.28,alpha=-.94,
            finite_reference_interval=(-25.,-21.))
        velocity=jnp.zeros((n**3,3));variance=jnp.full((n**3,3),10000.)
        masses=jnp.ones((5,n**3));angular=jnp.stack((jnp.linspace(.05,1.,n**3),
            jnp.linspace(1.,.05,n**3)))
        closure=dict(core_sigma_km_s=20.,dispersion_scale=.3,broad_fraction=.5)
        obs=dict(voxel=jnp.array([2,2,2]),radius=jnp.asarray(radius),dz=jnp.asarray(1.),
            ksmag=jnp.asarray(9.5),x=jnp.array([.3,2.2,2.7]),
            error_covariance=jnp.eye(3)*.002,richness=jnp.asarray(0.),
            cut_lower=jnp.array([2.,1.9]),cut_upper=jnp.array([4.,2.5]))
        params=jnp.asarray(POPULATION_ORIGIN)

        def score(direction,override_radius=None,logscale=0.):
            current=dict(closure,dispersion_scale=.3*jnp.exp(logscale))
            pos,vel,mass,sky,q,weights=observed_ray_components(direction,radius,velocity,
                variance,masses,angular,1,g,current,source_grid=n,order=16)
            radial=q if override_radius is None else override_radius
            a,b=chunk_log_terms(params,pos,vel,mass,sky,obs,population=1,geometry=g,
                cut_order=16,source_velocity_variances_km2_s2=jnp.full_like(pos,10000.),
                velocity_closure=current,radial_source_mass=weights,
                source_radius_cMpc_h=radial)
            return a-b,q,pos

        axis=jnp.array([1.,0.,0.]);diagonal=jnp.ones(3)/jnp.sqrt(3.)
        axis_value,q,pos=score(axis)
        diagonal_value,_,_=score(diagonal)
        np.testing.assert_allclose(float(axis_value),float(diagonal_value),rtol=0,atol=2e-8)
        face=box/2.
        wrapped=np.asarray(q)>face
        self.assertTrue(np.any(wrapped))
        min_image=(pos-jnp.full(3,24.)+box/2)%box-box/2
        wrong=jnp.linalg.norm(min_image,axis=1)
        wrong_value,_,_=score(axis,wrong)
        self.assertGreater(abs(float(axis_value-wrong_value)),1e-4)
        f=lambda scale:score(axis,logscale=scale)[0]
        grad=float(jax.jit(jax.grad(f))(jnp.asarray(0.)))
        eps=1e-4
        finite_difference=float((f(eps)-f(-eps))/(2*eps))
        np.testing.assert_allclose(grad,finite_difference,rtol=2e-5,atol=1e-9)


if __name__=='__main__':unittest.main()
