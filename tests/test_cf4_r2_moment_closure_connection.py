import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_native_to_count_cells import native_moments_to_count_cells
from cf4_r2_shell_cdf_count import predict_source_volume_intensity
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity


class MomentClosureConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_additive_moment_conservation_and_variance_pullback(self):
        rho=jnp.arange(64,dtype=float).reshape(4,4,4)/32+.1
        mean=jnp.stack((rho*12.,rho*-7.,rho*3.))
        var=jnp.stack((rho*100.,rho*200.,rho*300.))
        mass,velocity,variance=native_moments_to_count_cells(rho,mean,var,48.)
        np.testing.assert_allclose(mass.sum(),rho.sum(),rtol=1e-13)
        np.testing.assert_allclose((mass[None]*velocity).sum(axis=(1,2,3)),
            (rho[None]*mean).sum(axis=(1,2,3)),rtol=1e-13)
        np.testing.assert_allclose((mass[None]*(variance+velocity**2)).sum(axis=(1,2,3)),
            (rho[None]*(var+mean**2)).sum(axis=(1,2,3)),rtol=1e-13)
        zero=jnp.zeros_like(var)
        _,_,between=native_moments_to_count_cells(rho,mean,zero,48.)
        self.assertGreater(float(between.sum()),0.)
        def score(t):return native_moments_to_count_cells(rho,mean,var*jnp.exp(t),48.)[2].sum()
        grad=jax.grad(score)(0.);eps=1e-4
        np.testing.assert_allclose(grad,(score(eps)-score(-eps))/(2*eps),rtol=1e-7)

    def test_chunked_variance_slicing_padding_and_gradient(self):
        r=jnp.linspace(.001,48.,1001)
        positions=jnp.array([[36.4,24.8,24.9],[37.,24.9,24.8],[35.8,24.7,24.9]])
        velocities=jnp.zeros_like(positions);mass=jnp.ones((5,3));sky=jnp.ones((2,3))
        var=jnp.array([[2000.,3000.,4000.],[1000.,2000.,3000.],[3000.,4000.,5000.]])
        g=dict(observer=jnp.full(3,24.),box_size_cMpc_h=48.,hubble_km_s_Mpc=74.6,little_h=.746,
            radius_table_cMpc_h=r,modulus_table_h=5*jnp.log10(r)+25.,redshift_table=r/3000.,
            grid_size=8,sigma_los_km_s=20.,radial_min_cMpc_h=5.,radial_max_cMpc_h=20.,
            finite_reference_interval=(-25.,-21.),deposition='voxel_cdf',target_population=1,
            target_voxel=jnp.array([6,4,4]),source_spacing=1.5,volume_order=1,order=8)
        def full(t):return predict_source_volume_intensity(positions,velocities,mass,sky,
            **g,source_velocity_variances_km2_s2=var,dispersion_scale=t)
        def chunks(t):return predict_chunked_volume_intensity(positions,velocities,mass,sky,
            **g,source_velocity_variances_km2_s2=var,dispersion_scale=t,source_chunk_size=2)
        a,da=jax.jit(jax.value_and_grad(full))(.8)
        b,db=jax.jit(jax.value_and_grad(chunks))(.8)
        self.assertGreater(float(a),0.)
        np.testing.assert_allclose([a,da],[b,db],rtol=2e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
