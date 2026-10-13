import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_marked_tracer_jax import predict_source_marked_radial_key_density
from cf4_r2_shell_cdf_count import predict_shell_cdf_intensity,ray_voxel_interval
from cf4_r2_raw_volume_target import FreshRawSupport
from cf4_r2_velocity_closure import mixture_components


class SharedClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64',True)

    def setUp(self):
        r=jnp.linspace(.001,400.,4001)
        self.g=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=r,
            modulus_table_h=5*jnp.log10(r)+25.,redshift_table=r/3000.,grid_size=128,
            sigma_los_km_s=20.,finite_reference_interval=(-25.,-21.))
        self.pos=jnp.array([[222.4,192.8,192.9]])
        self.vel=jnp.zeros((1,3));self.mass=jnp.ones((5,1));self.sky=jnp.ones((2,1))
        self.var=jnp.array([[20000.,30000.,40000.]])
        self.voxel=jnp.array([74,64,64]);self.pop=1

    def test_radial_density_integrates_same_voxel_cdf_mixture(self):
        direction=(self.pos-self.g['observer'])/jnp.linalg.norm(self.pos-self.g['observer'],axis=1)[:,None]
        lo,hi,active=ray_voxel_interval(direction,self.g['observer'],self.voxel,128,384.,jnp.zeros(3))
        self.assertTrue(bool(active[0]))
        t,w=np.polynomial.legendre.leggauss(128)
        radii=(float(lo[0])+float(hi[0]))/2+(float(hi[0])-float(lo[0]))/2*t
        closure=dict(core_sigma_km_s=20.,dispersion_scale=.8,broad_fraction=.7)
        expected=0.;actual=0.
        for weight,width,scale in mixture_components(closure):
            g=dict(self.g,sigma_los_km_s=width,source_velocity_variances_km2_s2=self.var,
                dispersion_scale=scale)
            radial=jax.jit(jax.vmap(lambda r:predict_source_marked_radial_key_density(
                self.pos,self.vel,self.mass,self.sky,self.pop,self.voxel,r,
                **dict(g,deposition='voxel_cdf')).sum()))(jnp.asarray(radii))
            expected+=weight*jnp.sum(radial*jnp.asarray(w))*(hi[0]-lo[0])/2
            actual+=weight*predict_shell_cdf_intensity(self.pos,self.vel,self.mass,self.sky,
                **g,target_population=self.pop,target_voxel=self.voxel,
                deposition='voxel_cdf',order=32,segments=8)
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=1e-12)

    def test_variance_and_scale_radial_gradients(self):
        def score(scale,var):
            return predict_source_marked_radial_key_density(self.pos,self.vel,self.mass,self.sky,
                self.pop,self.voxel,31.,**self.g,deposition='voxel_cdf',
                source_velocity_variances_km2_s2=var,dispersion_scale=scale).sum()
        for axis in ('scale','variance'):
            f=(lambda t:score(t,self.var)) if axis=='scale' else (lambda t:score(.8,self.var*jnp.exp(t)))
            point=.8 if axis=='scale' else 0.
            reverse=jax.jit(jax.grad(f))(point)
            eps=1e-4;fd=(f(point+eps)-f(point-eps))/(2*eps)
            np.testing.assert_allclose(reverse,fd,rtol=1e-5,atol=1e-10)
            self.assertGreater(abs(float(reverse)),1e-10)

    def test_periodic_alias_and_tail_support(self):
        pos=jnp.array([[383.,192.1,192.1]])
        g=dict(self.g,sigma_los_km_s=2300.)
        voxel=jnp.array([59,64,64])
        value=predict_source_marked_radial_key_density(pos,self.vel,self.mass,self.sky,
            0,voxel,14.,**g,deposition='voxel_cdf')
        self.assertGreater(float(value.sum()),0.)
        no_tail=predict_source_marked_radial_key_density(pos,self.vel,self.mass,self.sky,
            0,voxel,14.,**dict(g,sigma_los_km_s=20.),deposition='voxel_cdf')
        self.assertEqual(float(no_tail.sum()),0.)

    def test_fresh_support_variance_contract(self):
        g={k:v for k,v in self.g.items() if k not in ('sigma_los_km_s','finite_reference_interval')}
        o=dict(voxel=np.asarray(self.voxel)[None],radius=np.array([31.]))
        support=FreshRawSupport(self.pos,self.sky,[self.pop],o,g,source_spacing=1.5,volume_order=2,block=64)
        closure=dict(core_sigma_km_s=20.,dispersion_scale=.8,broad_fraction=.7)
        packs,info=support.build(self.vel,np.zeros(9),velocity_variances=self.var,velocity_closure=closure)
        self.assertGreater(info['components'],0)
        self.assertTrue(np.asarray(packs[self.pop]['mask']).any())
        with self.assertRaises(ValueError):
            support.build(self.vel,np.zeros(9),velocity_variances=-self.var,velocity_closure=closure)


if __name__=='__main__':unittest.main()
