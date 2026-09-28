"""Analytic probability, alias/negative-branch and actual K/TSC controls."""
from itertools import product
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad

from cf4_r2_shell_cdf_count import shell_cdf_nodes,predict_shell_cdf_intensity


class ShellCDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64',True)

    def test_boundary_probability_and_derivative(self):
        def probability(mu):
            q,w=shell_cdf_nodes(jnp.array([mu]),jnp.array([[1.,0.,0.]]),
                               1.,jnp.zeros(3),order=12)
            return w.sum()
        for mu in (179.9,180.,180.1):
            self.assertAlmostEqual(float(probability(mu)),ndtr(180.-mu),places=13)
        derivative=float(jax.grad(probability)(180.))
        self.assertAlmostEqual(derivative,-1/np.sqrt(2*np.pi),places=12)

    def test_scalar_gather_equals_full_deposit_and_derivative(self):
        radius=jnp.linspace(.001,400.,4001)
        kwargs=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,
            grid_size=16,sigma_los_km_s=100.,order=4,segments=8)
        voxel=jnp.array([15,8,8]); pop=0
        def read(v,scalar):
            result=predict_shell_cdf_intensity(jnp.array([[371.9,192.,192.]]),
                jnp.array([[v,0.,0.]]),jnp.ones((5,1)),jnp.ones((2,1)),
                **kwargs,**(dict(target_population=pop,target_voxel=voxel) if scalar else {}))
            return result if scalar else result[pop,15,8,8]
        full=jax.jit(jax.value_and_grad(lambda v:read(v,False)))(0.)
        scalar=jax.jit(jax.value_and_grad(lambda v:read(v,True)))(0.)
        np.testing.assert_allclose(scalar,full,rtol=1e-11,atol=1e-12)
        self.assertGreater(float(scalar[0]),0.)

    def test_inner_exclusion_both_signed_branches_and_weights(self):
        q,w=shell_cdf_nodes(jnp.array([0.]),jnp.array([[1.,0.,0.]]),
                           3.,jnp.zeros(3),order=12)
        active=np.asarray(w)>0
        self.assertTrue((np.abs(np.asarray(q)[active])>=5.).all())
        self.assertAlmostEqual(float(w.sum()),2*ndtr(-5/3),places=13)
        self.assertAlmostEqual(float(w[0].sum()),float(w[1].sum()),places=14)

    def test_periodic_image_tail_is_not_omitted(self):
        mass=0.
        for image in product((-1,0,1),repeat=3):
            q,w=shell_cdf_nodes(jnp.array([190.]),jnp.array([[1.,0.,0.]]),
                               10.,jnp.array(image)*384.,order=8)
            mass+=float(w.sum())
        # Central sphere q<=180 and wrapped neighbour q>=384-180=204.
        self.assertAlmostEqual(mass,ndtr(-1)+ndtr(-1.4),places=13)

    def test_physical_strata_resolve_a_five_sigma_TSC_tail(self):
        def kernel(x):
            d=np.abs(x-105.)
            return np.where(d<.5,.75-d*d,np.where(d<1.5,.5*(1.5-d)**2,0.))
        expected=quad(lambda x:float(kernel(x))*np.exp(-.5*(x-100)**2)/np.sqrt(2*np.pi),
                      103.5,106.5,points=[104.5,105.5],epsabs=1e-15)[0]
        values=[]
        for order,segments in ((16,1),(4,16),(8,32)):
            q,w=shell_cdf_nodes(jnp.array([100.]),jnp.array([[1.,0.,0.]]),
                1.,jnp.zeros(3),order=order,segments=segments)
            values.append(float(np.sum(np.asarray(w)*kernel(np.asarray(q)))))
        print('five_sigma_TSC_tail',dict(reference=expected,global16=values[0],
              strata16_order4=values[1],strata32_order8=values[2]),flush=True)
        self.assertEqual(values[0],0.)
        self.assertGreater(values[1],0.)
        self.assertLess(abs(values[2]/expected-1),.001)

    def test_full_source_K_TSC_values_and_velocity_gradient(self):
        box=384.
        radius=jnp.linspace(.001,400.,4001)
        def intensity(v,order):
            # Real archive coordinates are float32, fields/masses float64.
            return predict_shell_cdf_intensity(jnp.array([[371.9,192.,192.]],dtype=jnp.float32),
                jnp.array([[v,0.,0.]]),jnp.ones((5,1)),jnp.ones((2,1)),
                observer=jnp.full(3,192.),box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,
                little_h=.746,radius_table_cMpc_h=radius,
                modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,
                grid_size=16,sigma_los_km_s=100.,order=order)
        evaluate=jax.jit(intensity,static_argnums=1)
        low=np.asarray(evaluate(0.,16)); high=np.asarray(evaluate(0.,32))
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue((high>=0).all())
        self.assertGreater(high.sum(),0.)
        self.assertLess(np.abs(low-high).sum()/high.sum(),.001)
        gradient=float(jax.jit(jax.grad(lambda v:intensity(v,16).sum()))(0.))
        fd=float((evaluate(.001,16).sum()-evaluate(-.001,16).sum())/.002)
        self.assertTrue(np.isfinite(gradient))
        self.assertGreater(abs(gradient),1e-8)
        np.testing.assert_allclose(gradient,fd,rtol=1e-4,atol=1e-8)


if __name__=='__main__':
    unittest.main()
