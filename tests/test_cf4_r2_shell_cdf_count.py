"""Analytic probability, alias/negative-branch and actual K/TSC controls."""
from itertools import product
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad

import cf4_r2_shell_cdf_count as shell_module
from cf4_r2_marked_tracer_jax import (
    _source_mark_transfer_reference, source_mark_transfer,
    intrinsic_biased_source_masses,intrinsic_biased_source_reference_rates,
)
from cf4_r2_shell_cdf_count import (shell_cdf_nodes,predict_shell_cdf_intensity,
    cell_averaged_tsc_weight,predict_source_volume_intensity,ngp_deposit_jax)
from cf4_2mpp_joint_likelihood_jax import tsc_exposure_weight_jax


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

    def test_finite_reference_volume_count_matches_legacy_and_supports_alpha_below_minus_one(self):
        table=jnp.linspace(.001,400.,4001)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=table,
            modulus_table_h=5*jnp.log10(table)+25.,redshift_table=table/3000.,
            grid_size=8,sigma_los_km_s=100.,order=2,segments=2)
        positions=jnp.array([[280.,192.,192.],[275.,193.,191.],[315.,190.,193.]])
        density=jnp.array([.45,.9,1.55]);bias=jnp.array([.6,.8,1.,1.25,1.45])
        angular=jnp.ones((2,3));log_rate=jnp.log(.12)
        cotangent=jnp.sin(jnp.arange(6*8**3,dtype=float)).reshape((6,8,8,8))

        def field(parameters,finite):
            alpha,velocity_scale=parameters
            if finite:
                intrinsic=intrinsic_biased_source_reference_rates(
                    density,log_rate,bias)
                interval={'finite_reference_interval':(-25.,-21.)}
            else:
                intrinsic=intrinsic_biased_source_masses(density,log_rate,bias,
                    mstar=-23.28,alpha=alpha,reference_interval=(-25.,-21.))
                interval={}
            velocity=jnp.array([[0.,0.,0.],[80.,-20.,10.],[-40.,5.,15.]])*velocity_scale
            return predict_source_volume_intensity(positions,velocity,intrinsic,angular,
                source_spacing=3.,volume_order=1,mstar=-23.28,alpha=alpha,
                **interval,**geometry)

        for alpha in (-.94,-.73):
            p=jnp.array([alpha,1.])
            old=jax.jit(lambda x:field(x,False))(p)
            new=jax.jit(lambda x:field(x,True))(p)
            np.testing.assert_allclose(old,new,rtol=3e-9,atol=3e-11)
            old_vg=jax.jit(jax.value_and_grad(
                lambda x:jnp.sum(field(x,False)*cotangent)))(p)
            new_vg=jax.jit(jax.value_and_grad(
                lambda x:jnp.sum(field(x,True)*cotangent)))(p)
            np.testing.assert_allclose(old_vg[0],new_vg[0],rtol=3e-9,atol=3e-10)
            np.testing.assert_allclose(old_vg[1],new_vg[1],rtol=3e-7,atol=3e-8)

        below=jax.jit(jax.value_and_grad(
            lambda a:jnp.sum(field(jnp.array([a,1.]),True))))(-1.35)
        self.assertTrue(np.isfinite(np.asarray(below)).all())
        self.assertGreater(float(below[0]),0.)

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

    def test_exposure_total_matches_full_TSC_grid_and_derivative(self):
        radius=jnp.linspace(.001,400.,4001)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,
            grid_size=16,sigma_los_km_s=100.,order=4,segments=8)
        flat=jnp.arange(16**3).reshape((16,16,16))
        mask=jnp.broadcast_to(((flat*17+3)%11)<5,(6,16,16,16))
        position=jnp.array([[280.3,192.,192.],[315.1,191.7,192.2]])
        intrinsic=jnp.asarray([[.7,1.1],[1.2,.8],[.9,1.4],[.6,.5],[1.3,.7]])
        angular=jnp.asarray([[.6,.9],[.8,.7]])

        def score(v,compact):
            velocity=jnp.asarray([[v,0.,0.],[0.,-v/3.,0.]])
            if compact:
                deposited=predict_shell_cdf_intensity(
                    position,velocity,intrinsic,angular,**geometry,
                    deposition='exposure_total',exposure_masks=mask)
                return deposited
            deposited=predict_shell_cdf_intensity(
                position,velocity,intrinsic,angular,**geometry)
            return jnp.sum(deposited*mask,axis=(1,2,3))

        full=jax.jit(jax.value_and_grad(lambda v:jnp.sum(score(v,False))))(35.)
        compact=jax.jit(jax.value_and_grad(lambda v:jnp.sum(score(v,True))))(35.)
        np.testing.assert_allclose(compact[0],full[0],rtol=2e-11,atol=2e-12)
        np.testing.assert_allclose(compact[1],full[1],rtol=3e-10,atol=3e-11)

        per_source=tsc_exposure_weight_jax(
            position,mask[0],geometry['grid_size'],geometry['box_size_cMpc_h'])
        self.assertTrue(np.all(np.asarray(per_source)>=0.))
        self.assertTrue(np.all(np.asarray(per_source)<=1.+1e-14))

    def test_inactive_clamped_sources_do_not_change_shell_cdf_gradient(self):
        radius=jnp.array([1.,192.])
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,
            redshift_table=radius/3000.,grid_size=8,sigma_los_km_s=100.,
            radial_min_cMpc_h=5.,radial_max_cMpc_h=180.,order=2,segments=1)
        # The first source contributes in the selected shell. The corner source
        # is beyond the lookup table and shell, so its inactive quadrature node
        # clamps true/observed modulus together and has exactly zero weight.
        positions=jnp.array([[292.,192.,192.],[350.,350.,350.]])
        intrinsic=jnp.ones((5,2));angular=jnp.ones((2,2))
        velocities=jnp.zeros((2,3))

        def evaluate(transfer):
            original=shell_module.source_mark_transfer
            try:
                shell_module.source_mark_transfer=transfer
                def value(v):
                    return jnp.sum(shell_module.predict_shell_cdf_intensity(
                        positions,v,intrinsic,angular,**geometry))
                return jax.jit(jax.value_and_grad(value))(velocities)
            finally:
                shell_module.source_mark_transfer=original

        fast=evaluate(source_mark_transfer)
        reference=evaluate(_source_mark_transfer_reference)
        np.testing.assert_allclose(fast[0],reference[0],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(fast[1],reference[1],rtol=1e-10,atol=1e-11)

    def test_ngp_deposit_matches_periodic_catalogue_floor_mapping(self):
        position=jnp.array([[.1,.9,1.2],[4.,4.,4.],[1.99,2.01,3.99]])
        mass=jnp.array([1.,2.,3.])
        got=np.asarray(ngp_deposit_jax(position,mass,4,4.))
        expected=np.zeros((4,4,4))
        expected[0,0,1]=1.
        expected[0,0,0]=2.
        expected[1,2,3]=3.
        np.testing.assert_array_equal(got,expected)
        self.assertEqual(got.sum(),6.)

    def test_source_count_ngp_closure_conserves_mass_but_changes_kernel(self):
        radius=jnp.linspace(.001,400.,4001)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,
            grid_size=16,sigma_los_km_s=100.,order=4,segments=8)
        pos=jnp.array([[280.3,192.,192.]])
        vel=jnp.zeros((1,3));intrinsic=jnp.ones((5,1));angular=jnp.ones((2,1))
        def field(deposition=None):
            extra={} if deposition is None else {'deposition':deposition}
            return predict_source_volume_intensity(pos,vel,intrinsic,angular,
                source_spacing=3.,volume_order=2,**geometry,**extra)
        default=jax.jit(field)(None)
        explicit=jax.jit(lambda:field('tsc'))()
        ngp=jax.jit(lambda:field('ngp'))()
        np.testing.assert_array_equal(default,explicit)
        np.testing.assert_allclose(np.asarray(default).sum(),np.asarray(ngp).sum(),
                                   rtol=1e-12,atol=1e-13)
        self.assertGreater(float(jnp.sum(jnp.abs(default-ngp))),1e-8)

    def test_scalar_ngp_gather_matches_full_ngp_deposit(self):
        radius=jnp.linspace(.001,400.,4001)
        kwargs=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,
            grid_size=16,sigma_los_km_s=100.,order=4,segments=8,deposition='ngp')
        position=jnp.array([[280.3,192.,192.]])
        full=predict_shell_cdf_intensity(position,jnp.zeros_like(position),
            jnp.ones((5,1)),jnp.ones((2,1)),**kwargs)
        voxel=jnp.array([11,8,8])
        scalar=predict_shell_cdf_intensity(position,jnp.zeros_like(position),
            jnp.ones((5,1)),jnp.ones((2,1)),**kwargs,
            target_population=0,target_voxel=voxel)
        np.testing.assert_allclose(scalar,full[0,11,8,8],rtol=1e-12,atol=1e-13)

    def test_cell_averaged_kernel_is_top_hat_integral_not_full_RSD_claim(self):
        # Unit-spacing periodic grid, separable integrals evaluated independently.
        voxel=np.array([5,6,7])
        for delta in ([.1,.7,1.4],[1.9,.1,.2],[-.6,.8,-1.1]):
            pos=(voxel+.5-np.asarray(delta))%16
            def tsc(d):
                d=abs(d)
                return .75-d*d if d<.5 else (.5*(1.5-d)**2 if d<1.5 else 0.)
            expected=np.prod([quad(lambda s:tsc(d-s),-.5,.5,epsabs=1e-12,
                points=[p for p in (d-1.5,d-.5,d+.5,d+1.5) if -.5<p<.5])[0] for d in delta])
            got=float(cell_averaged_tsc_weight(jnp.asarray(pos[None]),voxel,16,16.)[0])
            self.assertAlmostEqual(got,expected,places=12)

    def test_streamed_volume_equals_materialized_subnodes(self):
        radius=jnp.linspace(.001,400.,4001)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=radius,
            modulus_table_h=5*jnp.log10(radius)+25.,redshift_table=radius/3000.,
            grid_size=16,sigma_los_km_s=100.,order=4,segments=8)
        pos=jnp.array([[371.9,192.,192.]])
        offsets=jnp.asarray(list(product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3)))*1.5
        def score(v,stream):
            velocity=jnp.array([[v,0.,0.]])
            if stream:
                return predict_source_volume_intensity(pos,velocity,jnp.ones((5,1)),
                    jnp.ones((2,1)),source_spacing=3.,volume_order=2,**geometry)
            return predict_shell_cdf_intensity((pos+offsets)%384.,
                jnp.broadcast_to(velocity,(8,3)),jnp.ones((5,8))/8,jnp.ones((2,8)),**geometry)
        a=jax.jit(lambda v:score(v,True))(0.)
        b=jax.jit(lambda v:score(v,False))(0.)
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-13)
        da=jax.jit(jax.grad(lambda v:score(v,True).sum()))(0.)
        db=jax.jit(jax.grad(lambda v:score(v,False).sum()))(0.)
        np.testing.assert_allclose(da,db,rtol=1e-10,atol=1e-12)

    def test_image_gap_shortcut_preserves_value_and_gradient(self):
        radius=jnp.linspace(.001,400.,4001)
        def score(v,force_all):
            return predict_shell_cdf_intensity(jnp.array([[371.9,192.,192.]]),
                jnp.array([[v,0.,0.]]),jnp.ones((5,1)),jnp.ones((2,1)),
                observer=jnp.full(3,192.),box_size_cMpc_h=384.,hubble_km_s_Mpc=74.6,
                little_h=.746,radius_table_cMpc_h=radius,modulus_table_h=5*jnp.log10(radius)+25.,
                redshift_table=radius/3000.,grid_size=16,sigma_los_km_s=100.,order=4,segments=8,
                target_population=0,target_voxel=jnp.array([15,8,8]),force_all_images=force_all)
        a=jax.jit(jax.value_and_grad(lambda v:score(v,False)))(0.)
        b=jax.jit(jax.value_and_grad(lambda v:score(v,True)))(0.)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-13)

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
