"""Exact monotone-boundary reuse: values, derivatives and equality ties."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_marked_tracer_jax import (
    intrinsic_lf_reference_weights,source_mark_transfer,
    _source_mark_transfer_reference,
)


class LFReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64',True)

    def test_values_and_all_parameter_derivatives(self):
        true=jnp.linspace(25.,38.,19)
        offset=jnp.linspace(-3.,3.,19)
        z=jnp.linspace(.0001,.09,19)
        weight=jnp.arange(6*5*19,dtype=float).reshape(6,5,19)/570
        def output(parameters,method):
            return method(true,true+offset+parameters[0],z,z+parameters[1],
                          mstar=parameters[2],alpha=parameters[3])
        values_a=jax.jit(lambda x:output(x,source_mark_transfer))
        values_b=jax.jit(lambda x:output(x,_source_mark_transfer_reference))
        gradient_a=jax.jit(jax.grad(lambda x:jnp.sum(output(x,source_mark_transfer)*weight)))
        gradient_b=jax.jit(jax.grad(lambda x:jnp.sum(output(x,_source_mark_transfer_reference)*weight)))
        for alpha in (-.999,-.94,-.73):
            p=jnp.array([.017,.002,-23.28,alpha])
            a,b=values_a(p),values_b(p)
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-13)
            ga,gb=gradient_a(p),gradient_b(p)
            self.assertTrue(np.isfinite(np.asarray(ga)).all())
            np.testing.assert_allclose(ga,gb,rtol=1e-9,atol=1e-9)

    def test_identical_intrinsic_and_observed_edges_keep_subgradient(self):
        true=jnp.array([26.,30.,34.,38.]); z=jnp.array([.001,.01,.04,.08])
        weight=jnp.sin(jnp.arange(120,dtype=float).reshape(6,5,4)+.3)
        def total(shift,method):
            return jnp.sum(method(true,true+shift,z,z)*weight)
        for method in (source_mark_transfer,_source_mark_transfer_reference):
            value=float(total(0.,method))
            grad=float(jax.grad(lambda s:total(s,method))(0.))
            if method is source_mark_transfer:
                candidate=(value,grad)
            else:
                np.testing.assert_allclose(candidate,(value,grad),rtol=1e-10,atol=1e-10)

    def test_jitted_apparent_and_three_way_ties_match_all_reference_gradients(self):
        true=jnp.array([36.5])
        z=jnp.array([.01])
        weight=jnp.arange(30,dtype=float).reshape(6,5,1)/30

        def output(parameters,method):
            return jnp.sum(method(true,true+parameters[0],z,z,
                mstar=parameters[1],alpha=parameters[2])*weight)

        fast=jax.jit(jax.value_and_grad(
            lambda p:output(p,source_mark_transfer)))
        reference=jax.jit(jax.value_and_grad(
            lambda p:output(p,_source_mark_transfer_reference)))
        # shift=0 gives a three-way true/apparent/observed edge tie at M=-25;
        # shift=.17 keeps the true/apparent tie but separates observed edges.
        for shift in (0.,.17):
            parameters=jnp.array([shift,-23.28,-.94])
            fast_value,fast_gradient=fast(parameters)
            ref_value,ref_gradient=reference(parameters)
            np.testing.assert_allclose(fast_value,ref_value,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(fast_gradient,ref_gradient,
                                       rtol=1e-10,atol=1e-10)

    def test_finite_reference_transfer_preserves_legacy_factorization_values_and_gradients(self):
        true=jnp.linspace(25.,38.,19)
        offset=jnp.linspace(-3.,3.,19)
        z=jnp.linspace(.0001,.09,19)
        weight=jnp.arange(6*5*19,dtype=float).reshape(6,5,19)/570

        def old(parameters):
            weights=intrinsic_lf_reference_weights(
                mstar=parameters[2],alpha=parameters[3])
            transfer=source_mark_transfer(true,true+offset+parameters[0],z,z+parameters[1],
                mstar=parameters[2],alpha=parameters[3])
            return jnp.sum(weight*weights[None,:,None]*transfer)

        def finite(parameters):
            transfer=source_mark_transfer(true,true+offset+parameters[0],z,z+parameters[1],
                mstar=parameters[2],alpha=parameters[3],
                finite_reference_interval=(-25.,-21.))
            return jnp.sum(weight*transfer)

        old_value_gradient=jax.jit(jax.value_and_grad(old))
        finite_value_gradient=jax.jit(jax.value_and_grad(finite))
        for alpha in (-.999,-.94,-.73):
            p=jnp.array([.017,.002,-23.28,alpha])
            a,b=old_value_gradient(p),finite_value_gradient(p)
            np.testing.assert_allclose(a[0],b[0],rtol=2e-10,atol=2e-11)
            np.testing.assert_allclose(a[1],b[1],rtol=2e-8,atol=2e-8)

    def test_finite_reference_transfer_is_regular_across_alpha_minus_one(self):
        true=jnp.array([25.1,29.7,34.2,38.])
        observed=true+jnp.array([-.03,.01,.05,-.02])
        z=jnp.array([.0001,.01,.04,.09])
        weights=jnp.cos(jnp.arange(120,dtype=float).reshape(6,5,4)+.2)

        def objective(parameters):
            transfer=source_mark_transfer(true,observed,z,z,
                mstar=parameters[0],alpha=parameters[1],
                finite_reference_interval=(-25.,-21.))
            return jnp.sum(weights*transfer)

        value_gradient=jax.jit(jax.value_and_grad(objective))
        for alpha in (-1.7,-1.1,-1.,-.9,-.73):
            value,gradient=value_gradient(jnp.array([-23.28,alpha]))
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(np.isfinite(np.asarray(gradient)).all())
            self.assertGreater(float(jnp.sum(source_mark_transfer(
                true,observed,z,z,mstar=-23.28,alpha=alpha,
                finite_reference_interval=(-25.,-21.)))),0.)

    def test_finite_reference_three_way_tie_keeps_legacy_factorized_subgradient(self):
        true=jnp.array([36.5])
        z=jnp.array([.01])
        weight=jnp.arange(30,dtype=float).reshape(6,5,1)/30

        def factorized(parameters):
            lf_weight=intrinsic_lf_reference_weights(
                mstar=parameters[1],alpha=parameters[2])
            conditional=source_mark_transfer(true,true+parameters[0],z,z,
                mstar=parameters[1],alpha=parameters[2])
            return jnp.sum(weight*lf_weight[None,:,None]*conditional)

        def finite(parameters):
            transfer=source_mark_transfer(true,true+parameters[0],z,z,
                mstar=parameters[1],alpha=parameters[2],
                finite_reference_interval=(-25.,-21.))
            return jnp.sum(weight*transfer)

        p=jnp.array([0.,-23.28,-.94])
        a=jax.jit(jax.value_and_grad(factorized))(p)
        b=jax.jit(jax.value_and_grad(finite))(p)
        np.testing.assert_allclose(a[0],b[0],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(a[1],b[1],rtol=1e-8,atol=1e-8)


if __name__=='__main__':
    unittest.main()
