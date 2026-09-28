"""Exact monotone-boundary reuse: values, derivatives and equality ties."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_marked_tracer_jax import source_mark_transfer,_source_mark_transfer_reference


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


if __name__=='__main__':
    unittest.main()
