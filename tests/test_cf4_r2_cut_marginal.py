import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr
from cf4_r2_raw_selected_fp import rule,log_box_probability,log_cdf_interval
from cf4_r2_raw_selected_fp_jax import correlated_rectangle_logprob


class CutMarginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_broad_cut_exact_marginal_and_gradient_without_independence(self):
        t,w=map(jnp.asarray,rule(64))
        lower=jnp.array([[-12.,-.8]]);upper=jnp.array([[13.,.7]])
        def logprob(mean):
            return correlated_rectangle_logprob(lower-mean,upper-mean,jnp.array([.9]),t,w,
                integration_axis=1,marginal_tolerance=1e-12)[0]
        value,gradient=jax.jit(jax.value_and_grad(logprob))(jnp.zeros(2))
        probability=ndtr(.7)-ndtr(-.8)
        expected=(np.exp(-.5*.8**2)-np.exp(-.5*.7**2))/np.sqrt(2*np.pi)/probability
        self.assertAlmostEqual(float(value),np.log(probability),places=13)
        np.testing.assert_allclose(gradient,[0.,expected],atol=1e-12)

    def test_absolute_small_tail_is_not_enough_for_rare_retained_event(self):
        t,w=map(jnp.asarray,rule(256));lo=jnp.array([[-8.,9.]]);hi=jnp.array([[8.,9.1]])
        f=lambda tol:correlated_rectangle_logprob(lo,hi,jnp.array([.9]),t,w,
            integration_axis=1,marginal_tolerance=tol)[0]
        self.assertAlmostEqual(float(f(1e-12)),float(f(0.)),places=12)
        self.assertLess(float(f(1e-12)),float(log_cdf_interval(9.,9.1))-.1)

    def test_axis_change_preserves_correlated_rectangle(self):
        t,w=map(jnp.asarray,rule(256));lo=jnp.array([[-.3,-1.]]);hi=jnp.array([[.7,.6]])
        expected=float(log_box_probability([0.,0.],[[1.,.6],[.6,1.]],[-.3,-1.],[.7,.6],order=256))
        for axis in (0,1):
            value=correlated_rectangle_logprob(lo,hi,jnp.array([.6]),t,w,
                integration_axis=axis,marginal_tolerance=1e-12)
            self.assertAlmostEqual(float(value[0]),expected,places=11)


if __name__=='__main__':unittest.main()
