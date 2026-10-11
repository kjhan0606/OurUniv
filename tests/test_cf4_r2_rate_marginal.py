import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from cf4_r2_rate_marginal import gamma_poisson_log_marginal_sparse


class RateMarginalTest(unittest.TestCase):
    def test_two_cells_against_direct_integral(self):
        exposure = np.array([[.3,.8]]*6)
        keys = np.array([0,1,2,3,4,5,6,7,8,9,10,11], dtype=np.int32)
        counts = np.array([1,2,1,1,1,1,1,1,1,1,1,1], dtype=np.int32)
        shape = np.full(6,2.)
        mean = np.full(6,1.4)
        got = float(gamma_poisson_log_marginal_sparse(jnp.asarray(exposure),
            jnp.asarray(keys),jnp.asarray(counts),jnp.asarray(mean),jnp.asarray(shape)))
        beta=shape[0]/mean[0]
        def one(pop):
            nn=counts[2*pop:2*pop+2]
            def integrand(rate):
                return (beta**2/np.exp(gammaln(2))*rate*np.exp(-beta*rate)
                    * np.prod(np.exp(nn*np.log(rate*exposure[pop])
                        -rate*exposure[pop]-gammaln(nn+1))))
            return np.log(quad(integrand,0,np.inf,epsabs=1e-13,epsrel=1e-13)[0])
        self.assertAlmostEqual(got,sum(one(p) for p in range(6)),places=9)

    def test_positive_count_zero_support_is_impossible(self):
        exposure=jnp.ones((6,2)).at[0,0].set(0.)
        result=gamma_poisson_log_marginal_sparse(exposure,jnp.asarray([0]),
            jnp.asarray([1]),jnp.ones(6),jnp.ones(6))
        self.assertTrue(np.isneginf(float(result)))

    def test_log_derivative(self):
        exposure=jnp.array([[.4,.8]]*6)
        keys=jnp.asarray([0,1],dtype=jnp.int32)
        counts=jnp.asarray([1,2])
        def objective(x):
            return gamma_poisson_log_marginal_sparse(
                exposure.at[0,0].set(x),keys,counts,jnp.ones(6),jnp.ones(6))
        grad=float(jax.grad(objective)(.4))
        step=1e-4
        fd=(float(objective(.4+step))-float(objective(.4-step)))/(2*step)
        self.assertAlmostEqual(grad,fd,places=3)


if __name__=='__main__':
    unittest.main()
