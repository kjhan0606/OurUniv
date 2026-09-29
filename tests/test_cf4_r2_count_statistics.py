import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_2mpp_joint_likelihood_jax import tsc_deposit_jax
from cf4_r2_count_statistics import count_statistic_layout,tsc_count_statistics,poisson_from_statistics
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood


class CountStatisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_full_poisson_value_and_position_mass_gradients(self):
        n=8;box=384.;p=2
        position=jnp.array([[1.,2.,3.],[200.,201.,202.],[383.,381.,382.],[94.,145.,80.]])
        mass=jnp.array([1.,2.,3.,4.])
        base=np.asarray(tsc_deposit_jax(position,mass,n,box)).ravel()
        mask=np.random.default_rng(329).uniform(size=6*n**3)>.2
        keys=np.flatnonzero((base>0)&mask[p*n**3:(p+1)*n**3])[::3]+p*n**3
        counts=jnp.ones(len(keys));layout=count_statistic_layout(n,keys,mask)
        def full(x,m):
            grid=jnp.zeros((6,n,n,n)).at[p].set(tsc_deposit_jax(x,m,n,box))
            return sparse_marked_poisson_log_likelihood(grid,jnp.asarray(keys),counts,selected_voxel_mask=jnp.asarray(mask))
        def compact(x,m):return poisson_from_statistics(tsc_count_statistics(x,m,p,n,box,layout,len(keys)+1),counts)
        va,ga=jax.jit(jax.value_and_grad(full,argnums=(0,1)))(position,mass)
        vb,gb=jax.jit(jax.value_and_grad(compact,argnums=(0,1)))(position,mass)
        np.testing.assert_allclose(va,vb,rtol=1e-12,atol=1e-12)
        for a,b in zip(ga,gb):np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)

    def test_no_observed_keys_still_penalizes_all_exposed_mass(self):
        layout=count_statistic_layout(8,np.empty(0,dtype=int))
        x=jnp.array([[10.,20.,30.],[383.,382.,381.]]);m=jnp.array([2.,3.])
        statistics=tsc_count_statistics(x,m,5,8,384.,layout,1)
        # Missing keys must not scatter into the total and double its mass.
        np.testing.assert_allclose(statistics,[5.],atol=1e-13)
        self.assertAlmostEqual(float(poisson_from_statistics(statistics,jnp.empty(0))),-5.,places=13)

    def test_invalid_or_excluded_keys_rejected(self):
        for keys in ([1,1],[-1],[6*8**3]):
            with self.assertRaises(ValueError):count_statistic_layout(8,keys)
        with self.assertRaises(ValueError):count_statistic_layout(8,[1],np.zeros(8**3,dtype=bool))


if __name__=='__main__':unittest.main()
