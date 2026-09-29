import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_count_statistics import count_statistic_layout
from cf4_r2_count_statistic_integral import volume_count_statistics
from cf4_r2_shell_cdf_count import predict_source_volume_intensity


class StatisticIntegralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_same_full_volume_rule_and_gradient_including_aliases(self):
        radius=jnp.linspace(.001,400.,4001);n=8
        pos=jnp.array([[371.9,192.,192.],[382.,192.,192.],[301.,200.,190.]])
        intrinsic=jnp.arange(15,dtype=float).reshape(5,3)/15+.2;sky=jnp.ones((2,3))
        mask=np.random.default_rng(531).uniform(size=6*n**3)>.15
        keys=np.flatnonzero(mask)[::7];layout=count_statistic_layout(n,keys,mask)
        g=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,hubble_km_s_Mpc=74.6,
            little_h=.746,radius_table_cMpc_h=radius,modulus_table_h=5*jnp.log10(radius)+25.,
            redshift_table=radius/3000.,grid_size=n,sigma_los_km_s=1000.,order=4,segments=8)
        def read(v,compact):
            vel=jnp.array([[v,0.,0.],[0.,0.,0.],[100.,0.,0.]])
            if compact:return volume_count_statistics(pos,vel,intrinsic,sky,layout,len(keys)+1,
                source_spacing=3.,volume_order=2,source_chunk_size=2,**g)
            full=predict_source_volume_intensity(pos,vel,intrinsic,sky,source_spacing=3.,volume_order=2,**g).ravel()
            return jnp.r_[full[jnp.asarray(keys)],jnp.sum(full*jnp.asarray(mask))]
        a=jax.jit(lambda v:read(v,False))(0.);b=jax.jit(lambda v:read(v,True))(0.)
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)
        cotangent=jnp.cos(jnp.arange(len(keys)+1))
        ga=jax.jit(jax.grad(lambda v:read(v,False)@cotangent))(0.)
        gb=jax.jit(jax.grad(lambda v:read(v,True)@cotangent))(0.)
        np.testing.assert_allclose(ga,gb,rtol=1e-10,atol=1e-12)


if __name__=='__main__':unittest.main()
