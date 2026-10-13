import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_shell_cdf_count import predict_source_volume_intensity
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity


class ChunkedVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_values_and_gradient_preserved_with_partial_last_chunk(self):
        table=jnp.linspace(.001,400.,4001)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,hubble_km_s_Mpc=74.6,
            little_h=.746,radius_table_cMpc_h=table,modulus_table_h=5*jnp.log10(table)+25.,
            redshift_table=table/3000.,grid_size=16,sigma_los_km_s=100.,order=4,segments=8)
        pos=jnp.array([[371.9,192.,192.],[300.,193.,194.],[260.,194.,195.]])
        mass=jnp.arange(15,dtype=float).reshape(5,3)/15+.1;sky=jnp.ones((2,3))
        def intensity(scale,chunked):
            vel=jnp.array([[80.,3.,4.],[120.,5.,8.],[-70.,1.,2.]])*scale
            f=predict_chunked_volume_intensity if chunked else predict_source_volume_intensity
            return f(pos,vel,mass,sky,source_spacing=3.,volume_order=2,**geometry,
                **(dict(source_chunk_size=2) if chunked else {}))
        a=jax.jit(lambda x:intensity(x,False))(1.)
        b=jax.jit(lambda x:intensity(x,True))(1.)
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)
        # Nonuniform output cotangent, not merely total-mass conservation.
        weight=jnp.sin(jnp.arange(a.size)).reshape(a.shape)
        ga=jax.jit(jax.grad(lambda x:jnp.sum(intensity(x,False)*weight)))(1.)
        gb=jax.jit(jax.grad(lambda x:jnp.sum(intensity(x,True)*weight)))(1.)
        np.testing.assert_allclose(ga,gb,rtol=1e-10,atol=1e-12)


if __name__=='__main__':unittest.main()
