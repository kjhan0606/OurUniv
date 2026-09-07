import unittest
import numpy as np
from cf4_multires_z0 import cell_axis,child_origin,refine,restrict


class MultiresTest(unittest.TestCase):
    def test_native_geometry(self):
        self.assertEqual(child_origin(.25,8),-1.5)
        axis=cell_axis(256,1.5,-3.)
        np.testing.assert_allclose(axis.reshape(32,8).mean(axis=1),(np.arange(32)+.25)*12)
        local=cell_axis(128,.1875,177.)
        np.testing.assert_allclose(local.reshape(2,64).mean(axis=1),[183.,195.])

    def test_mass_and_momentum(self):
        rng=np.random.default_rng(2026090709)
        rho=np.exp(rng.normal(size=(2,)*3)); vel=rng.normal(size=(3,2,2,2))*200
        detail=rng.normal(size=(16,)*3)
        dv=rng.normal(size=(3,16,16,16))*100
        fine=refine(rho,vel,detail,dv,8)
        result=restrict(*fine,8)
        self.assertTrue(np.all(fine[0]>0))
        np.testing.assert_allclose(result[0],rho,rtol=1e-12)
        np.testing.assert_allclose(result[1],vel,atol=1e-10)
        # Invariance to unidentifiable constant log-detail and velocity offsets.
        same=refine(rho,vel,detail+20,dv+300,8)
        np.testing.assert_allclose(same[0],fine[0],atol=1e-12)
        np.testing.assert_allclose(same[1],fine[1],atol=1e-10)

    def test_jax_gradient_and_actual_parent(self):
        import jax
        import jax.numpy as jnp
        path="/gpfs/kjhan/CF4/z0_density/actual_data_longer_v3/task_0/posterior_fields.npz"
        with np.load(path,allow_pickle=False) as f:
            rho=1+f["density_mean"][15:17,15:17,15:17]
            vel=f["velocity_mean"][:,15:17,15:17,15:17]
        detail=jnp.zeros((16,)*3); dv=jnp.zeros((3,16,16,16))
        fn=jax.jit(lambda x:refine(jnp.asarray(rho),jnp.asarray(vel),x,dv,8,xp=jnp))
        fine=tuple(np.asarray(a) for a in fn(detail))
        back=restrict(*fine,8)
        np.testing.assert_allclose(back[0],rho,rtol=1e-12)
        np.testing.assert_allclose(back[1],vel,atol=1e-10)
        g=jax.grad(lambda x:fn(x)[0][0,0,0])(detail)
        self.assertTrue(np.isfinite(g).all())
        self.assertGreater(float(np.linalg.norm(g)),0)


if __name__=="__main__": unittest.main()
