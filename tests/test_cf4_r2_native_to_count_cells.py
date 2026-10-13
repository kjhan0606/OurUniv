import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells


def test_native_to_voxel_mass_momentum():
    n,box=4,8.
    rho=jnp.zeros((n,n,n)).at[1,1,1].set(8.)
    velocity=jnp.zeros((3,n,n,n)).at[:,1,1,1].set(jnp.array([30.,-20.,5.]))
    mass,v=native_mass_momentum_to_count_cells(rho,velocity,box)
    np.testing.assert_allclose(mass.sum(),rho.sum(),atol=1e-12)
    for k in range(3):
        np.testing.assert_allclose((mass*v[k]).sum(),(rho*velocity[k]).sum(),atol=1e-12)
    assert int(np.count_nonzero(np.asarray(mass)))==8
    np.testing.assert_allclose(np.asarray(mass)[np.asarray(mass)>0],np.ones(8),atol=1e-12)
    assert np.isfinite(np.asarray(v)).all()
    np.testing.assert_allclose(jnp.array([x.sum() for x in
        jax.grad(lambda r:native_mass_momentum_to_count_cells(r,velocity,box)[0].sum())(rho)]),
        jnp.full((n,),n*n),atol=1e-12)


if __name__=='__main__':
    test_native_to_voxel_mass_momentum()
    print('PASS: node-to-voxel mass, momentum and differentiability',flush=True)
