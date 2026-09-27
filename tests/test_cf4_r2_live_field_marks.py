import jax
import jax.numpy as jnp
import numpy as np
from cf4_z0_pm_bridge import enable_pmwd_type_description_compatibility
from cf4_z0_physical_field import read_centred
from cf4_r2_live_field_marks import bind_state_geometry, live_group_scores, joint_white_logdensity
from cf4_r2_joint_calibration import group_scores


def test_native_pm_node_contract():
    enable_pmwd_type_description_compatibility()
    from pmwd import Configuration, scatter, gather
    from pmwd.particles import Particles
    conf = Configuration(ptcl_spacing=2.,ptcl_grid_shape=(4,)*3,
                         float_dtype=jnp.float64,cosmo_dtype=jnp.float64)
    pos = jnp.array([[2.,2.,2.]])
    particles = Particles.from_pos(conf,pos)
    mesh = scatter(particles,conf,val=jnp.ones(1))
    np.testing.assert_allclose(mesh[1,1,1],1.,atol=1e-14)
    np.testing.assert_allclose(mesh.sum(),1.,atol=1e-14)
    queries = jnp.array([[2.,2.,2.],[2.4,2.6,1.8],[7.8,.3,1.1]])
    expected = gather(Particles.from_pos(conf,queries),conf,mesh)
    np.testing.assert_allclose(read_centred(mesh,queries,8.,0.),expected,atol=1e-14)
    assert abs(float(read_centred(mesh,pos,8.,.5)[0])-.125) < 1e-14


def test_live_binding_and_joint_derivative():
    n,box = 8,80.
    xyz = jnp.stack(jnp.meshgrid(*([jnp.arange(n)]*3),indexing='ij'),axis=0)
    rho = 1.+.1*jnp.sin(2*jnp.pi*xyz[0]/n)+.05*jnp.cos(2*jnp.pi*xyz[1]/n)
    velocity = 40*jnp.sin(2*jnp.pi*xyz/n)
    d = jnp.array([[10.,15.,20.],[15.,20.,25.]])
    geom = dict(distance=d,quadrature_weight=jnp.ones_like(d),
        directions=jnp.array([[1.,0.,0.],[0.,1.,0.]]),zcos=d/2997.92458,
        redshift_sufficient=jnp.array([[1/400.**2,1500.,0.,np.log(400.**2),1.],
                                     [1/400.**2,2000.,0.,np.log(400.**2),1.]]),
        row_group=jnp.array([0,0,1]),dz_row=jnp.array([15.,15.,20.]),
        eta_mean=jnp.array([.03,-.01,.02]),eta_std=jnp.array([.1,.12,.08]),eta_alpha=jnp.zeros(3),
        predicted_modulus=5*jnp.log10(d/.746)+25,
        anchor_group=jnp.array([0,1]),anchor_modulus=jnp.array([31.6,32.2]),
        anchor_error=jnp.array([.2,.3]),anchor_method=jnp.array([0,1]),
        group_holdout=jnp.array([False,True]))
    theta = jnp.array([.003,.05,-.1,.4])
    bound = bind_state_geometry(rho,velocity,geom,box=box)
    np.testing.assert_allclose(live_group_scores(rho,velocity,theta,geom,box=box),
                               group_scores(theta,bound),atol=1e-12)
    # The SAME changing density and velocity enter the conditional numerator
    # and denominator; field-independent caches cannot pass this derivative.
    sd = jnp.array([.004,1.,1.,2.])
    def f(x):
        # Both rays stay in one z plane. A z-only multiplier is constant
        # along each ray and must cancel, so use a genuinely radial shape.
        r = rho*jnp.exp(x[0]*.2*jnp.cos(xyz[0]+xyz[1]))
        v = velocity+x[1]*30*jnp.cos(xyz)
        return joint_white_logdensity(x[:2],x[2:],r,v,geom,sd,box=box)
    x = jnp.array([.1,-.2,.2,.1,-.1,.3])
    grad = np.asarray(jax.grad(f)(x))
    for k in range(len(x)):
        delta = jnp.eye(len(x))[k]*1e-5
        np.testing.assert_allclose(grad[k],(f(x+delta)-f(x-delta))/2e-5,rtol=2e-5,atol=1e-7)
    assert abs(grad[0]+float(x[0])) > 1e-5
    assert abs(grad[1]+float(x[1])) > 1e-5


if __name__ == '__main__':
    test_native_pm_node_contract()
    test_live_binding_and_joint_derivative()
    print('PASS: native PM scatter/gather origin; live joint mark binding and gradients',flush=True)
