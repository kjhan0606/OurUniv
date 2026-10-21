import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_moment_target import MomentObservationTarget
from cf4_r2_velocity_closure import active_tracer_coordinates,closure_from_coordinates


class MomentTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def test_active_coordinates_no_unused_sigma(self):
        t=jnp.arange(8.)
        np.testing.assert_array_equal(active_tracer_coordinates(t),[0,1,2,3,4,5,0,6,7])
        closure=closure_from_coordinates(jnp.array([jnp.log(30.),0.,0.]))
        self.assertAlmostEqual(float(closure['core_sigma_km_s']),30.)
        self.assertAlmostEqual(float(closure['broad_fraction']),.5)

    def test_joint_density_velocity_variance_nuisance_adjoint(self):
        n=4;box=48.;ng=8
        pos=(jnp.indices((n,)*3).reshape(3,-1).T+.5)*(box/n)
        source=dict(positions=pos,angular=jnp.ones((2,n**3)))
        rtab=jnp.linspace(.001,48.,1001)
        g=dict(observer=jnp.full(3,24.),box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,
            radius_table_cMpc_h=rtab,modulus_table_h=5*jnp.log10(rtab)+25.,redshift_table=rtab/3000.,
            grid_size=ng,radial_min_cMpc_h=5.,radial_max_cMpc_h=20.)
        o=dict(voxel=jnp.array([[5,5,5]]),radius=jnp.array([11.]),dz=jnp.array([11.]),
            ksmag=jnp.array([7.3]),x=jnp.array([[.3,2.2,2.7]]),error_covariance=jnp.eye(3)[None]*.002,
            richness=jnp.array([0.]),cut_lower=jnp.array([[2.,1.9]]),cut_upper=jnp.array([[4.,2.5]]))
        keys=jnp.array([ng**3+5*ng**2+5*ng+5]);counts=jnp.array([2.])
        target=MomentObservationTarget(n,source,[1],o,g,keys,counts,jnp.ones((6,ng,ng,ng)),
            volume_orders=(1,),source_chunk=32,raw_block=64,cut_order=16)
        rho=jnp.arange(n**3,dtype=float).reshape(n,n,n)/64+.5
        vel=jnp.stack((rho*3.,rho*-2.,rho))
        var=jnp.stack((rho*1000.,rho*2000.,rho*1500.))
        t8=jnp.zeros(8);p=jnp.zeros(15);c=jnp.array([jnp.log(30.),jnp.log(.3),0.])
        state=(rho,vel,var,t8,p,c)
        packs,_=target.support(rho,vel,var,t8,c,1)
        (value,parts),grad=target.derivative(*state,packs,source,o,1)
        direct=target.value(*state,packs,source,o,1)[0]
        np.testing.assert_allclose(value,direct,atol=1e-9)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.isfinite(np.asarray(parts)).all())
        directions=(rho*.01,vel*.03,var*.02,jnp.full(8,.01),jnp.full(15,.01),jnp.array([.02,-.03,.01]))
        self.assertGreater(float(jnp.linalg.norm(grad[2])),0.)
        for derivative in grad:self.assertTrue(np.isfinite(np.asarray(derivative)).all())
        reverse=sum(float(jnp.sum(a*b)) for a,b in zip(grad,directions))
        eps=1e-4;values=[]
        for sign in (1,-1):
            trial=tuple(a+sign*eps*b for a,b in zip(state,directions))
            pack,_=target.support(trial[0],trial[1],trial[2],trial[3],trial[5],1)
            values.append(float(target.value(*trial,pack,source,o,1)[0]))
        fd=(values[0]-values[1])/(2*eps)
        np.testing.assert_allclose(reverse,fd,rtol=2e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
