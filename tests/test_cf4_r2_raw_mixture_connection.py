"""The raw FP factor mixes intensities before conditional normalization."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_live_mark import chunk_log_terms,streaming_raw_mark,POPULATION_ORIGIN
from cf4_r2_velocity_closure import mixture_components


class RawMixtureConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):jax.config.update('jax_enable_x64',True)

    def setUp(self):
        r=jnp.linspace(.001,400.,4001)
        self.g=dict(observer=jnp.full(3,192.),box_size_cMpc_h=384.,
            hubble_km_s_Mpc=74.6,little_h=.746,radius_table_cMpc_h=r,
            modulus_table_h=5*jnp.log10(r)+25.,redshift_table=r/3000.,grid_size=128,
            sigma_los_km_s=20.,finite_reference_interval=(-25.,-21.),mstar=-23.28,alpha=-.94)
        self.pos=jnp.array([[222.4,192.8,192.9],[223.2,192.9,192.8]])
        self.vel=jnp.zeros((2,3));self.mass=jnp.ones((5,2));self.sky=jnp.ones((2,2))
        self.var=jnp.array([[20000.,30000.,40000.],[40000.,10000.,20000.]])
        self.o=dict(voxel=jnp.array([74,64,64]),radius=jnp.array(31.),dz=jnp.array(31.),
            ksmag=jnp.array(9.5),x=jnp.array([.3,2.2,2.7]),
            error_covariance=jnp.eye(3)*.002,richness=jnp.array(0.),
            cut_lower=jnp.array([2.,1.9]),cut_upper=jnp.array([4.,2.5]))
        self.closure=dict(core_sigma_km_s=20.,dispersion_scale=.8,broad_fraction=.7)

    def test_unnormalized_component_mixture_and_streaming(self):
        args=(jnp.asarray(POPULATION_ORIGIN),self.pos,self.vel,self.mass,self.sky,self.o)
        common=dict(population=1,cut_order=16)
        mixed=chunk_log_terms(*args,geometry=self.g,**common,
            source_velocity_variances_km2_s2=self.var,velocity_closure=self.closure)
        reference=[]
        for weight,width,scale in mixture_components(self.closure):
            g=dict(self.g,sigma_los_km_s=width,deposition='voxel_cdf',
                source_velocity_variances_km2_s2=self.var,dispersion_scale=scale)
            reference.append(jnp.asarray(chunk_log_terms(*args,geometry=g,**common))+jnp.log(weight))
        expected=jax.scipy.special.logsumexp(jnp.stack(reference),axis=0)
        self.assertTrue(np.isfinite(np.asarray(mixed)).all())
        np.testing.assert_allclose(mixed,expected,atol=2e-12)
        stream=streaming_raw_mark(args[0],self.pos[None],self.vel[None],self.mass[None],
            self.sky[None],self.o,geometry=self.g,**common,return_log_terms=True,
            source_velocity_variances_km2_s2=self.var[None],velocity_closure=self.closure)
        np.testing.assert_allclose(stream,mixed,atol=2e-12)

    def test_packed_rows_and_variance_adjoint(self):
        o={k:v[None] for k,v in self.o.items()}
        def score(logscale):
            closure=dict(self.closure,dispersion_scale=.8*jnp.exp(logscale))
            # Every (bin,source) occurs once, just as in the live padded packs.
            ids=jnp.tile(jnp.arange(2),5);bins=jnp.repeat(jnp.arange(5),2)
            return streaming_raw_mark(jnp.asarray(POPULATION_ORIGIN),self.pos[ids][None],
                self.vel[ids][None],self.mass[:,ids][None],self.sky[:,ids][None],o,
                population=1,geometry=self.g,component_bins=bins[None],
                component_rows=jnp.zeros((1,10),dtype=jnp.int32),cut_order=16,
                source_velocity_variances_km2_s2=self.var[ids][None],velocity_closure=closure)[0]
        direct=chunk_log_terms(jnp.asarray(POPULATION_ORIGIN),self.pos,self.vel,self.mass,self.sky,self.o,
            population=1,geometry=self.g,cut_order=16,
            source_velocity_variances_km2_s2=self.var,velocity_closure=self.closure)
        np.testing.assert_allclose(score(0.),direct[0]-direct[1],atol=2e-12)
        f=jax.jit(jax.value_and_grad(score));value,gradient=f(0.)
        eps=1e-4;fd=(score(eps)-score(-eps))/(2*eps)
        self.assertTrue(np.isfinite([float(value),float(gradient)]).all())
        np.testing.assert_allclose(gradient,fd,rtol=1e-5,atol=1e-9)


if __name__=='__main__':unittest.main()
