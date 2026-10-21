import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
from cf4_r2_joint_calibration import group_scores
from cf4_r2_hierarchical_marks import (shifted_group_scores, decode_hyper,
    joint_logdensity, marginal_group_scores)
from cf4_r2_live_field_marks import bind_state_geometry


def fixture():
    d = jnp.array([[20.,30.,40.],[25.,35.,45.]])
    g = dict(distance=d,quadrature_weight=jnp.ones_like(d),
        directions=jnp.array([[1.,0.,0.],[0.,1.,0.]]),zcos=d/2997.92458,
        redshift_sufficient=jnp.array([[1/400.**2,3000.,0.,np.log(400.**2),1.],
                                     [1/400.**2,3500.,0.,np.log(400.**2),1.]]),
        row_group=jnp.array([0,0,1]),dz_row=jnp.array([30.,30.,35.]),
        eta_mean=jnp.array([.03,-.01,.02]),eta_std=jnp.array([.1,.12,.08]),eta_alpha=jnp.zeros(3),
        predicted_modulus=5*jnp.log10(d/.746)+25,
        anchor_group=jnp.array([0,1]),anchor_modulus=jnp.array([33.1,33.5]),
        anchor_error=jnp.array([.2,.3]),anchor_method=jnp.array([0,1]),
        group_holdout=jnp.array([False,True]),train_group_index=jnp.array([0]))
    xyz = jnp.stack(jnp.meshgrid(*([jnp.arange(8)]*3),indexing='ij'),axis=0)
    rho = 1+.2*jnp.sin(xyz[0]+xyz[1])
    v = 30*jnp.cos(xyz)
    return g,rho,v


def test_zero_offset_recovers_original():
    g,rho,v = fixture()
    bound = bind_state_geometry(rho,v,g,box=160.)
    p = jnp.array([.003,.1,-.2,.7])
    np.testing.assert_allclose(shifted_group_scores(p,jnp.zeros(2),bound),
                               group_scores(p,bound),atol=1e-12)
    shifted = dict(bound,log_distance_weight=bound['log_distance_weight']+jnp.array([[17.],[-13.]]))
    np.testing.assert_allclose(shifted_group_scores(p,jnp.array([.02,-.01]),bound),
                               shifted_group_scores(p,jnp.array([.02,-.01]),shifted),atol=1e-12)


def test_shared_not_independent_gaussian_reference():
    g,rho,v = fixture()
    g = {k:(x[:,1:2] if x.ndim == 2 and k not in ('directions','redshift_sufficient') else x)
         for k,x in g.items()}
    bound = bind_state_geometry(rho,v,g,box=160.)
    p = jnp.array([.013,.1,-.2,.7])
    tau = .04
    nodes,weights = np.polynomial.hermite.hermgauss(33)
    got = np.array([shifted_group_scores(p,jnp.full(2,np.sqrt(2)*tau*x),bound)
                    for x in nodes])
    integrated = logsumexp(got+np.log(weights/np.sqrt(np.pi))[:,None],axis=0)
    expected,wrong = [],[]
    for group in range(2):
        rows = np.asarray(g['row_group']) == group
        y,s = np.asarray(g['eta_mean'])[rows],np.asarray(g['eta_std'])[rows]
        eta = np.log10(np.asarray(g['dz_row'])[rows]/float(g['distance'][group,0]))+float(p[0])
        cov = np.diag(s*s)+tau*tau*np.ones((len(s),len(s)))
        mu = float(g['predicted_modulus'][group,0]+p[group+1])
        obs,err = float(g['anchor_modulus'][group]),float(g['anchor_error'][group])
        nonfp = -.5*((obs-mu)**2-(obs-35.)**2)/err**2
        reference = norm.logpdf(y,0,s).sum()
        expected.append(multivariate_normal.logpdf(y,eta,cov)-reference+nonfp)
        wrong.append(norm.logpdf(y,eta,np.sqrt(s*s+tau*tau)).sum()-reference+nonfp)
    np.testing.assert_allclose(integrated,expected,atol=1e-10)
    assert abs(integrated[0]-wrong[0]) > 1e-4
    # Non-FP marks are NOT shifted by the FP common offset.
    changed = dict(bound,anchor_modulus=bound['anchor_modulus']+1.)
    contrast = lambda u: shifted_group_scores(p,jnp.full(2,u),changed)-shifted_group_scores(p,jnp.full(2,u),bound)
    np.testing.assert_allclose(contrast(0.),contrast(.03),atol=1e-11)


def test_joint_derivatives_and_holdout():
    g,rho,v = fixture()
    sd = jnp.array([.004,1.,1.,2.])
    # One IC-like field coordinate, six hyper coordinates, one training group.
    x = jnp.array([.1,.2,.1,-.1,.3,.25,-.2,.6])
    def target(x,data):
        r = rho*jnp.exp(x[0]*jnp.cos(jnp.arange(8))[:,None,None])
        return joint_logdensity(x[:1],x[1:7],x[7:],r,v,data,sd,box=160.)
    grad = np.asarray(jax.grad(target)(x,g))
    for k in range(len(x)):
        step = jnp.eye(len(x))[k]*1e-5
        np.testing.assert_allclose(grad[k],(target(x+step,g)-target(x-step,g))/2e-5,
                                   rtol=2e-5,atol=1e-7)
    changed = dict(g,eta_mean=g['eta_mean'].at[2].set(.6),
                  anchor_modulus=g['anchor_modulus'].at[1].set(40.))
    np.testing.assert_allclose(target(x,g),target(x,changed),atol=1e-12)
    np.testing.assert_allclose(jax.grad(target)(x,g),jax.grad(target)(x,changed),atol=1e-12)
    assert abs(grad[5]+float(x[5])) > 1e-6  # selected slope has data response
    assert abs(grad[6]+float(x[6])) > 1e-6  # shared scatter has data response


def test_heldout_offset_is_integrated():
    g,rho,v = fixture()
    sd = jnp.array([.004,1.,1.,2.])
    w = jnp.array([.1,.2,-.1,.3,.2,-.1])
    p,b,tau = decode_hyper(w,sd)
    bound = bind_state_geometry(rho,v,g,box=160.,selected_bias=b)
    nodes,weights = np.polynomial.hermite.hermgauss(17)
    nodes,logw = jnp.asarray(np.sqrt(2)*nodes),jnp.log(jnp.asarray(weights))
    expected = logsumexp(np.array([shifted_group_scores(p,jnp.full(2,tau*n),bound)
        for n in nodes])+np.asarray(logw)[:,None],axis=0)-logsumexp(np.asarray(logw))
    np.testing.assert_allclose(marginal_group_scores(rho,v,w,g,sd,nodes,logw,box=160.),expected,atol=1e-11)


if __name__ == '__main__':
    test_zero_offset_recovers_original()
    test_shared_not_independent_gaussian_reference()
    test_joint_derivatives_and_holdout()
    test_heldout_offset_is_integrated()
    print('PASS: four shared-error/selection/ownership/gradient/holdout controls',flush=True)
