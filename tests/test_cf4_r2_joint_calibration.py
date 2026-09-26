import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from cf4_r2_joint_calibration import group_scores, white_logdensity


def fixture():
    d = jnp.array([[20.,30.,40.],[25.,35.,45.]])
    return dict(distance=d,log_distance_weight=2*jnp.log(d),
        redshift_logkernel=jnp.array([[-1.,0.,-2.],[-2.,0.,-1.]]),
        predicted_modulus=5*jnp.log10(d/.746)+25,
        row_group=jnp.array([0,0,1]),dz_row=jnp.array([30.,30.,35.]),
        eta_mean=jnp.array([.03,-.01,.02]),eta_std=jnp.array([.1,.12,.08]),
        eta_alpha=jnp.zeros(3),anchor_group=jnp.array([0,1]),
        anchor_modulus=jnp.array([33.1,33.5]),anchor_error=jnp.array([.2,.3]),
        anchor_method=jnp.array([0,1]),group_holdout=jnp.array([False,True]))


def test_shared_correlated_target():
    data = fixture()
    data = {k:(v[:,1:2] if v.ndim == 2 else v) for k,v in data.items()}
    data['group_holdout'] = jnp.array([False,False])
    L = jnp.array([[.004,0.,0.,0.],[.02,.2,0.,0.],[-.01,.04,.3,0.],[0.,0.,.2,1.]])
    mean = jnp.zeros(4)
    # Exact linear Gaussian reference at known distance, including correlated
    # global calibration. Rows0/1 share beta; no repeated prior per group.
    A = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    y = np.r_[data['eta_mean'],data['anchor_modulus']-data['predicted_modulus'][:,0]]
    error = np.r_[data['eta_std'],data['anchor_error']]
    design = A @ np.asarray(L)
    precision = np.eye(4)+(design/error[:,None]).T@(design/error[:,None])
    centre = np.linalg.solve(precision,design.T@(y/error**2))
    f = lambda w: white_logdensity(w,data,mean,L)
    for white in (jnp.zeros(4),jnp.array([.7,-.2,.4,1.]),jnp.array(centre)):
        expected = -.5*(white-centre)@precision@(white-centre)+.5*centre@precision@centre
        np.testing.assert_allclose(f(white)-f(jnp.zeros(4)),expected,atol=1e-11)
    np.testing.assert_allclose(jax.hessian(f)(jnp.array(centre)),-precision,atol=1e-10)


def test_selected_shape_and_training_ownership():
    data = fixture()
    p = jnp.array([.004,.1,-.2,.7])
    got = np.asarray(group_scores(p,data))
    expected = []
    for g in range(2):
        d = np.asarray(data['distance'][g])
        base = np.asarray(data['log_distance_weight'][g]+data['redshift_logkernel'][g])+float(p[-1])*np.log(d/100.)
        marks = np.zeros(3)
        for i in np.flatnonzero(np.asarray(data['row_group']) == g):
            eta = np.log10(float(data['dz_row'][i])/d)+float(p[0])
            m,s = float(data['eta_mean'][i]),float(data['eta_std'][i])
            marks += -.5*((eta-m)**2-m**2)/s**2
        j = g
        mu = np.asarray(data['predicted_modulus'][g])+float(p[1+j])
        obs,err = float(data['anchor_modulus'][j]),float(data['anchor_error'][j])
        marks += -.5*((obs-mu)**2-(obs-35.)**2)/err**2
        expected.append(logsumexp(base+marks)-logsumexp(base))
    np.testing.assert_allclose(got,expected,atol=1e-11)
    shifted = dict(data,log_distance_weight=data['log_distance_weight']+jnp.array([[11.],[-9.]]))
    np.testing.assert_allclose(group_scores(p,shifted),got,atol=1e-11)
    f = lambda w: white_logdensity(w,data,jnp.zeros(4),jnp.eye(4))
    # Heldout source/anchor mutations cannot change the training target.
    changed = dict(data,eta_mean=data['eta_mean'].at[2].set(.4),
                   anchor_modulus=data['anchor_modulus'].at[1].set(40.))
    np.testing.assert_allclose(f(p),white_logdensity(p,changed,jnp.zeros(4),jnp.eye(4)),atol=1e-11)
    gradient = np.asarray(jax.grad(f)(p))
    for k in range(4):
        step = jnp.eye(4)[k]*1e-5
        np.testing.assert_allclose(gradient[k],(f(p+step)-f(p-step))/2e-5,rtol=2e-6,atol=1e-7)


if __name__ == '__main__':
    test_shared_correlated_target()
    test_selected_shape_and_training_ownership()
    print('PASS: shared correlated prior / exact Gaussian target; selection normalization / holdout / joint gradient',flush=True)
