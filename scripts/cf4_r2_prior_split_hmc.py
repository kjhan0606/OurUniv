"""Finite-dimensional, fixed-metric Gaussian-prior split HMC mechanics.

Canonical coordinates ALL have independent N(0,1) priors. Never feed the
optimizer's 100*tracer coordinates directly. C is inverse momentum mass, not
a claimed posterior covariance. It must stay fixed during production steps.
The oracle returns the FULL negative log target and canonical gradient.
Gaussian-prior/kinetic flow is exact; likelihood kicks use grad(U)-q.
The final Metropolis correction uses the FULL Hamiltonian, without tempering.

This is a finite-dimensional splitting implementation, not a claim of the
dimension-independent convergence results in Beskos et al.(2011),
https://authors.library.caltech.edu/records/kbprr-5m424 . Actual CF4 mixing,
metric construction, shared-data calibration and production are separate.
"""
import numpy as np
from scipy.fft import fftn, ifftn


def canonical_from_optimizer_oracle(optimizer_oracle,n_ic):
    """Adapt this project's IC +100*9tracer +zero optimizer coordinates.

    The constant coordinate Jacobian has no effect on MH energy differences;
    gradients DO require the chain rule. Priors are unchanged, not rescaled.
    """
    def oracle(q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(n_ic+10,):
            raise ValueError('expected IC plus9 white tracers and1 white zero')
        x=q.copy(); x[n_ic:n_ic+9]*=100.
        value,gradient=optimizer_oracle(x)
        if float(value)==np.inf:
            return value,None
        gradient=np.array(gradient,dtype=float,copy=True)
        gradient[n_ic:n_ic+9]*=100.
        return value,gradient
    return oracle


def inverse_laplacian_metric_symbol(n,fundamental_mass=6000.):
    """Positive even proposal metric guess, NOT a fitted Hessian or prior.

    M(k)=1+(fundamental_mass-1)/|integer_mode|^2 for k!=0; M(0)=1.
    The finite-grid high-k mass is greater than1, not exactly the prior mass.
    """
    if not isinstance(n,int) or n<2 or not np.isfinite(fundamental_mass) or fundamental_mass<1:
        raise ValueError('grid>=2 and finite fundamental mass>=1 required')
    modes=np.meshgrid(*[np.fft.fftfreq(n)*n]*3,indexing='ij',sparse=True)
    k2=sum(k*k for k in modes)
    mass=1.+np.divide(fundamental_mass-1.,k2,out=np.zeros_like(k2),where=k2>0)
    return 1./mass


class FixedSplitMetric:
    def __init__(self, ic_inverse_mass, nuisance_inverse_mass):
        c=np.asarray(ic_inverse_mass,dtype=float)
        b=np.asarray(nuisance_inverse_mass,dtype=float)
        if (c.ndim!=3 or len(set(c.shape))!=1 or not np.isfinite(c).all()
                or np.any(c<=0) or b.ndim!=2 or b.shape[0]!=b.shape[1]
                or not np.isfinite(b).all() or not np.allclose(b,b.T,rtol=0,atol=1e-12)):
            raise ValueError('positive Fourier symbol and symmetric nuisance inverse mass required')
        reflected=c
        for axis in range(3):
            reflected=np.take(reflected,(-np.arange(c.shape[axis]))%c.shape[axis],axis=axis)
        if not np.allclose(c,reflected,rtol=0,atol=1e-12):
            raise ValueError('Fourier metric must preserve real fields')
        eigen,vectors=np.linalg.eigh(b)
        if np.any(eigen<=0):
            raise ValueError('nuisance inverse mass must be SPD; no eigenvalue clipping')
        self.c,self.b=c.copy(),b.copy()
        self.root,self.eigen,self.vectors=np.sqrt(c),eigen,vectors
        self.n_ic=c.size
        self.size=c.size+len(eigen)

    def split(self,vector):
        x=np.asarray(vector,dtype=float)
        if x.shape!=(self.size,) or not np.isfinite(x).all():
            raise ValueError('finite canonical vector with matching dimension required')
        return x[:self.n_ic].reshape(self.c.shape),x[self.n_ic:]

    def momentum(self,rng):
        z=rng.standard_normal(self.size)
        field,nuisance=self.split(z)
        p=ifftn(fftn(field,norm='ortho',workers=1)/self.root,norm='ortho',workers=1).real
        tail=self.vectors@(nuisance/np.sqrt(self.eigen))
        return np.r_[p.ravel(),tail]

    def kinetic(self,p):
        field,tail=self.split(p)
        spectral=fftn(field,norm='ortho',workers=1)
        return .5*float(np.sum(self.c*np.abs(spectral)**2)+tail@self.b@tail)

    def prior_flow(self,q,p,step):
        """Exact flow of (q.q + p.C.p)/2 for positive OR negative time."""
        if not np.isfinite(step):
            raise ValueError('finite integration step required')
        field,tail=self.split(q); momentum,ptail=self.split(p)
        a=fftn(field,norm='ortho',workers=1)
        b=fftn(momentum,norm='ortho',workers=1)
        cosine,sine=np.cos(step*self.root),np.sin(step*self.root)
        anew=cosine*a+sine*self.root*b
        bnew=cosine*b-sine/self.root*a
        field=ifftn(anew,norm='ortho',workers=1).real
        momentum=ifftn(bnew,norm='ortho',workers=1).real
        a,b=self.vectors.T@tail,self.vectors.T@ptail
        root=np.sqrt(self.eigen)
        cosine,sine=np.cos(step*root),np.sin(step*root)
        tail=self.vectors@(cosine*a+sine*root*b)
        ptail=self.vectors@(cosine*b-sine/root*a)
        return np.r_[field.ravel(),tail],np.r_[momentum.ravel(),ptail]


def checked_oracle(oracle,q):
    value,gradient=oracle(q)
    value=float(value)
    if value==np.inf:
        return value,None  # A true zero-density target point; NOT a floor.
    gradient=np.asarray(gradient,dtype=float)
    if not np.isfinite(value) or gradient.shape!=q.shape or not np.isfinite(gradient).all():
        raise FloatingPointError('nonfinite target or derivative at finite-support state')
    return value,gradient


def split_trajectory(oracle,metric,q,p,step,steps,initial_evaluation=None):
    """Reversible kick/rotation/kick map; caller owns deterministic support."""
    if not isinstance(steps,int) or steps<1 or not np.isfinite(step) or step==0:
        raise ValueError('nonzero finite step and positive integer length required')
    q,p=np.array(q,dtype=float,copy=True),np.array(p,dtype=float,copy=True)
    metric.split(q); metric.split(p)
    value,gradient=(checked_oracle(oracle,q) if initial_evaluation is None else initial_evaluation)
    gradient=None if gradient is None else np.asarray(gradient,dtype=float)
    if (not np.isfinite(value) or gradient is None or gradient.shape!=q.shape
            or not np.isfinite(gradient).all()):
        raise ValueError('trajectory must start inside finite differentiable support')
    p-=.5*step*(gradient-q)
    for i in range(steps):
        q,p=metric.prior_flow(q,p,step)
        value,gradient=checked_oracle(oracle,q)
        if value==np.inf:
            return q,p,value,None
        p-=(.5 if i==steps-1 else 1.)*step*(gradient-q)
    return q,p,value,gradient


def split_hmc_step(oracle,metric,q,value,gradient,rng,*,step,steps,endpoint_value=None):
    """One fixed-metric MH-corrected proposal, including rejected states."""
    p=metric.momentum(rng)
    start=float(value)+metric.kinetic(p)
    if not np.isfinite(start):
        raise FloatingPointError('nonfinite initial Hamiltonian')
    calls=0
    def counted(position):
        nonlocal calls
        calls+=1
        return oracle(position)
    proposed,pend,new_value,new_gradient=split_trajectory(
        counted,metric,q,p,step,steps,initial_evaluation=(value,gradient))
    if endpoint_value is not None:
        exact=float(endpoint_value(proposed))
        if not ((exact==np.inf and new_value==np.inf) or
                (np.isfinite(exact) and np.isfinite(new_value)
                 and np.isclose(exact,new_value,rtol=0.,atol=1e-7))):
            raise FloatingPointError('HMC endpoint value/derivative primal disagreement')
        new_value=exact  # MH uses the independently evaluated full target.
    delta=(new_value+metric.kinetic(pend)-start if np.isfinite(new_value) else np.inf)
    if np.isfinite(new_value) and not np.isfinite(delta):
        raise FloatingPointError('nonfinite Hamiltonian error at finite target')
    log_accept=min(0.,-delta)
    accepted=bool(np.log(rng.uniform())<log_accept)
    if accepted:
        q,value,gradient=proposed,new_value,new_gradient
    return q,value,gradient,dict(accepted=accepted,log_acceptance=log_accept,
                                energy_error=delta,force_evaluations=calls)


class PilotBudgetStop(RuntimeError):
    pass


def bounded_split_pilot(oracle,metric,q,value,gradient,rng,*,step=.1,warmup=16,
                        retained=16,steps=2,seconds_left,callback,endpoint_value=None):
    """Bounded mechanics pilot; short retained trace is NOT posterior UQ.

    Adapt only the discarded warmup, then freeze the last step. Fixed metric
    throughout. A budget expiry mid-trajectory returns the last accepted
    state; a numerical/capacity failure propagates instead of truncating target.
    """
    if min(warmup,retained)<0 or warmup+retained<1 or not 1e-6<=step<=.3:
        raise ValueError('invalid pilot length/initial step')
    def checked(q):
        if seconds_left()<=0:
            raise PilotBudgetStop()
        return oracle(q)
    trace=[]; rejection_streak=0
    message='proposal limit'
    for i in range(warmup+retained):
        previous=q.copy()
        used_step=step
        try:
            q,value,gradient,info=split_hmc_step(checked,metric,q,value,gradient,rng,
                step=used_step,steps=steps,endpoint_value=endpoint_value)
        except PilotBudgetStop:
            message='application time budget'; break
        delta=q-previous
        cube=q[:metric.n_ic].reshape(metric.c.shape)
        wave=np.cos(2*np.pi*np.arange(cube.shape[0])/cube.shape[0])
        # Same normalized cosine direction as the historical curvature probe.
        norm=np.sqrt(cube.shape[0]**2*np.sum(wave**2))
        projections=[float(np.dot(cube.sum(axis=tuple(a for a in range(3) if a!=axis)),wave)/norm)
                     for axis in range(3)]
        dcube=delta[:metric.n_ic].reshape(cube.shape)
        jumps=[float(np.dot(dcube.sum(axis=tuple(a for a in range(3) if a!=axis)),wave)/norm)
               for axis in range(3)]
        row=dict(iteration=i+1,warmup=i<warmup,step_size=used_step,
            objective=float(value),**info,IC_mean_square=float(np.mean(cube*cube)),
            canonical_jump_rms=float(np.sqrt(np.mean(delta*delta))),
            fundamental_cosine=projections,fundamental_cosine_jump=jumps,
            white_nuisance=q[metric.n_ic:].tolist(),white_zero_jump=float(delta[-1]),
            canonical_nuisance_jump_l2=float(np.linalg.norm(delta[metric.n_ic:])))
        trace.append(row); callback(q,value,gradient,row)
        if i<warmup:
            probability=float(np.exp(info['log_acceptance']))
            step*=np.exp((probability-.65)/np.sqrt(i+1.))
            rejection_streak=0 if info['accepted'] else rejection_streak+1
            if rejection_streak==3:
                step*=.5; rejection_streak=0
            step=float(np.clip(step,1e-6,.3))
    return q,value,gradient,trace,message
