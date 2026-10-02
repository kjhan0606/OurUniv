"""Training-only native velocity closure readout; no count/field fitting."""
import json
import os
from pathlib import Path
import resource
import time
from itertools import product

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp

from cf4_r2_native_mock import place_native_halves, distance_tables, observe

BASE=Path('/gpfs/kjhan/CF4/z0_density')
ENDPOINT=BASE/'r2_native_rsd_mock_fit_20261002_v2/result.json'
OUT=Path(os.environ.get('CF4_R2_OUT_DIR',str(BASE/'r2_native_velocity_closure_20261002_v1')))


def fit_conditional_mixture(x,variance):
    """Narrow core plus broad width tied to local matter variance proxy."""
    def objective(q):
        core=np.exp(q[0]);scale=np.exp(q[1]);broad=np.sqrt(core**2+scale**2*variance)
        logn=lambda s:-.5*(x/s)**2-np.log(s)-.5*np.log(2*np.pi)
        return -np.sum(logsumexp(np.stack((logn(core)-np.logaddexp(0,q[2]),
            logn(broad)-np.logaddexp(0,-q[2]))),axis=0))
    fits=[minimize(objective,[np.log(20.),np.log(scale),0.],method='L-BFGS-B',
        bounds=[(np.log(.1),np.log(3000.)),(np.log(.001),np.log(100.)),(-12.,12.)],
        options={'maxiter':200,'ftol':1e-12}) for scale in (.5,1.)]
    best=min(fits,key=lambda r:r.fun)
    core=np.exp(best.x[0]);scale=np.exp(best.x[1]);broad=np.sqrt(core**2+scale**2*variance)
    return dict(success=bool(best.success),message=str(best.message),
        sigma_core=float(core),matter_dispersion_scale=float(scale),
        fraction_broad=float(expit(best.x[2])),log_likelihood=float(-best.fun),AIC=float(6+2*best.fun),
        broad_sigma_percentiles=np.percentile(broad,[0,50,90,99,100]).tolist(),
        competing_start_log_likelihoods=[float(-r.fun) for r in fits],
        limit='diagonal matter velocity covariance only; not exact LOS covariance or calibrated R2 prior')


def fit_laws(x):
    """Zero-mean laws; common narrow/broad labels, finite-variance t comparator."""
    if len(x)<50 or not np.isfinite(x).all():raise ValueError('insufficient residuals')
    rms=float(np.sqrt(np.mean(x*x)))
    gaussian_ll=float(np.sum(-.5*(x/rms)**2-np.log(rms)-.5*np.log(2*np.pi)))
    def mixture(q):
        core=np.exp(q[0]);broad=core+np.exp(q[1]);f=expit(q[2])
        logn=lambda s:-.5*(x/s)**2-np.log(s)-.5*np.log(2*np.pi)
        # stable weights, even for optimizer trials in very narrow components
        return -np.sum(logsumexp(np.stack((logn(core)-np.logaddexp(0,q[2]),
            logn(broad)-np.logaddexp(0,-q[2]))),axis=0))
    starts=[np.array([np.log(40.),np.log(300.),-1.]),
            np.array([np.log(max(rms/3,1.)),np.log(max(rms,1.)),0.])]
    fits=[minimize(mixture,q,method='L-BFGS-B',bounds=[(np.log(.1),np.log(3000.)),
        (np.log(.1),np.log(5000.)),(-12.,12.)],options={'maxiter':200,'ftol':1e-12}) for q in starts]
    best=min(fits,key=lambda r:r.fun)
    core=np.exp(best.x[0]);broad=core+np.exp(best.x[1]);fraction=expit(best.x[2])
    def student(q):
        scale=np.exp(q[0]);nu=2+np.exp(q[1]);u=x/scale
        return -np.sum(gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(nu*np.pi)
            -np.log(scale)-.5*(nu+1)*np.log1p(u*u/nu))
    t=minimize(student,[np.log(max(rms/2,1.)),np.log(2.)],method='L-BFGS-B',
        bounds=[(np.log(.1),np.log(3000.)),(-10.,np.log(10000.))],
        options={'maxiter':200,'ftol':1e-12})
    return dict(n=len(x),mean=float(x.mean()),rms=rms,
        absolute_percentiles=np.percentile(np.abs(x),[50,90,95,99,100]).tolist(),
        gaussian=dict(sigma=rms,log_likelihood=gaussian_ll,AIC=2-2*gaussian_ll),
        mixture=dict(success=bool(best.success),message=str(best.message),
            sigma_core=float(core),sigma_broad=float(broad),fraction_broad=float(fraction),
            log_likelihood=float(-best.fun),AIC=float(6+2*best.fun),
            competing_start_log_likelihoods=[float(-r.fun) for r in fits]),
        student_t=dict(success=bool(t.success),scale=float(np.exp(t.x[0])),
            nu=float(2+np.exp(t.x[1])),log_likelihood=float(-t.fun),AIC=float(4+2*t.fun)))


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    start=time.monotonic();endpoint=json.loads(ENDPOINT.read_text())
    OUT.mkdir(exist_ok=False)
    with np.load(endpoint['source_galaxies'],allow_pickle=False) as f:
        g={k:f[k] for k in f.files}
    with h5py.File(endpoint['source_matter'],'r') as f:
        m=f['coarse'][:];omega=float(f.attrs['Omega_m'])
    native=g['position']%75.;pos=place_native_halves(native)
    radial,z,modulus=distance_tables(omega)
    mock=observe(pos,g['velocity'],g['K_h_proxy'],radial,z,modulus)
    cells=np.floor(native/1.5).astype(int)
    mass=m[0];moment=np.moveaxis(m[1:4],0,-1)
    ngp=moment[tuple(cells.T)]/mass[tuple(cells.T)][:,None]
    # Interpolate mass and momentum, THEN divide. Not an unweighted average
    # of cell velocities, which would measure a different operator.
    second=np.moveaxis(m[4:7],0,-1)
    numerator=np.zeros_like(ngp);second_numerator=np.zeros_like(ngp);denominator=np.zeros(len(native))
    for offset in product((-1,0,1),repeat=3):
        unwrapped=cells+offset;index=unwrapped%50
        distance=np.abs(native/1.5-.5-unwrapped)
        weights=np.where(distance<.5,.75-distance**2,
            np.where(distance<1.5,.5*(1.5-distance)**2,0.)).prod(axis=1)
        denominator+=weights*mass[tuple(index.T)]
        numerator+=weights[:,None]*moment[tuple(index.T)]
        second_numerator+=weights[:,None]*second[tuple(index.T)]
    interpolated=numerator/denominator[:,None]
    true_train=pos[:,1]-192.<-6.
    resolved=g['star_count']>=300
    # Intrinsic K range, decided before apparent/radius/observed-K selection.
    luminosity=(g['K_h_proxy']>=-25.)&(g['K_h_proxy']<-21.)
    source_train=true_train&resolved&luminosity
    edge=np.minimum.reduce((native[:,0]%37.5,37.5-native[:,0]%37.5,
        native[:,1],75-native[:,1],native[:,2],75-native[:,2]))
    report=dict(job_id=os.environ['SLURM_JOB_ID'],endpoint=str(ENDPOINT),
        classification='PRESELECTION_NATIVE_TRAINING_RESIDUAL_NOT_R2_PRIOR',
        selection='true y<-6; >=300 stars; -25<=intrinsic K_h<-21, before observed selection',
        no_count_refit=True,no_field_fit=True,no_actual_CF4_outcomes=True,
        central_satellite_labels=False,results={},limits=endpoint['limits'],
        MW_M31=endpoint['MW_M31'],M33=endpoint['M33'])
    residuals={}
    for name,reference,raw_second in (('cell_mean',ngp,second[tuple(cells.T)]/mass[tuple(cells.T)][:,None]),
        ('TSC_mass_momentum',interpolated,second_numerator/denominator[:,None])):
        residual=mock['coherent_vlos']-np.sum(reference*mock['direction'],axis=1)
        residuals[name]=residual[source_train]
        variance=raw_second-reference**2
        if np.any(variance < -1e-8*np.maximum(1.,np.abs(raw_second))):
            raise ValueError('matter diagonal second moments violate nonnegative variance')
        los_variance=np.sum(np.maximum(variance,0.)*mock['direction']**2,axis=1)
        residuals[name+'_matter_LOS_variance_proxy']=los_variance[source_train]
        masks={'preselection':source_train,'interior_3Mpc':source_train&(edge>=3.),
            'edge_3Mpc':source_train&(edge<3.),
            'observed_selected_comparison_only':source_train&mock['selected']}
        report['results'][name]={key:fit_laws(residual[mask]) for key,mask in masks.items()}
        report['results'][name]['conditional_mixture']=fit_conditional_mixture(
            residual[source_train],los_variance[source_train])
        rho=mass[tuple(cells.T)]/mass.mean()
        report['results'][name]['density_split']={label:dict(n=int(mask.sum()),
            rms=float(np.sqrt(np.mean(residual[mask]**2))),
            absolute_p99=float(np.percentile(np.abs(residual[mask]),99)))
            for label,mask in (('rho_below1',source_train&(rho<1)),
                               ('rho_atleast1',source_train&(rho>=1))) if mask.any()}
    np.savez_compressed(OUT/'training_residuals.npz',**residuals,
        native_id=g['native_id'][source_train],K_h=g['K_h_proxy'][source_train],
        rho=rho[source_train],edge_distance=edge[source_train])
    report['elapsed_seconds']=time.monotonic()-start
    report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
