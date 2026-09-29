"""Read two already-computed immutable force anchors; no new target evaluation.

A secant is one-direction curvature evidence, NOT a validated new sampler or
posterior metric. Running chains/anchors are never modified.
"""
import json
import os
from pathlib import Path
import numpy as np
from scipy import fft

BASE=Path('/gpfs/kjhan/CF4/z0_density');N=256;NIC=N**3


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    paths=[BASE/'r2_n256_affine_pilot_v1',BASE/'r2_n256_chain_a_v1']
    with np.load(paths[0]/'fixed_force_anchor.npz',allow_pickle=False) as f:
        q0=f['anchor'];d0=f['gradient_correction']
    with np.load(paths[1]/'fixed_force_anchor.npz',allow_pickle=False) as f:
        q1=f['anchor'];d1=f['gradient_correction']
    with np.load(paths[0]/'accepted_checkpoint.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(q1,f['canonical'])
    pilot=json.loads((paths[0]/'result.json').read_text())
    if pilot['status']!='N256_RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR':raise ValueError('completed anchor source required')
    dq=q1-q0;dd=d1-d0
    if not np.isfinite(dq).all() or not np.isfinite(dd).all():raise ValueError('finite anchor differences required')
    residual=sum(r['fine_force_correction_change'] for r in pilot['trace'])
    prediction=.5*float(dq@dd)
    blocks={}
    for name,sl in [('IC',slice(None,NIC)),('nuisance',slice(NIC,None))]:
        x,y=dq[sl],dd[sl];xx=float(x@x);yy=float(y@y);xy=float(x@y)
        blocks[name]=dict(displacement_norm=float(np.sqrt(xx)),gradient_change_norm=float(np.sqrt(yy)),
            directional_secant_curvature=xy/xx,quadratic_energy_prediction=.5*xy,
            cosine=xy/np.sqrt(xx*yy) if yy>0 else None)
    qhat=fft.fftn(dq[:NIC].reshape((N,)*3),norm='ortho',workers=2)
    dhat=fft.fftn(dd[:NIC].reshape((N,)*3),norm='ortho',workers=2)
    modes=np.fft.fftfreq(N)*N
    k2=modes[:,None,None]**2+modes[None,:,None]**2+modes[None,None,:]**2
    bins=[0.,4.,16.,64.,256.,1024.,4096.,16384.,float('inf')]
    spectral=[]
    for lower,upper in zip(bins[:-1],bins[1:]):
        mask=(k2>=lower)&(k2<upper);x=qhat[mask];y=dhat[mask]
        xx=float(np.vdot(x,x).real);yy=float(np.vdot(y,y).real);xy=float(np.vdot(x,y).real)
        coefficient=xy/xx
        spectral.append(dict(k_integer_lower=float(np.sqrt(lower)),
            k_integer_upper=float(np.sqrt(upper)) if np.isfinite(upper) else None,
            modes=int(mask.sum()),secant_coefficient=coefficient,
            relative_gradient_residual=float(np.linalg.norm(y-coefficient*x)/np.sqrt(yy)) if yy>0 else None,
            quadratic_energy_prediction=.5*xy))
    report=dict(status='SINGLE_SECANT_DIAGNOSTIC_NOT_A_NEW_KERNEL',job_id=os.environ['SLURM_JOB_ID'],
        R2_complete=False,target_evaluations=0,PM_evolutions=0,chain_changes=0,
        measured_fine_minus_affine_change=residual,
        trapezoidal_gradient_prediction=prediction,prediction_minus_actual=prediction-residual,
        blocks=blocks,spectral_secant=spectral,
        limits='one displacement cannot identify a Hessian or justify a production correction; no heldout data')
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
