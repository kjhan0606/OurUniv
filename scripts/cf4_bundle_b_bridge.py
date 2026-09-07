"""Two bounded regularized MAP development bridges, never an IC posterior."""
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cf4_z6_native_physics import read_native
from cf4_z0_pm_bridge import make_forward
from cf4_datum_bearing_z0_phasec_pilot import build_pmwd_truth


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def metrics(rho, vel, target):
    tr,tv = target
    return dict(density_RMS=float(np.sqrt(np.mean((rho-tr)**2))),
        velocity_RMS_km_s=float(np.sqrt(np.mean((vel-tv)**2))),
        density_correlation=float(np.corrcoef(rho.ravel(),tr.ravel())[0,1]),
        velocity_correlation=float(np.corrcoef(vel.ravel(),tv.ravel())[0,1]),
        density_min=float(rho.min()), density_mean=float(rho.mean()))


def spectrum(field, box=384.):
    n = field.shape[0]
    k = 2*np.pi*np.fft.fftfreq(n,d=box/n)
    kk = np.sqrt(sum(x*x for x in np.meshgrid(k,k,k,indexing="ij")))
    power = abs(np.fft.fftn(field))**2 * box**3/n**6
    edges = np.linspace(2*np.pi/box,np.pi*n/box,17)
    bins = np.digitize(kk.ravel(),edges)-1
    result=[]
    for i in range(len(edges)-1):
        use=bins==i
        result.append([float(kk.ravel()[use].mean()),float(power.ravel()[use].mean()),int(use.sum())])
    return result


def main():
    if jax.default_backend() != "gpu":
        raise RuntimeError("allocated Slurm GPU required")
    config=json.loads((ROOT/"config/cf4_bundle_b_v1.json").read_text())
    plan=json.loads((ROOT/config["native_plan"]).read_text())
    program=json.loads((ROOT/config["PM_program"]).read_text())
    settings=config["bridge"]
    out=Path(config["output_root"])/"bridge"
    out.mkdir(parents=True,exist_ok=False)
    fields=[read_native(plan,i) for i in settings["training"]]
    # Fixed before reading either target: residual scales are engineering weights,
    # NOT observed uncertainties, likelihood calibration, or acceptance tuning.
    sr=settings["normalization_fraction"]*np.std(np.stack([x[0]-1 for x in fields]))
    sv=settings["normalization_fraction"]*np.std(np.stack([x[1] for x in fields]),axis=(0,2,3,4))
    write(out/"normalization.json",dict(training_indices=settings["training"],
        density_sigma=float(sr),velocity_sigma_km_s=sv.tolist(),fraction=settings["normalization_fraction"],
        objective="0.5 sum density residual^2/sr^2 + 0.5 sum all component velocity residual^2/sv^2 + 0.5 sum white^2",
        status="FROZEN_BEFORE_TARGET_FITS_ENGINEERING_DISCREPANCY_NOT_OBSERVATIONAL_ERROR"))
    del fields
    forward,linear=make_forward(program)
    srj,svj=jnp.asarray(sr),jnp.asarray(sv)[:,None,None,None]

    @jax.jit
    def objective(x,tr,tv):
        rho,vel,_=forward(x)
        return .5*(jnp.sum(((rho-tr)/srj)**2)+jnp.sum(((vel-tv)/svj)**2)+jnp.sum(x*x))
    value_grad=jax.jit(jax.value_and_grad(objective))
    reports=[]
    for index,seed in zip(settings["targets"],settings["seeds"]):
        start_time=time.monotonic()
        case=out/f"target_{index}"
        case.mkdir(exist_ok=False)
        target=read_native(plan,index)
        tr,tv=map(jnp.asarray,target)
        x0=np.random.default_rng(seed).standard_normal(64**3)
        initial=tuple(np.asarray(a) for a in forward(jnp.asarray(x0)))
        if index == settings["targets"][0]:
            # Regression uses a NEW random white field, not saved generating phases.
            old=build_pmwd_truth(x0.reshape((64,)*3),program)
            np.testing.assert_allclose(initial[0]-1,old["coarse_delta"],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(initial[1],np.moveaxis(old["coarse_velocity"],-1,0),rtol=1e-10,atol=1e-8)
            del old
        log=[]
        best={"value":float("inf"),"x":None}
        class CapReached(Exception):
            pass
        def fun(x):
            if len(log)>=settings["max_evaluations"]:
                raise CapReached()
            value,grad=value_grad(jnp.asarray(x),tr,tv)
            value,grad=float(value),np.asarray(grad)
            if not np.isfinite(value) or not np.isfinite(grad).all():
                raise FloatingPointError("nonfinite PM objective/gradient")
            row=dict(evaluation=len(log)+1,objective=value,prior_penalty=float(.5*np.dot(x,x)),
                     gradient_norm=float(np.linalg.norm(grad)),elapsed_seconds=time.monotonic()-start_time)
            log.append(row)
            with (case/"evaluations.jsonl").open("a") as f:
                f.write(json.dumps(row,allow_nan=False)+"\n")
            if value<best["value"]:
                best.update(value=value,x=x.copy())
            if len(log)==1 or len(log)%10==0:
                print(json.dumps(dict(target=index,**row)),flush=True)
            return value,grad
        f0,g0=fun(x0)
        direction=np.random.default_rng(seed+100).normal(size=x0.size)
        direction/=np.linalg.norm(direction)
        eps=1e-3
        # These two objective/gradient probes count toward the 200-call cap.
        plus,_=fun(x0+eps*direction); minus,_=fun(x0-eps*direction)
        finite_difference=(plus-minus)/(2*eps)
        adjoint=float(np.dot(g0,direction))
        relative_error=abs(adjoint-finite_difference)/max(abs(adjoint),abs(finite_difference),1e-6)
        write(case/"gradient_check.json",dict(adjoint=adjoint,finite_difference=finite_difference,
            relative_error=relative_error,epsilon=eps,regression="existing PMWD wrapper matched"))
        if relative_error>.02:
            raise RuntimeError("PM adjoint directional check exceeds frozen 2% tolerance")
        try:
            fit=minimize(fun,x0,jac=True,method="L-BFGS-B",options=dict(maxiter=200,maxfun=200,ftol=1e-9,gtol=1e-5,maxls=20))
            termination=str(fit.message)
        except CapReached:
            termination="200 evaluation cap reached; use minimum finite objective evaluated on this fixed trajectory"
        xf=best["x"]
        # Recompute, rather than reuse optimizer auxiliary state.
        final=tuple(np.asarray(a) for a in forward(jnp.asarray(xf)))
        before,after=metrics(*initial[:2],target),metrics(*final[:2],target)
        conservation=bool(np.max(abs(final[2]))<1e-6 and np.max(abs(initial[2]))<1e-6)
        passed=bool(after["density_min"]>0 and abs(after["density_mean"]-1)<2e-12 and conservation
            and after["density_RMS"]<=.5*before["density_RMS"]
            and after["velocity_RMS_km_s"]<=.5*before["velocity_RMS_km_s"])
        np.savez_compressed(case/"candidate.npz",white_MAP=xf.reshape((64,)*3),
            initial_density=initial[0],initial_velocity=initial[1],final_density=final[0],final_velocity=final[1],
            target_density=target[0],target_velocity=target[1])
        p0=spectrum(np.asarray(linear(jnp.asarray(x0))))
        pf=spectrum(np.asarray(linear(jnp.asarray(xf))))
        report=dict(target=index,seed=seed,status="PASS_DEVELOPMENT_BRIDGE" if passed else "FAIL_DEVELOPMENT_BRIDGE",
            before=before,after=after,conservation_pass=conservation,conservation_error=final[2].tolist(),
            evaluations=len(log),termination=termination,initial_prior_penalty=float(.5*np.dot(x0,x0)),
            final_prior_penalty=float(.5*np.dot(xf,xf)),initial_linear_power=p0,final_linear_power=pf,
            elapsed_seconds=time.monotonic()-start_time,
            limits="Regularized MAP candidates, not an IC posterior. Reused development targets/shared PM generator. No P(k) rescaling. No actual data/LG conditioning.")
        write(case/"result.json",report); reports.append(report)
        print(json.dumps({k:report[k] for k in ("target","status","before","after","elapsed_seconds")}),flush=True)
    write(out/"result.json",dict(status="COMPLETE_DEVELOPMENT_ONLY",cases=reports,
        all_pass=all(r["status"]=="PASS_DEVELOPMENT_BRIDGE" for r in reports)))


if __name__=="__main__":
    main()
