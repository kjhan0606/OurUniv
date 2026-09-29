"""One actual N256 LCDM initializer and PM adjoint resource measurement.

Not a posterior sample: low modes inherit an unconverged N128 initializer;
new high modes are a prior draw, never claimed to be observed information.
"""
import json
import os
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy import fft
from cf4_lg_highk_conditional_field import prolong_white_spectrum,restrict_spectrum_preserve_dtype
from cf4_r1_particle_forward import make_dynamics,particle_grid

BASE=Path('/gpfs/kjhan/CF4/z0_density')
ROOT=Path(__file__).resolve().parents[1]


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],
        source_commit=os.environ['CF4_EXPECTED_COMMIT'],N=256,box_cMpc_h=384.,dx_cMpc_h=1.5,
        R2_complete=False,posterior_draw=False,heldout_scored=False,seed=2026092915,
        purpose='actual N256 forward/adjoint resource pilot and reusable initializer, not data-target validation',
        limits='unconverged inherited low modes; added modes are prior, MW/M31 ambiguous and M33 unresolved')
    def save():
        report.update(seconds=time.monotonic()-started,
            host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        parent=BASE/'r2_raw_joint_pilot_v1'
        if json.loads((parent/'result.json').read_text())['status']!='RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR':
            raise ValueError('completed raw joint initializer required')
        with np.load(parent/'accepted_present_state.npz',allow_pickle=False) as f:
            coarse=f['white_ic'].reshape(128,128,128)
            tracer=f['tracer'];population=f['population_white']
        report['phase']='PRIOR_CONDITIONAL_HIGH_MODES';save()
        spectrum,coarse_spectrum=prolong_white_spectrum(coarse,256,report['seed'],
            float_dtype=np.dtype(np.float64),workers=2)
        error=float(np.max(np.abs(restrict_spectrum_preserve_dtype(spectrum,128)-coarse_spectrum)))
        white_complex=fft.ifftn(spectrum,norm='ortho',workers=2)
        imag=float(np.max(np.abs(white_complex.imag)))
        white=np.ascontiguousarray(white_complex.real).ravel()
        del spectrum,coarse_spectrum,white_complex,coarse
        report.update(white_restriction_max_abs=error,white_imaginary_max_abs=imag,
            white_mean_square=float(np.mean(white**2)))
        if max(error,imag)>1e-10:raise AssertionError('white restriction/Hermitian mismatch')
        settings=json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
        settings={k:settings[k] for k in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
        settings.update(n=256,box_cMpc_h=384.)
        common=json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
        if settings['cosmology']['h']!=.746 or settings['cosmology']['Om']!=common['Omega_m']:
            raise ValueError('common cosmology mismatch')
        evolve,_,conf,_,pmass=make_dynamics(settings)
        mass=jnp.full(256**3,pmass)
        wave=jnp.cos(2*jnp.pi*jnp.arange(256)/256)[:,None,None]
        def resource_probe(q):
            positions,velocities=evolve(q)
            state=particle_grid(positions,velocities,mass,conf)
            # Deterministic smooth readout exercises density AND momentum
            # through the PM adjoint; this is not a likelihood or constraint.
            value=jnp.mean(state['rho']*wave)+jnp.mean(state['momentum'][...,0]*wave)/(mass.sum()*1000/256**3)
            return value,(state['rho'],state['mean_velocity_km_s'],state['variance_km2_s2'],state['valid'])
        report['phase']='COMPILE_N256_FORWARD_ADJOINT';save()
        q=jnp.asarray(white)
        derivative=jax.jit(jax.value_and_grad(resource_probe,has_aux=True))
        compiled=derivative.lower(q).compile()
        memory=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
        peak=stats.get('bytes_in_use',0)+memory.temp_size_in_bytes+memory.output_size_in_bytes
        report.update(phase='N256_FORWARD_ADJOINT',device_temporary_GiB=memory.temp_size_in_bytes/1024**3,
            estimated_peak_GiB=peak/1024**3,particle_mass_Msun_h=pmass)
        save()
        if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
            raise MemoryError('N256 adjoint lacks20percent device memory margin')
        tic=time.monotonic();(value,fields),gradient=compiled(q)
        value=float(value)
        finite=bool(jnp.isfinite(gradient).all())
        r,v,var,valid=tuple(map(np.asarray,fields))
        if not finite or not all(np.isfinite(x).all() for x in (r,v,var)):
            raise FloatingPointError('nonfinite N256 dynamics')
        mean=float(np.mean(r))
        if abs(mean-1)>1e-10 or np.min(r)<0:raise AssertionError('N256 deposited mass not conserved')
        report.update(forward_adjoint_seconds=time.monotonic()-tic,probe_value=value,
            density_mean=mean,valid_fraction=float(np.mean(valid)),gradient_rms=float(jnp.sqrt(jnp.mean(gradient**2))),
            phase='SAVING_INITIALIZER')
        save()
        # One reusable field+IC only; no particles, trajectories or gradient archive.
        np.savez(out/'initial_present_state.npz',white_ic=white,rho=r,
            mean_velocity_km_s=np.moveaxis(v,-1,0),physical_velocity_variance_km2_s2=np.moveaxis(var,-1,0),
            velocity_valid=valid,tracer=tracer,population_white=population,
            box_cMpc_h=384.,native_mesh_origin_fraction=0.,R2_complete=False,posterior_draw=False)
        report.update(status='N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR',settings=settings);save()
    except Exception as error:
        report.update(status='FAILED_N256_DYNAMICS_PROFILE',error=repr(error));save();raise


if __name__=='__main__':main()
