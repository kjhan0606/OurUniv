#!/usr/bin/env python3
"""Localize saved R2 low-k likelihood force by shell and centered multipole."""
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np


N = 256
BOX = 384.0
NIC = N**3
MAX_K_INDEX = 8
BASE = Path('/gpfs/kjhan/CF4/z0_density')
CHECKPOINTS = {
    'a': BASE/'r2_n256_gl2_metric6000_control_a_v1/chain_a_accepted_checkpoint.npz',
    'b': BASE/'r2_n256_gl2_metric6000_control_b_v1/chain_b_accepted_checkpoint.npz',
}
ATTRIBUTION = BASE/'r2_n256_lowk_component_attribution_20261002_v2/result.json'
MODES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def fourier_geometry(n, max_k_index):
    """Return exact |k|^2 shell IDs, directions and centered-observer phase."""
    if n < 8 or n % 2 or max_k_index < 1:
        raise ValueError('even grid n>=8 and positive low-k shell limit required')
    axis = np.rint(np.fft.fftfreq(n)*n).astype(np.int32)
    kx = np.broadcast_to(axis[:, None, None], (n, n, n))
    ky = np.broadcast_to(axis[None, :, None], (n, n, n))
    kz = np.broadcast_to(axis[None, None, :], (n, n, n))
    k2 = kx*kx + ky*ky + kz*kz
    kmag = np.sqrt(k2, dtype=np.float64)
    shell_squared = k2.astype(np.int16)
    parity = (kx & 1) ^ (ky & 1) ^ (kz & 1)
    # The observer is at grid index n/2 in each dimension.
    center_phase = 1.0 - 2.0*parity
    directions = np.zeros((n, n, n, 3), dtype=np.float32)
    nonzero = kmag > 0
    directions[..., 0] = np.divide(kx, kmag, out=np.zeros_like(kmag), where=nonzero)
    directions[..., 1] = np.divide(ky, kmag, out=np.zeros_like(kmag), where=nonzero)
    directions[..., 2] = np.divide(kz, kmag, out=np.zeros_like(kmag), where=nonzero)
    return kx, ky, kz, kmag, shell_squared, center_phase, directions


def multipole_projection(force_coefficients, directions, center_phase):
    """Fit centered l=0, l=1, l=2 components; return their Fourier L2 shares."""
    values = np.asarray(force_coefficients, dtype=np.complex128)
    direction = np.asarray(directions, dtype=np.float64)
    phase = np.asarray(center_phase, dtype=np.float64)
    if values.ndim != 1 or direction.shape != (len(values), 3) or phase.shape != values.shape:
        raise ValueError('multipole inputs have inconsistent shapes')
    if len(values) < 1 or not np.isfinite(values).all():
        raise ValueError('at least one finite Fourier mode is required')

    # A radial field centered at x=(L/2,L/2,L/2) has phase
    # exp(-i*pi*(nx+ny+nz)) = (-1)**(nx+ny+nz).
    centered = values*phase
    x, y, z = direction.T
    quadrupole_basis = np.column_stack((x*x-y*y, x*x-z*z,
                                         2*x*y, 2*y*z, 2*z*x))
    even_design = np.column_stack((np.ones(len(values)), quadrupole_basis))
    even_coef, _, even_rank, _ = np.linalg.lstsq(even_design, centered.real, rcond=None)
    even_fit = even_design@even_coef
    dipole_coef, _, dipole_rank, _ = np.linalg.lstsq(direction, centered.imag, rcond=None)
    dipole_fit = direction@dipole_coef

    monopole = np.full(len(values), even_coef[0], dtype=np.float64)
    quadrupole = quadrupole_basis@even_coef[1:]
    dipole = 1j*dipole_fit
    residual = centered-(monopole+quadrupole+dipole)
    energies = {
        'monopole_l0': float(np.vdot(monopole, monopole).real),
        'dipole_l1': float(np.vdot(dipole, dipole).real),
        'quadrupole_l2': float(np.vdot(quadrupole, quadrupole).real),
        'higher_order_or_unmodelled': float(np.vdot(residual, residual).real),
    }
    total = float(np.vdot(centered, centered).real)
    fractions = {name+'_fraction': value/total if total else 0.0
                 for name, value in energies.items()}
    return dict(centered_monopole_coefficient=float(even_coef[0]),
        dipole_coefficients= dipole_coef.tolist(),
        quadrupole_coefficients=even_coef[1:].tolist(),
        even_design_rank=int(even_rank), dipole_design_rank=int(dipole_rank),
        component_energy=energies, component_fraction=fractions,
        total_fourier_energy=total)


def analyze_spectra(q_hat, target_gradient_hat, *, max_k_index=MAX_K_INDEX):
    """Summarize exact radial derivatives and centered angular structure."""
    q_hat = np.asarray(q_hat)
    target_gradient_hat = np.asarray(target_gradient_hat)
    if q_hat.ndim != 3 or q_hat.shape != target_gradient_hat.shape:
        raise ValueError('q and target-gradient Fourier arrays must have matching 3D shapes')
    n = q_hat.shape[0]
    if q_hat.shape != (n, n, n):
        raise ValueError('cubic Fourier grid required')
    nll_hat = target_gradient_hat-q_hat
    force_hat = -nll_hat
    kx, ky, kz, kmag, shell_squared, center_phase, directions = \
        fourier_geometry(n, max_k_index)

    all_mode_prior = float(np.vdot(q_hat, q_hat).real)
    all_mode_nll = float(np.vdot(q_hat, nll_hat).real)
    shell_rows = []
    shell_component_energy = {name: 0.0 for name in
        ('monopole_l0', 'dipole_l1', 'quadrupole_l2', 'higher_order_or_unmodelled')}
    lowk_prior = 0.0
    lowk_nll = 0.0
    for shell_index in range(1, max_k_index*max_k_index+1):
        mask = shell_squared == shell_index
        count = int(mask.sum())
        if count == 0:
            continue
        prior_radial = float(np.sum(np.abs(q_hat[mask])**2))
        nll_radial = float(np.sum(np.real(np.conj(q_hat[mask])*nll_hat[mask])))
        phase = center_phase[mask]
        dirs = directions[mask]
        multipoles = multipole_projection(force_hat[mask], dirs, phase)
        shell_rows.append(dict(k_squared_index=shell_index,
            k_norm_index=float(np.sqrt(shell_index)),
            k_h_Mpc=float(np.sqrt(shell_index)*2*np.pi/BOX),
            mode_count=count, conjugate_pairs=count//2,
            prior_radial_derivative=prior_radial,
            likelihood_nll_radial_derivative=nll_radial,
            total_nlogtarget_radial_derivative=prior_radial+nll_radial,
            force_multipoles=multipoles))
        lowk_prior += prior_radial
        lowk_nll += nll_radial
        for name, value in multipoles['component_energy'].items():
            shell_component_energy[name] += value

    lowk_force_energy = sum(shell_component_energy.values())
    lowk_multipole_fractions = {name+'_fraction': value/lowk_force_energy
        if lowk_force_energy else 0.0 for name, value in shell_component_energy.items()}
    fundamental_rows = []
    for mode in MODES:
        qk = q_hat[mode]
        nllk = nll_hat[mode]
        fundamental_rows.append(dict(mode=list(mode),
            radial_nll_derivative_pair=2*float(np.real(np.conj(qk)*nllk)),
            force_coefficient=[float((-nllk).real), float((-nllk).imag)]))
    return dict(all_mode_radial=dict(prior=float(all_mode_prior),
        likelihood_nll=float(all_mode_nll), total_nlogtarget=float(all_mode_prior+all_mode_nll)),
        lowk_shells=dict(max_k_index=max_k_index,
            index_definition='exact n_x^2+n_y^2+n_z^2 shells with 0<|n|<=max_k_index; DC excluded',
            prior_radial=float(lowk_prior), likelihood_nll_radial=float(lowk_nll),
            total_nlogtarget_radial=float(lowk_prior+lowk_nll),
            outside_lowk_likelihood_nll_radial=float(all_mode_nll-lowk_nll),
            multipole_force_energy=shell_component_energy,
            multipole_force_fraction=lowk_multipole_fractions,
            shells=shell_rows),
        fundamental_mode_checks=fundamental_rows)


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('run as a Slurm GPU job')
    output = Path(os.environ['CF4_R2_OUT_DIR'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attribution = json.loads(ATTRIBUTION.read_text())
    if attribution.get('status') != 'COMPONENT_ATTRIBUTION_COMPLETE_NOT_POSTERIOR':
        raise ValueError('completed two-endpoint count-vs-FP attribution is required')

    report = dict(status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        classification='SAVED_N256_LOWK_GEOMETRY_ATTRIBUTION_NOT_POSTERIOR',
        grid=N, box_cMpc_h=BOX, cell_cMpc_h=BOX/N,
        observer_cMpc_h=[BOX/2]*3, observer_grid_index=[N//2]*3,
        lowk_max_index=MAX_K_INDEX, checkpoints={}, heldout_scored=False,
        chain_transitions=0, PMWD_replays=0,
        analyzed_field='combined count-dominated likelihood NLL gradient = saved target gradient - q; raw FP contribution is small but nonzero',
        multipole='exact |n|^2 shells; Fourier coefficients phase-corrected by (-1)**(nx+ny+nz); shellwise real monopole, imaginary dipole, real trace-free quadrupole, residual; fit ranks are recorded because low shells may not resolve every l=2 component',
        Q_GOAL='localize the low-k R2 observation-model force before count-law edits, toward a credible same-field z=0 environment for LG constraints',
        Q_LEAN='two saved exact-target checkpoints; FFTs and small shell/multipole reductions only; no replay, chain, heldout or simulation',
        MW_M31='role-ambiguous; no candidate selected by truth identity',
        M33='unresolved; eventual observables must constrain the same NEW field',
        interpretation_limit='geometry of latent-IC likelihood force only; no z=0 density peak, posterior, significance or LG identification claim')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        temporary = output/'result.json.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(output/'result.json')

    save()
    try:
        @jax.jit
        def fft_pair(q, gradient):
            return (jnp.fft.fftn(q.reshape((N,)*3),norm='ortho'),
                    jnp.fft.fftn(gradient.reshape((N,)*3),norm='ortho'))

        for label, path in CHECKPOINTS.items():
            with np.load(path,allow_pickle=False) as archive:
                q = np.asarray(archive['canonical'][:NIC],dtype=np.float64)
                total_gradient = np.asarray(archive['fine_gradient'][:NIC],dtype=np.float64)
                transitions = int(archive['completed_transitions'])
            if (q.shape!=(NIC,) or total_gradient.shape!=(NIC,) or transitions!=1
                    or not np.isfinite(q).all() or not np.isfinite(total_gradient).all()):
                raise ValueError(f'{label}: invalid saved exact endpoint')
            q_hat_dev, gradient_hat_dev = fft_pair(jnp.asarray(q),jnp.asarray(total_gradient))
            jax.block_until_ready((q_hat_dev,gradient_hat_dev))
            q_hat = np.asarray(jax.device_get(q_hat_dev))
            gradient_hat = np.asarray(jax.device_get(gradient_hat_dev))
            result = analyze_spectra(q_hat,gradient_hat,max_k_index=MAX_K_INDEX)
            reference_modes = attribution['checkpoints'][label]['total_likelihood_gradient_modes']
            reference = {tuple(row['mode']):row['radial_nloglikelihood_derivative']
                         for row in reference_modes}
            for row in result['fundamental_mode_checks']:
                key = tuple(row['mode'])
                row['reference_job410059'] = float(reference[key])
                row['absolute_reference_error'] = abs(row['radial_nll_derivative_pair']-reference[key])
                if row['absolute_reference_error']>1e-7:
                    raise AssertionError(f'{label}/{key}: fundamental derivative reference mismatch')
            result['fundamental_references_pass'] = True
            result['checkpoint'] = str(path)
            result['completed_transitions'] = transitions
            report['checkpoints'][label] = result
            save()
            del q, total_gradient, q_hat_dev, gradient_hat_dev, q_hat, gradient_hat

        report['status'] = 'GEOMETRY_DIAGNOSTIC_COMPLETE_NOT_POSTERIOR'
        report['outcome_mapping'] = 'compare radial-shell derivatives and l=0/1/2 energy fractions; audit radial selection/LF/K correction for monopole dominance, angular completeness/exposure for anisotropy, or low-intensity counts/bias for incoherent remainder; no automatic likelihood edit'
        save()
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    except Exception as error:
        report['status'] = 'GEOMETRY_DIAGNOSTIC_FAILED_OR_INCOMPLETE_NOT_POSTERIOR'
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


if __name__ == '__main__':
    main()
