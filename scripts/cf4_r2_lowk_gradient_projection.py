#!/usr/bin/env python3
"""Project saved exact-target gradients onto the three N256 fundamental modes."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


N = 256
NIC = N**3
MODES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
BASE = Path('/gpfs/kjhan/CF4/z0_density')
CHECKPOINTS = {
    'a': BASE/'r2_n256_gl2_metric6000_control_a_v1/chain_a_accepted_checkpoint.npz',
    'b': BASE/'r2_n256_gl2_metric6000_control_b_v1/chain_b_accepted_checkpoint.npz',
}


def complex_pair(value):
    return [float(value.real), float(value.imag)]


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('run as a Slurm GPU job')
    output = Path(os.environ['CF4_R2_OUT_DIR'])
    ix, iy, iz = (jnp.asarray([m[axis] for m in MODES]) for axis in range(3))

    @jax.jit
    def project_modes(field):
        spectrum = jnp.fft.fftn(field.reshape((N, N, N)), norm='ortho')
        return spectrum[ix, iy, iz]

    report = {
        'classification': 'SAVED_R2_EXACT_GL2_LOWK_GRADIENT_PROJECTION_NOT_POSTERIOR',
        'job_id': os.environ['SLURM_JOB_ID'],
        'grid': N,
        'box_cMpc_h': 384.0,
        'cell_cMpc_h': 1.5,
        'modes': [list(m) for m in MODES],
        'fft_normalization': 'orthonormal; non-self-conjugate Re/Im prior SD=1/sqrt(2)',
        'target_gradient_definition': 'fine_gradient = grad[0.5*q.q - log_likelihood(q)]',
        'decomposition': 'prior_gradient=q; likelihood_gradient=fine_gradient-q; force=-gradient',
        'interpretation_limit': 'local directional diagnostic only; no posterior, stationarity, density-peak, or LG identification claim',
        'checkpoints': {},
    }
    for label, path in CHECKPOINTS.items():
        with np.load(path, allow_pickle=False) as archive:
            q = np.asarray(archive['canonical'][:NIC], dtype=np.float64)
            total_gradient = np.asarray(archive['fine_gradient'][:NIC], dtype=np.float64)
            completed = int(archive['completed_transitions'])
        if q.shape != (NIC,) or total_gradient.shape != (NIC,):
            raise ValueError(f'{label}: checkpoint field shape mismatch')
        if not np.isfinite(q).all() or not np.isfinite(total_gradient).all():
            raise FloatingPointError(f'{label}: non-finite checkpoint field/gradient')

        q_modes = np.asarray(jax.device_get(project_modes(jnp.asarray(q))))
        total_modes = np.asarray(jax.device_get(project_modes(jnp.asarray(total_gradient))))
        likelihood_modes = total_modes - q_modes
        rows = []
        for mode, qk, gk, lk in zip(MODES, q_modes, total_modes, likelihood_modes):
            q2 = float(abs(qk)**2)
            dot_qg = float(np.real(np.conj(qk)*gk))
            rows.append({
                'mode': list(mode),
                'q_prior_coordinate': complex_pair(qk),
                'q_prior_standardized_re_im': [float(np.sqrt(2)*qk.real), float(np.sqrt(2)*qk.imag)],
                'prior_potential_pair_nats': q2,
                'total_nlogtarget_gradient': complex_pair(gk),
                'likelihood_gradient': complex_pair(lk),
                'total_gradient_to_prior_mode_norm': float(abs(gk)/max(abs(qk), 1e-300)),
                'cosine_q_with_total_gradient': float(dot_qg/(abs(qk)*abs(gk))) if abs(qk)*abs(gk) else None,
                'radial_nlogtarget_derivative_for_q_scaling': 2*dot_qg,
                'radial_prior_derivative': 2*q2,
                'radial_likelihood_derivative': 2*float(np.real(np.conj(qk)*lk)),
            })
        report['checkpoints'][label] = {
            'path': str(path),
            'completed_transitions': completed,
            'modes': rows,
            'three_mode_prior_potential_pair_sum_nats': float(sum(r['prior_potential_pair_nats'] for r in rows)),
            'three_mode_radial_total_derivative': float(sum(r['radial_nlogtarget_derivative_for_q_scaling'] for r in rows)),
            'three_mode_radial_prior_derivative': float(sum(r['radial_prior_derivative'] for r in rows)),
            'three_mode_radial_likelihood_derivative': float(sum(r['radial_likelihood_derivative'] for r in rows)),
        }

    output.mkdir(parents=True, exist_ok=False)
    temporary = output/'result.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(output/'result.json')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
