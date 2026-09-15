"""Read-only radius sensitivity of existing independent endpoints; no evolution."""
import json
import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from cf4_r1_ramses_reference import load_snapshot
from cf4_r1_particle_forward import aperture_moments
from cf4_r1_particle_entry import ROOT, ready, write


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    cfg = json.loads((ROOT/'config/cf4_r1_ramses_reference_v7.json').read_text())
    model = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    reference = Path('/gpfs/kjhan/CF4/r1_ref_v7/job_360338')
    previous = json.loads((reference/'result.json').read_text())
    if not previous['status'].startswith('COMPLETE_INDEPENDENT_COMPARISON'):
        raise RuntimeError('independent endpoints not complete')
    out = Path('/gpfs/kjhan/CF4/r1_observable_scale')/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    radii = [.3, .75, 1.5]
    results = {}
    for label in ['cic', 'tsc', 'amr8', 'amr9']:
        if label in ['amr8', 'amr9']:
            x, v, info = load_snapshot(reference/label/'output_00002', 12., 128)
            np.testing.assert_allclose(info['aexp'], 1., atol=2e-6, rtol=0)
        else:
            path = (Path(cfg['reference_directory'])/f"reference_particles_seed{cfg['seed']}.npz"
                    if label == 'cic' else Path(cfg['tsc_directory'])/f"tsc_final_seed{cfg['seed']}.npz")
            with np.load(path) as data:
                x, v = data['position_cMpc_h'], data['velocity_km_s']
        with np.load(Path(cfg['reference_directory'])/f"reference_particles_seed{cfg['seed']}.npz") as data:
            mp = float(data['particle_mass_Msun_h'])
        results[label] = {}
        for radius in radii:
            moments = ready(aperture_moments(jnp.asarray(x), jnp.asarray(v),
                jnp.full(len(x), mp), jnp.asarray(model['aperture_centers_cMpc_h']), radius, 12.))
            row = {key: np.asarray(value).tolist() for key, value in moments.items()}
            row['sigma_km_s'] = np.sqrt(np.asarray(moments['variance_km2_s2'])).tolist()
            results[label][str(radius)] = row
            if radius == .75 and label.startswith('amr'):
                np.testing.assert_allclose(moments['variance_km2_s2'],
                    previous['comparisons'][label]['physical_variance_km2_s2'], rtol=1e-10)
        print(label+' measured', flush=True)
        del x, v
    contrasts = {}
    for left, right in [('amr9', 'amr8'), ('amr9', 'tsc'), ('amr9', 'cic')]:
        contrasts[left+'-'+right] = {}
        for radius in radii:
            a, b = results[left][str(radius)], results[right][str(radius)]
            row = {}
            for key in ['mass_Msun_h', 'sigma_km_s', 'variance_km2_s2']:
                row[key+'_max_relative_difference'] = float(np.max(np.abs(np.asarray(a[key])/np.asarray(b[key])-1)))
            for key in ['offset_cMpc_h', 'mean_velocity_km_s']:
                row[key+'_max_vector_difference'] = float(np.max(np.linalg.norm(np.asarray(a[key])-np.asarray(b[key]), axis=-1)))
            contrasts[left+'-'+right][str(radius)] = row
    write(out/'result.json', dict(status='COMPLETE', source_commit=os.environ['EXPECTED_COMMIT'],
        radii_cMpc_h=radii, source_directory=str(reference), moments=results, contrasts=contrasts,
        limitations='Fixed probes, one seed, no halo identities. Radius is Gaussian width, not grid resolution. No inferred discrepancy covariance or LG posterior.',
        observed_posterior=False, resolved_LG=False, R1_complete=False))
    print(json.dumps(contrasts), flush=True)


if __name__ == '__main__':
    main()
