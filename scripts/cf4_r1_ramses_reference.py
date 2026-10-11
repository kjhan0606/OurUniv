"""Exact supplied-state handoff to a pinned independent RAMSES DMO build."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from grafic_io import _header_bytes, _write_record
from cf4_zoom_z0_gate import _record, _skip_header, read_info, particle_files
from cf4_r1_particle_entry import ROOT, ready, write
from cf4_r1_particle_forward import make_configuration, aperture_moments, particle_grid, periodic_delta
from cf4_r1_three_dimensional import bands, vector


def write_ascii_ic(directory, displacement, velocity, model):
    """Total-matter masses; no ZA reconstruction, GRAFIC mass split or half-cell shift."""
    directory.mkdir(exist_ok=False)
    n, box, a = model['n'], model['box_cMpc_h'], model['a_start']
    c = model['cosmology']
    # ASCII init_cosmo reads this header only; this is NOT a gas-density field.
    header = _header_bytes(n, n, n, box/(n*c['h']), (0, 0, 0), a, c['Om'], 1-c['Om'], 100*c['h'])
    with (directory/'ic_deltab').open('xb') as handle:
        _write_record(handle, header+np.asarray(c['Ob'], dtype='<f4').tobytes())
    q = np.indices((n,)*3).reshape(3, -1).T*box/n
    x = np.mod(q+displacement.reshape((-1, 3)), box)/box-.5
    v = velocity.reshape((-1, 3))*a/(100*box)
    data = np.column_stack((x, v, np.full(n**3, 1/n**3)))
    with (directory/'ic_part').open('x') as handle:
        np.savetxt(handle, data, fmt='%.17e')


def namelist(ic, levelmax, model, *, initial_only):
    if len(str(ic)) > 65:
        raise ValueError('RAMSES80-character path buffer requires a short IC path')
    a, c = model['a_start'], model['cosmology']
    outputs = f'noutput=1\naout={a}' if initial_only else f'noutput=2\naout={a},1.0'
    flags = ['use_neutrino', 'sidm', 'de_perturb', 'use_mond', 'use_fR', 'use_nDGP',
             'use_symmetron', 'use_dilaton', 'use_galileon', 'use_coupled_de',
             'use_quintessence', 'use_kessence', 'use_chaplygin', 'use_rvm',
             'use_horndeski', 'use_ede', 'use_sgs', 'use_adm', 'use_fdm', 'use_pbh']
    inactive = '\n'.join(f'{flag}=.false.' for flag in flags)
    return f'''&RUN_PARAMS
cosmo=.true.
pic=.true.
poisson=.true.
hydro=.false.
rt=.false.
sink=.false.
clumpfind=.false.
use_fftw=.true.
dump_pk=.false.
{inactive}
nrestart=0
nstepmax={0 if initial_only else 20000}
ncontrol=25
nremap=10
nsubcycle=1,2,2
ordering='hilbert'
aexp_step_limit=0.01
/
&OUTPUT_PARAMS
{outputs}
match_aout=.true.
foutput=1000000000
fbackup=1000000000
/
&INIT_PARAMS
filetype='ascii'
initfile(1)='{ic}'
aexp_ini={a}
/
&AMR_PARAMS
levelmin=7
levelmax={levelmax}
ngridtot=1500000
nparttot=4000000
nexpand=1,1,1
boxlen=1.0
/
&COSMO_PARAMS
omega_m={c['Om']}
omega_l={1-c['Om']}
omega_b={c['Ob']}
h0={100*c['h']}
/
&REFINE_PARAMS
m_refine=8.,8.,8.
ivar_refine=0
/
&POISSON_PARAMS
epsilon=1.d-5
maxiter_fine=100
abort_on_mg_nonconvergence=.true.
/
'''


def load_snapshot(directory, box, particle_n):
    if not (directory/'COMPLETE').is_file():
        raise RuntimeError(f'incomplete snapshot {directory}')
    info = read_info(directory)
    rows = []
    for path in particle_files(directory):
        with path.open('rb') as handle:
            _, count = _skip_header(handle)
            x = np.stack([_record(handle, '<f8') for _ in range(3)], axis=-1)
            v = np.stack([_record(handle, '<f8') for _ in range(3)], axis=-1)
            mass, ids = _record(handle, '<f8'), _record(handle, '<i8')
        if len(ids) != count:
            raise RuntimeError('particle record count mismatch')
        rows.append((x, v, mass, ids))
    x, v, mass, ids = [np.concatenate([row[i] for row in rows]) for i in range(4)]
    order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order], np.arange(1, particle_n**3+1))
    x, v, mass = x[order]*box, v[order]*(info['unit_l']/info['unit_t']/1e5), mass[order]
    np.testing.assert_allclose(mass, 1/particle_n**3, rtol=1e-12)
    if not np.isfinite(np.concatenate((x.ravel(), v.ravel()))).all():
        raise FloatingPointError('nonfinite RAMSES state')
    return x, v, info


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'cpu':
        raise RuntimeError('Slurm CPU allocation required')
    cfg = json.loads((ROOT/'config/cf4_r1_ramses_reference_v7.json').read_text())
    out = Path(cfg['output_root'])/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    binary = Path(cfg['binary'])
    source = Path(cfg['reference_directory'])
    model = json.loads((Path(cfg['entry_directory'])/'result.json').read_text())['config']
    model = {**model, 'n': cfg['particle_n']}
    box, n = model['box_cMpc_h'], model['n']
    with np.load(source/f"reference_particles_seed{cfg['seed']}.npz") as stored:
        psi, vi = stored['initial_displacement_cMpc_h'], stored['initial_velocity_km_s']
        cic_x, cic_v = stored['position_cMpc_h'], stored['velocity_km_s']
        mp = float(stored['particle_mass_Msun_h'])
    start = time.monotonic()
    report = dict(status='RUNNING', config=cfg, source_commit=os.environ['EXPECTED_COMMIT'],
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  comparisons={}, observed_posterior=False, resolved_LG=False, R1_complete=False)

    def progress(stage, **more):
        write(out/'result.json', report)
        item = dict(stage=stage, elapsed_seconds=time.monotonic()-start, **more)
        write(out/'progress.json', item)
        print(json.dumps(item, allow_nan=False), flush=True)

    def run(label, levelmax, initial_only=False):
        directory = out/label
        directory.mkdir()
        nml = directory/'run.nml'
        nml.write_text(namelist(out/'ic', levelmax, model, initial_only=initial_only))
        remaining = cfg['application_seconds_cap']-(time.monotonic()-start)
        if remaining < 60:
            raise TimeoutError('bounded reference application budget exhausted')
        cap = min(remaining, 300 if initial_only else cfg['each_evolution_seconds_cap'])
        progress('ramses_start', label=label, namelist=str(nml), timeout_seconds=cap,
                 expected_snapshots=1 if initial_only else 2)
        with (directory/'run.log').open('x') as log:
            subprocess.run(['mpirun', '-np', str(cfg['mpi_ranks']), str(binary), str(nml)],
                           cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=cap)
        text = (directory/'run.log').read_text()
        for required in ('Run completed', 'FFTW3 CPU direct Poisson solver enabled', 'FFT direct solve DONE',
                         'IC/namelist omega_b consistent:'):
            if required not in text:
                raise RuntimeError(f'missing RAMSES marker: {required}')
        if re.search(r'Fine multigrid Poisson failed to converge|FATAL:|forrtl:|Some grid are outside', text):
            raise RuntimeError('RAMSES runtime/physics failure in log')
        outputs = sorted(directory.glob('output_[0-9][0-9][0-9][0-9][0-9]'))
        if len(outputs) != (1 if initial_only else 2):
            raise RuntimeError('unexpected RAMSES output count')
        return outputs

    try:
        write_ascii_ic(out/'ic', psi, vi, model)
        initial_output = run('initial_state', 7, True)[0]
        xi, vin, info = load_snapshot(initial_output, box, n)
        q = np.indices((n,)*3).reshape(3, -1).T*box/n
        initial_dx = np.asarray(periodic_delta(jnp.asarray(xi), jnp.asarray(q+psi.reshape((-1, 3))), box))
        initial_dv = vin-vi.reshape((-1, 3))
        report['initial_state'] = dict(info=info, max_position_error_cMpc_h=float(abs(initial_dx).max()),
                                      max_velocity_error_km_s=float(abs(initial_dv).max()), particle_count=len(xi))
        np.testing.assert_allclose(info['aexp'], model['a_start'], atol=1e-10)
        np.testing.assert_allclose(initial_dx, 0., atol=5e-6)
        np.testing.assert_allclose(initial_dv, 0., atol=1e-3)
        del xi, vin, q, psi, vi, initial_dx, initial_dv
        progress('initial_state_pass')
        readout, _ = make_configuration({**model, 'n': cfg['readout_n']})
        centers, radius = jnp.asarray(model['aperture_centers_cMpc_h']), model['aperture_radius_cMpc_h']
        mass = jnp.full(n**3, mp)
        mean_mass = mp*n**3/box**3*(2*np.pi)**1.5*radius**3
        references = {'cic': (cic_x, cic_v)}
        with np.load(Path(cfg['tsc_directory'])/f"tsc_final_seed{cfg['seed']}.npz") as stored:
            references['tsc'] = (stored['position_cMpc_h'], stored['velocity_km_s'])
        summaries = {}

        def summary(x, v):
            moments = ready(aperture_moments(jnp.asarray(x), jnp.asarray(v), mass, centers, radius, box))
            field = ready(particle_grid(jnp.asarray(x), jnp.asarray(v), mass, readout))
            return np.asarray(vector(moments, mean_mass)), np.asarray(field['rho']), np.asarray(moments['variance_km2_s2'])

        for label, (x, v) in references.items():
            summaries[label] = summary(x, v)
        for levelmax in cfg['reference_levelmax']:
            label = f'amr{levelmax}'
            outputs = run(label, levelmax)
            x, v, info = load_snapshot(outputs[-1], box, n)
            np.testing.assert_allclose(info['aexp'], 1., atol=2e-6, rtol=0.)
            obs, rho, variance = summary(x, v)
            row = dict(info=info, observables=obs.tolist(), physical_variance_km2_s2=variance.tolist(), contrasts={})
            for name, (other_obs, other_rho, other_variance) in summaries.items():
                row['contrasts'][name] = dict(relative_probe_mass_difference=np.expm1((obs-other_obs)[::7]).tolist(),
                    observable_difference=(obs-other_obs).tolist(), density_bands=bands(rho, other_rho, box),
                    physical_variance_difference_km2_s2=(variance-other_variance).tolist())
            report['comparisons'][label] = row
            np.savez_compressed(out/f'{label}_summary.npz', rho=rho, observables=obs, physical_variance_km2_s2=variance)
            summaries[label] = (obs, rho, variance)
            progress('reference_complete', label=label)
        report['status'] = 'COMPLETE_INDEPENDENT_COMPARISON_DRIVER_JUDGMENT_REQUIRED'
    except Exception as exc:
        report.update(status='FAILED_OR_INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        progress('finish', status=report['status'])


if __name__ == '__main__':
    main()
