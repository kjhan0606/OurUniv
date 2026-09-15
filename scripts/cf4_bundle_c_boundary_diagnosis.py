"""Fixed stored-field diagnosis, no fitting, resampling or acceptance changes."""
import json
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np

from cf4_bundle_c_continuous import read_periodic_patch

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT/'stable_field_boundary_v1'


def ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else None


def phase_profile(plane_energy, faces, period):
    means = np.array([plane_energy[faces % period == p].mean() for p in range(period)])
    return dict(means=means.tolist(), counts=[int(np.sum(faces % period == p)) for p in range(period)],
        normalized=[ratio(x, means.mean()) for x in means],
        boundary_to_other=ratio(means[0], means[1:].mean()))


def diagnostics(mass):
    if mass.shape != (64,)*3 or not np.isfinite(mass).all() or mass.min() < 0 or mass.mean() <= 0:
        raise ValueError('finite positive-total nonnegative64 mass cube required')
    normalized = mass/mass.mean()
    result = {}
    for name, field in (('density', normalized-1), ('log_density', np.log1p(normalized))):
        rows = []
        for axis in range(3):
            energy = np.moveaxis(np.diff(field, axis=axis)**2, axis, 0)
            faces = np.arange(1, 64)
            plane = energy.mean(axis=(1, 2))
            original = ratio(plane[faces % 8 == 0].mean(), plane[faces % 8 != 0].mean())
            # Axis-oriented face right indices8..55; no periodic wrap joins.
            interior = energy[7:55, 8:56, 8:56]
            interior_faces = np.arange(8, 56)
            balanced = interior.mean(axis=(1, 2))
            concentration = []
            for selected in (interior_faces % 8 == 0, interior_faces % 8 != 0):
                flat = interior[selected].ravel()
                largest = np.partition(flat, -10)[-10:]
                concentration.append(dict(total=float(flat.sum()),
                    largest_edge_fraction=ratio(largest.max(), flat.sum()),
                    largest10_edges_fraction=ratio(largest.sum(), flat.sum())))
            rows.append(dict(axis=axis, original_boundary_to_other=original,
                original_plane_mean_squared_gradient=plane.tolist(),
                balanced_plane_mean_squared_gradient=balanced.tolist(),
                phases={str(p): phase_profile(balanced, interior_faces, p) for p in (2, 4, 8)},
                concentration_boundary_then_other=concentration))
        result[name] = rows
    return result


def controls():
    faces = np.arange(8, 56)
    for period in (2, 4, 8):
        ramp = phase_profile(np.ones(48), faces, period)
        np.testing.assert_allclose(ramp['normalized'], 1.)
        assert len(set(ramp['counts'])) == 1
    # Compute real differences of fields, not just a hand-coded phase table.
    x = np.arange(64, dtype=float)[:, None, None]
    ramp = np.broadcast_to(1+x, (64,)*3)
    staircase = np.broadcast_to(1+np.floor(x/8), (64,)*3)
    a, b = diagnostics(ramp)['density'][0], diagnostics(staircase)['density'][0]
    np.testing.assert_allclose(a['phases']['8']['normalized'], 1., rtol=1e-12)
    np.testing.assert_allclose(b['phases']['8']['normalized'], [8., 0., 0., 0., 0., 0., 0., 0.])
    assert b['phases']['8']['boundary_to_other'] is None  # positive/zero, not silently finite
    return dict(tests=2, passed=2, zero_denominator='null, with numerator/phase means retained')


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run():
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm required')
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    dump('tests.json', controls())
    archive = ROOT/'stable_field_eval_v2'
    previous = json.loads((archive/'result.json').read_text())
    if not previous['evaluation_complete'] or previous['valid_draws'] != 16:
        raise ValueError('requires the completed frozen16-draw evaluation')
    draws = json.loads((archive/'draws.json').read_text())
    first = [r for r in draws if r['draw'] == 0]
    if len(first) != 8 or len({r['mw'] for r in first}) != 8:
        raise ValueError('all eight first draws required')
    cases = {c['mw']: c for c in json.loads((ROOT/'population_locations_v1/cases.json').read_text())['test']}
    rows = []
    with h5py.File(archive/'first_draw_fields.h5', 'r') as saved, h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as native:
        for row in first:
            mw = row['mw']
            source = read_periodic_patch(native['fine'], np.asarray(cases[mw]['origin']), 64)[0]
            generated = saved[f'mw_{mw}/moments80'][0, 8:-8, 8:-8, 8:-8]
            truth, draw = diagnostics(source), diagnostics(generated)
            reproduced = [ratio(draw['density'][a]['original_boundary_to_other'],
                truth['density'][a]['original_boundary_to_other']) for a in range(3)]
            np.testing.assert_allclose(reproduced, row['comparison']['ratios']['parent_boundary_gradient_ratio'], rtol=1e-12)
            rows.append(dict(mw=mw, native=truth, generated=draw, original_ratio_reproduced=reproduced))
            dump('profiles.json', rows)
            print(json.dumps(dict(status='SCORED', mw=mw, completed=len(rows), total=8)), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for i, name in enumerate(('density', 'log_density')):
        for j, period in enumerate((2, 4, 8)):
            ax = axes[i, j]
            for role, color in (('native', 'black'), ('generated', 'tab:orange')):
                values = np.array([r[role][name][a]['phases'][str(period)]['normalized'] for r in rows for a in range(3)])
                for value in values:
                    ax.plot(range(period), value, color=color, alpha=.12, linewidth=.7)
                ax.plot(range(period), np.median(values, axis=0), color=color, label=f'{role} median', linewidth=2)
            ax.axhline(1., color='grey', linestyle=':')
            ax.set(title=f'{name}: period{period}', xlabel='face phase (0=refinement boundary)', ylabel='gradient energy / phase mean')
            ax.legend()
    fig.suptitle('8 stored first draws x3 axes; correlated development regions; NOT a new acceptance gate')
    fig.savefig(OUT/'phase_profiles.png', dpi=120)
    plt.close(fig)
    summary = {}
    for name in ('density', 'log_density'):
        summary[name] = {role: {str(p): dict(
            median_phase_profile=np.median([r[role][name][a]['phases'][str(p)]['normalized'] for r in rows for a in range(3)], axis=0).tolist(),
            boundary_to_other=[r[role][name][a]['phases'][str(p)]['boundary_to_other'] for r in rows for a in range(3)])
            for p in (2, 4, 8)} for role in ('native', 'generated')}
    dump('result.json', dict(status='COMPLETE_DESCRIPTIVE_DIAGNOSIS_DRIVER_JUDGMENT_REQUIRED',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'],
        source_evaluation_job=previous['job_id'], cases=8, additional_draws=0, optimizer_steps=0,
        original_metric_reproduced=True, original_generator_verdict_unchanged=previous['status'],
        summary=summary, seconds=time.monotonic()-started,
        host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))


if __name__ == '__main__':
    run()
