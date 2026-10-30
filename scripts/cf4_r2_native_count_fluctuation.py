"""Selection-window count-fluctuation comparison, not a tracer calibration."""
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_native_count_fluctuation_v1')


def native_density(plan, index):
    arm = 'ABCD'[index // 2]
    name = f'posterior_v1_{index:02d}_mock_{index:02d}_seed_{2026083000+index}_arm_{arm}'
    path = Path(plan['data']['posterior_root']) / name / 'posterior_summary.npz'
    with np.load(path, allow_pickle=False) as saved:
        rho = 1.0 + saved['truth_coarse_density'].astype(np.float64)
    if rho.shape != (32, 32, 32) or np.any(rho <= 0) or not np.isfinite(rho).all():
        raise ValueError(f'invalid native PM density: {index}')
    # Native block centroid is 3 cMpc/h, observed count-cell centre is 6.
    # This is the archived mass-conserving recentering convention.
    for axis in range(3):
        rho = 0.75 * rho + 0.25 * np.roll(rho, -1, axis=axis)
    return rho / rho.mean()


def factorial_excess(counts, window, field=None):
    """Conditional-on-total second factorial moment against the window."""
    count = counts[window > 0]
    q = window[window > 0]
    n = count.sum()
    if n < 30 or q.sum() <= 0:
        return None
    baseline = n * q / q.sum()
    denominator = np.sum(baseline**2)
    observed = float(np.sum(count * (count - 1)) / denominator - 1.0)
    if field is None:
        return observed
    tracer = field[window > 0]
    predicted = n * q * tracer / np.sum(q * tracer)
    return float(np.sum(predicted**2) / denominator - 1.0)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit numerical comparison through Slurm')
    plan = json.loads((ROOT / 'config/cf4_z6_native_physics_plan_v1.json').read_text())
    with np.load('/gpfs/kjhan/CF4/z0_density/actual_data_corrected_v2_run1/corrected_counts.npz',
                 allow_pickle=False) as saved:
        counts = saved['counts_all'].copy()
        shells = saved['selection_shells'].copy()
        yes, no = saved['survival_yes'].copy(), saved['survival_no'].copy()
    if counts.shape != (6, 32, 32, 32) or shells.shape != (6, 6, 32, 32, 32):
        raise ValueError('actual count or selection geometry changed')
    if not np.array_equal(counts, counts.astype(np.int64)) or np.any(counts < 0):
        raise ValueError('counts must be nonnegative integers')
    survival = (yes + 1) / (yes + no + 2)
    selection = np.einsum('ps,psijk->pijk', survival, shells)
    if np.any((counts > 0) & (selection <= 0)):
        raise ValueError('actual occupied cell outside calibrated selection support')
    physical_plan = json.loads((ROOT / 'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())
    published = np.asarray(physical_plan['external_population_prior']['published_bias'], dtype=float)
    saved_bias_z = json.loads(Path('/gpfs/kjhan/CF4/z0_density/r2_tracer_identifiability_v1/job_404160.json').read_text())
    fitted = published * np.exp(0.25 * np.asarray([row['bias_z_mean'] for row in saved_bias_z['population_rows']]))
    densities = [native_density(plan, index) for index in range(6)]
    axis = (np.arange(32) + 0.5) * 12.0 - 192.0
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    radius = np.sqrt(x*x + y*y + z*z)
    radial_groups = [(5., 180.)] + list(zip((5., 30., 60., 90., 120., 150.),
                                           (30., 60., 90., 120., 150., 180.)))
    rows = []
    for p in range(6):
        for lower, upper in radial_groups:
            mask = (radius >= lower) & (radius < upper)
            window = np.where(mask, selection[p], 0.0)
            observed = factorial_excess(counts[p], window)
            if observed is None:
                continue
            model = {}
            for label, exponent in [('published', published[p]), ('old_fit', fitted[p])]:
                values = []
                for rho in densities:
                    tracer = rho**exponent
                    values.append(factorial_excess(counts[p], window, tracer / tracer.mean()))
                model[label] = values
            rows.append({'population': p, 'radius_cMpc_h': [lower, upper],
                         'counts': int(counts[p][mask].sum()),
                         'positive_selection_cells': int((window > 0).sum()),
                         'observed_factorial_excess': observed,
                         'published_bias': float(published[p]),
                         'old_fit_bias_diagnostic_only': float(fitted[p]),
                         'PM_factorial_excess': model})
    result = {
        'classification': 'N32_384_OBSERVED_COUNTS_VS_UNCONSTRAINED_PM_WINDOW_STATISTIC',
        'formula': 'sum[n_i(n_i-1)] / sum[(N q_i / sum q)^2] - 1; PM counterpart replaces n_i(n_i-1) with lambda_i^2 and normalizes lambda total to N',
        'selection': 'corrected ARES exposure times calibration-only Beta posterior mean survival; 80% retained count sample',
        'native_field_indices': list(range(6)), 'native_box_cMpc_h': 384.0,
        'count_cell_cMpc_h': 12.0,
        'physical_RSD_in_PM_comparison': False,
        'density_recentered_from_3_to_6_cMpc_h': True,
        'uncertainties_or_independent_validation': False,
        'R2_bias_calibration_or_posterior': False,
        'rows': rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"job_{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(path), 'rows': len(rows), 'classification': result['classification']}), flush=True)


if __name__ == '__main__':
    main()
