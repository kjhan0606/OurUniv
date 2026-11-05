"""Read saved actual-data nuisance projections; no refit or R2 promotion."""
import json
import os
from pathlib import Path

import numpy as np


SOURCE = Path('/gpfs/kjhan/CF4/z0_density/actual_data_longer_v3/task_0')
COUNTS = Path('/gpfs/kjhan/CF4/z0_density/actual_data_corrected_v2_run1/corrected_counts.npz')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_tracer_identifiability_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit saved-chain numerical analysis through Slurm')
    chains = []
    expected = [f'nuisance_{i}' for i in range(60)]
    for chain in range(4):
        with np.load(SOURCE / f'chain_{chain}.npz', allow_pickle=False) as saved:
            names = saved['projection_names'].tolist()
            if names[:60] != expected:
                raise ValueError('saved nuisance projection order changed')
            chains.append(saved['projections'][:, :60].copy())
    draws = np.asarray(chains)
    if draws.shape != (4, 2048, 60) or not np.isfinite(draws).all():
        raise ValueError('unexpected saved actual-data chain shape')
    pooled = draws.reshape(-1, 60)
    with np.load(COUNTS, allow_pickle=False) as saved:
        yes = saved['survival_yes'].copy()
        no = saved['survival_no'].copy()
        exposure = saved['selection_shells'].copy()
        observed = saved['counts_train'].copy()
    if yes.shape != no.shape or yes.shape != (6, 6) or exposure.shape[:2] != (6, 6):
        raise ValueError('saved selection calibration shape changed')
    support = exposure.sum(axis=(2, 3, 4)) > 0
    unsupported_calibration = (yes + no == 0) & support
    rows = []
    for pop in range(6):
        rate_z = pooled[:, pop]
        bias_z = pooled[:, 6 + pop]
        survival_z = pooled[:, 24 + 6 * pop:30 + 6 * pop]
        correlations = np.corrcoef(np.column_stack((bias_z, survival_z)), rowvar=False)[0, 1:]
        rows.append({
            'population': pop,
            'training_count': int(observed[pop].sum()),
            'rate_z_mean': float(rate_z.mean()),
            'rate_z_sd': float(rate_z.std()),
            'bias_z_mean': float(bias_z.mean()),
            'bias_z_sd': float(bias_z.std()),
            'bias_rate_correlation': float(np.corrcoef(bias_z, rate_z)[0, 1]),
            'max_abs_bias_survival_correlation': float(np.max(np.abs(correlations))),
            'supported_shells_without_calibration_marks': np.flatnonzero(
                unsupported_calibration[pop]).tolist(),
            'bias_chain_means_z': draws[:, :, 6 + pop].mean(axis=1).tolist(),
        })
    result = {
        'classification': 'OLD_12_CMPC_H_ACTUAL_MODEL_IDENTIFIABILITY_ONLY',
        'source': str(SOURCE),
        'chains': 4, 'saved_draws_per_chain': 2048,
        'bias_prior_standardized_mean_sd': [0.0, 1.0],
        'bias_prior_log_sd': 0.25,
        'population_rows': rows,
        'supported_population_shells_without_calibration_marks': int(
            unsupported_calibration.sum()),
        'survival_calibration_parent_marks': int((yes + no).sum()),
        'actual_CF4_and_2Mpp_used_in_saved_fit': True,
        'new_fit_or_high_resolution_calibration': False,
        'R2_posterior_delivered': False,
        'interpretation': 'Standardized posterior width and nuisance correlation diagnose the old approximate 12-cMpc/h model only; they do not transfer its bias law to 1.5-cMpc/h PM inference.'
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"job_{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
