"""One frozen native composite-proxy fit/report. Slurm only, no actual LG fit."""
from collections import Counter
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cf4_lg_field_candidates import match_for_evaluation
from cf4_lg_composite_proxy import (pair_context, fit, score, predictive_moments,
    latent_observables, latent_from_observables)
from cf4_lg_observation_contract import solar_reference

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT/'composite_proxy_v1'
TRAIN = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15]
HELD = [16, 17, 20]
DX = .1875


def truth(record, profiles, h):
    members = [profiles[i] for i in record['ids']]
    positions = np.array([(np.array(p['position_ckpc_h'])/1000-record['patch_lower_cMpc_h']) % 75 for p in members])
    velocities = np.array([p['COM_velocity_km_s'] for p in members])
    relative = np.array([(positions[1]-positions[0])*1000/h,
                         (positions[2]-positions[1])*1000/h,
                         velocities[1]-velocities[0], velocities[2]-velocities[1]])
    return positions, relative


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('EXPECTED_COMMIT'):
        raise RuntimeError('Slurm job and frozen source commit required')
    split = json.loads((ROOT/'spatial_conditional_v1/result.json').read_text())
    if (split['training_patches'] != TRAIN or split['heldout_patches'] != HELD
            or split['train_heldout_native_voxel_overlap'] or split['heldout_mutual_voxel_overlap']):
        raise ValueError('frozen geometry split mismatch')
    request = json.loads((ROOT/'spatial_calibration_v1/request.json').read_text())
    h = request['h']
    OUT.mkdir(exist_ok=False)
    report = dict(status='STARTED_NOT_COMPLETE', source_commit=os.environ['EXPECTED_COMMIT'],
        job_id=os.environ['SLURM_JOB_ID'], training_fixtures=TRAIN, heldout_fixtures=HELD,
        dx_cMpc_h=DX, h=h, neural_parameters=0, optimizer_updates=0,
        fitted_statistical_coefficients=14, fixed_shrinkage=.1,
        target_rows=['r31_xyz_kpc', 'r32_xyz_kpc', 'v31_xyz_km_s', 'v32_xyz_km_s'],
        context_status={}, field_aperture_summary={}, training_associations=[],
        limits=['Selected satellite cases from one native TNG box, not independent cosmologies.',
                'Cartesian kinematic proxy density, not resolved components, mass likelihood or actual LG posterior.',
                'No field is inferred or updated; field dependence is not calibrated information gain.',
                'Uniform candidate-pair probabilities and isotropic K tensor I are uncalibrated approximations.',
                'Missing p(E|F,O)/E-conditioned field prior, q_F morphology and observation systematics remain.',
                'Unsupported contexts are not low scientific scores or evidence against a universe.'])

    def save():
        (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    save()
    contexts, positions = {}, {}
    # Construct all contexts without loading truth profiles/positions.
    with h5py.File(ROOT/'lg_identification_v1/candidates.h5', 'r') as source:
        if source.attrs['status'] != 'FIELD_ONLY_CANDIDATES_NOT_IDENTIFIED_HALOS':
            raise ValueError('candidate source label mismatch')
        report['candidate_source_commit'] = str(source.attrs['source_commit'])
        for row in TRAIN+HELD:
            group = source[str(row)]
            p, velocity, sigma = group['positions'][:], group['mean_velocity'][0].T, group['physical_sigma'][0].T
            moments = group['moments'][0]
            pairs = group['mw_m31_edges'][:]
            try:
                if np.any(moments[0] <= 0) or not np.isfinite(moments).all():
                    raise ValueError('nonpositive/nonfinite aperture mass')
                np.testing.assert_allclose(velocity, (moments[1:4]/moments[0]).T, atol=1e-8)
                contexts[row] = pair_context(p, velocity, sigma, pairs, dx=DX, h=h)
            except (ValueError, AssertionError) as error:
                contexts[row] = dict(status='INVALID_FIELD_CONTEXT', reason=str(error))
            positions[row] = p
            report['context_status'][str(row)] = contexts[row]['status']
            report['field_aperture_summary'][str(row)] = dict(
                candidates=len(p), pairs=len(pairs),
                mass_Msun_minmax=[float(moments[0].min()), float(moments[0].max())] if len(p) and np.isfinite(moments[0]).all() else None,
                physical_sigma_km_s_minmax=[float(sigma.min()), float(sigma.max())] if len(p) and np.isfinite(sigma).all() else None,
                aperture2_radius_cells=2, aperture2_use='descriptive only; cached source preserved',
                warning='Total-matter aperture mass, NOT a bound member/M200c or disjoint mass sum')
    native = json.loads((ROOT/'spatial_calibration_v1/result.json').read_text())
    profiles = {p['sid']: p for p in native['profiles']}
    train_ids = {sid for row in TRAIN for sid in native['patches'][row]['ids']}
    held_ids = {sid for row in HELD for sid in native['patches'][row]['ids']}
    if train_ids & held_ids:
        raise ValueError('native member IDs leak across frozen split')
    seen, residuals, directions = set(), [], []
    member_occurrences = Counter()
    for row in TRAIN:
        patch = native['patches'][row]
        p, target = truth(patch, profiles, h)
        context = contexts[row]
        entry = dict(fixture=row, ids=patch['ids'], included=False)
        if context['status'] == 'SUPPORTED_DIAGNOSTIC_ONLY':
            match = match_for_evaluation(positions[row], p[:2], DX)
            entry['two_host_match'] = match
            pair = tuple(match['matched'])
            indices = np.flatnonzero(np.all(context['pairs'] == pair, axis=1))
            key = tuple(patch['ids'][:2])
            if min(pair) >= 0 and len(indices) == 1 and key not in seen:
                index = indices[0]
                seen.add(key)
                residuals.append((target-context['base'][index])/context['scales'][index, :, None])
                directions.append(context['direction'][index])
                member_occurrences.update(patch['ids'])
                entry.update(included=True, candidate_pair=list(pair), context_pair_index=int(index))
            else:
                entry['reason'] = 'unmatched/ineligible host pair or duplicate native host pair'
        else:
            entry['reason'] = context['status']
        report['training_associations'].append(entry)
    report['training_unique_pairs'] = len(seen)
    report['training_excluded'] = len(TRAIN)-len(seen)
    report['training_repeated_member_ids'] = {str(k): v for k, v in member_occurrences.items() if v > 1}
    try:
        model = fit(np.asarray(residuals), np.asarray(directions))
    except (ValueError, np.linalg.LinAlgError) as error:
        report.update(status='NO_GO_PROXY_FIT_NO_TUNED_REPAIR', failure=str(error))
        save()
        print(json.dumps(report), flush=True)
        return
    report['model'] = dict(beta=model['beta'].tolist(), K=model['K'].tolist(), **model['diagnostics'])
    report['cross_scores'] = []
    report['heldout_predictions'] = []
    matrices = {key: np.full((3, 3), np.nan) for key in ('log_joint', 'log_host', 'log_M33_given_host')}
    for i, row in enumerate(HELD):
        _, target = truth(native['patches'][row], profiles, h)
        for j, field in enumerate(HELD):
            result = score(target, model, contexts[field])
            report['cross_scores'].append(dict(target_fixture=row, field_fixture=field, **result))
            for key, matrix in matrices.items():
                if result[key] is not None:
                    matrix[i, j] = result[key]
        if contexts[row]['status'] == 'SUPPORTED_DIAGNOSTIC_ONLY':
            mean, std = predictive_moments(model, contexts[row])
            report['heldout_predictions'].append(dict(fixture=row, native_target=target.tolist(),
                equal_pair_predictive_mean=mean.tolist(), equal_pair_predictive_std=std.tolist(),
                uncertainty='Proxy predictive spread; NOT physical dispersion or inferred-field uncertainty'))
    report['matrix_axis_order'] = dict(rows_target=HELD, columns_field=HELD)
    report['own_field_minus_other_fields_mean_logscore'] = {}
    for key, matrix in matrices.items():
        report[key+'_matrix'] = [[None if not np.isfinite(v) else float(v) for v in row] for row in matrix]
        report['own_field_minus_other_fields_mean_logscore'][key] = [
            float(matrix[i, i]-np.delete(matrix[i], i).mean()) if np.isfinite(matrix[i]).all() else None for i in range(3)]
    # Fixed noiseless native mock, not same-generator calibration validation.
    _, target = truth(native['patches'][HELD[0]], profiles, h)
    contract = json.loads(Path('config/cf4_lg_observation_contract_v1.json').read_text())
    solar_pos, solar_vel = solar_reference(contract)
    kwargs = dict(h=h, solar_position_kpc=solar_pos, solar_velocity_km_s=solar_vel)
    obs = latent_observables(target, **kwargs)
    back = latent_from_observables(obs, **kwargs)
    np.testing.assert_allclose(back, target, atol=1e-8, rtol=1e-10)
    report['mock_roundtrip'] = dict(fixture=HELD[0], max_abs_error=float(abs(back-target).max()),
        observable_columns=['RA_deg', 'Dec_deg', 'DM_mag', 'vlos_km_s', 'pmra_star_uas_yr', 'pmdec_uas_yr'],
        observables=obs.tolist(), frame='BOX axes interpreted as mock ICRS only; not observed sky conditioning')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, matrix) in zip(axes, matrices.items()):
        # Row centered only for plotting; original unit-normalized log densities saved.
        centered = matrix-np.nanmean(matrix, axis=1, keepdims=True) if np.isfinite(matrix).any() else matrix
        im = ax.imshow(np.ma.masked_invalid(centered), cmap='coolwarm')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{centered[i,j]:.2f}' if np.isfinite(centered[i,j]) else 'UNSUPPORTED', ha='center', fontsize=8)
        ax.set_xticks(range(3), HELD); ax.set_yticks(range(3), HELD)
        ax.set_xlabel('Field context'); ax.set_ylabel('Native target')
        ax.set_title(key+'\nrow-centered nats; not information gain')
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout(); fig.savefig(OUT/'field_context_scores.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for i, record in enumerate(report['heldout_predictions']):
        y, mean, std = [np.asarray(record[k]) for k in ('native_target', 'equal_pair_predictive_mean', 'equal_pair_predictive_std')]
        for j, rows in enumerate(((0, 1), (2, 3))):
            axes[i, j].errorbar(range(6), mean[list(rows)].ravel(), yerr=std[list(rows)].ravel(), fmt='o', label='equal-pair proxy mean +/-1 SD')
            axes[i, j].scatter(range(6), y[list(rows)].ravel(), marker='x', label='native truth: evaluation only')
            axes[i, j].set_xticks(range(6), ['31x','31y','31z','32x','32y','32z'])
            axes[i, j].set_title(f'Fixture{record["fixture"]}: '+('position kpc' if j == 0 else 'peculiar velocity km/s'))
            axes[i, j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/'native_proxy_kinematics.png', dpi=140); plt.close(fig)
    report['status'] = 'NATIVE_COMPOSITE_PROXY_DIAGNOSTIC_COMPLETE_NOT_LG_POSTERIOR'
    save()
    print(json.dumps({k: report[k] for k in ('status', 'model', 'own_field_minus_other_fields_mean_logscore')}), flush=True)


if __name__ == '__main__':
    main()
