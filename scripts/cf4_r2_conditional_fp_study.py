"""Bounded conditional-FP discrimination using existing TRAINING error geometry.

No population fit, field evolution, sampler, held-out outcomes or parent census.
The legacy fitted covariance is a sensitivity reference, not a calibration prior.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from cf4_r2_conditional_fp import (
    CONDITION_MATRIX as A, component_terms, conditioned_log_weights, mixture_logpdf)
from cf4_r2_raw_selected_fp import optical_error_covariance


BASE = Path('/gpfs/kjhan/CF4/z0_density')


def quantiles(values):
    if not np.isfinite(values).all():
        raise ValueError('nonfinite statistic; do not silently discard rows')
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                    np.quantile(values, [0, .05, .5, .95, 1]).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    args.output_dir.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    paths = [BASE/'r2_raw_fp_inputs_v1/training_photometry.npz',
             BASE/'r2_raw_selected_component_v1/training_cut_geometry.npz',
             BASE/'r2_raw_population_fit_v1/final_fit.npz']
    with np.load(paths[0], allow_pickle=False) as f:
        names = list(f['columns'])
        cohorts = {key: (f[key+'_PGC'].copy(), f[key+'_photometry'].copy())
                   for key in ('parent', 'selected')}
        parent_groups = f['parent_source_group'].copy()
    with np.load(paths[2], allow_pickle=False) as f:
        parameters = f['parameters'].copy()
        np.testing.assert_array_equal(np.sort(f['PGC']), np.sort(cohorts['selected'][0]))
    if (len(cohorts['parent'][0]) != 8901 or len(cohorts['selected'][0]) != 1414
            or not set(cohorts['selected'][0]).issubset(cohorts['parent'][0])):
        raise ValueError('training input identity/size changed; review before reuse')
    if any(len(set(ids)) != len(ids) for ids, _ in cohorts.values()):
        raise ValueError('duplicate training PGC; no independent-row assumption')
    b = parameters[:9].reshape(3, 3)
    c = parameters[9:]
    lower = np.array([[np.exp(c[0]), 0, 0], [c[1], np.exp(c[2]), 0],
                      [c[3], c[4], np.exp(c[5])]])
    fitted_cov = lower @ lower.T
    cases = {'development_prior': (.09*np.eye(3), np.array([1., 0, 0])),
             'legacy_cov_fixed_M': (fitted_cov, np.array([1., 0, 0])),
             'legacy_cov_lowz_M_response': (fitted_cov, np.array([1., 0, 0])+5*b[1])}
    info, arrays = {}, {}
    for cohort, (ids, values) in cohorts.items():
        errors = optical_error_covariance(*[values[:, names.index(k)] for k in ('er', 'es', 'ei')])
        if cohort == 'selected':
            with np.load(paths[1], allow_pickle=False) as f:
                order = {int(p): i for i, p in enumerate(f['PGC'])}
                ix = np.array([order[int(p)] for p in ids])
                np.testing.assert_allclose(errors, f['optical_error_covariance'][ix], atol=1e-14)
                np.testing.assert_allclose(A, f['cut_matrix'], atol=1e-14)
        info[cohort] = {}
        for label, (cov, direction) in cases.items():
            terms = component_terms(np.zeros(3), cov+errors, np.zeros(2), mean_direction=direction)
            fraction = terms['conditional_fisher']/terms['joint_fisher']
            if np.any((fraction < -1e-12) | (fraction > 1+1e-12)):
                raise AssertionError('conditional information exceeds the joint information')
            info[cohort][label] = dict(rows=len(ids), information_fraction=quantiles(fraction),
                conditional_information=quantiles(terms['conditional_fisher']),
                mean_response=quantiles(terms['mean_response']),
                zero_information_rows=int(np.sum(terms['conditional_fisher'] == 0)))
            arrays[cohort+'_'+label] = fraction

    mean = np.array([.3, 2.2, 2.7])
    means = mean + np.array([[-.2, 0, 0], [.2, 0, 0]])
    u = A @ mean + [1.2, .024]
    mixture = component_terms(means, .09*np.eye(3), u)
    lw = np.log([.5, .5])
    weights = np.exp(conditioned_log_weights(lw, mixture['log_condition_density']))
    integral = quad(lambda t: np.exp(mixture_logpdf(t, mixture, lw)), -3, 4)[0]
    if abs(integral-1) > 1e-9:
        raise AssertionError('synthetic mixture normalization failed')
    stress = []
    for gamma in (0., .25, .5, 1.):
        mass = quad(lambda e: norm.pdf(e)*expit(gamma*e), -10, 10)[0]
        shift = quad(lambda e: e*norm.pdf(e)*expit(gamma*e)/mass, -10, 10)[0]
        stress.append(dict(gamma=gamma, selection_mass=mass, residual_mean_shift_sigma=shift))
    constant = [dict(probability=p, integral=quad(lambda e: p*norm.pdf(e), -10, 10)[0]/p)
                for p in (.05, .2, .9)]
    # Four examples correspond to the four scientific checks, not a text-only PDF.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'axes.unicode_minus': False,
                         'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 9,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=.08, right=.97, top=.91, bottom=.22, hspace=.4, wspace=.3)
    def title(ax, label):
        ax.text(.03, .97, label, transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    e = np.linspace(-4, 4, 401)
    ax = axes[0, 0]
    for p, style in zip((.05, .2, .9), ('-', '--', ':')):
        ax.plot(e, p*norm.pdf(e)/p, style, label=f'선택확률 {p:.2f}')
    title(ax, '1. 조건변수에만 의존하는 선택: 분포가 동일')
    ax.set(xlabel='표준화한 크기 잔차 (예제)', ylabel='조건부 확률밀도', ylim=(0, .56))
    ax.legend(loc='lower right')
    ax = axes[0, 1]
    grid = np.linspace(.2, 1.35, 501)
    correct = np.exp(mixture_logpdf(grid, mixture, lw))
    stale = np.mean(norm.pdf(grid[:, None], mixture['mean'], np.sqrt(mixture['variance'])), axis=1)
    ax.plot(grid, correct, label='조건변수로 후보 가중치 갱신')
    ax.plot(grid, stale, '--', label='이전 가중치 유지 (틀린 계산)')
    title(ax, f'2. 두 후보 거리 예제: 적분 {integral:.8f}')
    ax.set(xlabel='관측 log 크기 (예제)', ylabel='조건부 확률밀도', ylim=(0, 4.2))
    ax.legend(loc='lower left')
    ax = axes[1, 0]
    for cohort, label in (('parent', '기존 훈련 원자료 8901개'), ('selected', '연결 표본 1414개')):
        ax.hist(arrays[cohort+'_legacy_cov_fixed_M'], bins=np.linspace(0, 1, 31),
                density=True, histtype='step', label=label)
    title(ax, '3. 조건화 후 남는 국소 거리정보')
    ax.set(xlabel='조건부 / 공동 Fisher 정보 (고정 M·기존 공분산)', ylabel='행별 분포')
    ymin, ymax = ax.get_ylim(); ax.set_ylim(ymin, ymax*1.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1., .83))
    ax = axes[1, 1]
    for case in stress:
        gamma = case['gamma']
        ax.plot(e, norm.pdf(e)*expit(gamma*e)/case['selection_mass'], label=f'잔차 의존 강도 {gamma:g}')
    title(ax, '4. 잔차에 의존하는 선택: 거리 편향 가능')
    ax.set(xlabel='표준화한 크기 잔차 (예제)', ylabel='선택 후 확률밀도', ylim=(0, .61))
    ax.legend(loc='upper right', bbox_to_anchor=(1., .83))
    fig.suptitle('R2 조건부 FP 후보 — 작은 수학 검사와 실제 훈련 오차 구조', fontsize=14)
    fig.text(.08, .145, '1·2·4는 해석 가능한 예제이며 실제 우주의 보정 검사가 아닙니다. 3만 기존 훈련자료의 측정오차를 사용합니다.\n'
             '기존 fitted 공분산은 비교용입니다. 정보 비율은 보정변수를 고정한 국소값이며, 공유 영점과 거리는 여전히 혼동됩니다.\n'
             '형태·품질·이상치 선택은 미해결입니다. held-out·현재장·시뮬레이션은 사용하지 않았습니다. R2 미완료; MW/M31/M33 미식별.',
             va='top', fontsize=9, linespacing=1.6)
    fig.savefig(args.output_dir/'conditional_fp_examples.png', dpi=120)
    fig.savefig(args.output_dir/'R2_조건부_FP_예제보고.pdf')
    plt.close(fig)
    result = dict(status='CONDITIONAL_FP_COMPONENT_STUDY_NOT_SELECTION_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        inputs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        training_columns=names, parent_source_groups=len(np.unique(parent_groups)),
        parent_is_complete_source_catalogue=False, information=info,
        synthetic_mixture_integral=integral, synthetic_conditioned_weights=weights.tolist(),
        synthetic_correct_mean=float(weights @ mixture['mean']),
        synthetic_incorrect_unreweighted_mean=float(np.mean(mixture['mean'])),
        fixed_covariate_selection=constant, residual_selection_stress=stress,
        shared_zero_point_and_global_eta_information_rank=1,
        legacy_population_parameters=parameters.tolist(), legacy_covariance=fitted_cov.tolist(),
        parameter_fits=0, heldout_scored=False, PM_evolutions=0, live_target_changed=False,
        selection_calibrated=False, R2_complete=False,
        limitations=['legacy covariance fitted to the same selected data; not independent calibration',
            'per-row fixed-covariance local Fisher information, not marginalized uncertainty or field information',
            'low-z M direction omits K-corrections and changing latent-source weights; not full field derivative',
            'shared zero point remains degenerate; group members not independent replicates',
            'known measured-covariate cuts may cancel; morphology/quality/outlier/linkage selection not calibrated',
            'no native FP-only source kernel or same-field multi-member likelihood implemented',
            'stress strength is an illustrative axis, not an observed bound or adopted prior',
            'MW/M31 roles ambiguous and M33 unresolved; same NEW field constraints remain required'],
        elapsed_seconds=time.monotonic()-start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (args.output_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
