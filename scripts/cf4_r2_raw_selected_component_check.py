"""One raw-mark component bundle: mechanics and actual training cut geometry.

Not a population fit, selected-survey calibration or new field posterior.
"""
import json
import os
from pathlib import Path
import resource
import subprocess

import numpy as np

from cf4_r2_raw_selected_fp import (log_box_probability,
    optical_error_covariance, optical_cut_geometry)
from test_cf4_r2_raw_selected_fp import normalized_example


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(exist_ok=False)
    base = Path('/gpfs/kjhan/CF4/z0_density')
    with np.load(base/'r2_raw_fp_inputs_v1/training_photometry.npz', allow_pickle=False) as f:
        names = list(f['columns'])
        pgc, data = f['selected_PGC'], f['selected_photometry']
    if len(pgc) != 1414 or len(np.unique(pgc)) != 1414:
        raise ValueError('frozen cohort changed')
    wanted = set(map(int, pgc))
    raw = {}
    fields = ['SIGMA_STARS', 'SIGMA_STARS_ERR', 'plate', 'deVRad_r', 'deVAB_r']
    with Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat').open() as handle:
        cols = handle.readline().lstrip('#').split()
        pi, indexes = cols.index('PGC'), [cols.index(k) for k in fields]
        for line in handle:
            values = line.split()
            if values and int(values[pi]) in wanted:
                p = int(values[pi])
                if p in raw:
                    raise ValueError('duplicate source identity')
                raw[p] = [float(values[j]) for j in indexes]
    if set(raw) != wanted:
        raise ValueError('source join incomplete')
    extra = np.array([raw[int(p)] for p in pgc])
    x = data[:, [names.index(k) for k in ('r', 's', 'i')]]
    theta = extra[:, 3]*np.sqrt(extra[:, 4])
    aperture = np.where(extra[:, 2] < 3510, 1.5, 1.)
    reconstructed_s = np.log10(extra[:, 0])-.04*np.log10(theta/(8*aperture))
    aperture_error = reconstructed_s-x[:, 1]
    matrix, lower, upper = optical_cut_geometry(x, data[:, names.index('deVMag_r')], extra[:, 0])
    y = x@matrix.T
    outside = ((y < lower) | (y > upper)).any(axis=-1)
    covariance = optical_error_covariance(*[data[:, names.index(k)] for k in ('er','es','ei')])
    result = dict(status='RAW_MARK_COMPONENT_CHECK_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        training_rows=len(pgc),outside_known_cuts_PGC=pgc[outside].tolist(),
        max_aperture_s_discrepancy=float(np.max(np.abs(aperture_error))),
        geometry_consistent=bool(not outside.any() and np.max(np.abs(aperture_error)) < 2e-5),
        R2_complete=False,live_target_changed=False,heldout_scored=False,PM_evolutions=0,
        population_parameters_fitted=0,selection_calibrated=False,
        limitations=['known magnitude/raw-sigma cuts only; not morphology, rejection or graph incidence',
            'source optical covariance prescription, not independently measured covariance',
            'no new K-error model, independent calibration or actual-source mixture fitted',
            'synthetic normalization is mechanics only; MW/M31 ambiguous, M33 unresolved'])
    np.savez(out/'training_cut_geometry.npz',PGC=pgc,x=x,extra_columns=np.array(fields),
        extra=extra,cut_matrix=matrix,cut_lower=lower,cut_upper=upper,
        optical_error_covariance=covariance,aperture_s_discrepancy=aperture_error)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3})
    pdf_path = out/'R2_원자료관측모형_예제보고.pdf'
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=.83,bottom=.3,wspace=.32)
        rho = np.linspace(-.85,.85,35)
        probability = np.array([np.exp(log_box_probability([0.,0.],[[1.,v],[v,1.]],
            [-np.inf,-np.inf],[0.,0.],order=192)) for v in rho])
        analytic = .25+np.arcsin(rho)/(2*np.pi)
        axes[0].plot(rho,probability,label='구현 적분')
        axes[0].plot(rho,analytic,'--',label='독립적인 해석식')
        axes[0].axhline(.25,color='red',ls=':',label='독립이라고 곱한 값 (일반적으로 틀림)')
        axes[0].set(xlabel='두 측정값의 상관계수',ylabel='두 조건을 동시에 만족할 확률')
        axes[0].legend(fontsize=8)
        integral,r,k,density,_ = normalized_example()
        im = axes[1].pcolormesh(k,r,density,shading='auto')
        fig.colorbar(im,ax=axes[1],label='정규화된 모형 밀도')
        axes[1].set(xlabel='예제 K 겉보기 등급',ylabel='예제 관측 크기 (log)',
                    title=f'선택된 예제 분포: 면적 적분 {integral:.8f}')
        fig.suptitle('R2 진행 중 — 선택확률과 결합 정규화 구현',fontsize=17)
        fig.text(.07,.2,'왼쪽: 상관된 관측 조건의 확률은 각각의 확률을 곱하면 안 됩니다.\n'
            '오른쪽: 동일한 은하수 광도함수를 유지하면서, 밝기·크기를 함께 점수화한 작은 예제입니다.\n'
            '기존 거리 PDF를 또 곱하지 않습니다. 예제 적분 성공은 실제 CF4 보정 성공이 아닙니다.',
            va='top',fontsize=11,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
        result['orthant_max_abs_error']=float(np.max(np.abs(probability-analytic)))
        result['synthetic_joint_integral']=float(integral)
        fig,axes = plt.subplots(1,2,figsize=(11.69,8.27))
        fig.subplots_adjust(top=.83,bottom=.3,wspace=.32)
        axes[0].scatter(data[:,names.index('deVMag_r')],extra[:,0],s=4,alpha=.4)
        for v in (10.,17.): axes[0].axvline(v,color='red',ls='--')
        for v in (70.,420.): axes[0].axhline(v,color='red',ls='--')
        axes[0].set(xlabel='실제 훈련 은하 r 등급',ylabel='보정 전 측정 속도분산 (km/s)',
                    title=f'1414개 자료: 알려진 절단 밖 {outside.sum()}개')
        axes[1].scatter(x[:,1],aperture_error*1e6,s=4,alpha=.4)
        axes[1].axhline(0,color='grey')
        axes[1].set(xlabel='공개된 보정 후 log 속도분산',ylabel='재계산 − 공개 값 (백만분의 1 dex)',
                    title='실제 각반경·분광판 번호로 재계산')
        fig.suptitle('R2 진행 중 — 실제 훈련 원자료와 선택 좌표 연결',fontsize=17)
        fig.text(.07,.2,'왼쪽: 공개된 밝기와 원래 속도분산의 절단을 실제 1414개 훈련 자료에 대조했습니다.\n'
            '오른쪽: 관측 구경 보정식을 확인합니다. 크기·표면밝기 오차에는 원 논문의 반상관을 보존합니다.\n'
            '형태·매칭·그룹 선택, 공통 보정, 현재장 posterior는 미완료. MW/M31 미식별, M33 미해결.',
            va='top',fontsize=11,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
    result['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf_path)],check=True,timeout=90)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['geometry_consistent']:
        raise RuntimeError('actual source geometry mismatch; do not promote known-cut model')


if __name__ == '__main__':
    main()
