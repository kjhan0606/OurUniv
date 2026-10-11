"""Illustrated Korean R2 progress PDF from completed saved readouts only."""
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit report generation through Slurm')
    fit=BASE/'r2_v6_single_mark_map_v1'
    counts=BASE/'r2_shell_cdf_field_check_v14'
    distance=BASE/'r2_v6_single_mark_fp_readout_v1'
    prelude=BASE/'r2_fp_single_mark_response_v1'
    out=Path(os.environ['CF4_R2_OUT_DIR'])
    reports=[json.loads((p/'result.json').read_text()) for p in (fit,counts,distance,prelude)]
    fitted,check,fp,comparison=reports
    if (not (fit/'final_state.npz').is_file()
            or check['status']!='SAVED_FIELD_COMPARISON_COMPLETE_NOT_POSTERIOR'
            or 'CDF4x32_vs_CDF8x32' not in check
            or fp['status']!='TRAINING_FP_DISTANCE_READOUT_COMPLETE_NOT_POSTERIOR'
            or comparison['status']!='FP_ZERO_RESPONSE_COMPLETE_NOT_POSTERIOR'):
        raise ValueError('required saved result is incomplete; do not fabricate a report')
    font=Path('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'font.size':11,'pdf.fonttype':3})
    out.mkdir(exist_ok=False)
    provenance='계산 '+str(fitted['job_id'])+' · 수치/수자료 '+str(check['job_id'])+' · 거리 '+str(fp['job_id'])
    def page(title,caption):
        fig=plt.figure(figsize=(11.69,8.27))
        fig.text(.05,.955,title,fontsize=19,weight='bold')
        fig.text(.05,.88,'R1 → R2 진행 중 → R3 국부은하군 → R4 정밀 진화 → R5 줌 초기조건',fontsize=12)
        fig.text(.05,.17,caption,va='top',fontsize=11,linespacing=1.6)
        fig.text(.05,.025,provenance+' | 학습 자료 비교이며 독립 검증·posterior 완성 판정이 아닙니다.',fontsize=9)
        return fig
    def picture(fig,path):
        ax=fig.add_axes((.055,.25,.89,.57)); ax.imshow(plt.imread(path)); ax.axis('off')
    with PdfPages(out/'R2_진행보고_예제그림.pdf') as pdf:
        fig=page('1. 어떤 현재 우주 지도가 만들어졌나?',
            '색은 밀도(위)와 SGZ 방향 속도 성분(아래)을 나타냅니다. 왼쪽은 시작, 오른쪽은 이번 적합 후입니다.\n'
            '현재 한 칸은 3 cMpc/h입니다. 주변 목표 1–2, 국부은하군 목표 ≤0.3보다 아직 거칩니다.\n'
            '무늬가 보인다는 것만으로 관측 구조를 복원한 것은 아닙니다. MW·M31은 미식별, M33도 미해결입니다.')
        picture(fig,fit/'readout/field_comparison.png'); pdf.savefig(fig); plt.close(fig)

        integral=check['CDF4x32_vs_CDF8x32']
        fig=page('2. 적분을 더 촘촘히 해도 계산이 일치하나?',
            f"예측량의 상대 차이 합계: {integral['exposure_L1_relative']:.3g}; 거리별 적분 방식을 더 촘촘히 대조했습니다.\n"
            '그래프는 같은 저장 장에서 계산법만 바꾼 은하 수 점수 차이입니다. 0에 가까울수록 기준 계산과 가깝습니다.\n'
            '작은 수치 차이는 계산의 일관성이지, 관측 오차·선택효과·물리 모형이 맞다는 보장은 아닙니다.')
        ax=fig.add_axes((.14,.30,.75,.48))
        reference=check['comparisons']['CDF8x32']['score']
        labels=list(check['comparisons'])
        delta=[check['comparisons'][k]['score']-reference for k in labels]
        ax.barh(labels,delta,color=['#999999','#377eb8','#4daf4a'])
        ax.axvline(0,color='black',lw=.7); ax.set_xlabel('더 촘촘한 CDF8×32 기준과의 로그 점수 차이')
        for y,value in enumerate(delta):
            ax.annotate(f'{value:.6g}',(value,y),xytext=(5,8),textcoords='offset points')
        pdf.savefig(fig); plt.close(fig)

        grad=check['velocity_scale_adjoint']
        fig=page('3. 속도를 바꿀 때 점수의 기울기를 맞게 계산하나?',
            f"두 경로의 상대 차이: {grad['relative_discrepancy']:.3g}. 같은 장의 속도 크기를 아주 조금 바꾸어 대조했습니다.\n"
            '파란 막대는 자동미분, 주황 막대는 실제로 값을 양쪽으로 바꿔 계산한 기울기입니다.\n'
            '이는 이 방향의 수치 검사입니다. 모든 방향의 정확도나 posterior 수렴을 증명하지 않습니다.')
        ax=fig.add_axes((.14,.30,.75,.48))
        values=[grad['reverse'],grad['finite_difference']]
        ax.bar(['자동미분','양쪽 변화량으로 계산'],values,color=['#377eb8','#ff7f00'])
        ax.set_ylabel('속도 배율에 대한 점수 기울기')
        for x,value in enumerate(values):
            ax.annotate(f'{value:.8g}',(x,value),xytext=(0,8),ha='center',textcoords='offset points')
        pdf.savefig(fig); plt.close(fig)

        radial=check['training_radial_profile']
        fig=page('4. 관측된 은하 수를 거리별로 재현하나?',
            f"총 관측 {radial['observed_total']:,}개, 예측 합계 {radial['expected_total']:,.1f}개입니다.\n"
            '총합이 가까워도 특정 거리에서는 과대·과소 예측할 수 있습니다. 점과 곡선의 거리별 차이가 핵심입니다.\n'
            '학습에 사용한 자료를 다시 비교한 그림입니다. 보류해 둔 독립 자료는 아직 평가하지 않았습니다.')
        picture(fig,counts/'training_radial_counts.png'); pdf.savefig(fig); plt.close(fig)

        detail=fp['training_distance_readout']
        fig=page('5. CF4 거리 자료도 재현하나?',
            f"{detail['rows']:,}개 거리 관측과 예측을 직접 비교합니다. 현재 상관계수 {detail['correlation']:.3f}, "
            f"공유 보정값 {detail['zero_dex']:+.5f} dex입니다.\n"
            '왼쪽 점이 대각선 부근일수록 관측 평균과 예측이 가깝습니다. 관측 오차가 커서 상관 하나로 판정할 수 없습니다.\n'
            '오른쪽은 오차로 나눈 잔차입니다. 이 분포를 곧바로 정규분포 검정이나 독립 검증으로 해석하지 않습니다.')
        picture(fig,distance/'training_distance_prediction.png'); pdf.savefig(fig); plt.close(fig)

        fig=page('6. 이번 적합 전에는 어떤 문제가 있었나?',
            '이 그림은 이번 1,414개 공동 적합 전의 저장 장 비교입니다. 새 적합의 최종 결과와 혼동하지 마세요.\n'
            '당시 거리 점수는 균일장 기준보다 약 63.78 낮았습니다. 같은 보정 사전분포를 적용해도 부족함이 남았습니다.\n'
            '그래서 거리 자료를 확장해 장 자체를 다시 적합했습니다. 균일장이 실제 우주라는 결론이나 Bayes 증거는 아닙니다.')
        picture(fig,prelude/'zero_response.png'); pdf.savefig(fig); plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m',
        '-r90',f'-sOutputFile={out}/page_%02d.png',str(out/'R2_진행보고_예제그림.pdf')],
        check=True,timeout=90)
    manifest=dict(job_id=os.environ['SLURM_JOB_ID'],pages=6,R2_complete=False,
        numerical_work='saved readouts only; no fit, PM, heldout or new likelihood',
        source_directories=[str(p) for p in (fit,counts,distance,prelude)],
        report=str(out/'R2_진행보고_예제그림.pdf'))
    (out/'result.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
