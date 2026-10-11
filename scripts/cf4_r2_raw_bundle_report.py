"""Illustrated Korean report from terminal numerical evidence, no new inference."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    read=lambda name:json.loads((BASE/name/'result.json').read_text())
    dense=read('r2_raw_live_check_v1');sparse=read('r2_raw_live_packed_v1')
    pcs=read('r2_count_cell_average_v1');los=read('r2_volume_count_los_v1')
    cohort=read('r2_raw_live_cohort_v1')
    if cohort['status']!='WHOLE_COHORT_LIVE_RAW_MARK_CHECKED_NOT_POSTERIOR':
        raise ValueError('whole-cohort terminal success required for this report')
    with np.load(BASE/'r2_raw_live_cohort_v1/whole_cohort_readout.npz',allow_pickle=False) as f:
        pgc=f['PGC'].copy();values=f['raw_logpdf'].copy()
    with np.load(BASE/'r2_raw_population_fit_v1/final_fit.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(pgc,f['PGC']);old=f['final_logpdf'].copy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3})
    pdf=out/'R2_전체자료연결_예제보고.pdf'
    with PdfPages(pdf) as pages:
        d,s=dense['rows'],sparse['rows'];x=np.arange(len(d))
        fig,ax=plt.subplots(2,2,figsize=(11.69,8.27));fig.subplots_adjust(top=.86,bottom=.30,hspace=.55,wspace=.35)
        labels=[str(r['PGC']) for r in d]
        ax[0,0].plot(x,[r['raw_logpdf'] for r in d],'o',label='전체 성분');ax[0,0].plot(x,[r['raw_logpdf'] for r in s],'x',label='희소 성분')
        ax[0,0].set(title='실제 7개 은하: 같은 관측 점수');ax[0,0].legend()
        ax[0,1].bar(x,[r['dense_gradient_max_abs_difference'] for r in s]);ax[0,1].set(title='26개 기울기의 최대 차이',ylabel='절대 차이')
        ax[1,0].bar(x-.18,[r['first_value_gradient_seconds'] for r in d],.36,label='전체 성분')
        ax[1,0].bar(x+.18,[r['first_value_gradient_seconds'] for r in s],.36,label='희소 성분')
        ax[1,0].set(title='핵심 기울기 계산 시간',ylabel='초');ax[1,0].legend()
        ax[1,1].bar(x,[r['packing_seconds'] for r in s]);ax[1,1].set(title='별도로 남는 목록 생성·컴파일 비용',ylabel='초')
        for a in ax.flat:a.set_xticks(x,labels,rotation=35,fontsize=7);a.set_xlabel('실제 관측 은하의 PGC 번호')
        fig.suptitle('검사 1 — 관측모형을 바꾸지 않고 계산량 줄이기',fontsize=17)
        fig.text(.06,.12,'밝기·크기·속도분산·속도 정보를 같은 밀도장에 연결합니다. 기여하지 않는 성분만 제외합니다.\n'
            '핵심 계산의 가속과 전체 실행시간의 가속은 다릅니다. 위 비용은 소규모 시제품의 측정값입니다.',fontsize=11,linespacing=1.7,va='top')
        pages.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(2,2,figsize=(11.69,8.27));fig.subplots_adjust(top=.86,bottom=.30,hspace=.55,wspace=.3)
        x=np.arange(6);labels=[str(r['PGC']) for r in pcs['rows']]
        ax[0,0].bar(x,[100*r['count_relative_error'] for r in pcs['rows']],color='tomato')
        ax[0,0].set(title='채택하지 않은 단순 격자 평균 근사',ylabel='정밀 적분 대비 예상 개수 차이 (%)')
        ax[0,1].bar(x,[100*r['count_relative_error'] for r in los['rows']]);ax[0,1].set(title='같은 모형: 시선 적분점만 줄임',ylabel='정밀 적분 대비 차이 (%)')
        for a,r,title in ((ax[1,0],pcs['rows'],'단순 근사의 속도 기울기'),(ax[1,1],los['rows'],'시선 적분점 변경의 속도 기울기')):
            a.plot(x,[v['reference_derivative'] for v in r],'o-',label='정밀 기준')
            a.plot(x,[v['velocity_derivative'] for v in r],'x--',label='비교 방식');a.set_title(title);a.legend(fontsize=8)
        for a in ax.flat:a.set_xticks(x,labels,rotation=35,fontsize=7);a.set_xlabel('실제 관측 은하의 PGC 번호')
        fig.suptitle('검사 2 — 더 빠른 은하 개수 예측이 정확한가?',fontsize=17)
        fig.text(.06,.12,'왼쪽 근사는 최대 약 6% 오차로 채택하지 않았습니다. 오른쪽은 모형을 유지한 수치 적분 비교입니다.\n'
            '6개 실제 위치의 결과이지, 전체 밀도장·posterior의 오차가 이 값 이하라는 보장은 아닙니다.',fontsize=11,linespacing=1.7,va='top')
        pages.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(2,2,figsize=(11.69,8.27));fig.subplots_adjust(top=.86,bottom=.34,hspace=.55,wspace=.3)
        change=values-old
        ax[0,0].scatter(np.arange(len(values)),change,s=3);ax[0,0].set(title='1,414개 훈련 은하의 점수 변화',xlabel='고정된 자료 순서',ylabel='부피 적분 − 중심점 계산')
        ax[0,1].hist(change,bins=40);ax[0,1].set(title='그 변화의 분포',xlabel='관측 로그밀도 차이',ylabel='은하 수')
        examples=np.array([0,len(values)//2,len(values)-1]);x=np.arange(3)
        ax[1,0].bar(x-.18,old[examples],.36,label='기존 중심점');ax[1,0].bar(x+.18,values[examples],.36,label='부피 적분')
        ax[1,0].set_xticks(x,[f'PGC {pgc[i]}' for i in examples],fontsize=8);ax[1,0].set_title('사전에 정한 첫·가운데·마지막 은하');ax[1,0].legend(fontsize=8)
        ax[1,1].bar([0,1],[cohort['joint_direction_analytic'],cohort['joint_direction_finite_difference']])
        ax[1,1].set_xticks([0,1],['미분 계산','변수를 조금 바꿔 직접 계산'])
        ax[1,1].set_title('밀도·속도·공통 보정변수를 함께 바꿀 때')
        fig.suptitle('검사 3 — 실제 훈련 자료 전체를 하나의 관측모형에 연결',fontsize=17)
        fig.text(.06,.20,'같은 현재장·같은 보정변수를 평가한 결과입니다. 점수 상승을 재현 품질 향상으로 해석하지 않습니다.\n'
            '새 중력 진화·최적화·posterior 표본추출·검증자료 평가는 하지 않았습니다.\n'
            'R1 → R2 진행 중 → R3 동일 장의 MW/M31/M33 → R4 정밀 진화 → R5 줌 IC\n'
            '현재 격자는 3 cMpc/h 개발용. MW/M31 역할은 모호하고 M33는 미해결입니다.',fontsize=10,linespacing=1.7,va='top')
        pages.savefig(fig);plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf)],check=True,timeout=90)
    (out/'result.json').write_text(json.dumps(dict(status='ILLUSTRATED_READOUT_NOT_INFERENCE',
        job_id=os.environ['SLURM_JOB_ID'],pages=3,R2_complete=False,pdf=str(pdf)),indent=2)+'\n')


if __name__=='__main__':main()
