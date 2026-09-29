"""Actual-example Korean report; reads saved evidence, never evolves a field."""
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
    native=read('r2_raw_field_profile_v1');count=read('r2_volume_count_profile_v1')
    pilot=read('r2_raw_joint_pilot_v1')
    if pilot['status']!='RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR':raise ValueError('terminal pilot required')
    with np.load(BASE/'r2_cut_benchmark_v1/readouts.npz',allow_pickle=False) as f:cut={k:f[k] for k in f.files}
    with np.load(BASE/'r2_raw_joint_pilot_v1/accepted_present_state.npz',allow_pickle=False) as f:
        rho=f['rho']
        dispersion=np.where(f['velocity_valid'],np.sqrt(f['physical_velocity_variance_km2_s2'].mean(axis=0)),np.nan)
    with np.load(BASE/'r2_prior_split_long_v1/final_state.npz',allow_pickle=False) as f:oldrho=f['rho']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3,'font.size':9})
    pdf=out/'R2_현재장연결과시험표본_실제예제.pdf'
    def page(title,note,shape=(2,2)):
        fig,axes=plt.subplots(*shape,figsize=(11.69,8.27))
        fig.subplots_adjust(top=.85,bottom=.28,hspace=.55,wspace=.38)
        fig.suptitle(title,fontsize=16)
        fig.text(.05,.17,note+'\nR1 → R2 진행 중 → R3 같은 장의 MW/M31/M33 → R4 정밀 진화 → R5 줌 IC',
            fontsize=10,linespacing=1.6,va='top')
        return fig,np.ravel(axes)
    with PdfPages(pdf) as pages:
        fig,a=page('검사 1 — 실제 은하의 선택확률 계산을 간소화해도 되는가?',
            '사전 지정 7개와 실제 선택 경계에 가장 가까운 4개 은하를 비교했습니다.\n'
            '같은 정밀도에서는 시간을 줄였지만, 기존 저차 방식보다 빠르다는 뜻은 아닙니다.')
        x=np.arange(len(cut['PGC']))
        a[0].plot(x,cut['reference_values'],'o',label='정밀 기준');a[0].plot(x,cut['fast_values'],'x',label='조건부 간소화')
        a[0].set(title='각 은하의 실제 관측 로그밀도',xlabel='PGC 번호');a[0].legend()
        a[0].set_xticks(x,[str(p) for p in cut['PGC']],rotation=55,fontsize=6)
        a[1].bar(x,cut['fast_values']-cut['reference_values']);a[1].set(title='같은 정밀 기준 대비 차이',xlabel='동일한 11개 은하 순서')
        a[2].plot(cut['fast_gradient']-cut['reference_gradient'],'o');a[2].set(title='26개 변수의 미분 차이',xlabel='변수 번호')
        a[3].bar(x,cut['legacy_values']-cut['reference_values']);a[3].set(title='기존 저차 적분의 수치적 차이',xlabel='동일한 11개 은하 순서')
        pages.savefig(fig);plt.close(fig)
        fig,a=page('검사 2 — 전체 은하 수와 전체 밀도·속도 격자를 연결',
            '전체 은하 수가 거의 같아도 위치별 확률의 곱은 달라집니다. 거친 계산을 정밀 posterior로 대체하지 않습니다.\n'
            '미분 검사는 변수를 조금 움직여 직접 구한 변화와 비교합니다. 재현 품질이나 불확실성 검증은 별도입니다.')
        rules=count['rules'];labels=['격자당 2³점','격자당 4³점']
        a[0].bar(labels,[r['score']-rules[0]['score'] for r in rules]);a[0].set(title='전체 로그확률 변화',ylabel='거친 기준 대비')
        a[1].bar(labels,[r['expected_training_count']-rules[0]['expected_training_count'] for r in rules]);a[1].set(title='예상 총 은하 수의 변화',ylabel='거친 기준 대비 개수')
        a[2].bar(labels,[r['full_gradient_seconds'] for r in rules]);a[2].set(title='모든 격자 변수의 개수항 미분',ylabel='초')
        a[3].bar(['자동 미분','직접 변화'],[native['native_direction_AD'],native['native_direction_FD']]);a[3].set(title='실제 밀도·속도·공통 변수를 함께 변경',ylabel='원시 관측항 변화율')
        pages.savefig(fig);plt.close(fig)
        fig,a=page('시험 표본추출 — 현재장 자체를 함께 움직였는가?',
            'N128, 격자 3 cMpc/h의 개발 시험입니다. 지도는 수락한 마지막 한 상태이지 posterior 평균이 아닙니다.\n'
            '물리적 속도분산은 같은 격자 안 입자 운동의 분산이며, posterior 불확실성이나 관측오차가 아닙니다. 빈 격자는 제외합니다.\n'
            '표본추출은 손실 최소화가 아닙니다. 수락·기각은 운동에너지까지 포함한 변화로 판정합니다.',(2,3))
        # Pixel centres represent native PM NODES (origin0), not voxel centres.
        dx=384/rho.shape[0];extent=(-192-dx/2,192-dx/2,-192-dx/2,192-dx/2)
        for axis,density,title in zip(a[:2],(oldrho,rho),('시험 시작의 밀도 단면','마지막 수락 상태의 밀도 단면')):
            im=axis.imshow(np.log10(np.maximum(density[:,:,64].T,1e-6)),origin='lower',extent=extent,vmin=-1.5,vmax=1.5,cmap='RdBu_r')
            axis.set(title=title,xlabel='관측자 상대 x (cMpc/h)',ylabel='y (cMpc/h)');fig.colorbar(im,ax=axis,label='log₁₀(밀도/평균)')
        im=a[2].imshow(dispersion[:,:,64].T,origin='lower',extent=extent,cmap='magma')
        a[2].set(title='마지막 상태의 물리적 속도분산',xlabel='x (cMpc/h)');fig.colorbar(im,ax=a[2],label='km/s')
        trace=pilot['trace'];x=[r['iteration'] for r in trace]
        a[3].plot(x,[r['fine_energy'] for r in trace],'o-');a[3].set(title='수락 상태의 확률값 추이',xlabel='제안 횟수',ylabel='음의 로그 posterior')
        a[4].bar(x,[int(r['accepted']) for r in trace]);a[4].set(title='실제 수락(1) / 기각(0)',xlabel='제안 횟수',ylim=(0,1.2))
        a[5].plot(x,[r['white_mean_square'] for r in trace],'o-');a[5].set(title='IC 백색변수 평균 제곱',xlabel='제안 횟수')
        for axis in a[3:]:axis.axvspan(.5,4.5,color='grey',alpha=.12)
        pages.savefig(fig);plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf)],check=True,timeout=90)
    (out/'result.json').write_text(json.dumps(dict(status='ILLUSTRATED_TRANSITION_NOT_POSTERIOR',
        job_id=os.environ['SLURM_JOB_ID'],pages=3,pdf=str(pdf),R2_complete=False),indent=2)+'\n')


if __name__=='__main__':main()
