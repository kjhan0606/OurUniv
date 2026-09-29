"""Illustrated actual N256 pilot and explicitly labelled small mechanics examples."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from cf4_r2_posterior_moments import PresentMomentAccumulator
from cf4_r2_affine_force import AffineCorrectedForce

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    read=lambda name:json.loads((BASE/name/'result.json').read_text())
    pilot=read('r2_n256_joint_pilot_v1');dyn=read('r2_n256_dynamics_profile_v1')
    resource=read('r2_n256_source_profile_v3')
    if pilot['status']!='N256_RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR':raise ValueError('terminal N256 pilot required')
    slices=[]
    for name,file,n in [('r2_raw_joint_pilot_v1','accepted_present_state.npz',128),
                        ('r2_n256_dynamics_profile_v1','initial_present_state.npz',256)]:
        with np.load(BASE/name/file,allow_pickle=False) as f:slices.append(f['rho'][:,:,n//2].T.copy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3,'font.size':9})
    pdf=out/'R2_N256_현재장과표본이동_예제보고.pdf'
    def page(title,note):
        fig,a=plt.subplots(2,2,figsize=(11.69,8.27))
        fig.subplots_adjust(top=.85,bottom=.29,hspace=.55,wspace=.38)
        fig.suptitle(title,fontsize=16)
        fig.text(.05,.19,note+'\nR1 → R2 진행 중 → R3 같은 장의 MW/M31/M33 → R4 정밀 진화 → R5 줌 IC',
            fontsize=10,linespacing=1.6,va='top')
        return fig,a.ravel()
    with PdfPages(pdf) as pages:
        fig,a=page('실제 격자 연결 — 3에서 1.5 cMpc/h로',
            '두 지도는 같은 큰 규모 초기 모드에서 출발했습니다. 오른쪽의 새 작은 규모 모드는 사전분포로 추가했습니다.\n'
            'N256 지도는 추론의 초기값입니다. 작은 무늬가 많다는 것만으로 CF4가 이를 재현했다는 뜻은 아닙니다.\n'
            '오른쪽 아래는 서로 다른 범위의 자원 측정입니다. 중력 시간만으로 전체 추론 시간을 예측할 수 없습니다.')
        for axis,s,n,title in zip(a[:2],slices,(128,256),('N128 수락 상태: 격자 3','N256 초기 추정장: 격자 1.5')):
            dx=384/n
            im=axis.imshow(np.log10(np.maximum(s,1e-6)),origin='lower',cmap='RdBu_r',vmin=-1.5,vmax=1.5,
                extent=(-192-dx/2,192-dx/2,-192-dx/2,192-dx/2))
            axis.set(title=title,xlabel='x (cMpc/h)',ylabel='y (cMpc/h)');fig.colorbar(im,ax=axis,label='log10(밀도/평균)')
        parent=read('r2_raw_joint_pilot_v1')['final_white_mean_square']
        a[2].bar(['기존 큰 규모','새 장의 같은 대역','새 장 전체'],
            [parent,pilot['initial_white']['inherited_low_white_mean_square'],dyn['white_mean_square']])
        a[2].set(title='전체 분산만 보면 초기상태 의존성이 가려짐',ylabel='IC 백색변수 평균제곱')
        a[3].bar(['중력+시험 미분','전체 은하 수항 미분'],
            [dyn['forward_adjoint_seconds'],resource['full_gradient_seconds']])
        a[3].set(title='실측 계산 비용 (측정 범위는 다름)',ylabel='초')
        pages.savefig(fig);plt.close(fig)

        trace=pilot['trace'];x=np.arange(1,len(trace)+1)
        fig,a=page('표본 이동 검사 — 미분이 맞아도 제안은 기각될 수 있음',
            '정밀 목표 확률로 수락·기각합니다. 기각된 제안은 현재장을 바꾸지 않습니다.\n'
            '경로 적분 오차와 목표 보정의 합이 최종 에너지 오차입니다. 이 시험은 수렴한 posterior가 아닙니다.\n'
            '관측 격자는 3 cMpc/h 그대로입니다. 1.5 cMpc/h 격자 전체가 관측으로 제약됐다는 주장은 하지 않습니다.')
        direction=pilot['joint_PM_direction']
        a[0].bar(['자동 미분','직접 변화'],[direction['likelihood_AD'],direction['likelihood_FD']])
        a[0].set(title=f"전체 PM+관측 미분 비교: 상대차 {direction['relative_error']:.2g}",ylabel='같은 방향의 변화율')
        # This fixed four-proposal pilot has five startup evaluations, then
        # two coarse-force calls and one fine endpoint per completed proposal.
        initial_coarse=pilot['evaluations'][0]['energy'];initial_fine=pilot['initial_fine_energy']
        before_difference=initial_fine-initial_coarse;errors=[];changes=[]
        for i,row in enumerate(trace):
            evaluations=pilot['evaluations'][5+3*i:8+3*i]
            if len(evaluations)!=3 or evaluations[-1]['order']!=2:raise ValueError('unexpected pilot evaluation layout')
            proposed_difference=evaluations[-1]['energy']-evaluations[-2]['energy']
            change=proposed_difference-before_difference
            errors.append(row['energy_error']-change);changes.append(change)
            if row['accepted']:before_difference=proposed_difference
        a[1].bar(x-.22,errors,width=.22,label='거친 경로 적분');a[1].bar(x,changes,width=.22,label='정밀 목표 보정')
        a[1].bar(x+.22,[r['energy_error'] for r in trace],width=.22,label='합계')
        a[1].set(title='실제 제안별 에너지 오차 분해',xlabel='제안');a[1].legend(fontsize=8)
        a[2].bar(x,[int(r['accepted']) for r in trace]);a[2].set(title='수락(1) / 기각(0)',xlabel='제안',ylim=(0,1.2))
        a[3].plot(x,[r['inherited_low_white_mean_square'] for r in trace],'o-')
        a[3].set(title='물려받은 큰 규모 대역의 변화',xlabel='제안',ylabel='평균제곱')
        pages.savefig(fig);plt.close(fig)

        def toy(r,v,p):
            return dict(rho=np.full((1,1,1),float(r)),mean_velocity_km_s=np.full((3,1,1,1),float(v)),
                physical_velocity_variance_km2_s2=np.full((3,1,1,1),float(p)),velocity_valid=np.full((1,1,1),r>0))
        acc=PresentMomentAccumulator((1,1,1))
        for f in (toy(1,10,4),toy(1,10,4),toy(4,40,9)):acc.update(f)
        moments=acc.arrays();empty=PresentMomentAccumulator((1,1,1));empty.update(toy(0,0,0));empty.update(toy(1,30,9))
        fig,a=page('후속 구현의 작은 수학 예제 — 실제 은하 지도가 아닙니다',
            '기각으로 같은 상태가 반복되면 그 반복도 평균에 포함합니다. 빈 셀은 속도 0의 관측이 아닙니다.\n'
            '입자 운동의 분산과 장의 표본간 분산은 다릅니다. 표본 평균의 오차(MC 오차)는 별도 측정이 필요합니다.\n'
            '고정 힘 보정은 간단한 수학 예제만 통과했습니다. 실제 비선형 CF4 계산에서의 효율은 아직 미확인입니다.')
        a[0].plot([1,2,3],[1,1,4],'o-',label='상태 A, A, B')
        a[0].axhline(float(moments['rho_mean'][0,0,0]),label='반복 포함 평균 2',color='tab:green')
        a[0].axhline(2.5,label='반복 제외: 잘못된 평균 2.5',color='tab:red',ls='--')
        a[0].set(title='기각된 반복도 표본',xlabel='순서');a[0].legend(fontsize=8)
        a[1].bar(['빈 셀을 0으로 처리','비어 있지 않을 때 평균'],
            [15.,float(empty.arrays()['conditional_mean_velocity_km_s'][0,0,0,0])])
        a[1].set(title='빈 셀 + 속도 30인 셀의 예',ylabel='속도')
        a[2].bar(['물리적 속도 분산','장 표본간 속도 분산'],[
            float(moments['conditional_mean_physical_velocity_variance_km2_s2'][0,0,0,0]),
            float(moments['conditional_velocity_posterior_variance_km2_s2'][0,0,0,0])])
        a[2].set(title='서로 다른 두 분산',ylabel='속도 제곱 단위')
        coarse=lambda q:(.5*float(q@q)+float(q.sum()),q+1.)
        corrected=AffineCorrectedForce(coarse,np.array([.5]),np.array([-3.]))
        z=np.linspace(-2,2,41)
        a[3].plot(z,z+1,label='거친 포텐셜 기울기');a[3].plot(z,z-2,label='정밀 포텐셜 기울기')
        a[3].plot(z,[corrected(np.array([v]))[1][0] for v in z],'--',label='고정 보정 후')
        a[3].set(title='같은 곡률의 두 포텐셜: 보정 예제',xlabel='변수');a[3].legend(fontsize=8)
        pages.savefig(fig);plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf)],check=True,timeout=90)
    (out/'result.json').write_text(json.dumps(dict(status='N256_ILLUSTRATED_PILOT_NOT_POSTERIOR',
        job_id=os.environ['SLURM_JOB_ID'],pdf=str(pdf),pages=3,R2_complete=False),indent=2)+'\n')


if __name__=='__main__':main()
