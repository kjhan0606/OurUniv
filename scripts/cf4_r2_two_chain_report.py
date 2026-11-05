"""One terminal two-chain diagnostic/report job, never a new fit or heldout test."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from cf4_r2_chain_diagnostics import scalar_diagnostics,retained_scalar_matrix

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    directories=[BASE/'r2_n256_chain_a_v1',BASE/'r2_n256_chain_b_v1']
    reports=[];matrices=[]
    for directory in directories:
        path=directory/'result.json'
        report=json.loads(path.read_text()) if path.is_file() else {'status':'NO_RESULT','trace':[]}
        labels,values=retained_scalar_matrix(report)
        reports.append(report);matrices.append(values)
    n=min(len(x) for x in matrices)
    diagnostics={}
    if n>=8:
        for i,label in enumerate(labels):
            # Equal-length suffixes are declared; all original traces remain intact.
            diagnostics[label]=scalar_diagnostics(np.stack([x[-n:,i] for x in matrices]))
    summary=dict(status='R2_CHAIN_DIAGNOSTICS_NOT_FINAL_POSTERIOR',R2_complete=False,
        heldout_scored=False,job_id=os.environ['SLURM_JOB_ID'],
        chain_statuses=[r['status'] for r in reports],
        retained_per_chain=[len(x) for x in matrices],equal_length_suffix=n,
        diagnostics=diagnostics,
        remaining=['stationarity/MC error assessment','GL4 numerical sensitivity',
            'weak common-scale prior sensitivity','information-support maps','one untouched v6 predictive check'],
        caveat='No automatic promotion; short/shared low-mode ancestry limits global exploration. MW/M31 ambiguous, M33 unresolved.')
    (out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3,'font.size':9})
    pdf=out/'R2_두체인_현재장과수렴진단_예제.pdf'
    def page(title,note):
        fig,axes=plt.subplots(2,2,figsize=(11.69,8.27))
        fig.subplots_adjust(top=.85,bottom=.28,hspace=.5,wspace=.4)
        fig.suptitle(title,fontsize=16)
        fig.text(.05,.18,note+'\nR1 → R2 진행 중 → R3 같은 장의 MW/M31/M33 → R4 정밀 진화 → R5 줌 IC',
            fontsize=10,linespacing=1.6,va='top')
        return fig,axes.ravel()
    with PdfPages(pdf) as pages:
        fig,a=page('실제 두 체인 — 같은 초기 상태에 머물러 있지 않은가?',
            '점선 왼쪽은 버리는 초기 적응 구간입니다. 채택되지 않은 상태도 반복 표본으로 포함했습니다.\n'
            '서로 다른 작은 규모 위상으로 시작했지만 큰 규모 초기 상태의 계보는 공유합니다.\n'
            '에너지는 최적화 손실이 아닙니다. 두 선의 겹침이나 채택률 하나만으로 수렴을 선언하지 않습니다.')
        for report,name in zip(reports,('A','B')):
            rows=report['trace'];x=[r['iteration'] for r in rows]
            a[0].plot(x,[r['inherited_low_white_mean_square'] for r in rows],label=name)
            a[1].plot(x,[r['fine_energy'] for r in rows],label=name)
            a[2].plot(x,[r['nuisance_white'][6] for r in rows],label=name)
            a[3].plot(x,np.cumsum([r['accepted'] for r in rows]),label=name)
        for axis,title in zip(a,('큰 규모 IC 대역의 변화','정밀 목표 에너지','LOS 폭 보조변수','누적 채택 횟수')):
            axis.axvline(12.5,color='grey',ls='--');axis.set(title=title,xlabel='이동 횟수');axis.legend()
        pages.savefig(fig);plt.close(fig)

        fig,a=page('실제 현재 밀도장 표본 평균·표본간 차이 — 아직 미판정',
            '한가운데 z 단면입니다. 위는 표본 평균, 아래는 표본간 표준편차이며 평균의 MC 오차가 아닙니다.\n'
            '두 체인이 아직 충분히 섞이지 않았다면 아래 지도는 신뢰 가능한 posterior 불확실성이 아닙니다.\n'
            '격자 간격 1.5 cMpc/h와 실제 관측 정보의 분해능은 다릅니다. LG 최종 분해능은 후속 단계입니다.')
        for col,(directory,name) in enumerate(zip(directories,('A','B'))):
            file=directory/'present_moments_unassessed.npz'
            if not file.is_file():
                for axis in (a[col],a[2+col]):axis.text(.1,.5,f'체인 {name}: 누적 지도 없음');axis.set_axis_off()
                continue
            with np.load(file,allow_pickle=False) as f:
                mean=f['rho_mean'][:,:,128].T.copy()
                variance=f['rho_posterior_variance'][:,:,128].T.copy()
            fields=(np.log10(np.maximum(mean,1e-6)),np.sqrt(np.maximum(variance,0)))
            for axis,field,title,limit in zip((a[col],a[2+col]),fields,
                    (f'{name}: log10(평균 밀도/우주 평균)',f'{name}: 밀도 표본간 표준편차'),(1.5,1.)):
                im=axis.imshow(field,origin='lower',extent=(-192.75,191.25,-192.75,191.25),
                    cmap='RdBu_r' if axis is a[col] else 'magma',
                    vmin=-1.5 if axis is a[col] else 0,vmax=limit)
                axis.set(title=title,xlabel='x (cMpc/h)',ylabel='y (cMpc/h)');fig.colorbar(im,ax=axis)
        pages.savefig(fig);plt.close(fig)

        finite=[(k,v) for k,v in diagnostics.items() if v['rank_folded_split_rhat'] is not None]
        worst=sorted(finite,key=lambda kv:kv[1]['rank_folded_split_rhat'],reverse=True)[:8]
        fig,a=page('실제 진단 예제 — 지도와 함께 수치의 한계를 확인',
            f'보존 표본 수 A/B: {len(matrices[0])}/{len(matrices[1])}. 공통 길이의 끝 {n}개를 비교했습니다.\n'
            'Rhat은 체인 간 일치도 진단입니다. 유효 표본 수와 MC 오차는 정상상태를 가정한 거친 배치평균 추정입니다.\n'
            '수치 적분·사전분포 민감도, 정보 지도, 미사용 자료 예측 검증은 이 보고에서 수행하지 않았습니다.')
        if worst:
            a[0].barh(range(len(worst)),[v['rank_folded_split_rhat'] for _,v in worst])
            a[0].set_yticks(range(len(worst)),[k for k,_ in worst],fontsize=6)
            a[0].axvline(1,color='grey');a[0].set(title='불일치가 큰 변수 예제',xlabel='rank/folded split Rhat')
            key=worst[0][0];idx=labels.index(key)
            for values,name in zip(matrices,('A','B')):a[1].plot(values[:,idx],label=name)
            a[1].set(title=f'실제 추적: {key}',xlabel='보존 순서');a[1].legend()
            ess=[v['batch_means_ess'] for _,v in finite if v['batch_means_ess'] is not None]
            a[2].hist(ess,bins=10);a[2].set(title='변수별 거친 유효 표본 수',xlabel='ESS',ylabel='변수 수')
            for report,name in zip(reports,('A','B')):
                a[3].plot([r['energy_error'] for r in report['trace']],label=name)
            a[3].axhline(0,color='grey');a[3].set(title='정밀 Hamiltonian 오차',xlabel='이동 순서');a[3].legend()
        else:
            for axis in a:axis.text(.05,.5,'표본 부족 또는 정지한 체인: 수치 판정 불가');axis.set_axis_off()
        pages.savefig(fig);plt.close(fig)

        rng=np.random.default_rng(91);iid=rng.normal(size=(2,10000));shift=iid.copy();shift[1]+=3
        corr=np.zeros_like(iid)
        for i in range(1,corr.shape[1]):corr[:,i]=.95*corr[:,i-1]+iid[:,i]
        fig,a=page('진단 코드의 작은 예제 — 실제 CF4 자료가 아닙니다',
            '같은 분포의 독립 표본과 서로 다른 평균의 두 체인을 구분하는 검사입니다.\n'
            '강한 시간 상관이 있으면 표본 개수보다 유효 표본 수가 작아집니다. 기각된 반복도 반드시 셉니다.\n'
            '이 예제 통과는 진단 구현의 확인이며, 위의 실제 CF4 체인이 수렴했다는 증거가 아닙니다.')
        for values,label in ((iid,'같은 분포'),(shift,'다른 평균')):
            a[0].hist(values[1],bins=40,density=True,histtype='step',label=label)
        a[0].set(title='두 번째 체인의 분포 예제');a[0].legend()
        a[1].bar(['같은 분포','평균이 다른 체인'],[
            scalar_diagnostics(iid)['rank_folded_split_rhat'],scalar_diagnostics(shift)['rank_folded_split_rhat']])
        a[1].set(title='같음/다름을 구분하는 Rhat 예제')
        a[2].plot(corr[0,:150],label='강한 상관');a[2].plot(iid[0,:150],alpha=.5,label='독립')
        a[2].set(title='상관된 표본의 실제 예제',xlabel='표본 순서');a[2].legend()
        a[3].plot([1,2,3],[1,1,4],'o-');a[3].axhline(2,label='반복 포함 평균',color='tab:green')
        a[3].axhline(2.5,label='반복 제외: 잘못된 평균',color='tab:red',ls='--')
        a[3].set(title='기각된 같은 상태를 두 번 세는 검사',xlabel='순서');a[3].legend()
        pages.savefig(fig);plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf)],check=True,timeout=90)
    summary.update(pdf=str(pdf),pages=4,pdf_pages_visually_reviewed=False)
    (out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
