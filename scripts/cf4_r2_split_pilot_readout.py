"""Read saved short-pilot diagnostics; never infer ESS or posterior error bars."""
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    source=Path(os.environ['CF4_R2_PILOT_DIR'])
    out=Path(os.environ['CF4_R2_OUT_DIR'])
    r=json.loads((source/'result.json').read_text())
    rows=r.get('sampler_trace',[])
    if not rows:
        raise ValueError('no real CF4 proposals; do not read unit-test stdout as science')
    out.mkdir(exist_ok=False)
    font=font_manager.FontProperties(fname='/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    font_manager.fontManager.addfont(font.get_file())
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'pdf.fonttype':3})
    x=np.array([p['iteration'] for p in rows])
    accepted=np.array([p['accepted'] for p in rows],bool)
    retained=np.array([not p['warmup'] for p in rows],bool)
    energy=np.array([float(p['energy_error']) for p in rows])
    step=np.array([p['step_size'] for p in rows])
    jump=np.array([p['canonical_jump_rms'] for p in rows])
    finite=np.isfinite(energy)
    warmup=int(r['sampler']['warmup'])
    pdfpath=out/'R2_표본추출시험_그림보고.pdf'
    with PdfPages(pdfpath) as pdf:
        for page in (0,1):
            fig,axes=plt.subplots(2,2,figsize=(11.69,8.27))
            fig.subplots_adjust(top=.84,bottom=.25,hspace=.4,wspace=.28)
            fig.suptitle('R2 진행 중: 표본추출 계산성 시험 — '+('수락과 수치 오차' if page==0 else '실제 상태 이동'),fontsize=17,y=.95)
            fig.text(.07,.89,'R1 → R2 진행 중 → R3 국부은하군 → R4 정밀 진화 → R5 줌 초기조건',fontsize=11)
            a,b,c,d=axes.ravel()
            if page==0:
                a.step(x,accepted.astype(int),where='mid'); a.set(title='제안 수락=1 / 거절=0',yticks=[0,1])
                b.plot(x[finite],energy[finite],'.-'); b.set(title=f'에너지 오차 (무한대 {int((~finite).sum())}회)')
                c.plot(x,step,'.-'); c.set(title='적분 한 걸음 크기',yscale='log')
                d.plot(x,jump,'.-'); d.set(title='수락 후 백색변수 이동 RMS',ylim=(0,None))
                caption=(f'전체 {len(rows)}회 중 {int(accepted.sum())}회 수락. 준비 구간 이후는 '
                    f'{int(retained.sum())}회 중 {int((accepted & retained).sum())}회 수락입니다.\n'
                    '거절하면 이전 상태를 유지합니다. 수락률만 높고 이동량이 작으면 탐색 효율이 낮을 수 있습니다.\n'
                    '독립 값 계산으로 제안 끝점의 전체 목표값을 대조했습니다. 짧은 시험은 posterior 수렴 검사가 아닙니다.')
            else:
                a.plot(x,[p['IC_mean_square'] for p in rows],'.-'); a.set(title='IC 백색변수 제곱 평균')
                b.plot(x,[p['white_nuisance'][-1]*.004 for p in rows],'.-'); b.set(title='공통 거리 보정값 (dex)')
                modes=np.array([p['fundamental_cosine'] for p in rows])
                c.plot(x,modes-modes[0]); c.set(title='기본 파장 성분 변화 (첫 기록 대비)')
                c.legend(['x 방향','y 방향','z 방향'],fontsize=8)
                d.plot(x,[p['canonical_nuisance_jump_l2'] for p in rows],'.-'); d.set(title='관측모형 변수의 수락 이동량')
                caption=('긴 파장과 관측모형 변수가 실제로 움직였는지 별도로 봅니다. 이 값은 물리적 속도분산이 아닙니다.\n'
                    'IC 제곱 평균은 초기화 이후의 이동 진단입니다. 이를 1로 강제 보정하거나 LCDM 검증으로 대체하지 않습니다.\n'
                    'MW·M31 미식별, M33 미해결. 독립 자료 예측·모형 보정·1.5 cMpc/h 지도는 아직 완료하지 않았습니다.')
            for ax in axes.ravel():
                if warmup:
                    ax.axvline(warmup+.5,color='grey',ls='--',lw=.8)
                ax.set_xlabel(f'제안 번호 (1–{warmup} 준비 구간)' if warmup else '제안 번호 (모두 고정 걸음)')
            fig.text(.07,.15,caption,va='top',fontsize=10,linespacing=1.8)
            fig.text(.07,.035,f"실제 계산 {r['job_id']} | 짧은 조건부 탐색 시험, posterior 오차막대·ESS를 제공하지 않음",fontsize=9)
            pdf.savefig(fig); fig.savefig(out/f'pilot_page_{page+1}.png',dpi=110); plt.close(fig)
    summary=dict(source=str(source),source_job=r['job_id'],source_status=r['status'],
        proposals=len(rows),accepted=int(accepted.sum()),retained=int(retained.sum()),
        retained_accepted=int((accepted & retained).sum()),nonfinite_energy_errors=int((~finite).sum()),
        final_step=float(step[-1]),IC_mean_square_initial=r.get('initial_IC_mean_square'),
        IC_mean_square_first_proposal=rows[0]['IC_mean_square'],
        IC_mean_square_end=rows[-1]['IC_mean_square'],R2_complete=False,posterior_claimed=False,
        report=str(pdfpath))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdfpath)],check=True,timeout=90)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
