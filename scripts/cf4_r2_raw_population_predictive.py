"""Actual training-mark distribution check; no field evolution or re-fitting."""
import json
import os
from pathlib import Path
import resource
import subprocess

import numpy as np
from scipy.special import gammainc, gammaincinv

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    with np.load(BASE/'r2_raw_source_mixture_v1/raw_source_mixture.npz',allow_pickle=False) as f:
        d={k:f[k].copy() for k in f.files}
    with np.load(BASE/'r2_raw_population_fit_v1/final_fit.npz',allow_pickle=False) as f:
        p=f['parameters'];richness_center=float(f['richness_center'])
        np.testing.assert_array_equal(d['PGC'],f['PGC'])
    b=p[:9].reshape(3,3);c=p[9:]
    ch=np.array([[np.exp(c[0]),0.,0.],[c[1],np.exp(c[2]),0.],[c[3],c[4],np.exp(c[5])]])
    cov=ch@ch.T
    with np.load(BASE/'r2_raw_selected_component_v1/training_cut_geometry.npz',allow_pickle=False) as f:
        index={int(v):i for i,v in enumerate(f['PGC'])}; ix=np.array([index[int(v)] for v in d['PGC']])
        x=f['x'][ix];error=f['optical_error_covariance'][ix]
        lower,upper=f['cut_lower'][ix],f['cut_upper'][ix]
    with np.load(BASE/'r2_raw_fp_inputs_v1/training_photometry.npz',allow_pickle=False) as f:
        index={int(v):i for i,v in enumerate(f['selected_PGC'])};ix=np.array([index[int(v)] for v in d['PGC']])
        richness=np.log1p(f['selected_photometry'][ix,list(f['columns']).index('NgroupT17')])-richness_center
    n=len(x); draws=1000;seed=2026092906;rng=np.random.default_rng(seed)
    observations=np.column_stack((x,d['observed_ksmag']))
    pit=np.empty_like(observations);predicted=np.empty_like(observations);scatter=np.empty_like(observations)
    selected_fraction=np.empty(n); accepted_generated=0; proposals=0
    matrix=np.array([[2.,0.,1.],[.04,1.,0.]])
    shape=float(d['lf_alpha'])+1; mstar=float(d['mstar'])
    lum_hi=10**(.4*(mstar-d['magnitude_lower']))
    lum_lo=10**(.4*(mstar-d['magnitude_upper']))
    u_lo,u_hi=gammainc(shape,lum_lo),gammainc(shape,lum_hi)
    # Same LF as count transfer; K is its catalogue proxy, not true noiseless luminosity.
    examples={}; example_rows=np.linspace(0,n-1,3,dtype=int)
    for i in range(n):
        start,stop=d['row_ptr'][i:i+2];ids=np.arange(start,stop)
        weight=np.exp(d['log_weight'][start:stop]);weight/=weight.sum()
        chol=np.linalg.cholesky(cov+error[i]); pieces=[];kept=0;attempts=0;passes=0
        while kept<draws:
            size=max(128,2*(draws-kept))
            proposals+=size;attempts+=size
            if proposals>20_000_000 or attempts>100_000:
                raise RuntimeError('selected-mark rejection draw budget exceeded; no row deletion')
            chosen=rng.choice(ids,size=size,p=weight)
            if np.any(u_hi[chosen]<=u_lo[chosen]):
                raise FloatingPointError('LF interval lost CDF precision')
            u=u_lo[chosen]+rng.random(size)*(u_hi[chosen]-u_lo[chosen])
            magnitude=mstar-np.log10(gammaincinv(shape,u))/.4
            mean=b[0]+richness[i]*b[2]+(magnitude[:,None]+23.)*b[1]
            mean[:,0]+=d['eta'][chosen]
            optical=mean+rng.normal(size=(size,3))@chol.T
            projected=optical@matrix.T
            okay=((projected>=lower[i])&(projected<=upper[i])).all(axis=1)
            passes+=int(okay.sum())
            kval=magnitude+d['observed_ksmag'][i]-d['observed_M'][chosen]
            sample=np.column_stack((optical,kval))[okay]
            pieces.append(sample[:draws-kept]);kept+=min(draws-kept,len(sample))
        sample=np.concatenate(pieces)
        accepted_generated+=len(sample);selected_fraction[i]=passes/attempts
        pit[i]=(np.sum(sample<observations[i],axis=0)+.5)/(draws+1.)
        predicted[i]=sample.mean(axis=0);scatter[i]=sample.std(axis=0)
        if i in example_rows: examples[f'row_{i}']=sample
    np.savez_compressed(out/'training_mark_prediction.npz',PGC=d['PGC'],observations=observations,
        predicted_mean=predicted,predicted_std=scatter,pit=pit,
        selected_fraction=selected_fraction,example_rows=example_rows,**examples)
    result=dict(status='TRAINING_RAW_MARK_MODEL_CHECK_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        seed=seed,rows=n,accepted_draws=accepted_generated,proposals=proposals,
        variables=['r_z','s','i','K_catalogue'],mean_rank=pit.mean(axis=0).tolist(),
        fraction_rank_below_0p1=(pit<.1).mean(axis=0).tolist(),
        fraction_rank_above_0p9=(pit>.9).mean(axis=0).tolist(),
        selection_fraction_quantiles=np.quantile(selected_fraction,[0,.5,1]).tolist(),
        population_parameters=p.tolist(),intrinsic_covariance=cov.tolist(),
        R2_complete=False,heldout_scored=False,PM_evolutions=0,parameter_fits=0,
        host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        limits=['training data and a fixed fitted field, not independent validation or field UQ',
            'finite Monte Carlo ranks; fitted-parameter effects, no uniformity p-value',
            'only specified Gaussian population, LF, known cuts and constant incidence',
            'a mismatch is model evidence, not proof which physical assumption is wrong',
            'MW/M31 ambiguous; M33 unresolved'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3})
    labels=['관측 log 크기','log 속도분산','log 표면밝기','K 겉보기 등급']
    pdfpath=out/'R2_실제관측분포_예제보고.pdf'
    with PdfPages(pdfpath) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(11.69,8.27));fig.subplots_adjust(top=.83,bottom=.22,hspace=.38)
        for j,ax in enumerate(axes.ravel()):
            ax.hist(pit[:,j],bins=np.linspace(0,1,11),density=True,histtype='step')
            ax.axhline(1.,color='grey',ls='--',label='이상적인 균등 분포 (유의성 기준 아님)')
            ax.set(title=labels[j],xlabel='모형 예측 중 실제값보다 작은 비율',ylabel='행별 비율 분포')
        fig.suptitle('R2 진행 중 — 실제 훈련 관측과 원자료 모형 비교',fontsize=17)
        fig.text(.07,.135,'각 은하에서 모형 관측을 1000개씩 생성했습니다. 실제 우주나 중력 시뮬레이션을 새로 만든 것이 아닙니다.\n'
            '좌우 끝에 몰리면 실제값이 모형 분포의 끝에 자주 놓입니다. 같은 훈련자료로 맞춘 모형이라 독립 검증은 아닙니다.\n'
            '유의확률·현재장 posterior 오차막대·R2 완료를 뜻하지 않습니다. MW/M31 미식별, M33 미해결.',
            va='top',fontsize=10,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(11.69,8.27));fig.subplots_adjust(top=.83,bottom=.27,wspace=.35)
        for ax,i in zip(axes,example_rows):
            sample=examples[f'row_{i}']
            ax.scatter(sample[:,3],sample[:,0],s=3,alpha=.2,label='모형 관측')
            ax.plot(observations[i,3],observations[i,0],'r*',markersize=13,label='실제 관측')
            ax.set(title=f'PGC {int(d["PGC"][i])}',xlabel='K 겉보기 등급',ylabel='관측 log 크기')
            ax.legend(fontsize=8)
        fig.suptitle('실제 세 은하의 예: 밝기와 크기를 함께 예측',fontsize=17)
        fig.text(.07,.18,'자료 순서의 처음·중간·끝 행을 미리 정해 보여줍니다. 잘 맞는 은하를 골라내지 않았습니다.\n'
            '푸른 점은 같은 현재장의 후보 거리·광도와 공통 관측모형에서 생성했습니다. 빨간 별은 실제 측정입니다.\n'
            '별을 중심으로 인위적으로 오차를 늘리거나 미세 밀도 위상을 추가하지 않았습니다.',
            va='top',fontsize=10,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdfpath)],check=True,timeout=90)
    result['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
