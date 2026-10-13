"""Bounded15-parameter raw FP fit at one saved field, not a new posterior."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri_exp

from cf4_r2_raw_selected_fp import rule, log_schechter, selected_mark_logpdf
from cf4_r2_raw_selected_fp_jax import inverse_log_cdf, row_logpdf, unpack


BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']); out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],
        source_commit=os.environ['CF4_EXPECTED_COMMIT'],R2_complete=False,
        heldout_scored=False,PM_evolutions=0,field_fixed=True,parameters=15,
        independent_calibration_prior=False,trace=[])
    def save():
        report['seconds']=time.monotonic()-started
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        with np.load(BASE/'r2_raw_source_mixture_v1/raw_source_mixture.npz',allow_pickle=False) as f:
            mixture={k:f[k].copy() for k in f.files}
        with np.load(BASE/'r2_raw_selected_component_v1/training_cut_geometry.npz',allow_pickle=False) as f:
            order={int(p):i for i,p in enumerate(f['PGC'])}
            index=np.array([order[int(p)] for p in mixture['PGC']])
            x=f['x'][index]; errors=f['optical_error_covariance'][index]
            lower,upper=f['cut_lower'][index],f['cut_upper'][index]
        with np.load(BASE/'r2_raw_fp_inputs_v1/training_photometry.npz',allow_pickle=False) as f:
            order={int(p):i for i,p in enumerate(f['selected_PGC'])}
            index=np.array([order[int(p)] for p in mixture['PGC']])
            richness=np.log1p(f['selected_photometry'][index,list(f['columns']).index('NgroupT17')])
        richness_center=float(richness.mean()); richness-=richness_center
        n=len(x); ptr=mixture['row_ptr']; row=np.repeat(np.arange(n),np.diff(ptr)).astype(np.int32)
        if n!=1414 or not np.isfinite(richness).all():
            raise ValueError('training geometry changed')
        mt,mw=rule(24)
        ml,mh=mixture['magnitude_lower'],mixture['magnitude_upper']
        magnitude=ml[:,None]+(mh-ml)[:,None]*mt
        mstar,alpha=map(float,(mixture['mstar'],mixture['lf_alpha']))
        logq=log_schechter(magnitude,mstar=mstar,alpha=alpha)+np.log(mw)
        integral=np.log(mh-ml)+logsumexp(logq,axis=-1)
        logq-=logsumexp(logq,axis=-1)[:,None]
        inside=(mixture['observed_M']>=ml)&(mixture['observed_M']<=mh)
        logmd=np.where(inside,log_schechter(mixture['observed_M'],mstar=mstar,alpha=alpha)-integral,-np.inf)
        # Data-derived optimizer initialization only: no prior/covariance inference.
        lw=mixture['log_weight']+logmd
        logs=np.array([logsumexp(lw[ptr[i]:ptr[i+1]]) for i in range(n)])
        kw=np.exp(lw-logs[row])
        eta_bar=np.add.reduceat(kw*mixture['eta'],ptr[:-1])
        m_bar=np.add.reduceat(kw*mixture['observed_M'],ptr[:-1])
        design=np.column_stack((np.ones(n),m_bar+23.,richness))
        y=x.copy(); y[:,0]-=eta_bar
        coeff=np.linalg.lstsq(design,y,rcond=None)[0]
        residual=y-design@coeff
        covariance=np.cov(residual,rowvar=False)-errors.mean(axis=0)
        eigen,vec=np.linalg.eigh(covariance)
        covariance=(vec*np.maximum(eigen,1e-6))@vec.T
        chol=np.linalg.cholesky(covariance)
        origin=np.r_[coeff.ravel(),np.log(chol[0,0]),chol[1,0],np.log(chol[1,1]),
                      chol[2,0],chol[2,1],np.log(chol[2,2])]
        scales=np.array([.1,.05,.1,.05,.02,.05,.05,.02,.05,.3,.03,.3,.05,.03,.3])
        data={k:jnp.asarray(v) for k,v in dict(x=x,row=row,eta=mixture['eta'],
            error_covariance=errors,richness=richness,observed_M=mixture['observed_M'],
            magnitude=magnitude,logq=logq,log_weight=mixture['log_weight'],
            log_M_density=logmd,cut_lower=lower,cut_upper=upper).items()}
        cut_t,cut_w=map(jnp.asarray,rule(64))
        def objective(z):
            values=row_logpdf(jnp.asarray(origin)+jnp.asarray(scales)*z,data,cut_t,cut_w)
            return -jnp.mean(values),values
        derivative=jax.jit(jax.value_and_grad(objective,has_aux=True))
        compiled=derivative.lower(jnp.zeros(15)).compile()
        memory=compiled.memory_analysis()
        report['device_temporary_GiB']=memory.temp_size_in_bytes/1024**3
        if report['device_temporary_GiB']>60:
            raise RuntimeError('fixed-field fit exceeds bounded device-memory plan')
        initial=np.zeros(15); cache={}; calls=0
        def evaluate(z):
            nonlocal calls
            if time.monotonic()-started>600:
                raise TimeoutError('10-minute application bound; keep current fit only')
            (value,rows),gradient=compiled(jnp.asarray(z))
            value=float(value); gradient=np.asarray(gradient); rows=np.asarray(rows)
            if not np.isfinite(np.r_[value,gradient,rows]).all():
                raise FloatingPointError('nonfinite raw likelihood; no clipping or inflated error repair')
            calls+=1;cache.update(z=np.array(z),value=value,rows=rows,gradient=gradient)
            return value,gradient
        f0,g0=evaluate(initial); initial_rows=cache['rows'].copy()
        logps=np.array([-1e-7,-1.,-29.9,-30.1,-100.,-1000.])
        inverse_error=float(np.max(np.abs(np.asarray(inverse_log_cdf(jnp.asarray(logps)))-ndtri_exp(logps))))
        if inverse_error>1e-10:
            raise AssertionError('stable inverse-log-CDF failed independent SciPy comparison')
        def cpu_rows(params):
            b,cov=map(np.asarray,unpack(jnp.asarray(params)))
            values=[]
            for i in range(8):
                sl=slice(ptr[i],ptr[i+1])
                values.append(float(selected_mark_logpdf(x[i],mixture['observed_M'][sl],
                    mixture['log_weight'][sl],mixture['eta'][sl],ml[sl],mh[sl],
                    b[0]+richness[i]*b[2],b[1],cov+errors[i],
                    np.array([[2.,0.,1.],[.04,1.,0.]]),lower[i],upper[i],
                    mstar=mstar,alpha=alpha,magnitude_order=24,cut_order=64)))
            return np.array(values)
        discrepancy=float(np.max(np.abs(cpu_rows(origin)-initial_rows[:8])))
        if discrepancy>1e-7:
            raise AssertionError('raw likelihood CPU / gradient-primal mismatch')
        direction=np.random.default_rng(2026092905).normal(size=15);direction/=np.linalg.norm(direction)
        eps=2e-5
        fd=(evaluate(eps*direction)[0]-evaluate(-eps*direction)[0])/(2*eps)
        ad=float(g0@direction)
        grad_error=abs(fd-ad)/max(1.,abs(fd),abs(ad))
        if grad_error>2e-5:
            raise AssertionError('raw likelihood directional derivative mismatch')
        report.update(initial_objective=f0,initial_gradient_norm=float(np.linalg.norm(g0)),
            initial_CPU_max_abs_difference=discrepancy,inverse_log_CDF_max_abs_error=inverse_error,
            directional_derivative=dict(analytic=ad,finite_difference=fd,relative_error=grad_error),
            source_components=len(row),richness_center=richness_center)
        save();print(json.dumps({k:v for k,v in report.items() if k!='trace'}),flush=True)
        def callback(z):
            if not np.array_equal(z,cache['z']): evaluate(z)
            entry=dict(iteration=len(report['trace'])+1,objective=cache['value'],
                gradient_norm=float(np.linalg.norm(cache['gradient'])),seconds=time.monotonic()-started)
            report['trace'].append(entry)
            np.savez(out/'accepted_fit.npz',parameters=origin+scales*z,unit_parameters=z,
                origin=origin,scales=scales,richness_center=richness_center)
            save();print(json.dumps(entry),flush=True)
        fit=minimize(evaluate,initial,jac=True,method='L-BFGS-B',callback=callback,
                     options=dict(maxiter=32,maxls=12,ftol=1e-9,gtol=1e-5))
        value,gradient=evaluate(fit.x); final_rows=cache['rows'].copy()
        parameters=origin+scales*fit.x
        cpu_final=float(np.max(np.abs(cpu_rows(parameters)-final_rows[:8])))
        if cpu_final>1e-7: raise AssertionError('terminal CPU reference mismatch')
        # A finer rule on the SAME data/parameters, not a new target or fit.
        fine_t,fine_w=map(jnp.asarray,rule(96))
        finer=jax.jit(lambda p: row_logpdf(p,data,fine_t,fine_w))(jnp.asarray(parameters))
        fine_difference=np.asarray(finer)-final_rows
        report.update(status='RAW_POPULATION_FIXED_FIELD_FIT_NOT_CALIBRATION',
            optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
            iterations=int(fit.nit),evaluations=calls,final_objective=value,
            final_gradient_norm=float(np.linalg.norm(gradient)),final_CPU_max_abs_difference=cpu_final,
            cut64_to96_max_row_change=float(np.max(np.abs(fine_difference))),
            cut64_to96_total_change=float(fine_difference.sum()),
            limitations=['fixed field already fitted to summaries of these SAME observations',
                'conditional optical Gaussian with affine luminosity-proxy/richness mean',
                'constant graph/type incidence and inherited radial approximations uncalibrated',
                'no independent absolute calibration, population prior or posterior covariance',
                'not a calibrated present-field posterior; MW/M31 ambiguous, M33 unresolved'])
        np.savez(out/'final_fit.npz',parameters=parameters,initial_parameters=origin,
            richness_center=richness_center,PGC=mixture['PGC'],initial_logpdf=initial_rows,
            final_logpdf=final_rows,fine_logpdf=np.asarray(finer),mstar=mstar,lf_alpha=alpha)
        save()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
        plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3})
        fig,axes=plt.subplots(1,2,figsize=(11.69,8.27));fig.subplots_adjust(top=.82,bottom=.31,wspace=.3)
        axes[0].plot([0]+[p['iteration'] for p in report['trace']],
                     [f0]+[p['objective'] for p in report['trace']],'.-')
        axes[0].set(xlabel='공동 최적화 반복',ylabel='행당 음의 원자료 로그밀도',title='같은 관측모형 안에서만 비교')
        axes[1].scatter(np.arange(n),fine_difference,s=3)
        axes[1].set(xlabel='고정된 훈련 자료 번호',ylabel='정밀 적분 − 기본 적분 (행별 로그밀도)',
                    title='동일 매개변수의 수치 적분 차이')
        fig.suptitle('R2 진행 중 — 같은 현재장에 연결한 원자료 모형 적합',fontsize=17)
        fig.text(.07,.22,'1414개 자료를 함께 사용해 15개 공통 변수를 적합했습니다. 은하마다 보정값을 주지 않았습니다.\n'
            '기존 거리 PDF와 중복 점수화하지 않습니다. 중력 진화·새 장 표본추출·검증자료 평가는 없었습니다.\n'
            '이 장도 같은 자료로 추론했으므로 독립 검증이 아닙니다. 수치 적합이 실제 선택효과 보정을 보장하지 않습니다.\n'
            '절대 거리 보정·그룹/형태 선택·posterior 불확실성은 미해결. MW/M31 미식별, M33 미해결.',
            va='top',fontsize=10,linespacing=1.8)
        pdf=out/'R2_원자료공동적합_예제보고.pdf';fig.savefig(pdf);plt.close(fig)
        subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
            f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf)],check=True,timeout=90)
        print(json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='FAILED_RAW_POPULATION_FIT_NOT_CALIBRATION',error=repr(error));save()
        raise


if __name__=='__main__': main()
