"""Same-state source-cell integration sensitivity; no PM, fitting or heldout."""
from itertools import product
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from cf4_r2_linked_fp_sparse_train import load_train_singletons, select_training_single_mark_links, FP, SOURCE
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_marked_tracer_jax import (intrinsic_biased_source_masses,
    intrinsic_lf_bin_fractions, predict_source_marked_radial_key_density, TRUE_EDGES, OBS_EDGES)
from cf4_r2_shell_cdf_count import predict_shell_cdf_intensity
from cf4_r2_raw_selected_fp import selected_mark_logpdf
from cf4_r2_raw_selected_fp_jax import unpack

BASE=Path('/gpfs/kjhan/CF4/z0_density')
N,BOX=128,384.


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']); out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(status='STARTED',source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        job_id=os.environ['SLURM_JOB_ID'],PM_evolutions=0,heldout_scored=False,
        R2_complete=False,calibration_refitted=False,rows=[],
        definition='piecewise-constant source mass/velocity/angular; GL volume integration',
        limits=['conditional development field, not posterior ensemble',
            'radial marks omit periodic images, unlike count integral',
            'finite source support and8sigma count truncation',
            'no sky-completeness subcell variation or physical LOS calibration',
            'MW/M31 ambiguous; M33 unresolved'])
    def save():
        report['seconds']=time.monotonic()-started
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        followup=os.environ.get('CF4_R2_CELL_FOLLOWUP')=='1'
        pcs=os.environ.get('CF4_R2_CELL_PCS')=='1'
        count_only=os.environ.get('CF4_R2_COUNT_ONLY')=='1'
        cdf_segments=int(os.environ.get('CF4_R2_CDF_SEGMENTS','32'))
        if followup:
            previous_path=BASE/'r2_source_cell_quadrature_v1/result.json'
            previous=json.loads(previous_path.read_text())
            if len(previous['rows'])!=72:
                raise ValueError('complete three-rule comparison required')
            report.update(rows=previous['rows'],precision_control_rows=[],
                predecessor=str(previous_path),predecessor_status=previous['status'],
                precision_control='original float32 source geometry; same strict1e-7 reproduction check')
        with np.load(BASE/'r2_prior_split_long_v1/final_state.npz',allow_pickle=False) as f:
            state={k:f[k].copy() for k in ('rho','velocity_km_s','tracer')}
        with np.load(SOURCE,allow_pickle=False) as f:
            source={k:f[k].copy() for k in f.files}
        with np.load(BASE/'r2_raw_source_mixture_v1/raw_source_mixture.npz',allow_pickle=False) as f:
            mix={k:f[k].copy() for k in ('PGC','population','dz_row','observed_radius',
                'observed_ksmag','mstar','lf_alpha')}
        with np.load(BASE/'r2_raw_population_fit_v1/final_fit.npz',allow_pickle=False) as f:
            params=f['parameters'].copy(); richness_center=float(f['richness_center'])
            reference=f['final_logpdf'].copy()
            np.testing.assert_array_equal(mix['PGC'],f['PGC'])
        b,cov=map(np.asarray,unpack(jnp.asarray(params)))
        with np.load(BASE/'r2_raw_selected_component_v1/training_cut_geometry.npz',allow_pickle=False) as f:
            order={int(p):i for i,p in enumerate(f['PGC'])}
            ind=np.array([order[int(p)] for p in mix['PGC']])
            x,errors,lower,upper=[f[k][ind] for k in
                ('x','optical_error_covariance','cut_lower','cut_upper')]
        with np.load(BASE/'r2_raw_fp_inputs_v1/training_photometry.npz',allow_pickle=False) as f:
            order={int(p):i for i,p in enumerate(f['selected_PGC'])}
            ind=np.array([order[int(p)] for p in mix['PGC']])
            richness=np.log1p(f['selected_photometry'][ind,list(f['columns']).index('NgroupT17')])-richness_center
        options,point,_=load_train_singletons(BASE/'r2_sky_closed_split_v6/split.npz')
        with np.load(FP,allow_pickle=False) as f:
            options=select_training_single_mark_links(options,f['membership_state'].astype(str),include_grouped=True)
            chosen=[o for p in range(6) for o in options if point['population'][o[2]]==p]
            np.testing.assert_array_equal(mix['PGC'],[f['PGC'][o[3]] for o in chosen])
        rows=np.concatenate([np.flatnonzero(mix['population']==p)[:4] for p in range(6)])
        if len(rows)!=24: raise ValueError('prespecified24-row cohort unavailable')
        count_rows={int(np.flatnonzero(mix['population']==p)[0]) for p in range(6)}
        voxels=np.array([np.unravel_index(point['flat_cell'][chosen[i][2]],(N,)*3) for i in rows])
        tr=jnp.asarray(state['tracer'])
        @jax.jit
        def field_arrays(rho,v):
            density,vel=native_mass_momentum_to_count_cells(rho,v,BOX)
            mass=intrinsic_biased_source_masses(density,
                jnp.log(jnp.sum(intrinsic_lf_bin_fractions()[1:4]))+2*tr[0],
                jnp.exp(.5*tr[1:6]),mstar=-23.28+.2*tr[8],alpha=-1+.06*jnp.exp(.5*tr[7]),
                reference_interval=(-25.,-21.))
            return jnp.moveaxis(vel,0,-1).reshape(-1,3),mass
        velocity,masses=map(np.asarray,field_arrays(jnp.asarray(state['rho']),jnp.asarray(state['velocity_km_s'])))
        del state
        sigma=float(100*np.exp(.5*tr[6])); eps=1e-4
        # Unshifted centres plus GLOBAL velocity bound covers every subnode,
        # TSC support and both small derivative perturbations by triangle inequality.
        vmax=float(np.max(np.linalg.norm(velocity,axis=1)))
        radius=.01*(vmax*(1+2*eps)+8*sigma)+np.sqrt(3)*2*BOX/N
        tree=cKDTree(source['positions']%BOX,boxsize=BOX)
        neighbors=tree.query_ball_point((voxels+.5)*BOX/N,radius,workers=1)
        # Per-cell velocity bounds tighten the conservative global query,
        # without choosing cells from their realised likelihood weights.
        for k,(voxel,ids) in enumerate(zip(voxels,neighbors)):
            ids=np.array(ids,dtype=int)
            delta=(source['positions'][ids]-(voxel+.5)*BOX/N+BOX/2)%BOX-BOX/2
            bound=.01*(np.linalg.norm(velocity[ids],axis=1)*(1+2*eps)+8*sigma)+np.sqrt(3)*2*BOX/N
            neighbors[k]=ids[np.linalg.norm(delta,axis=1)<=bound].tolist()
        width=((max(map(len,neighbors))+255)//256)*256
        if width>16384: raise ValueError('source support exceeds bounded comparison budget')
        report.update(PGC= mix['PGC'][rows].tolist(),cohort_rows=rows.tolist(),sigma_km_s=sigma,
            support_radius=radius,maximum_cells=max(map(len,neighbors)),padded_cells=width,
            source_spacing=BOX/N,velocity_direction='multiply ALL cell velocities by1+epsilon')
        save(); print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
        geometry=dict(observer=jnp.full(3,192.),box_size_cMpc_h=BOX,hubble_km_s_Mpc=74.6,
            little_h=.746,radius_table_cMpc_h=jnp.asarray(source['radial_table']),
            modulus_table_h=jnp.asarray(source['modulus_table']),redshift_table=jnp.asarray(source['redshift_table']),
            grid_size=N,sigma_los_km_s=sigma,mstar=float(mix['mstar']),alpha=float(mix['lf_alpha']))
        def radial(pos,vel,mass,angular,voxel,obs,pop):
            f=lambda scale:predict_source_marked_radial_key_density(pos,scale*vel,mass,angular,
                pop,voxel,obs,**geometry)
            value,derivative=jax.jvp(f,(jnp.array(1.),),(jnp.array(1.),))
            return value,derivative,f(1+eps),f(1-eps)
        radial=jax.jit(radial,static_argnums=6)
        def count(pos,vel,mass,angular,voxel,pop):
            f=lambda scale:predict_shell_cdf_intensity(pos,scale*vel,mass,angular,**geometry,
                order=4,segments=cdf_segments,target_population=pop,target_voxel=voxel,source_cell_average=pcs)
            return jax.jvp(f,(jnp.array(1.),),(jnp.array(1.),))
        count=jax.jit(count,static_argnums=5)
        if pcs or count_only:
            reference=json.loads((BASE/'r2_source_cell_quadrature_v2/result.json').read_text())
            ref={r['row']:r for r in reference['rows'] if r['nodes_per_axis']==(8 if pcs else 4)}
            report['definition']=('centre LF/RSD with analytically cell-averaged TSC; approximate, NOT full volume integration'
                if pcs else 'same4^3 source-volume integrand; LOS strata sensitivity ONLY')
            report['cdf_segments']=cdf_segments
            q=1 if pcs else 4
            qt,qw=np.polynomial.legendre.leggauss(q)
            indices=np.array(list(product(range(q),repeat=3)))
            subpositions=qt[indices]*1.5
            subweight=np.prod(qw[indices]/2,axis=1)
            for row,voxel,ids in zip(rows,voxels,neighbors):
                if int(row) not in count_rows:continue
                ids=np.array(sorted(ids),dtype=int);used=len(ids)
                ids=np.pad(ids,(0,width-used),constant_values=ids[0])
                pos=(source['positions'][ids,None,:]+subpositions[None])%BOX
                vel=np.broadcast_to(velocity[ids,None,:],pos.shape).reshape(-1,3)
                mass=masses[:,ids,None]*subweight[None,None,:];mass[:,used:]=0.
                args=tuple(map(jnp.asarray,(pos.reshape(-1,3),vel,mass.reshape(5,-1),
                    np.repeat(source['angular'][:,ids],q**3,axis=1),voxel)))
                evaluated_at=time.monotonic()
                value,derivative=map(float,count(*args,int(mix['population'][row])))
                entry=dict(row=int(row),PGC=int(mix['PGC'][row]),count_mean=value,
                    reference_count=ref[int(row)]['count_mean'],
                    count_relative_error=value/ref[int(row)]['count_mean']-1,
                    velocity_derivative=derivative,
                    reference_derivative=ref[int(row)]['count_velocity_derivative'],
                    evaluation_including_compile_seconds=time.monotonic()-evaluated_at)
                report['rows'].append(entry);save();print(json.dumps(entry),flush=True)
            report['status']=('CELL_AVERAGED_KERNEL_APPROXIMATION_MEASURED_NOT_TARGET' if pcs else
                              'LOS_STRATA_SENSITIVITY_MEASURED_NOT_TARGET');save()
            return
        diagnostic_rows=set(count_rows)
        if followup:
            old={(r['row'],r['nodes_per_axis']):r for r in previous['rows']}
            worst=max(rows,key=lambda r:abs(old[int(r),4]['velocity_derivative']-
                                           old[int(r),2]['velocity_derivative']))
            diagnostic_rows.add(int(worst))
            report['order8_rows']=sorted(diagnostic_rows)
            report['order8_selection']='six prespecified count rows plus largest2->4 raw derivative change; diagnostic only'
        radius_function=jax.jit(lambda pos:jnp.linalg.norm(
            (pos-geometry['observer']+BOX/2)%BOX-BOX/2,axis=1))
        for q in ((1,8) if followup else (1,2,4)):
            nodes,weights=np.polynomial.legendre.leggauss(q)
            ijk=np.array(list(product(range(q),repeat=3)))
            offsets=nodes[ijk]*(BOX/N)/2
            volume=np.prod(weights[ijk]/2,axis=1)
            np.testing.assert_allclose(volume.sum(),1.,atol=2e-15)
            for row,voxel,ids in zip(rows,voxels,neighbors):
                if followup and q==8 and int(row) not in diagnostic_rows: continue
                if time.monotonic()-started>1000: raise TimeoutError('bounded source-volume comparison')
                ids=np.asarray(sorted(ids),dtype=int); used=len(ids)
                padded=np.pad(ids,(0,width-used),constant_values=ids[0])
                pos=(source['positions'][padded,None,:]+offsets[None,:,:])%BOX
                if followup and q==1:
                    pos=source['positions'][padded,None,:].copy()
                vel=np.broadcast_to(velocity[padded,None,:],pos.shape).reshape(-1,3)
                mass=(masses[:,padded,None]*volume[None,None,:])
                mass[:,used:,:]=0.;mass=mass.reshape(5,-1)
                angular=np.repeat(source['angular'][:,padded],len(offsets),axis=1)
                pos=pos.reshape(-1,3); pop=int(mix['population'][row])
                args=tuple(map(jnp.asarray,(pos,vel,mass,angular,voxel)))
                w,dw,wp,wm=map(np.asarray,radial(*args,float(mix['observed_radius'][row]),pop))
                # Keep every component with positive weight at ANY derivative point.
                bi,si=np.nonzero(np.maximum.reduce([w,wp,wm])>0)
                vectors=np.stack([a[bi,si] for a in (w,dw,wp,wm)])
                rt=np.asarray(radius_function(jnp.asarray(pos)))[si].astype(float)
                eta=np.log10(mix['dz_row'][row]/rt)
                mt=np.interp(rt,source['radial_table'],source['modulus_table'])
                zt=np.interp(rt,source['radial_table'],source['redshift_table'])
                ro=mix['observed_radius'][row]
                mo=np.interp(ro,source['radial_table'],source['modulus_table'])
                zo=np.interp(ro,source['radial_table'],source['redshift_table'])
                correction=1.16*2.9*(zo-zt)-1.6*np.log10((1+zo)/(1+zt))
                lo=np.maximum.reduce([np.asarray(TRUE_EDGES)[bi],
                    np.full_like(mt,-np.inf if pop//3==0 else 11.5)-mt-correction,
                    np.full_like(mt,OBS_EDGES[pop%3])+mo-mt-correction])
                hi=np.minimum.reduce([np.asarray(TRUE_EDGES)[bi+1],
                    (11.5 if pop//3==0 else 12.5)-mt-correction,
                    np.full_like(mt,OBS_EDGES[pop%3+1])+mo-mt-correction])
                observed_m=mix['observed_ksmag'][row]-mt-correction
                ln,ld=[],[]
                for start in range(0,len(si),512):
                    sl=slice(start,start+512)
                    a,c=selected_mark_logpdf(x[row],observed_m[sl],np.zeros(len(si[sl])),
                        eta[sl],lo[sl],hi[sl],b[0]+richness[row]*b[2],b[1],cov+errors[row],
                        np.array([[2.,0.,1.],[.04,1.,0.]]),lower[row],upper[row],
                        mstar=float(mix['mstar']),alpha=float(mix['lf_alpha']),cut_order=64,
                        return_component_terms=True)
                    ln.append(a);ld.append(c)
                ln,ld=np.concatenate(ln),np.concatenate(ld)
                def score(ww):
                    lw=np.full_like(ww,-np.inf); np.log(ww,out=lw,where=ww>0)
                    return float(logsumexp(lw+ln)-logsumexp(lw+ld))
                score0,plus,minus=map(score,vectors[[0,2,3]])
                # Signed directional derivative, with stable positive reference sums.
                lw=np.full_like(vectors[0],-np.inf)
                np.log(vectors[0],out=lw,where=vectors[0]>0)
                numerator,denominator=logsumexp(lw+ln),logsumexp(lw+ld)
                derivative=float(np.sum(vectors[1]*np.exp(ln-numerator))-
                                 np.sum(vectors[1]*np.exp(ld-denominator)))
                finite_difference=(plus-minus)/(2*eps)
                entry=dict(row=int(row),PGC=int(mix['PGC'][row]),population=pop,
                    nodes_per_axis=q,source_cells=used,positive_components=len(si),
                    raw_logpdf=score0,velocity_derivative=derivative,
                    finite_difference=finite_difference,
                    derivative_error=abs(derivative-finite_difference)/max(1.,abs(derivative),abs(finite_difference)),
                    reference_center_difference=score0-float(reference[row]) if q==1 else None)
                if row in count_rows and not (followup and q==1):
                    value,grad=map(float,count(*args,pop))
                    entry.update(count_mean=value,count_velocity_derivative=grad)
                if not np.isfinite([score0,derivative,finite_difference]).all():
                    raise FloatingPointError('nonfinite source-cell comparison')
                target='precision_control_rows' if followup and q==1 else 'rows'
                report[target].append(entry);save();print(json.dumps(entry),flush=True)
            if followup and q==1:
                error=max(abs(r['reference_center_difference']) for r in report['precision_control_rows'])
                report['native_precision_reproduction_error']=error;save()
                if error>1e-7: raise AssertionError('native-precision centre reproduction failed')
        baseline=(report['precision_control_rows'] if followup else
                  [r for r in report['rows'] if r['nodes_per_axis']==1])
        report['maximum_center_reproduction_error']=max(abs(r['reference_center_difference']) for r in baseline)
        report['maximum_derivative_error']=max(r['derivative_error'] for r in report['rows'])
        # Only implementation-reproduction checks; no arbitrary science pass gate.
        if report['maximum_center_reproduction_error']>1e-7 or report['maximum_derivative_error']>2e-5:
            raise AssertionError('centre reproduction or analytic directional derivative failed')
        report['status']='SOURCE_CELL_SENSITIVITY_MEASURED_NOT_POSTERIOR';save()
        make_pdf(out,report)
        save()
    except Exception as error:
        report.update(status='FAILED_SOURCE_CELL_COMPARISON',error=repr(error));save();raise


def make_pdf(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3})
    pdf=out/'R2_격자내부적분_실제예제.pdf'
    with PdfPages(pdf) as pages:
        fig,ax=plt.subplots(1,3,figsize=(11.69,8.27));fig.subplots_adjust(top=.8,bottom=.3,wspace=.35)
        for k,q in enumerate((1,2,4)):
            t,_=np.polynomial.legendre.leggauss(q); xx,yy=np.meshgrid(t*1.5,t*1.5)
            ax[k].scatter(xx,yy,s=35);ax[k].set(xlim=(-1.6,1.6),ylim=(-1.6,1.6),aspect='equal',
                xlabel='격자 중심으로부터 거리 (cMpc/h)',title=f'한 축 {q}점 / 3차원 {q**3}점')
            for edge in (-1.5,1.5): ax[k].axhline(edge,color='gray');ax[k].axvline(edge,color='gray')
        fig.suptitle('R2 — 같은 격자의 중심점 근사와 부피 적분 비교',fontsize=17)
        fig.text(.07,.2,'새 우주 시뮬레이션이 아닙니다. 같은 격자의 질량을 보존하며 내부 평가점을 늘립니다.\n'
            '밀도·속도·하늘 선택률은 격자 안에서 일정하게 유지합니다. 3 cMpc/h보다 작은 구조를 새로 만든 것이 아닙니다.\n'
            '관측 예측만 비교합니다. 훈련 24개 은하를 사전 선택했고 보정변수 재적합·검증자료 사용은 없습니다.',fontsize=11,linespacing=1.8)
        pages.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11.69,8.27));fig.subplots_adjust(top=.86,bottom=.23,hspace=.4,wspace=.3)
        for i,row in enumerate(report['cohort_rows']):
            rr=[r for r in report['rows'] if r['row']==row]
            label=f"PGC {rr[0]['PGC']}" if i%4==0 else None
            q=[r['nodes_per_axis'] for r in rr]
            axes[0,0].plot(q,[r['raw_logpdf']-rr[0]['raw_logpdf'] for r in rr],'.-',alpha=.6,label=label)
            axes[0,1].plot(q,[r['velocity_derivative'] for r in rr],'.-',alpha=.6)
            if 'count_mean' in rr[0]:
                axes[1,0].plot(q,[r['count_mean']/rr[0]['count_mean'] for r in rr],'.-',label=label)
                axes[1,1].plot(q,[r['count_velocity_derivative'] for r in rr],'.-')
        for a in axes.flat:
            a.set_xticks(sorted({r['nodes_per_axis'] for r in report['rows']}))
            a.set_xlabel('한 축 내부 적분점 수')
        axes[0,0].set_title('24개 실제 관측: 원자료 로그밀도 변화')
        axes[0,1].set_title('원자료 점수의 속도 변화에 대한 기울기')
        axes[1,0].set_title('6개 실제 관측 위치: 예상 개수 / 중심점 값')
        axes[1,1].set_title('예상 개수의 속도 변화에 대한 기울기')
        axes[1,0].legend(fontsize=7,ncol=2)
        fig.suptitle('실제 은하 위치의 예측과 기울기 — 같은 장·같은 보정변수',fontsize=16)
        fig.text(.07,.12,'위: 관측된 밝기·크기·속도분산 등을 설명하는 정도. 아래: 해당 관측 격자의 예상 은하 개수.\n'
            '수치 변화는 선택효과·절대 거리 보정의 검증을 뜻하지 않습니다. R2 미완료, MW/M31 모호, M33 미해결.',fontsize=10,linespacing=1.7)
        pages.savefig(fig);plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf)],check=True,timeout=90)


if __name__=='__main__':main()
