"""Same-grid GL2/GL4 scores at TWO saved pilot states, not posterior reweighting."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import source_geometry_at_resolution
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_row_streamed_raw import RowStreamedRawReadout
from cf4_r2_raw_volume_target import tracer_masses,tracer_geometry
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood
from cf4_r2_count_exposure import build_population_exposure_masks

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def render(report,out,stream_check):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3,'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(11.69,8.27));a=axes.ravel()
    fig.subplots_adjust(top=.85,bottom=.28,hspace=.45,wspace=.4)
    fig.suptitle('실제 저장장 두 개의 수치 적분 검사 — posterior 검증은 아직 아님',fontsize=15)
    a[0].bar(['전체 동시 계산','관측 행 분할 계산'],
        [stream_check['reference_raw_score'],stream_check['actual_raw_score']])
    a[0].set(title=f"GL2 동일성: 절대차 {stream_check['absolute_error']:.2g}",ylabel='전체 raw log 점수')
    x=np.arange(2);d=np.array([r['GL4_minus_GL2'] for r in report['states']])
    a[1].bar(x-.15,d[:,0],width=.3,label='은하 수항');a[1].bar(x+.15,d[:,1],width=.3,label='FP/K항')
    a[1].set_xticks(x,['초기장','두 이동 후']);a[1].set(title='GL2에서 GL4로 바꾼 점수 차이',ylabel='log 점수 차이');a[1].legend()
    with np.load(out/'two_accepted_moves_raw.npz',allow_pickle=False) as f:
        radius=f['observed_radius'];fine=f['raw_logpdf'];pgc=f['PGC']
    with np.load(BASE/'r2_n256_row_stream_check_v1/readout.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['PGC'],pgc);change=fine-f['raw_logpdf']
    a[2].scatter(radius,change,s=4,alpha=.5);a[2].set(title='실제 1,414행 각각의 차이',xlabel='관측 위치 반경 (cMpc/h)',ylabel='GL4 - GL2 raw log 점수')
    a[3].plot(x,d.sum(axis=1),'o-');a[3].set_xticks(x,['초기장','두 이동 후'])
    a[3].set(title=f"두 상태간 보정 변화: {report['correction_change_between_states']:.4g}",ylabel='전체 log 점수 보정')
    fig.text(.05,.18,'격자와 관측 자료는 그대로 두고 셀 안의 적분점만 늘린 검사입니다. 중력을 다시 진화시키지 않았습니다.\n'
        '공통 점수 오프셋보다 상태에 따른 보정의 변화가 중요합니다. 두 상태만으로 전체 posterior 오차를 알 수는 없습니다.\n'
        'R1 → R2 진행 중 → R3 같은 장의 MW/M31/M33 → R4 정밀 진화 → R5 줌 IC',fontsize=10,linespacing=1.6,va='top')
    pdf=out/'R2_관측행분할과정밀적분_두상태예제.pdf';fig.savefig(pdf);plt.close(fig)
    subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r90',
        f'-sOutputFile={out}/pdf_page_%02d.png',str(pdf)],check=True,timeout=90)
    report.update(pdf=str(pdf),pages=1,pdf_pages_visually_reviewed=False)


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],R2_complete=False,
        N=256,source_orders=[2,4],observed_count_grid=128,source_cell_rate_factor=.125,
        PM_evolutions=0,heldout_scored=False,states=[],
        limitation='two pilot states, not posterior draws or an ensemble/global quadrature error bound')
    def save():
        report['seconds']=time.monotonic()-started
        report['host_peak_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        checked=json.loads((BASE/'r2_n256_row_stream_check_v1/result.json').read_text())
        if checked['status']!='N256_ROW_STREAMING_VALUE_CHECKED_NOT_GL4_SENSITIVITY':
            raise ValueError('whole-cohort GL2 equivalence must finish first')
        _,_,_,_,source,mix,o,g=load_inputs();source=source_geometry_at_resolution(source,256)
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
            keys,counts=map(jnp.asarray,(f['train_keys'],f['train_counts']))
            exposure,_=build_population_exposure_masks(128,f['heldout_flat_voxels'],
                f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        exposure=jnp.asarray(exposure)
        def count_value(density,velocity,tracer,src):
            intensity=predict_chunked_volume_intensity(src['positions'],velocity,
                tracer_masses(density,tracer)*.125,src['angular'],source_chunk_size=2097152,
                source_spacing=1.5,volume_order=4,**tracer_geometry(tracer,g),order=4,segments=8)
            return sparse_marked_poisson_log_likelihood(intensity,keys,counts,selected_voxel_mask=exposure)
        count_value=jax.jit(count_value)
        centre=jax.jit(native_mass_momentum_to_count_cells,static_argnums=2)
        readout=RowStreamedRawReadout(source,mix['population'],o,g,source_spacing=1.5,volume_order=4)
        initial=json.loads((BASE/'r2_n256_joint_pilot_v1/result.json').read_text())
        final=json.loads((BASE/'r2_n256_affine_pilot_v1/result.json').read_text())
        if not final['trace'][-1]['accepted']:raise ValueError('final reference must match accepted state')
        initial_parts=next(e['parts'] for e in initial['evaluations'] if e['order']==2 and not e['gradient'])
        states=[('initial',BASE/'r2_n256_dynamics_profile_v1/initial_present_state.npz',initial_parts),
                ('two_accepted_moves',BASE/'r2_n256_affine_pilot_v1/accepted_present_state.npz',final['evaluations'][-1]['parts'])]
        for name,path,reference in states:
            report.update(phase='GL4_COUNT',current_state=name);save()
            with np.load(path,allow_pickle=False) as f:
                rho,velocity,tracer,pop=map(jnp.asarray,(f['rho'],f['mean_velocity_km_s'],f['tracer'],f['population_white']))
            density,velocity=centre(rho,velocity,384.)
            velocity=jnp.moveaxis(velocity,0,-1).reshape(-1,3)
            arguments=(density,velocity,tracer,source)
            compiled=count_value.lower(*arguments).compile()
            memory=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
            peak=stats.get('bytes_in_use',0)+memory.temp_size_in_bytes+memory.output_size_in_bytes
            report['count_estimated_peak_GiB']=peak/1024**3;save()
            if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
                raise MemoryError('fine count value lacks20percent device margin')
            tic=time.monotonic();count=float(compiled(*arguments));count_seconds=time.monotonic()-tic
            if not np.isfinite(count):raise FloatingPointError('nonfinite GL4 count score')
            report.update(phase='GL4_RAW',current_count_score=count);save()
            def progress(row):
                report['completed_raw_rows']=row['stop'];save();print(json.dumps(dict(state=name,**row)),flush=True)
            values,info=readout.evaluate(density,velocity,tracer,pop,progress=progress)
            fine=np.array([count,float(values.sum())]);difference=fine-np.asarray(reference)
            entry=dict(name=name,GL2_count_raw=reference,GL4_count_raw=fine.tolist(),
                GL4_minus_GL2=difference.tolist(),log_likelihood_change=float(difference.sum()),
                count_seconds=count_seconds,raw=info)
            report['states'].append(entry);save();print(json.dumps(entry),flush=True)
            np.savez(out/f'{name}_raw.npz',PGC=mix['PGC'],raw_logpdf=values,observed_radius=mix['observed_radius'])
        report.update(status='N256_GL4_TWO_STATE_SENSITIVITY_NOT_POSTERIOR',
            correction_change_between_states=report['states'][1]['log_likelihood_change']-report['states'][0]['log_likelihood_change'])
        render(report,out,checked)
        save()
    except Exception as error:
        report.update(status='FAILED_N256_GL4_PAIR',error=repr(error));save();raise


if __name__=='__main__':main()
