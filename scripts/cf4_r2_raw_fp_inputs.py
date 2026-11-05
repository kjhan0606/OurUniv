"""Join existing training photometry for selected-mark model development.

No new likelihood, calibration fit, heldout score, or gravity evolution.
Preserve published photometric systems; r-K is NOT automatically a colour.
"""
import json
import os
from pathlib import Path

import numpy as np

from cf4_r2_linked_fp_sparse_train import (
    load_train_singletons, select_training_single_mark_links, FP, POINTS)

BASE=Path('/gpfs/kjhan/CF4/z0_density')
SOURCE=Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')
FIELDS=('r','er','s','es','i','ei','deVMag_r','deVMagErr_r','deVMag_g',
        'extinction_r','extinction_g','kcor_r','kcor_g','Sn','zcmb',
        'zcmb_group','zhelio','NgroupT17')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    out=Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(exist_ok=False)
    split=BASE/'r2_sky_closed_split_v6/split.npz'
    options,_,_=load_train_singletons(split)
    with np.load(FP,allow_pickle=False) as f:
        pgc=f['PGC'].copy()
        groups=f['source_group'].astype(str)
        membership=f['membership_state'].astype(str)
    with np.load(split,allow_pickle=False) as f:
        training=set(f['fp_source_group'][f['fp_role']==0].astype(str))
    parent_rows=np.array([i for i,g in enumerate(groups) if g in training],dtype=int)
    parent_pgc=pgc[parent_rows]
    selected=select_training_single_mark_links(options,membership,include_grouped=True)
    if len(selected)!=1414:
        raise ValueError('current single-mark cohort changed')
    selected_pgc=pgc[[o[3] for o in selected]]
    if not set(selected_pgc).issubset(set(parent_pgc)):
        raise ValueError('selected cohort not in frozen training parent')
    wanted=set(map(int,parent_pgc)); raw={}
    with SOURCE.open() as handle:
        columns=handle.readline().lstrip('#').split()
        pi=columns.index('PGC'); indexes=[columns.index(k) for k in FIELDS]
        for line in handle:
            values=line.split()
            if not values or int(values[pi]) not in wanted:
                continue
            p=int(values[pi])
            if p in raw:
                raise ValueError('duplicate source PGC')
            # No source eta/likelihood columns are parsed for this assembly.
            raw[p]=[float(values[j]) for j in indexes]
    if set(raw)!=wanted:
        raise ValueError('missing training raw-photometry source row')
    parent=np.array([raw[int(p)] for p in parent_pgc])
    chosen=np.array([raw[int(p)] for p in selected_pgc])
    with np.load(POINTS,allow_pickle=False) as f:
        point_ids=np.array([o[2] for o in selected])
        recno=f['recno'][point_ids]
        ksmag=f['ksmag'][point_ids]
        population=f['population'][point_ids]
    if not np.isfinite(chosen).all() or not np.isfinite(ksmag).all():
        raise ValueError('selected source has nonfinite raw inputs; no silent row removal')
    if np.any(chosen[:,[FIELDS.index(k) for k in ('er','es','ei')]]<=0):
        raise ValueError('selected source has nonpositive reported measurement uncertainty')
    np.savez(out/'training_photometry.npz',columns=np.array(FIELDS),
        parent_PGC=parent_pgc,parent_source_group=groups[parent_rows],parent_photometry=parent,
        selected_PGC=selected_pgc,selected_source_group=np.array([o[0] for o in selected]),
        selected_photometry=chosen,point_recno=recno,point_ksmag=ksmag,population=population)
    result=dict(status='TRAINING_RAW_FP_INPUTS_READY_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        source=str(SOURCE),training_parent_rows=len(parent),selected_rows=len(chosen),
        training_parent_is_full_SDSS=False,heldout_scored=False,field_runs=0,fitted_parameters=0,
        R2_complete=False,fields=list(FIELDS),
        limitations=['parent is the existing CF4-linked training assembly, not full SDSS',
            'published r is a group-redshift-based log physical radius, not a true distance',
            'optical and K magnitudes kept in their source systems; no colour conversion',
            'cross-band selection and photometric-error covariance not supplied by this join',
            'no shared FP-parameter covariance or new zero prior inferred',
            'MW/M31 ambiguous, M33 unresolved; eventual constraints act on same NEW field'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf')
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False,'pdf.fonttype':3})
    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27))
    fig.subplots_adjust(top=.83,bottom=.32,wspace=.3)
    mag=FIELDS.index('deVMag_r')
    axes[0].hist(parent[:,mag][np.isfinite(parent[:,mag])],bins=np.arange(10,17.51,.5),
                 density=True,histtype='step',label=f'기존 훈련 부모 표본 {len(parent)}개')
    axes[0].hist(chosen[:,mag],bins=np.arange(10,17.51,.5),density=True,histtype='step',
                 label='현재 거리 조건 1414개')
    axes[0].set(xlabel='SDSS r 겉보기 등급',ylabel='정규화된 분포')
    axes[0].legend(fontsize=8)
    axes[1].scatter(ksmag,chosen[:,mag],s=3,alpha=.35)
    axes[1].set(xlabel='2M++ K 등급 (원자료 값)',ylabel='SDSS r 등급 (원자료 값)')
    fig.suptitle('R2 진행 중 — 관측 선택모형을 위한 실제 입력 연결',fontsize=17)
    fig.text(.07,.23,'왼쪽: 기존 훈련 자료와 현재 사용하는 표본의 밝기 분포를 비교합니다.\n'
        '오른쪽: 같은 은하의 광학·근적외선 측정값을 연결했습니다. 두 축의 등급계를 바꾸지 않았습니다.\n'
        '이 그림만으로 거리 편향의 크기나 원인을 확정할 수 없습니다. 부모 표본도 전체 SDSS가 아닙니다.\n'
        '새 보정·밀도장 학습·검증자료 점수 계산은 하지 않았습니다. R2 미완료, MW/M31/M33 미식별.',
        va='top',fontsize=10,linespacing=1.8)
    fig.savefig(out/'R2_관측입력_그림보고.pdf')
    fig.savefig(out/'training_photometry.png',dpi=110)
    plt.close(fig)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
