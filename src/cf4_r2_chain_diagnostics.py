"""Small scalar diagnostics, not an automatic stationarity certificate.

Rank/folded split Rhat checks between-chain agreement. Batch-means MCSE/ESS
are rough within-chain estimates conditional on stationarity, not guarantees
for short drifting chains. No heldout data or new scientific target is used.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _rhat(x):
    n=x.shape[1]
    within=float(np.mean(np.var(x,axis=1,ddof=1)))
    if within<=0:return None
    between=n*float(np.var(np.mean(x,axis=1),ddof=1))
    return float(np.sqrt(((n-1)*within/n+between/n)/within))


def scalar_diagnostics(chains):
    x=np.asarray(chains,dtype=float)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<8 or not np.isfinite(x).all():
        raise ValueError('at least two finite chains and eight retained states required')
    m,n=x.shape;half=n//2
    split=np.concatenate((x[:,:half],x[:,-half:]),axis=0)
    def ranked(values):
        ranks=rankdata(values.ravel(),method='average').reshape(values.shape)
        return ndtri((ranks-.375)/(values.size+.25))
    location=_rhat(ranked(split))
    folded=_rhat(ranked(np.abs(split-np.median(split))))
    estimates=[v for v in (location,folded) if v is not None]
    # Constant chains are not evidence of infinite effective sample size.
    degenerate=bool(np.any(np.var(x,axis=1)==0))
    rhat=max(estimates) if len(estimates)==2 and not degenerate else None
    size=int(np.sqrt(n));batches=n//size;used=batches*size
    means=x[:,:used].reshape(m,batches,size).mean(axis=2)
    se2=float(np.var(means,axis=1,ddof=1).sum()/batches/m**2)
    variance=float(np.var(x[:,:used],axis=1,ddof=1).mean())
    mcse=float(np.sqrt(se2)) if se2>0 and not degenerate else None
    ess=min(float(m*used),variance/se2) if mcse is not None else None
    sd=np.std(x,axis=1,ddof=1)
    drift=np.divide(x[:,-half:].mean(axis=1)-x[:,:half].mean(axis=1),sd,
        out=np.full(m,np.nan),where=sd>0)
    return dict(chains=m,states_per_chain=n,rank_folded_split_rhat=rhat,
        batch_means_mcse=mcse,batch_means_ess=ess,batch_size=size,
        batches_per_chain=batches,states_used_for_mcse=used,
        chain_means=x.mean(axis=1).tolist(),
        half_mean_change_in_chain_sd=[float(v) if np.isfinite(v) else None for v in drift],
        degenerate_chain=degenerate,
        caveat='rough within-chain MC error assumes stationarity; short/shared-ancestry chains cannot establish global exploration')


def retained_scalar_matrix(report):
    rows=[r for r in report['trace'] if not r['warmup']]
    names=['fine_energy','white_mean_square','inherited_low_white_mean_square']
    sizes=dict(nuisance_white=24,fundamental_real=3,fundamental_imag=3,
        count_raw_scores=2,density_octant_means=8)
    labels=names+[f'{k}_{i}' for k,n in sizes.items() for i in range(n)]
    values=[]
    for row in rows:
        if any(len(row[k])!=n for k,n in sizes.items()):raise ValueError('incomplete trace')
        values.append([row[k] for k in names]+[v for k in sizes for v in row[k]])
    return labels,np.asarray(values,dtype=float).reshape(len(rows),len(labels))
