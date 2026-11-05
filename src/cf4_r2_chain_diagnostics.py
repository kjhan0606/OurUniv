"""Small scalar diagnostics, not an automatic stationarity certificate.

Rank/folded split Rhat checks between-chain agreement. Batch-means MCSE/ESS
are rough within-chain estimates conditional on stationarity, not guarantees
for short drifting chains. No heldout data or new scientific target is used.
"""
import numpy as np
from scipy import fft
from scipy.special import ndtri
from scipy.stats import rankdata

R2_SCALAR_RHAT_MAX=1.01
R2_SCALAR_ESS_MIN=400.


def _rhat(x):
    n=x.shape[1]
    within=float(np.mean(np.var(x,axis=1,ddof=1)))
    if within<=0:return None
    between=n*float(np.var(np.mean(x,axis=1),ddof=1))
    return float(np.sqrt(((n-1)*within/n+between/n)/within))


def _rank_normalize(x):
    ranks=rankdata(x.ravel(),method='average').reshape(x.shape)
    return ndtri((ranks-.375)/(ranks.size+.25))


def _multi_chain_ess(x):
    """Geyer initial-monotone ESS using between- and within-chain variance."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<4 or not np.isfinite(x).all():
        raise ValueError('ESS requires at least two finite split chains of length four')
    chains,n=x.shape
    means=x.mean(axis=1)
    centered=x-means[:,None]
    within=float(np.mean(np.sum(centered*centered,axis=1)/(n-1)))
    between=float(n*np.var(means,ddof=1))
    if not np.isfinite(within) or within<=0:
        return None
    variance_plus=(n-1.)/n*within+between/n
    if not np.isfinite(variance_plus) or variance_plus<=0:
        return None

    # The FFT yields sums of products; division by n-1 makes lag zero equal
    # each chain's sample variance, matching W above.
    size=1 << (2*n-1).bit_length()
    spectrum=np.fft.rfft(centered,n=size,axis=1)
    autocov=np.fft.irfft(spectrum*np.conjugate(spectrum),n=size,axis=1)[:,:n]
    autocov/=(n-1.)
    rho=1.-(within-autocov.mean(axis=0))/variance_plus
    rho[0]=1.
    pair_count=n//2
    if pair_count<1:
        return None
    paired=rho[:2*pair_count:2]+rho[1:2*pair_count:2]
    # Keep the initial positive sequence, then enforce the initial monotone
    # sequence. Pair zero is retained; the positivity test starts at pair one.
    stop=1
    while stop<len(paired) and paired[stop]>0.:
        stop+=1
    paired=paired[:stop].copy()
    for i in range(1,len(paired)):
        paired[i]=min(paired[i],paired[i-1])
    tau=-1.+2.*float(np.sum(paired))
    if not np.isfinite(tau) or tau<=0.:
        return None
    return float(chains*n/tau)


def retained_field_observables(field, training_region, *, workers=1):
    """Return fixed low-k and training-region scalars from one retained map.

    Density modes use the native periodic PM-node convention. Velocity modes
    are those of the mass-weighted momentum density ``rho * v`` (zero in
    empty cells), avoiding an arbitrary velocity value for unoccupied cells.
    The regional velocity summary uses the same mass weighting.
    """
    rho=np.asarray(field['rho'],dtype=np.float64)
    velocity=np.asarray(field['mean_velocity_km_s'],dtype=np.float64)
    valid=np.asarray(field['velocity_valid'])
    region=np.asarray(training_region)
    if (rho.ndim!=3 or len(set(rho.shape))!=1 or rho.shape[0]<2
            or velocity.shape!=(3,)+rho.shape or valid.shape!=rho.shape
            or valid.dtype!=np.bool_ or region.shape!=rho.shape
            or region.dtype!=np.bool_ or not np.any(region)):
        raise ValueError('aligned cubic field and nonempty boolean training region required')
    if (not np.isfinite(rho).all() or not np.isfinite(velocity).all()
            or np.any(rho<0) or not np.array_equal(valid,rho>0)):
        raise ValueError('finite density and occupancy-aligned velocity field required')
    n=rho.shape[0]
    density_spectrum=fft.fftn(rho,norm='ortho',workers=workers)
    fundamental=((1,0,0),(0,1,0),(0,0,1))
    density_modes=np.asarray([density_spectrum[index] for index in fundamental])
    integer_k=np.fft.fftfreq(n)*n
    k2=(integer_k[:,None,None]**2+integer_k[None,:,None]**2
        +integer_k[None,None,:]**2)
    power=np.square(np.abs(density_spectrum))
    shells={}
    for name,mask in (('k2_1',k2==1),('k2_4_8',(k2>=4)&(k2<=8))):
        shells[name]=float(np.mean(power[mask])) if np.any(mask) else None

    momentum=np.where(valid[None],rho[None]*velocity,0.)
    momentum_modes=np.empty((3,3),dtype=np.complex128)
    for component in range(3):
        for axis in range(3):
            transverse=tuple(index for index in range(3) if index!=axis)
            line=np.mean(momentum[component],axis=transverse)
            # The 1-D orthonormal FFT of the transverse mean is N times the
            # corresponding 3-D orthonormal coefficient at this fundamental.
            momentum_modes[component,axis]=n*fft.fft(
                line,norm='ortho',workers=workers)[1]
    selected_density=rho[region]
    mass=float(np.sum(selected_density,dtype=np.float64))
    if not np.isfinite(mass) or mass<=0.:
        raise ValueError('training region must contain positive finite mass')
    regional_momentum=np.sum(momentum[:,region],axis=1,dtype=np.float64)
    return dict(
        density_mode_real=density_modes.real.tolist(),
        density_mode_imag=density_modes.imag.tolist(),
        density_band_power_k2_1=shells['k2_1'],
        density_band_power_k2_4_8=shells['k2_4_8'],
        momentum_mode_real=momentum_modes.real.reshape(-1).tolist(),
        momentum_mode_imag=momentum_modes.imag.reshape(-1).tolist(),
        training_region_density_mean=float(np.mean(selected_density)),
        training_region_occupancy_fraction=float(np.mean(valid[region])),
        training_region_mass_weighted_velocity_km_s=(regional_momentum/mass).tolist())


def scalar_diagnostics(chains):
    try:
        x=np.asarray(chains,dtype=float)
    except (TypeError,ValueError) as error:
        raise ValueError('equal-length finite chains are required') from error
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<8 or not np.isfinite(x).all():
        raise ValueError('at least two finite chains and eight retained states required')
    m,n=x.shape;half=n//2
    split=np.concatenate((x[:,:half],x[:,-half:]),axis=0)
    location=_rhat(_rank_normalize(split))
    folded=_rhat(_rank_normalize(np.abs(split-np.median(split))))
    estimates=[v for v in (location,folded) if v is not None]
    # Constant chains are not evidence of infinite effective sample size.
    degenerate=bool(np.any(np.var(x,axis=1)==0))
    rhat=max(estimates) if len(estimates)==2 and not degenerate else None
    bulk_ess=None
    tail_ess=None
    if not degenerate:
        bulk_ess=_multi_chain_ess(_rank_normalize(split))
        lower,upper=np.quantile(split,[.05,.95])
        lower_ess=_multi_chain_ess((split<=lower).astype(np.float64))
        upper_ess=_multi_chain_ess((split>=upper).astype(np.float64))
        if lower_ess is not None and upper_ess is not None:
            tail_ess=min(lower_ess,upper_ess)
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
        bulk_ess=bulk_ess,tail_ess=tail_ess,
        ess_method='rank-normalized split chains; Geyer initial monotone sequence; tail=min(5%,95%)',
        batch_means_mcse=mcse,batch_means_ess=ess,batch_size=size,
        batches_per_chain=batches,states_used_for_mcse=used,
        chain_means=x.mean(axis=1).tolist(),
        half_mean_change_in_chain_sd=[float(v) if np.isfinite(v) else None for v in drift],
        degenerate_chain=degenerate,
        caveat='rough within-chain MC error assumes stationarity; short/shared-ancestry chains cannot establish global exploration')


def retained_scalar_matrix(report, *, require_field_observables=False):
    rows=[r for r in report['trace'] if not r['warmup']]
    names=['fine_energy','white_mean_square','inherited_low_white_mean_square']
    sizes=dict(nuisance_white=24,fundamental_real=3,fundamental_imag=3,
        count_raw_scores=2,density_octant_means=8)
    field_arrays=dict(density_mode_real=3,density_mode_imag=3,
        momentum_mode_real=9,momentum_mode_imag=9,
        training_region_mass_weighted_velocity_km_s=3)
    field_scalars=('density_band_power_k2_1','density_band_power_k2_4_8',
        'training_region_density_mean','training_region_occupancy_fraction')
    field_keys=set(field_arrays)|set(field_scalars)
    field_rows=[field_keys.issubset(row) for row in rows]
    if any(field_rows) and not all(field_rows):
        raise ValueError('field-observable trace is incomplete across retained states')
    field_present=bool(rows) and all(field_rows)
    if require_field_observables and not field_present:
        raise ValueError('retained trace lacks required density/velocity field observables')
    if field_present:
        names.extend(field_scalars)
        sizes.update(field_arrays)
    labels=names+[f'{k}_{i}' for k,n in sizes.items() for i in range(n)]
    values=[]
    for row in rows:
        try:
            if any(len(row[k])!=n for k,n in sizes.items()):
                raise ValueError('incomplete trace')
            scalar_values=[float(row[k]) for k in names]
            array_values=[v for key in sizes for v in row[key]]
        except (KeyError,TypeError,ValueError) as error:
            raise ValueError('incomplete retained scalar trace') from error
        if not np.isfinite(scalar_values).all() or not np.isfinite(array_values).all():
            raise ValueError('retained scalar trace contains nonfinite values')
        values.append(scalar_values+array_values)
    return labels,np.asarray(values,dtype=float).reshape(len(rows),len(labels))


def four_chain_scalar_diagnostics(reports):
    """Evaluate the frozen scalar gate without trimming or promoting maps."""
    reports=list(reports)
    if len(reports)!=4:
        raise ValueError('exactly four independent chain reports are required')
    if any(report.get('status')!='N256_CHAIN_FINISHED_REQUIRES_DIAGNOSTICS'
            for report in reports):
        raise ValueError('all four chains must finish their declared schedule')
    if any(report.get('R2_complete') is not False
            or report.get('heldout_scored') is not False for report in reports):
        raise ValueError('chain inputs must remain pre-heldout and non-promoted')
    if any(report.get('initialization_policy')!='independent_prior'
            or report.get('long_chain') is not True for report in reports):
        raise ValueError('four independent prior-start long chains are required')
    start_hashes=[]
    for report in reports:
        selection=report.get('prior_start_selection',{})
        attempts=selection.get('attempts',[])
        digest=selection.get('selected_state_sha256')
        attempt_limit=selection.get('max_attempts')
        if (selection.get('status')!='FINITE_PRIOR_START_SELECTED'
                or not isinstance(attempts,list) or not 1<=len(attempts)<=4
                or not isinstance(attempt_limit,int) or isinstance(attempt_limit,bool)
                or not len(attempts)<=attempt_limit<=4
                or selection.get('selected_attempt')!=len(attempts)
                or not isinstance(digest,str) or len(digest)!=64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('each chain needs a recorded finite independent prior start')
        for attempt_number,attempt in enumerate(attempts,1):
            if (not isinstance(attempt,dict)
                    or attempt.get('attempt')!=attempt_number
                    or not isinstance(attempt.get('state_sha256'),str)
                    or len(attempt['state_sha256'])!=64
                    or any(char not in '0123456789abcdef'
                        for char in attempt['state_sha256'])):
                raise ValueError('prior-start attempt history is malformed')
            expected_status=('FINITE' if attempt_number==len(attempts)
                             else 'ZERO_COUNT_LIKELIHOOD')
            if attempt.get('status')!=expected_status:
                raise ValueError('prior-start attempt history has an invalid retry')
        if attempts[-1].get('state_sha256')!=digest:
            raise ValueError('selected prior-start digest differs from attempt history')
        start_hashes.append(digest)
    if len(set(start_hashes))!=4:
        raise ValueError('independent prior-start states are not distinct')
    designs=[report.get('chain_rng_design',{}) for report in reports]
    indices=[design.get('chain_index') for design in designs]
    seeds=[design.get('master_seed') for design in designs]
    if (any(not isinstance(index,int) or isinstance(index,bool) for index in indices)
            or sorted(indices)!=[0,1,2,3]
            or any(not isinstance(seed,int) or isinstance(seed,bool) or seed<0 for seed in seeds)
            or len(set(seeds))!=1
            or any(design.get('chain_count')!=4 for design in designs)):
        raise ValueError('four unique chain-indexed streams from one master seed required')
    expected_streams={'prior':0,'audit':1,'transition':2}
    for index,design in zip(indices,designs):
        streams=design.get('streams',{})
        if (design.get('bit_generator')!='PCG64'
                or set(streams)!=set(expected_streams)
                or any(streams[name].get('stream_id')!=stream_id
                    or streams[name].get('spawn_key')!=[index,stream_id]
                    for name,stream_id in expected_streams.items())):
            raise ValueError('independent prior/audit/transition RNG stream map differs')

    invariant_keys=('source_commit','target','N','box_cMpc_h',
        'fine_volume_order','force_volume_order','max_proposals',
        'warmup_proposals','schedule','field_mcse_contract',
        'association_ledger_source_commit','observation_target_revision',
        'source_conditioning_rule')
    for key in invariant_keys:
        if (any(key not in report for report in reports)
                or any(report.get(key)!=reports[0].get(key) for report in reports[1:])):
            raise ValueError(f'chain contract mismatch: {key}')
    chain_ids=[report.get('chain_id') for report in reports]
    if any(not value for value in chain_ids) or len(set(chain_ids))!=4:
        raise ValueError('four distinct nonempty retained-state chain IDs required')

    matrices=[]
    labels=None
    for report in reports:
        trace=report.get('trace')
        proposals=int(report['max_proposals'])
        warmup=int(report['warmup_proposals'])
        if (not isinstance(trace,list) or len(trace)!=proposals
                or [row.get('iteration') for row in trace]!=list(range(1,proposals+1))
                or [bool(row.get('warmup')) for row in trace]
                    !=[iteration<warmup for iteration in range(proposals)]):
            raise ValueError('chain trace is incomplete or does not match its schedule')
        current_labels,matrix=retained_scalar_matrix(
            report,require_field_observables=True)
        if labels is None:
            labels=current_labels
        elif current_labels!=labels:
            raise ValueError('chain observable definitions differ')
        expected=proposals-warmup
        if (matrix.shape[0]!=expected
                or report.get('retained_states')!=expected
                or report.get('retained_state_records')!=expected):
            raise ValueError('chain retained-state count differs from its declared schedule')
        matrices.append(matrix)
    lengths=[matrix.shape[0] for matrix in matrices]
    if len(set(lengths))!=1:
        raise ValueError('unequal retained chain lengths; no suffix trimming is allowed')
    stacked=np.stack(matrices)
    diagnostics={label:scalar_diagnostics(stacked[:,:,index])
        for index,label in enumerate(labels)}
    failing=[label for label,item in diagnostics.items()
        if (item['rank_folded_split_rhat'] is None
            or item['rank_folded_split_rhat']>R2_SCALAR_RHAT_MAX
            or item['bulk_ess'] is None or item['bulk_ess']<R2_SCALAR_ESS_MIN
            or item['tail_ess'] is None or item['tail_ess']<R2_SCALAR_ESS_MIN)]
    scalar_gate='PASS' if not failing else 'FAIL'
    return dict(status='FOUR_CHAIN_SCALAR_DIAGNOSTICS_ONLY_FIELD_GATE_PENDING',
        R2_complete=False,heldout_scored=False,n_chains=4,
        chain_ids=chain_ids,chain_indices=indices,master_seed=seeds[0],
        retained_per_chain=lengths,observables=labels,
        thresholds=dict(rank_folded_split_rhat_max=R2_SCALAR_RHAT_MAX,
            bulk_ess_min=R2_SCALAR_ESS_MIN,tail_ess_min=R2_SCALAR_ESS_MIN),
        scalar_gate=scalar_gate,failing_observables=failing,
        field_gate='NOT_EVALUATED',warmup_viability='NOT_EVALUATED',
        diagnostics=diagnostics,
        caveat='scalar diagnostics only; not full-field precision, held-out validation, or posterior promotion')
