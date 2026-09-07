"""Measure a continuous marked population and condition it on actual LG data.

Slurm numerical work; serve-stars on syntax only copies requested native data.
No density painting, fixed-field catalogue inheritance, CF4 multiplication,
native full snapshot rescan or claim of a .1875-cMpc/h posterior.
"""
import argparse
from itertools import product
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from cf4_bundle_c_tng_operator import load_catalog
from cf4_bundle_c_tng_stage import OUT as NATIVE, RAW
from cf4_lg_population import FEATURES, invariant_kinematics, fit_gaussian, gaussian_logpdf
from cf4_lg_population import conditional_gaussian, sky_observable_log_jacobian, summarize_weights
from cf4_lg_observation_contract import basis, solar_reference, approximate_covariance

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUTPUT = ROOT / 'lg_population_v1'
REPO = Path(__file__).resolve().parents[1]


def delta(x, y, box=75.):
    return (x-y+box/2) % box-box/2


def environment(position, velocity, field):
    """Native 1.5-cell shell summaries; not .1875 information or isotropic map."""
    dx = 1.5
    offsets = np.array(list(product(range(-6, 7), repeat=3)))
    unwrapped = np.floor(position/dx).astype(int) + offsets
    dr = (unwrapped+.5)*dx-position
    radius = np.linalg.norm(dr, axis=1)
    selected = (radius >= 2) & (radius < 8)
    dr, radius, cells = dr[selected], radius[selected], unwrapped[selected] % 50
    value = field[:, cells[:, 0], cells[:, 1], cells[:, 2]]
    mean_box = field[0].mean()
    rho, flow, dispersion = [], [], []
    for mask in (radius < 4, radius >= 4):
        moments = value[:, mask].sum(axis=1)
        mean = moments[1:4]/moments[0]
        variance = np.sum(moments[4:7]/moments[0]-mean**2)/3
        if variance <= 0:
            raise ValueError('invalid shell moments')
        rho.append(np.log(moments[0]/(mask.sum()*mean_box)))
        radial = ((value[1:4, mask].T-value[0, mask, None]*velocity)*dr[mask]/radius[mask, None]).sum()/moments[0]
        flow.append(radial/100)
        dispersion.append(np.log(np.sqrt(variance)/100))
    return np.r_[rho, flow, dispersion]


def population(cat, h, field):
    g, s = cat['Group'], cat['Subhalo']
    first = g['GroupFirstSub'].astype(np.int64)
    m200 = g['Group_M_Crit200'].astype(float)*1e10/h
    valid_group = (first >= 0) & (m200 >= 2e11) & (m200 <= 5e12)
    groups = np.flatnonzero(valid_group)
    sid = first[groups]
    groups = groups[s['SubhaloFlag'][sid] & (s['SubhaloLenType'][sid, 1] >= 1000)]
    sid = first[groups]
    positions = s['SubhaloPos'].astype(float)/1000
    velocities = s['SubhaloVel'].astype(float)
    bound = s['SubhaloMass'].astype(float)*1e10/h
    host_tree = cKDTree(positions[sid], boxsize=75)
    companions = np.flatnonzero(s['SubhaloFlag'] & (bound >= 1e9) & (bound <= 1e12)
        & (s['SubhaloLenType'][:, 1] >= 1000) & (s['SubhaloLenType'][:, 4] >= 100))
    satellite_tree = cKDTree(positions[companions], boxsize=75)
    rows, identities, branch, split, weights = [], [], [], [], []
    env_cache = {}
    for i, mw in enumerate(sid):
        mw = int(mw)
        candidates = []
        for j in host_tree.query_ball_point(positions[mw], 3*h):
            m31 = int(sid[j])
            if m31 == mw:
                continue
            r31 = delta(positions[m31], positions[mw])*1000/h
            if np.linalg.norm(r31) < 200:
                continue
            thirds = []
            for k in satellite_tree.query_ball_point(positions[m31], 1.5*h):
                m33 = int(companions[k])
                if m33 in (mw, m31) or bound[m33] > .5*m200[groups[j]]:
                    continue
                r32 = delta(positions[m33], positions[m31])*1000/h
                if np.linalg.norm(r32) < 50:
                    continue
                # Third object is either a satellite of M31's FoF or a
                # separate FoF primary. Exclude satellites of a fourth host.
                g33 = int(s['SubhaloGrNr'][m33])
                kind = 0 if g33 == groups[j] else 1 if first[g33] == m33 else -1
                if kind < 0:
                    continue
                xs = positions[[mw, m31, m33], 0]
                subset = 0 if np.all(xs < 50) else 1 if np.all(xs >= 50) else -1
                if subset < 0:
                    continue
                try:
                    kin = invariant_kinematics(r31, r32, velocities[m31]-velocities[mw], velocities[m33]-velocities[m31])
                except ValueError:
                    continue
                thirds.append((m33, kind, subset, kin))
            if thirds:
                candidates.append((j, m31, thirds))
        if not candidates:
            continue
        if mw not in env_cache:
            env_cache[mw] = environment(positions[mw], velocities[mw], field)
        for j, m31, thirds in candidates:
            for m33, kind, subset, kin in thirds:
                rows.append(np.r_[np.log([m200[groups[i]], m200[groups[j]], bound[m33]]), kin, env_cache[mw]])
                identities.append([mw, m31, m33])
                branch.append(kind)
                split.append(subset)
                weights.append(1/len(candidates)/len(thirds))
    return tuple(map(np.asarray, (rows, identities, branch, split, weights)))


def stellar_request(cat, counts, identities, split):
    """Fixed-seed32 centrals+32 satellites, selected without observed LG fit."""
    g, s = cat['Group'], cat['Subhalo']
    pool = np.unique(identities[split == 0])
    is_central = g['GroupFirstSub'][s['SubhaloGrNr'][pool]] == pool
    rng = np.random.default_rng(91821)
    chosen = []
    for kind, mask in ((0, is_central), (1, ~is_central)):
        eligible = pool[mask & (s['SubhaloLenType'][pool, 4] >= 200)]
        if len(eligible) < 32:
            raise ValueError(f'fewer than32 training stellar calibration objects in kind{kind}')
        chosen.extend((int(x), kind) for x in rng.choice(eligible, size=32, replace=False))
    group_prefix = np.r_[0, np.cumsum(g['GroupLenType'][:, 4], dtype=np.int64)]
    sub_prefix = np.r_[0, np.cumsum(s['SubhaloLenType'][:, 4], dtype=np.int64)]
    file_prefix = np.r_[0, np.cumsum(counts[:, 4], dtype=np.int64)]
    with h5py.File(NATIVE / 'native_catalog.h5', 'r') as f:
        catalog_prefix = np.r_[0, np.cumsum([int(f[f'chunks/{i}/Header'].attrs['Nsubgroups_ThisFile']) for i in range(448)])]
    objects = []
    for sid, kind in chosen:
        group_id = int(s['SubhaloGrNr'][sid])
        start = int(group_prefix[group_id] + sub_prefix[sid]-sub_prefix[g['GroupFirstSub'][group_id]])
        length = int(s['SubhaloLenType'][sid, 4])
        pieces = []
        for i in range(448):
            lo, hi = max(start, int(file_prefix[i])), min(start+length, int(file_prefix[i+1]))
            if hi > lo:
                pieces.append(dict(file=i, start=lo-int(file_prefix[i]), stop=hi-int(file_prefix[i])))
        if sum(p['stop']-p['start'] for p in pieces) != length:
            raise ValueError('stellar native ranges incomplete')
        ci = int(np.searchsorted(catalog_prefix, sid, side='right')-1)
        objects.append(dict(sid=sid, kind=kind, count=length, pieces=pieces, catalog_file=ci,
            catalog_row=sid-int(catalog_prefix[ci]), position_native=s['SubhaloPos'][sid].tolist(), velocity=s['SubhaloVel'][sid].tolist()))
    count = sum(obj['count'] for obj in objects)
    if count > 20000000:
        raise ValueError('stellar calibration exceeds20m-row read budget; no automatic expansion')
    return dict(objects=objects, total_rows=count, source='existing TNG100-1 snapshot99; requested stellar subsets only')


def serve_stars(request_path):
    """I/O only: copy native half-mass radius and requested stellar row slices."""
    request = json.loads(Path(request_path).read_text())
    stream = sys.stdout.buffer
    for obj in request['objects']:
        with h5py.File(RAW / f"groups_099/fof_subhalo_tab_099.{obj['catalog_file']}.hdf5", 'r') as f:
            radius = f['Subhalo/SubhaloHalfmassRadType'][obj['catalog_row'], 4]
        pickle.dump(('object', obj['sid'], radius), stream, protocol=5)
        for piece in obj['pieces']:
            with h5py.File(RAW / f"snapdir_099/snap_099.{piece['file']}.hdf5", 'r') as f:
                group = f['PartType4']
                for lo in range(piece['start'], piece['stop'], 131072):
                    hi = min(lo+131072, piece['stop'])
                    pickle.dump(('rows', *(group[name][lo:hi] for name in
                        ('Coordinates', 'Velocities', 'Masses', 'GFM_StellarFormationTime'))), stream, protocol=5)
        pickle.dump(('end_object', obj['sid']), stream, protocol=5)
        stream.flush()
    pickle.dump(('end',), stream, protocol=5)
    stream.flush()


def measure_stars(request, h):
    path = OUTPUT / 'stellar_request.json'
    path.write_text(json.dumps(request, indent=2))
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'syntax',
        'env', f'PYTHONPATH={REPO / "src"}:{REPO / "scripts"}',
        '/home/kjhan/miniconda3/bin/python3.13', str(Path(__file__).resolve()), 'serve-stars', '--request', str(path)]
    source = subprocess.Popen(command, stdout=subprocess.PIPE)
    records = []
    try:
        for obj in request['objects']:
            tag, sid, half = pickle.load(source.stdout)
            if tag != 'object' or sid != obj['sid'] or half <= 0:
                raise ValueError('invalid stellar source header')
            accum = np.zeros((2, 7))
            counts = np.zeros(2, dtype=int)
            streamed = 0
            while True:
                item = pickle.load(source.stdout)
                if item[0] == 'end_object':
                    if item[1] != sid or streamed != obj['count']:
                        raise ValueError('incomplete stellar object stream')
                    break
                _, x, v, m, formation = item
                streamed += len(m)
                dr = delta(x.astype(float), obj['position_native'], box=75000.)
                radial = np.linalg.norm(dr, axis=1)
                for k, aperture in enumerate((1, 2)):
                    keep = (formation > 0) & (radial <= aperture*half)
                    weights = m[keep].astype(float)
                    accum[k, 0] += weights.sum()
                    accum[k, 1:4] += (weights[:, None]*v[keep]).sum(axis=0)
                    accum[k, 4:7] += (weights[:, None]*dr[keep]).sum(axis=0)
                    counts[k] += keep.sum()
            if np.any(counts < 100) or np.any(accum[:, 0] <= 0):
                raise ValueError('insufficient actual stars in calibrated aperture')
            records.append(dict(sid=sid, kind=obj['kind'], counts=counts.tolist(),
                halfmass_kpc=float(half/h), velocity_offset_km_s=(accum[:, 1:4]/accum[:, 0, None]-obj['velocity']).tolist(),
                position_offset_kpc=(accum[:, 4:7]/accum[:, 0, None]/h).tolist()))
        if pickle.load(source.stdout) != ('end',) or source.wait(timeout=30) != 0:
            raise ValueError('stellar stream did not complete')
    finally:
        if source.poll() is None:
            source.terminate()
            try:
                source.wait(timeout=10)
            except subprocess.TimeoutExpired:
                source.kill()
                source.wait()
        source.stdout.close()
    return records


def discrepancy_model(records, aperture):
    """Isotropic joint stellar-minus-halo position/velocity proxy from TNG.

    Each6D covariance retains position/velocity cross-correlation, averaged
    over orientations. These are physical COM offsets, not observation errors
    or particle velocity dispersions. Pairs of different halos are assumed
    independent; the SAME MW offset is shared by both relative observables.
    """
    covariance, diagnostic = {}, {}
    for category in (0, 1):
        selected = [r for r in records if r['kind'] == category]
        x = np.array([r['position_offset_kpc'][aperture] for r in selected])
        v = np.array([r['velocity_offset_km_s'][aperture] for r in selected])
        xx, vv, xv = np.mean(x*x), np.mean(v*v), np.mean(x*v)
        covariance[category] = np.kron([[xx, xv], [xv, vv]], np.eye(3))
        np.linalg.cholesky(covariance[category])
        diagnostic[str(category)] = dict(objects=len(selected), sigma_position_kpc=float(np.sqrt(xx)),
            sigma_velocity_km_s=float(np.sqrt(vv)), position_velocity_correlation=float(xv/np.sqrt(xx*vv)),
            covariance=covariance[category].tolist())
    return covariance, diagnostic


def observables_to_kinematics(values, contract, h, offsets=None):
    solar_x, solar_v = solar_reference(contract)
    if offsets is None:
        offsets = np.zeros((len(values), 3, 6))
    positions, velocities, distances = [], [], []
    for j, name in enumerate(contract['galaxy_order']):
        observed = contract['measurements'][name]
        frame = basis(observed['ra_deg'], observed['dec_deg'])
        distance = 10**((values[:, 4*j]-10)/5)
        r = distance[:, None]*frame[0] + solar_x
        components = values[:, 4*j+1:4*j+4].copy()
        components[:, 1:] *= 4.74047*distance[:, None]/1000
        v = components @ frame + solar_v - .1*h*r
        positions.append(r+offsets[:, 0, :3]-offsets[:, j+1, :3])
        velocities.append(v+offsets[:, 0, 3:]-offsets[:, j+1, 3:])
        distances.append(distance)
    kin = invariant_kinematics(positions[0], positions[1]-positions[0], velocities[0], velocities[1]-velocities[0])
    return kin, np.array(distances).T


def bounds(features):
    lo = np.log([2e11, 2e11, 1e9, 200, 50])
    hi = np.log([5e12, 5e12, 1e12, 3000, 1500])
    return np.all((features[:, :5] >= lo) & (features[:, :5] <= hi), axis=1) & (features[:, 2] <= features[:, 1]+np.log(.5))


def condition(mean, covariance, contract, records, kind, aperture, seed, h):
    rng = np.random.default_rng(seed)
    obs_mean = np.concatenate([contract['measurements'][g]['value'] for g in contract['galaxy_order']])
    obs_cov = approximate_covariance(contract)  # Explicit partial-covariance development opt-in.
    proxy_cov, discrepancy = discrepancy_model(records, aperture)
    size = 65536
    proposals = rng.multivariate_normal(obs_mean, obs_cov, size=size)
    categories = [0, 0, 1 if kind == 0 else 0]
    offsets = np.stack([rng.multivariate_normal(np.zeros(6), proxy_cov[k], size=size) for k in categories], axis=1)
    kin, distances = observables_to_kinematics(proposals, contract, h, offsets)
    known = np.arange(3, 12)
    logw = gaussian_logpdf(kin, mean[known], covariance[np.ix_(known, known)]) + sky_observable_log_jacobian(kin, distances)
    other, conditional_mean, conditional_cov = conditional_gaussian(mean, covariance, known, kin)
    remainder = conditional_mean + rng.normal(size=(size, len(other))) @ np.linalg.cholesky(conditional_cov).T
    features = np.empty((size, len(mean)))
    features[:, known], features[:, other] = kin, remainder
    logw[~bounds(features)] = -np.inf
    weights, ess = summarize_weights(logw)
    chosen = rng.choice(size, size=4096, replace=True, p=weights)
    return dict(features=features[chosen], proposal_ids=chosen, observables=proposals[chosen], stellar_halo_offsets=offsets[chosen],
        importance_weights=weights, prior_features=prior_draws(mean, covariance, rng, 4096)), dict(
        branch='M33_satellite_of_M31' if kind == 0 else 'M33_separate_primary', aperture_halfmass_radii=aperture+1,
        proposal_count=size, proposal_ESS=ess, maximum_proposal_weight=float(weights.max()),
        rejected_selection_fraction=float(np.mean(~bounds(features))), unique_resampled_proposals=len(np.unique(chosen)),
        observational_covariance=obs_cov.tolist(), discrepancy=discrepancy,
        status='ACTUAL_LG_MARKS_DEVELOPMENT_POSTERIOR_NOT_CF4_DENSITY_MAP' if ess >= 1000 else 'NO_GO_OBSERVATIONAL_PROPOSAL_SUPPORT')


def prior_draws(mean, covariance, rng, count):
    # Bounded rejection from a normalized Gaussian truncated to population
    # support. Its normalizer cancels only within each fixed branch/model.
    proposals = rng.multivariate_normal(mean, covariance, size=131072)
    valid = proposals[bounds(proposals)]
    if len(valid) < count:
        raise ValueError('insufficient truncated-prior proposal support')
    return valid[:count]


def run():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('numerical population work requires Slurm')
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    cat, header, _, counts = load_catalog()
    h = float(header['HubbleParam'])
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as f:
        if f.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE':
            raise ValueError('complete native source required')
        field = f['coarse'][:]
    rows, identities, kinds, split, weights = population(cat, h, field)
    print(f'Native population rows={len(rows)}, observers={len(np.unique(identities[:, 0]))}', flush=True)
    request = stellar_request(cat, counts, identities, split)
    del cat
    records = measure_stars(request, h)
    with (OUTPUT / 'stellar_calibration.json').open('x') as f:
        json.dump(dict(requested_rows=request['total_rows'], records=records), f, indent=2)
    print(f'Stellar proxy calibrated on {len(records)} native objects', flush=True)
    contract = json.loads((REPO / 'config/cf4_lg_observation_contract_v1.json').read_text())
    reports = []
    with h5py.File(OUTPUT / 'population_and_marks.h5', 'x') as f:
        f.attrs.update(status='INCOMPLETE', h=h, feature_names=json.dumps(FEATURES), source_commit=os.environ['EXPECTED_COMMIT'])
        for name, value in (('features', rows), ('identities', identities), ('branch', kinds), ('heldout', split), ('row_weights', weights)):
            f.create_dataset(name, data=value, compression='gzip')
        for kind in (0, 1):
            train, held = (kinds == kind) & (split == 0), (kinds == kind) & (split == 1)
            observers = [len(np.unique(identities[m, 0])) for m in (train, held)]
            if min(observers) < 30 or train.sum() < 100 or held.sum() < 50:
                reports.append(dict(branch=kind, status='NO_GO_POPULATION_TOO_SMALL', observer_counts=observers,
                    row_counts=[int(train.sum()), int(held.sum())]))
                continue
            # Retarget each branch to uniform MW, then uniform represented M31,
            # then uniform eligible M33 within that pair. Counts cannot create
            # many independent copies of an observer with abundant satellites.
            def branch_weights(mask):
                ids = identities[mask]
                result = np.empty(len(ids))
                for mw in np.unique(ids[:, 0]):
                    own = ids[:, 0] == mw
                    for m31 in np.unique(ids[own, 1]):
                        pair = own & (ids[:, 1] == m31)
                        result[pair] = 1/len(np.unique(ids[own, 1]))/pair.sum()
                return result/result.sum()
            tw, hw = branch_weights(train), branch_weights(held)
            mean, covariance = fit_gaussian(rows[train], tw)
            group = f.create_group(f'branch_{kind}')
            group.create_dataset('mean', data=mean)
            group.create_dataset('covariance', data=covariance)
            group.create_dataset('training_weight', data=tw)
            group.create_dataset('heldout_weight', data=hw)
            logs = gaussian_logpdf(rows[held], mean, covariance)
            diagonal_logs = gaussian_logpdf(rows[held], mean, np.diag(np.diag(covariance)))
            # Compare the actual truncated models, not differently normalized
            # untruncated Gaussian scores. Freeze this MC precision in advance.
            norm_rng = np.random.default_rng(28170+kind)
            probabilities = []
            for cov in (covariance, np.diag(np.diag(covariance))):
                draws = norm_rng.multivariate_normal(mean, cov, size=131072)
                probabilities.append(float(bounds(draws).mean()))
            if min(probabilities) <= 0:
                raise ValueError('zero estimated population selection normalization')
            logs -= np.log(probabilities[0])
            diagonal_logs -= np.log(probabilities[1])
            held_diagnostic = dict(observer_counts=observers, row_counts=[int(train.sum()), int(held.sum())],
                weighted_heldout_logscore_gain_over_diagonal=float(hw @ (logs-diagonal_logs)),
                truncation_probability_full_diagonal=probabilities, normalization_proposals_per_model=131072,
                heldout_standardized_mean=((hw @ rows[held]-mean)/np.sqrt(np.diag(covariance))).tolist(),
                native_catalogue_identity_overlap=int(len(np.intersect1d(identities[train].ravel(), identities[held].ravel()))),
                limits='Disjoint native object IDs; shared box long modes and potentially overlapping shell voxels, NOT independent simulations. No tuning on these diagnostics.')
            for aperture in (0, 1):
                samples, report = condition(mean, covariance, contract, records, kind, aperture, 72410+kind*2+aperture, h)
                report['population_check'] = held_diagnostic
                out = group.create_group(f'aperture_{aperture+1}')
                for name, value in samples.items():
                    out.create_dataset(name, data=value, compression='gzip')
                for label in ('features', 'prior_features'):
                    report[label+'_quantiles_16_50_84'] = np.quantile(samples[label], [.16, .5, .84], axis=0).tolist()
                reports.append(report)
            print(f'Finished branch {kind}: training/heldout observers {observers}', flush=True)
        f.attrs['status'] = 'ACTUAL_LG_MARK_DISTRIBUTIONS_NOT_DENSITY_POSTERIOR'
    report = dict(status='DEVELOPMENT_LG_MARK_INFERENCE_REQUIRES_DRIVER_REVIEW', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=os.environ['EXPECTED_COMMIT'], total_population_rows=len(rows),
        unique_native_observers=len(np.unique(identities[:, 0])), h=h, stellar_rows_read=request['total_rows'],
        feature_names=FEATURES, cases=reports, elapsed_seconds=time.monotonic()-started,
        limits=['No CF4 data factor or fine density-map likelihood. This is a marked-population/shell-summary model, NOT the final field posterior.',
                'Gaussian in invariant log/asinh mark coordinates, with frozen5% diagonal shrinkage and explicit selection truncation; not Gaussian z0 density.',
                'Broad mass/separation/resolution eligibility defines an astrophysical population prior, not new measurements. No isolation or binding cut.',
                'M33 satellite and separate-primary cases are conditional alternatives; no posterior odds without branch/selection normalizers.',
                'Stellar COM offsets are aperture-dependent proxies, not calibrated Gaia disk-fit, gas LOS, LMC/reflex or PM-distance-reduction systematics.',
                'Fixed solar reference and partial published observational covariance remain assumptions. Stellar position/velocity offsets marginalized jointly, same MW draw shared by M31/M33; cross-halo physical correlations otherwise ignored.',
                'One TNG box/cosmology and correlated objects. Mark covariance fit does not calibrate arbitrary field/reservoir deformations or eliminate environmental support failures.',
                'Density profiles and angular environmental field remain absent; never convert shell summaries into a claimed .1875-cMpc/h LG map.'])
    with (OUTPUT / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


def summarize():
    """Read saved draws only: physical units and local reference-coverage check."""
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('numerical summaries require Slurm')
    result = json.loads((OUTPUT / 'result.json').read_text())
    request = json.loads((NATIVE / 'particle_request.json').read_text())
    resolution_mass = 1000*request['DM_mass_native']*1e10/request['h']
    names = ['MW_host_M200c_Msun', 'M31_host_M200c_Msun', 'M33_bound_mass_Msun',
        'MW_M31_kpc', 'M31_M33_kpc', 'triangle_cosine'] + [
        f'peculiar_v{p}_{a}_km_s' for p in ('31', '32') for a in ('r', 't', 'n')] + [
        'density_2_4_over_mean', 'density_4_8_over_mean', 'flow_2_4_km_s',
        'flow_4_8_km_s', 'physical_sigma_2_4_km_s', 'physical_sigma_4_8_km_s']
    def physical(x):
        x = x.copy()
        x[:, :5] = np.exp(x[:, :5])
        x[:, 5] = np.tanh(x[:, 5])
        x[:, 6:12] = 100*np.sinh(x[:, 6:12])
        x[:, 12:14] = np.exp(x[:, 12:14])
        x[:, 14:16] *= 100
        x[:, 16:] = 100*np.exp(x[:, 16:])
        return x
    cases = []
    with h5py.File(OUTPUT / 'population_and_marks.h5', 'r') as f:
        rows, kind, split = f['features'][:], f['branch'][:], f['heldout'][:]
        for report in result['cases']:
            if 'proposal_ESS' not in report:
                continue
            branch = 0 if report['branch'] == 'M33_satellite_of_M31' else 1
            aperture = report['aperture_halfmass_radii']
            group = f[f'branch_{branch}']
            covariance = group['covariance'][:][3:12, 3:12]
            L = np.linalg.cholesky(covariance)
            train, held = (kind == branch) & (split == 0), (kind == branch) & (split == 1)
            def white(x):
                return np.linalg.solve(L, x[:, 3:12].T).T
            tree = cKDTree(white(rows[train]))
            held_dist = tree.query(white(rows[held]))[0]
            draws = group[f'aperture_{aperture}']
            post, prior = draws['features'][:], draws['prior_features'][:]
            posterior_dist = tree.query(white(post))[0]
            cutoff = float(np.quantile(held_dist, .95))
            posterior_q, prior_q = np.quantile(physical(post), [.16, .5, .84], axis=0), np.quantile(physical(prior), [.16, .5, .84], axis=0)
            cases.append(dict(branch=report['branch'], aperture_halfmass_radii=aperture,
                proposal_ESS=report['proposal_ESS'], physical_quantiles={name: dict(
                    prior_16_50_84=prior_q[:, i].tolist(), posterior_16_50_84=posterior_q[:, i].tolist(),
                    interval_width_ratio=float((posterior_q[2, i]-posterior_q[0, i])/(prior_q[2, i]-prior_q[0, i]))) for i, name in enumerate(names)},
                native_nearest_neighbor_coverage=dict(metric='Euclidean in9 training-covariance-whitened kinematic coordinates; not template weighting or independent validation',
                    heldout_distance_50_95=np.quantile(held_dist, [.5, .95]).tolist(),
                    posterior_distance_50_95=np.quantile(posterior_dist, [.5, .95]).tolist(),
                    posterior_fraction_beyond_heldout_95=float(np.mean(posterior_dist > cutoff))),
                minimum_mass_of1000_native_DM_particles_Msun=resolution_mass,
                posterior_M33_fraction_below_that_mass=float(np.mean(np.exp(post[:, 2]) < resolution_mass))))
    summary = dict(status='LG_MARKS_AND_REFERENCE_COVERAGE_REVIEW_NOT_SPATIAL_MAP', source_job=result['job_id'],
        summary_job=os.environ['SLURM_JOB_ID'], cases=cases,
        limits='Nearest-neighbor distances only disclose local extrapolation; no new bank/support tuning. Interval shrinkage is not independent accuracy or spatial information resolution. Masses are native host M200c/bound definitions, not extra measurements or isolated-M33 M200c.')
    with (OUTPUT / 'physical_summary.json').open('x') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'serve-stars', 'summarize'])
    parser.add_argument('--request')
    args = parser.parse_args()
    if args.action == 'run':
        run()
    elif args.action == 'summarize':
        summarize()
    else:
        serve_stars(args.request)
