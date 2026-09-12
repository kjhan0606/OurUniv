"""Approved population-weighted small location fit and fixed endpoint, Slurm only."""
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time
import traceback
import unittest

import h5py
import numpy as np
import torch

from cf4_conditional_split_flow import configure_precision
from cf4_population_locations import (DX, CORE, HALO, OBSERVER, SLABS, PopulationRoleModel,
    native_features, population_cases, tuple_targets, training_shared_probability,
    calibrate_alpha, field_patch, selected)
from cf4_member_mass_readout import transform, inverse_symmetry
from cf4_role_locations import transform_positions, sample_triples

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
CFG = json.loads(Path('config/cf4_population_locations_v1.json').read_text())
OUT = ROOT / CFG['output_name']
START = time.monotonic()
CREATED = False
RESULT = dict(status='NOT_STARTED', updates=0, epochs_completed=0, observed_LG_posterior=False,
    independent_validation=False, physical_halo_catalogue=False, field_prior=False)


def dump(name, value):
    with (OUT / name).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def memory():
    return dict(host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)


def check():
    if time.monotonic()-START > CFG['application_seconds']:
        raise TimeoutError('TOTAL application cap includes preparation, fit and evaluation')
    usage = memory()
    if any(usage[key] > CFG[key] for key in usage):
        raise MemoryError(f'predeclared sizing exceeded: {usage}')


def event(status, **extra):
    value = dict(status=status, job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-START, **extra)
    dump('status.json', value)
    print(json.dumps(value), flush=True)


def prepare_cases():
    with h5py.File(ROOT / 'lg_population_v1/population_and_marks.h5', 'r') as source:
        identities, branches = source['identities'][:], source['branch'][:]
        if source.attrs['status'] != 'ACTUAL_LG_MARK_DISTRIBUTIONS_NOT_DENSITY_POSTERIOR':
            raise ValueError('incorrect archived population source')
    wanted = np.unique(identities[branches == 0])
    positions, offset = {}, 0
    with h5py.File(ROOT / 'tng_operator_v2/native_catalog.h5', 'r') as source:
        if source.attrs['status'] != 'COMPLETE_NATIVE_FIELD_COPY_NO_SELECTION':
            raise ValueError('incomplete staged catalogue')
        for index in range(448):
            chunk = source[f'chunks/{index}']
            count = int(chunk['Header'].attrs['Nsubgroups_ThisFile'])
            ids = wanted[(wanted >= offset) & (wanted < offset+count)]
            if len(ids):
                values = chunk['Subhalo/SubhaloPos'][ids-offset].astype(np.float64)/1000
                positions.update({int(sid): p for sid, p in zip(ids, values)})
            offset += count
    if len(positions) != len(wanted) or not all(np.isfinite(p).all() for p in positions.values()):
        raise ValueError('missing/nonfinite native positions')
    cases, report = population_cases(identities, branches, positions, CFG['previous_development_lowers_cMpc_h'])
    report.update(dx_cMpc_h=DX, core_cells=CORE, feature_halo_cells=HALO, slabs=SLABS,
        observer_local_face_coordinates=OBSERVER, source_rows=int(len(identities)),
        weighting='uniform observer, then uniform distinct eligible M31, then uniform distinct eligible M33',
        selection='existing branch0 eligibility; not actual MW observer selection',
        within_split_overlap='expected, not independent universe samples',
        former_development_exclusion='periodic core+feature-halo versus old full128 cubes',
        calibration_observer_bounds_cMpc_h=(dict(
            minimum=np.min([positions[c['mw']] for c in cases['calibration']], axis=0).tolist(),
            maximum=np.max([positions[c['mw']] for c in cases['calibration']], axis=0).tolist())
            if cases['calibration'] else None))
    dump('population.json', report)
    dump('cases.json', cases)
    if any(len(cases[name]) < minimum for name, minimum in CFG['min_observers'].items()):
        raise ValueError('INCONCLUSIVE_DATA_SCOPE: fixed minimum observer counts not reached; no split relaxation')
    chosen = np.random.default_rng(CFG['rollout_selection_seed']).choice(
        len(cases['test']), size=CFG['rollout_observers'], replace=False)
    rollout_cases = [cases['test'][int(i)] for i in chosen]
    pi = training_shared_probability(cases['train'])
    report.update(same_cell_pi=pi, same_cell_smoothing='one pseudocount each, one weight unit per observer; NOT independent-trial posterior',
        rollout_observers_frozen_before_fit=[c['mw'] for c in rollout_cases])
    dump('population.json', report)
    RESULT['observer_counts'] = report['observer_counts']
    event('POPULATION_READY', counts=report['observer_counts'], pairs=report['pair_counts'], dropped=report['dropped'])
    return cases, rollout_cases, pi


def contexts(case):
    yield 0, [], [case['center']], 1.
    yield 1, [case['center']], [p['center'] for p in case['pairs']], 1.
    for pair in case['pairs']:
        yield 2, [case['center'], pair['center']], pair['centers'], 1/len(case['pairs'])


def learn(model, field, cases):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'], foreach=False)
    rng = np.random.default_rng(CFG['shuffle_seed'])
    history, exposure = [], Counter()
    started = time.monotonic()
    class LearningBudget(Exception):
        pass
    try:
        for epoch in range(1, CFG['epochs']+1):
            order = rng.permutation(len(cases))
            for start in range(0, len(order), CFG['batch_observers']):
                check()
                if time.monotonic()-started >= CFG['learning_seconds']:
                    raise LearningBudget()
                batch = [cases[int(i)] for i in order[start:start+CFG['batch_observers']]]
                optimizer.zero_grad(set_to_none=True)
                losses = torch.zeros(3, device=field.device)
                tick = time.monotonic()
                for case in batch:
                    raw = field_patch(field, case['origin'])
                    for role, parents, targets, weight in contexts(case):
                        logp = model.raw_log_prob(raw, OBSERVER, parents, role)
                        loss = -selected(logp, targets).mean()*weight/len(batch)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('nonfinite expected training loss')
                        losses[role] += loss.detach()
                        (loss/3).backward()
                        del logp, loss
                    del raw
                ridge = CFG['ridge']/2*sum(p.square().sum() for p in model.parameters())
                ridge.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError('missing/nonfinite linear-model gradient')
                optimizer.step()
                torch.cuda.synchronize()
                RESULT['updates'] += 1
                exposure.update(c['mw'] for c in batch)
                history.append(dict(update=RESULT['updates'], epoch=epoch, observers=len(batch),
                    role_NLL=losses.cpu().tolist(), ridge=float(ridge.detach()), seconds=time.monotonic()-tick))
                if RESULT['updates'] == 1 or RESULT['updates'] % 50 == 0:
                    dump('history.json', history)
                    event('TRAINING', epoch=epoch, epochs=CFG['epochs'], update=RESULT['updates'],
                          visited_observers=sum(exposure.values()), last=history[-1], memory=memory())
            RESULT['epochs_completed'] = epoch
            event('EPOCH_COMPLETE', epoch=epoch, update=RESULT['updates'])
    except LearningBudget:
        event('LEARNING_BUDGET_REACHED', updates=RESULT['updates'], epochs_completed=RESULT['epochs_completed'])
    finally:
        dump('history.json', history)
        dump('exposure.json', dict(observer_visits=dict(exposure), weighting='all alternatives on EVERY visit'))
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=CFG,
            source_commit=os.environ['EXPECTED_COMMIT'], updates=RESULT['updates'],
            epochs_completed=RESULT['epochs_completed'], shuffle_rng=rng.bit_generator.state), OUT / 'checkpoint_final.pt')
    RESULT.update(training_seconds=time.monotonic()-started, training_complete=RESULT['epochs_completed'] == CFG['epochs'])
    model.eval()


@torch.no_grad()
def symmetry_check(model, field, case, phase):
    raw = field_patch(field, case['origin'])
    first = case['pairs'][0]
    labels = torch.tensor([case['center'], first['center'], first['centers'][0]], device=field.device)
    observer = raw.new_tensor(OBSERVER)
    base = [model.log_prob(raw, observer, labels[:r], r).exp() for r in range(3)]
    rows = []
    for symmetry in range(48):
        check()
        changed = transform(raw, symmetry)
        new_o = transform_positions(observer, symmetry, CORE)
        new_c = transform_positions(labels, symmetry, CORE, cell_indices=True)
        errors = []
        for role in range(3):
            logp = model.log_prob(changed, new_o, new_c[:role], role)
            if not bool(torch.isfinite(logp).all()) or abs(float(logp.exp().sum())-1) > 1e-4:
                raise ValueError('nonfinite/unnormalized field law')
            restored = transform(logp.exp()[None], inverse_symmetry(symmetry))[0]
            errors.append(float(abs(restored-base[role]).sum()))
        rows.append(errors)
    maximum = np.max(rows, axis=0).tolist()
    report = dict(phase=phase, mw=case['mw'], alpha=model.alpha, all48_role_distribution_L1=rows,
                  maximum=maximum, passed=all(v <= CFG['symmetry_L1_max'] for v in maximum))
    dump(f'symmetry_{phase}.json', report)
    if not report['passed']:
        raise ValueError('signed-cubic numerical equivariance failure')
    event('SYMMETRY_PASS', phase=phase, maximum=maximum)
    return maximum


@torch.no_grad()
def score(model, field, cases, pi, split, for_calibration=False):
    rows, calibration = [], [[], [], []]
    for i, case in enumerate(cases):
        check()
        raw = field_patch(field, case['origin'])
        scores = {name: np.zeros(3) for name in ('raw', 'calibrated', 'density', 'geometry', 'same_cell')}
        for role, parents, targets, weight in contexts(case):
            distributions = model.distributions(raw, OBSERVER, parents, role, pi=pi)
            chosen = {name: selected(q, targets).cpu().numpy().astype(np.float64) for name, q in distributions.items()}
            if not all(np.isfinite(q).all() for q in chosen.values()):
                raise ValueError('nonfinite native expected score')
            for name, q in chosen.items():
                scores[name][role] -= float(q.mean())*weight
            if for_calibration:
                calibration[0].extend(chosen['density'].tolist())
                calibration[1].extend(chosen['raw'].tolist())
                calibration[2].extend([weight/(3*len(cases)*len(targets))]*len(targets))
        rows.append(dict(mw=case['mw'], pair_count=len(case['pairs']), split=split,
            role_NLL={name: q.tolist() for name, q in scores.items()},
            joint_NLL={name: float(q.sum()) for name, q in scores.items()}))
        if (i+1) % 100 == 0:
            event('SCORING', split=split, observers=i+1, total=len(cases))
    dump(f'scores_{split}.json', rows)
    means = {name: np.mean([r['role_NLL'][name] for r in rows], axis=0).tolist() for name in scores}
    return dict(role_NLL=means, joint_NLL={name: sum(v) for name, v in means.items()}), calibration


def plot_rollout(case, raw, samples, weights, targets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    background = np.log1p(raw[0].cpu().numpy().sum(2)).T
    for row, (name, draws) in enumerate(samples.items()):
        for role in range(3):
            ax = axes[row, role]
            ax.imshow(background, origin='lower', cmap='Greys', extent=(0, 12, 0, 12))
            xy = (draws[:, role, :2]+.5)*DX
            ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=.4)
            native = (targets[:, role, :2]+.5)*DX
            ax.scatter(native[:, 0], native[:, 1], marker='x', c='red', s=20+100*weights)
            ax.set(title=f'{name}: {("MW", "M31", "M33")[role]}', xlabel='x [cMpc/h]', ylabel='y [cMpc/h]')
    fig.suptitle(f'Observer {case["mw"]}: NATIVE density background; samples are not recovered halos')
    fig.savefig(OUT / f'locations_{case["mw"]}.png', dpi=120)
    plt.close(fig)


@torch.no_grad()
def rollouts(model, field, cases):
    reports = []
    saved_alpha = model.alpha
    for i, case in enumerate(cases):
        raw = field_patch(field, case['origin'])
        generated = {}
        for name, alpha in (('calibrated', saved_alpha), ('density', 0.)):
            model.alpha = alpha
            draws, logq = sample_triples(model, raw, OBSERVER, CFG['rollouts_per_observer_per_law'],
                torch.Generator(device=field.device).manual_seed(CFG['rollout_seed']+i), check)
            # Freeze both laws before accessing any target/candidate set for comparison.
            dump(f'samples_{case["mw"]}_{name}.json', dict(mw=case['mw'], law=name, alpha=alpha,
                cells=draws.tolist(), sampled_joint_logp=logq, true_parents_used=False, failures=0))
            generated[name] = draws
        model.alpha = saved_alpha
        targets, weights = tuple_targets(case)
        for name, draws in generated.items():
            distance = np.linalg.norm((draws[:, None]-targets[None])*DX, axis=-1)
            expected = np.einsum('str,t->sr', distance, weights)
            reports.append(dict(mw=case['mw'], law=name, samples=len(draws), failures=0,
                distinct_triples=len(np.unique(draws.reshape(len(draws), -1), axis=0)),
                M31_M33_shared_fraction=float(np.mean(np.all(draws[:, 1] == draws[:, 2], axis=1))),
                mean_weighted_target_distance_cMpc_h=expected.mean(0).tolist(),
                coordinate_std_cMpc_h=(draws*DX).std(0).tolist(),
                target='hierarchically weighted native alternatives, NOT nearest-truth selection or coverage'))
        plot_rollout(case, raw, generated, weights, targets)
        dump('rollouts.json', reports)
        event('ROLLOUT_FROZEN', mw=case['mw'], draws_per_law=CFG['rollouts_per_observer_per_law'])


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm execution required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    torch.set_num_threads(2)
    RESULT.update(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'])
    dump('request.json', dict(**CFG, source_commit=RESULT['source_commit'], job_id=RESULT['job_id'],
        gpu=torch.cuda.get_device_name(), precision=configure_precision(True),
        total_cap_includes_learning=True, fixed_context='64^3/.1875, observer faces36, feature halo4'))
    event('TESTING')
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_population_locations'])
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    dump('tests.json', dict(tests=tested.testsRun, errors=len(tested.errors), failures=len(tested.failures)))
    if not tested.wasSuccessful():
        raise ValueError('focused regressions failed')
    cases, selected_cases, pi = prepare_cases()
    event('BUILDING_FIXED_FEATURES')
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as source:
        if source.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE' or source['fine'].shape != (7,400,400,400):
            raise ValueError('incorrect native total-matter input')
        host_field = native_features(source['fine'], check)
    field = torch.from_numpy(host_field).cuda()
    del host_field
    check()
    event('FEATURES_READY', memory=memory())
    torch.manual_seed(CFG['seed'])
    model = PopulationRoleModel().cuda().eval()
    if sum(p.numel() for p in model.parameters()) != CFG['parameters']:
        raise ValueError('unexpected model size')
    RESULT['symmetry_before'] = symmetry_check(model, field, cases['train'][0], 'before')
    learn(model, field, cases['train'])
    RESULT['symmetry_raw_after'] = symmetry_check(model, field, cases['train'][0], 'raw_after')
    calibration_scores, records = score(model, field, cases['calibration'], pi, 'calibration', True)
    fitted = calibrate_alpha(*records)
    del records
    model.alpha = fitted['alpha']
    RESULT['calibration'] = fitted
    dump('calibration.json', dict(**fitted, pre_mixture_scores=calibration_scores,
        alpha_is_one_scalar_for_all_roles=True, test_used=False,
        caveat='thin calibration slab and one-box correlations; not independent-universe calibration'))
    RESULT['symmetry_calibrated_after'] = symmetry_check(model, field, cases['train'][0], 'calibrated_after')
    rollouts(model, field, selected_cases)
    RESULT['training_scores'], _ = score(model, field, cases['train'], pi, 'train')
    RESULT['test_scores'], _ = score(model, field, cases['test'], pi, 'test')
    joint, roles = RESULT['test_scores']['joint_NLL'], RESULT['test_scores']['role_NLL']
    criteria = dict(alpha_positive=model.alpha > 0,
        joint_beats_geometry=joint['calibrated'] < joint['geometry'],
        joint_beats_density=joint['calibrated'] < joint['density'],
        M33_beats_density=roles['calibrated'][2] < roles['density'][2],
        M33_beats_same_cell=roles['calibrated'][2] < roles['same_cell'][2])
    status = ('INCONCLUSIVE_LEARNING_BUDGET' if not RESULT['training_complete'] else
        'PASS_POPULATION_LOCATION_FEASIBILITY_ONLY' if all(criteria.values()) else
        'NO_GO_POPULATION_LOCATION_COMPARISON')
    RESULT.update(status=status, criteria=criteria, evaluation_complete=True,
        theta=[p.detach().cpu().tolist() for p in model.theta], alpha=model.alpha,
        application_seconds=time.monotonic()-START, memory=memory(),
        next_step='Report and wait: no automatic location repair, field-prior training, observed inference or IC')
    check()
    dump('result.json', RESULT)
    event(status, alpha=model.alpha, test_joint_NLL=joint, criteria=criteria)


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if CREATED:
            RESULT.update(status=('INCONCLUSIVE_DATA_SCOPE' if 'INCONCLUSIVE_DATA_SCOPE' in str(error)
                else 'INCONCLUSIVE_TECHNICAL_OR_BUDGET_FAILURE'), error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(), application_seconds=time.monotonic()-START)
            dump('result.json', RESULT)
            event(RESULT['status'], error=RESULT['error'])
        raise
