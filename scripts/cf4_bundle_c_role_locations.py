"""One approved probabilistic center-location pilot; Slurm only, no follow-up."""
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

from cf4_bundle_c_continuous import read_periodic_patch
from cf4_bundle_c_spatial_model import overlaps
from cf4_conditional_split_flow import configure_precision
from cf4_member_mass_readout import transform, inverse_symmetry
from cf4_role_locations import (ROLES, WIDTHS, RoleLocationNet, native_state,
    native_center_cells, transform_state, transform_positions, location_features,
    reference_log_probs, sample_triples)

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
CFG = json.loads(Path('config/cf4_role_locations_v1.json').read_text())
OUT = ROOT / CFG['output_name']
START = time.monotonic()
CREATED = False
RESULT = dict(status='NOT_STARTED', updates=0, roles=ROLES,
    observed_LG_posterior=False, field_prior_training=False,
    member_masses_or_velocities_inferred=False, independent_validation=False)


def dump(name, value):
    with (OUT / name).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def memory():
    return dict(host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)


def check():
    if time.monotonic()-START > CFG['application_seconds']:
        raise TimeoutError('application budget; incomplete evidence, not scientific failure')
    measured = memory()
    if any(measured[key] > CFG[key] for key in measured):
        raise MemoryError(f'approved sizing exceeded: {measured}')


def event(status, **extra):
    value = dict(status=status, job_id=os.environ['SLURM_JOB_ID'],
                 seconds=time.monotonic()-START, **extra)
    dump('status.json', value)
    print(json.dumps(value), flush=True)


def load_field(native, record):
    origin = np.rint(np.asarray(record['patch_lower_cMpc_h'])/CFG['dx_cMpc_h']).astype(int)
    if not np.allclose(origin*CFG['dx_cMpc_h'], record['patch_lower_cMpc_h'], rtol=0, atol=1e-10):
        raise ValueError('patch origin not on native grid')
    return native_state(read_periodic_patch(native['fine'], origin % 400, 128))


def prepare():
    request = json.loads((ROOT / 'spatial_calibration_v1/request.json').read_text())
    objects = {int(obj['sid']): obj for obj in request['objects']}
    with h5py.File(ROOT / 'spatial_calibration_v1/spatial_components.h5', 'r') as source:
        if source.attrs['status'] != 'NATIVE_JOINT_PROFILES_AND_REMAINDER_NOT_CF4_LG_POSTERIOR':
            raise ValueError('incorrect native source status')
        records = {i: json.loads(source[f'patches/{i}'].attrs['record_json'])
                   for i in CFG['train']+CFG['development']}
    if any(overlaps(records[i]['patch_lower_cMpc_h'], records[j]['patch_lower_cMpc_h'])
           for i in CFG['train'] for j in CFG['development']):
        raise ValueError('fixed training/development voxel overlap')
    labels = {i: native_center_cells([objects[s]['position_ckpc_h'] for s in r['ids']],
              r['patch_lower_cMpc_h'], CFG['dx_cMpc_h']) for i, r in records.items()}
    shared = {i: bool(np.array_equal(c[1], c[2])) for i, c in labels.items()}
    pi = (sum(shared[i] for i in CFG['train'])+1)/(len(CFG['train'])+2)
    counts = Counter(s for i in CFG['train'] for s in records[i]['ids'])
    dump('source.json', dict(records=records, native_center_cells={i: c.tolist() for i, c in labels.items()},
        native_M31_M33_center_same_cell=shared, same_cell_probability=pi,
        mixture_source='training native SubhaloPos CENTER cells; Beta(1,1), no development tuning',
        training_shared_ids={str(k): v for k, v in counts.items() if v > 1},
        train_development_shared_ids=sorted(set(counts) & {s for i in CFG['development'] for s in records[i]['ids']}),
        training_overlapping_pairs=[[i, j] for k, i in enumerate(CFG['train']) for j in CFG['train'][k+1:]
            if overlaps(records[i]['patch_lower_cMpc_h'], records[j]['patch_lower_cMpc_h'])],
        label_use='teacher-forced training and proper joint scoring ONLY; never autoregressive parents',
        independent_validation=False, target_population='native two-primary/M31-satellite E_train'))
    cache = {}
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as native:
        if native.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE' or native['fine'].shape != (7, 400, 400, 400):
            raise ValueError('incorrect native total field')
        for i in CFG['train']:
            check()
            cache[i] = load_field(native, records[i])
            event('PREPARING', cached_fields=len(cache))
    return cache, records, labels, pi


def check_logp(logp):
    if not bool(torch.isfinite(logp).all()) or abs(float(logp.exp().sum())-1) > 1e-4:
        raise ValueError('nonfinite or unnormalized location probability')


@torch.no_grad()
def symmetry_test(model, raw_cpu, centers, phase):
    model.eval()
    raw = raw_cpu.cuda()
    observer = raw.new_tensor(CFG['observer_center_cells'])
    labels = torch.as_tensor(centers, device=raw.device)
    reference = []
    for role in range(3):
        logp = model.log_prob(raw, observer, labels[:role], role)
        check_logp(logp)
        reference.append(logp.exp())
    rows = []
    for symmetry in range(48):
        check()
        value = transform_state(raw, symmetry)
        new_o = transform_positions(observer, symmetry, 128)
        new_c = transform_positions(labels, symmetry, 128, cell_indices=True)
        errors = []
        for role in range(3):
            logp = model.log_prob(value, new_o, new_c[:role], role)
            check_logp(logp)
            restored = transform(logp.exp()[None], inverse_symmetry(symmetry))[0]
            errors.append(float(abs(restored-reference[role]).sum()))
        rows.append(dict(symmetry=symmetry, distribution_L1=errors))
    maximum = np.max([row['distribution_L1'] for row in rows], axis=0).tolist()
    passed = all(x <= CFG['symmetry_distribution_L1_max'] for x in maximum)
    report = dict(phase=phase, fixture=CFG['symmetry_fixture'], rows=rows,
                  maximum_distribution_L1=maximum, passed=passed,
                  scope='whole native128 cube, all48 signed permutations, teacher-forced contexts')
    dump(f'symmetry_{phase}.json', report)
    event('SYMMETRY_PASS' if passed else 'SYMMETRY_FAILURE', phase=phase, maximum_L1=maximum)
    if not passed:
        raise ValueError('whole-network symmetry implementation failure, no science verdict')
    return report


def learn(model, cache, labels):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'],
        weight_decay=CFG['weight_decay'], foreach=False)
    rng = np.random.default_rng(CFG['shuffle_seed'])
    order, history = np.asarray(CFG['train']), []
    started = time.monotonic()
    try:
        for update in range(1, CFG['joint_updates']+1):
            check()
            if time.monotonic()-started >= CFG['learning_seconds']:
                break
            if (update-1) % len(order) == 0:
                rng.shuffle(order)
            index = int(order[(update-1) % len(order)])
            raw = cache[index].cuda()
            centers = torch.as_tensor(labels[index], device=raw.device)
            optimizer.zero_grad(set_to_none=True)
            tick, terms = time.monotonic(), []
            for role in range(3):
                logp = model.log_prob(raw, CFG['observer_center_cells'], centers[:role], role)
                loss = -logp[tuple(centers[role])]/3
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite training loss')
                terms.append(float(loss.detach())*3)
                loss.backward()  # sequential graphs; one optimizer step per joint tuple
                del logp, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            RESULT['updates'] = update
            history.append(dict(update=update, fixture=index, role_negative_logp=terms,
                mean_negative_logp=sum(terms)/3, gradient_norm=float(norm), seconds=time.monotonic()-tick))
            del raw, centers
            if update == CFG['operational_updates']:
                elapsed = time.monotonic()-started
                dump('operational.json', dict(updates=update, mean_joint_seconds=elapsed/update,
                    estimated_learning_seconds=elapsed/update*CFG['joint_updates'], memory=memory(),
                    action='timing only; unchanged model, budget and ongoing fit'))
            if update == 1 or update == CFG['operational_updates'] or update % CFG['history_every'] == 0:
                check()
                dump('history.json', history)
                event('TRAINING', updates=update, target=CFG['joint_updates'], last=history[-1], memory=memory())
    finally:
        dump('history.json', history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=CFG,
            updates=RESULT['updates'], source_commit=os.environ['EXPECTED_COMMIT'],
            torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
            shuffle_state=rng.bit_generator.state, current_order=order.tolist(),
            status='SAVED_STATE_NOT_SCIENTIFIC_ACCEPTANCE'), OUT / 'checkpoint_final.pt')
    RESULT.update(training_seconds=time.monotonic()-started,
        training_complete=RESULT['updates'] == CFG['joint_updates'],
        training_exposures=dict(Counter(row['fixture'] for row in history)))
    return model.eval()


def plot_draws(index, raw, samples, truth):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    density = np.log1p(raw[0].numpy().sum(2)).T
    for role, ax in enumerate(axes):
        ax.imshow(density, origin='lower', cmap='Greys', extent=(0, 24, 0, 24))
        xy = (samples[:, role, :2]+.5)*CFG['dx_cMpc_h']
        ax.scatter(xy[:, 0], xy[:, 1], s=14, alpha=.4, label='autonomous draws')
        native = (truth[role, :2]+.5)*CFG['dx_cMpc_h']
        ax.scatter(*native, marker='x', c='red', s=55, label='native center cell')
        ax.set(xlabel='x [cMpc/h]', ylabel='y [cMpc/h]', title=ROLES[role])
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Development {index}: native density background, NOT reconstructed mass')
    fig.savefig(OUT / f'locations_{index}.png', dpi=140)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, cache, records, labels, pi):
    model.eval()
    rollouts, rows = [], []
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as native:
        # Freeze complete nonoracle draws BEFORE any comparison to their native targets.
        for index in CFG['development']:
            check()
            raw = load_field(native, records[index])
            draws, logq = sample_triples(model, raw.cuda(), CFG['observer_center_cells'],
                CFG['rollout_count_per_development'],
                torch.Generator(device='cuda').manual_seed(CFG['rollout_seed']+index), check)
            dump(f'samples_{index}.json', dict(fixture=index, samples_cells=draws.tolist(),
                joint_log_prob=logq, native_parents_used=False, catalogue_candidates_used=False,
                sampled_parents_replaced=False, failures=0,
                scope='q(center cells|native F,O,E_train); not observed posterior or physical halo certification'))
            # Samples are immutable now; native centers enter only this descriptive comparison.
            xyz = (draws+.5)*CFG['dx_cMpc_h']
            distance = np.linalg.norm((draws-labels[index])*CFG['dx_cMpc_h'], axis=2)
            rollouts.append(dict(fixture=index, count=len(draws), failures=0,
                unique_triples=len(np.unique(draws.reshape(len(draws), -1), axis=0)),
                M31_M33_same_cell_fraction=float(np.mean(np.all(draws[:, 1] == draws[:, 2], axis=1))),
                mean_position_cMpc_h=xyz.mean(0).tolist(), std_position_cMpc_h=xyz.std(0).tolist(),
                native_cell_distance_median_cMpc_h=np.median(distance, axis=0).tolist(),
                native_cell_distance_16_84_cMpc_h=np.quantile(distance, [.16, .84], axis=0).tolist(),
                uncertainty='model spread on reused development fields, NOT calibrated coverage'))
            dump('rollouts.json', rollouts)
            plot_draws(index, raw, draws, labels[index])
            del raw
            event('ROLLOUT_FROZEN', fixture=index, draws=len(draws))
        for index in CFG['train']+CFG['development']:
            check()
            raw = (cache[index] if index in cache else load_field(native, records[index])).cuda()
            centers = torch.as_tensor(labels[index], device=raw.device)
            scores = {key: [] for key in ('model', 'geometry', 'density', 'same_cell')}
            for role in range(3):
                logp = model.log_prob(raw, CFG['observer_center_cells'], centers[:role], role)
                _, geometry = location_features(raw, CFG['observer_center_cells'], centers[:role], role)
                references = reference_log_probs(raw, geometry, role, centers[:role], pi)
                for key, distribution in zip(scores, (logp, *references)):
                    check_logp(distribution)
                    scores[key].append(float(distribution[tuple(centers[role])]))
            rows.append(dict(fixture=index, split='train' if index in cache else 'development',
                role_logp=scores, joint_logp={k: sum(v) for k, v in scores.items()},
                teacher_forced_proper_joint_score=True, autonomous_detection_claimed=False))
            dump('evaluation.json', rows)
            del raw, centers, logp, geometry, references, distribution
    dev = [r for r in rows if r['split'] == 'development']
    means = {key: np.mean([r['role_logp'][key] for r in dev], axis=0).tolist()
             for key in ('model', 'geometry', 'density', 'same_cell')}
    joint = {key: sum(value) for key, value in means.items()}
    criteria = dict(joint_beats_geometry=joint['model'] > joint['geometry'],
        joint_beats_density=joint['model'] > joint['density'],
        M33_beats_geometry=means['model'][2] > means['geometry'][2],
        M33_beats_density=means['model'][2] > means['density'][2],
        M33_beats_same_cell=means['model'][2] > means['same_cell'][2])
    RESULT.update(development_role_mean_logp=means, development_joint_mean_logp=joint,
        criteria=criteria, rollout_completed=True, evaluation_complete=True)
    RESULT['status'] = ('INCONCLUSIVE_LEARNING_BUDGET' if not RESULT['training_complete'] else
        'PASS_ROLE_LOCATION_FEASIBILITY_ONLY' if all(criteria.values()) else 'NO_GO_ROLE_LOCATION_AT_FIXED_BUDGET')


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('Slurm source-pinned execution required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    torch.set_num_threads(2)
    if CFG['dx_cMpc_h'] != .1875 or tuple(CFG['geometry_widths_cMpc_h']) != WIDTHS or CFG['joint_updates'] % len(CFG['train']):
        raise ValueError('fixed spatial convention or complete-cycle budget changed')
    RESULT.update(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'])
    dump('request.json', dict(**CFG, job_id=RESULT['job_id'], source_commit=RESULT['source_commit'],
        gpu=torch.cuda.get_device_name(), precision=configure_precision(True)))
    event('TESTING')
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_role_locations',
        'test_cf4_spatial_diffusion.SpatialDiffusionTests.test_signed_symmetries_and_coarse_restriction'])
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    dump('tests.json', dict(tests=tested.testsRun, errors=len(tested.errors), failures=len(tested.failures)))
    if not tested.wasSuccessful():
        raise ValueError('focused numerical regression failed')
    cache, records, labels, pi = prepare()
    torch.manual_seed(CFG['seed'])
    model = RoleLocationNet().cuda().eval()
    if sum(p.numel() for p in model.parameters()) != CFG['parameters']:
        raise ValueError('model differs from approved static sizing')
    fixture = CFG['symmetry_fixture']
    RESULT['symmetry_before'] = symmetry_test(model, cache[fixture], labels[fixture], 'before')['maximum_distribution_L1']
    learn(model, cache, labels)
    RESULT['symmetry_after'] = symmetry_test(model, cache[fixture], labels[fixture], 'after')['maximum_distribution_L1']
    evaluate(model, cache, records, labels, pi)
    check()
    RESULT.update(application_seconds=time.monotonic()-START, memory=memory(),
        next_step='Report endpoint and wait; no automatic location extension, observed-data posterior or IC')
    dump('result.json', RESULT)
    event(RESULT['status'], updates=RESULT['updates'], criteria=RESULT['criteria'])


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if CREATED:
            RESULT.update(status='INCONCLUSIVE_TECHNICAL_OR_BUDGET_FAILURE',
                error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                application_seconds=time.monotonic()-START)
            dump('result.json', RESULT)
            event(RESULT['status'], updates=RESULT['updates'], error=RESULT['error'])
        raise
