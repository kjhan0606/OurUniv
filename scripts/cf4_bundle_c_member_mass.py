"""One approved native member-mass learner and fixed evaluation, no follow-up."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from cf4_bundle_c_continuous import read_periodic_patch
from cf4_bundle_c_spatial_model import overlaps
from cf4_conditional_split_flow import configure_precision
from cf4_member_mass_readout import MemberMassNet, features, transform, map_loss, metrics, ROLES

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
CFG = json.loads(Path('config/cf4_member_mass_pilot_v1.json').read_text())
OUT = ROOT / CFG['output_name']
START = time.monotonic()
CREATED = False
RESULT = dict(status='NOT_STARTED', updates=0, roles=ROLES, observed_LG_posterior=False,
              field_prior_training=False, member_velocities_inferred=False)


def dump(name, value):
    with (OUT / name).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def memory():
    return dict(host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)


def check():
    if time.monotonic()-START > CFG['application_seconds']:
        raise TimeoutError('application cap, not a completed scientific evaluation')
    measured = memory()
    if measured['host_peak_GiB'] > CFG['host_peak_GiB'] or measured['gpu_reserved_peak_GiB'] > CFG['gpu_reserved_peak_GiB']:
        raise MemoryError(f'approved engineering sizing exceeded: {measured}')


def event(status, **kwargs):
    dump('status.json', dict(status=status, job_id=os.environ['SLURM_JOB_ID'],
         seconds=time.monotonic()-START, **kwargs))
    print(json.dumps(dict(status=status, **kwargs)), flush=True)


def load_case(native, source, index):
    record = json.loads(source[f'patches/{index}'].attrs['record_json'])
    origin = np.rint(np.array(record['patch_lower_cMpc_h'])/.1875).astype(int)
    total = read_periodic_patch(native['fine'], origin, 128)
    truth = np.zeros((4, 128, 128, 128), np.float64)
    truth[3] = source[f'patches/{index}/remainder'][0]
    for role, sid in enumerate(record['ids']):
        halo = source[f'halos/{sid}']
        cells = (halo['global_cell'][:]-origin) % 400
        if np.any(cells >= 128):
            raise ValueError('native target tail outside fixed full patch')
        truth[role].reshape(-1)[np.ravel_multi_index(cells.T, (128,)*3)] = halo['moments'][0]
    if not np.isfinite(total).all() or not np.isfinite(truth).all() or np.any(truth < 0):
        raise ValueError('invalid native target or total')
    if np.any(truth.sum((1, 2, 3)) <= 0):
        raise ValueError(f'INCONCLUSIVE_TARGET_UNAVAILABLE fixture{index}')
    error = float(abs(truth.sum(0)-total[0]).max()/max(total[0].max(), 1))
    if error > 1e-9:
        raise ValueError('cached component mass does not reconstruct native total')
    inp = features(total, CFG['observer_center_cells'])
    unit = CFG['mass_unit_Msun']
    return dict(index=index, record=record, input=torch.from_numpy(inp),
        total=torch.from_numpy((total[:1]/unit).astype(np.float32)),
        truth=torch.from_numpy((truth/unit).astype(np.float32)), native_error=error)


def prepare():
    cases = []
    baseline = np.zeros((4, 128, 128, 128), np.float64)
    means = np.zeros(7)
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as native, \
         h5py.File(ROOT / 'spatial_calibration_v1/spatial_components.h5', 'r') as source:
        if source.attrs['status'] != 'NATIVE_JOINT_PROFILES_AND_REMAINDER_NOT_CF4_LG_POSTERIOR':
            raise ValueError('incorrect native source status')
        records = {i: json.loads(source[f'patches/{i}'].attrs['record_json']) for i in CFG['train']+CFG['development']}
        for i in CFG['train']:
            for j in CFG['development']:
                if overlaps(records[i]['patch_lower_cMpc_h'], records[j]['patch_lower_cMpc_h']):
                    raise ValueError('fixed training/development voxel overlap')
        counts = Counter(s for i in CFG['train'] for s in records[i]['ids'])
        dump('source.json', dict(records=records, native_ids_for_targets_only=True,
             training_shared_ids={str(k): v for k, v in counts.items() if v > 1},
             train_development_shared_ids=sorted(set(counts) & {s for i in CFG['development'] for s in records[i]['ids']}),
             training_overlapping_pairs=[[i, j] for k, i in enumerate(CFG['train']) for j in CFG['train'][k+1:]
                 if overlaps(records[i]['patch_lower_cMpc_h'], records[j]['patch_lower_cMpc_h'])],
             independent_validation=False))
        for i in CFG['train']:
            check()
            case = load_case(native, source, i)
            t, m = case['truth'].numpy(), case['total'].numpy()
            fractions = np.divide(t, m, out=np.zeros_like(t), where=m > 0)
            fractions[3][m[0] == 0] = 1
            baseline += fractions/len(CFG['train'])
            means += case['input'][:7].numpy().mean((1, 2, 3), dtype=np.float64)/len(CFG['train'])
            cases.append(case)
            event('PREPARING', fixtures=len(cases))
    baseline = baseline.astype(np.float32)
    dump('input_semantics.json', dict(feature_means=means.tolist(), observer_center_cells=CFG['observer_center_cells'],
         observer='coarse1.5-cell center, not a true subcell galaxy position',
         field_channels='log1p mass/(1e11 dx^3), asinh mean velocity/300, log1p directional sigma/100',
         coordinate_channels='relative to observer coarse-cell center in half-patch units',
         target='native disjoint SUBFIND-bound mass plus remainder; no M200c conversion'))
    with h5py.File(OUT / 'baseline.h5', 'x') as out:
        out.create_dataset('fractions', data=baseline)
    check()
    return cases, baseline, means


def learn(cases):
    torch.manual_seed(CFG['seed'])
    rng = np.random.default_rng(CFG['seed'])
    model = MemberMassNet().cuda()
    count = sum(p.numel() for p in model.parameters())
    if count != CFG['parameters'] or count > 1000000:
        raise ValueError('model size differs from submitted static sizing')
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'], foreach=False)
    history = []
    started = time.monotonic()
    order = np.arange(len(cases))
    try:
        for step in range(1, CFG['steps']+1):
            check()
            if time.monotonic()-started >= CFG['learning_seconds']:
                break
            if (step-1) % len(cases) == 0:
                rng.shuffle(order)
            case = cases[order[(step-1) % len(cases)]]
            symmetry = int(rng.integers(48))
            x = transform(case['input'].cuda(), symmetry, True)[None]
            mass = transform(case['total'].cuda(), symmetry)[None]
            target = transform(case['truth'].cuda(), symmetry)[None]
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x)*mass
            loss, errors = map_loss(predicted, target)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite supervised loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            RESULT['updates'] = step
            history.append(dict(step=step, fixture=case['index'], symmetry=symmetry, loss=float(loss.detach()),
                role_L1=errors[0].detach().cpu().tolist(), gradient_norm=float(norm), seconds=time.monotonic()-tick))
            del x, mass, target, predicted, loss, errors
            if step == 1 or step % 50 == 0:
                check()
                dump('history.json', history)
                event('TRAINING', updates=step, target=CFG['steps'], last=history[-1], memory=memory())
    finally:
        dump('history.json', history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=CFG,
             updates=RESULT['updates'], source_commit=os.environ['EXPECTED_COMMIT'],
             status='FINAL_SAVED_STATE_NOT_SCIENTIFIC_ACCEPTANCE'), OUT / 'checkpoint_final.pt')
    RESULT['training_seconds'] = time.monotonic()-started
    RESULT['training_complete'] = RESULT['updates'] == CFG['steps']
    return model.eval(), history


def plot_case(index, native, predicted, baseline):
    arrays = [v.sum(axis=3) for v in (native, predicted, baseline)]
    fig, axes = plt.subplots(4, 3, figsize=(11, 12), squeeze=False)
    for r in range(4):
        vmax = max(float(a[r].max()) for a in arrays)
        norm = LogNorm(vmin=vmax*1e-6, vmax=vmax) if vmax > 0 else None
        for c, a in enumerate(arrays):
            axes[r, c].imshow(np.ma.masked_less_equal(a[r].T, 0), origin='lower', extent=[0,24,0,24], norm=norm)
            axes[r, c].set_title(f'{ROLES[r]} / {("native", "predicted", "geometry baseline")[c]}')
        fig.colorbar(axes[r, 0].images[0], ax=list(axes[r]), label='projected cell mass [Msun]', fraction=.025)
    fig.suptitle(f'Fixture {index}: native simulated field, NOT observed LG; same row color scale\nLog color lower bound is display only; stored masses unchanged')
    fig.savefig(OUT / f'components_{index}.png', dpi=130)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, cases, baseline, means):
    rows = []
    with h5py.File(OUT / 'development_mass_maps.h5', 'x') as out, \
         h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as native, \
         h5py.File(ROOT / 'spatial_calibration_v1/spatial_components.h5', 'r') as source:
        out.attrs.update(status='INCOMPLETE_NATIVE_MASS_READOUT', roles=json.dumps(ROLES), dx_cMpc_h=.1875)
        for split, indices in [('training', CFG['train']), ('development', CFG['development'])]:
            for k, i in enumerate(indices):
                check()
                case = cases[k] if split == 'training' else load_case(native, source, i)
                unit = CFG['mass_unit_Msun']
                total = case['total'][0].numpy().astype(np.float64)*unit
                truth = case['truth'].numpy().astype(np.float64)*unit
                x = case['input'][None].cuda()
                prediction = model(x)[0].cpu().numpy().astype(np.float64)*total
                base = baseline.astype(np.float64)*total
                row = dict(split=split, fixture=i, predicted=metrics(prediction, truth, total),
                     baseline=metrics(base, truth, total),
                     baseline_zero_support_target_cells=[int(np.sum((base[r] == 0) & (truth[r] > 0))) for r in range(4)])
                if max(row['predicted']['conservation_relative_max'], row['baseline']['conservation_relative_max']) > 1e-6:
                    raise ValueError('allocation mass conservation failed')
                if split == 'development':
                    x[:, :7] = torch.as_tensor(means, device='cuda', dtype=x.dtype)[None, :, None, None, None]
                    ablated = model(x)[0].cpu().numpy().astype(np.float64)*total
                    row['mean_field_ablation_OOD'] = metrics(ablated, truth, total)
                    group = out.create_group(str(i))
                    for name, a in [('native', truth), ('predicted', prediction), ('baseline', base), ('mean_field_ablation', ablated)]:
                        group.create_dataset(name, data=a, compression='gzip', compression_opts=1)
                    plot_case(i, truth, prediction, base)
                    del ablated
                rows.append(row)
                dump('evaluation.json', rows)
                event('EVALUATING', split=split, fixture=i, finished=len(rows))
                del x, total, truth, prediction, base
        out.attrs['status'] = 'COMPLETE_NATIVE_COMPONENT_MASS_HYPOTHESES_NOT_OBSERVED_LG'
    def array(split, kind):
        return np.array([r[kind]['map_L1'] for r in rows if r['split'] == split])
    tp, tb = array('training', 'predicted'), array('training', 'baseline')
    dp, db = array('development', 'predicted'), array('development', 'baseline')
    checks = dict(training_each_role_20percent=(tp.mean(0) <= .8*tb.mean(0)).tolist(),
        development_M33_each_improves=(dp[:, 2] < db[:, 2]).tolist(),
        development_MW_M31_median_nonworsening=(np.median(dp[:, :2], axis=0) <= np.median(db[:, :2], axis=0)).tolist())
    success = all(all(v) for v in checks.values())
    RESULT.update(criteria=checks, training_mean_predicted_L1=tp.mean(0).tolist(), training_mean_baseline_L1=tb.mean(0).tolist(),
        status=('INCONCLUSIVE_BUDGET' if not RESULT['training_complete'] else
            'PASS_MASS_MAP_FEASIBILITY_ONLY' if success else 'NO_GO_MEMBER_MASS_READOUT'),
        caution='Fixed engineering criteria, not certified member detections or posterior. Compare also zero-mass L1=1 and mass/centroid errors. No member COM velocities.')


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Slurm required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    RESULT.update(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'])
    dump('request.json', CFG)
    torch.set_num_threads(2)
    dump('precision.json', configure_precision(True))
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_member_mass_readout',
        'test_cf4_spatial_diffusion.SpatialDiffusionTests.test_signed_symmetries_and_coarse_restriction'])
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    dump('tests.json', dict(tests=tested.testsRun, errors=len(tested.errors), failures=len(tested.failures)))
    if not tested.wasSuccessful():
        raise ValueError('focused regression failed before training')
    cases, baseline, means = prepare()
    model, history = learn(cases)
    dump('result.json', RESULT)
    evaluate(model, cases, baseline, means)
    fig, ax = plt.subplots()
    ax.plot([r['step'] for r in history], [r['loss'] for r in history])
    ax.set(xlabel='optimizer updates', ylabel='mean role-normalized map L1', yscale='log')
    fig.savefig(OUT / 'learning_curve.png', dpi=130)
    plt.close(fig)
    RESULT.update(seconds=time.monotonic()-START, memory=memory())
    dump('result.json', RESULT)
    event(RESULT['status'], updates=RESULT['updates'], criteria=RESULT['criteria'])


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if CREATED:
            status = 'INCONCLUSIVE_TARGET_UNAVAILABLE' if 'INCONCLUSIVE_TARGET_UNAVAILABLE' in str(exc) else 'INCOMPLETE_EXECUTION'
            RESULT.update(status=status, error=str(exc), traceback=traceback.format_exc(), memory=memory())
            dump('result.json', RESULT)
            event(status, error=str(exc))
        raise
