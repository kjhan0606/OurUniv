"""One E-conditional field fit and fixed generated-state endpoint, Slurm only."""
import copy
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

from cf4_stable_field import StableField, conditioning, loss, noisy_record, sample
from cf4_spatial_diffusion import pack, unpack, augment
from cf4_continuous_matter import restrict, check_realizable
from cf4_role_locations import transform_positions, sample_triples
from cf4_population_locations import PopulationRoleModel, DX, OBSERVER
from cf4_position_link import features_torch, observation_kernel, log_likelihood
from cf4_bundle_c_continuous import read_periodic_patch
from cf4_bundle_c_flow_pilot import metrics, compare
from cf4_conditional_split_flow import configure_precision
from cf4_split_moments import state

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
CFG = json.loads(Path('config/cf4_stable_field_v1.json').read_text())
OUT = ROOT/CFG['output_name']
START = time.monotonic()
CREATED = False
OBSERVER_FULL = np.full(3, 44.)*DX
RESULT = dict(status='NOT_STARTED', updates=0, E_conditional=True, observed_LG_posterior=False,
    physical_halo_catalogue=False, coarse_CF4_prior=False)


def dump(name, value):
    with (OUT/name).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def check():
    if time.monotonic()-START > CFG['application_seconds']:
        raise TimeoutError('total application3h50 includes preparation, learning and evaluation')
    usage = dict(host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
    if any(value > CFG[name] for name, value in usage.items()):
        raise MemoryError(f'fixed sizing exceeded: {usage}')
    return usage


def event(status, **extra):
    value = dict(status=status, seconds=time.monotonic()-START, job_id=os.environ['SLURM_JOB_ID'], **extra)
    dump('status.json', value)
    print(json.dumps(value), flush=True)


def cases_and_slab():
    archive = ROOT/'population_locations_v1'
    sets = json.loads((archive/'cases.json').read_text())
    selected = json.loads((archive/'population.json').read_text())['rollout_observers_frozen_before_fit']
    test_by_id = {c['mw']: c for c in sets['test']}
    train, test = sets['train'], [test_by_id[sid] for sid in selected]
    for cases, lo, hi in ((train, 0, 184), (test, 272, 400)):
        for case in cases:
            start = np.asarray(case['origin'])-8
            if start[0] < lo or start[0]+80 > hi or np.any(start % 8):
                raise ValueError('buffered80 source support leaves frozen spatial slab')
    with h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as source:
        if source.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE':
            raise ValueError('incorrect full-matter source')
        slab = source['fine'][:, :184, :, :]
    dump('population.json', dict(E_conditional=True, train_observers=len(train), test_observers=[c['mw'] for c in test],
        weighting='uniform archived eligible MW observer, not uniform unique volume',
        duplicates='overlapping and identical fields can recur under observer weighting; not independent universes',
        source_scope='native80 buffered cubes; within-fit disjoint x slabs; historically reused test fields'))
    event('DATA_READY', training_observers=len(train), test_observers=len(test), memory=check())
    return train, test, slab


def cube(source, case):
    return read_periodic_patch(source, (np.asarray(case['origin'])-8)%400, 80)


def normalize(slab, cases):
    rng = np.random.default_rng(CFG['normalization_seed'])
    chosen = rng.choice(len(cases), CFG['normalization_observers'], replace=False)
    sums, squares, count = np.zeros(49), np.zeros(49), np.zeros(49)
    identities, errors = [], []
    for index in chosen:
        check()
        case = cases[int(index)]
        fine = augment(cube(slab, case), int(rng.integers(48)))
        identities.append(case['mw'])
        for level in range(3):
            value = restrict(fine, 2**level) if level else fine
            z, codes, parent = pack(value)
            restored, info = unpack(z, codes, parent)
            if np.any(info['corrected_per_node_channel']):
                raise ValueError('native canonical branch mismatch')
            error = float(np.max(abs(restored-value)/np.maximum(abs(value), 1.)))
            # Native near-cold cancellation is judged with channel-scale tolerance.
            scaled = np.max(abs(restored-value), axis=(1, 2, 3))/np.maximum(np.max(abs(value), axis=(1, 2, 3)), 1.)
            if float(scaled.max()) > 1e-8:
                raise ValueError('native seven-moment chart roundtrip mismatch')
            errors.append(dict(mw=case['mw'], level=level, channel_scaled=float(scaled.max()), cell_scaled=error))
            sites = rng.choice(z[0].size, min(4096, z[0].size), replace=False)
            values = z.reshape(49, -1)[:, sites]
            active = codes.reshape(49, -1)[:, sites] == 2
            sums += (values*active).sum(1)
            squares += (values*values*active).sum(1)
            count += active.sum(1)
        if len(identities) % 16 == 0:
            event('NORMALIZING', observers=len(identities), total=len(chosen))
    if np.any(count == 0):
        raise ValueError('missing active-coordinate normalization support')
    location = sums/count
    spread = np.sqrt(np.maximum(squares/count-location**2, 0))
    if np.any(spread <= 0) or not np.isfinite(spread).all():
        raise ValueError('invalid/constant normalization, no invented spread')
    dump('representation.json', dict(E_conditional=True, observer_ids=identities, location=location.tolist(),
        spread=spread.tolist(), active_count=count.tolist(), native_roundtrip=errors))
    return location, spread


def inputs(field, level, observer, location, spread):
    value = restrict(field, 2**level) if level else field
    z, codes, parent = pack(value)
    clean = np.where(codes == 2, (z-location[:, None, None, None])/spread[:, None, None, None], 0.)
    tensors = [torch.from_numpy(np.ascontiguousarray(a))[None].cuda() for a in
        (clean.astype(np.float32), codes.astype(np.int64), conditioning(parent, DX*2**level, observer))]
    return tensors


def learn(slab, cases, location, spread):
    torch.manual_seed(CFG['training_seed'])
    rng = np.random.default_rng(CFG['training_seed'])
    model = StableField().cuda().train()
    if sum(p.numel() for p in model.parameters()) != CFG['parameters']:
        raise ValueError('submitted exact parameter count mismatch')
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'], foreach=False)
    history, visits, order = [], np.zeros(len(cases), np.int64), rng.permutation(len(cases))
    started = time.monotonic()
    def save(step):
        torch.save(dict(E_conditional=True, model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
            step=step, location=location, spread=spread, config=CFG, numpy_rng=rng.bit_generator.state,
            source_commit=os.environ['EXPECTED_COMMIT'], torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
            OUT/f'checkpoint_{step:05d}.pt')
        dump('history.json', history)
        dump('exposure.json', {c['mw']: int(n) for c, n in zip(cases, visits)})
    last_saved = -1
    try:
        for step in range(1, CFG['steps']+1):
            check()
            if time.monotonic()-started >= CFG['learning_seconds']:
                break
            offset = (step-1) % len(cases)
            if offset == 0 and step > 1:
                order = rng.permutation(len(cases))
            index = int(order[offset])
            level, symmetry = (step-1) % 3, int(rng.integers(48))
            tick = time.monotonic()
            field = augment(cube(slab, cases[index]), symmetry)
            observer = transform_positions(torch.tensor([44.]*3), symmetry, 80).numpy()*DX
            record = inputs(field, level, observer, location, spread)
            del field
            optimizer.zero_grad(set_to_none=True)
            continuous, categorical, report = loss(model, *record, torch.tensor([float(level)], device='cuda'), CFG['diffusion_steps'])
            if not bool(torch.isfinite(continuous)) or not bool(torch.isfinite(categorical)):
                raise ValueError('nonfinite training loss')
            continuous.backward()
            categorical.backward()
            norms = [float(torch.nn.utils.clip_grad_norm_(branch.parameters(), CFG['gradient_clip_per_branch'], error_if_nonfinite=True))
                for branch in (model.continuous, model.categorical)]
            optimizer.step()
            with torch.no_grad():
                for target, parameter in zip(ema.parameters(), model.parameters()):
                    target.lerp_(parameter, 1-CFG['ema_decay'])
            torch.cuda.synchronize()
            RESULT['updates'] = step
            visits[index] += 1
            history.append(dict(step=step, level=level, mw=cases[index]['mw'], seconds=time.monotonic()-tick,
                branch_gradient_norms=norms, **report))
            if step == 1 or step % 200 == 0:
                dump('history.json', history)
                event('TRAINING', step=step, target=CFG['steps'], last=history[-1], memory=check())
            if step % CFG['checkpoint_every'] == 0:
                save(step)
                last_saved = step
            del continuous, categorical, record
    finally:
        if last_saved != RESULT['updates']:
            save(RESULT['updates'])
    RESULT.update(training_complete=RESULT['updates'] == CFG['steps'], training_seconds=time.monotonic()-started)
    return ema


@torch.no_grad()
def denoising(model, cases, location, spread):
    torch.manual_seed(CFG['denoising_seed'])
    rows = []
    with h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as source:
        for case in cases:
            native = cube(source['fine'], case)
            for level in range(3):
                clean, codes, coarse = inputs(native, level, OBSERVER_FULL, location, spread)
                for t in (10, 50, 100):
                    check()
                    times = torch.tensor([t], device='cuda')
                    noisy, masked_codes, target, mask = noisy_record(clean, codes, times, CFG['diffusion_steps'])
                    v, logits = model(noisy, masked_codes, coarse, times, torch.tensor([float(level)], device='cuda'))
                    mse, zero = float((v-target).square().mean()), float(target.square().mean())
                    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), codes.flatten(0, 1), reduction='none').reshape_as(clean)
                    rows.append(dict(mw=case['mw'], level=level, t=t, v_MSE=mse, zero_v_MSE=zero,
                        ratio=mse/zero, masked_category_NLL=float(ce[mask].mean()), uniform_category_NLL=float(np.log(4)),
                        v_RMS=float(v.square().mean().sqrt())))
            dump('denoising.json', rows)
            event('DENOISING_SCORED', observers=len(rows)//9, total=len(cases))
    return rows


def draw_field(model, parent, location, spread):
    current, refinements = parent, []
    for level in (2, 1, 0):
        check()
        ctx = torch.from_numpy(conditioning(current, DX*2**level, OBSERVER_FULL))[None].cuda()
        z, codes, trace = sample(model, ctx, DX*2**level, location, spread, CFG['diffusion_steps'], check)
        current, report = unpack(z, codes, current)
        refinements.append(dict(level=level, trace=trace, decoder=report))
    check_realizable(current)
    error = np.max(abs(restrict(current, 8)-parent), axis=(1, 2, 3))/np.maximum(np.max(abs(parent), axis=(1, 2, 3)), 1.)
    if not np.isfinite(current).all() or float(error.max()) > 1e-8:
        raise ValueError('invalid full80 field or failed parent conservation')
    return current, refinements


def plot_field(native, generated, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    limits = []
    for row, (label, field) in enumerate((('native', native), ('E-conditional draw', generated))):
        velocity, variance = state(field)
        column_mass = field[0].sum(2)
        bulk = np.divide(field[1].sum(2), column_mass, out=np.zeros_like(column_mass), where=column_mass > 0)
        dispersion = np.divide((field[0]*variance.mean(0)).sum(2), column_mass,
            out=np.zeros_like(column_mass), where=column_mass > 0)
        maps = [np.log1p(column_mass), bulk, np.sqrt(dispersion)]
        for column, (value, title) in enumerate(zip(maps, ('log projected mass', 'mass-weighted mean vx [km/s]', 'physical sigma [km/s]'))):
            if row == 0:
                limits.append((float(value.min()), float(value.max())))
            artist = axes[row, column].imshow(value.T, origin='lower', extent=(0, 12, 0, 12), cmap='viridis',
                vmin=limits[column][0], vmax=limits[column][1])
            axes[row, column].set_title(f'{label}: {title}')
            fig.colorbar(artist, ax=axes[row, column])
    fig.suptitle('Native coarse-conditioned draw; shared native color limits — NOT observed LG reconstruction')
    fig.savefig(OUT/name, dpi=110)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, cases, location, spread):
    torch.manual_seed(CFG['sample_seed'])
    location_model = PopulationRoleModel().double().cuda().eval()
    saved = torch.load(ROOT/'population_locations_v1/checkpoint_final.pt', map_location='cpu', weights_only=False)
    location_model.load_state_dict(saved['model'], strict=True)
    location_model.alpha = json.loads((ROOT/'population_locations_v1/result.json').read_text())['alpha']
    location_model.requires_grad_(False)
    observations = json.loads((ROOT/'position_link_v2/observations.json').read_text())
    actual = observations['actual']
    kernel = observation_kernel(actual['point_w'], np.asarray(OBSERVER)*DX, actual['directions'], actual['modulus'], observations['covariance'])
    rows, ensembles, role_rows = [], [], []
    with h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as source, h5py.File(OUT/'first_draw_fields.h5', 'w') as output:
        output.attrs.update(status='INCOMPLETE_E_CONDITIONAL_NATIVE_PARENT_NOT_OBSERVED_LG', E_conditional=True)
        for index, case in enumerate(cases):
            check()
            native = cube(source['fine'], case)
            parent = restrict(native, 8)
            core = native[:, 8:-8, 8:-8, 8:-8]
            coarse = parent[:, 1:-1, 1:-1, 1:-1]
            reference = metrics(core, DX, coarse, 8)
            local = []
            for draw in range(CFG['draws_per_observer']):
                seed = CFG['sample_seed']+100*index+draw
                torch.manual_seed(seed)
                row = dict(mw=case['mw'], draw=draw, seed=seed, E_conditional=True, valid=False)
                try:
                    generated, refinements = draw_field(model, parent, location, spread)
                    inner = generated[:, 8:-8, 8:-8, 8:-8]
                    measured = metrics(inner, DX, coarse, 8)
                    if not all(np.isfinite(np.asarray(v)).all() for v in measured.values()) or max(measured['coarse_error']) > 1e-8:
                        raise ValueError('nonfinite inner64 metrics or failed conservation')
                    comparison = compare(measured, reference)
                    row.update(valid=True, refinements=refinements, comparison=comparison)
                    local.append(measured)
                    # ALL fine feature support comes from the generated80 field.
                    features = features_torch(torch.from_numpy(generated).cuda())[:, 4:-4, 4:-4, 4:-4]
                    logL, _ = log_likelihood(location_model, features, kernel)
                    row['actual_position_log_L_diagnostic'] = float(logL)
                    if draw == 0:
                        output.create_dataset(f'mw_{case["mw"]}/moments80', data=generated, compression='gzip', compression_opts=1)
                        samples, logp = sample_triples(location_model, features, OBSERVER, CFG['role_triples_per_first_draw'],
                            torch.Generator(device='cuda').manual_seed(seed+500000), check)
                        role_rows.append(dict(mw=case['mw'], cells=samples.tolist(), logp=logp, native_parent_roles_used=False,
                            labels='Probabilistic cells, NOT physical halo catalogue/mass/COM'))
                        dump('role_samples.json', role_rows)
                        plot_field(core, inner, f'field_{case["mw"]}.png')
                    del generated, features
                except ValueError as error:
                    if row['valid']:
                        raise  # readout/plot failure is technical, not an invalid physical draw
                    row.update(error=str(error), valid=False)
                rows.append(row)
                dump('draws.json', rows)
                event('DRAW_FINISHED', mw=case['mw'], draw=draw, valid=row['valid'], completed=len(rows), total=16)
            if len(local) == CFG['draws_per_observer']:
                mean = {key: np.mean([r[key] for r in local], axis=0).tolist() for key in reference}
                ensembles.append(dict(mw=case['mw'], **compare(mean, reference)))
            else:
                ensembles.append(dict(mw=case['mw'], development_pass=False, reason='one or more invalid draws'))
            dump('ensembles.json', ensembles)
        output.attrs['status'] = 'COMPLETE_E_CONDITIONAL_NATIVE_PARENT_NOT_OBSERVED_LG'
    return rows, ensembles


def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm allocation required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    torch.set_num_threads(2)
    RESULT.update(job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'])
    dump('request.json', dict(**CFG, **{k: RESULT[k] for k in ('job_id', 'source_commit')},
        E_conditional=True, gpu=torch.cuda.get_device_name(), precision=configure_precision(True)))
    tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_stable_field']))
    dump('tests.json', dict(tests=tested.testsRun, failures=len(tested.failures), errors=len(tested.errors)))
    if not tested.wasSuccessful():
        raise ValueError('stable field focused tests failed')
    train, test, slab = cases_and_slab()
    location, spread = normalize(slab, train)
    model = learn(slab, train, location, spread)
    del slab
    dump('result.json', dict(**RESULT, status='EVALUATING'))
    denoising(model, test, location, spread)
    draws, ensembles = evaluate(model, test, location, spread)
    valid = sum(r['valid'] for r in draws)
    individual = sum(r.get('comparison', {}).get('development_pass', False) for r in draws if r['valid'])
    passed = sum(r['development_pass'] for r in ensembles)
    RESULT.update(status=('INCONCLUSIVE_LEARNING_BUDGET' if not RESULT['training_complete'] else
        'PASS_E_CONDITIONAL_FIELD_GENERATOR_FEASIBILITY_ONLY' if valid == 16 and passed == 8 else
        'NO_GO_E_CONDITIONAL_FIELD_GENERATOR'), valid_draws=valid, total_draws=16,
        individual_morphology_pass=individual, ensemble_morphology_pass=passed,
        evaluation_complete=True, application_seconds=time.monotonic()-START, memory=check(),
        next='Driver reviews evidence; no automatic refit, unconditional-prior promotion, observed posterior or IC')
    dump('result.json', RESULT)
    event(RESULT['status'], valid_draws=valid, individual_pass=individual, ensemble_pass=passed)


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if CREATED:
            RESULT.update(status='INCONCLUSIVE_TECHNICAL_OR_BUDGET_FAILURE', error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(), application_seconds=time.monotonic()-START)
            dump('result.json', RESULT)
        raise
