"""Frozen two-checkpoint signed-view evaluation; no learning or follow-up job."""
from collections import defaultdict
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
from cf4_member_mass_readout import MemberMassNet, transform, inverse_symmetry, metrics, ROLES
from cf4_spatial_diffusion import SYMMETRIES

ROOT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1')
OUT = ROOT / 'member_mass_diagnosis_v5'
FIXTURES = (0, 10, 15, 16, 17, 20)
SOURCES = (
    ('single5000', 'member_mass_single5000_v3', '787497f', '342086', 5000,
     'PASS_SINGLE_FIELD_LEARNING_ONLY'),
    ('multi18000', 'member_mass_multifield_v4', '20253e1', '342129', 18000,
     'NO_GO_MEMBER_MASS_READOUT'),
)
IDENTITY = SYMMETRIES.index(((0, 1, 2), (1, 1, 1)))
RTOL, ATOL, MATERIAL_SPREAD, MIN_ALTERNATIVES = 1e-4, 1e-5, .1, 2
START = time.monotonic()
CREATED = False


def dump(name, value):
    with (OUT / name).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def usage():
    return dict(host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                gpu_reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)


def check():
    if time.monotonic()-START > 780:
        raise TimeoutError('incomplete evaluation: application budget')
    measured = usage()
    if measured['host_peak_GiB'] > 5 or measured['gpu_reserved_peak_GiB'] > 20:
        raise MemoryError(f'incomplete evaluation: memory cap {measured}')


def event(status, **extra):
    value = dict(status=status, job_id=os.environ['SLURM_JOB_ID'],
                 seconds=time.monotonic()-START, **extra)
    dump('status.json', value)
    print(json.dumps(value), flush=True)


def selection_context(source, indices):
    records = {int(i): json.loads(source[f'patches/{i}'].attrs['record_json'])
               for i in source['patches']}
    with h5py.File(ROOT / 'lg_population_v1/population_and_marks.h5', 'r') as population:
        ids, split, branch = (population[k][:] for k in ('identities', 'heldout', 'branch'))
    rows = []
    for i in indices:
        record = records[i]
        mw, m31, m33 = record['ids']
        selected = record['source_row']
        np.testing.assert_array_equal(ids[selected], record['ids'])
        if split[selected] != record['heldout'] or branch[selected] != 0:
            raise ValueError('source selection row mismatch')
        eligible = ids[(split == record['heldout']) & (branch == 0) & (ids[:, 0] == mw)]
        thirds = eligible[eligible[:, 1] == m31]
        n31, n33 = len(np.unique(eligible[:, 1])), len(np.unique(thirds[:, 2]))
        rows.append(dict(fixture=i, ids=record['ids'], eligible_M31=n31,
            eligible_M33_given_chosen_M31=n33, eligible_triple_rows=len(eligible),
            chosen_M31_triple_rows=len(thirds),
            omitted_assignment_choice=bool(n31 >= MIN_ALTERNATIVES or n33 >= MIN_ALTERNATIVES)))
    grouped = defaultdict(list)
    for i, record in records.items():
        grouped[tuple(record['patch_lower_cMpc_h'])].append(i)
    collisions = [dict(origin=list(origin), fixtures=items,
        ids=[records[i]['ids'] for i in items]) for origin, items in grouped.items()
        if len({tuple(records[i]['ids']) for i in items}) > 1]
    return dict(rows=rows, all_source_patch_count=len(records),
        same_input_different_label_groups=collisions,
        interpretation='Source selection multiplicity, NOT posterior weights or an empirical error bound. '
        'Same input means same native origin and fixed coarse-observer channels. '
        'Absence of collisions in these records does not prove identifiability.')


def load_models():
    models, references = {}, {}
    for name, directory, commit, job, updates, status in SOURCES:
        path = ROOT / directory
        result = json.loads((path / 'result.json').read_text())
        saved = torch.load(path / 'checkpoint_final.pt', map_location='cpu', weights_only=True)
        if (saved['source_commit'], saved['updates']) != (commit, updates) or \
                (result['source_commit'], result['job_id'], result['updates'], result['status']) != (commit, job, updates, status):
            raise ValueError('incorrect frozen source')
        if saved['config']['observer_center_cells'] != [68, 68, 68] or saved['config']['mass_unit_Msun'] != 1e12:
            raise ValueError('frozen feature/mass units changed')
        model = MemberMassNet().cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        models[name] = model
        references[name] = ({0: result['screen']['predicted']} if name == 'single5000' else
            {r['fixture']: r['predicted'] for r in json.loads((path / 'evaluation.json').read_text())})
        del saved
    return models, references


def summarize(rows):
    summaries = []
    for name, *_ in SOURCES:
        for fixture in FIXTURES:
            group = [r for r in rows if r['checkpoint'] == name and r['fixture'] == fixture]
            if len(group) != 48:
                raise ValueError('incomplete view coverage')
            identity = next(r for r in group if r['symmetry'] == IDENTITY)
            value = dict(checkpoint=name, fixture=fixture, identity=identity)
            for key in ('map_L1', 'overlap', 'mass_ratio', 'identity_discrepancy_L1', 'identity_discrepancy_Msun'):
                a = np.array([r[key] for r in group])
                value[key] = dict(mean=a.mean(0).tolist(), minimum=a.min(0).tolist(), maximum=a.max(0).tolist())
            l1_spread = np.ptp([r['map_L1'] for r in group], axis=0)
            overlap_spread = np.ptp([r['overlap'] for r in group], axis=0)
            value.update(L1_range=l1_spread.tolist(), overlap_range=overlap_spread.tolist(),
                material_orientation_sensitivity=((l1_spread > MATERIAL_SPREAD) | (overlap_spread > MATERIAL_SPREAD)).tolist())
            summaries.append(value)
    return summaries


@torch.inference_mode()
def run():
    global CREATED
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('Slurm source-pinned numerical execution required')
    OUT.mkdir(exist_ok=False)
    CREATED = True
    torch.set_num_threads(2)
    dump('request.json', dict(fixtures=FIXTURES, sources=SOURCES, forward_passes=576,
        optimizer_updates=0, identity_symmetry=IDENTITY, identity_rtol=RTOL, identity_atol=ATOL,
        material_L1_or_overlap_range=MATERIAL_SPREAD, minimum_assignment_alternatives=MIN_ALTERNATIVES,
        roles=ROLES, mass_floor_applied=False, application_seconds=780,
        source_commit=os.environ['EXPECTED_COMMIT'], gpu=torch.cuda.get_device_name(),
        precision=configure_precision(True)))
    # Reuse established physical symmetry/restore/loss tests plus inverse roundtrip.
    # These tests need gradients; diagnostic forward passes do not.
    with torch.inference_mode(False), torch.enable_grad():
        suite = unittest.defaultTestLoader.loadTestsFromNames(['test_cf4_member_mass_readout',
            'test_cf4_spatial_diffusion.SpatialDiffusionTests.test_signed_symmetries_and_coarse_restriction'])
        tested = unittest.TextTestRunner(verbosity=2).run(suite)
    dump('tests.json', dict(tests=tested.testsRun, errors=len(tested.errors), failures=len(tested.failures)))
    if not tested.wasSuccessful():
        raise ValueError('focused regression failed')
    # Import only after entry checks. Reuse loader, never call its run/learn/prepare.
    import cf4_bundle_c_member_mass as training
    training.CFG = json.loads((ROOT / 'member_mass_multifield_v4/request.json').read_text())
    models, references = load_models()
    rows = []
    with h5py.File(ROOT / 'total_matter_v1/matter_moments.h5', 'r') as native, \
         h5py.File(ROOT / 'spatial_calibration_v1/spatial_components.h5', 'r') as source:
        if source.attrs['status'] != 'NATIVE_JOINT_PROFILES_AND_REMAINDER_NOT_CF4_LG_POSTERIOR':
            raise ValueError('incorrect native source status')
        selection = selection_context(source, training.CFG['train']+training.CFG['development'])
        dump('selection_context.json', selection)
        for fixture in FIXTURES:
            check()
            case = training.load_case(native, source, fixture)
            x = case['input'].cuda()
            total = case['total'][0].numpy().astype(np.float64)*1e12
            truth = case['truth'].numpy().astype(np.float64)*1e12
            truth_mass = truth.sum((1, 2, 3))
            if not np.isfinite(truth_mass).all() or np.any(truth_mass <= 0):
                raise ValueError('unavailable native role denominator; no floor permitted')
            for name, model in models.items():
                for symmetry in [IDENTITY]+[s for s in range(48) if s != IDENTITY]:
                    check()
                    fractions = model(transform(x, symmetry, True)[None])[0]
                    predicted = transform(fractions, inverse_symmetry(symmetry)).cpu().numpy().astype(np.float64)*total
                    measured = metrics(predicted, truth, total)
                    if measured['conservation_relative_max'] > 1e-6:
                        raise ValueError('inverse-view mass conservation failed')
                    if symmetry == IDENTITY:
                        canonical = predicted.copy()
                        if fixture in references[name]:
                            for key in ('map_L1', 'overlap'):
                                np.testing.assert_allclose(measured[key], references[name][fixture][key], rtol=RTOL, atol=ATOL)
                    absolute_difference = np.abs(predicted-canonical).sum((1, 2, 3))
                    rows.append(dict(checkpoint=name, fixture=fixture, symmetry=symmetry, **measured,
                        mass_ratio=(np.array(measured['predicted_mass_Msun'])/truth_mass).tolist(),
                        identity_discrepancy_L1=(absolute_difference/truth_mass).tolist(),
                        identity_discrepancy_Msun=absolute_difference.tolist(),
                        native_map_absolute_error_Msun=(np.array(measured['map_L1'])*truth_mass).tolist()))
                    del predicted, fractions
                del canonical
                dump('views.json', rows)
                event('EVALUATING', fixture=fixture, checkpoint=name, forwards=len(rows), memory=usage())
            del case, x, total, truth
    result = dict(status='COMPLETE_FROZEN_DIAGNOSIS_NOT_MEMBER_OR_FIELD_ACCEPTANCE',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT'],
        optimizer_updates=0, forward_passes=len(rows), identity_reproduction_passed=True,
        summaries=summarize(rows), memory=usage(), seconds=time.monotonic()-START,
        caution='Sensitivity/selection evidence, not isolated augmentation causality or a posterior. No follow-up job.')
    dump('result.json', result)
    event(result['status'], forward_passes=len(rows), optimizer_updates=0)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if CREATED:
            event('INCOMPLETE_EXECUTION', error=str(exc), optimizer_updates=0)
        traceback.print_exc()
        raise
