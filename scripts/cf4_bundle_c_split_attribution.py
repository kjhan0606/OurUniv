"""Saved-field decoder interventions, NOT samples from a repaired prior."""
import json
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np

from cf4_bundle_c_stable_field import ROOT, cube
from cf4_bundle_c_scale_link import score
from cf4_continuous_matter import restrict
from cf4_spatial_diffusion import pack, unpack

OUT = ROOT/'stable_field_split_attribution_v1'


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run():
    if 'SLURM_JOB_ID' not in os.environ or 'EXPECTED_COMMIT' not in os.environ:
        raise RuntimeError('source-pinned Slurm required')
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    dump('request.json', dict(job_id=os.environ['SLURM_JOB_ID'],
        source_commit=os.environ['EXPECTED_COMMIT'], optimizer_steps=0,
        new_stochastic_draws=0, oracle_interventions=True, observed_posterior=False))
    cases = {c['mw']: c for c in json.loads((ROOT/'population_locations_v1/cases.json').read_text())['test']}
    rows = []
    try:
        with h5py.File(ROOT/'total_matter_v1/matter_moments.h5', 'r') as source:
            for branch in ('control', 'linked_budget'):
                folder = ROOT/'stable_field_link_repair_v1'/branch
                first = [r for r in json.loads((folder/'draws.json').read_text()) if r['draw'] == 0]
                if len(first) != 8 or not all(r['valid'] for r in first):
                    raise ValueError('eight valid preserved first draws required')
                with h5py.File(folder/'first_draw_fields.h5', 'r') as saved:
                    for case in first:
                        if time.monotonic()-started > 240:
                            raise TimeoutError('four-minute application limit')
                        native = restrict(cube(source['fine'], cases[case['mw']]), 4)
                        generated = restrict(saved[f'mw_{case["mw"]}/moments80'][...], 4)
                        nz, nc, parent = pack(native)
                        gz, gc, gp = pack(generated)
                        scales = np.maximum(np.max(abs(parent), axis=(1, 2, 3)), 1)
                        if np.max(np.max(abs(gp-parent), axis=(1, 2, 3))/scales) > 1e-8:
                            raise ValueError('different conditioned1.5 parents')
                        # Re-encoded legal codes: do not confuse these with the
                        # raw categorical network output before canonicalization.
                        combinations = [('native_roundtrip', nz, nc),
                            ('generated_roundtrip', gz, gc),
                            ('native_codes_generated_values', gz, nc),
                            ('generated_codes_native_values', nz, gc)]
                        for mode, z, codes in combinations:
                            field, report = unpack(z, codes, parent)
                            if mode.endswith('roundtrip'):
                                reference = native if mode == 'native_roundtrip' else generated
                                relative = np.max(np.max(abs(field-reference), axis=(1, 2, 3))/np.maximum(np.max(abs(reference), axis=(1, 2, 3)), 1))
                                if relative > 1e-8:
                                    raise ValueError(f'{mode} failed: {relative}')
                            row = dict(branch=branch, mw=case['mw'], mode=mode,
                                dx_cMpc_h=.75, oracle_intervention='codes_' in mode,
                                score=score(field[:, 2:-2, 2:-2, 2:-2], native[:, 2:-2, 2:-2, 2:-2],
                                    parent[:, 1:-1, 1:-1, 1:-1], .75, 2), decoder=report,
                                legal_category_mismatch_fraction=float(np.mean(nc != gc)),
                                native_category_counts=np.bincount(nc.ravel(), minlength=4).tolist(),
                                generated_category_counts=np.bincount(gc.ravel(), minlength=4).tolist())
                            rows.append(row)
                        dump('scores.json', rows)
                        print(json.dumps(dict(branch=branch, mw=case['mw'], rows=len(rows))), flush=True)
        dump('result.json', dict(status='COMPLETE_DECODER_INTERVENTION_NOT_POSTERIOR',
            rows=len(rows), roundtrip_controls=32, optimizer_steps=0, new_stochastic_draws=0,
            seconds=time.monotonic()-started,
            host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['EXPECTED_COMMIT']))
    except Exception as error:
        dump('failure.json', dict(error=str(error), completed_rows=len(rows)))
        raise


if __name__ == '__main__':
    run()
