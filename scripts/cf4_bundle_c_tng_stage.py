"""I/O-only staging from syntax-local TNG storage; numerical work stays in Slurm."""
import argparse
import json
from pathlib import Path
import h5py

RAW = Path('/scratch/kjhan/IllustrisTNG/TNG100-1/output')
OUT = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/tng_operator_v1')
FIELDS = {
    'Group': ('GroupFirstSub', 'GroupNsubs', 'GroupLenType', 'Group_M_Crit200', 'GroupPos'),
    'Subhalo': ('SubhaloPos', 'SubhaloVel', 'SubhaloMass', 'SubhaloLenType', 'SubhaloGrNr', 'SubhaloFlag'),
}


def stage_catalog():
    OUT.mkdir(exist_ok=False)
    with h5py.File(OUT / 'native_catalog.h5', 'x') as target:
        for index in range(448):
            chunk = target.create_group(f'chunks/{index}')
            with h5py.File(RAW / f'groups_099/fof_subhalo_tab_099.{index}.hdf5', 'r') as source:
                source.copy('Header', chunk)
                for group, fields in FIELDS.items():
                    dest = chunk.create_group(group)
                    if group in source:
                        for field in fields:
                            source.copy(source[group][field], dest, name=field)
            # Only headers are staged here, not the multi-TB snapshot.
            with h5py.File(RAW / f'snapdir_099/snap_099.{index}.hdf5', 'r') as source:
                source.copy('Header', chunk, name='SnapshotHeader')
            if (index + 1) % 64 == 0:
                print(f'catalog I/O {index + 1}/448', flush=True)
        target.attrs['source'] = str(RAW)
        target.attrs['status'] = 'COMPLETE_NATIVE_FIELD_COPY_NO_SELECTION'


def stage_particles():
    request = json.loads((OUT / 'particle_request.json').read_text())
    with h5py.File(OUT / 'native_particles.h5', 'x') as target:
        for type_request in request['particle_types']:
            name = f"PartType{type_request['type']}"
            group = target.create_group(name)
            for piece in type_request['pieces']:
                path = RAW / f"snapdir_099/snap_099.{piece['file']}.hdf5"
                with h5py.File(path, 'r') as source:
                    for field in type_request['fields']:
                        data = source[f'{name}/{field}']
                        if field not in group:
                            group.create_dataset(field, shape=(type_request['count'],) + data.shape[1:], dtype=data.dtype)
                        group[field][piece['destination_start']:piece['destination_stop']] = data[piece['start']:piece['stop']]
            print(f"particle I/O {name}, rows={type_request['count']}", flush=True)
        target.attrs['source'] = str(RAW)
        target.attrs['status'] = 'COMPLETE_NATIVE_FOF_COPY_NOT_SPATIAL_CUTOUT'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['catalog', 'particles'])
    action = parser.parse_args().action
    (stage_catalog if action == 'catalog' else stage_particles)()
