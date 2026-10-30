"""Bounded I/O-only extraction of known syntax-visible catalogue fields.

No fitting, field conversion, simulation or numerical calibration. Copies
about150MB to shared project storage for the Slurm response calculation.
"""
from pathlib import Path
import h5py

RAW = Path('/scratch/kjhan/IllustrisTNG/TNG100-1/output/groups_099')
TARGET = Path('/gpfs/kjhan/CF4/z0_density/r2_tng_native_k_catalog_20261002_v1.h5')


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    temporary = TARGET.with_suffix('.h5.partial')
    with h5py.File(RAW/'fof_subhalo_tab_099.0.hdf5', 'r') as source:
        files = int(source['Header'].attrs['NumFiles'])
        expected = int(source['Header'].attrs['Nsubgroups_Total'])
    rows = 0
    with h5py.File(temporary, 'x') as target:
        target.attrs['status'] = 'INCOMPLETE'
        target.attrs['source'] = str(RAW)
        for index in range(files):
            with h5py.File(RAW/f'fof_subhalo_tab_099.{index}.hdf5','r') as source:
                chunk = target.create_group(f'chunks/{index}')
                source.copy('Header', chunk)
                n = int(source['Header'].attrs['Nsubgroups_ThisFile'])
                if n:
                    group = chunk.create_group('Subhalo')
                    for key in ('SubhaloPos','SubhaloVel','SubhaloFlag'):
                        source.copy(source[f'Subhalo/{key}'], group, name=key)
                    group.create_dataset('star_count', data=source['Subhalo/SubhaloLenType'][:,4])
                    group.create_dataset('K_physical', data=source['Subhalo/SubhaloStellarPhotometrics'][:,3])
                rows += n
            if (index+1) % 64 == 0:
                print(f'copied {index+1}/{files} catalogue chunks', flush=True)
        if rows != expected:
            raise ValueError('incomplete native catalogue copy')
        target.attrs.update(status='COMPLETE_NATIVE_FIELD_COPY_NO_SELECTION', rows=rows, files=files)
    temporary.rename(TARGET)
    print(f'copied native fields: {rows} rows; {TARGET.stat().st_size} bytes; {TARGET}', flush=True)


if __name__ == '__main__':
    main()
