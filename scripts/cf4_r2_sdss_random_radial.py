"""Source-backed SDSS FP radial-selection shape, without a field fit.

Stream the published random catalogue; retain only a small radial summary.
Its redshifts were sampled from the *observed* SDSS FP n(z), so this is not
by itself the real-distance selection of CF4 source groups.
"""

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_sdss_random_radial_v1'
RANDOM_URL = 'https://zenodo.org/api/records/6824749/files/SDSS_randoms_public.dat/content'
RANDOM_MD5 = '8627b4063e8a71572e333e0ac65d6657'
SOURCE = Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')
SOURCE_MD5 = 'b5b6e31caf7ea469c2ac2cb775fa8d14'
OBS = BASE/'r2_source_observation_assembly_v1/observations.npz'
GEOMETRY = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
SPLIT = BASE/'r2_sky_closed_split_v5/split.npz'
EDGES = np.linspace(0., .1, 21)


def bin_index(redshift):
    if not np.isfinite(redshift) or redshift < 0 or redshift > .2:
        raise ValueError(f'published redshift outside physical read range: {redshift}')
    if redshift > .1:
        return None
    return min(int(redshift/.005), len(EDGES)-2)


def source_radii(z):
    z_grid = np.linspace(0., .1, 10001)
    d_grid = 2997.92458*cumulative_trapezoid(
        1./np.sqrt(.31*(1.+z_grid)**3+.69), z_grid, initial=0.)
    return np.interp(z, z_grid, d_grid)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    # A fixed toy shell control catches a bin-edge/volume-normalization error.
    assert [bin_index(v) for v in (0., .0049, .005, .1, .1001)] == [0, 0, 1, 19, None]
    radii = source_radii(EDGES)
    assert np.all(np.diff(radii) > 0)

    random_counts = np.zeros(20, dtype=np.int64)
    nbar_sum = np.zeros(20, dtype=np.float64)
    random_digest = hashlib.md5()
    transferred = 0
    random_rows = random_outside = 0
    random_z_max = 0.
    with urlopen(RANDOM_URL, timeout=120) as response:
        declared = int(response.headers.get('Content-Length', '0'))
        if declared != 284000071:
            raise ValueError(f'published random catalogue length changed: {declared}')
        header = response.readline()
        random_digest.update(header)
        transferred += len(header)
        if header.lstrip(b'#').split() != [b'RA', b'Dec', b'zcmb', b'nbar']:
            raise ValueError('unexpected published random columns')
        for line in response:
            transferred += len(line)
            if transferred > declared:
                raise ValueError('random catalogue exceeds published length')
            random_digest.update(line)
            values = line.split()
            if len(values) != 4:
                raise ValueError('unexpected random row width')
            z, nbar = float(values[2]), float(values[3])
            if not np.isfinite(nbar) or nbar <= 0:
                raise ValueError('invalid published random nbar')
            k = bin_index(z)
            random_rows += 1
            random_z_max = max(random_z_max, z)
            if k is None:
                random_outside += 1
                continue
            random_counts[k] += 1
            nbar_sum[k] += nbar
    if transferred != 284000071 or random_digest.hexdigest() != RANDOM_MD5:
        raise ValueError('published random catalogue hash/length mismatch')

    with np.load(OBS, allow_pickle=False) as f:
        eligible_pgc = f['PGC'].astype(np.int64)
        eligible_source = f['source_group'].copy()
    with np.load(GEOMETRY, allow_pickle=False) as f:
        group_labels = f['group_labels'].copy()
    with np.load(SPLIT, allow_pickle=False) as f:
        np.testing.assert_array_equal(group_labels, f['fp_source_group'])
        roles = f['fp_role'].copy()
    group_index = np.searchsorted(group_labels, eligible_source)
    if (len(eligible_pgc) != 10020 or np.any(group_index >= len(group_labels))
            or np.any(group_labels[group_index] != eligible_source)):
        raise ValueError('frozen FP source-group identities changed')
    train_pgc = set(map(int, eligible_pgc[roles[group_index] == 0]))
    full_counts = np.zeros(20, dtype=np.int64)
    cf4_train_counts = np.zeros(20, dtype=np.int64)
    source_digest = hashlib.md5()
    source_rows = source_in_mask = 0
    selected_train_rows = 0
    source_outside = 0
    with SOURCE.open('rb') as stream:
        header = stream.readline()
        source_digest.update(header)
        columns = header.lstrip(b'#').split()
        wanted = {name:columns.index(name) for name in (b'PGC',b'zcmb',b'in_mask')}
        for line in stream:
            source_digest.update(line)
            values = line.split()
            if not values:
                continue
            source_rows += 1
            if len(values) != len(columns):
                raise ValueError('unexpected SDSS FP row width')
            if int(values[wanted[b'in_mask']]) != 1:
                continue
            source_in_mask += 1
            k = bin_index(float(values[wanted[b'zcmb']]))
            if k is None:
                source_outside += 1
                continue
            full_counts[k] += 1
            if int(values[wanted[b'PGC']]) in train_pgc:
                cf4_train_counts[k] += 1
                selected_train_rows += 1
    if source_digest.hexdigest() != SOURCE_MD5 or source_rows != 34059:
        raise ValueError('published SDSS FP source hash/row count changed')
    # The paper's 33,618 in-mask figure precedes its final FP outlier cut.
    # The hash-frozen v1.1 source itself has 33,121 in-mask rows.
    if source_in_mask != 33121:
        raise ValueError('final published FP mask count changed')
    shell_volumes = np.diff(radii**3)/3.
    shell_density = random_counts/shell_volumes
    reference_bin = 6  # z in [.030,.035), fixed before reading marks.
    if random_counts[reference_bin] == 0:
        raise ValueError('published randoms lack fixed reference shell')
    relative = [float(v/shell_density[reference_bin]) if n else None
                for v,n in zip(shell_density,random_counts)]
    nbar_mean = [float(total/n) if n else None
                 for total,n in zip(nbar_sum,random_counts)]
    fraction = np.divide(cf4_train_counts, full_counts,
                         out=np.zeros(20, dtype=float), where=full_counts > 0)
    report = dict(
        classification='R2_PUBLIC_SDSS_FP_RANDOM_RADIAL_SHAPE_NOT_GROUP_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'], source_url=RANDOM_URL,
        random_bytes=transferred, random_md5=random_digest.hexdigest(),
        random_rows=random_rows, random_rows_in_profile=int(random_counts.sum()),
        random_rows_above_z0p1=random_outside, random_z_max=random_z_max,
        raw_random_retained=False,
        source_path=str(SOURCE), source_md5=source_digest.hexdigest(),
        full_source_rows=source_rows, full_in_mask_rows=source_in_mask,
        full_in_mask_rows_above_z0p1=source_outside,
        CF4_FP_eligible_rows=len(eligible_pgc),
        CF4_FP_sky_train_in_mask_rows=selected_train_rows,
        heldout_FP_marks_read=False, fp_group_selection_calibrated=False,
        true_distance_selection_calibrated=False, count_FP_overlap_calibrated=False,
        observed_redshift_bin_edges=EDGES.tolist(),
        comoving_cMpc_h_edges=radii.tolist(),
        random_count_by_bin=random_counts.tolist(),
        full_in_mask_count_by_bin=full_counts.tolist(),
        CF4_FP_sky_train_in_mask_count_by_bin=cf4_train_counts.tolist(),
        published_nbar_mean_by_bin=nbar_mean,
        relative_random_number_density_by_bin=relative,
        CF4_train_to_full_in_mask_fraction_by_bin=fraction.tolist(),
        reference_redshift_bin=[float(EDGES[reference_bin]),float(EDGES[reference_bin+1])],
        interpretation='Published random z values sample a smoothed observed SDSS FP '
                       'redshift distribution. Dividing their shell counts by comoving '
                       'shell volume gives an angular-averaged redshift-space selection '
                       'shape, NOT a true-distance selected-group prior, CF4 subset '
                       'inclusion law or independent sky-holdout calibration.',
        MW_M31_M33='New-field latent roles; MW/M31 ambiguous and M33 may be '
                    'unresolved. No native/mock identity seeds candidates.',
        R2_posterior=False, N256=False)
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
