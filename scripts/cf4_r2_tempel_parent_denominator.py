"""Build a full Tempel-parent/SDSS-FP denominator and independent 2M++ link.

This is a source-selection diagnostic only. It never fits a selection law or
uses component truth to choose a field candidate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_tempel_parent import (  # noqa: E402
    nearest_random_arcsec,
    parent_indices,
    unique_velocity_matches,
)

EXTERNAL = Path('/gpfs/kjhan/CF4/external/tempel2017_cds')
FP_SOURCE = Path('/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat')
BASE = Path('/gpfs/kjhan/CF4/z0_density')
R2_LINK = BASE/'r2_sdss_fp_source_link_v1/source_link.npz'
SPLIT = BASE/'r2_sky_closed_split_v5/split.npz'
M2PP = ROOT/'data/2mpp_catalog.csv'
OUT = BASE/'r2_tempel_parent_denominator_v1'
FP_MD5 = 'b5b6e31caf7ea469c2ac2cb775fa8d14'
RANDOM_MD5 = '8627b4063e8a71572e333e0ac65d6657'
RANDOM_BYTES = 284000071
RANDOM_URL = 'https://zenodo.org/api/records/6824749/files/SDSS_randoms_public.dat/content'
TABLE_ROWS = {'table1': 584449, 'table2': 88662}
C_KMS = 299792.458


def digest(path: Path, algorithm: str = 'sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def acquire_vizier(table: str, columns: list[str], filename: str) -> tuple[Path, str]:
    path = EXTERNAL/filename
    url = 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?'+urlencode({
        '-source': f'J/A+A/602/A100/{table}', '-out': ','.join(columns),
        '-out.max': 'unlimited'})
    if not path.exists():
        partial = path.with_suffix(path.suffix+'.partial')
        transferred = 0
        with urlopen(url, timeout=180) as response, partial.open('wb') as target:
            for chunk in iter(lambda: response.read(1024*1024), b''):
                transferred += len(chunk)
                if transferred > 100_000_000:
                    raise ValueError('Tempel selected-column response exceeds 100MB bound')
                target.write(chunk)
        if transferred == 0:
            raise ValueError('empty Tempel table response')
        partial.replace(path)
    return path, url


def numeric_table(path: Path, columns: list[str], count: int,
                  dtypes: dict[str, object]) -> dict[str, np.ndarray]:
    out = {name: np.empty(count, dtype=dtypes[name]) for name in columns}
    row_count = 0
    with path.open() as stream:
        for raw in stream:
            if not raw.strip() or raw.startswith('#'):
                continue
            values = raw.rstrip('\n').split('\t')
            if not values[0].strip().isdigit():
                continue
            if len(values) != len(columns) or row_count >= count:
                raise ValueError(f'unexpected {path.name} row width/count')
            for name, value in zip(columns, values, strict=True):
                out[name][row_count] = value.strip()
            row_count += 1
    if row_count != count:
        raise ValueError(f'{path.name}: expected {count} rows, got {row_count}')
    return out


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    return np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))


def random_positions() -> tuple[np.ndarray, dict[str, object]]:
    xyz = np.empty((4_000_000, 3), dtype=np.float64)
    md5 = hashlib.md5()
    transferred = rows = 0
    with urlopen(RANDOM_URL, timeout=180) as response:
        if int(response.headers.get('Content-Length', '0')) != RANDOM_BYTES:
            raise ValueError('published SDSS random catalogue length changed')
        header = response.readline()
        md5.update(header)
        transferred += len(header)
        if header.lstrip(b'#').split() != [b'RA', b'Dec', b'zcmb', b'nbar']:
            raise ValueError('unexpected SDSS random catalogue columns')
        for raw in response:
            transferred += len(raw)
            md5.update(raw)
            values = raw.split()
            if len(values) != 4 or rows >= len(xyz):
                raise ValueError('unexpected SDSS random row width/count')
            ra, dec = np.deg2rad(float(values[0])), np.deg2rad(float(values[1]))
            xyz[rows] = (np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec))
            rows += 1
    if transferred != RANDOM_BYTES or md5.hexdigest() != RANDOM_MD5 or rows != len(xyz):
        raise ValueError('published SDSS random catalogue length/hash/rows changed')
    return xyz, dict(url=RANDOM_URL, bytes=transferred, md5=md5.hexdigest(), rows=rows)


def source_fp_rows(path: Path) -> dict[str, np.ndarray]:
    if digest(path, 'md5') != FP_MD5:
        raise ValueError('published SDSS PV source checksum mismatch')
    with path.open() as stream:
        header = stream.readline().lstrip('#').split()
        columns = ['PGC', 'objid', 'RA', 'Dec', 'zcmb', 'in_mask', 'IDgroupT17', 'NgroupT17']
        ix = {name: header.index(name) for name in columns}
        result = {name: [] for name in columns}
        rows = 0
        for raw in stream:
            if not raw.strip():
                continue
            v = raw.split()
            if len(v) != len(header):
                raise ValueError('unexpected SDSS PV source row width')
            for name in columns:
                result[name].append(v[ix[name]])
            rows += 1
    if rows != 34059:
        raise ValueError(f'expected 34059 SDSS PV rows, got {rows}')
    casts = {'PGC': np.int64, 'objid': np.uint64, 'RA': np.float64, 'Dec': np.float64,
             'zcmb': np.float64, 'in_mask': np.int8, 'IDgroupT17': np.int64,
             'NgroupT17': np.int16}
    return {name: np.asarray(values, dtype=casts[name]) for name, values in result.items()}


def m2pp_rows(path: Path) -> dict[str, np.ndarray]:
    names, recno, gid, ra, dec, velocity = [], [], [], [], [], []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'recno', 'GID', 'Vcmb', '_RA', '_DE'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('unexpected 2M++ catalogue columns')
        for row in reader:
            names.append(row['Name'])
            recno.append(int(row['recno']))
            gid.append(int(row['GID']) if row['GID'].strip() else -1)
            ra.append(float(row['_RA']))
            dec.append(float(row['_DE']))
            velocity.append(float(row['Vcmb']))
    out = dict(name=np.asarray(names), recno=np.asarray(recno, dtype=np.int64),
               gid=np.asarray(gid, dtype=np.int64), ra=np.asarray(ra), dec=np.asarray(dec),
               velocity=np.asarray(velocity))
    if len(out['recno']) == 0 or len(np.unique(out['recno'])) != len(out['recno']):
        raise ValueError('2M++ source is empty or recno is not unique')
    return out


def distance_h_mpc(z: np.ndarray) -> np.ndarray:
    grid = np.linspace(0., .2, 20001)
    radial = 2997.92458*cumulative_trapezoid(
        1./np.sqrt(.31*(1.+grid)**3+.69), grid, initial=0.)
    return np.interp(z, grid, radial)


def count_categories(counts: np.ndarray, use: np.ndarray) -> dict[str, int]:
    values = counts[np.asarray(use, dtype=bool)]
    return dict(parents=int(len(values)), zero=int(np.sum(values == 0)),
                one=int(np.sum(values == 1)), multiple=int(np.sum(values >= 2)))


def main() -> None:
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    table1_path, table1_url = acquire_vizier(
        'table1', ['GalID', 'objID', 'GroupID', 'Ngal', 'zcmb', 'RAJ2000', 'DEJ2000'],
        'table1_parent_positions_v2.tsv')
    table2_path, table2_url = acquire_vizier(
        'table2', ['GroupID', 'Ngal', 'zcmb', 'RAJ2000', 'DEJ2000'],
        'table2_parent_centers_v2.tsv')
    members = numeric_table(table1_path,
        ['GalID', 'objID', 'GroupID', 'Ngal', 'zcmb', 'RAJ2000', 'DEJ2000'], TABLE_ROWS['table1'],
        dict(GalID=np.int64, objID=np.uint64, GroupID=np.int32, Ngal=np.int16,
             zcmb=np.float64, RAJ2000=np.float64, DEJ2000=np.float64))
    groups = numeric_table(table2_path,
        ['GroupID', 'Ngal', 'zcmb', 'RAJ2000', 'DEJ2000'], TABLE_ROWS['table2'],
        dict(GroupID=np.int32, Ngal=np.int16, zcmb=np.float64,
             RAJ2000=np.float64, DEJ2000=np.float64))
    order_g = np.argsort(groups['GroupID'])
    groups = {k: v[order_g] for k, v in groups.items()}
    expected_ids = np.arange(1, TABLE_ROWS['table2']+1, dtype=np.int32)
    if not np.array_equal(groups['GroupID'], expected_ids):
        raise ValueError('Tempel group IDs are not the complete expected range')
    member_counts = np.bincount(members['GroupID'][members['GroupID'] > 0],
                                minlength=len(expected_ids)+1)[1:]
    if not np.array_equal(member_counts, groups['Ngal']):
        raise ValueError('Tempel group member counts disagree with full parent table')
    if len(np.unique(members['objID'])) != len(members['objID']):
        raise ValueError('Tempel photometric objID is not unique')

    n_group = len(groups['GroupID'])
    tempel_parent, parent_kind, parent_id = parent_indices(
        members['GroupID'], members['GalID'], n_group)
    n_parent = n_group + int(np.sum(members['GroupID'] == 0))
    single_rows = np.flatnonzero(members['GroupID'] == 0)
    parent_richness = np.concatenate((groups['Ngal'], np.ones(len(single_rows), dtype=np.int16)))
    parent_z = np.concatenate((groups['zcmb'], members['zcmb'][single_rows]))
    parent_ra = np.concatenate((groups['RAJ2000'], members['RAJ2000'][single_rows]))
    parent_dec = np.concatenate((groups['DEJ2000'], members['DEJ2000'][single_rows]))
    parent_xyz = unit_vectors(parent_ra, parent_dec)

    fp = source_fp_rows(FP_SOURCE)
    member_order = np.argsort(members['objID'])
    sorted_objid = members['objID'][member_order]
    fp_pos = np.searchsorted(sorted_objid, fp['objid'])
    candidate = fp_pos < len(sorted_objid)
    fp_found = np.zeros(len(fp_pos), dtype=bool)
    fp_found[candidate] = sorted_objid[fp_pos[candidate]] == fp['objid'][candidate]
    fp_member = np.full(len(fp['PGC']), -1, dtype=np.int64)
    fp_member[fp_found] = member_order[fp_pos[fp_found]]
    source_group_mismatch = np.zeros(len(fp_found), dtype=bool)
    source_richness_mismatch = np.zeros(len(fp_found), dtype=bool)
    source_group_mismatch[fp_found] = (
        members['GroupID'][fp_member[fp_found]] != fp['IDgroupT17'][fp_found])
    source_richness_mismatch[fp_found] = (
        members['Ngal'][fp_member[fp_found]] != fp['NgroupT17'][fp_found])

    with np.load(R2_LINK, allow_pickle=False) as f:
        eligible_pgc = f['PGC'].astype(np.int64)
        eligible_group = f['source_group'].astype(str)
    with np.load(SPLIT, allow_pickle=False) as f:
        fp_split_group = f['fp_source_group'].astype(str)
        fp_split_role = f['fp_role'].astype(np.int8)
        count_recno = f['point_recno'].astype(np.int64)
        count_heldout = f['point_heldout'].astype(bool)
    role_lookup = dict(zip(fp_split_group.tolist(), fp_split_role.tolist(), strict=True))
    eligible_role = {int(p): role_lookup[str(g)] for p, g in zip(eligible_pgc, eligible_group, strict=True)}
    is_r2_eligible = np.isin(fp['PGC'], eligible_pgc)
    fp_roles = np.array([eligible_role.get(int(p), -1) for p in fp['PGC']], dtype=np.int8)

    n_fp_all = np.zeros(n_parent, dtype=np.int32)
    n_fp_mask = np.zeros(n_parent, dtype=np.int32)
    n_r2_all = np.zeros(n_parent, dtype=np.int32)
    n_r2_mask = np.zeros(n_parent, dtype=np.int32)
    n_fp_train = np.zeros(n_parent, dtype=np.int32)
    n_fp_heldout = np.zeros(n_parent, dtype=np.int32)
    matched_parent = tempel_parent[fp_member[fp_found]]
    np.add.at(n_fp_all, matched_parent, 1)
    matched_inmask = fp_found & (fp['in_mask'] == 1)
    np.add.at(n_fp_mask, tempel_parent[fp_member[matched_inmask]], 1)
    matched_eligible = fp_found & is_r2_eligible
    np.add.at(n_r2_all, tempel_parent[fp_member[matched_eligible]], 1)
    matched_eligible_mask = matched_eligible & (fp['in_mask'] == 1)
    np.add.at(n_r2_mask, tempel_parent[fp_member[matched_eligible_mask]], 1)
    for select, target in ((matched_eligible & (fp_roles == 0), n_fp_train),
                           (matched_eligible & (fp_roles == 1), n_fp_heldout)):
        np.add.at(target, tempel_parent[fp_member[select]], 1)

    # Link all Tempel members to 2M++ galaxies without using FP identities.
    m2 = m2pp_rows(M2PP)
    tempel_xyz = unit_vectors(members['RAJ2000'], members['DEJ2000'])
    source_i, member_i, link_diag = unique_velocity_matches(
        tempel_xyz, C_KMS*members['zcmb'], unit_vectors(m2['ra'], m2['dec']),
        m2['velocity'], max_sep_arcsec=10.0, max_dv_km_s=300.0)
    mpp_parent = tempel_parent[member_i]
    n_mpp_members = np.bincount(mpp_parent, minlength=n_parent).astype(np.int32)
    mpp_gids_by_parent: dict[int, set[int]] = defaultdict(set)
    for p, m in zip(mpp_parent, source_i, strict=True):
        if m2['gid'][m] >= 0:
            mpp_gids_by_parent[int(p)].add(int(m2['gid'][m]))
    n_mpp_gids = np.zeros(n_parent, dtype=np.int16)
    for p, gids in mpp_gids_by_parent.items():
        n_mpp_gids[p] = len(gids)

    count_role = {int(r): bool(h) for r, h in zip(count_recno, count_heldout, strict=True)}
    mpp_train = np.zeros(n_parent, dtype=np.int32)
    mpp_heldout = np.zeros(n_parent, dtype=np.int32)
    for p, m in zip(mpp_parent, source_i, strict=True):
        held = count_role.get(int(m2['recno'][m]))
        if held is not None:
            (mpp_heldout if held else mpp_train)[p] += 1
    role_train = (n_fp_train + mpp_train) > 0
    role_held = (n_fp_heldout + mpp_heldout) > 0
    parent_role = np.zeros(n_parent, dtype=np.int8)  # 0 no R2 role, 1 train, 2 heldout, 3 mixed
    parent_role[role_train & ~role_held] = 1
    parent_role[role_held & ~role_train] = 2
    parent_role[role_train & role_held] = 3

    # Reconstruct the public angular support from its official random sample;
    # calibrate its hard threshold against the catalogue's native in_mask flag.
    random_xyz, random_meta = random_positions()
    random_tree = cKDTree(random_xyz)
    del random_xyz
    source_distance = nearest_random_arcsec(random_tree, unit_vectors(fp['RA'], fp['Dec']), workers=2)
    parent_distance = nearest_random_arcsec(random_tree, parent_xyz, workers=2)
    pos = fp['in_mask'] == 1
    neg = ~pos
    if int(pos.sum()) != 33121 or int(neg.sum()) != 938:
        raise ValueError('published SDSS PV in_mask totals changed')
    mask_threshold = float(np.quantile(source_distance[pos], .99))
    mask_sensitivity = float(np.mean(source_distance[pos] <= mask_threshold))
    mask_false_positive = float(np.mean(source_distance[neg] <= mask_threshold))
    parent_in_mask = parent_distance <= mask_threshold
    parent_radius = distance_h_mpc(parent_z)
    parent_in_window = (parent_radius > 15.) & (parent_radius < 180.)
    parent_common = parent_in_mask & parent_in_window
    linked_parent = n_mpp_members > 0

    richness_bin = np.select([parent_richness == 1, parent_richness == 2,
        parent_richness == 3, parent_richness <= 5, parent_richness <= 10],
        [0, 1, 2, 3, 4], default=5).astype(np.int8)
    richness_names = ['single', '2', '3', '4-5', '6-10', '11+']
    redshift_bin = np.minimum(np.digitize(parent_z, [0., .03, .06, .10, .20], right=False)-1, 3)
    strata = []
    for kind in (0, 1):
        for rbin, rname in enumerate(richness_names):
            for zbin, zname in enumerate(['0-.03', '.03-.06', '.06-.10', '.10-.20']):
                use = parent_common & (parent_kind == kind) & (richness_bin == rbin) & (redshift_bin == zbin)
                if np.any(use):
                    strata.append(dict(parent='multi' if kind == 0 else 'singleton',
                        richness=rname, observed_z=zname,
                        total=count_categories(n_r2_mask, use),
                        train=count_categories(n_r2_mask, use & (parent_role == 1)),
                        heldout=count_categories(n_r2_mask, use & (parent_role == 2)),
                        mixed_role=int(np.sum(use & (parent_role == 3)))))

    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'parents.npz', parent_kind=parent_kind, parent_id=parent_id,
        richness=parent_richness, zcmb=parent_z, ra_deg=parent_ra, dec_deg=parent_dec,
        nearest_random_arcsec=parent_distance, common_footprint_z_window=parent_common,
        role=parent_role, n_fp_catalogue=n_fp_all, n_fp_in_mask=n_fp_mask,
        n_r2_eligible=n_r2_all, n_r2_eligible_in_mask=n_r2_mask,
        n_m2pp_member_links=n_mpp_members, n_m2pp_gids=n_mpp_gids,
        n_m2pp_train_links=mpp_train, n_m2pp_heldout_links=mpp_heldout)
    np.savez_compressed(OUT/'independent_m2pp_links.npz',
        m2pp_recno=m2['recno'][source_i], m2pp_gid=m2['gid'][source_i],
        tempel_galid=members['GalID'][member_i], tempel_groupid=members['GroupID'][member_i],
        parent_index=mpp_parent)
    report = dict(classification='OBSERVED_TEMPEL_PARENT_FP_DENOMINATOR_NOT_LATENT_SELECTION_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'], elapsed_s=round(time.monotonic()-started, 3),
        source_counts=dict(tempel_multimember_groups=n_group, tempel_member_rows=len(members),
            tempel_singleton_parents=len(single_rows), full_fp_rows=len(fp),
            full_fp_exact_tempel_member_matches=int(fp_found.sum()),
            full_fp_catalogue_absent_from_tempel=int((~fp_found).sum()),
            matched_source_group_id_mismatches=int(source_group_mismatch.sum()),
            matched_source_richness_mismatches=int(source_richness_mismatch.sum()),
            full_fp_in_mask=int(pos.sum()), r2_eligible_fp=len(eligible_pgc),
            r2_eligible_exact_tempel_matches=int((is_r2_eligible & fp_found).sum()),
            r2_eligible_absent_from_tempel=int((is_r2_eligible & ~fp_found).sum()),
            r2_eligible_in_mask=int(matched_eligible_mask.sum()),
            m2pp_source_rows=len(m2['recno']), m2pp_to_tempel_unique_links=link_diag['unique_pairs']),
        footprint=dict(random_source=random_meta, threshold_arcsec_p99_in_mask=mask_threshold,
            positive_control_sensitivity=mask_sensitivity, out_of_mask_false_positive_rate=mask_false_positive,
            in_mask_parent_centres=int(parent_in_mask.sum()), note='Random-neighbour support is an approximate mask reconstruction; native flag confusion is reported, not hidden.'),
        parent_denominator=dict(all_common_zwindow=count_categories(n_r2_mask, parent_common),
            common_zwindow_with_independent_m2pp_member=count_categories(n_r2_mask, parent_common & linked_parent),
            common_zwindow_without_independent_m2pp_member=count_categories(n_r2_mask, parent_common & ~linked_parent),
            groupid_zero_singletons=count_categories(n_r2_mask, parent_common & (parent_kind == 1)),
            eligible_train=count_categories(n_r2_mask, parent_common & (parent_role == 1)),
            eligible_heldout=count_categories(n_r2_mask, parent_common & (parent_role == 2)),
            mixed_role=int(np.sum(parent_common & (parent_role == 3))),
            independent_tempel_to_2mpp_group_links=dict(parents_with_member_link=int(linked_parent.sum()),
                no_m2pp_gid=int(np.sum(linked_parent & (n_mpp_gids == 0))),
                one_m2pp_gid=int(np.sum(n_mpp_gids == 1)), multiple_m2pp_gids=int(np.sum(n_mpp_gids > 1)))),
        richness_redshift_strata=strata,
        independent_link_rule='2M++ galaxy RA/Dec within10arcsec and |Vcmb-c*zTempel|<=300km/s; reciprocal one-to-one only; FP IDs/flags not used for matching.',
        fp_eligibility_rule='Full published SDSS PV rows, native in_mask flag, and current 10,020-row R2 eligibility are carried as separate counts.',
        no_fit=True, no_field_scores=True, heldout_marks_not_scored=True,
        selected_group_law_calibrated=False, true_distance_selection_calibrated=False,
        shared_covariance_calibrated=False, R2_posterior=False, new_gravity_runs=0,
        Q_GOAL='Adds source-defined parent/zero-FP denominator and an independent route to 2M++ parent associations for the R2 CF4/LG present-field observation law; it does not itself constrain a latent field.',
        Q_LEAN='One bounded source join; no sampler, new simulation, mock archive, model fit, or repeated mask sweep.',
        MW_M31_M33='No component labels are inferred or used. In R3, identify MW/M31 ambiguously from each NEW evolved field, retain unresolved M33, and constrain the SAME field with observed positions, distances, masses, and velocities; native IDs are evaluation-only.',
        source_sha256=dict(tempel_table1=digest(table1_path), tempel_table2=digest(table2_path),
            sdss_fp=digest(FP_SOURCE), two_mpp=digest(M2PP), split=digest(SPLIT)),
        code_sha256=dict(helper=digest(ROOT/'src/cf4_r2_tempel_parent.py'),
            driver=digest(Path(__file__)), tests=digest(ROOT/'tests/test_cf4_r2_tempel_parent.py')),
        tempel_urls=dict(table1=table1_url, table2=table2_url),
        split_path=str(SPLIT), R2_link_path=str(R2_LINK))
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
