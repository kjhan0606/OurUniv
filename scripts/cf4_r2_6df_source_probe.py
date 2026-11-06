"""Inspect ONE author release: archive/schema/identities only, never FP values.

Do not parse acceptance bits either: a residual-rejection bit is a derived
outcome. Split/identity ownership must be established before examining it.
This is not a source selection fit or an amendment to the frozen v6 sample.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import time
import zipfile


ARCHIVE = Path('/gpfs/kjhan/CF4/external/6dfgs_2014_source_probe_20261107/'
               'stu1198_Supplementary_Data.zip')
EXPECTED_SHA = '474c82f5f1338d1ed457ee7e86806388960e6b0b7c79f5cbebfb1343cb074283'
SOURCE_ID = re.compile(r'(?<!\w)g\d{7}[+-]\d{6}(?!\w)')
XSC_ID = re.compile(r'(?<!\w)2MASX\s*J\d{8}[+-]\d{7}(?!\w)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    started = time.monotonic()
    content = ARCHIVE.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    if sha != EXPECTED_SHA or len(content) > 10_000_000:
        raise ValueError('author archive differs from pinned bounded download')
    with zipfile.ZipFile(io.BytesIO(content)) as outer:
        members = [dict(name=x.filename, bytes=x.file_size, crc=x.CRC) for x in outer.infolist()]
        raw = outer.read('campbell_table8.zip')
    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
        # Inspect in memory; never extract arbitrary archive paths or executables.
        entries = [x for x in inner.infolist() if not x.is_dir()
                   and not x.filename.startswith('__MACOSX/')
                   and not Path(x.filename).name.startswith('.')]
        if len(entries) != 1 or entries[0].file_size > 10_000_000:
            raise ValueError(f'unexpected nested table entries: {[x.filename for x in entries]}')
        table_name = entries[0].filename
        table = inner.read(table_name)
    identities, header, columns = [], [], Counter()
    for line in table.decode('utf-8').splitlines():
        if not line.strip():
            continue
        source, xsc = SOURCE_ID.search(line), XSC_ID.search(line)
        if source is None and xsc is None and not identities:
            header.append(line)  # before the first identified record only
            continue
        if source is None or xsc is None:
            raise ValueError('non-header line missing a source identity; no science fallback')
        identities.append((source.group(), re.sub(r'\s+', '', xsc.group())))
        columns[len(line.split())] += 1
    if len(identities) != 11102 or any(len({r[k] for r in identities}) != 11102 for k in (0, 1)):
        raise ValueError('expected 11102 distinct raw-table identities; review version/schema')
    # No radius, surface brightness, dispersion, velocity, group, acceptance,
    # or final-PV membership value has been converted, scored or summarized.
    args.output_dir.mkdir(parents=False, exist_ok=False)
    with (args.output_dir/'identities.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('sixdf_id', 'two_mass_id'))
        writer.writerows(identities)
    result = dict(status='AUTHOR_PRE_FINAL_CUT_TABLE_IDENTITIES_AVAILABLE_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        archive=str(ARCHIVE), archive_sha256=sha, archive_bytes=len(content),
        official_landing='https://academic.oup.com/mnras/article/443/2/1231/1060810',
        outer_members=members, nested_table=table_name, table_bytes=len(table),
        table_sha256=hashlib.sha256(table).hexdigest(), header=header,
        rows=len(identities), distinct_sixdf_ids=11102, distinct_two_mass_ids=11102,
        whitespace_field_counts=dict(columns), acceptance_bits_parsed=False,
        published_reject_retention_is_row_verified=False,
        FP_or_distance_values_parsed=False, heldout_scored=False,
        frozen_split_changed=False, source_selection_calibrated=False,
        source_identity_split_linkage_complete=False, live_target_changed=False,
        R2_complete=False, elapsed_seconds=time.monotonic()-started,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (args.output_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
