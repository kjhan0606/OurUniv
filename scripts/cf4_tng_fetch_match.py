"""Fetch the official TNG100-1 hydro-to-Dark matching file once, with a key file."""

import argparse
import os
from pathlib import Path
from urllib.request import Request, urlopen

import h5py


URL = "https://www.tng-project.org/api/TNG100-1/files/subhalo_matching_to_dark.hdf5"


def fetch(key_file: Path, output: Path):
    if key_file.stat().st_mode & 0o077:
        raise PermissionError("API key file must have mode 0600 or stricter")
    key = key_file.read_text().strip()
    if not key:
        raise ValueError("empty API key file")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refuse to replace existing complete or partial download")
    request = Request(URL, headers={"API-Key": key})
    with urlopen(request, timeout=60) as response, partial.open("xb") as stream:
        size = response.headers.get("Content-Length")
        written = 0
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
            written += len(block)
        if size is not None and written != int(size):
            raise IOError("download length mismatch")
    with h5py.File(partial, "r") as source:
        group = source["Snapshot_99"]
        a = group["SubhaloIndexDark_LHaloTree"]
        b = group["SubhaloIndexDark_SubLink"]
        if a.shape != b.shape or len(a.shape) != 1 or a.shape[0] != 4371211:
            raise ValueError("unexpected z=0 matching-table shape")
        nrows = a.shape[0]
    os.rename(partial, output)
    print(f"verified matching table: {output}, {written} bytes, {nrows} z=0 rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.key_file, args.output)


if __name__ == "__main__":
    main()
