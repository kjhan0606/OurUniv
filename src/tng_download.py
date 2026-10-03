#!/usr/bin/env python
"""Authenticated, resumable downloader for TNG100-1 snapshot 99.

The TNG API authenticates the first request and redirects to a short-lived
signed data-server URL.  As of 2026-07, forwarding the ``api-key`` header to
that data server produces HTTP 403, so this downloader deliberately removes
the header after resolving each redirect.

Examples
--------
python src/tng_download.py groupcat
python src/tng_download.py snapshot --workers 2
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests


SIMULATION = "TNG100-1"
SNAPSHOT = 99
API_ROOT = "https://www.tng-project.org/api"
DEFAULT_ROOT = Path("/scratch/kjhan/IllustrisTNG/TNG100-1")
DEFAULT_KEY_FILE = Path("/home/kjhan/.config/illustris-tng/api_key")
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class DownloadResult:
    index: int
    path: str
    size_bytes: int
    elapsed_seconds: float
    resumed_from_bytes: int
    skipped: bool


def output_directory(root: Path, kind: str) -> Path:
    leaf = "groups_099" if kind == "groupcat" else "snapdir_099"
    return root / "output" / leaf


def output_name(kind: str, index: int) -> str:
    stem = "fof_subhalo_tab_099" if kind == "groupcat" else "snap_099"
    return f"{stem}.{index}.hdf5"


def read_key(path: Path) -> str:
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"{path} must not be accessible by group/other")
    key = path.read_text().strip()
    if not key:
        raise ValueError(f"{path} is empty")
    return key


def list_files(kind: str, key: str) -> list[str]:
    url = f"{API_ROOT}/{SIMULATION}/files/{kind}-{SNAPSHOT}/"
    response = requests.get(
        url,
        headers={"api-key": key, "Accept": "application/json"},
        timeout=(30, 120),
    )
    response.raise_for_status()
    files = response.json()["files"]
    if len(files) != 448:
        raise RuntimeError(f"expected 448 {kind} chunks, received {len(files)}")
    return [url.replace("http://", "https://", 1) for url in files]


def chunk_index(url: str) -> int:
    match = re.search(r"\.(\d+)\.hdf5$", url)
    if match is None:
        raise ValueError(f"cannot parse chunk index from {url}")
    return int(match.group(1))


def signed_url(api_url: str, key: str) -> str:
    response = requests.get(
        api_url,
        headers={"api-key": key},
        allow_redirects=False,
        stream=True,
        timeout=(30, 120),
    )
    try:
        if response.status_code not in (301, 302, 303, 307, 308):
            response.raise_for_status()
            raise RuntimeError(f"expected redirect for {api_url}")
        location = response.headers.get("location")
        if not location:
            raise RuntimeError(f"redirect from {api_url} has no Location")
        return urljoin(api_url, location)
    finally:
        response.close()


def total_from_response(response: requests.Response, start: int) -> int:
    content_range = response.headers.get("content-range")
    if content_range:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
    length = response.headers.get("content-length")
    if length is None:
        raise RuntimeError("data response has neither Content-Range nor Content-Length")
    return start + int(length) if response.status_code == 206 else int(length)


def download_one(
    api_url: str,
    kind: str,
    destination: Path,
    key: str,
    retries: int,
) -> DownloadResult:
    index = chunk_index(api_url)
    final = destination / output_name(kind, index)
    partial = final.with_suffix(final.suffix + ".part")
    if final.is_file():
        return DownloadResult(
            index, str(final), final.stat().st_size, 0.0, 0, True
        )

    started = time.time()
    initial_size = partial.stat().st_size if partial.exists() else 0
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            # Do not send the API key to the redirected data host.
            url = signed_url(api_url, key)
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=False,
                timeout=(30, 300),
            ) as response:
                if response.status_code == 416 and partial.exists():
                    total_match = re.search(
                        r"\*/(\d+)$", response.headers.get("content-range", "")
                    )
                    if total_match and partial.stat().st_size == int(total_match.group(1)):
                        partial.replace(final)
                        return DownloadResult(
                            index,
                            str(final),
                            final.stat().st_size,
                            time.time() - started,
                            initial_size,
                            False,
                        )
                response.raise_for_status()
                if start and response.status_code != 206:
                    start = 0
                total = total_from_response(response, start)
                mode = "ab" if start and response.status_code == 206 else "wb"
                with partial.open(mode) as handle:
                    for block in response.iter_content(CHUNK_BYTES):
                        if block:
                            handle.write(block)
            size = partial.stat().st_size
            if size != total:
                raise IOError(f"{partial}: received {size} bytes, expected {total}")
            partial.replace(final)
            return DownloadResult(
                index,
                str(final),
                size,
                time.time() - started,
                initial_size,
                False,
            )
        except (OSError, requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"chunk {index} failed after {retries} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("groupcat", "snapshot"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument(
        "--max-files", type=int, default=None, help="smoke-test prefix only"
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    key = read_key(args.key_file)
    files = list_files(args.kind, key)
    if args.max_files is not None:
        files = files[: args.max_files]
    destination = output_directory(args.root, args.kind)
    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"[start] {SIMULATION} {args.kind} snapshot={SNAPSHOT} "
        f"files={len(files)} workers={args.workers} destination={destination}",
        flush=True,
    )

    results: list[DownloadResult] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                download_one,
                url,
                args.kind,
                destination,
                key,
                args.retries,
            ): chunk_index(url)
            for url in files
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            done = len(results)
            downloaded = sum(item.size_bytes for item in results)
            print(
                f"[{done:03d}/{len(files):03d}] chunk={result.index:03d} "
                f"size={result.size_bytes/2**20:.1f} MiB "
                f"elapsed={result.elapsed_seconds:.1f}s skipped={result.skipped} "
                f"total={downloaded/2**30:.2f} GiB",
                flush=True,
            )

    results.sort(key=lambda item: item.index)
    manifest = {
        "simulation": SIMULATION,
        "snapshot": SNAPSHOT,
        "kind": args.kind,
        "destination": str(destination),
        "files_expected": len(files),
        "files_complete": len(results),
        "bytes_complete": sum(item.size_bytes for item in results),
        "elapsed_seconds": time.time() - started,
        "key_recorded": False,
        "results": [asdict(item) for item in results],
    }
    manifest_path = destination / f"download_{args.kind}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[complete] {args.kind}: {manifest['bytes_complete']/2**30:.2f} GiB "
        f"in {manifest['elapsed_seconds']/3600:.2f} h; {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
