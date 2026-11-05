#!/usr/bin/env python3
"""Fail-closed GPU visibility and VRAM guard for high-k PM recovery."""

from __future__ import annotations

import argparse
import re
import subprocess
from typing import Sequence


MINIMUM_GPU_MEMORY_MIB = 115_000


def parse_memory_totals(output: str) -> list[int]:
    """Parse the nounits CSV output from nvidia-smi without accepting noise."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    totals: list[int] = []
    for line in lines:
        if re.fullmatch(r"[0-9]+", line) is None:
            raise RuntimeError(f"invalid nvidia-smi memory.total value: {line!r}")
        totals.append(int(line))
    return totals


def validate_single_visible_gpu(
    memory_totals_mib: Sequence[int], *, minimum_mib: int = MINIMUM_GPU_MEMORY_MIB
) -> int:
    """Return the sole GPU's VRAM or reject zero/multiple/undersized GPUs."""
    if len(memory_totals_mib) != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, found {len(memory_totals_mib)}"
        )
    memory_mib = int(memory_totals_mib[0])
    if memory_mib < minimum_mib:
        raise RuntimeError(
            f"visible GPU has {memory_mib} MiB; at least {minimum_mib} MiB is required"
        )
    return memory_mib


def query_and_validate(*, minimum_mib: int = MINIMUM_GPU_MEMORY_MIB) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return validate_single_visible_gpu(
        parse_memory_totals(completed.stdout), minimum_mib=minimum_mib
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-mib", type=int, default=MINIMUM_GPU_MEMORY_MIB)
    args = parser.parse_args()
    memory_mib = query_and_validate(minimum_mib=args.minimum_mib)
    print(f"validated_visible_gpu_memory_mib={memory_mib}")


if __name__ == "__main__":
    main()
