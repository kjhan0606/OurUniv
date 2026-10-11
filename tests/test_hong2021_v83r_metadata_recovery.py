from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import hong2021_v83r_metadata_recovery as recovery
from hong2021_v18_init import sha256_file
from hong2021_v83_sample import SCHEMA


def make_ensemble(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "sample", data=np.arange(48, dtype=np.float32).reshape(2, 2, 1, 2, 2, 3)
        )
        handle.create_dataset("source_index", data=np.asarray([7, 11], dtype=np.int64))
        handle.attrs.update(
            {
                "schema": SCHEMA,
                "program_sha256": recovery.V83_PROGRAM_SHA256,
                "complete": True,
                "unchanged": "yes",
            }
        )


def test_copy_recovery_preserves_source_and_dataset_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    temporary = tmp_path / "partial" / "ensemble16.h5"
    final = tmp_path / "final" / "ensemble16.h5"
    make_ensemble(source)
    source_sha = sha256_file(source)
    source_bytes = source.stat().st_size
    row = recovery.copy_and_repair(
        source, temporary, final, source_sha, source_bytes
    )
    assert sha256_file(source) == source_sha
    assert source.stat().st_size == source_bytes
    assert row["all_dataset_bytes_identical"] is True
    assert row["sealed_source_unchanged"] is True
    assert row["only_added_attribute"] == {"diagnostic_k_h_mpc": 1.0}
    with h5py.File(source, "r") as original, h5py.File(temporary, "r") as copied:
        assert "diagnostic_k_h_mpc" not in original.attrs
        assert float(copied.attrs["diagnostic_k_h_mpc"]) == 1.0
        assert copied.attrs["unchanged"] == "yes"
        assert recovery.dataset_manifest(original) == recovery.dataset_manifest(copied)


def test_copy_recovery_refuses_hash_size_and_existing_attribute(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    make_ensemble(source)
    with pytest.raises(ValueError, match="bytes or hash"):
        recovery.copy_and_repair(
            source,
            tmp_path / "partial1" / "ensemble16.h5",
            tmp_path / "final1" / "ensemble16.h5",
            "0" * 64,
            source.stat().st_size,
        )
    with h5py.File(source, "r+") as handle:
        handle.attrs["diagnostic_k_h_mpc"] = 1.0
    with pytest.raises(ValueError, match="precondition"):
        recovery.copy_and_repair(
            source,
            tmp_path / "partial2" / "ensemble16.h5",
            tmp_path / "final2" / "ensemble16.h5",
            sha256_file(source),
            source.stat().st_size,
        )
