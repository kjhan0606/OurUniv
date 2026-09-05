import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import hong2021_v80dr_metadata_recovery as recovery
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v80_sample import ENSEMBLE_SCHEMA


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v80dr_metadata_only_recovery_program.json"


def test_recovery_program_is_source_bound_and_copy_only() -> None:
    program = recovery.load_program(PROGRAM, REPO)
    assert program["engineering_only"] is True
    assert program["statistically_valid_V79_reexecution"] is False
    assert program["only_authorized_mutation"] == {
        **program["only_authorized_mutation"],
        "attribute": {"diagnostic_k_h_mpc": 1.0},
        "modify_original_ensembles": False,
    }
    assert len(program["sealed_source_ensembles"]) == 6
    assert (
        program["outputs"]["recovered_ensemble_root"]
        != program["frozen_failure_state"]["original_ensemble_root"]
    )


def make_ensemble(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "sample", data=np.arange(48, dtype=np.float32).reshape(2, 2, 1, 2, 2, 3)
        )
        handle.create_dataset("source_index", data=np.asarray([7, 11], dtype=np.int64))
        handle.attrs.update(
            {
                "schema": ENSEMBLE_SCHEMA,
                "candidate_program_sha256": recovery.V80D_PROGRAM_SHA256,
                "complete": True,
                "another_attribute": "unchanged",
            }
        )


def test_copy_recovery_preserves_source_and_all_dataset_bytes(tmp_path: Path) -> None:
    source = tmp_path / "sealed.h5"
    temporary = tmp_path / "recovered.partial" / "ensemble16.h5"
    final = tmp_path / "recovered" / "ensemble16.h5"
    make_ensemble(source)
    source_sha = sha256_file(source)
    row = recovery.copy_and_repair(source, temporary, final, source_sha)
    assert sha256_file(source) == source_sha
    assert row["sealed_source_unchanged"] is True
    assert row["all_dataset_bytes_identical"] is True
    assert row["only_added_attribute"] == {"diagnostic_k_h_mpc": 1.0}
    assert row["recovered_path"] == str(final.resolve())
    with h5py.File(source, "r") as original, h5py.File(temporary, "r") as copied:
        assert "diagnostic_k_h_mpc" not in original.attrs
        assert float(copied.attrs["diagnostic_k_h_mpc"]) == 1.0
        assert copied.attrs["another_attribute"] == "unchanged"
        assert recovery.dataset_manifest(original) == recovery.dataset_manifest(copied)


def test_copy_recovery_refuses_an_already_modified_source(tmp_path: Path) -> None:
    source = tmp_path / "sealed.h5"
    make_ensemble(source)
    with h5py.File(source, "r+") as handle:
        handle.attrs["diagnostic_k_h_mpc"] = 1.0
    with pytest.raises(ValueError, match="precondition differs"):
        recovery.copy_and_repair(
            source,
            tmp_path / "partial" / "ensemble16.h5",
            tmp_path / "final" / "ensemble16.h5",
            sha256_file(source),
        )


def test_copy_recovery_refuses_a_source_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "sealed.h5"
    make_ensemble(source)
    with pytest.raises(ValueError, match="hash differs"):
        recovery.copy_and_repair(
            source,
            tmp_path / "partial" / "ensemble16.h5",
            tmp_path / "final" / "ensemble16.h5",
            "0" * 64,
        )


def test_recovery_record_digest_contract() -> None:
    row = {
        "schema": recovery.RECORD_SCHEMA,
        "status": "complete_metadata_only_copy_recovery_evaluation_may_run_once",
        "engineering_only": True,
    }
    row["decision_digest_sha256"] = canonical_digest(row)
    encoded = json.loads(json.dumps(row))
    assert canonical_digest(encoded) == encoded["decision_digest_sha256"]


def test_frozen_evaluator_attribute_is_exactly_the_recovered_attribute() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "src/hong2021_residual_evaluate.py").read_text()
    assert 'handle.attrs["diagnostic_k_h_mpc"]' in source
    assert recovery.ADDED_ATTRIBUTE == "diagnostic_k_h_mpc"
    assert recovery.ADDED_VALUE == 1.0
