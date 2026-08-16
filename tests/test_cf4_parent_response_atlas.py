import hashlib
import json
from pathlib import Path

import numpy as np

import cf4_parent_response_atlas as atlas_module
from cf4_aggregate_evidence_oracle import AtlasBounds, parent_response_grid
from cf4_parent_response_atlas import (
    atomic_npy,
    atlas_parent_case,
    validate_program,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_atomic_npy_is_exclusive_and_preserves_float64(tmp_path):
    path = tmp_path / "atlas.npy"
    value = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    atomic_npy(path, value)
    actual = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(actual, value)
    assert actual.dtype == np.float64
    with np.testing.assert_raises(FileExistsError):
        atomic_npy(path, value)


def test_parent_atlas_case_matches_exact_response_subcube(tmp_path):
    rng = np.random.default_rng(94)
    seed = 3193
    coarse = rng.normal(size=(4, 4, 4)).astype(np.float32)
    parent_path = tmp_path / "parent.npz"
    np.savez(parent_path, sample_seed=seed, s_out=coarse)
    kernel = rng.normal(size=(12, 12, 12))
    filter_full = np.fft.fftn(kernel, norm="ortho")
    bounds = AtlasBounds(
        relative_min=(-1, -1, -1),
        relative_max=(1, 1, 1),
        padded_min=(-2, -2, -2),
        padded_max=(2, 2, 2),
    )
    output = tmp_path / "output"
    output.mkdir()
    atlas_module._WORKER_FILTER = filter_full
    atlas_module._WORKER_BOUNDS = bounds
    atlas_module._WORKER_OUTPUT_DIRECTORY = output
    try:
        entry = atlas_parent_case({
            "seed": seed,
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
        })
    finally:
        atlas_module._WORKER_FILTER = None
        atlas_module._WORKER_BOUNDS = None
        atlas_module._WORKER_OUTPUT_DIRECTORY = None
    shard = Path(entry["atlas"])
    assert entry["seed"] == seed
    assert entry["atlas_sha256"] == sha256_file(shard)
    assert entry["shape"] == [5, 5, 5]
    assert entry["dtype"] == "float64"
    response = parent_response_grid(coarse, filter_full)
    indices = np.mod(np.arange(-2, 3) + 6, 12)
    expected = response[np.ix_(indices, indices, indices)]
    np.testing.assert_array_equal(np.load(shard, allow_pickle=False), expected)


def frozen_program_contract():
    design_path = ROOT / "config/cf4_aggregate_evidence_annealed_smc_design.json"
    design = json.loads(design_path.read_text())
    fixed = design["fixed_inputs"]
    atlas = design["oracle_and_cache"]["parent_response_atlas"]
    paths = [
        "src/cf4_aggregate_evidence_oracle.py",
        "src/cf4_parent_response_atlas.py",
        "src/cf4_peak_evidence_phase_cache.py",
        "src/cf4_projection_contract.py",
        "config/cf4_aggregate_evidence_annealed_smc_design.json",
    ]
    program_path = ROOT / "config/test_response_atlas_program.json"
    return program_path, {
        "status": "frozen_before_response_atlas_construction",
        "design": {
            "path": "config/cf4_aggregate_evidence_annealed_smc_design.json",
            "sha256": sha256_file(design_path),
        },
        "atlas": {
            "prior_mean_mpc_h": fixed["midpoint_prior_mean_mpc_h"],
            "prior_sigma_mpc_h": fixed["midpoint_prior_sigma_mpc_h"],
            "dx_mpc_h": fixed["dx_mpc_h"],
            "sigma_extent": 10.0,
            "padding_cells": atlas["point_padding_cells"],
            "shape": [101, 101, 101],
            "dtype": atlas["dtype"],
            "outside_atlas_policy": atlas["outside_atlas_policy"],
        },
        "parents": {
            "seed_range_inclusive": fixed["parent_seed_range_inclusive"],
            "count": fixed["parent_count"],
        },
        "density_filter": fixed["density_filter"],
        "reference_calibration": {
            **fixed["reference_calibration"],
            "status": "complete_reference_calibration_parent3429_pass",
        },
        "execution": {
            "host": "LagEunha",
            "worker_processes": 8,
            "threads_per_worker": 1,
            "process_table_polling": False,
        },
        "pinned_local_files": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in paths
        ],
        "storage": {
            "program": str(program_path),
            "directory": "/gpfs/kjhan/CF4/recon/linear_cr/test_atlas",
            "manifest": "/gpfs/kjhan/CF4/recon/linear_cr/test_atlas/manifest.json",
        },
    }


def test_atlas_preflight_rejects_design_constant_and_source_set_changes():
    program_path, program = frozen_program_contract()
    program["atlas"]["sigma_extent"] = 9.0
    with np.testing.assert_raises_regex(RuntimeError, "constants differ"):
        validate_program(program, program_path)
    program_path, program = frozen_program_contract()
    program["execution"]["worker_processes"] = 7
    with np.testing.assert_raises_regex(RuntimeError, "execution contract"):
        validate_program(program, program_path)
    program_path, program = frozen_program_contract()
    program["pinned_local_files"] = program["pinned_local_files"][:-1]
    with np.testing.assert_raises_regex(RuntimeError, "source set"):
        validate_program(program, program_path)
