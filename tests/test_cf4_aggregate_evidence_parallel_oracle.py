import json
from pathlib import Path
import zipfile

import numpy as np

from cf4_aggregate_evidence_oracle import PRODUCTION_PARENT_SEEDS, logmeanexp_parent
from cf4_aggregate_evidence_parallel_oracle import (
    INSIDE_CONTROL_ROWS,
    OUTSIDE_CONTROL_ROWS,
    AppendOnlyEvidenceCache,
    evaluate_fixed_parent_blocks,
    fixed_parent_blocks,
    run_sealed_regression_control,
)


REGRESSION_ARRAYS = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1/arrays.npz"
)


def _entries():
    return [{"seed": seed} for seed in PRODUCTION_PARENT_SEEDS]


def _synthetic_block_worker(task):
    start, entries, keys = task
    seeds = tuple(row["seed"] for row in entries)
    key_signal = np.sum(keys.astype(np.float64), axis=1)
    values = key_signal[:, None] + np.asarray(seeds, dtype=np.float64)[None]
    return start, seeds, values


def test_fixed_parent_blocks_are_exactly_eight_ordered_blocks_of_32():
    blocks = fixed_parent_blocks(_entries())
    assert len(blocks) == 8
    assert [start for start, _ in blocks] == list(range(0, 256, 32))
    assert [tuple(row["seed"] for row in block) for _, block in blocks] == [
        tuple(range(3193 + 32 * index, 3193 + 32 * (index + 1)))
        for index in range(8)
    ]
    bad = _entries()
    bad[40], bad[41] = bad[41], bad[40]
    with np.testing.assert_raises_regex(RuntimeError, "exactly seeds"):
        fixed_parent_blocks(bad)


def test_parallel_and_sequential_parent_reassembly_are_bitwise_equal():
    keys = np.asarray([
        [12, 20, 31, 3, 0, 0],
        [12, 21, 31, 3, 0, 0],
        [13, 20, 31, 2, 2, 0],
    ], dtype=np.int16)
    sequential = evaluate_fixed_parent_blocks(
        keys, _entries(), _synthetic_block_worker, parallel=False
    )
    parallel = evaluate_fixed_parent_blocks(
        keys, _entries(), _synthetic_block_worker, parallel=True
    )
    np.testing.assert_array_equal(parallel, sequential)
    np.testing.assert_array_equal(
        parallel[0], np.sum(keys[0]) + np.asarray(PRODUCTION_PARENT_SEEDS)
    )


def test_cache_shards_are_uncompressed_atomic_immutable_and_cross_unique(tmp_path):
    cache = AppendOnlyEvidenceCache(tmp_path / "cache")
    keys = np.asarray([
        [1, 2, 3, 1, 0, 0],
        [1, 2, 4, 1, 0, 0],
    ], dtype=np.int16)
    log_z = np.arange(2 * 256, dtype=np.float64).reshape(2, 256) / 100.0
    shard = cache.append(keys, log_z)
    path = Path(shard.path)
    assert path.is_file()
    assert not list(path.parent.glob(".*.tmp"))
    with zipfile.ZipFile(path) as archive:
        assert all(row.compress_type == zipfile.ZIP_STORED for row in archive.infolist())
    with np.load(path, allow_pickle=False) as item:
        assert tuple(item.files) == ("keys", "log_Z", "log_Z_bar")
        np.testing.assert_array_equal(item["keys"], keys)
        np.testing.assert_array_equal(item["log_Z"], log_z)
        np.testing.assert_array_equal(item["log_Z_bar"], logmeanexp_parent(log_z))
    before = sorted(path.parent.iterdir())
    with np.testing.assert_raises_regex(RuntimeError, "cross-shard duplicate"):
        cache.append(keys[1:], log_z[1:])
    assert sorted(path.parent.iterdir()) == before
    manifest, digest = cache.seal()
    value = json.loads(manifest.read_text())
    assert len(digest) == 64
    assert value["shard_count"] == 1
    assert value["total_row_count"] == 2
    assert value["restart_or_checkpoint_imported"] is False
    with np.testing.assert_raises_regex(RuntimeError, "already sealed"):
        cache.append(
            np.asarray([[2, 2, 4, 1, 0, 0]], dtype=np.int16),
            np.zeros((1, 256), dtype=np.float64),
        )
    with np.testing.assert_raises(FileExistsError):
        AppendOnlyEvidenceCache(path.parent)


def test_control_uses_exact_frozen_rows_then_discards_namespace(tmp_path):
    with np.load(REGRESSION_ARRAYS, allow_pickle=False) as item:
        key_blocks = (
            item["inside_keys"][INSIDE_CONTROL_ROWS],
            item["outside_keys"][OUTSIDE_CONTROL_ROWS],
        )
        value_blocks = (
            item["inside_direct_log_Z"][INSIDE_CONTROL_ROWS],
            item["outside_direct_log_Z"][OUTSIDE_CONTROL_ROWS],
        )
        lookup = {
            tuple(int(value) for value in key): row.copy()
            for keys, rows in zip(key_blocks, value_blocks)
            for key, row in zip(keys, rows)
        }
        inside_lookup = {
            tuple(int(value) for value in key) for key in key_blocks[0]
        }

    class Cache:
        def __init__(self):
            self.evaluated_covariance_keys = 0
            self.evaluation_batches = 0

    class Evaluator:
        def __init__(self, control):
            self.control = control
            self.covariance_cache = Cache()
            self.closed = False
            self.close_count = 0

        def __call__(self, keys):
            values = [tuple(int(item) for item in key) for key in keys]
            self.covariance_cache.evaluated_covariance_keys = len(values)
            self.covariance_cache.evaluation_batches = 1
            actual = np.stack([lookup[key] for key in values])
            if self.control:
                offsets = np.asarray([
                    1e-12 if key in inside_lookup else 2e-12 for key in values
                ])
                actual = actual + offsets[:, None]
            return values, actual

        def close(self):
            self.closed = True
            self.close_count += 1

    created = []

    def factory():
        evaluator = Evaluator(control=not created)
        created.append(evaluator)
        return evaluator

    result, production_evaluator, production = run_sealed_regression_control(
        factory, REGRESSION_ARRAYS, tmp_path / "namespaces"
    )
    assert abs(result.inside_max_abs_difference - 1e-12) < 5e-14
    assert abs(result.outside_max_abs_difference - 2e-12) < 5e-14
    assert result.outside_max_abs_difference > result.inside_max_abs_difference
    assert result.control_cache_discarded is True
    assert result.control_evaluator_discarded is True
    assert result.covariance_cache_identity_distinct is True
    assert result.production_covariance_cached_key_count == 0
    assert result.production_covariance_evaluation_batches == 0
    assert result.production_cache_empty is True
    assert created[0].closed is True
    assert created[0].close_count == 1
    assert created[0] is not production_evaluator
    assert created[0].covariance_cache is not production_evaluator.covariance_cache
    assert not (tmp_path / "namespaces/control_cache").exists()
    assert (tmp_path / "namespaces/production_cache").is_dir()
    assert production.shard_count == 0
    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["global_unique_key_count"] == 24
    assert summary["inside_row_count"] == 16
    assert summary["outside_row_count"] == 8
    assert summary["selection_sha256"] == result.selection_sha256
    assert len(result.summary_sha256) == 64


def _control_expected_lookup():
    with np.load(REGRESSION_ARRAYS, allow_pickle=False) as item:
        keys = np.concatenate((
            item["inside_keys"][INSIDE_CONTROL_ROWS],
            item["outside_keys"][OUTSIDE_CONTROL_ROWS],
        ))
        rows = np.concatenate((
            item["inside_direct_log_Z"][INSIDE_CONTROL_ROWS],
            item["outside_direct_log_Z"][OUTSIDE_CONTROL_ROWS],
        ))
    return {
        tuple(int(value) for value in key): row
        for key, row in zip(keys, rows)
    }


def test_control_evaluator_and_threshold_failures_close_pool_exactly_once(tmp_path):
    lookup = _control_expected_lookup()

    class Cache:
        evaluated_covariance_keys = 0
        evaluation_batches = 0

    class Evaluator:
        def __init__(self, mode):
            self.mode = mode
            self.covariance_cache = Cache()
            self.close_count = 0

        def __call__(self, keys):
            if self.mode == "evaluator":
                raise RuntimeError("synthetic evaluator failure")
            values = [tuple(int(item) for item in key) for key in keys]
            return values, np.stack([lookup[key] for key in values]) + 1.0

        def close(self):
            self.close_count += 1

    for mode, message in (
        ("evaluator", "synthetic evaluator failure"),
        ("threshold", "control failed"),
    ):
        created = []

        def factory():
            value = Evaluator(mode)
            created.append(value)
            return value

        with np.testing.assert_raises_regex(RuntimeError, message):
            run_sealed_regression_control(
                factory, REGRESSION_ARRAYS, tmp_path / mode
            )
        assert len(created) == 1
        assert created[0].close_count == 1


def test_control_cache_failure_closes_pool_exactly_once(tmp_path, monkeypatch):
    lookup = _control_expected_lookup()

    class Cache:
        evaluated_covariance_keys = 0
        evaluation_batches = 0

    class Evaluator:
        def __init__(self):
            self.covariance_cache = Cache()
            self.close_count = 0

        def __call__(self, keys):
            values = [tuple(int(item) for item in key) for key in keys]
            return values, np.stack([lookup[key] for key in values])

        def close(self):
            self.close_count += 1

    created = []

    def factory():
        value = Evaluator()
        created.append(value)
        return value

    def fail_append(*args, **kwargs):
        raise RuntimeError("synthetic cache failure")

    monkeypatch.setattr(AppendOnlyEvidenceCache, "append", fail_append)
    with np.testing.assert_raises_regex(RuntimeError, "synthetic cache failure"):
        run_sealed_regression_control(
            factory, REGRESSION_ARRAYS, tmp_path / "cache_failure"
        )
    assert len(created) == 1
    assert created[0].close_count == 1
