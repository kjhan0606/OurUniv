import json
from pathlib import Path

import numpy as np
import pytest

from cf4_lg_highk_covariance_cache import (
    build_covariance_cache,
    load_covariance_for_schedule_row,
    validate_covariance_cache,
)


def _synthetic_schedule(path: Path, keys: np.ndarray) -> None:
    np.savez(path, schedule_index=np.arange(len(keys)), keys=keys)


def test_cache_round_trip_preserves_exact_schedule_mapping(tmp_path: Path) -> None:
    keys = np.asarray([
        [6, 6, 6, 2, 0, 0],
        [3, 7, 4, 0, 2, 0],
        [6, 6, 6, 2, 0, 0],
    ])
    schedule = tmp_path / "schedule.npz"
    density_filter = tmp_path / "filter.npy"
    cache = tmp_path / "cache.npz"
    _synthetic_schedule(schedule, keys)
    # A real-input rFFT is enough to exercise the same exact CPU cache path;
    # the test deliberately does not import JAX, PM, or FoF.
    np.save(density_filter, np.fft.rfftn(np.ones((12, 12, 12)), norm="ortho"))
    result = build_covariance_cache(
        schedule_path=schedule, filter_path=density_filter, output_path=cache,
        coarse_n=4,
    )
    assert result["diagnostics"]["unique_key_count"] == 2
    key, points, covariance = load_covariance_for_schedule_row(cache, 2)
    np.testing.assert_array_equal(key, keys[0])
    assert points.shape == (14, 3)
    assert covariance.shape == (14, 14)
    np.testing.assert_allclose(covariance, covariance.T)


def test_cache_rejects_schedule_with_changed_row_mapping(tmp_path: Path) -> None:
    keys = np.asarray([[6, 6, 6, 2, 0, 0], [3, 7, 4, 0, 2, 0]])
    schedule = tmp_path / "schedule.npz"
    changed_schedule = tmp_path / "changed.npz"
    density_filter = tmp_path / "filter.npy"
    cache = tmp_path / "cache.npz"
    _synthetic_schedule(schedule, keys)
    _synthetic_schedule(changed_schedule, keys[::-1])
    np.save(density_filter, np.fft.rfftn(np.ones((12, 12, 12)), norm="ortho"))
    build_covariance_cache(
        schedule_path=schedule, filter_path=density_filter, output_path=cache,
        coarse_n=4,
    )
    with pytest.raises(ValueError, match="does not map"):
        validate_covariance_cache(cache, schedule_path=changed_schedule, filter_path=density_filter)


def test_cache_result_is_atomic_and_records_no_field(tmp_path: Path) -> None:
    keys = np.asarray([[6, 6, 6, 2, 0, 0]])
    schedule, density_filter = tmp_path / "schedule.npz", tmp_path / "filter.npy"
    cache, result_path = tmp_path / "cache.npz", tmp_path / "result.json"
    _synthetic_schedule(schedule, keys)
    np.save(density_filter, np.fft.rfftn(np.ones((12, 12, 12)), norm="ortho"))
    build_covariance_cache(
        schedule_path=schedule, filter_path=density_filter, output_path=cache,
        coarse_n=4, result_path=result_path,
    )
    result = json.loads(result_path.read_text())
    assert result["status"] == "complete_exact_covariance_cache"
    assert result["field_generated"] is False
    assert result["cache_validated"] is True


def test_cache_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    keys = np.asarray([[6, 6, 6, 2, 0, 0]])
    schedule, density_filter = tmp_path / "schedule.npz", tmp_path / "filter.npy"
    cache = tmp_path / "cache.npz"
    _synthetic_schedule(schedule, keys)
    np.save(density_filter, np.fft.rfftn(np.ones((12, 12, 12)), norm="ortho"))
    build_covariance_cache(
        schedule_path=schedule, filter_path=density_filter, output_path=cache,
        coarse_n=4,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_covariance_cache(
            schedule_path=schedule, filter_path=density_filter,
            output_path=cache, coarse_n=4,
        )
