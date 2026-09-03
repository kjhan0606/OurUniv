from __future__ import annotations

import pytest

from cf4_q3_resolution_resource_preflight import (
    MAX_DEVICE_MEMORY_BYTES,
    PreflightInputError,
    derive_k_max,
    q3_resolution_preflight,
    resolution_point,
)


def test_k_max_matches_frozen_q2_development_point() -> None:
    assert derive_k_max(12.0) == 19
    assert derive_k_max(2.0) == 64
    assert derive_k_max(0.3) == 379


def test_q2_12_cmpc_h_bytes_match_record() -> None:
    point = resolution_point("development", 12.0)
    assert point.grid_size == 32
    assert point.source_particles == 192**3
    assert point.padded_array_bytes_total == 31_271_936
    assert point.fits_device_memory


def test_release_point_is_full_pm_basis_and_fits_static_memory() -> None:
    point = resolution_point("release", 2.0)
    assert point.grid_size == 192
    assert point.source_particles == 7_077_888
    assert point.padded_array_bytes_total > 0
    assert point.padded_array_bytes_total < MAX_DEVICE_MEMORY_BYTES
    assert not point.observational_information_claim_authorized


def test_deferred_03_projection_is_not_a_science_claim_and_exceeds_host() -> None:
    result = q3_resolution_preflight()
    point = result["points"][2]
    assert point["grid_size"] == 1280
    assert point["k_max"] == 379
    assert point["fits_host_memory"] is False
    assert point["observational_information_claim_authorized"] is False
    assert "separate conditional" in result["deferred_target"]


def test_invalid_grid_and_inputs_fail_closed() -> None:
    with pytest.raises(PreflightInputError):
        resolution_point("bad", 7.0)
    with pytest.raises(PreflightInputError):
        resolution_point("bad", 2.0, source_particles=0)
    with pytest.raises(PreflightInputError):
        derive_k_max(0.0)
