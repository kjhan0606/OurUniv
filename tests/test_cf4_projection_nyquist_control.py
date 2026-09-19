import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_make_ic import fourier_resample_white_field
from cf4_projection_nyquist_control import (
    legacy_projection_rfft,
    normalized_errors,
    projection_geometry,
    spatial_from_output_rfft,
    variance_preserving_projection_rfft,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_helper_is_exact_production_operation():
    rng = np.random.default_rng(17)
    source = rng.standard_normal((24, 24, 24))
    source_fft = np.fft.rfftn(source)
    helper = spatial_from_output_rfft(
        legacy_projection_rfft(source_fft, 24, 8), 8
    )
    production = fourier_resample_white_field(source, 8)
    np.testing.assert_array_equal(helper, production)


def test_variance_preserving_fold_is_valid_real_rfft():
    rng = np.random.default_rng(23)
    source = rng.standard_normal((24, 24, 24))
    folded_fft = variance_preserving_projection_rfft(
        np.fft.rfftn(source), 24, 8
    )
    folded = spatial_from_output_rfft(folded_fft, 8)
    roundtrip = np.fft.rfftn(folded)
    errors = normalized_errors(roundtrip, folded_fft)
    assert errors["relative_RMS"] < 1e-7
    assert errors["maximum_normalized_error"] < 1e-6


def test_variance_preserving_fold_repairs_boundary_power_in_ensemble():
    rng = np.random.default_rng(29)
    legacy_boundary = []
    folded_boundary = []
    for _ in range(128):
        source = rng.standard_normal((24, 24, 24))
        source_fft = np.fft.rfftn(source)
        fields = [
            spatial_from_output_rfft(
                legacy_projection_rfft(source_fft, 24, 8), 8
            ),
            spatial_from_output_rfft(
                variance_preserving_projection_rfft(source_fft, 24, 8), 8
            ),
        ]
        values = []
        for field in fields:
            field_fft = np.fft.rfftn(field, norm="ortho")
            boundary = np.zeros(field_fft.shape, dtype=bool)
            boundary[4, :, :] = True
            boundary[:, 4, :] = True
            boundary[:, :, 4] = True
            weights = np.full((1, 1, 5), 2.0)
            weights[..., 0] = 1.0
            weights[..., -1] = 1.0
            weights = np.broadcast_to(weights, field_fft.shape)
            values.append(float(
                np.sum(weights[boundary] * np.abs(field_fft[boundary]) ** 2)
                / np.sum(weights[boundary])
            ))
        legacy_boundary.append(values[0])
        folded_boundary.append(values[1])
    assert abs(np.mean(folded_boundary) - 1.0) < 0.04
    assert np.mean(folded_boundary) > np.mean(legacy_boundary) + 0.15


def test_projection_geometry_marks_only_actual_output_nyquist_shells():
    geometry = projection_geometry(
        12, 24.0, 4, np.asarray([0.0, 1.3, 1.8, 2.4, np.inf])
    )
    fractions = geometry["boundary_weight_fractions"]
    assert fractions.shape == (4,)
    assert np.all((fractions >= 0.0) & (fractions <= 1.0))
    assert np.any(fractions == 0.0)
    assert np.any(fractions > 0.0)


def test_control_program_is_hash_pinned_and_firewalled():
    program = json.loads((
        ROOT / "config/cf4_projection_nyquist_control_program.json"
    ).read_text())
    implementation = ROOT / program["implementation"]["path"]
    legacy = ROOT / program["legacy_projection"]["path"]
    assert sha256_file(implementation) == program["implementation"]["sha256"]
    assert sha256_file(legacy) == program["legacy_projection"]["sha256"]
    assert program["sampling"]["count"] == len(program["sampling"]["seeds"])
    firewall = program["information_firewall"]
    assert firewall["consumed_white_field_control_only"] is True
    assert firewall["CF4_catalog_read"] is False
    assert firewall["existing_V8_field_read"] is False
    assert firewall["new_constrained_field_constructed"] is False
    assert firewall["proposal_or_seed_selected"] is False
    assert firewall["V9_or_RAMSES_authorized"] is False


def test_control_lifecycle_scripts_do_not_poll_process_table():
    scripts = [
        ROOT / "scripts/run_cf4_projection_nyquist_control_lageunha.sh",
        ROOT / "scripts/launch_cf4_projection_nyquist_control_lageunha.sh",
        ROOT / "scripts/status_cf4_projection_nyquist_control.sh",
    ]
    for script in scripts:
        text = script.read_text()
        assert "pgrep" not in text
        assert "while " not in text
        assert "sleep " not in text


def test_result_record_separates_projection_pass_from_single_parent_failure():
    record = json.loads((
        ROOT / "config/cf4_projection_nyquist_control_result_record.json"
    ).read_text())
    assert record["status"] == (
        "complete_pass_output_Nyquist_boundary_mechanism_isolated"
    )
    assert sha256_file(ROOT / record["lineage"]["program"]) == (
        record["lineage"]["program_sha256"]
    )
    assert sha256_file(ROOT / record["lineage"]["implementation"]) == (
        record["lineage"]["implementation_sha256"]
    )
    assert record["gates"]["mechanism_isolated"] is True
    decision = record["decision"]
    assert decision["future_only_variance_preserving_projection_contract_authorized"] is True
    assert decision["retroactive_V8_mutation_authorized"] is False
    assert decision["current_single_parent_N64_freeze_architecture_reopened"] is False
    assert decision["V9_or_seed_promotion_authorized"] is False
    assert decision["RAMSES_authorized"] is False
