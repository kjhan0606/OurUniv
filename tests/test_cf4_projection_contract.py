import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_projection_contract import (
    add_restriction_adjoint,
    contract_metadata,
    normalized_errors,
    prolong_white_field,
    restrict_spectrum,
    restrict_white_field,
    restriction_adjoint_spectrum,
    white_moments,
)
from cf4_projection_nyquist_control import (
    spatial_from_output_rfft,
    variance_preserving_projection_rfft,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restriction_adjoint_is_exact_right_inverse():
    rng = np.random.default_rng(31)
    target = rng.standard_normal((8, 8, 8))
    target_fft = np.fft.fftn(target, norm="ortho")
    prolonged_fft = restriction_adjoint_spectrum(target_fft, 24)
    roundtrip_fft = restrict_spectrum(prolonged_fft, 8)
    np.testing.assert_allclose(roundtrip_fft, target_fft, rtol=2e-15, atol=2e-15)


def test_restriction_and_prolongation_obey_adjoint_identity():
    rng = np.random.default_rng(37)
    source_fft = np.fft.fftn(rng.standard_normal((24, 24, 24)), norm="ortho")
    target_fft = np.fft.fftn(rng.standard_normal((8, 8, 8)), norm="ortho")
    left = np.vdot(restrict_spectrum(source_fft, 8), target_fft)
    right = np.vdot(source_fft, restriction_adjoint_spectrum(target_fft, 24))
    np.testing.assert_allclose(left, right, rtol=2e-15, atol=2e-14)


def test_prolonged_fields_roundtrip_to_identical_coarse_field():
    rng = np.random.default_rng(41)
    coarse = rng.standard_normal((8, 8, 8)).astype(np.float32)
    fine_a = prolong_white_field(coarse, 24, 101)
    fine_b = prolong_white_field(coarse, 24, 102)
    error_a = normalized_errors(restrict_white_field(fine_a, 8), coarse)
    error_b = normalized_errors(restrict_white_field(fine_b, 8), coarse)
    assert error_a["relative_RMS"] < 1e-7
    assert error_a["maximum_normalized_error"] < 5e-7
    assert error_b["relative_RMS"] < 1e-7
    assert error_b["maximum_normalized_error"] < 5e-7
    assert np.sqrt(np.mean((fine_a.astype(float) - fine_b.astype(float)) ** 2)) > 1.0


def test_full_spectrum_restriction_matches_rfft_control_implementation():
    rng = np.random.default_rng(43)
    source = rng.standard_normal((24, 24, 24))
    full_contract = restrict_white_field(source, 8)
    rfft_contract = spatial_from_output_rfft(
        variance_preserving_projection_rfft(np.fft.rfftn(source), 24, 8), 8
    )
    np.testing.assert_allclose(full_contract, rfft_contract, rtol=2e-6, atol=2e-6)


def test_conditional_prolongation_has_white_marginal_when_coarse_is_white():
    rng = np.random.default_rng(47)
    moments = []
    for sample in range(32):
        coarse = rng.standard_normal((8, 8, 8))
        fine = prolong_white_field(coarse, 24, 1000 + sample)
        moments.append(white_moments(fine))
    assert abs(np.mean([row["mean"] for row in moments])) < 0.01
    assert abs(np.mean([row["std"] for row in moments]) - 1.0) < 0.01
    assert abs(np.mean([row["skew"] for row in moments])) < 0.02
    assert abs(np.mean([row["excess_kurtosis"] for row in moments])) < 0.04


def test_adjoint_addition_does_not_mutate_input():
    source = np.zeros((12, 12, 12), dtype=np.complex128)
    target = np.ones((4, 4, 4), dtype=np.complex128)
    result = add_restriction_adjoint(source, target)
    assert np.all(source == 0.0)
    assert np.any(result != 0.0)


def test_contract_metadata_keeps_v8_immutable():
    metadata = contract_metadata()
    assert metadata["FFT_normalization"] == "ortho"
    assert metadata["prolongation"] == "z + R* (y - R z)"
    assert metadata["legacy_V8_products_modified"] is False


def test_full_size_contract_program_is_hash_pinned_and_nonselective():
    program = json.loads((
        ROOT / "config/cf4_projection_contract_control_program.json"
    ).read_text())
    for key in ("implementation", "contract_implementation", "rfft_control"):
        item = program[key]
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    firewall = program["information_firewall"]
    assert firewall["numerical_contract_control_only"] is True
    assert firewall["CF4_likelihood_or_structure_metric_computed"] is False
    assert firewall["fine_fields_persisted"] is False
    assert firewall["candidate_or_proposal_constructed"] is False
    assert firewall["parent_or_seed_selected"] is False
    assert firewall["PM_or_RAMSES_authorized"] is False


def test_contract_control_scripts_are_single_shot_without_process_polling():
    paths = [
        ROOT / "scripts/run_cf4_projection_contract_control_lageunha.sh",
        ROOT / "scripts/launch_cf4_projection_contract_control_lageunha.sh",
        ROOT / "scripts/status_cf4_projection_contract_control.sh",
    ]
    for path in paths:
        text = path.read_text()
        assert "pgrep" not in text
        assert "while " not in text
        assert "sleep " not in text


def test_contract_result_record_authorizes_design_but_not_generation():
    record = json.loads((
        ROOT / "config/cf4_projection_contract_control_result_record.json"
    ).read_text())
    assert record["status"] == "complete_pass_future_projection_contract"
    assert sha256_file(ROOT / record["lineage"]["program"]) == (
        record["lineage"]["program_sha256"]
    )
    assert sha256_file(ROOT / record["lineage"]["contract_implementation"]) == (
        record["lineage"]["contract_implementation_sha256"]
    )
    assert record["gates"]["contract_pass"] is True
    decision = record["decision"]
    assert decision["paired_projection_contract_authorized"] is True
    assert decision["independent_parent_architecture_design_authorized"] is True
    assert decision["candidate_generation_authorized"] is False
    assert decision["seed_selection_authorized"] is False
    assert decision["PM_or_RAMSES_authorized"] is False
