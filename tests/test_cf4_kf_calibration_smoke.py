import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cf4_kf_calibration_smoke as smoke  # noqa: E402


CONFIG = ROOT / "config" / "cf4_kf_calibration_smoke_execution_v1.json"
MANIFEST = ROOT / "config" / "cf4_kf_bin_manifest_v1.json"
COMMIT = "a" * 40


def _bound_config(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    value = json.loads(CONFIG.read_text())
    for record in value["source_bindings"].values():
        path = ROOT / record["path"]
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = tmp_path / "config.json"
    path.write_bytes(smoke.canonical_json_bytes(value))
    return path


def test_smoke_run_and_validate_are_fail_closed(tmp_path):
    config = _bound_config(tmp_path)
    output = tmp_path / "staging"
    output.mkdir()
    result = smoke.run_smoke(config, MANIFEST, output, COMMIT)
    audit = smoke.validate_smoke(output, config, MANIFEST)
    assert result["status"] == "SMOKE_PASS"
    assert audit["status"] == "PASS"
    assert result["metrics"]["CF4_selection_noise_truth_mock_provenance_validated"] is False
    assert result["metrics"]["development_science_metric_allowed"] is False
    assert not any(result["metrics"]["strict_gate_before_geometry"])
    assert not any(result["metrics"]["strict_gate_intersection_with_geometry"])
    assert (output / "COMPLETE").exists()


def test_source_hash_tamper_and_existing_output_fail(tmp_path):
    config = _bound_config(tmp_path)
    value = json.loads(config.read_text())
    value["source_bindings"]["calibration_implementation"]["sha256"] = "0" * 64
    config.write_bytes(smoke.canonical_json_bytes(value))
    output = tmp_path / "staging"
    output.mkdir()
    with pytest.raises(smoke.SmokeError, match="source SHA256 mismatch"):
        smoke.run_smoke(config, MANIFEST, output, COMMIT)
    (output / "existing").write_text("preserve")
    with pytest.raises(smoke.SmokeError, match="must exist and be empty"):
        smoke.run_smoke(_bound_config(tmp_path / "again"), MANIFEST, output, COMMIT)


def test_validation_rejects_result_tamper_even_with_canonical_json(tmp_path):
    config = _bound_config(tmp_path)
    output = tmp_path / "staging"
    output.mkdir()
    smoke.run_smoke(config, MANIFEST, output, COMMIT)
    result = json.loads((output / "result.json").read_text())
    result["metrics"]["coverage68_status"] = "EVALUATED"
    (output / "result.json").write_bytes(smoke.canonical_json_bytes(result))
    with pytest.raises(smoke.SmokeError):
        smoke.validate_smoke(output, config, MANIFEST)
