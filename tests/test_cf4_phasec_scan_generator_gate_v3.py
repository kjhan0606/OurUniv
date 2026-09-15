import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_scan_generator_gate_v3_hardware_quarantine.json"
RUNNER = ROOT / "scripts/run_cf4_phasec_scan_generator_gate_v3.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_phasec_scan_generator_gate_v3.sbatch"
RESULT_RECORD = ROOT / "config/cf4_phasec_scan_generator_gate_v3_result_record.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    return json.loads(PROGRAM.read_text())


def test_v3_changes_hardware_placement_only_and_preserves_all_eight_science_gate():
    program = load_program()
    assert [row["seed"] for row in program["assignments"]] == list(range(2026083000, 2026083008))
    assert [row["arm"] for row in program["assignments"]] == list("AABBCCDD")
    science = program["unchanged_science_contract"]
    assert science["a_nbody_maxsteps"] == [1 / 128, 1 / 256]
    assert science["density_cross_correlation_min"] == 0.999
    assert science["density_relative_L2_max"] == 0.03
    assert science["velocity_cross_correlation_min"] == 0.995
    assert science["velocity_relative_L2_max"] == 0.05
    assert science["all_eight_seeds_must_pass"] is True
    assert science["seed_replacement_or_threshold_relaxation"] is False


def test_v3_lineage_and_scope_are_frozen():
    program = load_program()
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]
    assert not any(program["scope_firewall"].values())
    for key in ("sampler", "actual_observational_data", "validation_seed", "Phase_D_or_later"):
        assert program["authorization"][key] is False


def test_v3_quarantines_syn06_and_records_every_task_device():
    program = load_program()
    quarantine = program["hardware_quarantine"]
    assert quarantine["excluded_node"] == "syn06"
    assert quarantine["expected_nodes"] == ["syn05", "syn07"]
    assert quarantine["physical_GPU_identity_recorded_per_task"] is True
    for text in (RUNNER.read_text(), AGGREGATOR.read_text()):
        assert "#SBATCH --exclude=syn06" in text
        assert "GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6" in text
        assert "scripts/tripwire/**" in text
    runner = RUNNER.read_text()
    assert "capture-device" in runner
    assert '"$host_name" == syn05 || "$host_name" == syn07' in runner
    assert "#SBATCH --array=0-7" in runner


def test_v3_memory_headroom_and_no_forbidden_process_or_manual_execution():
    execution = load_program()["execution"]
    assert execution["requested_host_memory_MiB_per_task"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB_per_task"
    ]
    assert execution["aggregate_requested_host_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_host_memory_MiB"
    ]
    assert "#SBATCH --mem=9216M" in RUNNER.read_text()
    assert "#SBATCH --mem=1024M" in AGGREGATOR.read_text()
    for text in (RUNNER.read_text(), AGGREGATOR.read_text()):
        for forbidden in ("pgrep", "renameat2", "/tmp", "blackjax", "prepare_catalog"):
            assert forbidden not in text


def test_v3_does_not_release_sampler_directly():
    decision = load_program()["decision_rule"]
    assert decision["sampler_mechanics_pilot_indices_after_PASS"] == [0, 6]
    assert decision["sampler_release_by_this_execution"] is False
    assert decision["actual_observational_posterior_allowed"] is False
    assert decision["validation_or_Phase_D_allowed"] is False


def test_v3_result_record_passes_all_eight_and_releases_only_mock_pilot_indices():
    result = json.loads(RESULT_RECORD.read_text())
    assert result["lineage"]["commit"] == "214a2fc03ec3be0bacc10b9241c026350387e046"
    summary = result["summary"]
    assert summary["valid_task_artifact_count"] == 8
    assert summary["passing_seed_count"] == 8
    assert summary["all_seed_gate_pass"] is True
    assert len(result["hardware_provenance"]) == 4
    decision = result["decision"]
    assert decision["sampler_mechanics_pilot_indices_allowed"] == [0, 6]
    assert decision["sampler_mechanics_pilot_started"] is False
    assert decision["actual_observational_posterior_allowed"] is False
    assert decision["validation_or_Phase_D_allowed"] is False
