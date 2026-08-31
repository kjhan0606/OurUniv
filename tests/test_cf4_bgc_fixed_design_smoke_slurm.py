import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "cf4_bgc_fixed_design_smoke_execution_v1.json"
RUNNER = ROOT / "scripts" / "run_cf4_bgc_fixed_design_smoke_v1.sbatch"


def test_execution_contract_is_single_fixed_design_no_claim_smoke():
    config = json.loads(CONFIG.read_text())
    assert config["status"] == "USER_APPROVED_SINGLE_FIXED_DESIGN_TRUTH_MOCK_SMOKE"
    authorization = config["authorization"]
    assert authorization["single_Slurm_execution_authorized"] is True
    assert authorization["development_truth_seed_count_authorized"] == 1
    assert authorization["population_selection_mock_authorized"] is False
    assert authorization["development_64_mock_execution_authorized"] is False
    assert authorization["untouched_256_mock_validation_authorized"] is False
    assert authorization["retry_authorized"] is False
    smoke = config["smoke_contract"]
    assert smoke["selection_semantics"] == "observed_grouped_CF4_fixed_design_conditioned"
    assert smoke["mock_datum"] == "u_mock=A*s_truth+B*q_truth+epsilon"
    assert (smoke["grid_N"], smoke["cell_size_cMpc_h"]) == (32, 12.0)
    assert smoke["development_truth_seed_count"] == 1
    assert smoke["posterior_draw_count"] == 4
    assert smoke["canonical_independent_real_mode_count"] == 8538
    assert smoke["science_claim_allowed"] is False


def test_source_and_input_hashes_are_current():
    config = json.loads(CONFIG.read_text())
    for section in ("input_bindings", "source_bindings"):
        for record in config[section].values():
            expected = record.get("sha256", record.get("file_sha256"))
            assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == expected


def test_slurm_runner_matches_resource_and_controller_contract():
    config = json.loads(CONFIG.read_text())
    source = RUNNER.read_text()
    execution = config["execution"]
    assert "#SBATCH --partition=a10" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=4096M" in source
    assert "#SBATCH --time=00:30:00" in source
    assert execution["memory_request_MiB"] >= 1.2 * execution["memory_expected_peak_MiB"]
    assert execution["maximum_submissions"] == execution["maximum_executions"] == 1
    assert "SUBMISSION_CONTROLLER\" == syntax" in source
    assert "host_name\" != syntax" in source
    assert "host_name\" != syn101" in source
    assert "scripts/tripwire/**" in source
    assert "JAX_PLATFORMS=cpu" in source
    assert "--implementation-commit \"$EXPECTED_COMMIT\"" in source
    assert "renameat2" not in source
    assert "pgrep" not in source


def test_runner_binds_exact_execution_config_hash():
    source = RUNNER.read_text()
    expected = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert f"config_sha={expected}" in source
