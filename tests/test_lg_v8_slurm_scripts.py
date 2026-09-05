from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v8_pipeline_is_one_shot_and_stops_before_ramses():
    script = (REPO / "scripts/run_lg_v8_z0_importance_slurm.sh").read_text()
    assert "--partition=h200,h100,a100" in script
    assert "a40" not in script.lower()
    assert "src/cf4_lg_peak_cr.py" in script
    assert "src/cf4_p2_screen.py" in script
    assert "src/cf4_lg_z0_importance.py score" in script
    assert "src/cf4_lg_z0_importance.py gate" in script
    assert "READY_FOR_RAMSSES_REVIEW" in script
    assert "ramses3d" not in script.lower()
    assert "RAMSES_launched=false" in script


def test_v8_pipeline_pins_program_and_implementation_hashes():
    script = (REPO / "scripts/run_lg_v8_z0_importance_slurm.sh").read_text()
    assert "6a89f5027f253282e18f21201146dde384837f0d689d725a25022def8ea7e6f2" in script
    assert "fde4384fee2ee39bea2fff7b967db6702eec9a3ad2c193d4eddd7537a57eb3f8" in script
    assert "Frozen V8 input SHA-256 mismatch" in script


def test_v8_pipeline_separates_pytest_from_the_pmwd_environment():
    script = (REPO / "scripts/run_lg_v8_z0_importance_slurm.sh").read_text()
    assert "test_python=/home/kjhan/miniconda3/bin/python" in script
    assert '"$test_python" -m pytest' in script
    assert "python=/home/kjhan/miniconda3/envs/circle/bin/python" in script
