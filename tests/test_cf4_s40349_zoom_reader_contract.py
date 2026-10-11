from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reader_gate_uses_actual_ic_ceiling_and_suppresses_outputs() -> None:
    text = (ROOT / "config/ramses_cf4_lg_b3292_s40349_zoom_l12_reader_v1.nml").read_text()
    assert "nstepmax=0" in text
    assert "noutput=1" in text and "aout=1.1" in text
    assert "tout=1.d100" in text
    assert "foutput=1000000000" in text and "fbackup=1000000000" in text
    assert "levelmin=9" in text and "levelmax=12" in text
    assert "hydro=.false." in text
    assert "abort_on_mg_nonconvergence=.true." in text
    assert text.count("traceonly_l12_v2/level_") == 4


def test_reader_runner_is_bounded_and_trace_only() -> None:
    text = (ROOT / "scripts/run_cf4_lg_b3292_s40349_zoom_l12_reader_v1.sbatch").read_text()
    assert "#SBATCH --mem=64G" in text
    assert "parent_promoted=false" in text
    assert "m33_resolved=false" in text
    assert "expected_outputs=0" in text
    assert "extended omega_b header mismatch" in text
    assert "reader gate unexpectedly wrote a full snapshot" in text
