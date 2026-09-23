from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_twostep_gate_enables_runtime_ceiling_without_outputs() -> None:
    text = (ROOT / "config/ramses_cf4_lg_b3292_s40349_zoom_l19_twostep_v1.nml").read_text()
    assert "nstepmax=2" in text
    assert "noutput=1" in text and "aout=1.1" in text
    assert "foutput=1000000000" in text and "fbackup=1000000000" in text
    assert "levelmin=9" in text and "levelmax=19" in text
    assert "nparttot=190000000" in text
    assert "ngridtot=24000000" in text
    assert "abort_on_mg_nonconvergence=.true." in text


def test_twostep_runner_requires_reader_pass_and_preserves_limits() -> None:
    text = (ROOT / "scripts/run_cf4_lg_b3292_s40349_zoom_l19_twostep_v1.sbatch").read_text()
    assert "TRACE_ONLY_ZOOM_L12_READER_PASS" in text
    assert "parent_promoted=false" in text
    assert "m33_resolved=false" in text
    assert "expected_outputs=0" in text
    assert "two-step gate unexpectedly wrote a full snapshot" in text
