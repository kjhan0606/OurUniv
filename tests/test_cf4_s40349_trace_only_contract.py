from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trace_only_namelist_has_one_initial_dump_and_no_evolution() -> None:
    text = (ROOT / "config/ramses_cf4_lg_b3292_s40349_trace_only_v1.nml").read_text()
    assert "nstepmax=0" in text
    assert "noutput=1" in text
    assert "aout=0.02" in text
    assert "aout=1.0" not in text
    assert "foutput=1000000000" in text
    assert "fbackup=1000000000" in text
    assert "levelmin=9" in text and "levelmax=9" in text
    assert "hydro=.false." in text
    assert "abort_on_mg_nonconvergence=.true." in text


def test_runner_labels_trace_only_status_and_one_measured_output() -> None:
    text = (ROOT / "scripts/run_cf4_lg_b3292_s40349_trace_only_v1.sbatch").read_text()
    assert "purpose=trace-only-zoom-candidate" in text
    assert "parent_promoted=false" in text
    assert "m33_resolved=false" in text
    assert "noutput=1 aout=0.02 measured_GiB_each=18 expected_GiB_total=18" in text
    assert "--selection-status trace-only-zoom-candidate" in text
    assert "[[ \"${#outputs[@]}\" -eq 1 ]]" in text
