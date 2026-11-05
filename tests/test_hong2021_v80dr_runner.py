from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/hong2021_v80dr_metadata_recovery_lageunha.sh"


def test_v80dr_runner_is_recovery_only_and_orders_first_evaluation_after_proof() -> None:
    source = RUNNER.read_text()
    assert "program_sha=b36fd4db6e5a9fbe89fecd36cf17caeb3102fc698cd053c894a7da0a160a1b5a" in source
    assert "program_freeze=38e75b8ba7079aae9e7aabe616f8ca1efc09a325" in source
    recovery = source.index("hong2021_v80dr_metadata_recovery.py")
    evaluation = source.index("hong2021_v80_evaluate.py")
    report = source.index("hong2021_v80dr_engineering_report.py")
    assert recovery < evaluation < report
    assert "hong2021_v80_sample.py" not in source
    assert "hong2021_v79_complete_gate.py" not in source
    assert "hong2021_v80_manifest.py" not in source
    assert "tng100_simba_swift_v80dr_metadata_recovery_ensembles" in source
    assert "failed_terminal_V80DR_no_additional_repair_evaluator_or_report_retry" in source


def test_v80dr_runner_keeps_ramses_cpu_set_disjoint() -> None:
    source = RUNNER.read_text()
    assert "taskset -c 64-95" in source
    assert "low=$((64 + slot * 10))" in source
    assert "high=$((low + 9))" in source
