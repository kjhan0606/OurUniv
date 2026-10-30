from __future__ import annotations

from hong2021_v17_development_gate import _next_after_failure


def _row(*, tng: bool, simba: bool, swift_peak_only: bool):
    names = (
        "high_k_total_power_within_10_percent",
        "residual_rms_within_10_percent",
        "rank_histogram_tv_at_most_0.05",
        "finite_ensemble_coverage_within_0.03",
        "density_pdf_improves_deterministic",
        "two_point_improves_deterministic_all_scales",
        "all_selected_peak_void_statistics_improve_deterministic",
        "exact_dc_projection",
    )
    checks = {name: True for name in names}
    if swift_peak_only:
        checks["all_selected_peak_void_statistics_improve_deterministic"] = False
    return {
        "domains": {
            "tng": {"field_gate": {"pass": tng}},
            "simba_dev": {"field_gate": {"pass": simba}},
            "swift_dev": {
                "field_gate": {"pass": not swift_peak_only, "checks": checks}
            },
        }
    }


def test_v17_conditional_scale_branch_requires_only_swift_peak_failure() -> None:
    assert _next_after_failure(
        _row(tng=True, simba=True, swift_peak_only=True)
    ) == "stop_unopened_then_predeclare_all_band_target_free_conditional_scale"


def test_v17_every_other_failure_requires_new_audit() -> None:
    assert _next_after_failure(
        _row(tng=False, simba=True, swift_peak_only=True)
    ) == "stop_unopened_and_obtain_new_independent_design_audit"
