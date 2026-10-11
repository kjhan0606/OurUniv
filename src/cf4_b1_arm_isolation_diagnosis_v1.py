"""Fresh-seed common-random-number arm-isolation diagnosis for B1.

This is a development-only experiment.  It never touches the sealed 64-member
calibration or validation range.  All native A--D arms share the same truth,
count-split, mark, and posterior seeds within each replicate.  D-only controls
then remove (one at a time) the D count overdispersion and the D model-
discrepancy intensity terms.  The purpose is causal attribution, not width
fitting or gate promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

import cf4_b1_development_mock_calibration as base
import cf4_b1_integrated_joint_calibration_v3 as integrated


REPLICATES = 16
D_CONTROLS = ("native", "no_phi", "no_discrepancy", "no_phi_no_discrepancy")
FRESH_TRUTH_START = 2026084000
FRESH_COUNTS_START = 2026404000
FRESH_SPLIT_START = 2026405000
FRESH_MARKS_START = 2026504000
FRESH_POSTERIOR_START = 2026604000
SEALED_DEVELOPMENT = range(2026083000, 2026083064)
SEALED_QUARANTINE = range(2026083064, 2026083128)
SEALED_VALIDATION = range(2026083320, 2026083576)


def common_seed_schedule(index: int) -> dict[str, int]:
    if not isinstance(index, int) or not 0 <= index < REPLICATES:
        raise ValueError("index outside fresh arm-isolation block")
    return {
        "truth": FRESH_TRUTH_START + index,
        "counts": FRESH_COUNTS_START + index,
        "count_split": FRESH_SPLIT_START + index,
        "marks": FRESH_MARKS_START + index,
        "posterior": FRESH_POSTERIOR_START + index,
    }


def _assert_disjoint_seed_firewall() -> None:
    fresh = set(common_seed_schedule(i)["truth"] for i in range(REPLICATES))
    if fresh & (set(SEALED_DEVELOPMENT) | set(SEALED_QUARANTINE) | set(SEALED_VALIDATION)):
        raise RuntimeError("fresh arm-isolation truth seeds overlap sealed ranges")


@contextmanager
def _patched_seed_schedule() -> Iterator[None]:
    original = integrated.seed_schedule
    integrated.seed_schedule = common_seed_schedule
    try:
        yield
    finally:
        integrated.seed_schedule = original


@contextmanager
def _d_control(control: str) -> Iterator[None]:
    if control not in set(D_CONTROLS):
        raise ValueError(f"unknown D control {control}")
    original_draw = base._draw_counts
    original_positive = base._positive_intensity
    original_phi = integrated.D_OVERDISPERSION_PHI

    if "no_phi" in control:
        def poisson_draw(intensity: np.ndarray, arm: str, seed: int) -> np.ndarray:
            return np.random.default_rng(seed).poisson(intensity).astype(np.int64)

        base._draw_counts = poisson_draw
        integrated.D_OVERDISPERSION_PHI = 0.0

    if "no_discrepancy" in control:
        def no_discrepancy(eta: np.ndarray, arm: str, velocity: np.ndarray) -> np.ndarray:
            return original_positive(eta, "C" if arm == "D" else arm, velocity)

        base._positive_intensity = no_discrepancy
    try:
        yield
    finally:
        base._draw_counts = original_draw
        base._positive_intensity = original_positive
        integrated.D_OVERDISPERSION_PHI = original_phi


def _run_control(control: str) -> list[dict[str, object]]:
    with _patched_seed_schedule(), _d_control(control):
        return [integrated.run_mock(index, "D") for index in range(REPLICATES)]


def _run_native_arms() -> list[dict[str, object]]:
    with _patched_seed_schedule():
        return [integrated.run_mock(index, arm)
                for index in range(REPLICATES) for arm in ("A", "B", "C", "D")]


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = [row["metrics"] for row in rows]
    names = ("response", "correlation_r", "residual_power_ratio", "coverage68",
             "coverage95", "heldout_log_score_improvement", "joint_log_likelihood_abs_error")
    out: dict[str, object] = {
        "member_count": len(rows),
        "strict_low_k_gate_pass_count": int(sum(bool(m["strict_low_k_gate"]) for m in metrics)),
        "metric_mean": {name: float(np.mean([m[name] for m in metrics])) for name in names},
        "metric_median": {name: float(np.median([m[name] for m in metrics])) for name in names},
        "seed_range": {
            key: [min(int(row["seed"][key]) for row in rows), max(int(row["seed"][key]) for row in rows)]
            for key in ("truth", "counts", "count_split", "marks", "posterior")
        },
        "members": [
            {
                "index": int(row["index"]),
                "strict_low_k_gate": bool(row["metrics"]["strict_low_k_gate"]),
                "coverage68": float(row["metrics"]["coverage68"]),
                "coverage95": float(row["metrics"]["coverage95"]),
                "response": float(row["metrics"]["response"]),
                "residual_power_ratio": float(row["metrics"]["residual_power_ratio"]),
                "heldout_log_score_improvement": float(row["metrics"]["heldout_log_score_improvement"]),
            }
            for row in rows
        ],
    }
    return out


def run_diagnosis() -> dict[str, object]:
    _assert_disjoint_seed_firewall()
    native = _run_native_arms()
    controls = {control: _run_control(control) for control in D_CONTROLS}
    by_arm = {arm: _summarize([row for row in native if row["arm"] == arm]) for arm in "ABCD"}
    by_control = {control: _summarize(rows) for control, rows in controls.items()}
    native_d = {int(row["index"]): row["metrics"] for row in controls["native"]}
    paired = {}
    for control, rows in controls.items():
        current = {int(row["index"]): row["metrics"] for row in rows}
        paired[control] = {
            "strict_flip_count_vs_native": int(sum(bool(native_d[i]["strict_low_k_gate"]) != bool(current[i]["strict_low_k_gate"]) for i in native_d)),
            "strict_gain_count_vs_native": int(sum((not bool(native_d[i]["strict_low_k_gate"])) and bool(current[i]["strict_low_k_gate"]) for i in native_d)),
            "strict_loss_count_vs_native": int(sum(bool(native_d[i]["strict_low_k_gate"]) and (not bool(current[i]["strict_low_k_gate"])) for i in native_d)),
            "paired_mean_delta": {
                name: float(np.mean([float(current[i][name]) - float(native_d[i][name]) for i in native_d]))
                for name in ("coverage68", "coverage95", "response", "residual_power_ratio", "heldout_log_score_improvement")
            },
        }
    return {
        "schema": "ouruniv-cf4-b1-arm-isolation-diagnosis-result-v1",
        "status": "COMPLETE_DEVELOPMENT_ONLY_NO_SCIENCE_CLAIM",
        "purpose": "common-random-number causal attribution of uniform native strict-pass counts; no width refit",
        "source_artifacts": [
            {"path": str(Path(integrated.__file__).resolve()), "bytes": Path(integrated.__file__).stat().st_size,
             "sha256": hashlib.sha256(Path(integrated.__file__).read_bytes()).hexdigest()},
            {"path": str(Path(base.__file__).resolve()), "bytes": Path(base.__file__).stat().st_size,
             "sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()},
        ],
        "design": {
            "replicates": REPLICATES,
            "native_arms": "A-D, all four arms share each replicate's truth/count-split/mark/posterior seeds",
            "D_controls": {
                "native": "D intensity discrepancy and gamma-Poisson overdispersion retained",
                "no_phi": "D intensity discrepancy retained; D count generation and Fisher variance made Poisson",
                "no_discrepancy": "D gamma-Poisson overdispersion retained; D intensity replaced by C intensity (RSD+FoG, no D discrepancy)",
                "no_phi_no_discrepancy": "both D-specific overdispersion and discrepancy removed",
            },
            "fresh_seed_ranges": {
                "truth": [FRESH_TRUTH_START, FRESH_TRUTH_START + REPLICATES],
                "counts": [FRESH_COUNTS_START, FRESH_COUNTS_START + REPLICATES],
                "count_split": [FRESH_SPLIT_START, FRESH_SPLIT_START + REPLICATES],
                "marks": [FRESH_MARKS_START, FRESH_MARKS_START + REPLICATES],
                "posterior": [FRESH_POSTERIOR_START, FRESH_POSTERIOR_START + REPLICATES],
            },
            "sealed_ranges_untouched": {
                "development_truth": [2026083000, 2026083064],
                "quarantine_truth": [2026083064, 2026083128],
                "validation_truth": [2026083320, 2026083576],
            },
        },
        "native_by_arm": by_arm,
        "D_by_control": by_control,
        "D_paired_effects_vs_native": paired,
        "interpretation": {
            "native_uniform_pass_count": {arm: by_arm[arm]["strict_low_k_gate_pass_count"] for arm in "ABCD"},
            "causal_rule": "A D-specific mechanism is supported only if the paired D control changes coverage/pass behavior while common seeds and other interfaces remain fixed.",
            "width_policy": "diagnostic-only; no posterior width or mean-model refit",
            "validation_opened": False,
            "B2_IC_FORWARD": "NOT_STARTED",
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_diagnosis()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"native_pass": result["interpretation"]["native_uniform_pass_count"],
                      "D_controls": {k: v["strict_low_k_gate_pass_count"] for k, v in result["D_by_control"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
