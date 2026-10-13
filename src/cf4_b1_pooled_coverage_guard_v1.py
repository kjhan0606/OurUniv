"""Machine-reproducible, development-only pooled coverage guard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


TARGETS = {"coverage68": 0.6826894921370859, "coverage95": 0.9544997361036416}
TOLERANCES = {"coverage68": 0.05, "coverage95": 0.02}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260905
ARMS = ("A", "B", "C", "D")


def _bootstrap_upper(values: np.ndarray, seed: int = BOOTSTRAP_SEED) -> float:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [values[rng.integers(0, len(values), size=len(values))].mean()
         for _ in range(BOOTSTRAP_REPLICATES)],
        dtype=np.float64,
    )
    return float(np.percentile(draws, 97.5))


def evaluate(input_path: Path) -> dict[str, object]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload["members"]
    if len(rows) != 64:
        raise ValueError("guard requires exactly 64 development members")
    implementation = Path(__file__).resolve()
    dependency = payload.get("dependency_artifact")
    if not isinstance(dependency, dict) or not {"path", "bytes", "sha256"} <= dependency.keys():
        raise ValueError("guard input is missing the repaired base dependency artifact")
    result: dict[str, object] = {
        "schema": "ouruniv-cf4-b1-pooled-coverage-guard-result-v1",
        "status": "COMPLETE_DEVELOPMENT_ONLY_NO_SCIENCE_CLAIM",
        "implementation_artifact": {
            "path": str(implementation),
            "bytes": implementation.stat().st_size,
            "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        },
        "dependency_artifact": dependency,
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "bootstrap": {
            "unit": "independent truth seed",
            "replicates": BOOTSTRAP_REPLICATES,
            "rng": f"PCG64({BOOTSTRAP_SEED})",
            "percentile": 97.5,
        },
        "guard_definition": {
            "statistic": "per-seed mean of eight mode coverage indicators",
            "tolerances": TOLERANCES,
            "pass_rule": "one-sided bootstrap upper percentile <= target + tolerance",
            "strict_promotion_gate_changed": False,
        },
        "levels": {},
        "by_arm": {},
        "scientific_boundary": {
            "validation_opened": False,
            "observational_z0_posterior": "NOT_CREATED",
            "frontier_promotion": "BLOCKED",
            "B2_IC_FORWARD": "NOT_STARTED",
        },
    }
    for key, target in TARGETS.items():
        indicators = np.asarray([row[f"{key}_by_mode"] for row in rows], dtype=bool)
        per_seed = indicators.mean(axis=1)
        upper = _bootstrap_upper(per_seed)
        level = {
            "target": target,
            "tolerance": TOLERANCES[key],
            "upper_limit": target + TOLERANCES[key],
            "member_count": len(per_seed),
            "mode_count": indicators.shape[1],
            "pooled_count": int(indicators.sum()),
            "pooled_total": int(indicators.size),
            "mean": float(per_seed.mean()),
            "bootstrap_upper_97_5": upper,
            "pass": bool(upper <= target + TOLERANCES[key]),
        }
        result["levels"][key] = level
        for arm in ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            arm_indicators = np.asarray([row[f"{key}_by_mode"] for row in arm_rows], dtype=bool)
            arm_seed_means = arm_indicators.mean(axis=1)
            arm_result = result["by_arm"].setdefault(arm, {})
            arm_upper = _bootstrap_upper(arm_seed_means)
            arm_result[key] = {
                "member_count": len(arm_seed_means),
                "mean": float(arm_seed_means.mean()),
                "bootstrap_upper_97_5": arm_upper,
                "upper_limit": target + TOLERANCES[key],
                "pass": bool(arm_upper <= target + TOLERANCES[key]),
            }
    result["primary_reference_arm"] = "B"
    result["primary_reference_pass"] = bool(
        result["by_arm"]["B"]["coverage68"]["pass"]
        and result["by_arm"]["B"]["coverage95"]["pass"]
    )
    result["overall_pass"] = bool(
        result["levels"]["coverage68"]["pass"]
        and result["levels"]["coverage95"]["pass"]
        and result["primary_reference_pass"]
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("config/cf4_b1_mode_coverage_diagnosis_result_v3.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = evaluate(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall_pass": result["overall_pass"], "levels": result["levels"], "primary_reference_pass": result["primary_reference_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
