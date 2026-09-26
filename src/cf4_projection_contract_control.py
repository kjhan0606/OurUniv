#!/usr/bin/env python3
"""Full-size consumed-only control for the future Fourier projection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cf4_projection_contract import (
    contract_metadata,
    normalized_errors,
    prolong_white_field,
    restrict_white_field,
    white_moments,
)
from cf4_projection_nyquist_control import (
    spatial_from_output_rfft,
    variance_preserving_projection_rfft,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_nonfinite(value: Any) -> Any:
    if isinstance(value, list):
        return [_replace_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _replace_nonfinite(value.tolist())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_reference_field(case: dict[str, Any]) -> np.ndarray:
    path = Path(case["path"])
    if sha256_file(path) != case["sha256"]:
        raise RuntimeError(f"reference field hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as item:
        seed = int(item["sample_seed"])
        field = item[case["array"]].astype(np.float32)
    if seed != int(case["parent_seed"]):
        raise RuntimeError(f"reference seed mismatch: {seed} != {case['parent_seed']}")
    return field


def roundtrip_case(
    coarse: np.ndarray,
    fine_n: int,
    fine_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fine = prolong_white_field(coarse, fine_n, fine_seed)
    restricted = restrict_white_field(fine, coarse.shape[0])
    return fine, {
        "fine_seed": fine_seed,
        "roundtrip": normalized_errors(restricted, coarse),
        "fine_moments": white_moments(fine),
    }


def run(program: dict[str, Any]) -> dict[str, Any]:
    coarse_n = int(program["mesh"]["coarse_N"])
    fine_n = int(program["mesh"]["fine_N"])
    iid_rows = []
    reference_rows = []
    rfft_equivalence = None

    for number, case in enumerate(program["IID_cases"], start=1):
        rng = np.random.Generator(np.random.PCG64DXSM(int(case["coarse_seed"])))
        coarse = rng.standard_normal((coarse_n, coarse_n, coarse_n)).astype(np.float32)
        fine, row = roundtrip_case(coarse, fine_n, int(case["fine_seed"]))
        row["coarse_seed"] = int(case["coarse_seed"])
        row["coarse_moments"] = white_moments(coarse)
        if rfft_equivalence is None:
            full_restriction = restrict_white_field(fine, coarse_n)
            rfft_restriction = spatial_from_output_rfft(
                variance_preserving_projection_rfft(
                    np.fft.rfftn(fine.astype(np.float64)), fine_n, coarse_n
                ),
                coarse_n,
            )
            rfft_equivalence = normalized_errors(
                full_restriction, rfft_restriction
            )
        iid_rows.append(row)
        print(
            f"[contract] IID {number}/{len(program['IID_cases'])} "
            f"coarse={case['coarse_seed']} fine={case['fine_seed']}",
            flush=True,
        )
        del coarse, fine

    retained_by_parent: dict[int, np.ndarray] = {}
    diversity_rows = []
    for number, case in enumerate(program["reference_cases"], start=1):
        coarse = load_reference_field(case)
        parent_seed = int(case["parent_seed"])
        fine_seed = int(case["fine_seed"])
        fine, row = roundtrip_case(coarse, fine_n, fine_seed)
        row.update({
            "parent_seed": parent_seed,
            "parent_field_sha256": case["sha256"],
        })
        if parent_seed in retained_by_parent:
            previous = retained_by_parent.pop(parent_seed)
            diversity_rows.append({
                "parent_seed": parent_seed,
                "fine_seed_pair": [int(case["paired_with_fine_seed"]), fine_seed],
                "fine_field_difference_RMS": float(np.sqrt(np.mean(
                    (previous.astype(np.float64) - fine.astype(np.float64)) ** 2
                ))),
            })
        elif case.get("retain_for_pair", False):
            retained_by_parent[parent_seed] = fine.copy()
        reference_rows.append(row)
        print(
            f"[contract] reference {number}/{len(program['reference_cases'])} "
            f"parent={parent_seed} fine={fine_seed}",
            flush=True,
        )
        del coarse, fine
    if retained_by_parent:
        raise RuntimeError("unpaired retained reference prolongation")

    all_rows = iid_rows + reference_rows
    gate = program["gates"]
    roundtrip_pass = all(
        row["roundtrip"]["relative_RMS"] <= gate["roundtrip_relative_RMS_max"]
        and row["roundtrip"]["maximum_normalized_error"]
        <= gate["roundtrip_maximum_normalized_error_max"]
        for row in all_rows
    )
    rfft_pass = bool(
        rfft_equivalence["relative_RMS"]
        <= gate["full_vs_rfft_relative_RMS_max"]
        and rfft_equivalence["maximum_normalized_error"]
        <= gate["full_vs_rfft_maximum_normalized_error_max"]
    )
    iid_moment_pass = all(
        abs(row["fine_moments"]["mean"]) <= gate["IID_fine_abs_mean_max"]
        and abs(row["fine_moments"]["std"] - 1.0)
        <= gate["IID_fine_abs_std_minus_one_max"]
        and abs(row["fine_moments"]["skew"])
        <= gate["IID_fine_abs_skew_max"]
        and abs(row["fine_moments"]["excess_kurtosis"])
        <= gate["IID_fine_abs_excess_kurtosis_max"]
        for row in iid_rows
    )
    diversity_pass = bool(
        diversity_rows
        and all(
            row["fine_field_difference_RMS"]
            >= gate["same_parent_fine_difference_RMS_min"]
            for row in diversity_rows
        )
    )
    contract_pass = bool(
        roundtrip_pass and rfft_pass and iid_moment_pass and diversity_pass
    )
    return {
        "schema": "ouruniv-cf4-projection-contract-control-result-v1",
        "status": (
            "complete_pass_future_projection_contract"
            if contract_pass else "complete_fail_future_projection_contract"
        ),
        "information_firewall": program["information_firewall"],
        "contract": contract_metadata(),
        "mesh": program["mesh"],
        "IID_rows": iid_rows,
        "reference_rows": reference_rows,
        "same_parent_diversity": diversity_rows,
        "full_spectrum_vs_rfft_restriction": rfft_equivalence,
        "gates": {
            "all_roundtrips": roundtrip_pass,
            "full_spectrum_matches_rfft_control": rfft_pass,
            "IID_fine_moments": iid_moment_pass,
            "same_parent_null_space_diversity": diversity_pass,
            "contract_pass": contract_pass,
        },
        "decision": {
            "paired_projection_contract_authorized": contract_pass,
            "independent_parent_architecture_design_authorized": contract_pass,
            "candidate_generation_authorized": False,
            "seed_selection_authorized": False,
            "PM_or_RAMSES_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    program_path = Path(args.program).resolve()
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with program_path.open() as stream:
        program = json.load(stream)

    for key in ("implementation", "contract_implementation", "rfft_control"):
        item = program[key]
        path = (ROOT / item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"{key} hash mismatch")

    result = run(program)
    result["lineage"] = {
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "implementation_sha256": program["implementation"]["sha256"],
        "contract_implementation_sha256": program[
            "contract_implementation"
        ]["sha256"],
        "rfft_control_sha256": program["rfft_control"]["sha256"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(
            result, stream, indent=2, sort_keys=True,
            default=json_default, allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(output_path)
    print(f"[contract] status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
