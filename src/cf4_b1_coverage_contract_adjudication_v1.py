"""Make the B1 legacy coverage denominator and contingency contract explicit.

The historical diagnosis stores integer member-failure counts, not ratios.  This
development-only checker derives both counts and fractions from serialized
member gate failures so an audit packet cannot accidentally render ``47`` as
``47/47``.  It does not alter the strict gate, open validation, or fit any seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = Path("config/cf4_b1_calibration_diagnosis_result_v3.json")


def summarize(input_path: Path = DEFAULT_INPUT) -> dict[str, object]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = data["members"]
    n = len(rows)
    if n <= 0:
        raise ValueError("diagnosis contains no serialized members")
    has68 = ["coverage68" in row["gate_failures"] for row in rows]
    has95 = ["coverage95" in row["gate_failures"] for row in rows]
    contingency = Counter(zip(has68, has95))
    values = {
        "coverage68_only": int(contingency[(True, False)]),
        "coverage95_only": int(contingency[(False, True)]),
        "both": int(contingency[(True, True)]),
        "neither": int(contingency[(False, False)]),
    }
    fail68 = values["coverage68_only"] + values["both"]
    fail95 = values["coverage95_only"] + values["both"]
    union = values["coverage68_only"] + values["coverage95_only"] + values["both"]
    if fail68 != int(data["failure_histogram"]["coverage68"]):
        raise ValueError("coverage68 serialized count disagrees with diagnosis histogram")
    if fail95 != int(data["failure_histogram"]["coverage95"]):
        raise ValueError("coverage95 serialized count disagrees with diagnosis histogram")
    if sum(values.values()) != n or union + values["neither"] != n:
        raise ValueError("coverage contingency does not partition the member population")
    return {
        "schema": "ouruniv-cf4-b1-coverage-contract-adjudication-result-v1",
        "status": "COMPLETE_DEVELOPMENT_ONLY_NO_SCIENCE_CLAIM",
        "source": str(input_path),
        "source_artifact": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "metric_scope": "legacy single-member per-voxel diagnostic; member denominator is serialized development population",
        "member_count": n,
        "failure_counts": {"coverage68": fail68, "coverage95": fail95, "union": union},
        "failure_fractions": {"coverage68": fail68 / n, "coverage95": fail95 / n, "union": union / n},
        "contingency": values,
        "arithmetic": {
            "component68": f"{values['coverage68_only']}+{values['both']}={fail68}",
            "component95": f"{values['coverage95_only']}+{values['both']}={fail95}",
            "partition": f"{values['coverage68_only']}+{values['coverage95_only']}+{values['both']}+{values['neither']}={n}",
        },
        "interpretation": "The stored 47 values are counts (47/64 = 0.734375), not 47/47 ratios. The contingency is internally consistent; the strict gate remains unchanged and fails 56/64.",
        "validation_opened": False,
        "B2_IC_FORWARD": "NOT_STARTED",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = summarize(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"member_count": result["member_count"], "failure_counts": result["failure_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
