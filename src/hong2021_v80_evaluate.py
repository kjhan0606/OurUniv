#!/usr/bin/env python
"""Evaluate a V80 centered ensemble with the frozen evaluator."""
import hong2021_residual_evaluate as frozen

from hong2021_v80_sample import ENSEMBLE_SCHEMA


def main() -> None:
    frozen.CENTERED_SCHEMAS.add(ENSEMBLE_SCHEMA)
    frozen.main()


if __name__ == "__main__":
    main()
