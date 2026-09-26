#!/usr/bin/env python
"""Evaluate a V70 full-band centered ensemble with the frozen evaluator."""
from __future__ import annotations

import hong2021_residual_evaluate as frozen

from hong2021_v70_development_sample import ENSEMBLE_SCHEMA


def main() -> None:
    # Schema registration changes no statistic or threshold in the frozen
    # evaluator; it only declares the already enforced full-band DC convention.
    frozen.CENTERED_SCHEMAS.add(ENSEMBLE_SCHEMA)
    frozen.main()


if __name__ == "__main__":
    main()
