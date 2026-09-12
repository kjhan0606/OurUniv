#!/usr/bin/env python
"""Evaluate a V72 centered ensemble with the unchanged frozen evaluator."""
from __future__ import annotations

import hong2021_residual_evaluate as frozen

from hong2021_v72_sqt import ENSEMBLE_SCHEMA


def main() -> None:
    frozen.CENTERED_SCHEMAS.add(ENSEMBLE_SCHEMA)
    frozen.main()


if __name__ == "__main__":
    main()
