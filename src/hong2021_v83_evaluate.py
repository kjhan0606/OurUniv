#!/usr/bin/env python
"""Evaluate a centered V83 ensemble with the frozen residual evaluator."""
import hong2021_residual_evaluate as frozen

from hong2021_v83_sample import SCHEMA


def main() -> None:
    frozen.CENTERED_SCHEMAS.add(SCHEMA)
    frozen.main()


if __name__ == "__main__":
    main()
