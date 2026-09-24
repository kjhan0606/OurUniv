import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "cf4_tng_hydro_dark_match",
    Path(__file__).resolve().parents[1] / "scripts/cf4_tng_hydro_dark_match.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mass_bins_and_distinct_match_algorithms():
    mass = np.array([1e8, 1e9, 1e10, 1e11])
    central = np.array([True, False, False, True])
    left = np.array([4, -1, 9, 7])
    right = np.array([4, 6, 10, -1])
    counts = {}
    MODULE.tally(counts, mass, central, left, right)
    assert sum(row["total"] for row in counts.values()) == 4
    assert counts["central:0"]["same_id"] == 1
    assert counts["satellite:1"]["sublink"] == 1
    assert counts["satellite:1"]["lhalotree"] == 0
    assert counts["satellite:2"]["both"] == 1
    assert counts["satellite:2"]["same_id"] == 0
    assert counts["central:3"]["either"] == 1
    assert counts["central:3"]["both"] == 0
