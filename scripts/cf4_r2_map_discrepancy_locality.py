"""Check whether actual catalogue/map discrepancies are local pixel changes.

Neighbour agreement is diagnostic, not a corrected selection model. Source
marks do not determine a spatially normalized effective survey mask.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import healpy as hp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_twompp_disjoint_tracer_pilot_v3 import base

POINTS = Path("/gpfs/kjhan/CF4/z0_density/r2_point_mark_manifest_v1/points.npz")
OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_map_discrepancy_locality_v1")


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    tracer = json.loads((ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json").read_text())
    maps = []
    for key in ("completeness_11_5", "completeness_12_5"):
        binding = tracer["inputs"][key]
        path = Path(binding["path"])
        raw = path.read_bytes()
        if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError(f"source map changed: {key}")
        maps.append(base.load_completeness_map(path, 512))
    with np.load(POINTS, allow_pickle=False) as saved:
        points = {key: saved[key] for key in saved.files}
    mismatch = points["mark_mismatch"]
    ids = np.flatnonzero(mismatch)
    theta = np.deg2rad(90. - points["dec_deg"][ids])
    phi = np.deg2rad(points["ra_deg"][ids])
    pixel = hp.ang2pix(512, theta, phi, nest=False)
    neighbours = hp.get_all_neighbours(512, pixel, nest=False)
    mark = points["catalogue_completeness"][ids]
    own = points["map_completeness"][ids]
    app = points["population"][ids] // 3
    min_neighbor_error = np.full(len(ids), np.inf)
    for a in (0, 1):
        selected = np.flatnonzero(app == a)
        for n in neighbours[:, selected]:
            valid = n >= 0
            error = np.full(len(selected), np.inf)
            error[valid] = np.abs(maps[a][n[valid]] - mark[selected[valid]])
            min_neighbor_error[selected] = np.minimum(min_neighbor_error[selected], error)
    finite = np.isfinite(mark)
    neighbour_agreement = finite & (min_neighbor_error <= .05)
    source_subset = {
        "all": np.ones(len(ids), dtype=bool),
        "train": points["train"][ids],
        "holdout": points["holdout"][ids],
        "calibration": points["calibration"][ids],
        "prior_exception": points["prior_exception"][ids],
        "map_zero": points["map_zero"][ids],
    }
    summary = {name: dict(n=int(mask.sum()), finite_marks=int(np.count_nonzero(mask & finite)),
                          neighbour_agreement=int(np.count_nonzero(mask & neighbour_agreement)))
               for name, mask in source_subset.items()}
    report = dict(classification="R2_MAP_DISCREPANCY_LOCALITY_NOT_SELECTION_CORRECTION",
                  tolerance=0.05, healpix_nside=512, groups=summary,
                  zero_point_recno=points["recno"][ids[points["map_zero"][ids]]].astype(int).tolist(),
                  interpretation="A neighbouring-pixel match is a locality clue only. It does not establish coordinate uncertainty, source-map provenance, a calibrated smoothing width, or a normalized spatial selection function.")
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
