"""Source-bound coordinate bridge: published 2M++ CSV versus ARES example.

The ARES example's 2MPP.txt is the input paired with the released HEALPix
masks. This audit does not assert that its full selection/noise model is
calibrated for CF4, nor replace the frozen scientific catalogue.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_twompp_disjoint_tracer_pilot_v3 import load_catalog

ARES = Path("/gpfs/kjhan/CF4/software/ares-6cf608ed/examples/2MPP.txt")
ARES_SHA256 = "06bf20e023592bff06530bb91694185dc833da02d219f6d4ce9c6592b7196357"
POINTS = Path("/gpfs/kjhan/CF4/z0_density/r2_point_mark_manifest_v1/points.npz")
OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_ares_catalogue_bridge_v1")
ARCSEC = np.pi / (180.0 * 3600.0)
LIGHT_SPEED = 299792.458


def unit(ra, dec):
    return np.stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                     np.sin(dec)), axis=-1)


def quantiles(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), maximum=float(np.max(x))) if x.size else None


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    program = json.loads((ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json").read_text())
    binding = program["inputs"]["twompp_catalog"]
    source_path = ROOT / binding["path"]
    raw = source_path.read_bytes()
    if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("2M++ source catalogue binding changed")
    if hashlib.sha256(ARES.read_bytes()).hexdigest() != ARES_SHA256:
        raise ValueError("ARES example catalogue binding changed")
    source = load_catalog(source_path)
    example = np.loadtxt(ARES, dtype=np.float64)
    if example.shape != (67224, 7) or not np.array_equal(example[:, 0], np.arange(67224)):
        raise ValueError("ARES text schema or sequential index changed")
    source_unit = unit(np.deg2rad(source["RA"]), np.deg2rad(source["DEC"]))
    example_unit = unit(example[:, 1], example[:, 2])
    chord, index = cKDTree(source_unit).query(example_unit, k=1, workers=1)
    angle_arcsec = np.rad2deg(2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0))) * 3600.0
    redshift_delta = np.abs(example[:, 3] * LIGHT_SPEED - source["Vcmb"][index])
    magnitude_delta = np.abs(example[:, 4] - source["Ksmag"][index])
    # Radian/velocity/magnitude columns are serialized with finite precision;
    # a match must pass all three, and cannot claim two ARES rows for one recno.
    close = (angle_arcsec <= 2.0) & (redshift_delta <= 2.0) & (magnitude_delta <= 0.1)
    matched_index = index[close]
    if len(np.unique(matched_index)) != len(matched_index):
        raise ValueError("coordinate bridge is not one-to-one")
    accepted = np.zeros(len(source["recno"]), dtype=bool)
    accepted[matched_index] = True
    with np.load(POINTS, allow_pickle=False) as saved:
        points = {key: saved[key] for key in saved.files}
    sorted_source = np.argsort(source["recno"])
    lookup = sorted_source[np.searchsorted(source["recno"][sorted_source], points["recno"])]
    np.testing.assert_array_equal(source["recno"][lookup], points["recno"])
    eligible_in_example = accepted[lookup]
    target = points["recno"] == 67100
    if target.sum() != 1:
        raise ValueError("known map-zero point not unique in eligible list")
    n_source = len(source["recno"])
    report = dict(classification="R2_ARES_EXAMPLE_SOURCE_BRIDGE_NO_LIKELIHOOD",
                  source_sha256=binding["sha256"], ares_sha256=ARES_SHA256,
                  source_rows=n_source, example_rows=int(len(example)),
                  matched_example_rows=int(close.sum()), unmatched_example_rows=int((~close).sum()),
                  match_tolerances=dict(angle_arcsec=2.0, vcmb_km_s=2.0, ksmag=0.1),
                  matched_angle_arcsec=quantiles(angle_arcsec[close]),
                  matched_redshift_delta_km_s=quantiles(redshift_delta[close]),
                  matched_magnitude_delta=quantiles(magnitude_delta[close]),
                  unmatched_example_nearest_angle_arcsec=quantiles(angle_arcsec[~close]),
                  source_rows_not_in_example=int((~accepted).sum()),
                  source_not_in_example_cloned=int(np.count_nonzero(~accepted & (source["Cln"] == 1))),
                  source_not_in_example_zoa=int(np.count_nonzero(~accepted &
                      (np.char.lower(np.char.strip(source["Ref"].astype(str))) == "zoa"))),
                  eligible_rows_in_example=int(eligible_in_example.sum()),
                  eligible_rows_not_in_example=int((~eligible_in_example).sum()),
                  eligible_map_zero_in_example=int(np.count_nonzero(eligible_in_example & points["map_zero"])),
                  eligible_map_zero_outside_example=int(np.count_nonzero(~eligible_in_example & points["map_zero"])),
                  eligible_mismatch_in_example=int(np.count_nonzero(eligible_in_example & points["mark_mismatch"])),
                  eligible_mismatch_outside_example=int(np.count_nonzero(~eligible_in_example & points["mark_mismatch"])),
                  known_zero_point_in_example=bool(eligible_in_example[target][0]),
                  limitation="ARES example membership is not by itself a calibrated joint CF4/2M++ selection or CF4 conditional distance likelihood. If matching is incomplete, inspect source precision before scientific use.")
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "bridge.npz", example_index=example[close, 0].astype(np.int32),
                        source_recno=source["recno"][matched_index],
                        eligible_recno=points["recno"], eligible_in_example=eligible_in_example)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
