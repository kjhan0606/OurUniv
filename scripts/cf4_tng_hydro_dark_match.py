"""Descriptive z=0 TNG hydro-to-Dark Subfind matching; no LG likelihood."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


METHODS = ("SubhaloIndexDark_LHaloTree", "SubhaloIndexDark_SubLink")
STAR_MASS_EDGES_MSUN = (1e8, 1e9, 1e10, 1e11, float("inf"))
MIN_STAR_PARTICLES = 100


def bin_index(mass):
    return np.searchsorted(STAR_MASS_EDGES_MSUN, mass, side="right") - 1


def tally(summary, mass, central, match_a, match_b):
    """Count eligible objects, not independent draws or detection probabilities."""
    a, b = match_a >= 0, match_b >= 0
    for role, role_mask in (("central", central), ("satellite", ~central)):
        for index in range(len(STAR_MASS_EDGES_MSUN) - 1):
            mask = role_mask & (bin_index(mass) == index)
            key = f"{role}:{index}"
            row = summary.setdefault(key, dict(total=0, lhalotree=0, sublink=0,
                                               both=0, either=0, same_id=0))
            row["total"] += int(mask.sum())
            row["lhalotree"] += int((mask & a).sum())
            row["sublink"] += int((mask & b).sum())
            row["both"] += int((mask & a & b).sum())
            row["either"] += int((mask & (a | b)).sum())
            row["same_id"] += int((mask & a & b & (match_a == match_b)).sum())


def analyze(match_path, hydro_dir):
    files = sorted(hydro_dir.glob("fof_subhalo_tab_099.*.hdf5"),
                   key=lambda p: int(p.name.split(".")[-2]))
    if len(files) != 448 or [int(p.name.split(".")[-2]) for p in files] != list(range(448)):
        raise ValueError("expected all 448 hydro group-catalogue chunks")
    with h5py.File(match_path) as match:
        group = match["Snapshot_99"]
        left, right = (group[name][:] for name in METHODS)
    with h5py.File(files[0]) as source:
        head = source["Header"].attrs
        nsub = int(head["Nsubgroups_Total"])
        ngroup = int(head["Ngroups_Total"])
        h = float(head["HubbleParam"])
    if left.shape != (nsub,) or right.shape != (nsub,):
        raise ValueError("matching table must align with hydro global Subfind IDs")
    if not (0 < h < 1):
        raise ValueError("invalid HubbleParam")

    # Global GroupFirstSub is needed to distinguish central and satellite.
    first = np.empty(ngroup, dtype=np.int64)
    group_offset = 0
    for path in files:
        with h5py.File(path) as source:
            count = int(source["Header"].attrs["Ngroups_ThisFile"])
            if count:
                first[group_offset:group_offset + count] = source["Group/GroupFirstSub"][:]
            group_offset += count
    if group_offset != ngroup:
        raise ValueError("incomplete hydro group count")

    summary = {}
    source_count = eligible_count = 0
    for path in files:
        with h5py.File(path) as source:
            count = int(source["Header"].attrs["Nsubgroups_ThisFile"])
            if not count:
                continue
            sub = source["Subhalo"]
            mass = sub["SubhaloMassType"][:, 4].astype(np.float64) * 1e10 / h
            stars = sub["SubhaloLenType"][:, 4]
            host = sub["SubhaloGrNr"][:]
            flag = sub["SubhaloFlag"][:]
            if np.any((host < 0) | (host >= ngroup)):
                raise ValueError("invalid host group ID")
            ids = np.arange(source_count, source_count + count, dtype=np.int64)
            eligible = flag & (stars >= MIN_STAR_PARTICLES) & (mass >= STAR_MASS_EDGES_MSUN[0])
            eligible_count += int(eligible.sum())
            central = ids == first[host]
            tally(summary, mass[eligible], central[eligible],
                  left[ids][eligible], right[ids][eligible])
            source_count += count
    if source_count != nsub:
        raise ValueError("incomplete hydro subhalo count")
    if sum(row["total"] for row in summary.values()) != eligible_count:
        raise ValueError("summary count mismatch")
    return {
        "status": "TNG_HYDRO_DARK_SUBFIND_MATCH_DESCRIPTIVE_ONLY",
        "source_hydro": str(hydro_dir), "source_match": str(match_path),
        "snapshot": 99, "hydro_subhalos": nsub, "eligible_galaxies": eligible_count,
        "eligibility": {"SubhaloFlag": True, "min_star_particles": MIN_STAR_PARTICLES,
                        "min_stellar_mass_Msun": STAR_MASS_EDGES_MSUN[0]},
        "stellar_mass_bins_Msun": [[STAR_MASS_EDGES_MSUN[i],
                                    None if np.isinf(STAR_MASS_EDGES_MSUN[i + 1])
                                    else STAR_MASS_EDGES_MSUN[i + 1]]
                                   for i in range(len(STAR_MASS_EDGES_MSUN) - 1)],
        "counts": summary,
        "limits": "One correlated TNG volume; hydro-to-Dark Subfind matching only. "
                  "No NewGalFinder detection calibration, observer selection, "
                  "MW/M31/M33 role prior, or CF4/LG likelihood."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--hydro-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.match, args.hydro_dir)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "eligible_galaxies": result["eligible_galaxies"]}))


if __name__ == "__main__":
    main()
