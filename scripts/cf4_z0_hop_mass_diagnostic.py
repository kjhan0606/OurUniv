#!/usr/bin/env python3
"""Read-only z=0 HOP mass-ranked diagnostic; no target promotion."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cf4_zoom_z0_gate import catalog_from_hop_tags, particle_files, read_info, scan_mass_species


MSUN_G = 1.98847e33


def main() -> None:
    output = Path("/gpfs/kjhan/CF4/ramses/cf4_production_ic_z0_dmo_v1/job_388424/output_00002")
    hop = Path("/gpfs/kjhan/CF4/hop/cf4_z0_v1/job_388595")
    out = Path("/gpfs/kjhan/CF4/diagnostics/cf4_z0_hop_mass_v1.json")
    info = read_info(output)
    files = particle_files(output)
    mass_unit = info["unit_d"] * info["unit_l"] ** 3 / MSUN_G * (info["H0"] / 100.0)
    ntotal, species = scan_mass_species(files, mass_unit)
    fine_mass_code = species[0]["mass_code"]
    cat = catalog_from_hop_tags(output, hop / "grp.tag", 384.0, mass_unit,
                                info["unit_l"] / info["unit_t"] / 1e5,
                                fine_mass_code)
    mass = np.asarray(cat["mass"], float)
    count = np.asarray(cat["n"], int)
    order = np.argsort(-mass)
    rows = []
    for i in order[:200]:
        rows.append({"group_id": int(cat["group_id"][i]), "count": int(count[i]),
                     "mass_msun_h": float(mass[i]),
                     "position_cMpc_h": np.asarray(cat["pos"][i]).tolist(),
                     "velocity_km_s": np.asarray(cat["vel"][i]).tolist()})
    result = {
        "schema": "ouruniv-cf4-z0-hop-mass-diagnostic-v1",
        "stage": "7/8",
        "output": str(output), "hop_work": str(hop), "aexp": float(info["aexp"]),
        "particle_count": int(ntotal), "group_count": int(len(mass)),
        "groups_count_ge_100": int(np.count_nonzero(count >= 100)),
        "mass_species": species, "top_groups": rows,
        "interpretation": "Mass-ranked HOP groups only; no MW/M31/M33 or environment promotion."
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "particle_count": int(ntotal),
                      "group_count": int(len(mass)),
                      "groups_count_ge_100": int(np.count_nonzero(count >= 100))},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
