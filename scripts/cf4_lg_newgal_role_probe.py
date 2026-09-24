#!/usr/bin/env python3
"""One bounded role-support readout from the saved z=0 NewGalFinder catalogue."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from cf4_lg_generated_roles import enumerate_role_hypotheses
from cf4_newgalfinder_zoom_validation import read_catalog


CATALOG = Path("/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_z0_v1/job_1109268/"
               "FoF_Data/FoF.00003/GALCATALOG.LIST.00003")
BOX = 384.0
OBSERVER = np.full(3, BOX / 2)
LOCAL_RADIUS = 5.0  # pre-existing P1 Local-Sheet aperture, not target-ranked


def main():
    hosts, children, parent = read_catalog(CATALOG)
    xyz = np.column_stack([hosts[key] for key in ("x", "y", "z")])
    displacement = (xyz - OBSERVER + BOX / 2) % BOX - BOX / 2
    host_ids = np.flatnonzero(np.linalg.norm(displacement, axis=1) <= LOCAL_RADIUS)
    component_ids = np.flatnonzero(np.isin(parent, host_ids))
    n = len(component_ids)
    count = n * (n - 1) ** 2
    report = {
        "status": "GENERATED_STATE_SUPPORT_DIAGNOSTIC_ONLY",
        "catalog": str(CATALOG),
        "catalog_size_bytes": CATALOG.stat().st_size,
        "observer_SG_cMpc_h": OBSERVER.tolist(),
        "fixed_local_radius_cMpc_h": LOCAL_RADIUS,
        "support_rule": "All FoF hosts within the fixed 5 cMpc/h observer aperture and every one of their bound components; no mass, observed-sky or truth-ID ranking.",
        "all_host_count": int(len(hosts)),
        "all_bound_component_count": int(len(children)),
        "local_host_count": int(len(host_ids)),
        "local_bound_component_count": int(n),
        "ordered_role_hypothesis_count_including_unresolved": int(count),
        "local_host_indices": host_ids.tolist(),
        "limits": "This is support cardinality, not a role prior, likelihood, posterior, M33 identification, or an IC update. A shared FoF host does not prove a bound MW-M31 pair.",
    }
    if count <= 100000:
        hypotheses = enumerate_role_hypotheses(children, parent, host_ids)
        assert len(hypotheses) == count
        report["branch_counts"] = dict(Counter(row["M33_branch"] for row in hypotheses))
        report["MW_M31_shared_FoF_hypotheses"] = sum(
            row["MW_M31_share_FoF_host"] for row in hypotheses)
    else:
        report["enumeration"] = "NOT_RUN_UNTRUNCATED_SUPPORT_EXCEEDS_100000"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
