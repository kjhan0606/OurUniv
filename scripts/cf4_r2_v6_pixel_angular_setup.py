"""Create the small score-blind HEALPix control input for the GPU diagnostic."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cf4_r2_selection_ray_integral import load_physics_inputs  # noqa: E402
from cf4_r2_v6_pixel_angular_geometry import (  # noqa: E402
    choose_controls,
    flatten_cell_indices,
)

BASE = Path("/gpfs/kjhan/CF4/z0_density")
SOURCE_PATH = BASE / "r2_marked_source_geometry_v1/geometry.npz"


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("control preparation must run inside the Slurm job")
    out = Path(os.environ["CF4_R2_OUT_DIR"])
    out.mkdir(parents=True, exist_ok=False)
    physics = load_physics_inputs()
    with np.load(SOURCE_PATH, allow_pickle=False) as source:
        controls = choose_controls(physics["maps"], physics["rotation"],
                                   source["angular"])
    if not np.array_equal(controls["flat_ids"],
                          flatten_cell_indices(controls["ijk"], n=256)):
        raise ValueError("control flattened IDs do not match their N256 cells")
    np.savez_compressed(
        out / "controls.npz",
        candidate_indices=controls["candidate_indices"],
        flat_ids=controls["flat_ids"], ijk=controls["ijk"],
        positions=controls["positions"], radius=controls["radius"],
        parent_angular=controls["parent_angular"],
        direct_angular=controls["direct_angular"],
        direct_contrast=controls["direct_contrast"],
        quadrature_offsets=controls["quadrature_offsets"],
    )
    result = dict(
        status="FROZEN_SCORE_BLIND_GEOMETRY_CONTROLS",
        selection_rule="12 deterministic N256 cells selected from Fibonacci sky candidates before reading field values, training keys/counts, or heldout outcomes: one maximum-contrast boundary and one minimum-contrast nonzero-map interior in each of four radial strata (18-55, 55-95, 95-135, 135-168 cMpc/h), plus four nearest radial-boundary controls",
        control_count=len(controls["flat_ids"]),
        labels=controls["labels"],
        candidate_indices=controls["candidate_indices"].tolist(),
        source_flat_ids_N256=controls["flat_ids"].tolist(),
        source_ijk_N256=controls["ijk"].tolist(),
        source_radius_cMpc_h=controls["radius"].tolist(),
        within_cell_map_range_max=controls["direct_contrast"].tolist(),
        angular_maps={"native_nside": 512, "ordering": "RING",
                      "sha256": physics["map_hashes"]},
        pmwd_state_or_count_data_read=False,
        heldout_values_read=False,
    )
    (out / "control_selection.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
