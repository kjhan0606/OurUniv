#!/usr/bin/env python3
"""Trace a CF4 LG anchor sphere from z=0 RAMSES output to the IC output."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cf4_zoom_z0_gate import _record, _skip_header, particle_files
from cf4_lagrangian_mask import build_mask, SCHEMA


BOX = 384.0
TARGET = np.array([189.25386007706237, 191.60229489929446, 192.32376166515513])
RADIUS = 5.0


def read_particles(output: Path, want_ids: set[int] | None = None):
    rows = []
    for path in particle_files(output):
        with path.open("rb") as fh:
            _, n = _skip_header(fh)
            xyz = np.stack([_record(fh, "<f8") for _ in range(3)], axis=1) * BOX
            for _ in range(3):
                _record(fh, "<f8")
            _record(fh, "<f8")
            ids = _record(fh, "<i8")
        if want_ids is None:
            rows.append((ids, xyz))
        else:
            keep = np.isin(ids, np.fromiter(want_ids, dtype=np.int64))
            if np.any(keep):
                rows.append((ids[keep], xyz[keep]))
        if sum(len(x[0]) for x in rows) and want_ids is not None:
            pass
    if not rows:
        return np.empty((0,), np.int64), np.empty((0, 3), float)
    return np.concatenate([x[0] for x in rows]), np.concatenate([x[1] for x in rows])


def main() -> None:
    z0 = Path("/gpfs/kjhan/CF4/ramses/cf4_production_ic_z0_dmo_v1/job_388424/output_00002")
    ini = Path("/gpfs/kjhan/CF4/ramses/cf4_production_ic_z0_dmo_v1/job_388424/output_00001")
    root = Path("/gpfs/kjhan/CF4/zoom/cf4_lg_trace_v1")
    root.mkdir(parents=True, exist_ok=True)
    ids, pos = read_particles(z0)
    dr = (pos - TARGET + BOX / 2) % BOX - BOX / 2
    keep = np.einsum("ij,ij->i", dr, dr) <= RADIUS ** 2
    selected = ids[keep]
    if len(selected) < 100:
        raise RuntimeError(f"too few anchor particles: {len(selected)}")
    init_ids, init_pos = read_particles(ini, set(map(int, selected)))
    order = np.argsort(init_ids)
    traced = init_pos[order]
    np.savez(root / "traced_lagrangian.npz", lagrangian=traced,
             particle_ids=init_ids[order], target=TARGET, radius=RADIUS,
             box_size_mpc_h=BOX)
    built = build_mask(traced, BOX, base_level=9, buffer_mpc_h=1.5,
                       subbox_pad_base_cells=2)
    np.savez(root / "lg_mask_l9.npz", schema=SCHEMA,
             box_size_mpc_h=BOX, base_level=9, base_cells=built["cells"],
             subbox_lo_base=built["lo"], subbox_hi_base=built["hi"],
             requires_periodic_origin_shift=built["requires_shift"])
    report = {"schema": "ouruniv-cf4-traced-lg-mask-v1", "stage": "8/8",
              "z0_output": str(z0), "initial_output": str(ini),
              "target_cMpc_h": TARGET.tolist(), "radius_cMpc_h": RADIUS,
              "selected_z0_particles": int(len(selected)),
              "traced_initial_particles": int(len(traced)),
              "mask": str(root / "lg_mask_l9.npz"),
              "subbox_lo_base": built["lo"].tolist(),
              "subbox_hi_base": built["hi"].tolist(),
              "requires_periodic_origin_shift": bool(built["requires_shift"]),
              "interpretation": "Trace-derived mask; no zoom IC generated yet."}
    (root / "trace_mask_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
