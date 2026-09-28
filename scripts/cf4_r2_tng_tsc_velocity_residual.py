"""Read existing TNG products for a bounded halo-minus-coarse-flow residual.

This is one-box auxiliary scale evidence, not a CF4-selected galaxy/FoG
calibration. It downloads nothing, reads no raw snapshot particles, and writes
only an aggregate summary.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import h5py
import numpy as np

from cf4_r2_velocity_residual import (
    residual_summary,
    tsc_deposit_2x_moments,
    tsc_interpolate_periodic,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
MOMENT_ROOT = BASE / "bundle_c_v1/total_matter_v1"
MOMENTS = MOMENT_ROOT / "matter_moments.h5"
MOMENT_RESULT = MOMENT_ROOT / "result.json"
CATALOG = BASE / "bundle_c_v1/tng_operator_v2/native_catalog.h5"
OUT = BASE / "r2_tng_tsc_velocity_residual_v1"
BOX_CMPc_H = 75.0
GRID_COARSE = 25
MIN_DM_PARTICLES = 1000


def sha256_array(array):
    view = np.ascontiguousarray(array).view(np.uint8)
    return hashlib.sha256(view).hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm is required for catalog reads")
    expected = os.environ.get("CF4_EXPECTED_COMMIT")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    if not expected or head != expected:
        raise RuntimeError(f"source commit is not pinned: {expected} != {head}")
    checked = subprocess.run(["git", "diff", "--quiet", expected, "--",
        "src/cf4_r2_velocity_residual.py",
        "scripts/cf4_r2_tng_tsc_velocity_residual.py",
        "tests/test_cf4_r2_velocity_residual.py"], cwd=ROOT)
    if checked.returncode:
        raise RuntimeError("submitted residual-diagnostic sources changed after pin")
    if OUT.exists():
        raise FileExistsError(OUT)

    started = time.monotonic()
    provenance = json.loads(MOMENT_RESULT.read_text())
    if provenance.get("status") != "TOTAL_MATTER_PRIOR_SOURCE_NOT_CF4_POSTERIOR":
        raise ValueError("existing TNG moment provenance changed")
    with h5py.File(MOMENTS, "r") as handle:
        if handle.attrs.get("status") != "NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE":
            raise ValueError("native moment field provenance changed")
        if handle.attrs.get("field_order") != (
                "mass_Msun, momentum_xyz_Msun_km_s, diagonal_second_xyz_Msun_km2_s2"):
            raise ValueError("native moment channel order changed")
        tng_h = float(handle.attrs["h"])
        tng_omega_m = float(handle.attrs["Omega_m"])
        if abs(tng_h - 0.6774) > 1e-8 or abs(tng_omega_m - 0.3089) > 1e-8:
            raise ValueError("TNG cosmology metadata changed")
        fine_moments = handle["coarse"][:]
    coarse_moments = tsc_deposit_2x_moments(fine_moments)
    mass, momentum = coarse_moments[0], coarse_moments[1:4]
    if np.any(mass <= 0):
        raise ValueError("TSC coarse flow grid contains empty mass cells")
    velocity_grid = momentum / mass[None]
    source_sums = fine_moments.sum(axis=(1, 2, 3), dtype=np.float64)
    target_sums = coarse_moments.sum(axis=(1, 2, 3), dtype=np.float64)
    conservation = np.abs(target_sums - source_sums) / np.maximum(
        np.abs(source_sums), np.sum(np.abs(fine_moments), axis=(1, 2, 3)) * 1e-15 + 1.0)
    if np.max(conservation) > 3e-12:
        raise AssertionError(f"TSC restriction failed moment conservation: {conservation}")

    residual_parts = {"central": [], "satellite": []}
    counts = dict(cosmological_subhalos=0, dm_particles_ge_threshold=0,
                  central_dm_particles_ge_threshold=0,
                  satellite_dm_particles_ge_threshold=0)
    catalog_hash = hashlib.sha256()
    with h5py.File(CATALOG, "r") as catalog:
        if catalog.attrs.get("status") != "COMPLETE_NATIVE_FIELD_COPY_NO_SELECTION":
            raise ValueError("native catalog provenance changed")
        chunk_names = sorted(catalog["chunks"].keys(), key=int)
        if len(chunk_names) != 448:
            raise ValueError(f"native catalog chunk count changed: {len(chunk_names)}")
        header = catalog["chunks/0/Header"].attrs
        if (abs(float(header["Time"]) - 1.0) > 1e-8
                or abs(float(header["BoxSize"]) / 1000.0 - BOX_CMPc_H) > 1e-8
                or abs(float(header["HubbleParam"]) - tng_h) > 1e-8
                or abs(float(header["Omega0"]) - tng_omega_m) > 1e-8):
            raise ValueError("catalog and moment redshift/box/cosmology disagree")
        group_first_parts = []
        for chunk_name in chunk_names:
            group = catalog["chunks"][chunk_name]["Group"]
            if "GroupFirstSub" in group:
                group_first_parts.append(group["GroupFirstSub"][:].astype(np.int64))
        group_first = np.concatenate(group_first_parts)
        if len(group_first) != int(header["Ngroups_Total"]):
            raise ValueError("native group-first-subhalo table is incomplete")
        catalog_hash.update(np.ascontiguousarray(group_first).view(np.uint8))
        subhalo_offset = 0
        total_subhalos = 0
        for chunk_name in chunk_names:
            chunk = catalog["chunks"][chunk_name]
            if "Subhalo" not in chunk or "SubhaloFlag" not in chunk["Subhalo"]:
                continue
            sub = chunk["Subhalo"]
            required = ("SubhaloFlag", "SubhaloLenType", "SubhaloPos", "SubhaloVel",
                        "SubhaloGrNr")
            if any(name not in sub for name in required):
                raise ValueError(f"incomplete subhalo fields in chunk {chunk_name}")
            flag = sub["SubhaloFlag"][:].astype(bool)
            n_dm = sub["SubhaloLenType"][:, 1]
            pos_all = sub["SubhaloPos"][:]
            vel_all = sub["SubhaloVel"][:]
            if (pos_all.shape != vel_all.shape or pos_all.shape != (len(flag), 3)
                    or len(n_dm) != len(flag)):
                raise ValueError(f"subhalo arrays misaligned in chunk {chunk_name}")
            chunk_subhalo_count = int(chunk["Header"].attrs["Nsubgroups_ThisFile"])
            if chunk_subhalo_count != len(flag):
                raise ValueError(f"native subhalo offset mismatch in chunk {chunk_name}")
            global_subhalo_id = subhalo_offset + np.arange(len(flag), dtype=np.int64)
            group_id = sub["SubhaloGrNr"][:].astype(np.int64)
            if np.any((group_id < 0) | (group_id >= len(group_first))):
                raise ValueError(f"subhalo parent-group index out of range in chunk {chunk_name}")
            central = group_first[group_id] == global_subhalo_id
            counts["cosmological_subhalos"] += int(flag.sum())
            keep = flag & (n_dm >= MIN_DM_PARTICLES)
            counts["dm_particles_ge_threshold"] += int(keep.sum())
            for category, category_mask in (("central", central), ("satellite", ~central)):
                selected = keep & category_mask
                if not np.any(selected):
                    continue
                pos = np.mod(pos_all[selected].astype(np.float64) / 1000.0, BOX_CMPc_H)
                vel = vel_all[selected].astype(np.float64)
                if not np.isfinite(pos).all() or not np.isfinite(vel).all():
                    raise ValueError(f"nonfinite selected {category} phase space in chunk {chunk_name}")
                counts[f"{category}_dm_particles_ge_threshold"] += len(pos)
                for name, array in (("global_subhalo_id", global_subhalo_id[selected]),
                                    ("group_id", group_id[selected]),
                                    ("pos", pos_all[selected]), ("vel", vel_all[selected])):
                    catalog_hash.update(name.encode("ascii"))
                    catalog_hash.update(np.ascontiguousarray(array).view(np.uint8))
                flow = tsc_interpolate_periodic(velocity_grid, pos, BOX_CMPc_H)
                residual_parts[category].append(vel - flow)
            subhalo_offset += len(flag)
            total_subhalos += len(flag)
        if total_subhalos != int(header["Nsubgroups_Total"]):
            raise ValueError("native subhalo table is incomplete")

    if not all(residual_parts.values()):
        raise ValueError("no resolved cosmological subhalos passed the fixed DM-count cut")
    summaries = {name: residual_summary(np.concatenate(parts, axis=0))
                 for name, parts in residual_parts.items()}
    result = dict(
        classification="ONE_TNG_BOX_RESOLVED_SUBHALO_MINUS_TSC_3CMPC_H_MATTER_FLOW",
        status="AUXILIARY_SCALE_DIAGNOSTIC_NOT_CF4_FOG_CALIBRATION",
        job_id=os.environ["SLURM_JOB_ID"],
        source_commit=head,
        source=dict(moment_path=str(MOMENTS), catalog_path=str(CATALOG),
                    moment_result_status=provenance["status"], h=tng_h,
                    Omega_m=tng_omega_m, box_cMpc_h=BOX_CMPc_H,
                    native_catalog_chunks=len(chunk_names),
                    coarse_moment_sha256=sha256_array(fine_moments)),
        operator=dict(input_moment_grid=50, input_cell_cMpc_h=1.5,
                      output_grid=GRID_COARSE, output_cell_cMpc_h=3.0,
                      moment_transfer="periodic cell-centred TSC of mass/momentum/diagonal second moments",
                      subhalo_velocity_readout="periodic cell-centred TSC gather of mass-weighted momentum/mass velocity",
                      minimum_subhalo_dm_particles=MIN_DM_PARTICLES,
                      global_moment_conservation_relative_error=conservation.tolist()),
        counts=counts, selected_catalog_phase_space_sha256=catalog_hash.hexdigest(),
        residual_by_role=summaries,
        comparison_100_km_s=dict(input_likelihood_width=100.0,
            ratio_to_central_isotropic_1d_sigma=(
                summaries["central"]["isotropic_1d_sigma_km_s"] / 100.0),
            ratio_to_satellite_isotropic_1d_sigma=(
                summaries["satellite"]["isotropic_1d_sigma_km_s"] / 100.0)),
        interpretation=("Compares resolved native TNG central/satellite subhalo velocities "
            "with a 3 cMpc/h TSC-smoothed total-matter velocity field. A single "
            "75-cMpc/h TNG box, its different cosmology, subhalo-only tracer sample "
            "and missing CF4/2M++ selection/group law prevent calibration or direct "
            "transfer to ungrouped 2M++ galaxies. It is auxiliary scale evidence only."),
        Q_GOAL=("Tests whether the fixed 100 km/s source residual scale is of the right "
            "order for a same-field R2 operator, without using LG identities."),
        Q_LEAN=("Reuses existing TNG moments and native catalog; one bounded streaming "
            "read, no new data download, raw snapshot, gravity run, sampler, or gate ladder."),
        MW_M31_M33=("No truth identities or observed matching used. MW/M31 stay ambiguous; "
            "M33 remains unresolved until all three observables constrain the same new field."),
        field_posterior=False, heldout_scored=False, R2_complete=False,
        elapsed_seconds=time.monotonic() - started,
        host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
