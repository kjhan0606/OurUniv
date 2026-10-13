"""Fixed-state three-arm ray-volume reference for the active v6 count term.

The diagnostic compares active N256/GL2 source cells with a ray-box source-
volume reference, holding the parent angular map fixed in one arm and using
the pinned native NSIDE512 map in the other. It is not a likelihood or fit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cf4_r2_count_exposure import (  # noqa: E402
    build_population_exposure_masks,
    reshape_population_exposure_masks,
)
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells  # noqa: E402
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses  # noqa: E402
from cf4_r2_ray_cost_profile import candidate_cap, ray_box_intervals  # noqa: E402
from cf4_r2_selection_ray_integral import load_physics_inputs  # noqa: E402
from cf4_r2_shell_cdf_count import (  # noqa: E402
    predict_shell_cdf_intensity,
    predict_source_volume_intensity,
)

BASE = Path("/gpfs/kjhan/CF4/z0_density")
FIELD_DIR = BASE / "r2_conditional_map_20261005_v3"
FIELD_PATH = FIELD_DIR / "conditional_best_field.npz"
PARAMETER_PATH = FIELD_DIR / "best_parameters.npz"
SOURCE_PATH = BASE / "r2_marked_source_geometry_v1/geometry.npz"
SPLIT_PATH = BASE / "r2_sky_closed_split_v6/split.npz"
N = 256
N_COUNT = 128
NVOXEL = N_COUNT**3
BOX = 384.0
DX = BOX / N
COUNT_DX = BOX / N_COUNT
OBSERVER = np.full(3, BOX / 2.0)
POPULATIONS = 6
SOURCE_VOLUME_FACTOR = (DX / COUNT_DX)**3
RAY_NSIDE = 512
RADIAL_ORDER = 2
ANGULAR_CHECK_NSIDE = 1024
RADIAL_CHECK_ORDER = 4
SOURCE_BATCH = 1024


def source_cell_ray_volume_rule(ijk, ray_nside, radial_order, rotation):
    """Reuse exact slab intersections for ``dOmega r^2 dr/Vcell`` samples."""
    ijk = np.asarray(ijk)
    rotation = np.asarray(rotation, dtype=np.float64)
    if (ijk.shape != (3,) or not np.issubdtype(ijk.dtype, np.integer)
            or np.any(ijk < 0) or np.any(ijk >= N)
            or ray_nside < 1 or ray_nside & (ray_nside - 1)
            or radial_order < 1 or rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-10)):
        raise ValueError("invalid source-cell ray quadrature settings")
    low = ijk.astype(np.float64) * DX - BOX / 2.
    high = low + DX
    center_sky = rotation @ ((low + high) / 2.)
    axis, cap = candidate_cap(center_sky, low, high, rotation)
    candidates = hp.query_disc(ray_nside, axis, cap, inclusive=True, nest=False)
    sky_directions = np.asarray(hp.pix2vec(ray_nside, candidates, nest=False), dtype=np.float64)
    grid_directions = rotation.T @ sky_directions
    enter, leave, intersects = ray_box_intervals(grid_directions, low, high)
    valid = intersects & (leave > enter)
    if not np.any(valid):
        raise ValueError("ray subdivision has no pixel-centre ray through source cell")
    directions = grid_directions[:, valid].T
    rlo, rhi = enter[valid], leave[valid]
    nodes, gl_weights = np.polynomial.legendre.leggauss(radial_order)
    half = (rhi - rlo) / 2.
    radii = (rhi + rlo)[:, None] / 2. + half[:, None] * nodes[None, :]
    weights = (hp.nside2pixarea(ray_nside) * half[:, None]
               * gl_weights[None, :] * radii**2 / DX**3)
    positions = OBSERVER[None, None, :] + directions[:, None, :] * radii[:, :, None]
    native_pixels = hp.vec2pix(512, *sky_directions[:, valid], nest=False).astype(np.int64)
    return {
        "positions": positions.reshape(-1, 3),
        "weights": weights.reshape(-1),
        "native_pixels": native_pixels,
        "candidate_pixels": int(len(candidates)),
        "intersecting_rays": int(np.count_nonzero(valid)),
        "volume_weight_sum": float(np.sum(weights)),
        "volume_closure_error": float(np.sum(weights) - 1.),
        "ray_nside": int(ray_nside),
        "radial_order": int(radial_order),
    }


def _relative_delta(reference, value):
    reference = np.asarray(reference, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    denominator = np.maximum(np.abs(reference), np.abs(value))
    delta = value - reference
    relative = np.empty(np.broadcast_shapes(reference.shape, value.shape), dtype=object)
    for index in np.ndindex(relative.shape):
        relative[index] = (float(delta[index] / denominator[index])
                           if denominator[index] > 0. else None)
    return delta.tolist(), relative.tolist()


def _padded_chunk(positions, weights, angular, velocity, intrinsic_rate, start, stop):
    p = np.asarray(positions[start:stop], dtype=np.float64)
    w = np.asarray(weights[start:stop], dtype=np.float64)
    a = np.asarray(angular[:, start:stop], dtype=np.float64)
    length = len(p)
    if length < SOURCE_BATCH:
        pad = SOURCE_BATCH - length
        p = np.concatenate((p, np.broadcast_to(p[:1], (pad, 3))), axis=0)
        w = np.pad(w, (0, pad))
        a = np.concatenate((a, np.broadcast_to(a[:, :1], (2, pad))), axis=1)
    v = np.broadcast_to(np.asarray(velocity)[None, :], (SOURCE_BATCH, 3))
    intrinsic = (np.asarray(intrinsic_rate)[:, None]
                 * (w * SOURCE_VOLUME_FACTOR)[None, :])
    return p, v, intrinsic, a


def _ray_angles(sample, angular_maps, parent_angular, radial_order):
    per_ray = np.stack([angular_maps[0][sample["native_pixels"]],
                        angular_maps[1][sample["native_pixels"]]])
    pixel = np.repeat(per_ray, radial_order, axis=1)
    constant = np.broadcast_to(
        np.asarray(parent_angular, dtype=np.float64)[:, None], pixel.shape).copy()
    return constant, pixel


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("angular ray reference must run on a Slurm GPU")
    expected_commit = os.environ.get("CF4_EXPECTED_COMMIT")
    expected_bundle = os.environ.get("CF4_EXPECTED_SOURCE_BUNDLE_SHA256")
    out = Path(os.environ["CF4_R2_OUT_DIR"])
    if (not expected_commit or not expected_bundle or not out.is_dir()
            or not (out / "controls.npz").is_file()):
        raise RuntimeError("source commit and fresh score-blind controls are required")
    if (out / "result.json").exists():
        raise FileExistsError(out / "result.json")
    started = time.monotonic()
    report = {
        "status": "STARTED",
        "classification": "LOCAL_V6_COUNT_OPERATOR_RAY_VOLUME_REFERENCE_NOT_LIKELIHOOD",
        "job_id": os.environ["SLURM_JOB_ID"],
        "base_commit": expected_commit,
        "source_bundle_sha256": expected_bundle,
        "field_path": str(FIELD_PATH),
        "field_result_status": None,
        "heldout_count_values_loaded": False,
        "heldout_FP_values_loaded": False,
        "training_count_values_loaded": False,
        "PMWD_evolutions": 0,
        "optimizer_steps": 0,
        "full_likelihood_evaluated": False,
        "R2_complete": False,
        "LG_roles": "MW/M31 remain ambiguous; M33 unresolved; no LG observables or truth IDs enter this diagnostic.",
        "completed_source_cells": 0,
        "source_cell_results": [],
        "Q_GOAL": "tests source-volume and angular-selection approximations inside the same CF4-conditioned z=0 count operator; no posterior/map is delivered",
        "Q_LEAN": "12 fixed geometry controls, one angular subdivision and one radial-order check; no full-grid exposure, fit, PM replay, heldout score or simulation",
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        report["max_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        field_result = json.loads((FIELD_DIR / "result.json").read_text())
        if (field_result.get("status") != "CONDITIONAL_MAP_ATTEMPT_INCOMPLETE"
                or field_result.get("best_evaluation") != 6
                or field_result.get("heldout_count_values_loaded")
                or field_result.get("heldout_FP_marks_loaded")):
            raise ValueError("saved conditional-map diagnostic failed its provenance gate")
        report["field_result_status"] = field_result["status"]
        report["field_best_evaluation"] = field_result["best_evaluation"]

        selection = json.loads((out / "control_selection.json").read_text())
        with np.load(out / "controls.npz", allow_pickle=False) as archive:
            controls = {key: archive[key].copy() for key in archive.files}
        ijk = np.asarray(controls["ijk"], dtype=np.int64)
        flat_ids = np.asarray(controls["flat_ids"], dtype=np.int64)
        if ijk.shape != (12, 3) or flat_ids.shape != (12,):
            raise ValueError("expected the frozen twelve source-cell controls")
        if not np.array_equal(flat_ids, (ijk[:, 0] * N + ijk[:, 1]) * N + ijk[:, 2]):
            raise ValueError("flattened source-cell IDs do not match their declared N256 cells")
        if selection.get("source_ijk_N256") != ijk.tolist():
            raise ValueError("score-blind control selection and NPZ cells differ")
        labels = selection["labels"]
        if len(labels) != len(ijk) or "map_boundary_55_95" not in labels:
            raise ValueError("frozen control labels do not include the reference control")
        reference_index = labels.index("map_boundary_55_95")

        with np.load(SOURCE_PATH, allow_pickle=False) as archive:
            source = {key: archive[key].copy() for key in archive.files}
        required_source_keys = {
            "positions", "angular", "radial_table", "modulus_table", "redshift_table"
        }
        missing_source_keys = required_source_keys.difference(source)
        if missing_source_keys:
            raise ValueError(
                f"pinned source geometry is missing keys: {sorted(missing_source_keys)}")
        if source["positions"].shape != (128**3, 3):
            raise ValueError("pinned N128 source geometry changed")
        parent = ijk // 2
        parent_flat = (parent[:, 0] * 128 + parent[:, 1]) * 128 + parent[:, 2]
        expected_parent_angle = source["angular"][:, parent_flat].T
        if not np.array_equal(controls["parent_angular"], expected_parent_angle):
            raise ValueError("parent-cell map values do not match the pinned source geometry")

        physics = load_physics_inputs()
        if physics["map_hashes"] != selection["angular_maps"]["sha256"]:
            raise ValueError("control selection and pinned angular-map hashes differ")
        with np.load(FIELD_PATH, allow_pickle=False) as archive:
            rho_node = jnp.asarray(archive["rho"])
            velocity_node = jnp.asarray(archive["mean_velocity_km_s"])
        with np.load(PARAMETER_PATH, allow_pickle=False) as archive:
            tracer = jnp.asarray(archive["tracer_white"])
        if rho_node.shape != (N, N, N) or velocity_node.shape != (3, N, N, N):
            raise ValueError("saved N256 field has unexpected geometry")
        if tracer.shape != (9,):
            raise ValueError("saved conditional count nuisance vector changed")

        split = np.load(SPLIT_PATH, allow_pickle=False)
        train_keys = np.asarray(split["train_keys"], dtype=np.int64).copy()
        exposure, _ = build_population_exposure_masks(
            N_COUNT, split["heldout_flat_voxels"],
            split["train_window_excluded_keys"],
            split["heldout_window_excluded_keys"])
        del split
        if (train_keys.ndim != 1 or np.any(train_keys < 0)
                or np.any(train_keys >= POPULATIONS * NVOXEL)
                or len(np.unique(train_keys)) != len(train_keys)):
            raise ValueError("v6 training-key geometry is malformed")
        exposure_flat = np.asarray(exposure, dtype=bool)
        exposure_grid = reshape_population_exposure_masks(
            exposure_flat, N_COUNT, population_count=POPULATIONS)
        exposure_jax = jnp.asarray(exposure_grid)
        report["training_key_geometry_loaded"] = True
        report["exposure_contract"] = (
            "v6 training exposure geometry only; empty exposed cells included; "
            "heldout and buffered geometry excluded; no observed counts loaded")

        density, count_velocity = native_mass_momentum_to_count_cells(
            rho_node, velocity_node, BOX)
        rates_np = np.asarray(tracer_masses(density, tracer).reshape(5, -1))
        count_velocity_np = np.asarray(jnp.moveaxis(count_velocity, 0, -1).reshape(-1, 3))
        geom = dict(
            observer=jnp.asarray(OBSERVER), box_size_cMpc_h=BOX,
            hubble_km_s_Mpc=74.6, little_h=0.746,
            radius_table_cMpc_h=jnp.asarray(source["radial_table"]),
            modulus_table_h=jnp.asarray(source["modulus_table"]),
            redshift_table=jnp.asarray(source["redshift_table"]),
            grid_size=N_COUNT,
        )
        geom.update(tracer_geometry(tracer, geom))
        exposure_flat = exposure_flat.astype(np.float64).reshape(POPULATIONS, NVOXEL)

        def active_gl2(position, velocity, intrinsic, angle):
            return predict_source_volume_intensity(
                position, velocity, intrinsic, angle,
                source_spacing=DX, volume_order=2, **geom,
                order=4, segments=8, deposition="tsc")

        active_gl2_jit = jax.jit(active_gl2)

        def exposure_kernel(position, velocity, intrinsic, angle):
            return predict_shell_cdf_intensity(
                position, velocity, intrinsic, angle,
                **geom, order=4, segments=8, deposition="exposure_total",
                exposure_masks=exposure_jax)

        exposure_kernel_jit = jax.jit(exposure_kernel)

        active_expected = np.zeros(POPULATIONS, dtype=np.float64)
        parent_expected = np.zeros(POPULATIONS, dtype=np.float64)
        pixel_expected = np.zeros(POPULATIONS, dtype=np.float64)
        reference_grid = None
        reference_velocity = None
        reference_rate = None
        geometry_rows = []

        for cell_i, cell_ijk in enumerate(ijk):
            parent_id = int(parent_flat[cell_i])
            source_position = jnp.asarray(controls["positions"][cell_i:cell_i+1])
            source_velocity = jnp.asarray(count_velocity_np[parent_id:parent_id+1])
            source_rate = jnp.asarray(rates_np[:, parent_id:parent_id+1]
                                      * SOURCE_VOLUME_FACTOR)
            parent_angle = jnp.asarray(controls["parent_angular"][cell_i:cell_i+1].T)
            grid = np.asarray(active_gl2_jit(
                source_position, source_velocity, source_rate, parent_angle))
            active_cell_expected = np.sum(
                grid.reshape(POPULATIONS, NVOXEL) * exposure_flat, axis=1)
            active_expected += active_cell_expected
            if cell_i == reference_index:
                reference_grid = grid.reshape(POPULATIONS, NVOXEL)
                reference_velocity = count_velocity_np[parent_id].copy()
                reference_rate = rates_np[:, parent_id].copy()

            coarse = source_cell_ray_volume_rule(
                cell_ijk, RAY_NSIDE, RADIAL_ORDER, physics["rotation"])
            angle_parent, angle_pixel = _ray_angles(
                coarse, physics["maps"], controls["parent_angular"][cell_i], RADIAL_ORDER)
            arms = []
            for angle_array in (angle_parent, angle_pixel):
                expected = np.zeros(POPULATIONS, dtype=np.float64)
                for start in range(0, len(coarse["weights"]), SOURCE_BATCH):
                    stop = min(start + SOURCE_BATCH, len(coarse["weights"]))
                    pos, vel, intrinsic, angular = _padded_chunk(
                        coarse["positions"], coarse["weights"], angle_array,
                        reference_velocity if cell_i == reference_index
                        else count_velocity_np[parent_id],
                        reference_rate if cell_i == reference_index
                        else rates_np[:, parent_id], start, stop)
                    expected += np.asarray(exposure_kernel_jit(
                        jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(intrinsic),
                        jnp.asarray(angular)))
                arms.append(expected)
            parent_expected += arms[0]
            pixel_expected += arms[1]
            geometry_rows.append({
                "label": labels[cell_i],
                "source_ijk_N256": cell_ijk.tolist(),
                "source_flat_id_N256": int(flat_ids[cell_i]),
                "source_radius_cMpc_h": float(controls["radius"][cell_i]),
                "ray_candidate_pixels": coarse["candidate_pixels"],
                "ray_intersecting_pixels": coarse["intersecting_rays"],
                "radial_order": RADIAL_ORDER,
                "volume_weight_sum_unrenormalized": coarse["volume_weight_sum"],
                "volume_closure_error_unrenormalized": coarse["volume_closure_error"],
                "active_GL2_cell_constant_expected_by_population": active_cell_expected.tolist(),
                "refined_parent_map_expected_by_population": arms[0].tolist(),
                "refined_native_NSIDE512_map_expected_by_population": arms[1].tolist(),
            })
            report["source_cell_results"] = geometry_rows.copy()
            report["completed_source_cells"] = cell_i + 1
            report["status"] = "RUNNING_FIXED_STATE_RAY_REFERENCE"
            save()

        if reference_grid is None or reference_velocity is None or reference_rate is None:
            raise RuntimeError("predeclared support-reference source cell was not evaluated")
        report["active_GL2_cell_constant_selected_patch_expected_sum"] = active_expected.tolist()
        report["refined_parent_map_selected_patch_expected_sum"] = parent_expected.tolist()
        report["refined_NSIDE512_map_selected_patch_expected_sum"] = pixel_expected.tolist()
        report["refined_parent_minus_active_GL2_delta"] = _relative_delta(
            active_expected, parent_expected)[0]
        report["refined_parent_minus_active_GL2_relative_delta"] = _relative_delta(
            active_expected, parent_expected)[1]
        report["refined_pixel_minus_parent_delta"] = _relative_delta(
            parent_expected, pixel_expected)[0]
        report["refined_pixel_minus_parent_relative_delta"] = _relative_delta(
            parent_expected, pixel_expected)[1]
        for index, row in enumerate(geometry_rows):
            row["refined_parent_minus_active_GL2_delta_by_population"] = (
                np.asarray(row["refined_parent_map_expected_by_population"])
                - np.asarray(row["active_GL2_cell_constant_expected_by_population"])).tolist()
            row["refined_pixel_minus_parent_delta_by_population"] = (
                np.asarray(row["refined_native_NSIDE512_map_expected_by_population"])
                - np.asarray(row["refined_parent_map_expected_by_population"])).tolist()
            row["refined_pixel_minus_parent_relative_delta_by_population"] = _relative_delta(
                row["refined_parent_map_expected_by_population"],
                row["refined_native_NSIDE512_map_expected_by_population"])[1]
        report["source_cell_results"] = geometry_rows

        # Freeze one score-blind, actually supported training key per population
        # from a predeclared map-boundary control. No observed count is read.
        selected_keys = []
        for population in range(POPULATIONS):
            population_keys = train_keys[train_keys // NVOXEL == population]
            key_values = reference_grid[population, population_keys % NVOXEL]
            supported = np.isfinite(key_values) & (key_values > 0.)
            if not np.any(supported):
                selected_keys.append({
                    "population": population,
                    "status": "NO_SUPPORTED_TRAINING_KEY",
                    "training_keys_checked": int(len(population_keys)),
                })
                continue
            candidates = np.flatnonzero(supported)
            chosen_local = int(candidates[np.argmax(key_values[candidates])])
            key = int(population_keys[chosen_local])
            voxel = np.asarray(np.unravel_index(key % NVOXEL, (N_COUNT,) * 3), dtype=np.int32)
            selected_keys.append({
                "population": population,
                "status": "SUPPORTED",
                "key": key,
                "voxel_ijk": voxel.tolist(),
                "selection": "maximum positive active-GL2 model intensity among v6 training keys; no observed counts used",
                "active_GL2_intensity": float(key_values[chosen_local]),
                "supported_training_keys_in_control": int(np.count_nonzero(supported)),
                "_voxel": voxel,
            })

        def key_intensity(sample, angular, population, voxel, rate, velocity):
            total = 0.0
            def kernel(position, vel, intrinsic, angle):
                return predict_shell_cdf_intensity(
                    position, vel, intrinsic, angle, **geom,
                    order=4, segments=8, deposition="tsc",
                    target_population=int(population), target_voxel=jnp.asarray(voxel))
            compiled = jax.jit(kernel)
            for start in range(0, len(sample["weights"]), SOURCE_BATCH):
                stop = min(start + SOURCE_BATCH, len(sample["weights"]))
                pos, vel, intrinsic, angle = _padded_chunk(
                    sample["positions"], sample["weights"], angular,
                    velocity, rate, start, stop)
                total += float(compiled(jnp.asarray(pos), jnp.asarray(vel),
                                        jnp.asarray(intrinsic), jnp.asarray(angle)))
            return total

        reference = source_cell_ray_volume_rule(
            ijk[reference_index], RAY_NSIDE, RADIAL_ORDER, physics["rotation"])
        ref_parent_angle, ref_pixel_angle = _ray_angles(
            reference, physics["maps"], controls["parent_angular"][reference_index],
            RADIAL_ORDER)
        supported_key_results = []
        for item in selected_keys:
            if item["status"] != "SUPPORTED":
                supported_key_results.append({k: v for k, v in item.items() if k != "_voxel"})
                continue
            population = item["population"]
            refined_parent_value = key_intensity(
                reference, ref_parent_angle, population, item["_voxel"],
                reference_rate, reference_velocity)
            refined_pixel_value = key_intensity(
                reference, ref_pixel_angle, population, item["_voxel"],
                reference_rate, reference_velocity)
            item["refined_parent_map_intensity"] = refined_parent_value
            item["refined_NSIDE512_map_intensity"] = refined_pixel_value
            item["parent_minus_active_absolute_delta"] = (
                refined_parent_value - item["active_GL2_intensity"])
            item["pixel_minus_parent_absolute_delta"] = (
                refined_pixel_value - refined_parent_value)
            item["pixel_minus_parent_relative_delta"] = (
                (refined_pixel_value - refined_parent_value)
                / max(abs(refined_pixel_value), abs(refined_parent_value))
                if max(abs(refined_pixel_value), abs(refined_parent_value)) > 0.
                else None)
            supported_key_results.append({k: v for k, v in item.items() if k != "_voxel"})
        report["support_aware_training_key_results"] = supported_key_results
        report["support_key_selection_contract"] = (
            "one maximum positive active-kernel key per population from the "
            "predeclared map_boundary_55_95 source cell; training-key IDs only, "
            "no observed count values or scores")

        # One predeclared angular subdivision and radial-order check on the
        # same geometry-selected cell; no adaptive follow-up ladder.
        checks = {}
        for name, nside, order in (
                ("angular_subdivision", ANGULAR_CHECK_NSIDE, RADIAL_ORDER),
                ("radial_order", RAY_NSIDE, RADIAL_CHECK_ORDER)):
            sample = source_cell_ray_volume_rule(
                ijk[reference_index], nside, order, physics["rotation"])
            parent_angle, pixel_angle = _ray_angles(
                sample, physics["maps"], controls["parent_angular"][reference_index], order)
            arm_outputs = []
            for angle in (parent_angle, pixel_angle):
                expected = np.zeros(POPULATIONS, dtype=np.float64)
                for start in range(0, len(sample["weights"]), SOURCE_BATCH):
                    stop = min(start + SOURCE_BATCH, len(sample["weights"]))
                    pos, vel, intrinsic, angular = _padded_chunk(
                        sample["positions"], sample["weights"], angle,
                        reference_velocity, reference_rate, start, stop)
                    expected += np.asarray(exposure_kernel_jit(
                        jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(intrinsic),
                        jnp.asarray(angular)))
                arm_outputs.append(expected)
            check = {
                "ray_nside": nside,
                "radial_order": order,
                "intersecting_rays": sample["intersecting_rays"],
                "volume_weight_sum_unrenormalized": sample["volume_weight_sum"],
                "volume_closure_error_unrenormalized": sample["volume_closure_error"],
                "parent_map_expected_by_population": arm_outputs[0].tolist(),
                "native_map_expected_by_population": arm_outputs[1].tolist(),
            }
            if name == "angular_subdivision":
                reference_parent = np.asarray(
                    geometry_rows[reference_index]["refined_parent_map_expected_by_population"])
                reference_pixel = np.asarray(
                    geometry_rows[reference_index]["refined_native_NSIDE512_map_expected_by_population"])
                parent_abs, parent_rel = _relative_delta(reference_parent, arm_outputs[0])
                pixel_abs, pixel_rel = _relative_delta(reference_pixel, arm_outputs[1])
                check["parent_NSIDE1024_minus_NSIDE512_delta"] = parent_abs
                check["parent_NSIDE1024_minus_NSIDE512_relative_delta"] = parent_rel
                check["native_map_NSIDE1024_minus_NSIDE512_delta"] = pixel_abs
                check["native_map_NSIDE1024_minus_NSIDE512_relative_delta"] = pixel_rel
                check["pixel_minus_parent_delta"] = _relative_delta(
                    arm_outputs[0], arm_outputs[1])[0]
                check["pixel_minus_parent_relative_delta"] = _relative_delta(
                    arm_outputs[0], arm_outputs[1])[1]
            else:
                reference_parent = np.asarray(
                    geometry_rows[reference_index]["refined_parent_map_expected_by_population"])
                reference_pixel = np.asarray(
                    geometry_rows[reference_index]["refined_native_NSIDE512_map_expected_by_population"])
                parent_abs, parent_rel = _relative_delta(reference_parent, arm_outputs[0])
                pixel_abs, pixel_rel = _relative_delta(reference_pixel, arm_outputs[1])
                check["parent_radial_order4_minus_order2_delta"] = parent_abs
                check["parent_radial_order4_minus_order2_relative_delta"] = parent_rel
                check["pixel_radial_order4_minus_order2_delta"] = pixel_abs
                check["pixel_radial_order4_minus_order2_relative_delta"] = pixel_rel
            checks[name] = check
        report["operator_convergence_checks"] = checks
        report["operator_contract"] = {
            "arm_1": "active N256 GL2 source-volume nodes, parent N128 angular average held constant",
            "arm_2": "exact ray-box source-cell intersections, dOmega r^2 dr/Vcell GL radial nodes, parent angular average held constant",
            "arm_3": "same ray-box quadrature with pinned NSIDE512 RING pixel value at each ray direction",
            "shared_physics": "same saved field, source rates, coherent spherical RSD, source transfer, order-4/8 LOS count kernel, periodic handling, TSC deposition and v6 training exposure mask",
            "source_radial_cut": "source cells are integrated over their full ray-box intervals; only the existing observed-space count operator applies its 5/180 cMpc/h limits",
            "volume_weights": "physical dOmega r^2 dr/Vcell; no renormalization; closure error is reported",
            "counts_and_likelihood": "no observed count values loaded; no full Poisson score computed",
        }
        report["status"] = "COMPLETE_FIXED_STATE_LOCAL_RAY_VOLUME_REFERENCE"
        report["classification"] = "LOCAL_V6_COUNT_OPERATOR_RAY_VOLUME_REFERENCE_NOT_LIKELIHOOD"
        report["R2_complete"] = False
        report["decision"] = (
            "Compare only these 12 geometry-selected source-cell patches. If the "
            "predeclared angular/radial checks or unrenormalized geometry closure "
            "are materially unstable, treat the reference as inconclusive; no "
            "full-grid escalation or production operator change follows.")
        save()
        print(json.dumps({k: report[k] for k in (
            "status", "active_GL2_cell_constant_selected_patch_expected_sum",
            "refined_parent_map_selected_patch_expected_sum",
            "refined_NSIDE512_map_selected_patch_expected_sum",
            "operator_convergence_checks", "decision")}, allow_nan=False), flush=True)
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
