"""Fixed-state local comparison of v6 cell-constant and pixel-aware sky maps.

This is a diagnostic of the active source-volume -> RSD/K -> TSC count
operator, not a likelihood, posterior update, or field-convergence test.
Only a small, geometry-selected set of N256 source cells is evaluated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cf4_r2_count_exposure import build_population_exposure_masks  # noqa: E402
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells  # noqa: E402
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses  # noqa: E402
from cf4_r2_shell_cdf_count import predict_shell_cdf_intensity  # noqa: E402

BASE = Path("/gpfs/kjhan/CF4/z0_density")
FIELD_DIR = BASE / "r2_conditional_map_20261005_v3"
FIELD_PATH = FIELD_DIR / "conditional_best_field.npz"
PARAMETER_PATH = FIELD_DIR / "best_parameters.npz"
SOURCE_PATH = BASE / "r2_marked_source_geometry_v1/geometry.npz"
SPLIT_PATH = BASE / "r2_sky_closed_split_v6/split.npz"
N = 256
N_COUNT = 128
BOX = 384.0
DX = BOX / N
COUNT_DX = BOX / N_COUNT
OBSERVER = np.full(3, BOX / 2.0)
POPULATIONS = 6
SOURCE_VOLUME_FACTOR = (DX / COUNT_DX) ** 3
NVOXEL = N_COUNT ** 3


def source_volume_rule(order: int = 2):
    """The exact GL2 volume nodes/weights used by the active N256 target."""
    if order != 2:
        raise ValueError("this diagnostic is frozen to the active GL2 source rule")
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    ijk = np.asarray([(i, j, k) for i in range(order)
                      for j in range(order) for k in range(order)])
    offsets = nodes_1d[ijk] * DX / 2.0
    weights = np.prod(weights_1d[ijk] / 2.0, axis=1)
    return offsets, weights


def nearest_training_keys(keys, positions, velocities):
    """Nearest score-blind training key to each coherent-RSD control centre."""
    keys = np.asarray(keys, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    relative = (positions - OBSERVER + BOX / 2.0) % BOX - BOX / 2.0
    radius = np.linalg.norm(relative, axis=1)
    direction = relative / radius[:, None]
    radial_velocity = np.sum(velocities * direction, axis=1)
    displacement = 0.746 * radial_velocity / 74.6
    shifted = (positions + displacement[:, None] * direction) % BOX
    targets, distances = [], []
    for population in range(POPULATIONS):
        pop_keys = keys[keys // NVOXEL == population]
        if not len(pop_keys):
            raise ValueError(f"no training key in population {population}")
        vox = pop_keys % NVOXEL
        ijk = np.column_stack(np.unravel_index(vox, (N_COUNT,) * 3))
        key_positions = (ijk + 0.5) * COUNT_DX
        tree = cKDTree(key_positions, boxsize=BOX)
        d, local = tree.query(shifted, k=1)
        targets.append(pop_keys[local])
        distances.append(d)
    return np.asarray(targets, dtype=np.int64).T, np.asarray(distances).T, shifted


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("this fixed-state diagnostic must run on a Slurm GPU")
    out = Path(os.environ["CF4_R2_OUT_DIR"])
    if not out.is_dir() or not (out / "controls.npz").is_file():
        raise FileNotFoundError("the score-blind Slurm setup step must create controls.npz first")
    if (out / "result.json").exists():
        raise FileExistsError(out / "result.json")
    started = time.monotonic()
    report = dict(
        status="STARTED",
        classification="LOCAL_PIXEL_AWARE_ANGULAR_OPERATOR_COMPARISON_NOT_LIKELIHOOD",
        job_id=os.environ["SLURM_JOB_ID"],
        source_commit=os.environ["CF4_EXPECTED_COMMIT"],
        field_path=str(FIELD_PATH),
        field_result_status=None,
        heldout_count_values_loaded=False,
        heldout_FP_values_loaded=False,
        training_count_values_loaded=False,
        PMWD_evolutions=0,
        optimizer_steps=0,
        full_likelihood_evaluated=False,
        R2_complete=False,
        LG_roles="MW/M31 remain ambiguous; M33 unresolved; no LG observables or truth IDs enter this diagnostic.",
    )

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        field_result = json.loads((FIELD_DIR / "result.json").read_text())
        if (field_result.get("status") != "CONDITIONAL_MAP_ATTEMPT_INCOMPLETE"
                or field_result.get("best_evaluation") != 6
                or field_result.get("heldout_count_values_loaded")
                or field_result.get("heldout_FP_marks_loaded")):
            raise ValueError("saved conditional-map diagnostic state failed its provenance gate")
        report["field_result_status"] = field_result["status"]
        report["field_best_evaluation"] = field_result["best_evaluation"]

        control_selection = json.loads((out / "control_selection.json").read_text())
        with np.load(out / "controls.npz", allow_pickle=False) as f:
            controls = {key: f[key].copy() for key in f.files}
        candidate_indices = controls["candidate_indices"]
        flat_ids = controls["flat_ids"]
        ijk = controls["ijk"]
        if (candidate_indices.ndim != 1 or flat_ids.shape != candidate_indices.shape
                or ijk.shape != (len(flat_ids), 3)
                or not np.issubdtype(candidate_indices.dtype, np.integer)
                or not np.issubdtype(flat_ids.dtype, np.integer)
                or not np.issubdtype(ijk.dtype, np.integer)):
            raise ValueError("frozen control index arrays have invalid shape or dtype")
        expected_flat_ids = (ijk[:, 0].astype(np.int64) * N + ijk[:, 1]) * N + ijk[:, 2]
        if (np.any((ijk < 0) | (ijk >= N))
                or np.any((flat_ids < 0) | (flat_ids >= N**3))
                or not np.array_equal(flat_ids, expected_flat_ids)):
            raise ValueError("frozen N256 flattened IDs do not map to their declared cells")
        if (not np.array_equal(control_selection["candidate_indices"], candidate_indices.tolist())
                or not np.array_equal(control_selection["source_flat_ids_N256"], flat_ids.tolist())
                or not np.array_equal(control_selection["source_ijk_N256"], ijk.tolist())):
            raise ValueError("control NPZ arrays disagree with the score-blind selection record")
        with np.load(SOURCE_PATH, allow_pickle=False) as f:
            coarse_source = {key: f[key].copy() for key in f.files}
        if coarse_source["positions"].shape != (128**3, 3):
            raise ValueError("the pinned N128 source geometry changed")
        parent = ijk // 2
        parent_flat = (parent[:, 0] * 128 + parent[:, 1]) * 128 + parent[:, 2]
        expected_parent_angular = coarse_source["angular"][:, parent_flat].T
        if not np.array_equal(controls["parent_angular"], expected_parent_angular):
            raise ValueError("precomputed cell-constant angular values do not match source geometry")
        with np.load(FIELD_PATH, allow_pickle=False) as f:
            rho_node = jnp.asarray(f["rho"])
            velocity_node = jnp.asarray(f["mean_velocity_km_s"])
        with np.load(PARAMETER_PATH, allow_pickle=False) as f:
            tracer = jnp.asarray(f["tracer_white"])
        if rho_node.shape != (N, N, N) or velocity_node.shape != (3, N, N, N):
            raise ValueError("saved N256 field geometry changed")
        if tracer.shape != (9,):
            raise ValueError("saved conditional-map count nuisance vector changed")

        report["control_selection"] = {
            "rule": control_selection["selection_rule"],
            "count": int(len(flat_ids)),
            "labels": control_selection["labels"],
            "candidate_indices_not_grid_indices": candidate_indices.tolist(),
            "source_flat_ids_N256": flat_ids.tolist(),
            "source_ijk_N256": ijk.tolist(),
            "source_radius_cMpc_h": controls["radius"].tolist(),
            "angular_map_within_cell_range_max": controls["direct_contrast"].tolist(),
            "pinned_map_hashes": control_selection["angular_maps"]["sha256"],
        }

        split = np.load(SPLIT_PATH, allow_pickle=False)
        train_keys = split["train_keys"].copy()
        exposure, _ = build_population_exposure_masks(
            N_COUNT, split["heldout_flat_voxels"],
            split["train_window_excluded_keys"],
            split["heldout_window_excluded_keys"])
        del split
        report["training_key_geometry_loaded"] = True
        report["training_count_values_loaded"] = False
        report["exposure_integral"] = "training exposure only; includes empty exposed cells and excludes heldout/buffered cells"

        density, count_velocity = native_mass_momentum_to_count_cells(
            rho_node, velocity_node, BOX)
        rates = tracer_masses(density, tracer)
        geom = dict(
            observer=jnp.asarray(OBSERVER), box_size_cMpc_h=BOX,
            hubble_km_s_Mpc=74.6, little_h=0.746,
            radius_table_cMpc_h=jnp.asarray(coarse_source["radial_table"]),
            modulus_table_h=jnp.asarray(coarse_source["modulus_table"]),
            redshift_table=jnp.asarray(coarse_source["redshift_table"]),
            grid_size=N_COUNT,
        )
        geom.update(tracer_geometry(tracer, geom))
        offsets, volume_weights = source_volume_rule()
        current_angles = controls["parent_angular"]
        direct_angles = controls["direct_angular"]
        source_velocity = jnp.moveaxis(count_velocity, 0, -1).reshape(-1, 3)
        velocity_controls = source_velocity[flat_ids]
        target_keys, target_key_distances, shifted = nearest_training_keys(
            train_keys, controls["positions"],
            np.asarray(velocity_controls))

        exposure_jax = jnp.asarray(exposure.reshape(POPULATIONS, NVOXEL))
        target_keys_jax = jnp.asarray(target_keys)
        offsets_jax = jnp.asarray(offsets)
        weights_jax = jnp.asarray(volume_weights)
        source_rates = rates.reshape(5, -1)
        source_positions = jnp.asarray(controls["positions"])
        rates_controls = source_rates[:, flat_ids] * SOURCE_VOLUME_FACTOR

        def patch_measure(position, velocity, intrinsic, coarse_angle, node_angle,
                          key_vector):
            def accumulate(carry, item):
                constant_sum, pixel_sum = carry
                offset, qweight, pixel_angle = item
                node_position = (position + offset[None, :]) % BOX
                node_mass = intrinsic * qweight
                constant = predict_shell_cdf_intensity(
                    node_position, velocity, node_mass, coarse_angle,
                    **geom, order=4, segments=8, deposition="tsc")
                pixel = predict_shell_cdf_intensity(
                    node_position, velocity, node_mass, pixel_angle,
                    **geom, order=4, segments=8, deposition="tsc")
                cgrid = constant.reshape(POPULATIONS, NVOXEL)
                pgrid = pixel.reshape(POPULATIONS, NVOXEL)
                cexpected = jnp.sum(cgrid * exposure_jax, axis=1)
                pexpected = jnp.sum(pgrid * exposure_jax, axis=1)
                ckeys = jnp.take(constant.reshape(-1), key_vector)
                pkeys = jnp.take(pixel.reshape(-1), key_vector)
                return ((constant_sum[0] + cexpected, constant_sum[1] + ckeys),
                        (pixel_sum[0] + pexpected, pixel_sum[1] + pkeys)), None

            initial = ((jnp.zeros(POPULATIONS), jnp.zeros(POPULATIONS)),
                       (jnp.zeros(POPULATIONS), jnp.zeros(POPULATIONS)))
            (constant, pixel), _ = jax.lax.scan(
                accumulate, initial,
                (offsets_jax, weights_jax, jnp.moveaxis(node_angle, 0, 0)))
            return constant, pixel

        measure = jax.jit(patch_measure)
        controls_out = []
        for i, cell_id in enumerate(flat_ids):
            coarse_angle = jnp.asarray(current_angles[i, :, None])
            node_angle = jnp.asarray(direct_angles[i].transpose(1, 0)[:, :, None])
            constant, pixel = measure(
                source_positions[i], velocity_controls[i:i+1],
                rates_controls[:, i:i+1], coarse_angle, node_angle,
                target_keys_jax[i])
            (expected0, keys0), (expected1, keys1) = map(
                lambda pair: tuple(np.asarray(x) for x in pair), (constant, pixel))
            controls_out.append(dict(
                label=control_selection["labels"][i],
                source_flat_id_N256=int(cell_id),
                source_ijk_N256=controls["ijk"][i].tolist(),
                source_radius_cMpc_h=float(controls["radius"][i]),
                coherent_rsd_center_cMpc_h=shifted[i].tolist(),
                source_to_nearest_training_key_distance_cMpc_h=target_key_distances[i].tolist(),
                nearest_training_keys=target_keys[i].tolist(),
                expected_count_by_population_cell_constant=expected0.tolist(),
                expected_count_by_population_pixel_aware=expected1.tolist(),
                expected_count_delta_by_population=(expected1-expected0).tolist(),
                nearest_key_intensity_cell_constant=keys0.tolist(),
                nearest_key_intensity_pixel_aware=keys1.tolist(),
                nearest_key_intensity_delta=(keys1-keys0).tolist(),
            ))
        report["operator_contract"] = {
            "active_count_path": "N256 GL2 source-volume nodes; current per-source angular values are the pinned N128 eight-sightline cell averages replicated to N256; same coherent radial RSD, order-4/8 LOS kernel, K/LF transfer, TSC deposit, and v6 training exposure mask",
            "reference": "same fixed source positions/velocities/rates/nuisance and GL2 weights, but direct pinned NSIDE512 RING map lookup at each N256 source-volume node",
            "conditional_FP_path": "the raw FP numerator and denominator both apply the same pinned angular map as a source weight through raw_field_logpdf/streaming_raw_mark; this count-only local comparison does not test that normalized mark ratio",
            "count_occurrence": "not scored; expected-rate contributions and selected training-key intensities are reported once, with no extra count or f_n factor",
            "heldout": "only the geometric heldout mask is used to construct the training exposure; no heldout count or mark values loaded",
        }
        report["source_patch_results"] = controls_out
        totals0 = np.sum([x["expected_count_by_population_cell_constant"] for x in controls_out], axis=0)
        totals1 = np.sum([x["expected_count_by_population_pixel_aware"] for x in controls_out], axis=0)
        report["selected_patch_training_expected_count_sum"] = {
            "cell_constant_by_population": totals0.tolist(),
            "pixel_aware_by_population": totals1.tolist(),
            "delta_by_population": (totals1-totals0).tolist(),
            "relative_delta_by_population": np.divide(
                totals1-totals0, totals0, out=np.zeros_like(totals0), where=totals0 != 0).tolist(),
        }
        report["classification"] = "LOCAL_PIXEL_AWARE_ANGULAR_OPERATOR_COMPARISON_NOT_LIKELIHOOD"
        report["status"] = "COMPLETE_FIXED_STATE_LOCAL_OPERATOR_COMPARISON"
        report["Q_GOAL"] = "tests one active observation-operator approximation relevant to the same CF4-conditioned z=0 field; not a z=0 posterior or LG result"
        report["Q_LEAN"] = "12 geometry-selected source cells, fixed saved state, no PM replay, optimization, heldout outcomes, global resolution ladder, or simulation"
        report["R2_complete"] = False
        save()
        print(json.dumps({key: report[key] for key in (
            "status", "selected_patch_training_expected_count_sum", "Q_GOAL", "Q_LEAN")},
            allow_nan=False), flush=True)
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
