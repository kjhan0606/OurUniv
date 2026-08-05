#!/usr/bin/env python3
"""Exact band-limited Gaussian conditioning for Local-Group peak proposals.

The configured long-wave Fourier coefficients remain numerically unchanged.
The remaining white-noise degrees of freedom receive an explicitly labelled
Gaussian likelihood on translated, smoothed linear-density values. Matheron's
rule then produces posterior samples rather than a hand-edited MAP field. When
the frozen mesh is smaller than the CF4 parent mesh, every parent-resolution
projection must be revalidated because some CF4-resolved modes may change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import fft as spfft


def free_rfft_mask(n: int, coarse_n: int) -> np.ndarray:
    """Coefficients not overwritten by ``cf4_make_ic.embed_ic``."""
    if n <= coarse_n or n % 2 or coarse_n % 2:
        raise ValueError("n and coarse_n must be even, with n > coarse_n")
    free = np.ones((n, n, n // 2 + 1), dtype=bool)
    half = coarse_n // 2
    mapped = np.array([i if i < half else n - (coarse_n - i)
                       for i in range(coarse_n)])
    keep = np.arange(coarse_n) != half
    idx = mapped[keep]
    free[np.ix_(idx, idx, np.arange(half))] = False
    return free


def condition_translated_constraints(
    base: np.ndarray,
    density_filter: np.ndarray,
    free_mask: np.ndarray,
    points: np.ndarray,
    targets: np.ndarray,
    sigma: float | np.ndarray,
    noise_seed: int,
    prepared: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """Matheron sample for translated smoothed-density constraints.

    ``density_filter`` maps the white field to the desired smoothed linear
    density through ``irfftn(rfftn(s) * density_filter)``.  Points are integer
    periodic grid indices.  Only the subspace selected by ``free_mask`` is
    corrected.
    """
    # N=576 contains 191 million cells.  Keep the volume-sized work arrays in
    # single precision (SciPy then uses complex64 FFTs); the 14x14 likelihood
    # system below remains double precision.
    base = np.asarray(base, dtype=np.float32)
    n = base.shape[0]
    if base.shape != (n, n, n):
        raise ValueError("base must be cubic")
    expected = (n, n, n // 2 + 1)
    filt = np.asarray(density_filter, dtype=np.float32)
    if filt.shape != expected or free_mask.shape != expected:
        raise ValueError("filter/mask shape mismatch")
    points = np.mod(np.asarray(points, dtype=np.int64), n)
    targets = np.asarray(targets, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(targets):
        raise ValueError("points must be (m,3) and match targets")
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), targets.shape)
    if np.any(sig <= 0):
        raise ValueError("all likelihood sigmas must be positive")

    # One origin template and its free-subspace projection suffice because all
    # constraints are translations of the same isotropic smoothing kernel.
    axes = (0, 1, 2)
    if prepared is None:
        template = spfft.irfftn(filt, s=base.shape, axes=axes, workers=-1)
        template_k = spfft.rfftn(template, axes=axes, workers=-1)
        template_k[~free_mask] = 0.0
        template_free = spfft.irfftn(
            template_k, s=base.shape, axes=axes, workers=-1)
        covariance_grid = spfft.irfftn(
            np.abs(template_k) ** 2, s=base.shape, axes=axes, workers=-1)
    else:
        template_free, covariance_grid = prepared
        template_free = np.asarray(template_free, dtype=np.float32)
        covariance_grid = np.asarray(covariance_grid, dtype=np.float32)

    m = len(points)
    covariance = np.empty((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            displacement = tuple(np.mod(points[i] - points[j], n))
            covariance[i, j] = covariance_grid[displacement]

    base_k = spfft.rfftn(base, axes=axes, workers=-1)
    smooth_base = spfft.irfftn(
        base_k * filt,
        s=base.shape, axes=axes, workers=-1)
    predicted = smooth_base[tuple(points.T)]
    rng = np.random.default_rng(noise_seed)
    mock_noise = rng.normal(0.0, sig)
    system = covariance + np.diag(sig**2)
    weights = np.linalg.solve(system, targets - predicted - mock_noise)

    correction = np.zeros_like(base)
    for weight, point in zip(weights, points):
        correction += np.float32(weight) * np.roll(
            template_free, shift=tuple(point), axis=(0, 1, 2))
    conditioned = (base + correction).astype(np.float32, copy=False)
    conditioned_k = spfft.rfftn(conditioned, axes=axes, workers=-1)
    achieved = spfft.irfftn(
        conditioned_k * filt,
        s=base.shape, axes=axes, workers=-1)[tuple(points.T)]
    correction_k = spfft.rfftn(correction, axes=axes, workers=-1)
    frozen_difference = conditioned_k[~free_mask] - base_k[~free_mask]
    frozen_reference_rms = float(np.sqrt(
        np.mean(np.abs(base_k[~free_mask]) ** 2)))
    frozen_difference_rms = float(np.sqrt(
        np.mean(np.abs(frozen_difference) ** 2)))
    frozen_relative_rms = frozen_difference_rms / max(frozen_reference_rms, 1e-30)
    if frozen_relative_rms > 2e-6:
        raise RuntimeError(
            "conditioning changed the frozen coarse modes: "
            f"relative RMS={frozen_relative_rms:.3e}")
    meta = {
        "predicted_before": predicted.tolist(),
        "achieved_after": achieved.tolist(),
        "targets": targets.tolist(),
        "sigma": sig.tolist(),
        "mock_noise": mock_noise.tolist(),
        "weights": weights.tolist(),
        "constraint_covariance": covariance.tolist(),
        "correction_rms": float(correction.std()),
        "frozen_mode_correction_max": float(
            np.max(np.abs(correction_k[~free_mask]), initial=0.0)),
        "frozen_mode_final_difference_max": float(
            np.max(np.abs(frozen_difference), initial=0.0)),
        "frozen_mode_final_difference_rms": frozen_difference_rms,
        "frozen_mode_reference_rms": frozen_reference_rms,
        "frozen_mode_final_relative_rms": frozen_relative_rms,
    }
    return conditioned, meta


def prepare_translated_conditioner(
    density_filter: np.ndarray, free_mask: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the free template and its translation covariance grid."""
    density_filter = np.asarray(density_filter, dtype=np.float32)
    template = spfft.irfftn(density_filter, s=shape, workers=-1)
    template_k = spfft.rfftn(template, workers=-1)
    template_k[~free_mask] = 0.0
    template_free = spfft.irfftn(template_k, s=shape, workers=-1)
    covariance_grid = spfft.irfftn(
        np.abs(template_k) ** 2, s=shape, workers=-1)
    return (
        template_free.astype(np.float32, copy=False),
        covariance_grid.astype(np.float32, copy=False),
    )


def linear_density_filter(
    n: int, box_size: float, radius: float, cosmology: dict
) -> np.ndarray:
    """White-noise to Gaussian-smoothed z=0 linear-density multiplier."""
    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann
    from pmwd.boltzmann import linear_power

    dx = box_size / n
    conf = Configuration(
        ptcl_spacing=float(dx), ptcl_grid_shape=(n,) * 3, mesh_shape=1,
        cosmo_dtype=jnp.float64, float_dtype=jnp.float32)
    cosmo = boltzmann(SimpleLCDM(
        conf, Omega_m=float(cosmology["Om"]), Omega_b=float(cosmology["Ob"]),
        h=float(cosmology["h"]), A_s_1e9=float(cosmology["A_s_1e9"]),
        n_s=float(cosmology["ns"])), conf)
    kmax = np.sqrt(3.0) * np.pi / dx
    sample_k = np.linspace(0.0, kmax, 32769, dtype=np.float64)
    sample_p = np.array(linear_power(
        jnp.asarray(sample_k), 1.0, cosmo, conf), dtype=np.float64, copy=True)
    sample_p[~np.isfinite(sample_p)] = 0.0
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    ky2 = kx[:, None] ** 2
    kz2 = kz[None, :] ** 2
    result = np.empty((n, n, n // 2 + 1), dtype=np.float32)
    for i, value in enumerate(kx):
        k2 = value**2 + ky2 + kz2
        power = np.interp(np.sqrt(k2), sample_k, sample_p)
        result[i] = (
            np.sqrt(np.maximum(power, 0.0) / dx**3)
            * np.exp(-0.5 * k2 * radius**2)
        ).astype(np.float32)
    result[0, 0, 0] = 0.0
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def two_peak_points(
    n: int,
    midpoint: np.ndarray,
    axis: np.ndarray,
    separation_cells: int,
    shell_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two centres plus six axial shell probes around each centre."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    offset = np.rint(0.5 * separation_cells * axis).astype(np.int64)
    centres = np.vstack((midpoint - offset, midpoint + offset))
    unit = np.eye(3, dtype=np.int64) * int(shell_cells)
    points, kinds = [], []
    for centre in centres:
        points.append(centre); kinds.append(1)
        for delta in np.vstack((unit, -unit)):
            points.append(centre + delta); kinds.append(0)
    return np.mod(np.asarray(points), n), np.asarray(kinds, dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    n = int(config["mesh_size"])
    box = float(config["box_size_mpc_h"])
    coarse_path = Path(config["parent_field"])
    outdir = args.outdir or Path(config["storage"]["proposal_directory"])
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(coarse_path, allow_pickle=False) as data:
        coarse = data["s_out"].astype(np.float32)
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    peak = config["peak_constraints"]
    dx = box / n
    filt = linear_density_filter(
        n, box, float(peak["gaussian_radius_mpc_h"]), cosmology)
    frozen_n = int(config.get("frozen_mode_mesh_size", coarse.shape[0]))
    if frozen_n > coarse.shape[0]:
        raise ValueError("frozen_mode_mesh_size cannot exceed the parent mesh")
    free = free_rfft_mask(n, frozen_n)
    prepared = prepare_translated_conditioner(filt, free, (n, n, n))
    separation_cells = int(round(peak["protohalo_separation_mpc_h"] / dx))
    shell_cells = int(round(peak["shell_radius_mpc_h"] / dx))
    midpoint_offset = np.asarray(
        peak.get("protohalo_midpoint_offset_mpc_h", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if midpoint_offset.shape != (3,):
        raise ValueError("protohalo_midpoint_offset_mpc_h must have three values")
    midpoint = np.full(3, n // 2, dtype=np.int64) + np.rint(
        midpoint_offset / dx).astype(np.int64)

    from cf4_make_ic import embed_ic, fourier_resample_white_field

    projection_n = config.get("parent_projection_mesh_size")
    projection_n = int(projection_n) if projection_n is not None else None
    projection_dir = None
    projection_entries = []
    if projection_n is not None:
        if projection_n > n:
            raise ValueError("parent projection mesh cannot exceed proposal mesh")
        projection_dir = Path(config["storage"]["parent_projection_directory"])
        projection_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    triplets = zip(
        config["proposal_seeds"], config["geometry_seeds"],
        config["likelihood_noise_seeds"])
    for number, (seed, geometry_seed, noise_seed) in enumerate(triplets, 1):
        axis = np.random.default_rng(geometry_seed).normal(size=3)
        points, kinds = two_peak_points(
            n, midpoint, axis, separation_cells, shell_cells)
        targets = np.where(
            kinds == 1, peak["centre_target_delta_linear"],
            peak["six_shell_target_delta_linear"])
        base = embed_ic(coarse, n, int(seed))
        conditioned, metadata = condition_translated_constraints(
            base, filt, free, points, targets,
            float(peak["likelihood_sigma_delta"]), int(noise_seed),
            prepared=prepared)
        path = outdir / f"lg_peak_p{config['parent_seed']}_s{seed}.npz"
        record = {
            "parent_seed": int(config["parent_seed"]),
            "proposal_seed": int(seed),
            "geometry_seed": int(geometry_seed),
            "likelihood_noise_seed": int(noise_seed),
            "axis": (axis / np.linalg.norm(axis)).tolist(),
            "protohalo_midpoint_grid": np.mod(midpoint, n).tolist(),
            "protohalo_midpoint_offset_mpc_h": midpoint_offset.tolist(),
            "points_grid": points.tolist(),
            "kinds": kinds.tolist(),
            "field": str(path),
            "field_mean": float(conditioned.mean()),
            "field_std": float(conditioned.std()),
            "conditioning": metadata,
        }
        np.savez(
            path, s_conditioned=conditioned, N=np.int32(n), L=np.float64(box),
            parent_seed=np.int32(config["parent_seed"]),
            proposal_seed=np.int32(seed),
            metadata_json=np.array(json.dumps(record, sort_keys=True)))
        record["field_sha256"] = sha256_file(path)
        if projection_n is not None:
            projected = fourier_resample_white_field(conditioned, projection_n)
            projected_path = projection_dir / (
                f"lg_peak_parent_p{config['parent_seed']}_s{seed}.npz")
            np.savez(
                projected_path,
                s_out=projected,
                sample_seed=np.int64(seed),
                source_parent_seed=np.int64(config["parent_seed"]),
                N=np.int64(projection_n),
                spacing=np.float64(box / projection_n),
                L=np.float64(box),
                hh=np.float64(cosmology["h"]),
                Om=np.float64(cosmology["Om"]),
                Ob=np.float64(cosmology["Ob"]),
                A_s_1e9=np.float64(cosmology["A_s_1e9"]),
                ns=np.float64(cosmology["ns"]),
            )
            projection_record = {
                "parent_seed": int(config["parent_seed"]),
                "proposal_seed": int(seed),
                "field": str(projected_path.resolve()),
                "field_sha256": sha256_file(projected_path),
                "field_mean": float(projected.mean()),
                "field_std": float(projected.std()),
            }
            record["parent_projection"] = projection_record
            projection_entries.append(projection_record)
            del projected
        entries.append(record)
        print(
            f"[LG-CR] {number}/{len(config['proposal_seeds'])} seed={seed} "
            f"std={record['field_std']:.5f} "
            f"corr_rms={metadata['correction_rms']:.4f} "
            f"frozen_max={metadata['frozen_mode_correction_max']:.2e}",
            flush=True)

    manifest = {
        "schema": "ouruniv-lg-peak-proposals-v1",
        "status": "complete",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "parent_field": str(coarse_path.resolve()),
        "parent_field_sha256": sha256_file(coarse_path),
        "mesh_size": n,
        "frozen_mode_mesh_size": frozen_n,
        "box_size_mpc_h": box,
        "cosmology": cosmology,
        "entries": entries,
    }
    manifest_path = outdir / "lg_peak_proposals_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[LG-CR] wrote {manifest_path}", flush=True)
    if projection_n is not None:
        projection_manifest = {
            "schema": "ouruniv-lg-peak-parent-projections-v1",
            "status": "all_data",
            "method": "Fourier projection of explicit LG-conditioned proposals",
            "source_proposal_manifest": str(manifest_path.resolve()),
            "source_proposal_manifest_sha256": sha256_file(manifest_path),
            "configuration": {
                "N": projection_n,
                "box_size": box,
                "Om": cosmology["Om"],
                "Ob": cosmology["Ob"],
                "h": cosmology["h"],
                "A_s_1e9": cosmology["A_s_1e9"],
                "ns": cosmology["ns"],
                "sample_seeds": [row["proposal_seed"] for row in projection_entries],
            },
            "outputs": [row["field"] for row in projection_entries],
            "entries": projection_entries,
        }
        projection_manifest_path = projection_dir / "parent_projection_manifest.json"
        projection_manifest_path.write_text(
            json.dumps(projection_manifest, indent=2) + "\n")
        print(f"[LG-CR] wrote {projection_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
