#!/usr/bin/env python
"""Forward and score the preregistered CF4 parent ensemble at z=0.

Thresholds are read from config/p1_targets_v1.json.  This program does not tune
them, use the held-out velocity test, or search for a Local-Group analogue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RHO_CRIT = 2.775e11

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sg_xyz(sgl_deg: float, sgb_deg: float, distance: float) -> np.ndarray:
    lon, lat = np.radians([sgl_deg, sgb_deg])
    return distance * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


class DensityScorer:
    """Measurements on a periodic, observer-centred Cartesian density mesh."""

    def __init__(
        self,
        delta: np.ndarray,
        spacing: float,
        shell_half_width: float,
        observer_offset: np.ndarray | None = None,
    ):
        self.delta = np.asarray(delta, np.float32)
        self.nmesh = delta.shape[0]
        self.spacing = float(spacing)
        self.box_size = self.nmesh * self.spacing
        self.centre = self.box_size / 2.0
        self.observer_offset = (
            np.zeros(3, dtype=np.float64)
            if observer_offset is None else np.asarray(observer_offset, np.float64)
        )
        if self.observer_offset.shape != (3,):
            raise ValueError("observer_offset must contain three Cartesian coordinates")
        self.observer_position = self.centre + self.observer_offset
        self.shell_half_width = float(shell_half_width)
        axes = []
        for coordinate in self.observer_position:
            axis = (np.arange(self.nmesh, dtype=np.float32) + 0.5) * spacing
            axis = (axis - coordinate + self.box_size / 2) % self.box_size
            axes.append(axis - self.box_size / 2)
        self.observer_radius = np.sqrt(
            axes[0][:, None, None] ** 2
            + axes[1][None, :, None] ** 2
            + axes[2][None, None, :] ** 2
        )

    def value(self, offset: np.ndarray) -> float:
        index = np.floor(
            (offset + self.observer_position) / self.spacing).astype(int)
        return float(self.delta[tuple(index % self.nmesh)])

    def shell_percentile(self, offset: np.ndarray, value: float) -> float:
        radius = float(np.linalg.norm(offset))
        shell = np.abs(self.observer_radius - radius) <= self.shell_half_width
        return float(100.0 * np.mean(self.delta[shell] < value))

    def _local_cube(self, offset: np.ndarray, radius: float):
        box_pos = offset + self.observer_position
        lower = np.floor((box_pos - radius) / self.spacing).astype(int)
        upper = np.floor((box_pos + radius) / self.spacing).astype(int) + 1
        raw_axes = [np.arange(a, b) for a, b in zip(lower, upper)]
        mesh_axes = [axis % self.nmesh for axis in raw_axes]
        coord_axes = [
            (axis + 0.5) * self.spacing - self.centre for axis in raw_axes
        ]
        displacements = [
            ((coord - offset[i] + self.box_size / 2) % self.box_size)
            - self.box_size / 2
            for i, coord in enumerate(coord_axes)
        ]
        radius2 = (
            displacements[0][:, None, None] ** 2
            + displacements[1][None, :, None] ** 2
            + displacements[2][None, None, :] ** 2
        )
        cube = self.delta[np.ix_(*mesh_axes)]
        return cube, radius2 <= radius**2, raw_axes

    def sphere_mean(self, offset: np.ndarray, radius: float) -> float:
        cube, mask, _ = self._local_cube(offset, radius)
        return float(cube[mask].mean())

    def peak(self, offset: np.ndarray, radius: float) -> tuple[float, np.ndarray, float]:
        cube, mask, raw_axes = self._local_cube(offset, radius)
        flat_index = int(np.argmax(np.where(mask, cube, -np.inf)))
        local_index = np.array(np.unravel_index(flat_index, cube.shape))
        peak = np.array(
            [
                (raw_axes[i][local_index[i]] + 0.5) * self.spacing
                - self.observer_position[i]
                for i in range(3)
            ]
        )
        peak = ((peak + self.box_size / 2) % self.box_size) - self.box_size / 2
        delta_peak = float(cube[tuple(local_index)])
        displacement = ((peak - offset + self.box_size / 2) % self.box_size)
        displacement -= self.box_size / 2
        return delta_peak, peak, float(np.linalg.norm(displacement))


def cluster_metrics(
    name: str, spec: dict, scorer: DensityScorer, h: float, hard: bool
) -> dict:
    target = sg_xyz(spec["sgl_deg"], spec["sgb_deg"], spec["distance_mpc"] * h)
    target_delta = scorer.value(target)
    target_percentile = scorer.shell_percentile(target, target_delta)
    peak_delta, peak_position, peak_separation = scorer.peak(
        target, spec["search_radius_mpc_h"]
    )
    peak_percentile = scorer.shell_percentile(target, peak_delta)
    row = {
        "name": name,
        "target_offset_mpc_h": target.tolist(),
        "target_delta": target_delta,
        "target_shell_percentile": target_percentile,
        "peak_delta": peak_delta,
        "peak_shell_percentile": peak_percentile,
        "peak_offset_mpc_h": peak_position.tolist(),
        "peak_separation_mpc_h": peak_separation,
    }
    if hard:
        row["pass"] = bool(
            (not spec["require_positive_target_delta"] or target_delta > 0.0)
            and target_percentile >= spec["minimum_target_percentile"]
            and peak_percentile >= spec["minimum_peak_percentile"]
            and peak_separation <= spec["search_radius_mpc_h"]
        )
    return row


def local_void_metrics(spec: dict, scorer: DensityScorer) -> dict:
    probes = []
    for name, xyz in spec["probes"].items():
        position = np.asarray(xyz, np.float64)
        mean_delta = scorer.sphere_mean(position, spec["probe_radius_mpc_h"])
        centre_delta = scorer.value(position)
        probes.append(
            {
                "name": name,
                "offset_mpc_h": position.tolist(),
                "mean_delta": mean_delta,
                "centre_delta": centre_delta,
                "centre_shell_percentile": scorer.shell_percentile(
                    position, centre_delta
                ),
                "underdense": bool(mean_delta < 0.0),
            }
        )
    n_underdense = sum(row["underdense"] for row in probes)
    probe_mean = float(np.mean([row["mean_delta"] for row in probes]))
    median_percentile = float(
        np.median([row["centre_shell_percentile"] for row in probes])
    )
    passed = (
        n_underdense >= spec["minimum_underdense_probes"]
        and (not spec["require_negative_probe_mean"] or probe_mean < 0.0)
        and median_percentile <= spec["maximum_median_shell_percentile"]
    )
    return {
        "probes": probes,
        "n_underdense": int(n_underdense),
        "probe_mean_delta": probe_mean,
        "median_centre_shell_percentile": median_percentile,
        "pass": bool(passed),
    }


def bootes_metrics(spec: dict, scorer: DensityScorer) -> dict:
    target = sg_xyz(spec["sgl_deg"], spec["sgb_deg"], spec["distance_mpc_h"])
    centre_delta = scorer.value(target)
    centre_percentile = scorer.shell_percentile(target, centre_delta)
    profile = {
        str(float(radius)): scorer.sphere_mean(target, float(radius))
        for radius in spec["profile_radii_mpc_h"]
    }
    negative_required = all(
        profile[str(float(radius))] < 0.0
        for radius in spec["require_negative_mean_at_radii_mpc_h"]
    )
    return {
        "target_offset_mpc_h": target.tolist(),
        "centre_delta": centre_delta,
        "centre_shell_percentile": centre_percentile,
        "mean_delta_profile": profile,
        "nearest_box_face_mpc_h": float(np.min(np.minimum(
            (scorer.observer_position + target) % scorer.box_size,
            scorer.box_size - (scorer.observer_position + target) % scorer.box_size,
        ))),
        "pass": bool(
            centre_percentile <= spec["maximum_center_shell_percentile"]
            and negative_required
        ),
    }


def observer_environment_metrics(spec: dict, scorer: DensityScorer,
                                 omega_m: float) -> dict:
    """Coarse Local-Volume mass proxy, independent of LG pair morphology."""
    rows = {}
    passed = True
    for radius_text, maximum_excess in spec["maximum_excess_mass_msun_h"].items():
        radius = float(radius_text)
        mean_delta = scorer.sphere_mean(np.zeros(3), radius)
        mean_mass = omega_m * RHO_CRIT * (4.0 * np.pi / 3.0) * radius**3
        excess_mass = mean_delta * mean_mass
        row_pass = excess_mass <= float(maximum_excess)
        if np.isclose(radius, spec["local_sheet_radius_mpc_h"]):
            row_pass = row_pass and mean_delta >= spec["minimum_local_sheet_mean_delta"]
        rows[radius_text] = {
            "radius_mpc_h": radius,
            "mean_delta": mean_delta,
            "cosmic_mean_mass_msun_h": mean_mass,
            "excess_mass_msun_h": excess_mass,
            "maximum_excess_mass_msun_h": float(maximum_excess),
            "pass": bool(row_pass),
        }
        passed = passed and row_pass
    return {
        "spheres": rows,
        "minimum_local_sheet_mean_delta": spec["minimum_local_sheet_mean_delta"],
        "pass": bool(passed),
        "status": "coarse smoothed-density proxy; definitive exclusion uses particle HOP",
    }


def score_member(
    delta: np.ndarray,
    spacing: float,
    config: dict,
    omega_m: float = 0.31,
    observer_offset: np.ndarray | None = None,
) -> dict:
    scorer = DensityScorer(
        delta, spacing, config["shell_half_width_mpc_h"], observer_offset)
    h = float(config["cosmology_h"])
    clusters = {
        name: cluster_metrics(name, spec, scorer, h, hard=True)
        for name, spec in config["clusters"].items()
    }
    secondary = {
        name: cluster_metrics(name, spec, scorer, h, hard=False)
        for name, spec in config["secondary_cluster_anchors"].items()
    }
    local_void = local_void_metrics(config["local_void"], scorer)
    bootes_void = bootes_metrics(config["bootes_void"], scorer)
    observer = (
        observer_environment_metrics(config["observer_environment"], scorer, omega_m)
        if "observer_environment" in config else None
    )
    gates = {
        **{name: row["pass"] for name, row in clusters.items()},
        "LocalVoid": local_void["pass"],
        "BootesVoid": bootes_void["pass"],
    }
    if observer is not None:
        gates["ObserverEnvironment"] = observer["pass"]
    passed = all(gates[name] for name in config["hard_gate_policy"]["pass_requires"])
    return {
        "clusters": clusters,
        "secondary_cluster_anchors": secondary,
        "local_void": local_void,
        "bootes_void": bootes_void,
        "observer_environment": observer,
        "gates": gates,
        "n_gates_passed": int(sum(gates.values())),
        "pass": bool(passed),
    }


def markdown_report(result: dict) -> str:
    lines = [
        "# P1 parent-ensemble result",
        "",
        f"- Config SHA-256: `{result['config_sha256']}`",
        f"- Members: {len(result['members'])}",
        f"- Full P1 passes: {len(result['passing_seeds'])}",
        f"- Passing seeds: {result['passing_seeds'] or 'none'}",
        "",
        "| seed | Virgo | Coma | Local Void | Boötes | Observer | gates | P1 |",
        "|---:|:---:|:---:|:---:|:---:|:---:|---:|:---:|",
    ]
    label = lambda value: "PASS" if value else "fail"
    for row in result["members"]:
        gates = row["gates"]
        lines.append(
            f"| {row['seed']} | {label(gates['Virgo'])} | "
            f"{label(gates['Coma'])} | {label(gates['LocalVoid'])} | "
            f"{label(gates['BootesVoid'])} | "
            f"{label(gates.get('ObserverEnvironment', True))} | "
            f"{row['n_gates_passed']}/{len(gates)} | "
            f"{label(row['pass'])} |"
        )
    lines += [
        "",
        "Thresholds were frozen before forwarding. A failure is not repaired by",
        "retuning them. Secondary cluster anchors are blind, non-gating validation.",
        "Halo masses, velocities, contamination, and Local-Group morphology are",
        "not P1 claims.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "recon/linear_cr/manifest_parent_v1_all.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/p1_targets_v1.json"
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "recon/linear_cr/p1_parent_v1"
    )
    parser.add_argument("--limit", type=int, default=0, help="development only")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    if not config.get("frozen_before_forwarding"):
        raise RuntimeError("P1 config is not frozen")
    inputs = manifest["outputs"][: args.limit or None]
    if not inputs:
        raise RuntimeError("parent manifest has no outputs")

    import jax
    import jax.numpy as jnp
    from scipy.ndimage import gaussian_filter
    from mock_pipeline import make_forward

    with np.load(inputs[0]) as first:
        nmesh = int(first["N"])
        spacing = float(first["spacing"])
        box_size = float(first["L"])
    expected = (192, 2.0, 384.0)
    if (nmesh, spacing, box_size) != expected:
        raise RuntimeError(
            f"P1 v1 expects N,spacing,L={expected}; got {nmesh, spacing, box_size}"
        )

    config_sha = file_hash(args.config)
    print(f"[P1] JAX {jax.__version__}; device={jax.devices()[0]}", flush=True)
    print(f"[P1] frozen config SHA-256={config_sha}", flush=True)
    print(f"[P1] identical PM forward for {len(inputs)} members", flush=True)
    model = manifest["configuration"]
    cosmology = {
        "Om": model["Om"],
        "Ob": model["Ob"],
        "h": model["h"],
        "A_s_1e9": model["A_s_1e9"],
        "ns": model["ns"],
    }
    if not np.isclose(cosmology["h"], config["cosmology_h"]):
        raise RuntimeError("P1 target conversion h differs from the IC cosmology")
    _, _, forward = make_forward(
        nmesh, spacing, jnp.float32, return_dens=True, cosmology=cosmology
    )

    members = []
    for index, input_name in enumerate(inputs, 1):
        input_path = Path(input_name)
        with np.load(input_path) as data:
            initial_field = data["s_out"].astype(np.float32)
            seed = int(data["sample_seed"])
        started = time.time()
        density, _ = forward(jnp.asarray(initial_field))
        density.block_until_ready()
        smoothed = gaussian_filter(
            np.asarray(density, np.float32),
            config["density_smoothing_mpc_h"] / spacing,
            mode="wrap",
        )
        delta = smoothed / np.mean(smoothed, dtype=np.float64) - 1.0
        row = {
            "seed": seed,
            "input": str(input_path.resolve()),
            "seconds": time.time() - started,
            **score_member(delta, spacing, config, omega_m=cosmology["Om"]),
        }
        members.append(row)
        gates = " ".join(
            f"{name}={'Y' if value else 'n'}" for name, value in row["gates"].items()
        )
        print(
            f"[P1] {index:02d}/{len(inputs)} seed={seed} {gates} "
            f"=> {'PASS' if row['pass'] else 'fail'} ({row['seconds']:.1f}s)",
            flush=True,
        )

    result = {
        "schema": (
            "cf4-p1-result-v2-observer"
            if "observer_environment" in config else "cf4-p1-result-v1"
        ),
        "status": "complete" if not args.limit else "development_subset",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_hash(args.manifest),
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "N": nmesh,
        "spacing_mpc_h": spacing,
        "box_size_mpc_h": box_size,
        "cosmology": cosmology,
        "members": members,
        "passing_seeds": [row["seed"] for row in members if row["pass"]],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    result_path = args.outdir / "p1_result.json"
    report_path = args.outdir / "P1_REPORT.md"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(markdown_report(result))
    print(f"[P1] wrote {result_path}", flush=True)
    print(f"[P1] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
