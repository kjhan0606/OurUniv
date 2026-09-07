"""Corrected magnitude convention and independent mark-calibration datum.

Historical builders are unchanged. All numerical preparation runs under Slurm.
Survival probabilities are conditional on luminosity population and radius;
environment-dependent selection is not claimed to have been eliminated.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_twompp_disjoint_tracer_pilot_v3 import base
from cf4_twompp_joint_information_budget_pilot_v1 import _cosmology_distance_table, schechter_fraction
from cf4_datum_bearing_z0_twompp_datum_builder_v1 import (
    hash_holdout_mask, integer_count_grids, read_excluded_recnos,
)


def magnitude_h(physical_magnitude, h):
    if not np.isfinite(h) or h <= 0:
        raise ValueError("positive finite h required")
    return np.asarray(physical_magnitude) - 5*np.log10(h)


def three_way_split(recnos, salt):
    """Preserve old <1/5 heldout, reserve [1/5,2/5) for calibration."""
    held = hash_holdout_mask(recnos, salt, 1, 5)
    first_two = hash_holdout_mask(recnos, salt, 2, 5)
    return ~first_two, first_two & ~held, held


def mark_counts(population, shell, calibration, survives, shells=6):
    joint = np.asarray(population)*shells + np.asarray(shell)
    def counts(mask):
        return np.bincount(joint[mask], minlength=6*shells).reshape(6, shells)
    return counts(calibration & survives), counts(calibration & ~survives)


def shell_exposure(tracer, inputs, shell_edges, n, box, order):
    import healpy as hp
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    design, cosmology = tracer["tracer_design"], tracer["cosmology"]
    maps = [base.load_completeness_map(inputs[k]["path"], 512)
            for k in ("completeness_11_5", "completeness_12_5")]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    axis = (np.arange(n)+.5)*box/n-box/2
    output = np.zeros((6, len(shell_edges)-1, n, n, n))
    flat = output.reshape(6, len(shell_edges)-1, -1)
    edges = design["absolute_K_edges"]
    for ix, ox in enumerate(nodes):
        for iy, oy in enumerate(nodes):
            for iz, oz in enumerate(nodes):
                xyz = np.array(np.meshgrid(axis+ox*box/n/2, axis+oy*box/n/2,
                                           axis+oz*box/n/2, indexing="ij"))
                radius = np.linalg.norm(xyz, axis=0)
                active = (radius >= shell_edges[0]) & (radius <= shell_edges[-1])
                ids = np.flatnonzero(active)
                x, y, z = xyz[:, active]
                r = radius[active]
                sg = SkyCoord(sgl=np.arctan2(y,x)*u.rad, sgb=np.arcsin(z/r)*u.rad,
                              frame="supergalactic")
                pixels = hp.ang2pix(512, .5*np.pi-sg.icrs.dec.rad,
                                   np.mod(sg.icrs.ra.rad, 2*np.pi), nest=False)
                # M_h convention: distance modulus uses luminosity distance in Mpc/h.
                dl_h = _cosmology_distance_table(r, cosmology)*cosmology["h"]
                shell = np.clip(np.searchsorted(shell_edges, r, side="right")-1,
                                0, len(shell_edges)-2)
                weight = weights[ix]*weights[iy]*weights[iz]/8
                for p in range(6):
                    apparent, absolute = divmod(p, 3)
                    fraction = schechter_fraction(dl_h, None if apparent == 0 else 11.5,
                        11.5 if apparent == 0 else 12.5, edges[absolute], edges[absolute+1],
                        -23.28, -.94)
                    flat[p, shell, ids] += weight*maps[apparent][pixels]*fraction
        print(f"corrected selection quadrature {ix+1}/{order}", flush=True)
    return output


def build(plan, root):
    import healpy as hp
    cfg = plan["selection_correction"]
    tracer = json.loads(Path(cfg["tracer_program"]).read_text())
    old = json.loads(Path(cfg["old_builder_program"]).read_text())
    inputs = tracer["inputs"]
    # Bind only the small, actually used scientific inputs; no storage probes.
    bindings = {k: inputs[k] for k in ("twompp_catalog", "cf4_twompp_crossmatch",
                                      "completeness_11_5", "completeness_12_5")}
    bindings["excluded_recnos"] = old["bindings"]["excluded_recnos"]
    for key, item in bindings.items():
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"changed scientific input: {key}")
    catalog = base.load_catalog(inputs["twompp_catalog"]["path"])
    distance, absolute_phys = base.distance_and_absolute_magnitude(
        catalog["Vcmb"], catalog["Ksmag"], tracer["cosmology"])
    absolute = magnitude_h(absolute_phys, tracer["cosmology"]["h"])
    eligible, _, apparent, ab = base.classify_disjoint_tracer(
        catalog, set(), distance, absolute, tracer["tracer_design"])
    excluded, _ = base.read_crossmatch_exclusions(inputs["cf4_twompp_crossmatch"]["path"], 17007)
    prior_excluded = read_excluded_recnos(old["bindings"]["excluded_recnos"]["path"], 319)
    # Apply the same published-map consistency rule to newly admitted magnitude rows.
    pixels = hp.ang2pix(512, np.deg2rad(90-catalog["DEC"]), np.deg2rad(catalog["RA"]), nest=False)
    completeness = np.zeros(len(distance))
    for a in (0, 1):
        c = base.load_completeness_map(inputs[f"completeness_{'11_5' if a == 0 else '12_5'}"]["path"], 512)
        completeness[apparent == a] = c[pixels[apparent == a]]
    mark = np.where(apparent == 0, catalog["c11_5"], catalog["c12_5"])
    metadata_ok = (completeness > 0) & np.isfinite(mark) & (np.abs(mark-completeness) <= .05)
    survives = (~np.isin(catalog["recno"], list(excluded)) & metadata_ok
                & ~np.isin(catalog["recno"], prior_excluded))
    recnos = catalog["recno"][eligible]
    population = (3*apparent+ab)[eligible]
    edges = np.asarray(cfg["radius_edges_cMpc_h"])
    shell = np.clip(np.searchsorted(edges, distance[eligible], side="right")-1, 0, len(edges)-2)
    train, calibration, held = three_way_split(recnos, old["split"]["salt"])
    survives = survives[eligible]
    yes, no = mark_counts(population, shell, calibration, survives, len(edges)-1)
    n, box = plan["grid"]["N"], plan["grid"]["box_cMpc_h"]
    directions = base.supergalactic_unit_vectors(catalog["RA"], catalog["DEC"])[eligible]
    position = distance[eligible, None]*directions+box/2
    cells = np.floor(position/(box/n)).astype(int)
    if np.any(cells < 0) or np.any(cells >= n):
        raise ValueError("count outside box")
    flat = np.ravel_multi_index(cells.T, (n,)*3)
    used = survives & ~calibration
    allcounts, counts, holdcounts = integer_count_grids(population[used], flat[used], held[used], 6, n)
    exposure_shells = shell_exposure(tracer, inputs, edges, n, box, cfg["quadrature_order"])
    exposure = exposure_shells.sum(axis=1)
    if np.any((allcounts > 0) & (exposure <= 0)):
        raise ValueError("corrected counts outside quadrature support")
    if np.any((counts.sum(axis=(1,2,3)) == 0) | (holdcounts.sum(axis=(1,2,3)) == 0)):
        raise ValueError("empty training or heldout population")
    arrays = dict(counts_all=allcounts, counts_train=counts, counts_holdout=holdcounts,
                  raw_selection_exposure=exposure, selection_shells=exposure_shells,
                  survival_yes=yes, survival_no=no)
    np.savez_compressed(root/"corrected_counts.npz", **arrays)
    np.savez_compressed(root/"corrected_rows.npz", recno=recnos, population=population,
        shell=shell, survives=survives, calibration=calibration, train=train, holdout=held)
    report = dict(status="PASS_CORRECTED_INPUTS_NOT_SELECTION_CERTIFICATION",
        magnitude_convention="M_phys - 5log10(h); luminosity distance in Mpc/h",
        magnitude_shift=float(-5*np.log10(tracer["cosmology"]["h"])),
        parent_rows=len(recnos), retained_counts=int(allcounts.sum()), training_counts=int(counts.sum()),
        holdout_counts=int(holdcounts.sum()), count_population_totals=allcounts.sum(axis=(1,2,3)).tolist(),
        calibration_parent_rows=int(calibration.sum()), calibration_survivors=int((calibration & survives).sum()),
        survival_yes=yes.tolist(), survival_no=no.tolist(),
        survival_beta_mean=((yes+1)/(yes+no+2)).tolist(),
        train_fraction=.6, holdout_fraction=.2, calibration_fraction=.2,
        reused_heldout_in_calibration=False, previous_rejected_rows_reintroduced=False,
        calibration_model="Beta(1,1) updated by disjoint calibration-only survival marks per population/radial shell",
        limitations="Conditional ignorability within population/radial shell is approximate; assembly/environment/sky dependence and LF uncertainty remain. Calibration rows supply marks only, not a density likelihood or rate centring.",
        bindings=bindings)
    (root/"corrected_counts.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    import sys
    # Catalog preparation uses the existing astronomy environment, not the JAX environment.
    plan = json.loads(Path(sys.argv[1]).read_text())
    parent = json.loads(Path(plan["extends"]).read_text())
    parent.update(plan)
    root = Path(parent["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    build(parent, root)
