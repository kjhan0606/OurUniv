# OurUniv / CF4 constrained-initial-condition plan

## Scientific objective

Construct an ensemble of ΛCDM constrained realizations that simultaneously
reproduce, within declared observational tolerances:

- the MW–M31–M33 configuration and dynamics;
- Virgo and Coma;
- the Local Void and Boötes void;
- the CF4 peculiar-velocity data and large-scale density/velocity field;
- the target ΛCDM initial power spectrum and phase statistics.

The deliverable is one validated parent realization plus phase-consistent
zoom-IC families. LG, Virgo, and Coma should normally be separate zoom runs
sharing the same parent phases; resolving all three in one L14 run is not
required.

## Current state

- **P0 passed on 2026-07-29.** The accepted statistical product is the
  16-member N=192, L=384 Mpc/h linear-Gaussian parent ensemble in
  `recon/linear_cr/manifest_parent_v1_all.json`.
- The sampler uses the Gaussian Watkins--Feldman log-distance velocity
  estimator, a matrix-free Wiener filter, and exact Matheron conditional
  draws. It marginalizes an external bulk flow and the CF4 distance-scale
  zero point.
- A previously unseen 20% test split passed at N=192: standardized residual
  mean/std `+0.0077/1.0492`, 68/95% coverage `0.7050/0.9444`, and
  `Delta log predictive density=+815.0` relative to noise-only.
- All 16 all-data draws pass the numerical, Gaussianity, and ΛCDM shell-power
  gates. See `recon/linear_cr/gate_v1/CR_GATE_REPORT.md`.
- Seed 1001 is only the deterministic reference member. No realization has
  yet been selected using Local-Group or named-structure morphology.
- The cr6/e19 L14 RAMSES run completed to `a=1.002955`.
- A multi-mass HOP catalog and direct `M200c`/contamination measurements were
  completed.
- The run fails the LG gate: there are no finest particles in the
  screen-selected LG region. The production IC enlarged the initial box
  center rather than the target's Lagrangian patch.
- This run is retained as a diagnostic only and is rejected for LG/M33/dwarf
  science.
- The old `power_complete` routine had the correct physical aim of restoring
  Wiener-suppressed high-k power with random phases. It approximated the
  residual covariance by an isotropic shell filter, so it and the nonlinear-MAP
  subtraction products remain archived for provenance rather than serving as
  inputs to the accepted `linear_cr/parent_v1_all` ensemble.

## Ordered gates

### P0 — statistically valid parent field

**Status: complete for the frozen linear-Gaussian model.**

1. Freeze cosmology, CF4 sample, frame, likelihood, selection function, and
   distance-error model.
2. Produce posterior constrained realizations, not only a hand-tuned/MAP
   field.
3. Require held-out velocity-likelihood improvement and calibrated residuals.
4. Verify `P(k)`, transfer function, Gaussianity, and cross-power/phase
   preservation against the target ΛCDM model.
5. Archive seed, code commit, configuration, and checksums for every candidate.

### P1 — full-box Local-Universe scorecard

**Status: complete at parent resolution.**

- Definitions and thresholds were frozen in `config/p1_targets_v1.json`.
- All 16 parent members were evolved with the same `pmwd` model and the exact
  cosmology stored in the parent manifest.
- Seeds 1002 and 1009 pass all four coarse environment gates: Virgo, Coma,
  the four-probe Local Void, and the Boötes-void center/profile.
- Fornax, Hydra, Centaurus, Norma, Perseus, and the Shapley core were measured
  as blind non-gating anchors; they were not used to retune the hard gates.
- Exact cluster masses, relative velocities, and extended basin/filament
  topology remain higher-resolution measurements.
- Both passing parents advance to P2; no single physical winner is selected
  at P1.

### P2 — LG ensemble selection

**Status: first paired screen complete; no candidate passed.**

At sufficient parent resolution, rank random small-scale realizations using:

- two distinct MW/M31 halos with declared `M200c` ranges;
- separation and total radial/tangential velocities;
- isolation from a nearby massive group;
- an M33-mass halo near M31 with the correct hierarchy;
- retained Virgo/void geometry.

Use an ensemble and report the selection probability; do not treat one seed as
deterministic reconstruction of observed dwarfs.

The first frozen paired screen applied small-scale seeds 2001--2008 to both P1
parents. None of the 16 combinations contained an eligible MW-mass halo within
6 Mpc/h of the observer. Parent-level diagnostics show a shared central cavity,
so increasing only high-k seeds is not justified. Before repeating P2, the
local low-redshift likelihood and the explicitly external Local-Sheet/LG
selection must be separated and validated.

### P3 — exact Lagrangian masks

**Resolution hierarchy frozen on 2026-07-29.** See
`config/ic_resolution_v1.json` and `IC_RESOLUTION.md`.

- Full box: `L=384 Mpc/h`, global RAMSES `levelmin=9` (`512^3`).
- Pilot LG load: finest IC particles L12, runtime `levelmax=19`.
- Production LG load: finest IC particles L13, runtime `levelmax=21`.

1. Select each z=0 target using halo membership plus a buffered Eulerian
   region.
2. Trace stable particle IDs to the initial snapshot.
3. Construct a non-spherical convex/voxel mask with convergence buffers.
4. Verify that all nested levels share long modes and that added short modes
   contain only the intended unconstrained band.
5. For LG, include material that forms MW, M31, M33, and the immediate tidal
   environment.

### P4 — inexpensive z=0 pilot

Before the aggressive production run:

- run an L12-particle/L19-AMR pilot to z=0;
- repeat HOP and spherical-overdensity measurements;
- require the target to lie inside the high-resolution region;
- require a declared contamination limit (default `<1%` mass inside each
  `R200c`, aiming for zero);
- verify parent–zoom large-scale cross-power and halo displacement.

Only a passing pilot authorizes the production zoom.

### P5 — production and reproducibility

- Run the accepted LG mask with L13 particles and runtime L21. This gives
  `m_DM=8.86e6 Msun/h` and a formal minimum cell size of
  `0.183 ckpc/h` (`0.245 physical kpc` at z=0).
- Create separate Virgo/Coma zooms as needed, all from the same parent phases.
- Re-run the complete scorecard at every final snapshot.
- Store code, configs, manifests, small catalogs, figures, and checksums in
  GitHub. Keep raw RAMSES/HOP binaries and multi-GB tags on GPFS, referenced by
  manifest rather than committed.

## Immediate next action

Build and validate a v2 parent/local likelihood, then repeat P2 at a resolution
that can actually form an LG candidate. For every accepted candidate:

1. write an initial RAMSES particle snapshot at `a=0.02`;
2. run HOP at z=0 and choose the LG midpoint without moving it to the box
   centre;
3. use `cf4_trace_ramses_ids.py` for the z=0-to-initial ID join;
4. use `cf4_lagrangian_mask.py` to build the buffered sparse L9 mask;
5. generate the L12/L19 pilot with `cf4_zoom_ic2.py`;
6. authorize L13/L21 only after the HOP and contamination gates pass.

The failed cr6/e19 run remains diagnostic only and is not a candidate parent.
