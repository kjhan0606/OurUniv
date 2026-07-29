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
- The old `power_complete` and nonlinear-MAP subtraction products remain
  rejected as posterior samples. They must not be mixed with the accepted
  `linear_cr/parent_v1_all` ensemble.

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

Run the same HOP/void analysis on each parent candidate.

- Virgo: position, `M200c`, LG–Virgo relative velocity/infall.
- Coma: position and cluster mass at parent resolution.
- Local Void: center, radial density profile, effective radius/ellipticity.
- Boötes void: first freeze an observational catalog definition and coordinate
  transform; the current project has no explicit target coordinate.
- Reject candidates that recover one object by destroying another or by
  violating the CF4 likelihood/power gate.

### P2 — LG ensemble selection

At sufficient parent resolution, rank random small-scale realizations using:

- two distinct MW/M31 halos with declared `M200c` ranges;
- separation and total radial/tangential velocities;
- isolation from a nearby massive group;
- an M33-mass halo near M31 with the correct hierarchy;
- retained Virgo/void geometry.

Use an ensemble and report the selection probability; do not treat one seed as
deterministic reconstruction of observed dwarfs.

### P3 — exact Lagrangian masks

1. Select each z=0 target using halo membership plus a buffered Eulerian
   region.
2. Trace stable particle IDs to the initial snapshot.
3. Construct a non-spherical convex/voxel mask with convergence buffers.
4. Verify that all nested levels share long modes and that added short modes
   contain only the intended unconstrained band.
5. For LG, include material that forms MW, M31, M33, and the immediate tidal
   environment.

### P4 — inexpensive z=0 pilot

Before L14:

- run an L11/L12 pilot to z=0;
- repeat HOP and spherical-overdensity measurements;
- require the target to lie inside the high-resolution region;
- require a declared contamination limit (default `<1%` mass inside each
  `R200c`, aiming for zero);
- verify parent–zoom large-scale cross-power and halo displacement.

Only a passing pilot authorizes the production zoom.

### P5 — production and reproducibility

- Run LG L14 (or higher if M33/satellite science requires it).
- Create separate Virgo/Coma zooms as needed, all from the same parent phases.
- Re-run the complete scorecard at every final snapshot.
- Store code, configs, manifests, small catalogs, figures, and checksums in
  GitHub. Keep raw RAMSES/HOP binaries and multi-GB tags on GPFS, referenced by
  manifest rather than committed.

## Immediate next action

Freeze the P1/P2 acceptance table before looking at the 16 members, including
the Boötes-void observational definition. Then forward all 16 accepted CRs at
the same inexpensive resolution and score Virgo, Coma, the Local/Boötes
voids, and LG environment. Only after that pre-registered selection should a
single physical parent and its small-scale LG seed family be chosen.

In parallel, the failed cr6/e19 run can still be used for a diagnostic
Lagrangian backtrace, but it is no longer a candidate parent.
