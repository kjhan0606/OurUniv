# OurUniv / CF4 hybrid IC-inference plan

## Scientific objective

Infer an ensemble of ΛCDM initial conditions using the original CF4
observation-space likelihood directly on PM-forwarded fields. A full
present-day density/velocity posterior supplies proposals, initialization,
preconditioning, and diagnostics, but is not a second likelihood. The
forward-evolved ensemble must reproduce, within declared observational
tolerances:

- the MW–M31–M33 configuration and dynamics;
- Virgo and Coma;
- the Local Void and Boötes void;
- the CF4 peculiar-velocity data and large-scale density/velocity field;
- the target ΛCDM initial power spectrum and phase statistics.

The deliverables are a calibrated CF4-supported low-k posterior, a dynamically
closed hybrid IC posterior, and phase-consistent zoom-IC families. A numerical
grid or force target of `<=0.3 cMpc/h` does not establish observational
effective resolution at `0.3 cMpc/h`. LG, Virgo, and Coma should normally be
separate zoom runs sharing the same parent phases; resolving all three in one
L14 run is not required.

## Authoritative route and non-regression rule

The machine-readable authority is
`config/cf4_science_route_v2.json`. The mandatory order is:

1. **RES-CAL:** use CF4 selection/noise truth mocks to determine actual low-k
   `k_eff`, information gain, held-out prediction, and calibrated uncertainty
   coverage;
2. **Z0-PROP:** after RES-CAL passes, construct a full z=0 density/velocity
   posterior for IC proposal, initialization, preconditioning, and diagnosis;
3. **IC-HYBRID:** use that proposal while applying the original/raw CF4
   observation-space likelihood directly to PM-forwarded ICs through
   importance reweighting with evaluated `q`, exact-ratio MH accept/reject, or
   SMC reweight/resample/move with a declared normalized target sequence;
4. **HIRES/ZOOM:** only after hybrid closure, add explicitly labelled high-k
   conditional-prior modes and separate tracer likelihoods, then consider
   zoom/HOP/RAMSES validation.

The final target is
`p(IC|CF4) proportional to p(IC) p(CF4|F_z0(IC))`. The z=0 posterior must
retain its full ensemble/covariance; one deterministic point map must never be
treated as truth and literally reversed. Because that posterior was derived
from CF4, multiplying it as an independent likelihood would double count the
same data and is forbidden. Pure direct `CF4 -> IC` sampling is the
statistically clean target but is not required to be the sole search mechanism
because of its mixing and cost. Existing direct-route artifacts remain only
low-k proposals, controls, and historical evidence and cannot authorize
parent selection, high-k IC production, zoom promotion, HOP, or RAMSES.

For proposal `q(IC)`, the importance weight is
`w proportional to p(IC)p(CF4|F_z0(IC))/q(IC)`; MH accept/reject must contain
the exact target ratio and forward/reverse proposal ratio, and SMC must declare
its normalized target sequence and perform the corresponding
reweight/resample/move steps. An implicit proposal is ineligible for final
posterior correction if `q` cannot be evaluated and no target-invariant kernel
has been proved.

A change in this order requires explicit user approval and an earlier
committed update to the authoritative route record; a status summary or
historical result cannot change it.

## Current state

- **Active stage: RES-CAL, effective-resolution and coverage calibration.**
  Final scientific approval of the CF4-supported low-k reconstruction is
  currently **NO-GO**.
- Z0-PROP is blocked by RES-CAL; IC-HYBRID is blocked by Z0-PROP; HIRES/ZOOM is
  blocked by IC-HYBRID closure.
- The Hong-style model family is closed after a negative cross-code result.
  This is an estimator-only negative result and does not close the hybrid
  route.
- No direct parent/seed promotion, new high-k IC production, zoom advancement,
  HOP, or RAMSES execution is active or authorized by this plan.
- The existing CF4 linear posterior and the 768-member unconstrained P1
  reference are reusable low-k proposals, controls, and historical evidence,
  not evidence for promotion or observational high-k recovery.

### Resolution claim policy

Numerical grid/force resolution and observational effective resolution are
separate quantities. CF4 alone does not currently support a claim of
`0.3 cMpc/h` observational density-phase reconstruction. Random high-k
completion, interpolation, super-resolution, and zoom dynamics must be labelled
as `conditional-prior` or `numerically-resolved` content.

An observational `0.3 cMpc/h` claim requires independent truth mocks to pass a
preregistered gate over the declared volume and continuous k-band, never only
selected locations or bins. Cross-response must remain within `0.8--1.2`,
`r(k) >= 0.7`, and the residual-power ratio must be `<=0.5`. Phase coherence
must show a preregistered significant separation from the mock random-phase
null, and information gain must be a material variance reduction relative to
the prior. The 68/95% coverage thresholds and held-out-prediction improvement
must also be preregistered on mocks. With the cell-scale convention, the
claimed band must extend to `k_claim = pi / 0.3 ~= 10.47 h/Mpc`.

### Archived direct-route evidence

- The original WF15 P0 ensemble passed on 2026-07-29 and remains archived as
  the control realization set.
- The sampler uses the Gaussian Watkins--Feldman log-distance velocity
  estimator, a matrix-free Wiener filter, and exact Matheron conditional
  draws. It marginalizes an external bulk flow and the CF4 distance-scale
  zero point.
- A previously unseen 20% test split passed at N=192: standardized residual
  mean/std `+0.0077/1.0492`, 68/95% coverage `0.7050/0.9444`, and
  `Delta log predictive density=+815.0` relative to noise-only.
- **The replacement BGc P0 passed on 2026-08-02.** Its all-data product is the
  16-member N=192, L=384 Mpc/h ensemble at
  `/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_all_v1`, seeds 3001--3016.
  All numerical, Gaussianity, and ΛCDM shell-power gates passed; the maximum
  CG residual was `3.10e-5`, white-field standard deviations span
  `0.999379--1.000378`, and the independent predictive test retained
  `z=+0.023 +/- 1.012` and `Delta log score=+374.9`.
- The first BGc P1 pass selected seed 3003 on Virgo, Coma, Local Void, and
  Boötes alone. The subsequent P2 audit found persistent `2--4e13 Msun/h`
  groups only about `3 Mpc/h` from the observer. A new observer-centred mass
  gate therefore rejects seed 3003. None of the first 16 members passes all
  five gates. A preregistered 48-member posterior extension passed P0, and
  seed 3023 was its unique five-gate P1 survivor. An independently frozen
  128-member extension (seeds 3065--3192) also passed P0; seed 3096 is its
  sole five-gate P1 survivor. Its observer-centred mean densities are
  `delta_R5=-0.317` and `delta_R8=-0.312`, so it does not repeat seed 3003's
  nearby massive-group failure. Its unchanged 32-seed N576 P2 ensemble
  completed with zero screen pairs. Across the two physically valid P1
  parents, 3023 and 3096, the frozen result is therefore 0/64. The final
  preregistered 256-parent extension (3193--3448) passed P0; seed 3429 was its
  sole five-gate P1 survivor and its unchanged 32-seed N576 P2 screen also
  returned zero pairs. The CF4-only result is therefore 0/96 across the three
  valid P1 parents.
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
- **BGc N192 validation passed on 2026-08-02.** The leakage-free BGc model
  uses an 801-row reference window, `1500 <= cz <= 18000 km/s`, and a
  development-selected error scale of `0.90`. On a reserved N192 split it
  obtained `z=+0.023 +/- 1.012`, 68/95% coverage `0.708/0.951`, and
  `Delta log score=+374.9`. Its observer `R=5 Mpc/h` posterior density is
  `+0.430`; all eight validation realizations are positive. This removes the
  shared central-cavity failure at the likelihood level without adding a
  direct density constraint.

## Archived direct-route gates

The P0--P5 material below records completed or attempted work on the superseded
direct route. Its status labels are historical and do not override the active
Z0 route above.

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
- In the original WF15 control, seeds 1002 and 1009 passed all four gates.
- Repeating the original scorecard on the accepted BGc ensemble left seed
  3003 as the only field passing Virgo, Coma, the four-probe Local Void, and
  the Boötes-void center/profile. That historical result is stored at
  `/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_parent_v1/p1_result.json`.
- P2 exposed a missing Local-Volume condition: seed 3003 forms group-scale
  halos much nearer than Virgo. `config/p1_targets_v2_observer.json` adds
  fixed excess-mass bounds within 5 and 8 Mpc/h while leaving every old gate
  unchanged. Seed 3003 has excess masses `4.19e13` and `1.43e14 Msun/h` and
  fails. The first 16 have zero five-gate passes.
- Fornax, Hydra, Centaurus, Norma, Perseus, and the Shapley core were measured
  as blind non-gating anchors; they were not used to retune the hard gates.
- Exact cluster masses, relative velocities, and extended basin/filament
  topology remain higher-resolution measurements.
- The same-model seeds 3017--3064 all passed the statistical P0 gate. Seed
  3023 alone passes the five-gate P1: its coarse Local-Volume excess masses
  are `3.09e12 Msun/h` (R=5) and `4.23e13 Msun/h` (R=8). However, neither its
  eight canonical-L10 proposals nor its frozen 32-member N576 ensemble formed
  a P2 screen pair. The conditional success estimate is therefore 0/32.

### P2 — LG ensemble selection

**Status: CF4-only 0/96; the first explicit-LG pilot and fixed-midpoint
proposal model are rejected; a fresh latent-midpoint bank is frozen.**

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

After BGc repaired the central likelihood, an exploratory N576 screen of parent 3003
found passing morphology for small-scale seeds 2001, 2005, and 2008. Seed
2008 had the best frozen ranking score (`2.558`), but its screen-scale total
radial velocity was receding (`+217 km/s`) and is not a definitive dynamical
pass. More importantly, calling the RNG with the same seed at N512 does not
preserve the N576 Fourier realization. Therefore N576 candidates are not
promoted directly. Independent N512 redraw and N576-to-N512 canonical
projection checks both yielded zero passes. More decisively, their apparent
N576 passes sit beside unphysical observer-neighbour groups and are rejected.

The frozen 128-member parent extension (3065--3192) passed P0, and seed 3096
alone passed all five P1 gates. Config
`config/p2_lg_targets_v8_bgc_n576_parent3096.json` applies exactly the same
small-scale seeds 2001--2032, physical cuts, and ranking weights used for seed
3023. Only the parent, P1-result path, and output directory differ. It
completed with zero passes, giving 0/64 across parents 3023 and 3096.

The final parent 3429 CF4-only bank also returned 0/32. An explicitly labelled
additional MW/M31 peak likelihood then produced three coarse P2 pairs in eight
draws. The promoted p3429/s5108 L12/L19 RAMSES run was clean and phase
consistent, and HOP found an M33-scale subpeak, but the definitive pair failed:
`M200c=(3.23,1.65)e12 Msun/h` and separation `1.224 Mpc/h` violate the frozen
first-member mass and separation gates. Its environment gate also failed.

Keeping the same inverse-mapped Lagrangian midpoint and drawing 96 more
explicit-likelihood proposals (v4/v5) gave six PM screen realizations and nine
pairs, but none retained the five P1 gates when the observer was placed at the
evolved pair midpoint. Every pair failed Virgo and Coma, with midpoint offsets
`2.006--4.641 Mpc/h`. The unchanged fixed-midpoint model is therefore closed;
see `config/cf4_lg_v5_result_record.json`.

If this ensemble has zero passes, the final blind extension is preregistered
in `config/v3_bgc_parent_extension_v3.json`: seeds 3193--3448 receive the same
P0 and observer-aware P1, and every P1 survivor receives the same 32 P2
small-scale seeds. No further blind extension or threshold retuning follows a
failure of that batch; the next model must label MW/M31/M33 information as an
explicit additional likelihood rather than attributing it to CF4.

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

Complete the bounded, no-production **RES-CAL** design and preregistration:

1. specify independent CF4 selection/noise truth mocks and untouched held-out
   predictions;
2. freeze a continuous k-band and estimators for cross-response, `r(k)`, phase
   coherence, residual power, and information gain;
3. freeze 68/95% posterior-coverage tests and the rule that derives actual
   low-k `k_eff` rather than equating it with voxel size;
4. label random high-k, interpolation, super-resolution, and zoom-generated
   content as conditional-prior/numerically-resolved;
5. preregister pass/fail thresholds before any Z0-PROP implementation.

This step does not authorize Slurm submission, model training, parent
selection, IC generation, PM/HOP/RAMSES execution, or production artifacts.
The superseded v8 program and rejected zooms remain historical diagnostics.
