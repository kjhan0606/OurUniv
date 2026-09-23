# Seed 40349 trace-only zoom: z=0 forward plan

Date: 2026-09-23

## Scope and claim ceiling

This calculation restarts the numerically accepted `a=0.10` checkpoint of the
seed-40349 L9--L12 conditional zoom hierarchy and evolves it to `a=1`.  It is
the required forward test of one conditional high-k realization.  It does not
promote seed 40349, resolve M33, establish a posterior ensemble, or by itself
show that MW/M31/M33 are reproduced.

The strongest possible runtime result is
`TRACE_ONLY_ZOOM_L19_Z0_FORWARD_PASS`.  Scientific acceptance remains blocked
until the final snapshot is processed with the RAMSES GalaxyFinder newDD/opFoF
path and a complementary HOP peak/deblend diagnostic, followed by explicit
MW/M31/M33, environment, and contamination checks.

## Fixed execution contract

- Restart: `/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_early_a010_v2/job_1108481/output_00002`
- RAMSES binary: `/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d`
- DMO: `cosmo`, `pic`, and `poisson` true; `hydro` false; `omega_b=0`
- Mesh: global L9, IC hierarchy through L12, runtime ceiling L19
- Parallel layout: 32 MPI ranks x 2 OpenMP threads, identical MPI rank count
  to the checkpoint writer
- Capacity: `nparttot=190000000`, `ngridtot=48000000`, automatic maxima enabled
- Output: exactly one new full dump at `aout=1.0`; no periodic time or backup
  dumps
- Memory: measured checkpoint peak 48.89 GiB; conservative late-time estimate
  80 GiB; request 96 GiB (20% margin)
- Walltime: grammar normal maximum 48 hours.  The L9-only reference took about
  27 minutes on 16 MPI ranks and the previous, 4.6-times-larger L12 mask zoom
  took 22 h 56 min on 8 MPI ranks.  The current 32-rank run is therefore
  expected to finish well inside the allocation, but the limit is retained
  because late nonlinear refinement is not accurately predictable from the
  early checkpoint.
- Storage: the final dump is expected to be about 20--30 GiB; 319 TiB was free
  at planning time.  No intermediate dump is scheduled.

## Runtime gates

The wrapper refuses to start unless the early decision remains the exact
`TRACE_ONLY_ZOOM_L19_EARLY_NONLINEAR_PASS`, its final info file is present,
and the source scale factor is 0.10.  It requires a formal `INIT_PARAMS` block
even though this is a restart, uses the same 32-rank layout, and checks the
effective namelist before launch.

A runtime pass requires a restart marker, normal RAMSES completion, no fatal,
OOM, boundary, restart, or fine-multigrid nonconvergence marker, exactly one
new physical output directory, and a final scale factor in `[0.99, 1.05)`.
The maximum populated AMR level and per-level grid counts are measurements,
not silently relaxed acceptance criteria.  They feed the subsequent science
decision.

## Post-run science sequence

1. Record Slurm state, elapsed time, peak RSS, final scale factor, AMR levels,
   solver diagnostics, output size, and hashes in a tracked decision record.
2. Convert the final RAMSES dump with the GalaxyFinder `newDD` executable and
   run its RAMSES-compatible opFoF.  Preserve particle masses; do not use the
   historical GOTPM converter.
3. Run the existing RAMSES HOP diagnostic independently to expose peak
   merging/deblending ambiguity.
4. Test whether an MW/M31 pair can be identified without truth-label
   selection, search separately for an M33 analogue, and measure separation,
   radial velocity, masses, Virgo/environment drift, and low-resolution
   contamination.  Any M33 analogue in this run is a product of the random
   conditional high-k phases, not M33 information recovered from CF4.
5. Return PASS, CONDITIONAL/AMBIGUOUS, or NO-GO.  M33 unresolved status is not
   waived, and a numerically successful forward run cannot promote the seed
   by itself.

## Raw-output retention

The superseded `a=0.05` raw dump was removed after the independent `a=0.10`
checkpoint had completed and been validated; logs, namelists, and tracked
decisions remain.  Keep `a=0.10` until the z=0 dump is validated.  Keep the z=0
dump until both halo-finder products and all structure/contamination metrics
are sealed.  Only then remove superseded raw dumps, retaining provenance,
catalogues, diagnostics, hashes, and decision records.
