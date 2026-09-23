# Seed 40349 bounded early-nonlinear zoom gate

## Position in the eight-stage route

This is a stage 8 numerical gate following the completed stage 6 RAMSES
startup checks.  Seed 40349 remains a **trace-only** candidate: the L9
GalaxyFinder pair is not promoted by the frozen HOP criterion and M33 remains
unresolved.  This calculation tests numerical survival and nonlinear AMR
refinement of one conditional L9--L12 IC realization.  It does not validate a
CF4 posterior, promote the parent, identify MW/M31/M33, or run to z=0.

## Frozen calculation

- IC: `/gpfs/kjhan/CF4/zoom/cf4_lg_b3292_s40349_traceonly_l12_v2`
- Binary: `/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d`
- DMO: `cosmo`, `pic`, `poisson` true; `hydro` false
- Mesh: L9 base, loaded through L12, runtime ceiling L19
- Evolution: from a=0.02 to the first completed step at or beyond a=0.05
- Output: exactly one final full dump; no initial or periodic dump
- Acceptance: normal completion, FFT base solve, all refined MG solves
  converged, no boundary/OOM/fatal marker, exactly one target dump, and at
  least one populated level above L12

The target is intentionally the same early epoch used by the historical
L8--L12 gate, which reached a=0.05447 and L17.  It is early enough to bound
the test while exercising nonlinear refinement that the two-step startup gate
did not reach.

## Resource and storage basis

The authoritative two-step L9--L12 run used 48.72 GiB.  In the historical
L8 hierarchy, the reader/startup peak was 6.45 GiB and the comparable early
gate peak was 10.58 GiB, a factor 1.64.  Applying that measured factor gives
79.9 GiB; the request is 96 GiB, 20.1% above the estimate.  The job uses one
64-core grammar node as 32 MPI ranks x 2 OpenMP threads with a two-hour limit.
This is expected to finish in tens of minutes but retains scaling margin.

The historical final early dump was 3.5 GiB at L8.  Scaling the global base
by eight and allowing zoom metadata/refinement gives a conservative 32 GiB
estimate for the single intended dump.  `/gpfs` had 327 TiB free at plan time.
No checkpoint or additional science output is requested.

## Interpretation branches

- PASS: retain the one early dump for refinement, contamination and density
  inspection; this only permits planning a separately reviewed z=0 run.
- Numerical/resource failure: diagnose the concrete RAMSES failure without
  changing the scientific target or silently increasing capacity.
- No refinement above L12: reject this as an early-nonlinear gate even if the
  process exits normally; do not infer z=0 zoom viability.

MW/M31 can only be re-identified in a later new-field halo catalogue.  M33 is
explicitly unresolved here and cannot be inferred from this a~0.05 dump.
Native truth IDs are not used to promote or label a generated z=0 system.
