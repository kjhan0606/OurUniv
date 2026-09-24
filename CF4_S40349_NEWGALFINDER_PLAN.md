# Seed 40349 z=0 NewGalFinder structure plan

Date: 2026-09-24

## Goal and claim boundary

Use the user's latest GalaxyFinder pipeline to identify the z=0 halo and
subhalo structure in seed40349's completed trace-only zoom. NewGalFinder is
preferred over HOP here because it operates inside opFoF hosts with density
peaks, watershed membership, iterative boundedness, and tidal-radius tests.
Its DMO fallback can therefore test whether an M33-scale bound subhalo is
separable from an M31-scale host instead of relying on HOP regroup thresholds.

This analysis may identify a viable MW/M31/M33 analogue, ambiguity, or a
failure. It cannot promote the parent by itself. Any fine substructure comes
from one random conditional high-k realization, not directly recovered CF4
information.

## Frozen inputs and software

- RAMSES snapshot:
  `/gpfs/kjhan/CF4/ramses/s40349_zoom_l19_z0_v1/job_1109268/output_00003`
- NewDD/opFoF working source and binaries:
  `/home/kjhan/BACKUP/GalaxyFinder`; preserve its existing staged DMO-reader
  changes and record exact binary/source hashes.
- NewGalFinder source: GalaxyFinder branch
  `agent/fix-newgalfinder-periodic-unwrapping`, commit
  `96560d9ceef34a0143304d8a32534649634d5116`, checked out separately at
  `/gpfs/kjhan/CF4/external/GalaxyFinder_newgal_586a62e` so the user's dirty
  local tree is not overwritten.
- NewGalFinder binary: the same finder physics, rebuilt with `NMEG=8000L` and
  `NBODY`. SHA256:
  `2702135f6195714257a7ca3298c5bda2f128a7bc2f987587ca336f8d091dff09`.
- Input/output ABI: `INDEX`, `VarPM`, `XYZDBL`, `NBODY`; the compact `DmType`
  is compile-time asserted to be 72 bytes, exactly matching the measured opFoF
  member layout. The branch also fixes value sorting and post-shift bounds in
  periodic halo unwrapping and includes the latest DMO peak fallback fixes.

## Execution order

1. Run NewDD with all DMO particle masses preserved, 16 MPI ranks and 512
   slabs, then run opFoF on the same slabs. Validate catalog and member files.
2. Inspect the opFoF header count and largest host before releasing the larger
   NewGalFinder allocation.
3. Run NewGalFinder on the validated opFoF catalog. It contains 146,681 hosts;
   the largest contains 165,853 particles. This is safely below the 8 GB worker
   allocator ceiling. The four-rank build reserves 80 GB for the master and
   8 GB for each of three workers; allowing for OpenMP stacks and overhead gives
   an expected ceiling near 125 GB. Request 150 GB (at least 20% margin), use
   4 MPI ranks x16 OpenMP threads on one grammar normal node, and impose a
   24-hour ceiling.
4. Parse the resulting subhalo catalog without oracle identities. Locate an
   isolated MW/M31-scale pair near the constrained observer/environment,
   search for an M33-scale bound satellite around either component, and measure
   masses, separation, radial/tangential velocities, environment drift, and
   low-resolution contamination from the RAMSES particle masses.

The first launch, grammar job1113477, was cancelled after54 seconds when its
preflight-incomplete hydro build read the 72-byte DMO records with a 168-byte
stride. The resulting false coordinates triggered grid-size overflow. It made
no usable output. The corrected DMO ABI and periodic-unwrapping branch above
are mandatory for the rerun; adding memory would not fix this error.

The corrected-ABI launch, job1113504, demonstrated valid coordinates and DMO
subhalo separation, reaching host number4992 without an allocation failure.
It was then cancelled deliberately after9m47s: with `DM_DENSITY_WEIGHT=0`, the
code still computed an empty stellar FFT before every dedicated DMO search.
Commit96560d9 routes zero-star hosts directly to the same adaptive DMO finder
and zero-initializes per-halo state. This removes a scientifically inert cost
and the associated large-host memory risk without changing the DMO algorithm.

## Q-GOAL and Q-LEAN driver review

Q-GOAL: strong. This directly attacks the unresolved MW/M31/M33 structural
identification that blocks scientific promotion after the z=0 forward pass.

Q-LEAN: acceptable. It reuses the one final RAMSES dump and one required
NewDD/opFoF conversion, replaces rather than duplicates HOP, and performs one
fixed NewGalFinder run. No RAMSES rerun, finder parameter sweep, or extra raw
snapshot is authorized. NewDD slabs are transient and should be removed after
the opFoF/NewGalFinder products and science decision are sealed.
