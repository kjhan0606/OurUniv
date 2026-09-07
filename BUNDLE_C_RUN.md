# Bundle C execution

User approved C entry after B closure. Design: BUNDLE_C_DESIGN.md.
Native-observation builder and conservative multiresolution parameterization
implemented. Initial CPU job:2 cores,3600 MiB (3000+20%),20min cap.
No GPU, new IC/PM/RAMSES run or fine posterior fit submitted at this point.

CPU334407 stopped before output generation: NumPy2 does not multiply an int8
population array by32768 without explicit promotion. Corrected to int64,
matching the existing fine-key construction. Preserve the first directory;
retry uses native_data_v2. No scientific selection or input rows changed.

The current task is C1–2; C3–5 remain. Do not call input preparation or
zero-detail cell refinement a reconstruction. Do not claim0.1875 information
resolution from the configured cell size. Continue within C after the native
inputs and geometry tests; D still needs separate user approval.

Native-data retry334408 completed11s; all3 geometry/conservation/JAX tests pass.
Counts:24998 training,8378 heldout,32162 occupied population/cell keys at1.5.
CF4:19313 rows, minimum retained radius15.8152 cMpc/h and cz1501 km/s.
Inside R2 cMpc/h: ZERO direct CF4 rows and ZERO retained galaxy-count rows.
The buffered LG patch has59 CF4 and56 count rows, but these are NOT direct
MW/M31/M33 measurements. Explicit LG likelihood is therefore indispensable.
Native data: `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2`.

Next execution integrates N256 selection directly (not old-grid interpolation),
with the same2 CPUs/3600 MiB and30min cap, <3 GiB compressed output. Existing
TNG raw group catalog is available under /scratch, not the derived-only /gpfs
TNG root, if later needed for resolved-operator calibration. No TNG training,
template insertion or halo selection has been run. Fine-field prior and LG
halo/subhalo operator remain to be specified; C has not been completed.

## Selection integration and support repair

Selection334409 completed on syn05 under Slurm, source commit66804dd,
2026-09-07 23:40:30 KST, elapsed7m43s, MaxRSS394792K.
2 CPUs,3600 MiB,30min limit; no GPU. Rotation/radial-tabulation and all3
multiresolution tests passed. Scientific result is
UNRESOLVED_POSITIVE_COUNT_SUPPORT, not a selection pass.
Output: `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/selection_1p5_v1`.
Logs: `/gpfs/kjhan/CF4/logs/cf4_C_selection_334409.{out,err}`.
Only trust result.json after completion; selection.h5 is partial while running.
No downstream field inference is queued automatically. Next in the approved
C bundle is review of selection support followed by the resolved LG operator
and scale-dependent field-prior specification, then the bounded inference.
No new user approval is needed for ordinary work inside C; D remains gated.

The 2026-09-08 Slurm diagnostic identifies key40599936 = population2,
cell[107,129,128], occupied by training rows6921/6935. At both observed
positions source completeness=0.51851851 and LF fraction=1. The order4
Gauss nodes all land outside the positive footprint. Orders8/16/32/64 give
0.0159044/0.0171779/0.0154469/0.0157119: an angular quadrature miss, not an
observational magnitude/distance inconsistency. No rows or masks are changed.

Implemented repair: source-mask boundary pixels plus conservative cell angular
caps and radial LF support identify potentially missed ZERO exposure entries.
2048 fixed scrambled Sobol volume points re-integrate those entries, using no
catalogue occupancy to choose cells or set probabilities. Nonzero v1 entries
are preserved. Empty cells receive exactly the same policy. This is a bounded
zero-support repair, not a claim that all angular quadrature is converged.
Up to128 spatially fixed geometry controls use an independent32768-point
reference. Counts are read only after all replacement integrals are fixed.
New output selection_1p5_v2; v1 remains untouched. Three focused regression
tests and the repair will run in one Slurm job,2 CPUs,3600 MiB (3000+20%),
30min limit, <3 GiB output. No new field inference or LG halo operator yet.
