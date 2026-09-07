# Bundle C execution

User approved C entry after B closure. Design: BUNDLE_C_DESIGN.md.
Native-observation builder and conservative multiresolution parameterization
implemented. Initial CPU job:2 cores,3600 MiB (3000+20%),20min cap.
No GPU, new IC/PM/RAMSES run or fine posterior fit submitted at this point.

CPU334407 stopped before output generation: NumPy2 does not multiply an int8
population array by32768 without explicit promotion. Corrected to int64,
matching the existing fine-key construction. Preserve the first directory;
retry uses native_data_v2. No scientific selection or input rows changed.

The first delivery covered C1–2; the current work is C3 selection, with the
fine prior/LG operator and C4–5 still remaining. Do not call input preparation or
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

Job334507 stopped in the synthetic strip regression before creating v2:
the test strip's pixel-centre selection unintentionally included an old Gauss
node's entire HEALPix pixel. Moved the synthetic strip farther toward the cube
face, leaving a full-pixel margin. Production masks/geometry/Sobol settings
are unchanged; this corrects the test fixture, not the scientific algorithm.

Retry334508 COMPLETED on syn07 under Slurm (source854d057),2026-09-08
00:45:51–00:55:39 KST,9m48s, MaxRSS967352K. All3 focused regressions passed.
Status: SUPPORT_REPAIRED_NOT_SELECTION_CALIBRATION.

- Geometry-selected candidate population/cells:305732. Added positive
  exposure:70261 entries, of which70260 have no observed galaxy count.
- Occupied zero-support keys:1→0 across all32162 occupied population/cells.
  The failed key40599936 now has exposure0.0154009052 (previously exactly0).
  The earlier order64 diagnostic was0.0157118549; neither is exact truth.
- Preservation check334512, afterok334508, completed00:55:48 KST in9s.
  All15862681 originally positive population/shell/cell entries are bitwise
  unchanged, and no entry decreased. The observed-support check passes.
-120 occupancy-blind geometry control cells compared against an independent
 32768-point Sobol rule: max/mean tested shell-L1 absolute differences
 9.37195e-4/4.66319e-5. In33 population/cell control entries, the denser rule
 found tiny positive support missed by2048 points. Thus neither full support
 completeness nor global quadrature precision has been certified. No
 occupied entry remains in that zero-support class.

Outputs: `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/selection_1p5_v2/`
contains selection.h5, support_changes.npz and result.json. Preserve v1.
This resolves the immediate actual-count likelihood impossibility; it does
not calibrate the source angular mask, survival/bias, or high-resolution
matter prior. No C calculation or downstream inference remains queued.
Next work within C is specifying the physical fine-field prior and explicit
LG operator, then the bounded LG-on/off field inference. Additional generic
validation infrastructure is not the deliverable; D is still unapproved.
