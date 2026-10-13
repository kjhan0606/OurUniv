# R2 map-aware exposure precompute — bundled plan, 2026-10-04

Status: driver-approved implementation plan after Slurm geometry census
412034 and a read-only Fable5 advisory. This is one numerical observation-
operator bundle, not a density-field delivery.

## Purpose and limits

Produce map-aware 2M++ selection exposure for the N128, 3 cMpc/h count grid
used by the current coarse R2 likelihood controls. The larger R2 field target
remains N256 at 1.5 cMpc/h; LG resolution remains <=0.3 cMpc/h. This result
does not satisfy either field-resolution target, calibrate galaxy bias or
selection survival, resolve shared group covariance, or create a posterior.
It must not be promoted to the final z=0 field or zoom initial conditions.

Inputs are the hash-pinned NSIDE=512 RING completeness maps, common-cosmology
contract, the six fixed absolute-K population intervals, and exact voxel/ray
geometry. The existing order-six N128 exposure file may be read only after
construction for a cellwise old/new error summary; it never changes new
exposure values. Do not read catalogues, count/key arrays, train/holdout
splits, field states, or native truth identities. Preserve map pixel values;
NSIDE=1024 and 2048 are angular quadrature refinements, not new observational
detail.

## One-job implementation and checks

Implementation is staged in `scripts/cf4_r2_selection_ray_integral.py`,
with focused regressions in `tests/test_cf4_r2_selection_ray_integral.py` and
the typed-H100 runner in `scripts/run_cf4_r2_selection_ray_integral.sbatch`.
The source census records the complete 2×2×2 cross-tab of inner-boundary,
outer-boundary, and cap-limit status; it does not infer overlap from equal
marginal totals. The driver source review also confirms that the old HDF5
exposure is used only for the post-integration comparison, while the new
exposure values are calculated from pinned maps and the frozen cosmology.

1. Enumerate all N128 cells whose volume intersects `5 < r < 180 cMpc/h`.
   Compute and record the exact joint table of inner/outer shell status and
   cap-area candidate-limit status; do not assume the equal census totals
   (56 and 56) describe identical cells.
2. Remove the profiler's 1.5-million candidate skip. Use the conservative
   angular cap and inclusive HEALPix query, then process candidate directions
   in fixed-size chunks (initially 250,000). For each cell and direction,
   clip the ray interval at `5, 30, 60, 90, 120, 150, 180` cMpc/h and
   accumulate all six populations by all six radial shells. The coarse count
   operator consumes the shell sum per population; do not assign a whole cell
   to a shell by its center.
3. Before full integration, compare two chunk sizes and the unchunked
   reference on the eight maximum-cap cells; reproduce the two existing
   NSIDE=2048 exposure references within `1e-7` absolute on their feasible
   controls. Check the existing fixed geometry controls. These checks run in
   the same allocation, before the full grid.
4. Compute full N128 exposures at NSIDE=1024 and NSIDE=2048 in that same job.
   NSIDE=1024 is a convergence diagnostic, not a promotion gate. Record
   per-population/per-shell cellwise relative-change quantiles and zero-ray
   active cells. Do not launch NSIDE=4096 or adaptive refinement unless these
   results expose a material issue.
5. Enforce a global closure identity. Since the radial shell ends at 180 and
   the half-box is 192 cMpc/h, every sampled ray's full shell interval lies
   inside the cube. For each population `p` and radial shell `s`, the sum of
   cell exposures must equal
   `dOmega * sum_pix(c_p(parent512)) * [F_p(r_hi)-F_p(r_lo)] / dx^3`.
   Also run a pure-geometry channel (`c=1`, radial integrand `r^2`) whose
   shell sum must equal `4*pi*(r_hi^3-r_lo^3)/(3*dx^3)`. Use float64
   accumulation and a relative tolerance of `1e-10`.
6. Compare the finished exposure to the existing order-six N128 artifact by
   population, radial shell, and radius summary. Report zero-support changes
   explicitly; do not tune the integral to those values.
7. Save one versioned HDF5 artifact with source commit, map hashes, cosmology,
   NSIDE, shell edges, active flat-cell indices, six-by-six exposures, ray-hit
   counts, closure residuals, resolution comparison, zero-ray list, runtime,
   and peak memory. Label it
   `NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY`. A partial file must
   remain explicitly `INCOMPLETE`; only set complete after all closure checks.

The full output is computed without count/key arrays. Any later support check
uses training counts only and is separate from construction; no holdout is
opened. The historically contaminated holdout is replaced only after this
integration method is frozen.

## Q-GOAL and Q-LEAN

- Q-GOAL: fixes a measured cellwise selection-operator defect required by the
  same CF4-conditioned z=0 inference. It is groundwork at the 3 cMpc/h count
  grid, not a z=0 field result. Selection/bias calibration and the shared-
  latent count/mark conditional remain the actual science blockers.
- Q-LEAN: chunking directly addresses the measured 10.64-million-ray maximum
  and prevents valid cells from being skipped. Two chunk sizes, a few fixed
  references, a full-grid closure, and one 1024/2048 comparison are in the
  same job; no separate approval stages, fit, posterior, simulation, holdout
  score, or high-resolution adaptive sweep.

MW and M31 roles remain ambiguous; M33 remains unresolved. This bundle
identifies no LG member. Their observables must later constrain those same
roles in the same NEW evolved field at LG <=0.3 cMpc/h. Native truth identities
are reserved for calibration/evaluation, never candidate selection.

## Fable5 advice and driver assessment

Fable5 (`claude -p`, `--model opus`, plan/read-only permissions) returned
**CONDITIONAL PASS**. It recommended one job combining self-tests, bounded
chunk comparisons, full NSIDE=2048 integration, and exact closure; it opposed
separate validation jobs and suggested 16–32 CPU parallelism. It also
recommended recording candidate-hit counts, source hashes, and explicit
population-by-shell semantics. Its reasoning was that the 1.5M guard is an
implementation cap, the measured maximum query has 10.64M candidates, and
the full exposure is a selection denominator rather than a field or LG
result.

The driver adopts the no-skip path, one-job structure, exact closure,
population-by-shell clipping, chunk comparisons, and a full-grid NSIDE=1024
diagnostic paired with NSIDE=2048. Parallelism will be bounded by the node's
available CPU and memory configuration; expected peak is estimated from the
measured 0.205 GiB single-process stress result times the worker count plus
the parent/output buffers, then requested memory includes >=20% headroom.
H200 is reserved and A100 has one down node; use one compatible typed H100
Slurm job, never manual execution.

Fable inferred that the 56 inner-boundary cells may be exactly the 56 cells
above the cap and estimated candidate counts for their geometric shells. The
census did not save that cross-tab or all exact query counts, so the new
program computes and reports it rather than treating the inference as proven.
Fable proposed failing on every geometrically active cell with zero pixel
center hits. The driver instead records those cells as quadrature diagnostics:
a positive-volume sliver can fall between finite pixel centers. A later
training-support check must still reject any occupied cell with zero exposure;
global closure must pass before the artifact can be marked complete.

Fable advised that a full NSIDE=1024 pass and old-versus-new cellwise
comparison are deferrable. The driver includes the former in the same job
because prior resolution controls omitted the observer-near geometry; the
old-versus-new cellwise comparison is reported in the same job without
becoming a gate. The advice is not authority: the driver checked it against
the exact census, the active count-grid semantics, and the unchanged R2/LG
goal.

Node check on syntax found 64 effective CPUs and ample host memory on H100
node syn08; request one typed H100, 16 CPUs, 8 GiB host memory, and a 4-hour
limit. The measured chunked maximum-cap geometry peak was0.205GiB for one
process; budget conservatively for up to15 worker processes plus the parent,
maps, and output buffers. This yields an estimated peak below6GiB, so 8GiB
exceeds the required20% headroom. The job uses bounded chunks in all workers;
no manual node run.
