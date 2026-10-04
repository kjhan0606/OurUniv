# R2 map-aware ray exposure — result, 2026-10-04

Status: **completed numerical selection artifact; not likelihood-ready; R2
remains NO-GO.** The full result is
`/gpfs/kjhan/CF4/z0_density/r2_ray_selection_n128_nside1024_2048_20261004_v1/result.json`.
The HDF5 status is
`COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY`.

## Execution and implementation checks

Typed-H100 Slurm job412060 ran on `syn08` for1:34:51, exit0, requesting16
CPUs and8GiB. Slurm MaxRSS was3,961,908KiB (3.78GiB); the result JSON's
24.78GiB worker-sum bound is a deliberately loose theoretical sum of per-child
peak RSS, not the observed concurrent job peak. The previous unchunked census
job was412059; its eight tests passed but preflight stopped before output
creation because it assumed the geometry-only legacy routine returned an
exposure. Commit `c46c693` reconstructed that independent reference and added
a regression; the corrected job passed all nine tests.

The complete active-cell count is938,128. The exact2×2×2 table establishes
that all56 cells above the former1.5-million cap are exactly the56 inner-shell
partial cells;67,544 are outer-shell partial and870,528 fully inside both
radial boundaries. At NSIDE2048, the actual inclusive-disc candidate total is
7,088,692,124, about2.7% above the6,901,444,420 cap-area estimate. NSIDE1024
used1,819,724,558 candidates. No active cell was dropped.

Nine Slurm regressions passed; all eight maximum-cap cells agreed across
65,536-,250,000-, and one-piece chunks to at most7.7e-15 relative difference.
Six existing fixed-reference checks passed. Both full-sky closure identities
are excellent: population-shell relative error2.01e-15 (NSIDE1024) and
5.26e-15 (2048), with pure-geometry error below8e-16.

## Scientific interpretation

Global closure proves the sum of the implemented voxel integrals recovers the
full-sky reference. It does **not** prove cellwise angular convergence. The
maximum binned p95 NSIDE1024-to-2048 relative change is33.2% (p99 62.3%, max
89.1%) for population5, radial shell1, center radii15–30 cMpc/h; several
outer bins have binned p95 changes12–24%, with p99/max reaching100% where one
quadrature has zero exposure. NSIDE1024 records160 active zero-ray cells;
NSIDE2048 records8. Therefore NSIDE2048 is a no-skip production artifact but
is not yet certified as a sufficiently converged cellwise operator.

Against the old order-six map, the all-grid summary counts25,773
population-shell-cell entries that were zero before and positive now, and
zero entries that were positive before and zero now. This is not training
support or a count-weighted score: the construction intentionally did not
read keys or counts. A training-only audit is required before interpreting
the large relative changes or selecting any finer angular rule.

The output is only for the N128/384-cMpc/h (3-cMpc/h) count-grid selection
operator. It adds no observational resolution, does not calibrate survival,
galaxy bias, group inclusion or covariance, and creates no count likelihood,
posterior, z=0 density/velocity field, or IC. R2 stays NO-GO.

## Next bundled check

Fable5 reviewed this material resolution discrepancy read-only and returned
CONDITIONAL PASS; the driver adopted the bounded audit and added a data-free
interior-geometry closure check. Recompute the full active-cell NSIDE1024
operator, close it, write and verify its SHA256, and only then open the frozen
`train_keys`/`train_counts`. Compare shell-summed old order-six, NSIDE1024 and
NSIDE2048 exposure at those keys. Any occupied zero exposure is a hard support
failure: do not floor/smooth it or omit the row, and mark the total count
weighted delta undefined if support differs. Save exact key/cell coordinates,
radial-edge crossings, ray hits, and classify zero support as outside the
selection domain, a positive-volume domain sliver missed by finite rays, or a
ray-hit cell with zero population exposure. Heldout/all-key arrays, fits,
posteriors, PM evolution and simulation remain out of scope.

Decision thresholds are frozen in the source before count access: per
population, `abs(delta log L) < 1 nat`, no cell with
`abs(log(E2048/E1024))*sqrt(Ncell) > 0.5`, and less than1% of training counts
above0.1 by that metric. This is only a pairwise NSIDE1024/2048 proxy, not a
direct NSIDE2048 error bound. A failed gate may motivate NSIDE4096 only for
the entire relevant geometry-defined edge class, never selected individual
cells and never a default full-grid 4096/8192 sweep.

Q-GOAL: a selection-denominator diagnostic for the same CF4-conditioned R2
field, not a field/posterior result; the operator starts at5cMpc/h and does
not identify the LG. Q-LEAN: one data-free all-active-cell lower-resolution
artifact plus one sparse training-only pass, with no heldout exposure. R2
remains NO-GO. MW/M31 roles remain ambiguous and M33 unresolved; their
observables must constrain those same roles on the same NEW evolved LG field
at `<=0.3 cMpc/h`, with native truth IDs reserved for calibration/evaluation.

MW and M31 identities remain role-ambiguous and M33 remains unresolved. Their
observables must eventually constrain those same latent roles on the same NEW
evolved Local Group field at `<=0.3 cMpc/h`; native truth identities are
calibration/evaluation-only and may not seed or select candidates.
