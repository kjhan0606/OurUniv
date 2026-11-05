# Fable5 advice on the ray-exposure result — 2026-10-04

Verdict: **CONDITIONAL PASS** for one training-only impact audit before any
fit or posterior. This is advisory; the driver compared it with the saved
result, the observation operator and the project master plan. Fable inspected
the committed plan, small JSON result, and output-writing section of the
integrator; it did not open the HDF5 or any count/key archive.

## Evidence and adopted amendments

1. The completed result stores only the NSIDE2048 exposure cube. The
   NSIDE1024 cube was held in memory for aggregate cellwise summaries and was
   not saved. A cellwise comparison therefore requires recomputing the
   NSIDE1024 operator.
2. Compare the quantity the count likelihood consumes: the sum of all six
   radial shells for each population and cell. Per-shell differences can
   cancel exactly at internal radial edges and can overstate the error in the
   actual count operator.
3. Freeze the numerical decision thresholds in source before reading any
   counts. Per population, the suggested negligible rule is
   `abs(delta log L) < 1`, no training cell with
   `abs(log(E2048/E1024))*sqrt(Ncell) > 0.5`, and fewer than1% of training
   galaxies in cells above0.1 by that same shot-noise-scaled measure.
4. Treat every occupied zero-exposure training cell as a hard support failure.
   Never floor, smooth, or omit it. Distinguish a data-independent window /
   domain mismatch from a positive-volume geometric sliver missed by finite
   angular rays.
5. Compute and close the data-free full-grid NSIDE1024 artifact first, then
   hash and freeze it, and only afterward read frozen training keys/counts.
   NSIDE1024-versus2048 differences are a proxy for NSIDE2048 error, not a
   direct bound; report that limitation.
6. Only if the predeclared support or count-weighted resolution conditions
   fail, refine NSIDE4096 by the entire geometry-defined edge class containing
   affected training cells, not by selected cells. Keep heldout arrays
   untouched. Do not run full-grid NSIDE4096/8192 by default.

Driver adopts all six for the next bundle. The exact, independent interior
geometry check from saved `geometry_shells` is also included: for the
870,528 cells wholly inside `5<r<180`, the shell-summed pure-geometry channel
should be1 within angular-quadrature error. This uses no catalogue/count
information and directly diagnoses local angular integration without another
field or gravity calculation.

## Scope decision

Q-GOAL: aligned only as the selection denominator for the same CF4-conditioned
R2 count operator at N128/3cMpc/h. It creates no field, posterior, zoom IC, or
Local Group identification. Since the exposure starts at5cMpc/h, it does not
condition the MW/M31/M33 region. Those roles remain MW/M31-ambiguous and
M33-unresolved; their observables must later constrain those same latent
roles on the same NEW evolved LG field at `<=0.3cMpc/h`, with native truth
identities limited to calibration/evaluation.

Q-LEAN: one data-free all-active-cell NSIDE1024 precompute plus one sparse
training-only pass is proportionate. A direct decision uses likelihood-level
shell sums and fixed thresholds, not an unweighted maximum over every sparse
cell. If the rule passes, freeze NSIDE2048 and return to the genuine R2
blockers: selection/survival/bias calibration and a defensible shared-latent
count/mark law. If it fails, refine only the full geometry edge class needed
to fix it; no general pixel-resolution ladder.

The audit's memory note is accepted: the previous Slurm MaxRSS was3.78GiB for
an8GiB request. The result JSON's24.78GiB theoretical sum multiplies
per-child peaks as if all workers peak simultaneously and is not the observed
concurrent memory; preserve both values and label them correctly.
