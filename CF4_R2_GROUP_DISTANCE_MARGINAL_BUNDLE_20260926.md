# R2/5: joint group-distance and global zero-point integration

Autonomously authorized next implementation bundle. No new gravity run.
Use the same source-linked10,020 FP rows and existing N128/384 PM state.

At fixed field F, source group g and global calibration b, compute

`C_g(b,F) = integral dd w_g(d) R_g(Z_g|d,F) product_i L_i(eta_i(d)+b)
            / integral dd w_g(d) R_g(Z_g|d,F)`.

Then integrate `product_g C_g(b,F)` over ONE global b prior. Do not integrate
b independently per galaxy/group. Z contains the source group redshift and
unique secure eligible2M++ member redshifts. Their correlation enters the
supplied covariance; a repeated datum ID is rejected. R is conditioned on,
not multiplied into the distance score again. A heldout predictive factor is
the all-group marginal divided by the training-group marginal, sharing b.

Implementation controls use provisional common-COM scatter100 km/s,
member scatter150/300 km/s, group mean/member covariance sigma^2/NgroupT17,
and extra group-catalogue scatter50 km/s. Published member measurement errors
are included at the reference group redshift. These are sensitivity models,
NOT calibrated actual CF4/Tempel/2M++ covariance estimates. The caller-facing
kernel takes a general covariance rather than hardcoding this trial law.
Use a declared r^2 dd radial prior, not a claim of calibrated selected-group
selection. Its fixed +/-40 cMpc/h interval is clipped only to positive distance
and the actual cube boundary. Compare257/513 nodes and record edge mass.
Refine the common zero-point quadrature129/257 nodes across +/-8 prior SD
alongside the distance quadrature: a sparse prior-centred GH rule can miss
the narrow global calibration peak after thousands of observations.

Howlett et al.2022 section5.4, https://arxiv.org/html/2201.03112, gives relative
zero-point uncertainty0.004 dex, excluding CF3 anchor uncertainty0.0116 dex
and separately discussed cosmic variance. Evaluate0.004 and their quadrature
sum as sensitivity only. Do not add the paper's cosmic-variance term as an
independent noise if already represented by the latent field. The source FP
fit used its whole sample; these splits are not independent validation.

This conditional kernel does NOT by itself complete the joint model with
coarsened2M++ counts: the within-cell location law, association/selected-group
prior and physical covariance need justification. Ambiguous cross-group
redshift associations are retained in the report, not duplicated across
independent groups. No actual field inference or production promotion here.

Q-GOAL: handles shared distances/velocities and calibration in the actual-data
observation model required by the z=0-first science delivery. Q-LEAN: one
small implementation, two focused tests plus prior regressions, eight bounded
quadrature/sensitivity evaluations on an existing state, no sampler or new
validation framework. H200/H100/A100 checked; typed H200 chosen.2 CPUs,
16GiB (estimated peak<=13GiB plus margin),20-minute Slurm cap.

MW/M31/M33: this global FP bundle does not identify them. R3 uses candidates
from each NEW evolved field with MW/M31 role ambiguity and unresolved M33
explicit; their observed positions/masses/velocities constrain that SAME
field. Observational group labels here never seed generated LG identities.

Execution note:406004 passed all four focused/reused tests, then stopped in
source ingestion because unmatched crossmatch rows have blank 2M++ recnos.
The reader now checks match class and presence before integer conversion;
no unmatched record becomes a velocity datum. No numerical result was produced.

## Completed result and driver decision

Corrected H200 Slurm406005 COMPLETED/exit0 in1m49s; all four focused/reused
tests pass. Batch MaxRSS2,711,528 KiB. Numerical control time48.7s; no fit
or gravity evolution. Output is
`/gpfs/kjhan/CF4/z0_density/r2_fp_group_marginal_v1/result.json`, with
`group_factors.npz` retaining baseline per-group factors across the shared
zero-point nodes for reuse.

All10,020 FP rows enter6,821 **source Tempel/singleton groups**, not the5,450
CF4 groups from the previous bundle. Source-group split5,331/1,490 preserves
the prior group-closed allocation. Secure conditioning uses3,062 distinct
2M++ member redshifts in2,413 groups; zero ambiguous cross-group recnos.
No separate member-redshift factor is added to the conditional mark score.

The four fixed sigma/zero-prior cases all have finite results at both joint
quadrature resolutions. Maximum train log-ratio change0.0004234; maximum
all-group change0.0025020. Baseline field-amplitude derivative-45.89623900
agrees with finite difference-45.89623894. The largest fine-grid endpoint
quadrature weight is1.69e-6; this is NOT a rigorous bound on unintegrated
radial tails. The baseline source-group factors are saved, not a fitted field.

Fine-grid train log ratios against zero peculiar velocity range-24.5407 to
-26.0229. All-group ratios range-33.7446 to-35.9540. This unconditional PM
phase is not favored over the zero-velocity reference under these trial
observation laws. It is not evidence against the latent-IC reconstruction
route, a recovered IC, or calibrated likelihood evidence between physical
models. No velocity amplitude or zero-point was fitted to improve a score.

Accept the **conditional group integration implementation**, including one
global zero-point integral and shared-redshift normalization. Do not equate
that denominator with a complete joint2M++/CF4 likelihood: coarsened counts
do not already specify the full within-cell redshift/location distribution.
The trial COM/member/catalogue dispersions and r^2 radial prior are explicitly
uncalibrated; the narrow numerical sensitivity range does not certify them.
The next substantive task is to give these inputs a source-backed membership,
selection and covariance model (or source-mock calibration), and connect the
within-cell conditional law before actual posterior sampling. Do not restart
generic group-residual sweeps, cold-root tuning or a new gravity simulation.
