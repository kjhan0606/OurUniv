# Fixed-center enclosed-moment cause test — 2026-09-16

Reuse the four archived endpoints from HOP job361459. Select AMR9 top32 HOP
groups >=1000 particles only to provide declared diagnostic centers, not LG
candidates or truth labels. At fixed centers, measure direct spherical enclosed
mass, weighted mean velocity, per-axis sigma and particle count at radii
.1875,.3,.5,.75,1,1.5 cMpc/h in every solver. HOP tags do not enter the
measurement. Compare AMR9 against CIC/TSC/AMR8 per center/radius.

If discrepancies shrink relative to HOP group masses, boundary membership is
implicated; residuals are dynamical/distributional. Fixed AMR9 centers are a
diagnostic reference and do not establish MW/M31/M33, M200c, boundness or LG
posterior. No threshold tuning, particle exclusion, new evolution or GPU.

Q-GOAL: distinguish HOP boundary effects from field dynamics before LG
observables. Q-LEAN: one readout, no new framework/simulation. CPU4/8GiB
(6.5GiB estimate+20%),20min, JSON only. Code implementation by driver;
Opus5 code audit required before closing this bundle.

Initial run362840 exposed a driver error in this newly added code: the AMR9
velocity array was accidentally passed as the mass argument when ranking
diagnostic centers. No physical result was consumed from that run. The call
is corrected to pass the archived particle masses explicitly; the corrected
run must replace362840 for interpretation. Requested Opus5 CLI audit was
attempted, but this environment rejected model name `opus-5`; the `opus`
alias produced no response and exited after timeout. Treat that as audit
unavailable, not approval; driver performs an additional source review and
records this limitation.

Corrected run362869 completed in23s (source7f31041). With centers correctly
ranked by AMR9 HOP mass, AMR9 enclosed-vs-comparator maximum/median absolute
mass differences are CIC11.15%/1.33%, TSC12.32%/.75%, and AMR8 7.73%/.32%
across the six radii and32 centers. Maximum mean-velocity vector differences
are24.98,12.10,13.46 km/s respectively. The reduction from HOP group-level
mass differences supports a substantial boundary-membership contribution,
but nonzero fixed-center residuals show that dynamics/distribution differences
remain. This is still a one-seed diagnostic and not a convergence certificate.

Opus audit using the safe `claude --model opus` read-only settings returned
CONDITIONAL PASS. It found no numerical bug in the corrected run, but required
like-for-like reporting: pooled enclosed statistics cannot be compared with
counts of HOP pairs, and AMR9-centered spheres are asymmetric. The code now
saves matched HOP-group mass differences and per-radius summaries for the same
32 centers, while retaining the AMR9-center bias explicitly. The previous
boundary interpretation is therefore provisional until this corrected readout
is executed and re-audited.

Fable bundle-close audit: **CONDITIONAL PASS**. Corrected run363094 completed
in45s from commitf9b8737; all32 centers matched reciprocally for each
comparator. Per-radius summaries are saved in
`/gpfs/kjhan/CF4/r1_enclosed_cause/job_363094/result.json`.

At r=.1875 cMpc/h, enclosed median absolute mass differences are CIC4.52%,
TSC5.10%, AMR82.41%, while matched HOP medians are2.34%,2.67%,1.02%.
At r=.3 they are1.85%,1.17%,.78%; at r=.5 they are1.32%,.91%,.33%; at
r=1.5 they are.68%,.29%,.09%. HOP maximum differences on this same sample
are47.65%,32.82%,16.33% respectively. Thus HOP tails can be boundary-sensitive,
but core-scale enclosed differences are not reduced below HOP differences and
are consistent with force resolution and matter-distribution differences.
The cause remains undetermined. This diagnostic bundle is closed as a
conditional solver-readout result, not as R1 completion or LG inference.

Fable bundle-close audit of run362967: **CONDITIONAL PASS**. The calculation
is sound, but the prior interpretation was too strong. At r=.1875 cMpc/h,
enclosed median absolute mass differences are CIC4.5%, TSC5.1%, AMR82.4%,
versus matched HOP medians2.3%,2.7%,1.0%; at r=.3 they are1.9%,1.2%,.8%.
Thus core-scale residuals do not support an HOP-boundary-only explanation;
force resolution and matter-distribution differences remain viable. The code
now records HOP maxima and reciprocal-pair counts and uses only reciprocal
matches. HOP groups remain diagnostic, with no MW/M31/M33 or posterior claim.
