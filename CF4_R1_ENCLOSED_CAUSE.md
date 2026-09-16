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
