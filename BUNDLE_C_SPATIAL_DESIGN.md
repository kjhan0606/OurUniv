# C-spatial — from actual LG marks to a joint present-day spatial model

User authorized entry2026-09-08. This is a new execution bundle WITHIN the
unfinished master Bundle C, not entry to D or a declaration that C succeeded.
Astra driver handles implementation/evaluation; external audits remain waived.

## Deliverable and order

1. Calibrate spatial components, not only catalogue marks: native MW-role and
   M31-role primary member profiles, M33-role satellite members, and the exact
   remaining total matter, with density/momentum/second moments. Keep bound
   member mass separate from host M200c and measure their relation.
2. Define the joint continuous spatial distribution using those components
   and their environments. Specify profile/shape scatter and remainder-field
   covariance/response. Test against retained native spatial cases. The old
   global remainder multiplier and a mark Gaussian are not that distribution.
3. Use the actual native CF4/count responses and identity-preserving LG model
   once each in that joint target. Compare LG-on/off maps and uncertainty with
   the same spatial prior, and disclose conditioning, prior content, sensitivity
   and achieved numerical versus information resolution. If the model cannot
   support this, report the missing physical component rather than paint a map.

Target stays surroundings1.5 cMpc/h, LG0.1875 cMpc/h, existing384 domain and
24-cMpc/h LG patch. No new IC/RAMSES, D, direct-CF4 route, finite-bank expansion
or halo-only mass-likelihood recycling. A saved LG posterior can be a proposal,
but cannot be used as an independent LG likelihood or combined with a changed
spatial prior without the appropriate density ratio/correction.

## First spatial calibration calculation

Reuse `lg_population_v1/population_and_marks.h5` native SATELLITE population,
not its posterior-selected catalogue neighbours. Select16 training and16 heldout
MW identities uniformly with fixed seed; then choose one eligible M31 and M33
uniformly in that order. Preserve source labels/rows. Different selected MWs
can share companions; read each distinct native subhalo only once and report
shared IDs. Train/heldout catalogue IDs are disjoint by the earlier native-x
split, but24-cube fields can overlap. These32 patches are calibration cases,
not32 independent universes or a new finite prior component bank. Do not fit
population frequencies from this small selected set; the existing full native
population supplies mark statistics. The satellite-only scope is conditional,
not a claim that the separate-primary case has been physically disproved.

The object-level split is suitable for separate MEMBER-PROFILE checks, not
automatically a whole-field ML holdout: a training patch may contain voxels
of a retained patch. Before learning/testing a remainder-field distribution,
exclude shared native voxels (including periodic wrapping) from its heldout
score or choose a genuinely separated spatial subset. Do not report the
32-patch labels as leak-free full-field validation without that separation.

For each unique selected subhalo stream gas/DM/stars+wind/BH DYNAMICAL mass
members using existing catalogue offsets. Type4 wind mass belongs in total
matter, unlike the stellar-COM proxy where it was excluded. Check bound mass
and COM against native SUBFIND. Record cumulative mass-radius quantiles and
the actual total3D velocity covariance, not an assumed NFW profile or invented
observational mass error. Read at most50m selected particle/cell rows, through
one I/O-only SSH source on syntax; no raw full-snapshot pass or source copies.

Use a native aligned24 cube around each MW-role primary,128^3 at0.1875.
Subtract the three DISJOINT native member components from the full native
total-matter moments. The remainder includes other halos, group fuzz and
non-FoF matter; do not call it pure diffuse matter. Never subtract an inclusive
host M200c sphere and then subtract its satellite again. Check positive/
realizable residual moments, original reconstruction, and1.5 restriction.
Save sparse native components plus full remainder and coarse total moments,
so all native matter is reproducible without re-reading raw particles.

No frame rotation/interpolation, component transport or observation fitting
in this calibration job. Native axes and periodic source geometry are explicit.
Diagonal grid second moments do not determine a general rotated dispersion
tensor. Scalar trace and vector momentum can be rotated later; anisotropic
rotation requires the missing cross-moment field, not fabricated zeros.
The native catalogue is valid only for these unchanged source realizations.

Budget: Slurm2 CPUs, estimated4000 MiB peak+20%=4800 MiB,30min cap,
<8 GiB output in `bundle_c_v1/spatial_calibration_v1`, source subset<=50m rows.
No new simulation, download, storage test, process scan or audit micro-stages.
Reuse the existing moment/transport checks. Completion of this first job is
DATA/PROFILE CALIBRATION INPUT, not completion of this spatial bundle. Review
the actual particle read/mass/COM/remaining-matter report, then implement the
joint spatial distribution; do not substitute another mark-only fit.
