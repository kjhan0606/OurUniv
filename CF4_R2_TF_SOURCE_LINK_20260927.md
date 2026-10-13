# R2/5 — corrected CF4 TF-only distance groups on the same current state

R2 is unfinished: the N128/384 count+SDSS-FP development chains drift and
their maps disagree. Longer HMC or N256 is not the immediate remedy while
the all-sky TF information and selected-group law are absent. This bundle
adds an *observation component*, not a posterior fit or a new gravity run.

The [CF4 group table](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allgroup)
provides corrected group `DMtf`, its reported `e_DMtf`, method counts and
CMB-frame group redshift. The [CF4 TF source study](https://academic.oup.com/mnras/article/511/4/6160/6523366)
describes H I flux selection and the corrections incorporated in its final
distances. Use the reported **group-level** corrected TF modulus once, not
all constituent passbands, a repeated bias polynomial or the combined `DMzp`.
The Gaussian summary is an explicit approximation to the published group
product, not its exact per-galaxy measurement likelihood.

Choose native CF4 groups with TF and no FP/SN/calibrator/SBF/SNII method
contribution; exclude any group whose member PGC or Tempel T17 component
overlaps the existing source FP likelihood. Close the frozen native holdout
across any remaining shared T17 components. Do not treat TF/FP group IDs as
generated halo labels. Retain the 15–180 cMpc/h observed-redshift support.

At one archived **unconditional** N128 PM density/velocity state, evaluate
one normalized group conditional distance-mark ratio. Use the same native
origin0 mass and velocity field at each distance node. The numerator has the
corrected TF group Gaussian modulus; numerator and denominator share the
observed group-redshift kernel and radial `d² rho^b` measure. The trial
`b=1`, Gaussian group velocity width150 km/s, and absent group-inclusion
factor are deliberately **uncalibrated**. They are not fitted to the
unconditional field. Compare 257/513 distance quadrature and one velocity-
amplitude derivative to check implementation only. Save source identity,
support and split arrays for later calibration and live-state coupling.

One typed-H100 Slurm job,4CPU/16GiB/20min; the source catalogue is small,
and the peak array construction is roughly two to four million group-distance
points. No data download, new simulation, HMC, N256 or raw snapshot. Two
focused source-factor tests run in the same job. Preserve every old output.

Q-GOAL: TF-only groups cover an important missing CF4 method and sky region,
so their observables can eventually constrain the **same** current density
and velocity posterior as the galaxy counts and FP marks. The bundle makes
their ownership and numerical field connection explicit; it cannot alone
deliver R2 or calibrate group inclusion, source covariance, TF systematic
scatter and tracer bias/FoG.

Q-LEAN: reuse pinned CF4 groups/native split, existing PM state, distance
measure and CIC reads. No TF retraining, second survey download or new test
framework. A failed overlap/support check stops this route rather than
silently thinning the data.

MW/M31/M33 are not resolved by this N128 field. At R3, candidates must come
from each NEW evolved field without native truth IDs, preserving MW/M31 role
ambiguity and shared/missing M33 branches. Their measured positions,
distances, masses and velocities must constrain that same field under a
normalized role law; a CF4 group ID here does not supply that law.

## Execution and decision

Typed-H100 Slurm **406479 COMPLETED/exit0 in25s**. Both focused tests pass;
process peak is1.271GiB under16GiB requested. The pinned native CF4 split
provides8,502 TF-only groups, of which6,745 are training and1,757 heldout;
they summarize9,124 published TF measurements. There are1,019 groups with
another CF4 method,513 outside the native/broad velocity window, and one
outside the exact comoving-distance interval. No candidate is missing its
published TF modulus/error, and no selected TF-only group shares a PGC/T17
source component with the existing FP source. A separate read-only check also
finds **zero** overlap with the114 existing non-FP FP-group anchors by CF4
group ID. Preserve mixed-method groups for a future shared-distance law;
do not multiply their combined `DMzp` or silently append them here.

On the archived *unconditional* N128 PM state, the TF conditional mark
log-ratio is29,293.982 training /7,337.208 heldout relative to a fixed
35-mag data reference. The absolute ratio is **not** a fit statistic or
evidence comparison. Changing only the velocity amplitude has training
derivative-519.68006577 versus finite-difference-519.68006559. Q257→Q513
changes the total training ratio by-0.0397 nat and the largest single group
by0.00870 nat; this is adequate for this wiring control, not an accuracy
certification across inferred fields. Exact numerical output and the
disjoint source catalogue are in
`/gpfs/kjhan/CF4/z0_density/r2_tf_source_link_v1/`.

Decision: **PASS source ownership and same-state numerical TF wiring;
NO-GO for calibrated TF likelihood, joint R2 posterior and N256.** The
partial Gaussian TF group factor can next be joined to the live count+FP
target, sharing the existing TF relative calibration parameter and using
each observed group redshift only *inside* its conditional numerator and
denominator. Before scientific promotion, bound the TF summary covariance,
group velocity/FoG, selected-group radial law and method/sky dependence with
source-consistent mocks or independent heldout data. The drifted HMC chains
are not rescued by this new term; sampler equilibration remains a separate
R2 requirement.
