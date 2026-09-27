# R2/5 — secure singleton TF/2M++ point conditional

The TF-only group source has8,502 groups,95.01% with catalogue `Ngal=1`.
Of the full source,3,458 groups have at least one secure 2M++ count-point
link, and their group/point CMB redshifts differ by median9/p90 115km/s.
The current TF factor conditions on CF4 group cz for all groups while the
count factor already uses the linked point's observed redshift-space cell.

Build a stricter source bridge: exactly one secure edge, exactly one edge
total, and catalogue `Ngal=1`. For these groups, condition the distance
mark on the **counted point's observed cz**, never multiply the group cz as
a second independent velocity datum. Tie their radial-density exponent and
redshift/FoG width to that point's existing 2M++ count population. Keep
unmatched/ambiguous groups' original source cz and provisional TF-specific
response, visibly labelled. Preserve the original TF train/holdout split and
the single shared relative TF modulus zero from FP anchors. No truth-label
matching, new gravity or source distance rebiasing.

This is a more explicit conditional *development* model, NOT a complete
joint likelihood: selecting a TF measurement among count points, shared
source covariance and within-cell mark-position dependencies remain. Compare
old/new factor and IC derivative at the same predeclared state; no sampler
until that numerical control and source limitations are assessed.

Q-GOAL: removes an avoidable observed-redshift ownership ambiguity in the
CF4+galaxy field target needed for R2. Q-LEAN: one existing secure graph,
one vector-parameter extension and one fixed-state derivative control, not
a new selection-fitting ladder or long HMC run.

MW/M31/M33 stay R3 latent roles on each NEW field, with MW/M31 ambiguity and
possibly unresolved M33 retained. Their observations must constrain that
same field; native truth cannot seed or select candidates. N128/3cMpc/h is
not an LG member-resolving result.

## Execution and result

Typed-H100 source bridge **406494 COMPLETED/exit0**: exactly3,108
secure one-edge `Ngal=1` TF-only groups bind to distinct counted 2M++
points (2,484 training,624 heldout), spanning all six count populations.
The remaining5,394 groups retain original CF4 group cz. Matched absolute
group-minus-point redshift differences have median5/p90 85.3/p99 276km/s.
Pinned catalogue/report:
`/gpfs/kjhan/CF4/z0_density/r2_tf_matched_point_bridge_v1/`.

The first linked-control H100 **406495 FAILED after19s in its focused
vector-parameter unit test**, before any gravity or result output. The
per-group width was correctly reshaped for quadrature, but its `(G,1)`
validity mask broadcast a `(G,)` score into `(G,G)`. A focused correction
squeezes the width solely for the final group-validity mask; the vector
test must pass before the numerical control runs. This is an implementation
error, not a scientific failed posterior or reason to alter observations.

Corrected typed-H100 **406496 COMPLETED/exit0 in3m52s**, with all three
focused TF tests passing. The linked count+FP+TF factor reads one predefined
evolved IC; the2,484 securely matched TF training groups use counted-point
cz and their six-population count response, while4,261 unmatched training
groups retain the clearly provisional TF response. At this one IC the TF
log-ratio changes by-9.3879nat relative to the earlier all-group-cz factor;
count and FP factors are unchanged. The TF IC directional reverse gradient
agrees with finite difference to relative3.19e-8; count occupied support is
positive. Pinned control:
`/gpfs/kjhan/CF4/z0_density/r2_live_tf_matched_point_control_v1/result.json`.

**Numerical/source-link pass only.** A single-state -9.39nat change does
not show posterior robustness, calibrated selection, or correctness of the
count-linked population model. For the next sampler-development run the
unmatched/ambiguous TF groups will be excluded rather than kept as a
provisional training likelihood. The existing FP+count factors remain.

Explicit residual mismatch for interpretation: matched TF groups share the
count population's **FoG parameter**, but the TF conditional redshift kernel
does not separately convolve the count model's fixed35–50km/s redshift-error
term; its selected radial measure also omits the 2M++ exposure and the
probability of receiving a TF measurement. Therefore the current linked
factor is not algebraically identical to the full marked point-process
conditional, even for secure one-to-one groups. This is recorded before
the development-chain verdict; do not promote its source model solely from
sampler stationarity.

Post-chain scoped correction: the matched conditional now uses the same
population's fixed redshift measurement error in quadrature with its FoG
width. H100 **406514 COMPLETED/exit0 in4m04s**, all four focused tests pass,
including vector/scalar Gaussian-convolution identity and invalid-width
rejection. At the **same predeclared IC** as406496 the TF factor changes by
+1.85127nat; count, FP and white-prior factors are unchanged. The TF IC
directional reverse/finite-difference discrepancy is9.81e-8. Frozen406501
chains remain from the preceding target and are not retroactively corrected.
This numerical fix does **not** calibrate the count/TF true-distance
selection, shared source covariance or group inclusion. Output:
`/gpfs/kjhan/CF4/z0_density/r2_live_tf_matched_point_control_v2_redshift/`.

## Public individual-TF source option (not a calibrated replacement)

The user directed **no email** for the Qin et al. CF4TF mocks. No author
request was sent. A public, smaller source path exists: CDS/VizieR
[J/ApJ/902/145](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/902/145)
provides Kourkchi et al. (2020) `table1` with 10,737 individual PGC IDs,
H I linewidths/flux proxies and optical/WISE magnitudes, and `table4` with
9,792 published distances. Its `ReadMe` gives fixed-width definitions and
flags. The exact files were downloaded from the CDS `/ftp/cats/J/ApJ/902/145/`
archive to `/gpfs/kjhan/CF4/source_catalogues/J_ApJ_902_145/`; both gzip
streams pass integrity checks and have the documented record counts. SHA256:
`ReadMe` 45fa3cca5346aff29bcf9785e7b08e8ab34b1d67e80b359c0d28bdbd18e42581,
`table1.dat.gz` 59a16536ff62aa63729f1b1a6fc11155b0f63613e80ebd8c3469966385a64b0c,
`table4.dat.gz` 7199960877ccd38b4f499b342e9732951d32e7bc643bb90a08211ab524de7246.

[Boubel et al. (2024)](https://arxiv.org/abs/2301.12648) derive a
forward conditional likelihood for individual TF apparent magnitude given
linewidth, flux, observed redshift and a velocity prediction; a magnitude-
dependent selection factor appears in both numerator and normalization.
This suggests a *possible* source-constrained alternative to treating every
grouped `DMtf` as an independently selected Gaussian distance. It does not
license simply inserting their fitted velocity parameters or selection curve
into this project: their analysis conditions on a fixed redshift-space
velocity model, whereas ours infers the same evolving IC/field as the count
data; their sample, photometry availability and group aggregation may differ
from the CF4 `DMtf` rows.

Next bounded source task: determine exact PGC-level overlap among this
individual catalogue, the frozen CF4 TF-only group membership, and the
secure 2M++ counted points; preserve unavailable/ambiguous associations.
Only if that bridge and the paper's observable/selection definitions cover a
useful disjoint training subset should one field-dependent conditional-TF
factor be implemented and checked on untouched heldout galaxies. Do **not**
multiply it by the existing group-`DMtf` factor for the same measurements,
call a selected-only overlap an inclusion denominator, or claim this TF-only
option calibrates the FP/group source model. Q-GOAL: directly probes the R2
CF4 selection barrier. Q-LEAN: reuse a 1.1-MB public source table and one
bounded bridge before considering a new likelihood, rather than procuring
large mocks or expanding the sampler blindly. MW/M31/M33 remain unresolved
R3 latent roles on each NEW field, including an unresolved-M33 branch;
native identities cannot seed or rank generated states, and their
observables must eventually constrain that same evolved field.

The first PGC-only bridge **406523 FAILED/exit1 after4s** before writing a
result because it assumed every CDS `table4` line had exactly100 characters;
the archive omits trailing absent columns on some rows. The parser now
accepts these variable-length rows without changing source bytes or the
specified PGC/`DMbest` columns. Same-scope typed-H200 **406524
COMPLETED/exit0 in1s**. Its frozen
[`result.json`](/gpfs/kjhan/CF4/z0_density/r2_public_tf_source_bridge_v1/result.json)
shows 7,020/8,502 TF-only groups have at least one 2020 raw-TF member and
6,924 have a 2020 distance member. Of the **secure counted-point singleton**
groups, the exact CF4 edge member appears in the raw source for1,801/2,484
training and472/624 heldout groups; the 2020 distance table covers1,792
and470 respectively. Thus a useful public overlap exists, but it is neither
complete CF4 TF membership nor the missing 2M++ parent selection law.

For the directly comparable individual member rows, absolute current-CF4
`DMtf` minus 2020 `DMbest` is median/p90 **0.125/0.345 mag** in training and
**0.135/0.355 mag** in heldout (1,792/469 finite comparisons). These are
*differences of published products*, not measurement residuals, calibration
errors, or evidence that either catalogue is wrong; source-vintage, modulus
zero, estimator/selection and group summarization may all contribute. The
bridge therefore does **not** authorize replacing all group marks with raw
TF observables, treating the 2020 `DMbest` as an independent datum, or
mixing overlapping marks in one target. A prospective forward-TF factor
would need a disjoint ownership contract and its own selected-sample mock/
heldout calibration. FP/group selection remains a separate R2 barrier.
