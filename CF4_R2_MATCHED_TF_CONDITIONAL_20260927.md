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
