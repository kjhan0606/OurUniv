# R2/5: source-owned coarsened counts with live CF4 marks

R1 mechanics -> **R2 actual present-field inference, unfinished** -> R3 LG
conditioning -> R4 precise forward validation -> R5 zoom ICs.

## Target and observation ownership

Evaluate on one fresh N128/384 prior IC and its *same* evolved z=0 matter and
velocity field:

`p(C,U,A,M | F) = p(C | F) q(U | C) q(A | C,U) p(M | C,U,A,F)`.

`C` is the six-population full eligible 2M++ N128 count map. `U` carries
each observed point's within-cell position/redshift; `A` is the preserved
association/selected CF4 source-group graph; `M` comprises selected FP
source distance PDFs and linked non-FP distance moduli. The two q factors
are CONDITIONAL DESIGN APPROXIMATIONS: we do not know whether within-cell
locations, crossmatch completeness and CF4 group selection are independent
of the field. Their normalizers cannot be silently promoted to an exact
survey likelihood. The source FP fit itself used the full source sample,
limiting independent holdout interpretation.

Use the **full eligible**57,238 2M++ points once, projected to45,776
population/cell counts, with the order-six voxel-integrated ARES selection.
Do not apply the historical CF4-exclusion survival fraction, a `0.6` thinning
factor, or a pointwise map selection that has an actual zero-support object.
No arbitrary intensity floor. The coarsened count score has the empty-cell
expectation and a six-population Gamma rate integral under the existing broad
development prior. Published six linear-regime biases and inherited RSD/FoG
widths are held fixed as a sensitivity reference; they are NOT calibrated for
the nonlinear PM model. Total-count amplitudes are NOT set from observations.

The CF4 term is the prior bundle's live shared-group conditional mark factor:
one source group distance and FP excess offset, global calibration and
non-FP method offsets, with selected radial density response and the source
group+2M++ member redshifts inside ONE joint kernel. It contributes one
conditional mark score per training source group. The member redshifts are
observed covariates here, not a separate velocity likelihood. Group presence,
association and source fit covariance are not given an independently
calibrated likelihood. No inclusive-count × historical BGc product.

The native PM density/velocity mesh is node-centred. The galaxy count data
and integrated selection are voxel-centred. Interpolate native *mass and
momentum* to the galaxy cell centres with origin0 reads, divide to form the
cell velocity, then use the count operator's cell-centred origin0.5. This
keeps mass/momentum and sky geometry consistent without translating the
physical observer or redoing gravity. A constant/one-node unit test checks
mass and momentum conservation. The CF4 source rays keep their origin0
native reads.

## One bounded implementation and numerical decision

Reuse the source-bound inclusive counts, same-cosmology shell selection,
live hierarchical geometry, PMWD forward, count RSD/FoG kernel and proper
rate marginal. One fresh prior draw; no posterior chain, seed search, fitted
offset or legacy frozen-field cache. Compute the count, mark and prior parts
and their directional gradients through the same IC. Compare their small-step
finite differences after subtracting the analytic Gaussian prior tangent.
An occupied zero-intensity cell or a nonfinite count/mark gradient stops the
pilot and is reported as a model/support failure, not repaired by an epsilon.
Store scalar report and the specific initial draw/observation source hashes;
do not save another large full field for a one-state control.

One H100 GPU job,4CPU,16GiB host memory (estimated peak<=13GiB plus20%),
20min cap, all tests/calculation via Slurm. Prior comparable PM+marks host
peak7.44GiB; the six-count RSD/adjoint adds multiple full-grid arrays, so
this larger bound is intentionally provisional. Preserve prior outputs.
No external audit for this routine bounded calculation.

Q-GOAL: this is the first live same-IC numerical target with both the bulk
2M++ density counts and the source CF4 distance marks. It reveals whether
the chosen conditional law and native/voxel coordinate contract can be
evaluated and differentiated. It is a development target, not yet the
calibrated N256 present-field posterior.

Q-LEAN: reuse source files and tested operators, one small conservation test
and one actual-state gradient/control; no new simulation, long sampling,
count-rate fit, selection-map patch or diagnostic ladder.

MW/M31/M33: the N128 voxel map does not resolve them. R3 must enumerate
candidate bound components from each NEW evolved state, retain MW/M31 role
ambiguity and missing/shared M33 cases, and let their observations constrain
that SAME state under a normalized role/detection law. Native truth IDs may
calibrate/evaluate only. This bundle contains no LG role selection or
historical zoom seed reuse.

## Execution

Implementation pending.
