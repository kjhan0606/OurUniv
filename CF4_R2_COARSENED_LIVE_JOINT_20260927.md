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

The first H100 Slurm run, 406376, passed the native-node/voxel conservation
test and evaluated the actual-data joint value and reverse gradient on its
fresh prior IC. All 45,776 occupied population/cells had positive unit-rate
intensity. The run then FAILED in the *diagnostic*: PMWD's evolution uses a
custom VJP and cannot be called through JAX forward-mode `jvp`. Its partial
score and support readout are preserved in
`/gpfs/kjhan/CF4/z0_density/r2_coarsened_live_joint_v1/result.json`;
neither a physical support failure nor a derivative pass is inferred from it.

Retry 406377 uses the same data, IC seed and physical target in a new `v2`
directory. It obtains the mark tangent by reverse differentiation, takes
the analytic white-prior tangent, and derives the count tangent by subtracting
those from the full reverse gradient. Each count/mark tangent is then compared
with a separately evaluated symmetric finite difference. The identity of the
decomposed sum is not counted as an independent test. This is a scoped
technical recovery, not a new model, posterior or approval to run N256.

## Result and decision

H100 Slurm **406377 COMPLETED/exit0** in 3m53s. The conservation test passed;
the numerical report is
`/gpfs/kjhan/CF4/z0_density/r2_coarsened_live_joint_v2/result.json`.
All 45,776 occupied population/cells have positive model intensity with
zero diffuse/floor component. At the frozen unconditional IC, log factors
are count −279,661.856, conditional CF4 mark +1,391.895, and Gaussian
white prior −1,051,731.380. These absolute factors have different
normalizations and are **not** a goodness-of-fit ranking or a fitted field.
Full-IC/mark gradient RMS is finite (2.25934); the count directional
reverse/finite-difference discrepancy is 0.1324% at step2e-5 and 0.0946%
at step1e-5, while the mark discrepancy is 0.000124%/0.00696% respectively.
PMWD's custom VJP was respected; the count reverse tangent was decomposed
from the full target and independently checked against finite differences.
Host peak was 5.083GiB (16GiB requested), and first value+gradient including
compilation took 75.39s. No chain or z=0 sample map was generated.

Rate-one count exposure sums to 757,343.745 over six populations. Multiplying
each population by its *published prior-mean* rate yields totals
`[5952.64,9842.12,2420.36,6989.20,18464.97,4941.55]`, sum ~48,610.84
versus 57,238 observed. This one random state and the broad integrated Gamma
rates do not calibrate bias, selection, FoG or a posterior. In particular,
the fixed nonlinear `rho^b` response is a development law; its published
linear-regime biases must not be treated as measured effective N128 biases.

Decision: **PASS numerical same-state partial-target wiring; R2 posterior
remains NO-GO.** The missing field-dependent point/association/inclusion
probabilities and calibrated tracer/RSD/group covariance remain substantive,
not numerical, omissions. Next scientific step should use this live target to
quantify actual count–CF4 field tension and selection/bias sensitivity before
any N256 inference; a longer chain or a nominally finer grid alone would
not resolve those omissions. MW/M31/M33 identification and <=0.3-cMpc/h LG
constraints remain R3 work on each new evolved state, including ambiguous
MW/M31 and unresolved M33 branches.
