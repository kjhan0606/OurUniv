# C repair proposal: member learnability first, stable denoising second

2026-09-11. DESIGN ONLY, requested by the user after frozen diagnosis338746.
No new calculation or model patch is authorized by writing this proposal.
The next implementation bundle awaits user approval after Fable5 plan review.

## What actually failed

338746 completed2026-09-10 18:42:44 KST,36s allocation,19.7s application,
zero optimizer steps. Raw and EMA models give epsilon RMS about.005 versus
target RMS1, essentially zero correlation, MSE about1 on both tested scales.
At t100 the algebraic noisy/sqrt(1-alpha_bar) reference has MSE2.4e-7.
The unchanged reverse chain grows from RMS1 to2032; the first mass split has
2707/4096 active fractions outside floating-point support. This explains
generation failure, NOT its unique learning/architectural root cause.
Training/checkpoint normalization is identical. Continuous-head gradients are
nonzero; plain/checkpointed relative gradient difference is9.9e-10. No missing
parameter gradients. Early stem gradient RMS4.4e-8 versus head5.35e-4 suggests
poor learning signal upstream but does not prove category competition or a
particular fix. The categorical probe is99.923–100% majority class; its low
loss is not evidence of learning cosmological density structure.

Do not resume the30k checkpoint, increase training length, clip invalid latent
values, floor fractions, or alter power-spectrum amplitudes. Preserve failures.
The confirmed outcome is failed denoising; changing the model is an engineering
hypothesis, not a discovered one-line software bug.

## Proposed corrections and order

Two unresolved requirements are distinct: (a) infer explicit LG members from
the same field, (b) learn a useful spatial field prior. Do not reclassify a
successful denoiser as either member identification or a CF4 posterior.

### 1. Next executable bundle: ONE small native member-readout learner

Purpose: establish whether the available.1875 total mass/mean-velocity/three
dispersion fields carry learnable member separation beyond a position-only
selected-population template. Unlike the closed peak/proxy/budget probes,
this produces soft component MASS MAPS, not another peak list or mark score.
It need not find a separate M33 maximum. Do NOT train another total-field prior
inside this bundle, even if this pilot passes. This explicitly keeps the prior
member-learnability-before-further-field-training requirement.

Reuse existing `spatial_calibration_v1/spatial_components.h5` native disjoint
bound-member moment targets and total fields. Same fixed13 training fixtures
[0,2,3,4,5,7,8,9,10,11,12,14,15], three retained[16,17,20]; no raw-particle
pass, new simulation, bank enlargement, or split selected by test results.
These three repeatedly used fields are DEVELOPMENT checks, not independent
validation. Report shared native IDs and spatial overlap already known; do
not count rotations as new objects. Thirteen cases cannot calibrate a full
cosmological member distribution or its selection probability.

Inputs at inference: native/generated total seven-moment field only, fixed
coarse observer cell O and relative grid coordinates. O is the known coarse
observer condition, NOT a native MW subcell center. No halo IDs, member masks,
true M31/M33 positions, true masses or truth-selected peaks enter the network.
Native component labels are supervised TRAINING targets and later EVALUATION
truth only. At new-field inference no original component lookup is permitted.

One compact3D U-Net, proposed widths16/32/64, two down/up levels, four softmax
mass-allocation outputs (MW,M31,M33,remainder); one seed, <=1million trainable
parameters, <=2000 updates. Whole128^3 source support with bounded one-patch
read/cache; no member-tail truncation. Signed axis rotations/reflections act
on field vectors and observer coordinates consistently. Use normalized mass,
mean velocity and directional physical dispersion as continuous inputs, not
rare/common split categories as an auxiliary task. No architecture search.
Proposed AdamW1e-4; final checkpoint only, no heldout checkpoint selection.

If w_r(x) are output fractions, M_r(x)=w_r(x) M_total(x), so masses are
nonnegative and add exactly to the supplied total. Train equal-per-component
mass-normalized soft-label cross entropy: each component contributes its
native spatial mass distribution against log w_r, and zero-mass components
are reported unavailable, not filled with targets. This is a supervised
readout objective, NOT a normalized q(S|F,O,E) or a posterior likelihood.
Its deterministic outputs cannot represent full assignment uncertainty.
MW is observer-associated; M31/M33 role labels are supervised hypotheses,
not guaranteed detections. Soft spatial overlap is legal, including all three
components in one cell; no three-distinct-peak requirement or forced truth match.

Baseline: training-average allocation maps in the SAME observer-cell frame,
with no heldout labels and no target-field features, applied to each supplied
total mass. This measures what geometry/selection alone can explain. It is
not an independent cosmological prior. Any baseline zero-probability support
failure is reported explicitly; compare finite mass-map/COM errors rather
than adding a likelihood floor. Also evaluate the SAME trained learner with
all field inputs replaced by training means, keeping O/grid unchanged. No
second model fit. This ablation is out-of-distribution and is only secondary
evidence, not a calibrated null or likelihood-ratio test.

Deliver ONE report plus true/predicted/baseline component mass maps. Report
per role and per retained field: bound mass relative error, mass-weighted map
L1/overlap, mass-centroid error, and own-total conservation. Do not turn a
component centroid into a stellar center or call it a bound halo detection.
No component momentum or dispersion is inferred by multiplying total moments
by w_r: that would force all members to share the cell velocity. Distinct
member COM velocities/subcell stellar offsets remain explicitly unresolved.

Decision: first show training learnability (each role's aggregate normalized
map-L1 error below the geometry baseline by >=20%, an engineering criterion).
Then all three retained cases must improve M33 map-L1 versus the same baseline,
and median MW/M31 map-L1 must not worsen; report mass/centroid errors without
calling them observationally accurate. No averaged all-role pass that hides
failed M33. Passing means only worth designing the joint member state; failure
or budget truncation means no new field training and report insufficient
information/learning evidence. Do not claim impossibility from13/3 examples.
These thresholds are proposal choices, not literature-established science
acceptance. No retry, alternate seed, radius or new gate ladder in this job.

Resource proposal: ONE Slurm GPU on a40,a100,h100,h200, exclude syn06;
2 CPUs, host peak estimate5GiB+20%=6GiB,90min wall cap including preparation
and maps; <=2000 updates or70min learning, whichever first. GPU preliminary
peak20GiB+20%=24GiB; size actual architecture/buffers before submission using
the existing sizing style, not a separate framework. These are bounds, not
measured performance. No numerical work on syntax/login or manual syn101.
Reuse existing conservation/augmentation checks; one changed output-allocation
check in the same job. No separate jobs/audits between its internal steps.

### 2. Concrete denoiser repair design, DEFERRED numerical training

Propose v-prediction, NOT simply a hard-wired epsilon baseline counted as a
learned result. For z_t=alpha*x+sigma*epsilon, alpha^2+sigma^2=1:

    v_target = alpha*epsilon - sigma*x
    x_hat    = alpha*z_t - sigma*v_theta
    eps_hat  = sigma*z_t + alpha*v_theta

Here v is the diffusion parameterization, NOT physical peculiar velocity.
Use x_hat in the standard Gaussian posterior-mean coefficients to avoid
subtracting large near-equal quantities in the old epsilon formula. Reuse
the100-step schedule and exact mixed-support conservative decoder initially.
The zero-v model contains a unit-Gaussian shrinkage reference, not cosmological
learning: its stability or tiny t100 epsilon error MUST NOT count as success.
Any future trial must beat zero-v in reconstructed-clean error at intermediate
noise on retained cases AND show actual valid-field morphology. Clipping or
discarding invalid draws is not the repair. No claim that v alone guarantees
valid fractions or stable chains for arbitrary network outputs.

This parameterization is supported by Salimans & Ho, ICLR2022, section4 and
Appendix D: https://arxiv.org/abs/2202.00512 . It supports numerical design,
NOT success for our mixed seven-moment cosmological field or LG posterior.
Preconditioning as a separate design choice is also discussed by Karras et al.,
NeurIPS2022: https://arxiv.org/abs/2206.00364 . Do not implement distillation,
EDM noise/sampler changes or a library migration merely because cited.

Separate categorical training gradients from the continuous denoising trunk:
a small categorical head can condition on detached shared features plus noisy
codes/context, while continuous loss alone trains the shared spatial trunk.
Keep noisy codes available to continuous denoising and a stochastic reverse
category sampler: no replacement by always-interior class and no truth codes
at generation. This changes optimization, not the declared joint state.
It tests an unconfirmed competition hypothesis; it is not evidence that
categorical training caused the original collapse. Do not freeze a failed
continuous trunk or claim detachment supplies adequate category modeling.

Before any future long fit, one bounded known-input learning check must show
nontrivial clean reconstruction beyond the zero-v reference with fresh noise.
Avoid the old error of allowing30k updates despite loss at the null baseline.
Exact future budget/architecture/stop rule requires a separately approved
concrete bundle, not authorization from this design document.

### 3. Connection to the final objective, not a new route

Future complete state: total field F with explicit member state S. Recovered
members are latent hypotheses, including unresolved M33 within M31 cells.
Observation predictions must use member centers, distinct COM velocities and
mass-definition/stellar offsets, with uncertainties and role marginalization.
The next mass-map pilot is only the missing membership learnability component;
it is NOT that complete law or its observational likelihood.

The target remains CF4+galaxy+LG -> z=0 density/velocity posterior -> LCDM IC
-> forward validation -> LG zoom; LG<=.3 cMpc/h, environment1–2. No direct-IC
fallback or arbitrary fine phases relabelled as observed. Same-field LG-on/off
information must eventually be shown, not just more confident labels on an
unchanged F. Physical velocity dispersion is distinct from posterior error.
An E-selected member law requires p(E|F,O) with an unselected field prior, or
an explicitly E-conditioned JOINT field/member/environment prior. The32 positive
fixtures alone cannot estimate that selection. This pilot does not supply it.
Bound SUBFIND member mass != host M200c; no mass-likelihood conversion invented.
Global1.5 environment prior, calibrated member velocity/stellar offsets,
efficient posterior inference and actual fine LG-on/off maps remain missing.

## Requested plan audit

Fable5: Q-GOAL, Q-LEAN, feasibility, essential/deferred. Does ONE supervised
member mass-map learner resolve a necessary feasibility question after the
closed peak/proxy/budget tests, or is it still an unjustified detour with13/3
fixtures? Critically assess the baseline, softmax nonidentifiability, ambiguity,
M33 sharing a cell and the lack of member-velocity inference. Does this order
honor member learnability before further field-prior training? Is v-prediction
plus detached categorical gradients a justified subsequent repair hypothesis,
with the trivial stable baseline explicitly excluded from scientific success?
Give GO/CONDITIONAL GO/NO-GO and the smallest worthwhile next deliverable.
Do not automatically replace our scientific objective with prior-only LG ICs.
