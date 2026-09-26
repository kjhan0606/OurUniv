# Next correction: probabilistic, cubic-equivariant LG role locations

2026-09-12 driver proposal after343469; user requests progress. Plan audit is
advisory: the driver verifies recommendations against evidence and records
adoption/rejection. This document specifies the proposed learning bundle;
no numerical launch before its scope and approval boundary are settled.

## Why change the task, and what this does NOT deliver

Frozen diagnosis343469 reproduced both checkpoints and passed8 regressions.
The single-field model already failed rotated copies before multi-field
training; final multi model is orientation-sensitive in6/6 tested fields.
12/16 native targets involve a random eligible companion choice, but a unique-
choice development field fails too. Neither issue is the sole proven cause.
The native draws are valid calibration samples. The error was interpreting
one random draw as uniquely recoverable by one deterministic mass allocation.

Replace that allocation target with a normalized DISTRIBUTION over three
role-center cells on the same .1875 cMpc/h,128^3 native field. This is a
location/assignment component of q_S, not halo mass partition, member COM
velocities, calibrated full q_S, total-field q_F, or the observed z=0 posterior.
Do not call a sampled cell a resolved halo or a reconstructed galaxy.
Final destination unchanged: CF4+galaxy+LG -> present density/velocity posterior
-> LCDM IC -> forward LG zoom; environment1–2,LG<=.3 cMpc/h.

## Explicit probabilistic task

Fit q_theta(c_W,c_A,c_T | F,O,E_train) = q_W(c_W|F,O) *
q_A(c_A|F,O,c_W) * q_T(c_T|F,O,c_W,c_A), each factor a spatial categorical
distribution over the SAME128^3 cells. E_train denotes the existing native
two-primary/M31-satellite selection; it is not actual MW observer selection.
M31/M33 may occupy the same cell. Do not exclude cells occupied by previous
roles, require three distinct peaks, or introduce a permanent M33 waiver.

Training targets: existing native SubhaloPos from the93-object source request,
converted by the existing periodic patch origin to a cell; not halo mass peak
positions or labels inferred from a smoothed total field. No new snapshot
read/profile collection. Fix the previous13 training/3 development split.
Use native parents for teacher-forced conditional log-probabilities during
training and proper joint log-score evaluation. This evaluates q at the
labelled tuple; it is NOT successful autonomous detection. At NEW-field
sampling, draw MW from q_W, then M31/M33 conditioned ONLY on earlier draws.
No true IDs, positions, nearest-truth seeding or true-parent replacement.

Negative joint log probability, averaged over the3 factors, is the objective;
random eligible selections are samples, not hard uniqueness requirements.
One continuous probability mass function can represent separated modes without
averaging their positions into a fake intermediate halo. This pilot is NOT
uniform sampling over all candidate tuples: the selected native population
and stochastic source selection define its calibration law and limitations.

## Structural correction for signed cubic symmetries

Reuse the small three-scale16/32/64-channel U-Net layout, with these changes:
all learned3x3x3 scalar kernels tie coefficients within four signed-cubic
offset orbits (center,face,edge,corner); downsampling uses nonoverlapping2^3
average pooling plus1x1 convolution, not asymmetric stride2 padded convolution.
Nearest integer upsampling, GroupNorm, pointwise activations and scalar heads
preserve the same48 signed axis symmetries. No claim of arbitrary-angle SO(3)
equivariance, and no48-forward ensemble. One shared conditional network,
role one-hot and presence masks; no warm start from the orientation-memorized
mass model. Small exact-feature/kernel/whole-network regression, not a new
general group-convolution library.

Scalar inputs derived from the full native seven moments: log density,
velocity norm, dispersion trace; for O and each available parent cell, distance,
radial velocity projection, radial diagonal-variance projection, presence mask;
role one-hot. Signed permutations transform V, diagonal variance and anchor
positions consistently. All directional variances remain in source state;
invariant contractions are model features, NOT a replacement physical field.
No absolute xyz memorization channels or catalogue inputs on new fields.

Spatial logits are finite everywhere. Add a fixed broad geometry term centered
on O for MW, on sampled/conditioned MW for M31, and on M31 for M33, with Gaussian
widths .75/1.5/1.0 cMpc/h respectively. This is a declared candidate-position
reference measure, NOT extra physical mass. No hard mask or mass epsilon floor.
Normalize log probabilities with logsumexp across cells. Learn residual logits.

## One proposed fit and fixed endpoint evidence

Fresh initialization, AdamW1e-4/.01, strict FP32, gradient clip10. Proposed
6500 joint updates,500 per each13 fields, shuffled complete cycles. Every
joint update uses all3 teacher-forced factors; no separate single-field
memorization phase, checkpoint selection, LR/seed/model sweep or auto-extension.
6500 is a bounded feasibility budget, not proof of sufficient data/training.
Structural equivariance replaces reliance on randomly learning48 orientations.

Evaluate final joint/per-role log scores on all13 training and3 development
cases; all rows, not just the average. References: the declared geometry-only
distribution; a geometry-times-positive-density reference; and for M33 a
training-calibrated same-M31-cell mixture plus broad reference, to avoid
crediting a trivial shared-cell prediction as new M33 information. Declare
the mixture coefficient from training counts with a uniform Beta(1,1) prior,
not development tuning. All probability references normalized identically.

Also draw64 complete autoregressive triples for each development field,
reporting role location spread, shared-cell frequency and failures; score or
plot native targets only AFTER samples are frozen. These are correlated
conditional draws from reused development fields, not64 new universes or
independent astronomical validation. Require all48-view distribution-L1
differences <=1e-4 in the fixed whole-network symmetry check, before and after
fitting; inspect every role. Numerical failure never becomes a science verdict.

Bounded continuation decision: mean development joint log score improves over
both general references; mean M33 conditional log score improves over both
broad and same-cell references; report each individual field. If not, no-go
for adoption of this location component, not fundamental unidentifiability.
Even pass is FEASIBILITY_ONLY: three reused fields cannot certify coverage,
full q_S, actual observations, or a .1875 observed mass map. No mask-L1 gates
are imported from the different mass-allocation task.

## Same-field observations and remaining gap

This q_theta is a distribution conditional on F, not on LG observations d.
Future location evidence for F is an explicit sum/integral
L_pos(d|F,O)=sum_c q_theta(c|F,O,E_train) int_cell p(d|x,offsets) p(x|c,F) dx.
Thus observation preference can affect the SAME field through its member law;
it is not enough to paint labelled halos onto an unrelated field. Subcell
and stellar/potential/COM offsets, real observer selection versus E_train,
mass/velocity/member profiles and q_F require calibration BEFORE actual use.
The current candidate law does not supply those missing terms. Do not feed a
data-conditioned q(S|F,d) back as an independent likelihood or count d twice.
No actual-data weighting, fine-field-prior restart, IC or production in this job.

## Q-GOAL / Q-LEAN and proposed resources

Q-GOAL: a probabilistic member-location operator that does not require oracle
parents at inference is a concrete missing part of connecting LG observations
to F. But center-only progress must not become another indefinite prerequisite
to the mass/velocity posterior. Audit whether this limited output is worth
one fit, or whether a smaller/different next step serves the final goal better.
Q-LEAN: one model and one fit, existing data/primitives, two focused new tests,
proper-score references and one endpoint; no more frozen-diagnostic stages.
Deferred: exhaustive role-source catalogue, new simulations, equivariant model
sweep, mass/velocity posterior, observed inference and high-resolution IC.

Provisional1GPU/2CPU/6GiB host/4h Slurm, a40,a100,h100,h200 excluding syn06;
host estimate<=5GiB+20%, derived from earlier13-case cache, streaming three
conditional passes/gradients to limit memory. Exact parameter/activation sizing
must precede launch; one early operational check is INSIDE the same allocation.
Learning190min/application230min; incomplete budget is inconclusive. No
hardware/process/filesystem diagnostics. New output role_locations_v1,
preserve prior models. A larger resource request requires explicit approval.
This is a redesigned task, not a correction that silently resumes the old fit.
