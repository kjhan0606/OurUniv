# C — spatial diffusion replacement and the LG observation connection

2026-09-10. User approved preparing the recommended replacement after closing
the flow/energy-score repair. This is a DESIGN/audit bundle, not a submitted
training job. The next implementation bundle requires user approval.

## Decision and scientific scope

Keep observations -> present field -> IC -> forward validation -> LG zoom.
Replace the failed local flow learner with ONE spatial conditional diffusion
candidate. Reuse the native total-matter data and conservative seven-moment
algebra, not the failed checkpoint. Do not resume flow weight/seed/step repairs.

Saved evaluation338389: both branches pass0/16 original morphology cases;
retained finest power/native is control/repair .655/.664 with true parents
and .436/.462 after rollout. Individual draws lack connected fine structure.
This supports changing the generative learner, but does not uniquely establish
insufficient capacity, exposure bias, or the superiority of diffusion.

The immediate product would be a CONDITIONAL PRESENT-FIELD PRIOR candidate
for a24 cMpc/h patch,1.5 -> .75 -> .375 -> .1875 cMpc/h. It is not an actual
CF4 posterior, known LG, IC, or proof of dynamically admissible LCDM history.
Do not train a global384-volume environment model in this experiment.

Why this is worth one attempt: coherent spatial structure is currently absent
even with native coarse conditions. A wider multiresolution receptive field,
direct denoising training, and native translations/axis symmetries address
limitations the closed repair did not change. These are a combined design
change, not a causal ablation that isolates any one improvement.

## Evidence, and what the literature does NOT establish

Rouhiainen et al., PRD109,123536 (2024), arXiv:2311.05217v2, demonstrate3D
conditional diffusion of cosmological gas-density fields with U-Net spatial
context and translation/rotation/reflection crops. Their31.5M-parameter model
used two A100s for70h. Its gas-only264^3 TNG300 mesh is about .777 cMpc/h;
its low-resolution condition is a matched simulation, not our restricted
seven-moment total matter. It supports the method choice, NOT our resolution,
velocity accuracy, observational posterior, or training-time forecast.
https://arxiv.org/html/2311.05217v2

Austin et al. (2021), arXiv:2107.03006, provide discrete-state diffusion;
categorical branches cannot simply be treated as Gaussian continuous values.
https://arxiv.org/abs/2107.03006

## One concrete generative model

At each factor-two refinement, encode all seven binary nodes of each parent
octet together:49 real split coordinates and49 categorical states. The nodes
are CHANNELS in one spatial model, not seven separate locally independent
generation calls. Share one scale-conditioned3D U-Net across the three LG
refinements. Keep the three-scale cascade explicit: accumulation risk remains.

Proposed frozen architecture: widths48/96/192/384, two residual blocks per
resolution, GroupNorm, timestep/scale conditioning, a1x1 input compression,
attention only at the smallest8^3 grid for the largest64^3 parent lattice.
Smaller scales use the same downsampling count with smaller bottlenecks.
Condition on all seven root moments at matching spatial resolution. Largest
training and generation context is the entire24-cMpc/h cube, not an artificial
small crop surrounded by unseen padding. No outpainting or generated seams.

Continuous channels: training-only centering/scaling, Gaussian forward noise,
100 diffusion steps, noise prediction, no density-power rescaling. Categorical
channels: four physical codes plus an absorbing MASK state during diffusion,
standard finite-state forward/reverse transitions and a categorical denoising
objective. The network sees both noisy modalities and the coarse condition.
Use their joint reverse Markov chain, not independently generated masks.
Freeze equal averaged continuous/categorical loss-block weights, AdamW2e-4,
batch1, EMA.999 and one training seed; no heldout tuning. Exact transition
formulas and loss implementation must follow the cited discrete construction.

Reuse cf4_split_moments.py for FP64 physical decoding; network FP32 with TF32
disabled. Decode nodes in their existing parent order. Raw categorical outputs
must be mapped to the legal branches of the ACTUAL generated parent: empty
parents inactive; single-child/cold branches deterministic; otherwise inactive
raw codes map to continuous interior. Internal variance states are inactive
when relative velocity is an endpoint. Freeze this canonicalization before
training and report its frequency, including native-encode identity checks.
It is an explicit many-to-one decoder, NOT an invertible flow likelihood.
Retain both zero-mass/cold atoms and three physical directional dispersions.
No density floor, invented independent sigma noise, or per-band amplitude fix.

This defines a normalized SAMPLING law as the pushforward of a finite reverse
chain, even though a tractable marginal q(F|C) is not supplied. Neither the
denoising loss nor a decoder Jacobian is that marginal log likelihood. Invalid
numerical realizations are failures with retained counts, not silently redrawn
samples from a purported unchanged prior. Conservation does not imply correct
morphology, halo membership or dynamical consistency.

## Training data and one evaluation

Use only existing total_matter_v1/matter_moments.h5. Replace the fixed12 cubes
with uniform random native coarse-aligned24-cMpc/h origins in the SAME allowed
training slab: x-origin0..16, y/z0..49 in1.5 cells. Freeze the two historical
retained origins[33,0,0] and[33,17,17] and exclude overlap before crop selection.
There is no new cosmological volume and these are correlated augmentations.

Apply all48 signed axis permutations to native moments BEFORE encoding: mass
is a scalar, momentum transforms as a polar vector (including reflection
signs), diagonal second moments permute without those signs. Transform the
coarse condition consistently. No arbitrary rotations without cross moments.
Scale sampling is balanced. Normalization uses training data only. Report
training coarse overdensity/dispersion coverage versus retained contexts; do
not introduce environmental reweighting or claim uniform crops fix rare-LG
coverage. A small bounded in-memory crop cache is implementation detail, not
a new on-disk bank of posterior hypotheses.

One30,000-update fit, with a22h learning cap and24h total cap; last EMA model
only, no best-heldout checkpoint selection. Save lightweight training history
and recovery checkpoints, no field dump per epoch. Reuse the previous fixed
training/retained contexts for true-parent versus three-scale rollout and the
same fixed draw seeds. Reuse original morphology/physical-summary functions
and numerical tolerances unchanged. Show native/control/repair/new density
slices, power, connected structure, velocities and directional sigma together.
Previous factor-two/all-draw criterion remains a historical development rule,
not conditional cosmic coverage. Report every individual result, not a nicer
mean image. No claim to recover native fine phases from coarse data alone.

At the cap, report completed updates and learning trend. Insufficient training
is INCONCLUSIVE for the model family; it is not proof diffusion cannot work.
There is no automatic longer fit or architecture search. A morphology failure
does not get overridden by denoising loss, numerical conservation, or memory
success. Even a morphology pass is only permission to PROPOSE the LG-linked
next implementation, not promotion to an observed posterior.

## MW/M31/M33: same-field connection, not labels pasted onto an image

The required joint future state is (C,F,S,O,E,nuisance). S is a disjoint
allocation F_MW+F_M31+F_M33+F_remainder=F at the fine-cell moment level, plus
calibrated center/stellar-to-COM discrepancies. Shared spatial cells are
allowed: M33 need not be a third mesh peak. All role assignments are uncertain;
native SUBFIND IDs may label training targets, never choose NEW-field members.

Use the three binary component splits already specified in
BUNDLE_C_UNRESOLVED_MEMBER_DESIGN.md to define the possible conditional member
learner q_S(S|F,O,E). This is a distinct learner, not the old known-component
transport or the14-coefficient aperture proxy. Continuous positions cannot
be inferred from diagonal cell moments alone. Positive-member anchor measure,
center discrepancy, spatially coherent bound profiles and calibration remain
unsolved. Do NOT pretend a random allocation has learned bound halos. Existing
93 profiles/32 selected patches are too limited to promise such a joint fit.

This design deliberately does NOT train q_S in the first diffusion experiment.
It closes no member-identification milestone. Before a successful field prior
is used for actual inference, a separate approved deliverable must generate
new joint field/member states and show LG-on/off conditioning changes THAT
field, including conditional M33 effects or an explicit non-identifiability
result. Do not repeat the closed peak/proxy/necessary-budget diagnostics as a
substitute. A successful diffusion run alone cannot launch actual inference.

For an unselected field prior the future target requires

 p(C,O) q_F(F|C) p(E|F,O) q_S(S|F,O,E) p(nuisance)
       L_CF4/counts(D_env|F,C,nuisance) L_LG(D_LG|S,nuisance).

An explicitly E-conditioned JOINT prior is an alternative, but changes the
environment distribution too; q_S conditioned on E alone does not implement
selection. The32 positive fixtures do not estimate p(E|F,O). Missing global
1.5 environment law, selection, host/bound-mass distinctions, sky-conditioning
measure and calibrated discrepancies remain prerequisites for actual LG use.
Mass priors remain disabled; do not bypass resolved_halos=True on latent data.
The already data-conditioned12-grid map is not an independent prior that can
be multiplied by the same CF4 likelihood again.

A diffusion prior need not have an evaluable q(F|C) if inference is formulated
in its full noise/trajectory state with the correct base law and likelihood.
That is a possible target construction, NOT an efficient sampler established
here. Ad-hoc classifier guidance is not certified posterior sampling. The
latent dimension and poor proposal efficiency are risks to audit explicitly.

## Cost, essential work and deferred scope

Proposed one Slurm job: partitions a40,a100,h100,h200; exclude syn06;1 GPU,
4 CPU cores, host --mem=48G, --time=24:00:00. Preliminary host peak allowance
40GiB plus20%=48GiB; GPU engineering allowance36GiB plus20%=43.2GiB fits a48GB
card. These are planning bounds, NOT measurements from the smaller flow.
Account for weights/gradients/Adam/EMA, checkpointed3D activations, both noisy
modalities, native FP64 buffers and bounded data cache in implementation.
If the actual estimate exceeds them, revise before submission, not after OOM.

One in-job operational segment of32 updates (included in the30,000) measures
peak memory and throughput at64^3 parents, then continues the SAME fit without
a new approval/audit checkpoint if within the frozen budget. If infeasible,
save the reason and stop; no silent smaller model or automatic replacement.
The24h cap is a budget, not a convergence forecast; queue wait is additional.
Minimal changed-code tests: mixed transition/canonical decoder correctness,
conservative seven-moment roundtrip and signed-axis augmentation. Reuse other
checks. No extra monitoring framework, GPFS tests, process scans or new raw
simulation. No new training has been submitted by this DESIGN document.

Plan auditor must explicitly answer Q-GOAL, Q-LEAN, feasibility and whether the
missing member/selection/inference path makes even this bounded field-only
replacement unjustified. It may recommend NO-GO rather than blessing another
pretty-field detour. No adverse verdict is bypassed with a fallback auditor.
