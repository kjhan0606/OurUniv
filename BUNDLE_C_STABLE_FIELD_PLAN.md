# Autonomous C: one stable joint-field generator experiment

2026-09-12. User explicitly authorizes autonomous in-goal implementation,
Slurm execution, evaluation and commit/push without per-bundle approval waits.
Driver seeks Fable5 advice, independently checks it, and preserves failures.

## Goal and new evidence

CF4/galaxy/LG -> present density/velocity posterior -> LCDM IC -> forward LG
zoom. LG numerical scale<=.3 cMpc/h; surroundings1–2. Not direct-CF4 IC.
347085 now supplies a21-coefficient same-field role-position law;347087
supplies a correlated distance/sky likelihood with verified conservative
field derivatives. Neither supplies a field prior or physical mass/COM readout.
The8 mock original fields rank first jointly and for incremental M33; this
is reused one-box development evidence, not final local-universe recovery.

Previous field diffusion338402/338746 already used dense translated crops,
not the13-field member set. Its epsilon output remained nearzero (MSE~1),
terminal algebraic epsilon reference has MSE~2.4e-7, reverse latent RMS1->2032,
all draws invalid. Source shows a shared continuous/category backbone with
a48-channel1x1 stem for49 independent noisy coordinates plus category/context
inputs, and no explicit terminal noise-reference bypass. This is NOT proof
of the unique failure cause; more data/steps alone does not justify revival.
The earlier flow/objective/bank repairs remain CLOSED.

Test a materially changed small model: bounded v-parameterized reverse mean,
independent continuous/category learning paths, full-resolution convolutions
without a downsample bottleneck, a pointwise noisy-input bypass, and MATCHED
observer-conditional spatial data. One fresh fit; no checkpoint continuation,
hyperparameter sweep or automatic longer-training repair after a miss.

v here is a DIFFUSION target, NOT the physical peculiar velocity field.
Parameterization source: Salimans & Ho2022, section4,
https://arxiv.org/html/2202.00512v2#S4 (read from the primary paper).
This cosmological application is our hypothesis, not their demonstrated result.

## Physical state, data and observer selection

Reuse the EXACT native seven-moment split chart/49 continuous coordinates,
49 atom codes, canonicalization and conservative decoder already roundtrip-
tested. Preserve all M,Pxyz,Qdiag and three physical variances; no density
amplitude repair, physical mass epsilon, clipped fake profiles or halo painting.

Train on the1388 existing train observers, uniformly shuffled across epochs;
all captured native matter, not preidentified member components. Use80^3
buffered cubes: existing64^3 core plus8 native cells on each face,15cMpc/h
overall, .1875 fine. The old integer-aligned guarded slabs actually contain
these80 cubes too; assert this geometrically. Train/test slabs stay disjoint.
Use the same8 fixed test observers for evaluation; historically reused native
data are development only. No test-based normalization, early stopping or seed
selection. Calibration split is unused for this one generator.

The training distribution is explicitly q_E(F|coarse,O), conditional on the
archive's two-primary/M31-satellite eligibility E, not unconditional universe
fields. This matches the role-law population weighting. Real LG membership
in E and actual observer-selection completeness are unverified assumptions;
conditioning the whole field prior on E would avoid a missing p(E|F,O) factor
ONLY for a scientifically justified E-conditional target, not universal use.

Use3 refinement scales .75,.375,.1875 from a native1.5-cMpc/h parent10^3.
Both training and generation use complete spatial contexts of10^3/20^3/40^3
parents. Observer position is an explicit conditioning input: at native80
faces44, transformed with every one of48 signed cube augmentations BEFORE
packing. No augmentation silently moves the observer while retaining old O.
Provide three coordinate channels relative to O (physical units /12cMpc/h)
along with existing7 coarse context channels and timestep/scale embedding.
Native fine halo/buffer is NOT supplied at generation: generate the full80
and compute frozen location features on its inner64 using the generated halo.

## Network, loss and bounded execution

Two independent fully convolutional64-channel networks, each1x1 stem,
three residual blocks with two3x3 convolutions/dilations1,2,4, GroupNorm8,
time/scale conditioning and1x1 output. No downsampling/attention/checkpoint
dependency. Continuous output49, categorical output196; inputs are noisy49,
masked-code onehot245 and10 coarse/observer channels. Continuous path also
has learned49-channel timestep gains multiplying raw noisy coordinates,
initialized tozero; output heads zero-initialized. Separate parameters mean
categorical CE cannot directly overwhelm continuous gradients. ~1.5m parameters
(count must be computed exactly before source commit/launch), not34m.

z_t=sqrt(A_t)x+sqrt(1-A_t)eps; target v=sqrt(A_t)eps-sqrt(1-A_t)x.
Mean-v MSE plus unchanged masked absorbing-category variational CE, independent
branches. AdamW lr2e-4, decay.01, gradients clipped at10 PER branch, EMA.999.
24,000 total updates, balanced3 scales; all1388 observers repeatedly shuffled.
Training-only normalization from64 fixed-seed observer cubes, active chart
coordinates, all3 scales. Padded inactive clean coordinates remain zero.

Reverse mean in a cancellation-free algebraic form:
 mu_t=sqrt(1-beta_t)*z_t
      - beta_t*sqrt(A_{t-1})/sqrt(1-A_t)*v_pred.
Use the existing posterior variance beta_t*(1-A_{t-1})/(1-A_t),100 steps,
unchanged absorbing reverse. With v_pred=0, the latent reference is
NONAMPLIFYING and its variance stays<=1 (it is not a learned LCDM prior).
Test v/epsilon/x0 identities and reference stability, plus branch gradient
isolation, native roundtrip and observer transform in the SAME allocation.
No sampling-time clipping/rescaling to manufacture acceptable structure.

OneGPU/2CPU/6GiB/4h, a40,a100,h100,h200 excluding syn06. Host estimated4.5GiB
+20% rounded6: one7x184x400x400 FP64 training slab1.65GiB, streamed native
80^3/latent buffers, optimizer/model/runtime. GPU ceiling8GiB. Application
3h50 TOTAL, learning3h; save partial fit and report incomplete, no hidden
extension. Periodic6000-update checkpoints, final fit/evaluation in this job.
New output stable_field_v1, preserve earlier models. No filesystem probes,
manual node runs, new snapshot download or independent monitoring daemon.

## Endpoint, science judgment and immediate next outcome

Evaluate denoising at fixed t10/50/100 on all8 native test fields/3scales
against zero-v/reference, including continuous and categorical results.
Generate2 fresh full80^3 draws for EACH8 native parents, recording invalid
draws as failures rather than discarding/retrying them.16 draws,48 refinements.
Evaluate inner64 morphology versus SAME-context native: reuse existing power,
top1% mass, connected fraction, velocity/dispersion and parent-boundary metrics.
Keep published per-draw .5–2 ratio criteria unchanged and report ALL results.
For a stochastic feasibility decision ALSO report per-parent two-draw means;
do not silently call an individual failed draw passing from its ensemble mean.
Full feasibility requires all16 valid, each8 parent ensemble's existing tested
summary ratios in[.5,2], and numerical conservation/realizability. A partial
learning result remains incomplete regardless of attractive example images.

Run the frozen q_E(role cells|F,O) and Lpos on GENERATED fields (no oracle
members/parent positions). Shared M31/M33 cells remain legal, not silently
three resolved halos. On the8 first draws sample16 complete role triples;
save physical density/mean velocity/dispersion projections plus position
probabilities/samples and source-conditioned likelihood scores. This links
the generated state to observations but does NOT yet constrain it to actual
LG data or supply member masses/COM, q(coarse|CF4) or IC.

If useful, next substantive design is SAME-field LG conditioning with the
explicit latent generator/selection assumptions and mass/COM uncertainty,
not more score diagnostics. If it fails, driver analyzes actual evidence
before any different experiment; no preauthorized refit loop. Fable: assess
Q-GOAL, Q-LEAN, plausibility of the changed mechanism (not unique-cause proof),
MW/M31/M33 identification on generated fields, feasibility and essential/
deferred scope. Give your verdict and essential corrections, not an audit ladder.
