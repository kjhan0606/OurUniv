# R2 fixed HMC-drift-band block-MALA screen — plan (2026-11-02)

## Purpose and scope

Test one fixed, gradient-informed conditional Metropolis move on the two
pre-existing physical Fourier blocks, using two already-discarded exact-GL2
transport states. This is sampler engineering only. It neither replaces nor
establishes the existing full-space HMC kernel, is not posterior inference,
and does not close R2.

The motivating candidates are the two checked-in R2 outcomes: the learned
rank-4 geometry was unstable and is closed; the one-anchor/fixed-cap ESS
candidate reached no transition and is closed without extension. Neither is
evidence against all informed physical-mode proposals. Existing exact-GL2 HMC
traces show moves but also substantial target-energy/prior drift through40
transitions; those states remain discarded, not posterior draws.

## Frozen target, states, blocks, and scale

- Use the unchanged conditional-v6 N256 target in a384 cMpc/h box (1.5
  cMpc/h grid sampling):47,121 training count occurrences/37,951 keys,1,414
  linked raw-FP factors and the same24 standard-normal nuisance priors.
  Each proposal evaluates the complete registered GL2 value-and-gradient
  target. Rebuild support at every candidate under the unchanged live-support
  rule; do not freeze anchor support or change likelihood factors, nuisance
  priors, association ledger or target orders. GL2 is the registered numeric
  target, not a claim that support truncation or observation-model
  approximations are absent.
- Use the previously discarded checkpoint states from
  `/gpfs/kjhan/CF4/z0_density/r2_hmc_long_equilibrate_20261011/` (state38)
  and `/gpfs/kjhan/CF4/z0_density/r2_hmc_equil24h_20261013/` (state60).
  Verify the exact target fingerprint and one matched value+gradient replay
  at each state before proposals. They are two dependent states from a
  discarded transport history, **not independent chains or posterior draws**.
- Update fixed orthogonal Fourier blocks `k^2=1` (six real degrees of
  freedom) and `16 < k^2 <= 64`, equivalently `4 < |k| <= 8` (1,852 real
  degrees of freedom for N256). The second block is the exact shell whose
  continued drift was measured in the 24-hour HMC trace; it is **not** the
  different 66-coordinate `k^2=4..8` shell from the stopped ESS pilot. Build
  one direct orthogonal projector for this shell and test it; do not alter or
  reinterpret the closed ESS runner. These are physical lattice-mode
  partitions, not an inferred information cutoff. All other IC coordinates
  and all nuisance coordinates stay fixed in this screen.
- Work in the already-standard-normal prior coordinates `z`. Let `U(z)` be
  the complete negative log conditional-v6 target (IC and nuisance priors
  plus count and conditional-FP terms). For block projector `P`, fixed
  identity metric, and fixed block-specific `epsilon`, propose

  `z' = z - (epsilon^2/2) P grad U(z) + epsilon P xi`,

  where `xi ~ N(0,I)` in the full canonical space. Evaluate `U(z')` and its
  complete gradient. Accept with

  `log alpha = -U(z') + U(z) + log q(z | z') - log q(z' | z)`,

  using the forward and reverse Gaussian densities on the same block; the
  reverse mean must use `P grad U(z')`. The implementation must retain the
  full target difference and must not omit the Gaussian prior or substitute a
  partial count/FP score.
- The scale is frozen by a read-only single-thread CPU projection of the
  stored complete gradients at the two checkpoints (orthonormal real FFT,
  Parseval weights for the rFFT half-space). The measured per-coordinate RMS
  gradients at states38/60 are110.1802/103.1251 for `k^2=1` and
  10.8920/10.8539 for `16 < k^2 <=64`. Set fixed block steps
  `epsilon(k^2=1)=0.01` and `epsilon(16<k^2<=64)=0.05`. The corresponding
  drift-to-noise RMS ratios `epsilon*|P grad U|_RMS/2` are0.516–0.551 and
  0.271–0.272; expected proposal displacement RMS before MH rejection is
  0.0113–0.0114 and0.0518 per coordinate, respectively. These are fixed
  engineering scales, not acceptance or posterior-movement claims. They are
  also in the range of the existing HMC prior-only rotation scales: at its
  reported step/trajectory settings the fundamental is about0.01 rad, and
  the measured `4<|k|<=8` band spans approximately0.05–0.15 rad across its
  k modes and L=8/12. No epsilon ladder, adaptation, or post-result retuning.

## Bounded execution and evidence

- Freeze four NumPy RNG seeds before submission:
  state38/fundamental=`20261105`, state38/HMC-drift-band=`20261106`,
  state60/fundamental=`20261107`, state60/HMC-drift-band=`20261108`.
- At each state, replay once, then perform two proposals in each physical
  block, in the fixed order `k^2=1` then `16<k^2<=64`. Process only one
  checkpoint at a time; never hold both 257-MB NPZ state arrays resident
  simultaneously. Total maximum:2 replay evaluations plus8 exact proposal
  value+gradient evaluations. Replay both stored target value and cached full
  gradient using the existing checkpoint replay tolerances. If an anchor
  fails fingerprint/replay or an evaluation/resource guard, stop without
  proposal retries. Reuse a candidate value/gradient only when it is the
  current state after acceptance; a rejection leaves the current state and
  cached gradient unchanged.
- Before Slurm submission, pass CPU-only synthetic tests for proposal mean,
  block isolation, the forward/reverse log-density, full-target Metropolis
  correction, and the new shell projector dimensions/idempotence. Include
  deterministic nonzero-drift/asymmetric proposals and verify pairwise
  detailed balance on a low-dimensional Gaussian target, including rejection
  and cached-state behavior. Any math,
  non-finite, score/gradient, support, or target-fingerprint discrepancy is a
  hard stop.
- Record every attempt, random seed, block/state, `delta U` by prior/count/FP
  term, forward/reverse log proposal terms, `log alpha`, acceptance, projected
  gradient/noise norms, proposed displacement RMS, accepted displacement RMS,
  and retained displacement RMS (zero on rejection), target-evaluation time,
  support-build time, peak RSS/GPU memory, and exact target-evaluation count.
  Compare cost to the historical one-step exact-GL2 HMC context (about400s and
  prior-whitened **whole-IC** jump RMS about0.066) only as context. The later
  L8/L12 run used184 force evaluations over68,308 trajectory seconds for20
  transitions. Neither whole-IC displacement nor one-step cost is a matched
  efficiency baseline for these block moves; report no improvement ratio
  across unlike quantities. These HMC data are nonstationary diagnostics, not
  a calibrated efficiency baseline.
- Freeze a maximum of10 exact target/gradient evaluations,2h application /2.5h
  Slurm wall time,24GiB host request with a20GiB runtime stop, and the existing
  20%-headroom GPU guard before launch. The20GiB host planning maximum is the
  previous same-target 14.06GiB RSS peak plus5.94GiB reserve for a sequential
  checkpoint's state/gradient, proposal/noise/drift arrays, FFT work buffers,
  allocator and runtime overhead; only one checkpoint is resident at once.
  The24GiB request is20% above the planning maximum. Apply the GPU guard as
  `1.2 * max(43,799,579,648 bytes, reported_peak_bytes) <= allocated_limit`.
  Query H200 first and use `h200` with
  `--gres=gpu:H200:1` if idle; otherwise choose a permitted H100/A100 typed
  Slurm mode and record the choice. Do not run manually on a compute node.
- Report block outcomes separately at both states, including all rejected
  attempts. Do not define success by a single acceptance, a movement cutoff,
  or a pooled ratio. The diagnostic is informative only if it yields actual
  accepted movement in each targeted block across the multiple existing
  states, while exposing retained (zero on rejection) movement and cost.
  Acceptance in both blocks across the two states is necessary to consider a
  later sampler assessment, not evidence of mixing or a posterior. Zero
  acceptances are negative evidence about this fixed candidate at its tested
  scale, not a universal impossibility result. Do not tune or automatically
  retry. Assess whether any apparent
  improvement is only one-way downhill relaxation across the two dependent
  states. No state is promoted.

## Science guardrails

No held-out values, native truth IDs, separate production/RAMSES simulation,
HOP, field output, uncertainty estimate, or R3 work. The PMWD forward evolution
intrinsic to each likelihood evaluation is counted in the target-evaluation
budget. The observation model remains conditional on observed linked-mark
presence and is not full-survey calibrated.
R2 remains NO-GO until its conditional z=0 posterior, uncertainty and held-out
prediction requirements are met and its separate exit audit approves.

**Q-GOAL:** this tests sampler transport only on the route to the first
conditional-v6 z=0 density/velocity posterior. N256/384 is1.5 cMpc/h sampling,
not a claim of 0.3 cMpc/h observational resolution. MW/M31 remain
role-ambiguous and M33 unresolved; their observables must ultimately constrain
those same roles on the same NEW evolved LG field at `<=0.3 cMpc/h`. Native
truth IDs remain calibration/evaluation-only. This screen identifies none of
those components and supplies no LG refinement.

**Q-LEAN:** reuse the existing exact target and gradients; add one fixed
orthogonal shell projector and a minimal MALA transition;
two states, two blocks, eight proposals maximum and explicit resource caps.
No ESS retry, learned-rank geometry, mode/scale ladder, held-out access,
source census, field promotion, separate simulation or posterior claim. A
favorable screen can justify designing a later fixed composition with the
unchanged full-space HMC kernel; it does not authorize posterior production.

## Driver review status

Driver disposition: Astra's mid-course review is preserved in
`CF4_R2_BLOCK_MALA_SCREEN_ASTRA_REVIEW_20261102.md`. Its four clarifications
are adopted: rebuild live support at every candidate; do not compare unlike
HMC/block displacement denominators as an efficiency ratio; interpret zero
acceptances only for this frozen candidate; and test nonzero-drift asymmetric
proposal densities plus detailed balance. R2 remains NO-GO. The bounded
engineering implementation and CPU tests may now proceed; GPU submission
still requires passing those tests and the stated startup/resource checks.
