# Astra mid-course review — fixed-block MALA result (2026-11-02)

Read-only review of the plan, implementation, and job447590's preserved result.
This is advisory, not an R2 exit audit or R3 authorization.

---

**Verdict: accept the result and close this fixed-scale shell candidate. Continue R2; do not start posterior production.**

I reviewed the plan, runner, MALA transition implementation and authoritative result. The reported outcome is consistent with the implementation:

- Ten complete target evaluations; both discarded checkpoints replayed successfully.
- Fundamental block: three accepted proposals out of four.
- `4<|k|≤8` shell: zero accepted proposals out of four; count-likelihood deterioration dominates the target increase.
- The forward/reverse proposal terms enter the Hastings correction correctly. Rejection preserves the current position, target value and gradient.
- No posterior, uncertainty estimate, held-out prediction or field was delivered.

This is a **local failure of the identity-metric, ε=.05 shell proposal**, not evidence that MALA, the shell, or CF4 reconstruction is impossible. The fundamental result is promising only as a local transport observation. Although its three accepted moves lowered U, its uphill proposal had logα≈−0.232: its rejection was stochastic, not evidence that the kernel can only move downhill.

### Smallest useful next action

**Diagnose one already-rejected shell direction before designing another sampler.** Do not reduce ε and rerun the screen.

Use the source-pinned state38 history and frozen RNG to reconstruct the first shell proposal. The initial checkpoint gradient and first accepted fundamental move determine its starting point. Perform one bounded value/gradient consistency check:

1. Replay the reconstructed shell starting point and rejected endpoint; verify their recorded scores and proposal-ratio accounting.
2. Record the actual directional first-order prediction, `∇U(x)·d`, against the observed `U(x+d)−U(x)`. Endpoint gradient change can characterize nonlinearity along this direction, but is **not** a Hessian-spectrum estimate.
3. Check the starting directional derivative using a symmetric, predeclared small displacement along that same direction. Reuse any existing equivalent verified test rather than repeat it.

This can be limited to **two matched value/gradient evaluations plus two scalar evaluations**, with a hard budget and no adaptive extension. It reconstructs an existing failed direction; it is not another proposal trial, ε ladder, or geometry search.

The decision it should support is concrete:

- **Directional inconsistency beyond the registered numerical tolerance:** investigate the target/gradient/support wiring or expected nonsmooth behavior before any sampler redesign. Do not immediately label the gradient wrong.
- **Directional consistency, followed by strong finite-displacement nonlinearity:** the first-order drift/noise heuristic did not control this shell’s geometry. Design a separately justified metric/conditional proposal or reconsider the existing HMC transport strategy. Do not infer a successful replacement from these four failures.

The current evidence alone cannot distinguish those explanations. Repeating a different sampler without that distinction risks another arbitrary candidate.

### Q-GOAL

**Yes, this was relevant sampler engineering, but it supplied no actual z=0 posterior.** The shell corresponds to the documented HMC drift, and the full target was preserved.

Keep four issues separate: sampler equilibration/Monte Carlo error; uncertainty conditional on the frozen target; untouched-heldout predictive validation; and unresolved CF4 selection/group-observation calibration. Better sampling cannot repair an uncalibrated observation model.

MW/M31 roles remain ambiguous and M33 unresolved. Their observables must ultimately constrain those same roles on the **same NEW evolved LG field at≤0.3 cMpc/h**. Native truth identities remain calibration/evaluation-only. The current N256/384 cMpc/h grid samples1.5 cMpc/h and does not deliver that LG refinement.

### Q-LEAN

**The completed screen was proportionate. The next work should be one causal check, not another campaign.** Reuse the recorded failure, existing derivative tests, source-pinned checkpoints and target evaluator. No additional seeds, rank search, parameter ladder, long chain, separate simulation or held-out read.

There is no need to rerun successful fundamental tests now. Preserve that kernel as a candidate component, without promoting its endpoints.

This is a **mid-course advisory review**, not an R2 exit audit or R3 authorization.

---
