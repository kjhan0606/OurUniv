# Astra mid-course review — physical-mode ESS result (2026-11-02)

This is the read-only Astra response preserved verbatim. It is a mid-course
advisory review, not the separate R2 exit audit.

---

**Verdict: accept the reported candidate failure; continue R2. No posterior promotion, R2 exit, or R3 authorization.**

### 1. Result disposition

I independently read the authoritative `447541/result.json`.

- Checkpoint replay passed; the selected scalar backend reproduced the aggregate anchor score.
- Arm A exhausted four all-IC proposals. Arm B exhausted two fundamental-mode proposals.
- Neither accepted a transition. The 66-coordinate slow band, complement and nuisance updates were **not tested in B**.
- Both arms correctly stopped as incomplete; zero displacement is explicitly reporting-only, and comparison ratios are absent.
- There is no posterior, uncertainty, held-out score, promoted field or truth-ID use.

This rejects the **registered capped candidate**, not ESS generally, all physical-mode proposals, or R2 feasibility. The result does not estimate the acceptance rate or posterior conditional widths. Do not retry it with additional bracket evaluations, different seeds, or altered movement thresholds. The rank-4 branch remains closed.

The evaluated fundamental-mode proposals sharply worsened the count likelihood. This is consistent with prior-scale proposals being poorly matched to the conditional likelihood, but two proposals cannot establish that mechanism uniquely.

Resource limits were not responsible: reported host peak was12.219 GiB and device peak33.057 GB, both comfortably within the requested limits.

Two housekeeping issues deserve correction without another calculation:

- The JSON retains obsolete L-BFGS-B/optimizer and “MAP diagnostic” metadata despite being an ESS diagnostic.
- Its resource snapshot records three free H200 GPUs while H100 was selected. Document the actual reason; if none existed, acknowledge the departure from H200-first policy. This does not explain the scientific failure.

### 2. Recommended next design

**A bounded, gradient-informed conditional-Metropolis augmentation of the existing full-space HMC kernel**, not another prior-ESS experiment.

Use the already-defined fundamental and `k²=4..8` coordinate blocks. A conditional MALA proposal would use the projected gradient of the **complete negative log target**, including the Gaussian prior, to propose a local move. Its acceptance must include both the full target difference and the forward/reverse Gaussian proposal-density ratio. Evaluate the reverse proposal mean using the gradient at the proposed state.

Keep a fixed, positive-definite metric and freeze proposal settings before the diagnostic. Establish the initial scale from existing valid HMC displacement/metric evidence or explicitly bounded, discarded warmup—not from the failed ESS bracket or a rank/threshold ladder.

Retain the existing full-coordinate/nuisance HMC transition in the proposed eventual composition. The new blocks alone cannot sample the entire field. This is materially different from both rejected approaches:

- no learned low-rank geometry;
- no prior-scale elliptical orbit;
- no repeated HMC mass-factor search;
- explicit likelihood-directed proposals with a computable full-target correction.

**This is a justified next hypothesis, not a demonstrated cure.** Existing HMC nonstationarity remains unresolved until the combined sampler actually improves transport.

### 3. Minimum bounded evidence before a larger sampler assessment

Before GPU work, verify the proposal/reverse-density mathematics and projector wiring on small tests. Then conduct a **small, predeclared multi-state engineering screen** using existing discarded states—never held-out values or truth-selected states.

It should establish:

1. Full-target and gradient consistency, finite forward/reverse proposal terms, and complete evaluation accounting.
2. Actual accepted and rejection-inclusive movement in the troublesome modes at more than one state.
3. Cost relative to the existing HMC transition, including gradients and support rebuilding.
4. Whether improvement is only downhill transient relaxation rather than useful movement across states.

Freeze a total evaluation/time budget before launch. Do not label this screen’s endpoints posterior draws. If it succeeds, a separate frozen-kernel, multiple-chain assessment must address warmup, drift, autocorrelation/effective sample size and Monte Carlo error before posterior claims.

### 4. Q-GOAL

**Aligned, but only as sampler development.** Keep separate:

- **Sampling error:** insufficient equilibration or exploration of the frozen conditional-v6 target.
- **Conditional posterior uncertainty:** uncertainty under that particular target and its assumptions.
- **Held-out predictive validation:** later untouched-data performance; not a substitute for sampling validity.
- **Observation-model calibration:** unresolved linked-mark selection, CF4 group-incidence/survival and related assumptions. Successful sampling cannot fix these.

N256/384 cMpc/h gives1.5 cMpc/h field sampling, not proven observational resolution. MW/M31 remain ambiguous and M33 unresolved. Ultimately their observables must constrain those roles on the **same NEW evolved LG field at≤0.3 cMpc/h**. Truth identities remain calibration/evaluation-only. This screen neither identifies those components nor supplies that refinement.

### 5. Q-LEAN

**Proceed with one materially different, bounded transport test—not another instrumented geometry campaign.** Reuse the target, gradients, physical projectors and existing HMC backbone. Stop ESS bracket experiments, rank searches and unused geometry follow-ups. Correct the misleading metadata directly. No new RAMSES run, field production, held-out access or extensive monitoring framework is warranted now.

This is a mid-course advisory review only.

---
