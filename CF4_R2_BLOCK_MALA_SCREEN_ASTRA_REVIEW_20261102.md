# Astra advisory review — fixed HMC-drift-band block-MALA plan (2026-11-02)

This is Astra's read-only response preserved verbatim. It is a mid-course
design audit, not sampler validation, posterior approval, R2 exit approval,
or R3 authorization.

---

**Verdict: GO for the revised bounded engineering screen, with the minor clarifications below.** This is advisory approval of the design—not sampler efficacy, posterior production, or R2 exit.

The consequential correction is now present: `16<k²≤64` means `4<|k|≤8`, matching the preserved HMC drift evidence. The earlier 66-coordinate `k²=4..8` block did not. Correcting this before execution is appropriate; it is not a post-result band ladder.

### Mathematical assessment

The projected MALA proposal and full-target Hastings correction are correct provided:

- `P` is an orthogonal real-field projector in the prior-whitened coordinates, with consistent FFT normalization.
- Gaussian proposal densities are evaluated **on the block subspace**, not as nonsingular full-space Gaussians. The reverse mean uses the proposed-state gradient.
- Complement and nuisance coordinates remain unchanged.
- Full count, FP and prior terms enter the target difference exactly once.
- Candidate support is rebuilt using the unchanged live-support rule.

Fixed composition with a valid full-space/nuisance kernel can preserve the target. The screened blocks alone cannot sample the full posterior.

The frozen steps `.01/.05` have a defensible engineering motivation from saved-gradient amplitudes. Those amplitudes do **not** bound curvature or establish acceptance, particularly in the1,852-dimensional shell. The plan correctly promises neither. Retain the no-retuning commitment for this candidate.

### Concrete clarifications before execution

1. **Write “unchanged support rule, rebuilt at every candidate.”** “Do not change support” must not be implemented as frozen anchor support. “Exact-GL2” refers to the registered numerical target; it does not remove the existing support-truncation or observation-model approximations.

2. **Keep historical cost comparisons explicit.** Approximately400s and0.066 RMS describe historical **one-step HMC / whole-IC** measurements. The later L8/L12 run used184 force evaluations over68,308 trajectory seconds for20 transitions. Neither whole-IC displacement nor one-step cost is a matched efficiency baseline for the new block moves. Do not infer an improvement ratio from those unlike quantities.

3. **Distinguish informative failure from advancement.** Zero acceptances still give useful negative evidence about this fixed candidate. Accepted movement in both blocks across the two states is a necessary condition for considering a later assessment—not proof of mixing or a sufficient posterior criterion.

4. **Use deterministic math tests with nonzero drift and genuine asymmetry.** Check the forward/reverse proposal ratio and pairwise detailed-balance identity, including rejection/cache behavior and block isolation. These are small tests, not a separate validation project.

### Budget and evidence

Two dependent discarded states are suitable for this limited state-sensitivity screen, not independent chains. Two replays plus eight proposals correctly total ten complete value-and-gradient evaluations. The two-hour application cap is plausible against recent matched-target timing, but is a cap rather than a completion guarantee.

Sequential checkpoint loading,24 GiB requested/20 GiB planning maximum, and the stated GPU guard are proportionate. Preserve partial results if the budget ends; do not extend automatically.

### Q-GOAL

**Yes, narrowly:** this attacks a documented transport obstacle toward the conditional-v6 N256 z=0 posterior. Conditional uncertainty, sampler equilibration, untouched-heldout prediction and unresolved CF4 selection/group-observation calibration remain separate requirements.

MW/M31 roles remain ambiguous and M33 unresolved. Their observables must ultimately constrain those same roles on the **same NEW evolved LG field at≤0.3 cMpc/h**. Native truth IDs remain calibration/evaluation-only. This screen delivers neither identification nor refinement.

### Q-LEAN

**Yes.** One additional shell projector, a minimal corrected transition, ten evaluations and no parameter ladder are proportionate. No additional geometry campaign, separate simulation, held-out read or external audit is needed merely to apply these clarifications.

A favorable result supports only designing the next frozen-kernel assessment. **R2 remains open; R3 is not authorized.**

---
