# R2 fundamental-MALA / prior-split-HMC reconciliation (2026-11-02)

**Disposition: design only; do not submit an augmented sampler yet.** This
reconciliation uses stored results and endpoint checkpoints only; it runs no
new target evaluation, sampler or simulation. It implements Astra's
mid-course recommendation after job447634.

## Same-target evidence

The job447590 MALA screen and prior-split HMC traces at state38/state60 have
the same N256,384 cMpc/h conditional-v6 target: matched GL2 value/gradient,
47,121 training count occurrences/37,951 keys,1,414 linked FP factors,
`r2_sky_closed_split_v6`, the same association ledger and the same priors.
The HMC states and both MALA anchors are dependent, discarded transport
states—not posterior draws.

For the correct `16<k^2<=64` shell (1,852 real modes):

- The MALA `epsilon=0.05` block tried4 proposals at state38/state60; none was
  accepted, so retained shell movement was exactly0 for those four attempts.
  Their four full-target calls took1,553.68s total. This closes that fixed
  identity-metric candidate only.
- The saved HMC endpoint states `next_index=38` and `next_index=60` project to
  a net shell displacement RMS of0.28454 per real mode over22 intervening
  trace transitions. The same trace window records17/22 acceptances,208 full
  target/force evaluations and77,166.47s total (370.99s per evaluation).
  Its shell mean power changes from0.48722 to0.54596. These are segment-level
  net displacement and power drift—not per-transition shell movement, ESS,
  stationarity or a matched efficiency comparison. The historical HMC trace
  did not save a shell-projected retained jump for every proposal.
- At the shared state38 anchor, the first fundamental MALA move retained RMS
  `0.007823` in its six `k^2=1` coordinates with one full-target evaluation
  (389.60s). The historical HMC trace's next accepted full-space transition
  moved `0.012725` RMS in those same six coordinates using12 evaluations
  (4,390.25s), after the preceding HMC proposal rejected and left state38
  unchanged. This single-anchor contrast is descriptive only: the samplers
  are not stationary and it establishes no efficiency advantage.

## Candidate composition and why it is not launched

One mathematically valid candidate is to compose one fixed `k^2=1` MALA
update (`epsilon=0.01`, exact full-target Hastings correction) with the
existing full-space prior-split HMC transition unchanged. Since each kernel
is separately MH-corrected for the same conditional-v6 target, their fixed
composition also preserves that target; the HMC still transports the other
IC modes and all24 nuisance coordinates. One such MALA move adds one complete
value/gradient evaluation to an8–12-force HMC trajectory, about8.3–12.5%
additional target calls before measuring wall-time overhead.

Existing evidence does not justify claiming or testing benefit yet: the MALA
fundamental screen has only4 proposals over two dependent anchors, and its
three accepted moves were all downhill. HMC remains a discarded,
nonstationary transport history. Although its net high-shell displacement can
be recovered from the two saved endpoint states, the historical trace lacks
per-transition retained shell displacement, so there is no like-for-like
shell-motion/cost record. Do not fill that gap with whole-IC jump RMS or
band-power differences.

If a later bounded engineering pilot is justified, reuse the unchanged
target/HMC route and fixed fundamental-MALA component, and record
post-Metropolis `q_new-q_old` projected into the exact fundamental,
`16<k^2<=64`, remaining-IC and nuisance blocks, alongside every exact target
call and elapsed time. Any comparison must be explicitly costed and remain a
transport pilot—not posterior validation. No step-size ladder, Hessian or
learned metric is warranted by this evidence.

**Q-GOAL:** this is sampler-design work toward the first conditional-v6 z=0
posterior, not a posterior, heldout prediction, uncertainty or LG field.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must
ultimately constrain those same roles on the same NEW evolved LG field at
`<=0.3 cMpc/h`. Native truth IDs are calibration/evaluation-only.

**Q-LEAN:** stored JSON/checkpoints and one exact shell projection suffice;
no target re-evaluation, new simulation, heldout read, new sampler code or
extra validation framework was needed. R2 remains open/NO-GO; R3 has not
started.
