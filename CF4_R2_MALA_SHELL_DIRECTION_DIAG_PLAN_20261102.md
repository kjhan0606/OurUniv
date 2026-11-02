# R2 rejected-shell direction diagnosis — bounded plan (2026-11-02)

## Driver disposition of Astra's mid-course advice

Adopt the recommendation to diagnose one already-rejected first
`16<k^2<=64` shell direction before selecting any replacement kernel. The
`epsilon=0.05` shell MALA candidate is closed at its tested scale:0/4 moves
were accepted, with proposed target increases dominated by count NLL. Do not
retune that screen, launch an epsilon ladder, or call this a universal failure
of MALA/shell methods. The MALA code and saved Hastings terms pass their
mechanics checks. The causal question is narrower: does the registered
conditional-v6 gradient agree locally with the value function along this
direction, and how different is the finite-step score change?

## Fixed direction and frozen target

- Use the source-pinned discarded state38 checkpoint and the authoritative
  job447590 record. Verify the reference result commit/status and all recorded
  values/terms before calculation.
- Reconstruct the already-accepted first fundamental proposal from state38
  using its recorded checkpoint gradient, exact orthogonal `k^2=1` projector,
  `epsilon=0.01` and seed20261105. This is the starting state `x` of state38's
  first rejected shell proposal; do not select a new direction.
- Recompute `U(x), grad U(x)` and compare value and all target terms to the
  earlier fundamental proposal record. With that gradient, reconstruct the
  rejected shell endpoint `x+d` using the exact `16<k^2<=64` projector,
  `epsilon=0.05`, and seed20261106. Recompute its full value+gradient, target
  terms, forward/reverse Gaussian log densities and Metropolis log ratio;
  compare to its prior recorded values. This validates RNG reconstruction,
  endpoint scores and Hastings accounting without attempting an MH transition.
- Report the actual first-order prediction `grad U(x) dot d` versus the
  observed full finite-displacement difference `U(x+d)-U(x)`, along with the
  endpoint gradient change. These finite-step differences are not a
  Hessian-spectrum or curvature certification.
- Let `v=d/||d||` in canonical prior-white coordinates. Use the established
  conditional-target finite-difference step `h=1e-4` and compute the symmetric
  derivative `[U(x+h v)-U(x-h v)]/(2h)`. The two evaluations are scalar
  full-target evaluations on the unchanged value/score backend; rebuild live
  support separately at both points. Compare against `grad U(x) dot v` with
  the existing 1% relative convention `abs(FD-AD)/max(abs(AD),1)`.
- Exact target-evaluation ceiling: one mandatory state38 replay by the common
  runner, two additional matched value+gradient evaluations (`x`, `x+d`), and
  two scalar evaluations (`x±h v`): five total. No retry, adaptive extension,
  scale search, held-out access, or posterior/field write. Fixed 45-minute
  application cap, 90-minute Slurm wall time,24GiB request/20GiB RSS stop,
  and the existing 20%-headroom GPU check.

## Decision interpretation

- If reconstruction/score or proposal-ratio replay fails, stop: this candidate
  cannot diagnose the gradient until that discrepancy is resolved.
- If the small symmetric derivative disagrees beyond the registered tolerance,
  investigate target-gradient/support wiring and expected support
  non-smoothness before sampler redesign; do not immediately label the gradient
  wrong.
- If the small derivative agrees but the full `d` prediction differs greatly
  from the observed endpoint score, the local first-order drift/noise heuristic
  did not control this finite shell displacement. That supports considering a
  separately justified metric/conditional proposal or the existing HMC route;
  it does not choose or validate one.

## Science gates

**Q-GOAL:** one diagnostic of a recorded transport failure on the path to the
first conditional-v6 z=0 posterior. It delivers no density/velocity posterior,
uncertainty, held-out prediction, or LG result. MW/M31 stay role-ambiguous and
M33 unresolved; their observables must eventually constrain those same roles
on the same NEW evolved LG field at `<=0.3 cMpc/h`. Native truth IDs remain
calibration/evaluation-only.

**Q-LEAN:** reuse exactly one rejected direction, the existing state38 replay,
the existing target and finite-difference convention: five complete target
calls maximum, four after the mandatory replay. No step-size sweep, new sampler
campaign, extra checkpoint, separate simulation, held-out read, or field
promotion. This is not an R2 exit audit; R2 remains NO-GO and R3 unauthorized.
