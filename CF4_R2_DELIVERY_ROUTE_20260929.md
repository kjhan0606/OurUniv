# R2 delivery route after the source-volume finding

Order remains R1 -> R2 actual present-state posterior -> R3 same-field LG ->
R4 precise evolution -> R5 phase-consistent zoom IC. R2 is NOT complete.
Approved implementation target:384cMpc/h,N256/1.5. N128/3 remains development.
This document implements the master plan; it does not replace the science goal.

## Focused Fable advice and driver disposition

Request: `config/cf4_r2_delivery_advice_20260929.md`. Completed read-only CLI
advice: ADVISE PROCEED. Trigger was the substantive point/volume discrepancy
and impending whole-target/science-resolution computation, not a routine gate.

Adopt: a labelled FP-subset+galaxy-count CONDITIONAL posterior is a legitimate
first R2 delivery if it includes actual science-resolution draws/mean/UQ,
mixing evidence, information-support maps and heldout prediction. It is not
an all-CF4 reconstruction or calibrated absolute-scale claim. Additional CF4
methods and multi-FP groups remain explicit future scope, not secretly used
or permanently discarded. This does not waive the final LG/zoom objective.

Corrections independently checked by driver:

- The sky/graph-closedv6 training/heldout split is ALREADY frozen and every
  current1414 mark is training-only. Do not create a new split after fitting.
- Count LF/rate/bias/LOS nuisances are LIVE, not the "fixed LF nuisances" in
  the advice. Fixed current values belong only to the diagnostic readout.
- The old replan's15.8 minimum is NOT this FP cohort's measured support.
  Reading only the current small radius arrays gives observed-point radius
  22.301225–179.867507cMpc/h and FP-reference radius21.279958–179.988359.
  Counts begin at5. Therefore no direct LG FP constraints, and the named
  Virgo region may be informed by counts/environment rather than this FP
  distance cohort. Report actual support, not inherited coverage claims.
- HMC mean |DeltaH| near1 does not by itself imply acceptance "collapse".
  Assess its distribution, measured acceptance and cost per effective draw.
  The suggested3–5x speed threshold is advice, not an invented pass gate.
- "Fine" means a declared fixed target quadrature, not automatically8^3.
  Source4^3 with a measured8^3 sensitivity is the current development route.
  Re-scoring alone does not prove map shifts: any importance-reweighted map
  requires adequate overlap/ESS, otherwise another calculation is needed.

## Three substantive remaining actions, not a new gate ladder

1. Finish ONE coherent volume-integrated count+raw target, with shared proper
   population priors, state-refreshed source support and all1414 training
   marks. Profile a full value AND gradient including support generation,
   not only the fast sparse inner kernel.408353 performs the full raw-mark
   readout; counts/PM are not included in that runtime. Existing point/eta
   chains are initialization/history only, not samples of this new target.
2. Use measured-cost short sampling and an N256 resource pilot before two
   independent science-resolution chains. Keep actual covariance/mixing
   unclaimed until supported. If cost is excessive, consider deterministic
   coarse-force HMC with fixed fine-target Metropolis energies ONLY after
   comparing costs; it is not implemented or selected now. A support union
   frozen at a trajectory's start is NOT a reversible state-local rule.
   Fine energy/rule and metric stay fixed after warmup; rejected states and
   their accepted fine energy are retained. Do not substitute a smaller-grid
   target or tempered likelihood and call it the same posterior.
3. Deliver actual field draws, mean and UQ with TWO numerical sensitivities:
   weak common-scale/monopole prior dependence (reweight first if overlap
   supports it), and finer quadrature scores/map-summary dependence. Include
   one untouched CURRENT-split predictive evaluation, no subsequent refit
   disguised as that test's validation. Constant type/graph incidence and
   physical-selection transfer are conditional assumptions, not calibrations
   proven by training marginal agreement. Effective information resolution
   is separate from1.5cMpc/h grid spacing.

MW/M31 identities remain ambiguous; M33 remains unresolved. R3 must identify
roles from the NEW field/state and let their observables constrain that SAME
field. No truth IDs, hardwired known components or arbitrary high-k rescaling.
No GPFS diagnostics, new TNG dependency, mail, manual syntax runs or watchers.

## Immediate full-count cost measurement

Alongside the whole-cohort raw readout, implement the genuine source-volume
count integral by streaming GL nodes, not the rejected analytic surrogate.
Profile source2^3 and4^3 with LOS4x8 on the actual training counts at408337's
field. Differentiate ALL native density/velocity cells and9 tracer coordinates,
then compare the native velocity-direction derivative with finite differences.
No PM/fit/heldout or surrogate sampler. One H1002CPU8GiB45min bounded job;
estimated host<=6.5GiB plus20%, device temporary limit60GiB. Save only compact
summary and fine gradients, no large simulation snapshots.

One algebraically exact speedup: with wrapped coherent source positions and
8*sigma_radius < L/2-Rmax, no noncentral observer image can intersect the
truncated LOS path. Skip its large shell arrays, retaining all27-image logic
outside that domain. Check value/gradient against forced27-image evaluation.
This does NOT assert that untruncated radial-mark aliases are identically zero.
