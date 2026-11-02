# R2 count-geometry discovery/validation sketch — frozen plan (2026-11-02)

## Decision and scope

Astra's read-only review verdict is **CONDITIONAL GO** for one bounded
discovery/independent-validation geometry experiment and **NO-GO** for a
sampler launch from the tail screen alone. The driver checked the submitted
`result.json` against the declared screen contract and adopts that advice.
The tail/repeat screen passed; it does not estimate a reliable spectrum,
validate the likelihood, or establish any posterior.

Astra's follow-up found the proposed numeric gates defensible as stringent
engineering screens, not mathematical necessities or posterior-validation
criteria. It required a specified residual-trace upper confidence limit and
warned that the angle rule omits anchor displacement, FP curvature/coupling,
zero-row activation and finite-angle nonlinearities. The driver adopts the
one-sided Gaussian quadratic-form UCL below and fixed thresholds, while
keeping those limitations explicit. The rank-4 result applies only to this
candidate; failure does not prove the underlying operator or project is
impossible.

This experiment tests one deliberately limited rank-4 proposal geometry at
the same fixed conditional-v6 anchor. It is a falsification screen, not a
claim that the count information has rank four. No held-out values, native
truth labels, new simulation, candidate selection, row deletion, or intensity
floor are allowed. The frozen positive-row operator is used for proposal
geometry only; every training row remains in the unchanged exact target and
residual. Anchor-zero row activation remains unresolved.

## Frozen probe design

- Reuse the fixed anchor, training-exposed positive-row definition, N256
  count-intensity operator and typed-GPU runner from the passed screen.
- Discovery:16 independent Gaussian VJPs, seeds `20261102`–`20261117`.
  Construct one rank-4 subspace `U` from the leading four left singular
  directions of the discovery sketch. Split the discovery probes into the
  predeclared first/second halves of8 and8; compare their rank-4 spans by
  principal angles, not individually labelled eigenvectors. Require the
  minimum canonical-correlation singular value to be `>=0.80`. This is a
  stability gate for this rank-4 candidate, not an operator-rank claim.
- Validation:64 independent Gaussian VJPs, seeds `20261202`–`20261265`,
  independent of discovery and split into fixed batches of32. Keep `U` frozen.
  Stream projected vectors and residual norms; do not retain all validation
  field-sized gradients.
- Repeat the first discovery VJP once with the identical key. Require the
  repeat relative error and `||U^T U-I||_F` to be `<=1e-8`.
- Total cap:81 reverse pullbacks (16 discovery +64 validation +1 repeat),
  no automatic probe/rank ladder and no retry after failure/inconclusive
  results.

## Predeclared validation statistics and gates

For each validation VJP `s`, compute `p=U^T s` and
`s_perp=s-U p`. Use **uncentered second moments**, since the Gaussian probe
mean is known to be zero.

1. Require finite probes and summaries. Estimate `M=E[p p^T]` from all64
   validation probes and separately in each32-probe batch. The relative
   Frobenius difference between batch matrices is
   `||M1-M2||_F / ||(M1+M2)/2||_F`; require `<=0.50` and a nonzero denominator.
   This is a coarse reproducibility gate, not a confidence bound.
2. For each of the four discovery-fixed axes, construct the exact chi-square
   interval for its Gaussian second moment with64 degrees of freedom and
   Bonferroni family-wise95% coverage over four two-sided intervals. Define
   relative half-width as the larger distance from the point estimate to
   either endpoint, divided by that point estimate. Require each estimate to
   be positive and its relative half-width `<=0.50`; exact-zero or
   nonfinite directions fail closed.
3. Report the projected matrix's off-diagonal uncertainty and sorted
   eigenvalue uncertainty from a fixed-seed10,000-resample bootstrap of the
   64 validation probe vectors, conditional on frozen `U`. These intervals
   are descriptive Monte Carlo uncertainty, not a precise spectral
   certificate; do not use validation probes to alter `U`.
4. Estimate the **absolute** residual count-information
   `T_perp=E[||s_perp||^2]`, as well as its share of total sketch information.
   For a conservative one-sided95% Gaussian quadratic-form upper confidence
   limit, use
   `T_perp_UCL = mean(||s_perp||^2) / (1 - 2*sqrt(log(1/0.05)/64))`.
   This bound relies on `U` being frozen independently of all64 validation
   probes. Choose the minimum useful reference-whitened ESS angle
   `theta_min=0.10 rad` before observing results; require
   `theta_min^2*T_perp_UCL/2 <= 1 nat` (equivalently,
   `T_perp_UCL<=200`). This is a local count-quadratic screening rule only;
   it omits anchor displacement, FP curvature/coupling, zero-row activation
   and finite-angle nonlinear effects. It cannot guarantee accepted movement.

Any failed or inconclusive gate ends this rank-4 candidate. Do not increase
rank, add seeds, or weaken thresholds in response to its result. A different
proposal would require a separately justified design. Passing all gates
permits only the next separate `F_p` check; it does not permit sampler
submission.

## Resource plan

The passed screen used18:33 Slurm elapsed (application1,076.70s) for seven
pullbacks; warmed pullbacks were about134s each. This design therefore
estimates about3.0h for81 pulls plus setup. Request a3.5h application cap and
4h Slurm cap. Keep the discovery matrix on host:16,777,225 coordinates x16
float64 is about2.00GiB; the final and two half-split rank-4 bases together
are about1.50GiB. One retained/replayed full vector is about0.125GiB;
reserve a further2GiB for QR/LAPACK and transient workspace. Thus
6.42+2.00+1.50+0.25+2.00=12.17GiB, rounded up to a12.5GiB host peak
estimate. The20GiB request leaves more than20% headroom for this explicit
workspace estimate.

The previous device peak was41.13 decimal GB (38.31GiB); 20% headroom requires
45.97GiB, below the measured H100 device limit69.82GiB. Keep the sketch
subspace and validation summaries on host, not on device, and retain a
runtime memory guard. At submission, check H200 first, then H100/A100; use
the exact typed GRES, record all three free-GPU snapshots, and submit on
Syntax only. If the selected device lacks measured peak+20% headroom, do not
submit there. Request `--mem=20G`.

## After this experiment

Only if the geometry gates pass, run a separate, bounded fixed-anchor
15x15 raw-FP population Hessian check (`F_p`): inspect the unmodified signed
spectrum before constructing `PSD(-sym H_p)`; verify finite values,
symmetry and two fixed-direction Hessian actions using p-only gradient
differences at two predeclared step sizes; report clipping and conditioning
of `I+F_p`. Cap it at1h application/75m Slurm and stop inconclusively if
compilation/evaluation exceeds that budget. Count-VJP success is not evidence
about `F_p`; a block-diagonal reference does not resolve field-population
coupling.

Only after both independent geometry checks pass may a new, separately
audited full-target pilot be designed. Freeze `U,F_p,C,mu`, verify the exact
residual identity for the unchanged count+FP+prior target, retain every
training row and all complementary/nuisance directions, and predeclare a
small evaluation cap plus useful accepted movement per evaluation in
fundamental and `k^2=4-8` modes, informed coordinates, and the
reference-whitened population block. Acceptance rate or total IC RMS alone
does not pass. Exact correction preserves the frozen numerical target, not
efficiency.

**Q-GOAL:** this is a bounded proposal-geometry screen toward the first
conditional-v6 N256/1.5 cMpc/h z=0 density/velocity posterior in the
384 cMpc/h box; it is not that posterior. MW/M31 remain role-ambiguous and
M33 unresolved. Their observables must ultimately constrain those same roles
on the same NEW evolved LG field at `<=0.3 cMpc/h`; native truth identities
are calibration/evaluation labels only.

**Q-LEAN:** one fixed rank-4 candidate,16 discovery probes,64 independent
validation probes,one repeat,and a hard cap are the full geometry experiment.
No tail ladder, parameter ladder, held-out access or new gravity run. All
prior/current limitations on observed-mark-presence and survey-selection
calibration remain explicit; passing does not close R2 or authorize R3.
