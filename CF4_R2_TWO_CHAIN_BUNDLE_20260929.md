# R2: bounded N256 exploration after the affine pilot

Order: R1 -> **R2 current-state posterior, ONGOING** -> R3 same-field LG ->
R4 precise evolution -> R5 phase-consistent zoom IC. R2 is not complete.

408447 COMPLETED59m50s, host18.40GiB. Two proposals accepted, fine Hamiltonian
errors -4.86639/-5.89956. Proxy integration contributions -2.08411/-1.87149,
fine-minus-proxy changes -2.78228/-4.02807. Fine derivative/value identical;
first fine derivative1132.80s including compilation, estimated GPU30.56GiB.
Steady force147s, endpoint fine289–317s. Restricted low-band white power
.698109 -> .700069 still drifts. This establishes transport, NOT stationarity.
The changed RNG means this is not a matched-momentum controlled comparison.

## Finite execution

The previously completed Fable conditional advice required real transport and
cost/memory before up to2x24GPU-hours; the driver now finds those conditions
met for bounded EXPLORATION, not posterior promotion. No new routine audit.

- Two H100 jobs,2CPU,48GiB host,24h each,23h application deadline. Measured
  pilot18.4GiB plus accumulators, saved-array copies and working buffers are
  estimated below40GiB, leaving20% margin. Existing compiled GPU20% check
  remains in force. All numeric work via Slurm, no login computation.
- At most48 proposals per chain; first12 discarded. First6 use4 integration
  steps, remaining warmup and all retained proposals use8. Warmup adapts the
  step toward .65 acceptance, bounded above .08; thereafter step, length,
  metric, affine anchor/correction and fine target remain fixed. No per-path
  anchor refresh. Estimated total about20h/chain from measured costs, not an
  assurance of sufficient mixing. Budget exhaustion saves completed states.
- Chain A starts from408447's accepted state. Chain B uses another unselected
  prior completion of408389's N128 field with seed2026092921, then genuine
  N256 evolution. Independent sampler seeds2026092922/2026092923. The low-mode
  ancestry and nuisance starts remain shared/nearby: not independent global
  modes. All modes/nuisances remain live. Initializers are not posterior draws.
- Same1414 training marks+47121 counts, source256/observed128 keys, physical
  rate1/8, sourceGL2 fine energies, GL1+fixed affine proposal force, proper
  priors once, no new observation or selection model, heldout untouched.
- Accumulate native rho and occupancy-conditioned mean-velocity posterior
  moments at EVERY retained state, including rejections. Physical within-cell
  dispersion remains separate. These are UNASSESSED sample moments until
  mixing/MC-error checks; undefined variances remain NaN.
- Record24 nuisances, fundamental IC projections, restricted low-band power,
  total white power, count/raw scores, eight spatial octant density means,
  acceptance, jump RMS and Hamiltonian-error attribution. Save a field/IC draw
  each8 retained states plus final accepted full state, not particle histories.
  No automatic convergence claim or hidden thinning for scalar diagnostics.

## Next decisions, same bundle

Check actual transport, drift, autocorrelation/ESS and between-chain agreement;
inspect illustrative field slices. A short or unmixed chain is a failed/partial
science result even if Slurm exits0. Diagnose force drift or metric stiffness
from those records rather than weakening the fine target. No heldout score
before model freeze. Only usable draws support numerical/prior sensitivity,
information-support maps and the single untouched v6 predictive assessment.
Same-grid GL4 may require exact row streaming; it is not yet implemented.

Q-GOAL: obtain actual history-consistent z=0 density/velocity posterior draws
at the approved1.5-cMpc/h environment scale, not a picture from initialization.
Q-LEAN: two bounded measured-cost chains with existing target/sampler and lean
moment/trace storage; no additional proxy ladder, filesystem tests or surveys.
MW/M31 remain ambiguous and M33 unresolved. Later identification and their
observables must constrain this SAME NEW field. No truth IDs or fixed known
components are inserted. This environment calculation does not resolve LG
halos or deliver the final <=.3-cMpc/h LG/zoom objective by itself.

Submitted after408499 passed11 focused regressions in14s:408500 independent
high-mode initializer,408501 chain A,408502 chain B (afterok408500). All use
source4af620d; the two chains each reserve48GiB and24h. Source pushed to
origin/agent/freeze-zoom-pipeline. Jobs are not yet evidence of usable draws.

408500 completed the alternative initializer and both chains started.408504
passed3 diagnostic controls (IID vs shifted chains, autocorrelated/stuck
chains, preserving rejected repeats). Pending408503 was replaced before
execution to avoid waiting for an H100 for CPU-only scalar tests.
408505 is an afterany408501/408502 terminal diagnostic and4-page illustrated
Korean report, pinned to9225c0a. It does not refit or score heldout data and
cannot mark R2 complete. Rendered pages must be visually reviewed before
sharing. Batch-means MCSE/ESS are rough within-chain estimates conditional
on stationarity; rank/folded split Rhat and all shared-ancestry limits remain
explicit. A terminal report is not automatic production authorization.

408508 COMPLETED6m34s: the whole1414-row N256 GL2 raw score exactly matches
the saved affine-pilot state (5978.923105883818). Total18,843,365 components
retained; maximum batch1,069,371; host5.89GiB. This checks value equivalence,
not a new derivative or GL4 accuracy. New row-streamed code is separate from
the running chain source. It shares one spatial tree but refreshes physical
source support, keeping each observational row's full normalization intact.

While chains run, compare GL2/GL4 count+raw scores at the saved initial and
two-accepted-move states, with NO gravity rerun or refit. One bounded A100/H100
job,2CPU24GiB/150min, estimated host<=20GiB and compiled20% device margin;
full source4 counts plus row-streamed source4 raw marks. This can flag a
state-dependent numerical correction early, but cannot replace sensitivity
on usable posterior draws. No reweighting/ESS claim from two pilot states.
One illustrated PDF will show row-streaming equivalence, count/raw changes,
per-row examples and the change between states; view before sharing.

The pair job is408509, source4501657, on Slurm. Initial GL4 count estimated
device4.05GiB; its actual runtime/score and raw GL4 result remain pending.
408501 and408502 each completed one accepted L4 WARMUP proposal. Their
low-band powers are .704224/.702354 and Hamiltonian errors -19.976/-23.464;
the large negative errors include fine/proxy changes -14.675/-16.469. Movement
is real but a fixed local affine correction may degrade away from its anchor.
Track that directly; high early acceptance is not evidence of stationary mixing.

408509 COMPLETED1h55m17s, host7.25GiB. Full sourceGL4 raw component totals
150,581,357/150,695,010 were retained via row batches, max8,549,503 per batch.
Initial GL4-minus-GL2 count/raw scores +.27241766/+.09562372, sum+.36804138;
after the two accepted pilot moves +.22674394/+.07510420, sum+.30184814.
The between-state correction change is-.06619324nat. This does not expose a
large discrepancy on these TWO NEARBY PILOT states, but is NOT a bound for
the posterior or a reweighted-field/UQ result. Count GL4 costs2387s/state on
A100; row-streamed raw1169/944s. No new PM evolution or heldout evaluation.
The one-page actual-example PDF in `r2_n256_gl4_pair_v1` was rendered and
visually reviewed, and the review flag updated. It includes row-streaming
equality, per-row raw examples and both-state count/raw score changes.

408565 COMPLETED3s, source60fe8b4: read-only comparison of immutable affine
anchors at the initial and two-move states, with zero new target evaluations.
The measured fine-minus-fixed-affine change -6.81034nat is close to the
gradient-secant trapezoidal prediction -7.03873 (difference-.22838). IC
coordinate contribution -7.03327 dominates the prediction; nuisance-coordinate
contribution -.005459. This coordinate split is not causal separation of
cross-Hessian terms. It supports changing local curvature along this ONE
displacement, not an identified full Hessian or a validated new kernel.
Simple Fourier-shell scalar secants leave96.5–100% gradient residual across
shells. Therefore do NOT add an unvalidated isotropic/spectral rescaling or
claim it solves transport. Keep the existing running chains, check subsequent
mixing/force drift, and use more evidence if a proposal repair becomes needed.

At this point A/B each have6/6 accepted discarded-warmup proposals, restricted
low-band powers .746770/.744296 and still drifting. Both have entered the
planned L8 portion of warmup. Actual posterior moments and R2 are NOT complete.
