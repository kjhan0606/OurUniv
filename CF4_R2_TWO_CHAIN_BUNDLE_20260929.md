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
