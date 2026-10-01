# R2 exact-GL2 short-chain bundle — 2026-10-01

## Stage and purpose

R1 is complete; **R2, the actual z=0 field posterior, remains active and
incomplete**. R3 is same-field MW/M31/M33 identification, followed by R4
precision evolution and R5 zoom ICs. The current N256 mesh is 1.5 cMpc/h;
this is not the final <=0.3 cMpc/h LG zoom.

Job409484 completed the valid state-independent exact-GL2 one-step pilot at
two predetermined accepted states. Both proposals were accepted (p=.9896 and
.9311), with IC-white displacement RMS .0661/.0660. The matched saved GL2
eight-step trials had larger RMS .522 and p=.4798/.5505, but each arm has only
one momentum per endpoint; this is not a mixing comparison. Exact one-step
performance is plausible enough to test as an actual sequence, not enough to
authorize a long posterior run.

## Bounded computation

- Run one A and one B chain from the already accepted R2 checkpoints; those
  starts share history and are not independent initializations.
- Each chain performs at most eight sequential HMC transitions with the exact,
  state-independent GL2 target/force, one leapfrog step, fixed .08 step and
  the same fixed metric. No warmup, adaptation, force surrogate or target
  change. Rejected proposals retain the prior state, energy and gradient.
- Check the loaded state against the exact GL2 energy, use an independently
  evaluated GL2 endpoint value for MH, save an atomic accepted-state/gradient
  checkpoint after every transition, and record proposal decisions, energy
  error, accepted white-field jump, white power and fundamental modes.
- Two Slurm H100 tasks; each requests4 CPUs,24GiB host memory,3.5h application
  cap and4h wall cap. The previous exact-gradient peak was30.56GiB against69.81GiB device
  capacity (20% guard retained); the previous host MaxRSS was16.6GiB, so24GiB
  reserves >20% over that measured peak. Stop rather than exceed the cap.
- No new gravity evolution, data fit, held-out score, likelihood/prior change,
  or production posterior claim. Eight transitions can diagnose a short chain,
  not stationarity, useful ESS, or uncertainty. No density map is promoted.

## Goal and limits

Q-GOAL: this is still R2 sampling of the CF4-conditioned current field, not a
detour into IC-only generation. Q-LEAN: it reuses the tested exact target,
metric and states; the only new work is a bounded chain runner and the two
small Slurm pilots. MW/M31 remain role-ambiguous and M33 is unresolved; this
bundle does not identify them. Their later observables must arise from and
constrain the same NEW field. Native truth identities may evaluate/calibrate,
never seed or select candidates.

## Result

Pending Slurm completion. No posterior promotion is allowed from scheduler
success alone. Record acceptance, repeated rejected states, white-field drift,
time/force cost and any failure here and in the active master plan before
choosing a longer run. A future multi-hour/long-chain allocation is a separate
large-calculation decision; current short pilots do not authorize it.
