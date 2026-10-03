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

Jobs409590(A) and409591(B) both completed with exit0 in2:52:51 and2:55:33;
MaxRSS14.2/14.4GiB against24GiB requested. All16 exact-target transitions
completed. A accepted8/8, mean acceptance probability.9072; B accepted5/8,
mean probability.8857. Energy errors remained between-.282 and+.327. Accepted
IC-white jump RMS was about.0661 per move. Total white mean-square changed only
.996829->.996840 in A and.996939->.996914 in B.

This is a valid short-kernel check, not a stationary posterior. The three
tracked fundamental complex modes changed by only about1e-3–3e-3 in absolute
components across eight transitions (maximum component-relative change about
.6%). With the fixed metric's fundamental mass6000, inverse mass is1/6000;
the .08 one-step prior-flow phase advance is only.08/sqrt(6000)=.00103rad.
This is consistent with the observed weak low-k movement and makes a long run
with the same metric a poor next spend. It does not estimate ESS or prove that
the target posterior itself is narrow.

Driver next: run a bounded exact-GL2 metric sensitivity from these accepted
checkpoints, changing only the inverse-Laplacian fundamental-mass parameter
6000->600 (tenfold larger inverse mass at |k|=1, with the corresponding
smooth change across low modes) and using four leapfrog steps per proposal.
Keep the target, step .08, nuisance mass and MH correction fixed; compare
energy error, exact acceptance and fundamental-mode displacement. This
preserves the posterior target but tests a sampler geometry hypothesis; it is
not a production chain.
No external audit was performed for the 2026-10-01 short-chain result; the
bounded driver assessment and Q-GOAL/Q-LEAN rationale are recorded above.
Do not submit a longer run until this sensitivity is read.

## 2026-10-02 result and disposition

Jobs409763/409764 both completed/exit0 on H100. Each ran one state-independent
exact-GL2 proposal from the accepted terminal checkpoint of409590/409591,
using the unchanged target, step.08, four integrations, nuisance inverse
mass1e-5, and fundamental-mass parameter600. They were valid MH rejections:

| Chain | delta H | Acceptance probability | Force evaluations | White-field jump |
| --- | ---: | ---: | ---: | ---: |
| A | 6.5127 | .0014845 | 4 | 0 |
| B | 19.1332 | 4.90e-9 | 4 | 0 |

The endpoint energy/gradient primal check passed. Estimated device peaks were
30.56/30.07GiB under the69.81GiB device limit; Slurm host MaxRSS was
12.2/15.7GiB under24GiB. Both took about1h23–1h27. This is evidence against
the tested600/.08/four-integration proposal configuration, not target failure,
posterior nonexistence, or a metric-family conclusion. No heldout data or new
gravity evolution was used, and no map or posterior sample was promoted.

Code review found a reporting defect: the trace hardcoded
`integration_steps=1`, although the call passed four and `force_evaluations`
was four. The local recorder now writes the configured integration count.
The chain runner also previously reseeded on checkpoint continuation despite
already saving RNG state. It now restores that state by default, supports an
explicit seed override for deliberate matched replay, and records whether a
seed or restored state was used. Focused HMC tests pass9/9; Python compile,
batch-script syntax, and `git diff --check` pass. Historical results remain
unchanged.

Fable's read-only advice was usable. It judged the result a sampler-tuning
failure rather than an astrophysical discovery (driver agrees); Q-GOAL is
directly aligned with R2, while the original comparison was not fully lean
because metric and integration count changed together. The driver's
independent review rejects attributing the failure to metric600: the prior
6000 same-checkpoint chain had one integration, and the existing eight-step
6000 force trials used other states/momenta. Therefore the smallest useful
control is one exact-GL2 proposal per chain at6000, step.08/four integrations,
reusing the same initial states and RNG seeds2026100201/2026100202. Compared
with the already-saved600/four-step proposals, this isolates the metric at
fixed trajectory settings without rerunning the600 arm. Estimated cost is
about2.8 H100 GPU-hours total (two independent 4h wall-capped jobs,4CPU and
24GiB host each). The device estimate retains the20% guard. No per-step
Hamiltonian instrumentation or three-arm step-size sweep is adopted; if this
single contrast is inconclusive, stop and redesign the sampler rather than
automatically expanding the tuning grid.

The R2 science target itself remains provisional: no stationary posterior,
ESS, heldout predictive validation, or completed CF4 z=0 density/velocity
delivery follows. R3 same-field MW/M31/M33 identification has not started;
MW/M31 remain role-ambiguous, M33 unresolved, and their observations must
later constrain this same NEW field. Native truth identities remain evaluation
only.
