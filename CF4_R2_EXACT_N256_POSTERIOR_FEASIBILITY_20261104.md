# R2 exact-N256 posterior cost decision — 2026-11-04

## Evidence used

The latest exact-target timing comes from Syntax job `447829`, H100 with typed
`gpu:H100:1`. Its two unchanged retained-block HMC transitions took 4,766.57 s
and 3,457.48 s, respectively, or **1.1422 H100 GPU-hours per transition** on
average. The job elapsed 2.46 hours including checkpoint replay/startup and
recorded 21 exact target evaluations. The two states were dependent and
explicitly discarded for inference. The longer historical trace remains
non-stationary by available summaries and has no independent chains, R-hat,
ESS, or calibrated MC error.

The following are arithmetic sensitivity cases, not adequate sample-size
recommendations. They exclude queue time, setup, replay, restarts, and
cross-GPU speed differences:

| Four-chain scenario | Total transitions | H100-equivalent GPU hours | GPU-days | Elapsed days if four GPUs run continuously |
| --- | ---: | ---: | ---: | ---: |
| 64 warmup + 128 retained per chain | 768 | 877 | 36.5 | 9.1 |
| 64 warmup + 256 retained per chain | 1,280 | 1,462 | 60.9 | 15.2 |
| 128 warmup + 256 retained per chain | 1,536 | 1,754 | 73.1 | 18.3 |
| 128 warmup + 512 retained per chain | 2,560 | 2,924 | 121.8 | 30.5 |

Even these retained counts are not known to yield useful ESS or stable
field-level Monte Carlo errors. The run's exact-target kernel has moved the
measured modes, but field and nuisance summaries continued to drift. In
addition, the current target remains conditional on the frozen selected/linked
v6 observations and lacks a calibrated full-survey selection/group law. A
long chain would not repair that observation-model limitation.

## Driver decision

**NO-GO for launching an immediate multi-chain production calculation.** The
decision is not that the scientific target is impossible. It is that no
defensible transition budget or MC-error contract is yet available, while the
measured cost is already hundreds to thousands of GPU-hours and no
convergence guarantee follows from the illustrative retained counts. A
shortened one-chain run would repeat the same insufficient evidence. Do not
launch another short sampler pilot, reuse the DA surrogate, retune the closed
configuration, or relabel a MAP/transport diagnostic as a posterior.

The next production proposal must first specify: independent full-prior
initializations; warmup termination and a frozen retained phase; diagnostics
for density/velocity summaries, Fourier bands, and nuisance parameters;
MCSE/ESS and joint heldout-log-score stability targets; retained-state and
rejection-repeat persistence; an aggregate GPU-hour ceiling; and a terminal
rule if those criteria fail. The chosen ceiling must be explicitly acceptable
for a conditional-v6 result, with the selection/group limitations disclosed.
The current evidence does not support a smaller, scientifically sufficient
budget. No new solver, likelihood, or user resource budget is assumed here.

The next in-scope action is a joint science/resource decision on whether to
fund that large exact-target posterior run or first pursue a specifically
justified cost reduction. Routine additional sampler screens are closed.
R2 remains open/NO-GO; no field or posterior is promoted and R3 has not
started.

**Q-GOAL:** the required result is still the CF4+galaxy-conditioned z=0
density/velocity posterior at 1.5 cMpc/h, uncertainty and heldout prediction,
followed by same-field LG constraints at `<=0.3 cMpc/h`. MW/M31 roles remain
ambiguous and M33 unresolved; their observables must constrain those roles on
the same NEW evolved LG field. Native truth identities are calibration/
evaluation labels only.

**Q-LEAN:** the cost ruling reuses the measured exact HMC timings and the
closed surrogate result. It creates no new diagnostic ladder, target
evaluation, simulation, heldout read, or unnecessary gate code.
