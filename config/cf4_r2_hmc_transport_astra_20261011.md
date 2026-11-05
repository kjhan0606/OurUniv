# R2 HMC transport: Astra mid-course advice and driver disposition

Reviewer: existing user-authorized Astra science agent,
`/root/astra_cf4_science_audit_20261010`. Verdict: CONDITIONAL GO for one
sequential transport bundle, not production sampling or R2 exit approval.
Driver checked the inspected kernel/adapter and both terminal pilot results.

## Adopted calculation

Resume terminal q/value/gradient/RNG of 418838, never MAP best_parameters.
Replay the terminal value, component terms and gradient once. Keep the same
conditional-v6 matched-GL2 target and mass6000 / nuisance diag(D squared).
One typed GPU, 96 GiB host, six-hour Slurm, 5.5-hour application cap and
49 exact calls including replay. Warmup lengths: 2,2,4,4,6,6. Four diagnostic
retained transitions independently choose length 4 or 6 with equal probability.
Initial step .1; during warmup only multiply by
exp(.25*(accept_probability-.8)/sqrt(warmup_index)), clip [.01,.15]. Freeze
after six. No metric learning from these six points.

Require full-trajectory force/time headroom before launch, coordinate budget
exceptions, and checkpoint only complete states including rejection repeats.
Save q, gradient, value, current component terms, RNG, phase/index, step and
target/metric fingerprint. Partial endpoints are not rejections or samples;
capacity/compilation failures remain failures. Diagnostic traces include
signed energy errors, force cost, nuisance coordinates, fundamental sine and
cosine modes, fixed low-k band powers and proposal-metric-normalized motion.
Proposal normalization is not posterior normalization. No credible intervals,
reliable ESS/MCSE or convergence claim from four retained states.

## Driver assessment / next decision

Adopted: longer sequential trajectories test transport rather than endlessly
restarting two-proposal ladders. Both completed .1 transitions had negative
energy errors and accepted; neither proves stationarity or metric correctness.
Unknown posterior widths prevent diagnosing the small low-k jumps alone.
Preflight resume/schedule/budget correctness with small Slurm tests. Reuse
compact target/kernel; avoid independent simulations and mega-script branches.
If useful transport is demonstrated, proceed toward genuinely different
initial states, independent streams, substantive multi-chain sampling and
discarded equilibration. Diagnose a concrete failure rather than repeat an
identical budget if the fixed segment stalls or support/resources fail.

Q-GOAL: conditional pass for the same NEW N256/384 z=0 field. MW/M31 roles
remain ambiguous, M33 unresolved; their eventual observations must constrain
identified roles in the same evolved LG field <=.3 cMpc/h. Truth identities
are calibration/evaluation only, never candidate selection or seeding.
Conditional single-mark subset does not identify unrestricted CF4 incidence.
Mark availability, count selection, tracer/RSD response, shared calibration
and numerical sensitivity still require checks; freeze heldout protocol and
development-leakage assessment before opening heldout measurements.

Q-LEAN: no new catalogue search, TNG/RAMSES run or covariance prerequisite.
R2 remains open. No exit email and no R3 are approved by this advice.
