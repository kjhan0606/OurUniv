# R2 retained HMC block-jump result — 2026-11-03

## Outcome

Syntax Slurm job `447829` completed successfully (`COMPLETED`, exit `0:0`,
02:28:07; partition `h100`, typed request `gpu:H100:1`). It wrote the result to
`/gpfs/kjhan/CF4/z0_density/r2_hmc_retained_block_jumps_20261102/447829/`.
The source pin is `e623319d9c987099de93def954372dae3a3b6af9`. The authoritative
JSON reports a successful checkpoint replay, 21 exact target evaluations,
8,847 seconds of application time, and 16.251 GiB application host peak; Slurm
MaxRSS was 15,206,484 KiB. The selected H100 was queued only after all allowed
typed devices were occupied at submission.

This is an N256 grid in a 384 cMpc/h box (1.5 cMpc/h nominal cell spacing).
Both predeclared unchanged transitions from the discarded state-60 checkpoint
were accepted, and each transition's exact target-evaluation count matched its
force-evaluation count:

| Trace index | Steps | ΔH | Fundamental k²=1 RMS | HMC shell 16<k²≤64 RMS | Remaining IC RMS | 24 nuisance RMS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 12 | +0.536640 | 0.020812 | 0.113506 | 1.117153 | 0.014464 |
| 61 | 8 | −0.192212 | 0.009283 | 0.078486 | 0.771218 | 0.011151 |

These are post-Metropolis retained jumps per canonical coordinate, not
proposed distances. Both rows moved the measured Fourier blocks; neither row
was rejected. The accumulated trace contains 62 states, all explicitly marked
discarded. Its objective rises from 8,213,134 at index 0 to 8,516,907 at index
61. Most of that change is the IC-prior NLL moving from 8,077,223 to 8,386,209
while the training-count NLL improves from 141,887 to 136,666; it is not by
itself evidence of sampler failure. The later trace is not demonstrably
stationary either: `4<integer_k<=8` white-IC band power increases from 0.48722
at index 37 to 0.55163 at index 61, while the available density mean-square
readout increases from 11.8512 to 13.1449. The trace has no independent chains,
R-hat, ESS, stationarity certificate, or calibrated Monte Carlo error estimate.
The observer-sphere readout also switches geometry between the inherited
index-59 record (cell centres) and indices 60–61 (native nodes); those sphere
summaries are excluded from across-index comparison.
Its recorded transition times sum to 48.11 hours across the accumulated
history, excluding setup/replays and inter-job/queue time; the two new
transitions took 79.44 and 57.62 minutes.

The machine status is
`CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR`. No posterior sample,
posterior uncertainty, held-out score, or promoted density/velocity field was
produced. Held-out counts and marks were not read. The HMC kernel demonstrably
moves the measured modes at these two dependent states, but its posterior
adequacy remains unestablished. Do not resubmit this completed diagnostic or
promote its checkpoint as a draw.

## R2 disposition

This closes the planned two-transition measurement gap, not R2. R2 remains
open/NO-GO; R3 has not started. The next scientific decision is whether the
unchanged exact-target kernel can support a properly initialized, multi-chain
posterior run with convergence/MC-error diagnostics at a defensible cost, or
whether a substantive sampler redesign is necessary. Another two-step
transport readout by itself would not answer that decision.

**Q-GOAL:** the completed calculation is sampler engineering only. The required
R2 deliverable remains an observation-conditioned z=0 density/velocity
posterior at 1.5 cMpc/h, with uncertainty, held-out prediction, and calibrated
observation and sampler errors. MW/M31 roles remain ambiguous and M33 remains
unresolved; their observables must constrain those roles on the same NEW
evolved LG field at ≤0.3 cMpc/h. Native truth identities may be used for
calibration/evaluation only, never to seed or select generated-field
candidates. The present N256 grid does not deliver LG resolution.

**Q-LEAN:** reuse this completed block readout and the existing exact-target
kernel; do not repeat this diagnostic, launch another metric/scale ladder, or
read held-out outcomes before a valid sampling/validation contract. Any next
large sampler calculation requires a concrete cost and multi-chain diagnostic
plan; this result alone is not authorization for an unbounded chain.
