# R2 delayed-acceptance count-surrogate result and mid-course review — 2026-11-04

## Execution and result

Syntax Slurm job `449120` completed successfully (`COMPLETED`, exit `0:0`,
01:01:30) on `h100` with `gpu:H100:1`, 24 GiB host memory, and the fixed
4:30 wall cap. H200 was occupied (`MIXED`; three typed GPUs free but no idle
H200 node), so the permitted H100 fallback was used. The run used source
commit `cdc7f53d30977315f826a2f33bdb6deea132726f` and wrote its authoritative
record to
`/gpfs/kjhan/CF4/z0_density/r2_da_count_surrogate_screen_20261103/449120/result.json`.
The four startup regression suites passed (5, 11, 9, and 2 tests).

This was a fixed-anchor sampler-cost experiment on the discarded N256,
384 cMpc/h conditional-v6 state-60 checkpoint. It updated the N128-restricted
Fourier component and nine tracer nuisances; the fine orthogonal complement
and 15 FP-population nuisances stayed fixed. The target contained 47,121
training count occurrences, 37,951 keys, and 1,414 linked FP marks. No heldout
outcomes were read.

The anchor replay error was `4.47e-8 nat`, below `1e-7`. Across 65 surrogate
gradients the fixed `1e-30` event-log floor was never used; the minimum event
rate was `6.62e-5`. All eight fixed proposals passed stage one, but all eight
failed the exact N256 stage-two correction. The N128 count score improved by
`+201` to `+428 nat`; the fine N256 count score instead worsened by
`-4,408` to `-4,832 nat`. FP changes were only `-1.66` to `+1.85 nat`, and
`log(alpha2)` ranged from `-4,717` to `-5,033`. Every rejected move retained
zero displacement. The correction was computed in log space; a printed
probability of zero is underflow, not the acceptance decision.

The implementation identities passed: complement preservation was at most
`4.8e-16` RMS, and prior/discrepancy accounting errors were at most
`3.92e-9 nat`. The surrogate compile estimate was 19.70 GiB on an H100 with a
69.81 GiB limit; application host peak was 9.87 GiB. The run took 3,642 s of
the 14,400-s application allowance. Median N128 gradient time was about
32.9 s; median fine scalar time was about 136 s.

Status remains `R2_DA_COUNT_SURROGATE_COMPLETED_NOT_POSTERIOR`: no posterior
sample, uncertainty, calibrated field, or heldout score was produced.

## Astra mid-course review and driver disposition

Astra's read-only review accepts this as a correctly executed negative result
and finds no implementation defect that explains the rejections. The review
verified the job accounting, source pin, regression suites, anchor replay,
log-space correction, floor status, Fourier lift, and prior/discrepancy
identities. It interprets the large discrepancy as a mismatch between this
fixed N128 count proposal and the fine N256 count target at this anchor. It
does **not** invalidate delayed acceptance generally, the fine target, or the
CF4 reconstruction goal.

The driver adopts closing this exact surrogate configuration without
retuning or extending it. No step-size, seed, resolution, or floor ladder is
justified. Astra recommends a costed GO/NO-GO decision for a genuine
independent-chain exact-N256 posterior using the existing exact-target HMC,
not another speculative short sampler screen. That advice is adopted with
one addition: cost and sampler diagnostics cannot compensate for the distinct
selection/group calibration limits in the frozen conditional-v6 target.

## Scientific scope and limits

This result is sampler engineering only, not the R2 z=0 density/velocity
posterior at 1.5 cMpc/h. The current conditional-v6 mark component describes
the selected linked marks; it does not identify the missing parent,
redshift-success, FP-quality, linkage, and group-inclusion processes. Any
posterior under this target must be labelled conditional-v6, not full-survey
calibrated CF4. Heldout scoring and posterior uncertainty remain unavailable
until there are valid retained draws and an explicit MC-error assessment.

MW/M31 remain role-ambiguous and M33 unresolved. Their observables must
eventually constrain those same roles on the same **NEW evolved LG field at
`<=0.3 cMpc/h`**. Native truth identities may be used for calibration or
evaluation only, never to seed or select generated-field candidates. The
present N256/384 cMpc/h grid samples 1.5 cMpc/h and does not deliver LG
resolution.

**Q-GOAL:** the screen supplies no R2 science product and does not reduce the
LG-resolution or component-identification requirements.

**Q-LEAN:** the single fixed surrogate and eight exact corrections answered
the declared question. Reuse the evidence; do not add another sampler screen
or diagnostic ladder.

This is a mid-course review, not an R2 exit audit, exit-email trigger, or R3
authorization.
