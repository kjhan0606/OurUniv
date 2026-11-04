# R2 support-chunk replay — 2026-11-04

## Run and exactness

Syntax job `449672` completed on `syn102`, partition `a100`, with
`gpu:A100:1` (exit 0, elapsed 13:58). At submission, H200 had no idle/free
typed GPU, H100 had no free typed GPU, and A100 had three; the selected mode
and snapshot are preserved in the raw result. The job used commit
`828cbf9fb87b2f9c969477bd280183f8810e87aa` and a 24 GiB host-memory request.

The run replayed the same discarded N256, 384 cMpc/h conditional-v6 HMC
checkpoint used by job `447829`, changing only `support_chunk_cells` from
16,384 to 32,768. It ran exactly one value+gradient target evaluation,
`start_index=end_index=60`, hence **zero HMC transitions**. Four startup
regression suites passed (11, 11, 9 and 3 tests). Checkpoint replay passed;
the exact objective is `8,515,507.148618123`, matching job `447829`'s anchor
replay. Gradient RMS is `1.1360368589257783`; gradient infinity norm is
`1121.3421087652694`, consistent with the saved exact-target checkpoint.
The rebuilt support has the same 28,440,519 components and largest candidate
list of 44,018 cells. No rows were dropped and no likelihood/prior rule changed.

Raw machine record:
`/gpfs/kjhan/CF4/z0_density/r2_support_chunk_bench_20261104/449672/result.json`.

## Cost interpretation and disposition

The 32,768-chunk replay reports 761.81 s for the target evaluation and 92.13 s
for support construction. Job `447829`'s 16,384-chunk anchor replay reports
594.21 s and 98.42 s, respectively. These wall times are **not a valid A/B**:
the candidate ran on A100 while the reference ran on H100, and both are first
evaluations in their respective processes with compilation/setup effects.
The larger chunk also raises the padded work for a 44,018-cell row from three
16,384-wide chunks (49,152 padded cells) to two 32,768-wide chunks (65,536
padded cells), trading fewer calls for more padded work. The single replay
therefore establishes target/gradient equivalence, not a runtime improvement.

Keep 16,384 as the default. Close 32,768 as an unproven optimization; do not
launch a repeated chunk-size ladder or HMC transitions from this checkpoint.
This result does not meet any sampling, posterior, uncertainty, held-out, or
observation-calibration requirement and does not promote the field.

**Q-GOAL:** R2 remains open/NO-GO: there is no calibrated z=0 posterior,
uncertainty, held-out prediction, or full observation/sampler calibration.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must
constrain those roles on the same NEW evolved LG field at `<=0.3 cMpc/h`, with
native truth identities used only for calibration/evaluation.

**Q-LEAN:** one exact target replay tested a specifically motivated runtime
change; no transition, tuning sequence, held-out read, or separate simulation
was performed. No performance claim is made from unlike GPU runs.

## Next R2 action

Stop runtime micro-tuning. Resolve the observation-model calibration
limitation and define a defensible field-level Monte Carlo error/resource
contract before proposing the exact multi-chain target run. Do not read held-out
outcomes until valid sampling and the calibration contract are established.
