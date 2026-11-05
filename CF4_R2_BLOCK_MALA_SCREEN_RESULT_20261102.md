# R2 fixed-block MALA screen — result (2026-11-02)

## Execution

- Slurm job447590: `COMPLETED`, exit `0:0`, elapsed1:10:06; selected
  partition `h100`, typed GRES `gpu:H100:1` on `syn09`.
- H200 was occupied at submission (4/8 allocated); H100 had3/10 GPUs
  available; A100 was full (16/16 allocated). The permitted H100 mode was
  selected and recorded in the job environment.
- Source commit `eb0776e1b70fe03f190d90621550e77744161ecb`, pushed to
  `origin/agent/freeze-zoom-pipeline` before submission.
- Result: `/gpfs/kjhan/CF4/z0_density/r2_block_mala_screen_20261102/447590/result.json`.
  It is the authoritative per-proposal record. No checkpoint or raw result was
  changed.
- All10 exact full-target value+gradient evaluations completed, including both
  anchor replays. Replay errors: state38 objective `9.31e-9`, gradient max
  `1.01e-11` (relative L2 `5.16e-15`); state60 objective `4.47e-8`, gradient
  max `5.46e-12` (relative L2 `5.79e-15`).
- Python peak host RSS11.72GiB; Slurm batch MaxRSS12,022,436KiB. Allocated GPU
  limit74,962,354,176bytes; measured peak33,056,599,296bytes, versus the
  frozen43,799,579,648-byte prior peak and52,559,495,578-byte 20%-headroom
  minimum. No memory guard tripped.
- Status: `BLOCK_MALA_FIXED_SCALE_ZERO_ACCEPTANCE_IN_BLOCK_NOT_POSTERIOR`.
  Exactly8 proposals and10 target evaluations; held-out values/marks0, native
  truth IDs0, posterior sample/UQ false, no field written, R2 incomplete, R3
  not started.

## Outcomes

| Anchor (dependent/discarded) | Block | Step | Accepted | `log α` | Proposed RMS | Retained RMS | `ΔU` |
|---|---|---:|---:|---:|---:|---:|---:|
| state38 | `k²=1` (6D) | 1 | yes | +0.165 | 0.00782 | 0.00782 | −0.956 |
| state38 | `k²=1` (6D) | 2 | no | −0.232 | 0.01083 | 0 | +2.656 |
| state38 | `16<k²≤64` (1,852D) | 1 | no | −110.980 | 0.05174 | 0 | +120.728 |
| state38 | `16<k²≤64` (1,852D) | 2 | no | −85.989 | 0.05262 | 0 | +99.610 |
| state60 | `k²=1` (6D) | 1 | yes | +0.037 | 0.01118 | 0.01118 | −0.222 |
| state60 | `k²=1` (6D) | 2 | yes | +0.047 | 0.00973 | 0.00973 | −0.711 |
| state60 | `16<k²≤64` (1,852D) | 1 | no | −147.212 | 0.05133 | 0 | +166.912 |
| state60 | `16<k²≤64` (1,852D) | 2 | no | −130.602 | 0.05126 | 0 | +190.017 |

Every proposed/retained distinction is explicit in the raw report; rejected
proposals retain zero displacement. The target-component decomposition closes
to the full `ΔU` within about `1.1e-9` or better on every move.

The four fundamental-block moves had3/4 acceptances, and all three accepted
moves decreased `U`; these two starting anchors are dependent, discarded HMC
transport states. This is consistent with short downhill relaxation and is
not stationarity, mixing, effective sample size, or posterior evidence.

The measured HMC-drift shell had0/4 acceptances at the predeclared
`epsilon=0.05`. Its proposed `ΔU` was +99.6 to +190.0, dominated by count-NLL
increases of +96.6 to +187.5; IC-prior increases were +0.26 to +3.13, while
the FP term was comparatively small and changed sign. The forward/reverse
proposal correction recovered only about +9.7 to +59.4 log units, leaving
`log α` between −86.0 and −147.2. This is negative evidence about this fixed
block/scale candidate, not proof that all informed shell transitions are
impossible. There was no automatic step-size tuning, retry, or ladder.

Each proposal's exact target evaluation took385.8–407.0s; rebuilding live
support took82.9–100.5s. This was the registered training-only conditional-v6
target, with unchanged support rule rebuilt at every candidate. No separate
PM/RAMSES simulation, held-out scoring, or truth-ID inspection was performed.

## Scientific disposition

This sampler-mechanics screen does not deliver the conditional z=0 posterior,
uncertainty, held-out prediction, calibration, or an LG density map. N256 in
384cMpc/h has1.5cMpc/h field sampling, not the eventual `<=0.3cMpc/h` LG
resolution. MW/M31 roles remain ambiguous and M33 unresolved; their observables
must eventually constrain those same roles on the same NEW evolved LG field,
with native truth IDs used only for calibration/evaluation. This screen does
not identify or refine any LG component.

**R2 remains open/NO-GO. R3 is not authorized.** A mid-course Astra review of
this result and the proportionate next R2 action was requested; that advice is
not an exit audit and cannot close R2.
