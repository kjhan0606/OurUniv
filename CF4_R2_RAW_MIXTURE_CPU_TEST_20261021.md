# R2 raw-mixture regression test — 2026-10-21

Slurm job `431594` ran `tests/test_cf4_r2_raw_mixture_connection.py -v` on
`a100_pcie` / `syn103` as a CPU-only job (`JAX_PLATFORMS=cpu`, 2 CPUs,
12 GiB memory request, 20-minute limit). The job explicitly printed and
asserted `jax_backend=cpu`, then completed in 57 seconds with exit code 0 and
peak RSS 986,720 KiB.

All five tests passed: unnormalized component-mixture/streaming equivalence,
source-conditioning versus FP-mark radius separation, explicit observation
radius propagation, per-row linked-point radius handling, and packed-row
variance-adjoint finite-difference agreement. This resolves the earlier
incomplete regression attempt without initializing a GPU on the login node.

Scope: this is a code regression only. It creates no fitted state, posterior,
held-out score, calibration result, density map, or production IC; no held-out
measurements were read. R2 remains NO-GO.

At recording time, the separate ESS feasibility job `431345` remains pending
on `h200` for `QOSGrpGRES`. Its revision-guarded files, including
`CF4_MASTER_PLAN.md`, were left unchanged; do not duplicate or cancel that
job.
