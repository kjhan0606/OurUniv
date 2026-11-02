# R2 rejected-shell direction diagnostic — result (2026-11-02)

## Execution and reproduction

- Syntax Slurm job447634 completed (`COMPLETED`, exit0,37:13) on `h100`,
  typed GRES `gpu:H100:1`, node `syn09`, from source commit
  `85f085af688711a4a5638fff663a0356661e35e2`.
- H200 was checked and occupied (5/8 allocated,3 typed GPUs free); H100 had3
  free typed GPUs and A100 had none. No manual node execution was used.
- Five exact target evaluations were used: the mandatory state38 replay, two
  matched value/gradient evaluations, and the two scalar symmetric-FD values.
  The frozen five-evaluation ceiling and45-minute application limit were
  respected. Slurm MaxRSS was11,362,032KiB; Python peak RSS11.15GiB. GPU
  peak33,070,866,944bytes was below the74,962,354,176-byte limit and the
  existing20%-headroom check passed.
- The accepted fundamental proposal and rejected `16<k^2<=64` shell proposal
  were reconstructed against job447590. Both target values matched exactly;
  all listed components matched exactly except count log-likelihood at the
  shell endpoint, differing by2.91e-11. The fundamental and shell log-alpha
  and forward/reverse proposal densities reproduced the stored values within
  the implementation tolerances; the recorded shell rejection was recovered.
- Raw result: `/gpfs/kjhan/CF4/z0_density/r2_mala_shell_direction_20261102/447634/result.json`.
  Slurm stdout/stderr are `/gpfs/kjhan/CF4/logs/cf4_r2_mala_fd_447634.out`
  and `.err`.

## Directional result

For the fixed unit direction `v=d/||d||` and the predeclared `h=1e-4`:

- Autodiff directional derivative: `-130.2329147293`.
- Central finite difference: `-130.2327960730`.
- Relative discrepancy, `|FD-AD|/max(|AD|,1)`: `9.11e-7`, below the frozen
  `1e-2` tolerance.
- For the original finite shell displacement, `grad U(x)·d=-289.9766`,
  whereas the fully evaluated `U(x+d)-U(x)=+120.7285`; finite-step minus
  first-order prediction is `+410.7051`. The endpoint gradient changes by
  relative L2 `0.2205`.

Thus the tested starting directional derivative is locally consistent with
the target value. The first-order change is nevertheless a poor prediction
for this one finite displacement. This is evidence of finite-step
nonlinearity along one direction, not a Hessian-spectrum estimate, a sampler
replacement, or evidence that all shell moves fail.

## Scientific disposition

Status: `SHELL_DIRECTION_DIAGNOSTIC_COMPLETE_FD_PASS_NOT_POSTERIOR`.
No transition was attempted or state/field promoted. No heldout values/marks
or native truth IDs were accessed. No posterior, uncertainty, heldout
prediction or LG map was produced. The N256/384 cMpc/h grid samples1.5
cMpc/h; it does not satisfy the eventual `<=0.3 cMpc/h` LG resolution.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must
ultimately constrain those same roles on the same NEW evolved LG field, with
native truth IDs limited to calibration/evaluation.

The fixed `epsilon=0.05` identity-metric shell MALA candidate remains closed.
The diagnostic supports checking finite-step proposal geometry before
redesign; it does not determine whether a different metric, conditional
proposal or existing HMC route will work. R2 remains open/NO-GO; R3 has not
started. A separate mid-course Astra advisory review of this finding is
pending; this is not the R2 exit audit.
