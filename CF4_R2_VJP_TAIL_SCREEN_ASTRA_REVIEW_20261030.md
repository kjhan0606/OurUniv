# Astra mid-course advice — R2 VJP tail screen (2026-10-30)

Verdict: the result supports computational feasibility, not numerical stability or a reliable informed spectrum. It does not justify closing the Fisher-reference branch, but I recommend one bounded stability/concentration screen before releasing discovery/validation sketches. This is advice based on the supplied measurements, not a source-verified audit or R2-exit approval.

The distinction matters mathematically. For positive-row intensity Jacobian \(J\), the sketched operator is \(A=\operatorname{diag}(\lambda^{-1/2})J\). Row \(i\)’s actual contribution to its squared Frobenius norm is
\[
\|J_i\|^2/\lambda_i.
\]
When \(J_i=\lambda_i\nabla\log\lambda_i\), this becomes \(\lambda_i\|\nabla\log\lambda_i\|^2\). Thus \(2.26\times10^{18}\) amplification is a warning about intermediate arithmetic, not evidence that tiny-intensity rows dominate. A single finite Gaussian VJP cannot resolve that question.

Minimal next check:

- Freeze the same anchor, training rows, positive-row convention and operator. Define a diagnostic tail as the lowest-intensity 1% of positive rows, with deterministic tie-breaking. This partition changes neither weights nor target; no floor or row removal.
- Use two predeclared Gaussian seeds—the original and one independent seed. For each, compute full, tail-only and complementary-row VJPs: six pullbacks total. Repeat the original full pullback once: seven total.
- Check repeatability and the linear identity \(g_{\rm full}=g_{\rm tail}+g_{\rm rest}\), using the implementation’s actual dtype and numerical tolerance fixed before submission. Normalize decomposition error by \(\|g_{\rm tail}\|+\|g_{\rm rest}\|\), so legitimate cancellation does not manufacture a failure.
- Report full/tail/rest norms, their cross term, and \(r=\|g_{\rm tail}\|^2/(\|g_{\rm tail}\|^2+\|g_{\rm rest}\|^2)\), separately for IC and tracer coordinates. Do not describe two realizations as a precise estimate of trace share, rank or spectral concentration.

Suggested hard budget: seven pullbacks, no parameter changes or additional seeds, one-hour application cap, 75-minute Slurm cap. Retain the verified resource class and existing guarded memory allocation; do not turn this suggestion into a new larger allocation. The observed pullback cost makes this proportionate, although setup/replay overhead must be included.

Stop on nonfinite results, failed repeat/decomposition tolerance, resource guard or cap. Report an unresolved numerical failure; do not introduce an intensity floor or automatically retry. A large tail share alone is **not** a failure: genuine information may be concentrated. It means discovery and independent validation must retain and explicitly interrogate that contribution. Passing the screen permits a separately budgeted sketch experiment, not a sampler. Independent spectral/residual validation and full-target correction remain necessary; exact correction protects the target, not efficiency.

Q-GOAL: yes, this addresses a concrete obstacle to the conditional-v6 N256 z=0 density/velocity posterior. It establishes no posterior, calibrated uncertainty, held-out prediction or R2 completion. MW/M31 roles remain ambiguous and M33 unresolved. Their future observables must constrain candidate roles identified from the same NEW evolved LG field at ≤0.3 cMpc/h; native truth identities remain calibration/evaluation labels only.

Q-LEAN: yes, if limited to this single frozen-anchor screen and then a decision. It is not proportionate to launch a tail-threshold ladder, precision ladder or extensive concentration study. There is currently insufficient evidence either to declare harmful tail dominance or to close this proposal branch.
