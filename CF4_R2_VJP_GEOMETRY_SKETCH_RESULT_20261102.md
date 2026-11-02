# R2 rank-4 VJP geometry sketch — result (2026-11-02)

**Disposition: close this rank-4 proposal candidate. R2 remains NO-GO.**

Syntax Slurm job447447 ran on H200 (`gpu:H200:1`, `syn104`) for36:01 and
exited1 at the predeclared discovery-subspace gate. The result is
`/gpfs/kjhan/CF4/z0_density/r2_intensity_vjp_geometry_sketch_20261102/447447/result.json`;
stderr records `DISCOVERY_SUBSPACE_GATE_FAILED_STOP`. It used source commit
`a0a61c0c9b99656843bbca49c21850bab8814ac5` and the fixed N256/384 cMpc/h
conditional-v6 anchor. All16 registered discovery pullbacks completed in
1994.50s total; their individual pullbacks after the first took about121.8s.

## Gate result

The registered8-vs-8 comparison formed rank-4 spans independently from the
first and second discovery halves. Their canonical correlations, in descending
order, were `[0.3922, 0.2069, 0.1106, 0.02269]`. The minimum is far below the
predeclared0.80 threshold. Both half bases and the combined basis were
numerically orthonormal (Frobenius errors `3.37e-15`, `1.14e-14`, and
`5.97e-15`). The failure is therefore not a QR orthogonality defect: this
fixed rank-4 range is not reproducible across the independent discovery
halves.

The16 singular values of the *finite random-probe matrix* run from1463.6 to
754.9 with no sharp drop at rank4. These are not eigenvalues of the Fisher
operator and do not establish its spectrum or global rank. In combination
with the failed independent-span comparison, they support rejecting this
low-rank proposal, not declaring the likelihood or project impossible.

The script stopped as designed. It did not repeat the first probe, form a
rank-4 artifact, or run any of64 independent validation pullbacks. The
`F_p` check, sampler, posterior, uncertainty and held-out evaluation were
not run; held-out outcomes/scores loaded were0. Do not increase this
candidate's rank/probe count, relax its gates, or rerun it.

## Audit of the registered design and resources

The planned directional interval gate was internally infeasible at its
registered sample size. With64 zero-mean Gaussian projections and four
two-sided Bonferroni intervals for family-wise95% coverage, the exact
chi-square interval relative half-width is `0.630972` for every positive
directional estimate, exceeding the required `0.50`. This was independently
verified by calling the committed interval helper with64 unit samples. Under
the same interval definition and threshold, at least91 independent
validation probes would be needed for precision alone (`0.496622` at91;
`0.500269` at90). This is a design-feasibility fact, not authority to add
probes to the failed candidate; any future plan must check gate feasibility
before submission.

The report's Python `ru_maxrss` host peak is16.9447GiB, above the12.5GiB
estimate. A20GiB request does not provide20% headroom over that conservative
measurement: `1.2 x 16.9447 = 20.3337GiB`. Slurm `sacct` reported batch
`MaxRSS=13574576K` (about12.95GiB), which is lower than the in-process peak;
retain both measurements and size future requests against the larger one.
The H200 device peak was43,799,579,648 bytes against a reported
112,582,508,544-byte device limit, so the GPU itself retained ample headroom.

## Independent Astra review

Astra's read-only verdict is **REJECT this rank-4 candidate** and **NO-GO**
for the unused validation probes, the conditional `F_p` calculation, and any
sampler launch. This is a substantive result, not an invocation failure. Astra
independently confirmed the failed discovery gate, absence of downstream
evidence, and that the runner stopped in the registered order. The review is
mid-course advice, not R2 exit approval.

Astra also identified that the unexecuted directional-precision gate could
never pass as written: for any positive estimate,64 Gaussian probes and four
Bonferroni-adjusted two-sided95% intervals imply relative half-width
`0.6309723942`, greater than0.50. This did not cause the observed discovery
failure and is not grounds to resume the rejected candidate. At least91 probes
would be needed for that width criterion alone, which likewise is not
authorization to add probes. The review further noted that the conservative
process peak leaves only18.0% headroom in a20GiB request, and that the runner
checks this after—not immediately after—the large host QR/basis allocations.

## Driver disposition and scope

The driver adopts the substantive rank-4 rejection and closes this candidate.
The `F_p` step is conditional on passing this geometry gate, so it is not
launched. The driver independently checked Astra's interval arithmetic with
the committed helper and confirmed the `n=64` width (`0.6309723942`), as well
as `0.5002689` at90 and `0.4966218` at91. A small pre-run width-feasibility
helper and regression test now prevent this class of impossible precision
gate from being specified without detection; the closed geometry runner also
fails this feasibility check before GPU setup. The incomplete interval design
and host-memory underestimate are correctable planning defects, not evidence
that another rank-4 run is warranted.

For a distinct future sampler candidate, Astra recommends fixed,
physically-indexed fundamental and `k^2=4–8` conditional moves while retaining
complete-field and nuisance transitions. The driver treats this as a design
lead only, not permission to reopen the closed million-dimensional low-block
ESS sweep or launch a sampler. Before any run, the proposal must preserve the
unchanged full count+FP+prior target, cover every coordinate with an invariant
fixed schedule, and have one capped comparison of rejection-inclusive
slow-mode movement per wall time/full-target evaluation. If the existing
costs do not support a credible bounded comparison, do not submit it.

**Q-GOAL:** this result supplies no field or posterior. The R2 endpoint
remains a calibrated CF4+galaxy-conditioned z=0 density/velocity posterior at
1.5 cMpc/h in the384 cMpc/h box with held-out prediction and uncertainty.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must
ultimately constrain those same roles on the same NEW evolved LG field at
`<=0.3 cMpc/h`, with native truth identities restricted to
calibration/evaluation. R2 is not complete.

**Q-LEAN:** one16-pullback candidate stopped at its predeclared stability
gate; no validation, `F_p`, sampler, held-out read or gravity run followed.
Do not repeat this candidate or turn the CI feasibility correction into a
probe-count ladder. The next design must test transport directly rather than
add another proposal-only gate. Preserve the observation-selection,
uncertainty and sampler-calibration blockers in the full R2 goal.
