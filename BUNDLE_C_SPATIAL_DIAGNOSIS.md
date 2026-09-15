# C-spatial: one cause-separation calculation and replacement design

User approved2026-09-08: one diagnosis using existing cases, then evidence-led
implementation design. This does NOT approve replacement training, another
copula variant, actual-data inference or IC work. Astra driver owns evaluation.

Use original retained patches16/17/20 and frozen335968 model/draws. Compare:
native remainder; native five-channel representation decoded by the existing
conservative operator; native channels through the frozen empirical CDF and
inverse before/after conservation; all twelve existing stochastic outcomes.
Do not fit quantiles, adjust thresholds, choose draws or add seeds. These
cases now inform design and must not become fresh final validation again.

Reuse mass_statistics and moment checks. Direct native density roundtrip must
have L1 error/native mass<=1e-10. Measure eight power bands, mass L1, per-parent
conservation, outside-training-quantile cell/mass fractions, mass-weighted
mean-velocity and directional physical-sigma discrepancies. The scalar fine
sigma channel is lossy for directional variances; density and velocity exactness
are not interchangeable. Original failed morphology gates remain unchanged.

Quantile reconstruction preserves spatial ordering except clipping/atoms, not
exact Fourier phases. A remaining generated-field gap does not uniquely isolate
random phases: conditional/environment dependence, sampling, spectral model,
and nonlinear conservation interact. Do not claim a single cause without evidence.

One Slurm CPU job2 cores, estimated4000 MiB+20%=4800 MiB,10min cap, JSON and
one comparison PNG (<5 MiB) in `bundle_c_v1/spatial_diagnosis_v1`. No raw source
pass, new full-field datasets, new generic tests, filesystem checks or monitoring
loops. Report results then design the next implementation; do not launch it.

Submitted336263, sourceebbe526 committed/pushed. Logs:
`/gpfs/kjhan/CF4/logs/cf4_C_spatial_diag_336263.{out,err}`.

## Outcome

336263 COMPLETED via Slurm on syn07, exit0,42s, sampled MaxRSS442120K.
Products `bundle_c_v1/spatial_diagnosis_v1/result.json` and `comparison.png`.
No new fitted model or stochastic realization was made. Status
SAVED_CASE_CAUSE_SEPARATION_COMPLETE_NOT_NEW_MODEL.

Native density representation roundtrip mass L1/native mass is2.26–2.58e-16.
Across all three cases/eight bands, native quantile roundtrip changes power
by at most0.1324% BEFORE parent conservation and0.09432% AFTER it. After-map
mass L1/native mass is0.01473–0.05756%. The outside-training-range mass-ratio
cells contain0.1111–0.5538% of native mass; that is mass residing in those cells,
NOT the mass removed by clipping. These few empirical-tail cells matter for
tail modelling but do not cause the previously observed large power failure
in the native-layout roundtrip.

In contrast, the existing generated remainders have fine-band power ratios
0.1350–2.3345. The failure arises in the GENERATIVE PATH and its interaction
with conservation, not a large deterministic density encode/decode loss.
The native-layout control does not rule out a bad marginal sampling law or
normalization when used on different random spatial arrangements; it also
does not separately identify missing higher-order phases versus missing
environment dependence. No speculative single-cause claim is justified.

The fixed-draw0 central-slab images show filamentary native layouts maintained
by both roundtrips and block-like structure in the generated remainder. This
is a visual diagnosis on three development cases, not a new calibrated topology
test or evidence that stochastic draws should match native phases exactly.

Separate confirmed representation defect: native five-channel encode/decode
changes mean velocity by3.354–16.832 km/s per-axis native-mass-weighted RMS,
and directional physical sigma by15.148–38.190 km/s. Coarse moments remain
conserved. Collapsing three fine variances to scalar trace and subsequently
reallocating per-axis energy is lossy even with native input. Future internal
representation must retain all three diagonal fine variances. Report scalar
sigma as a derived observable, not the only stored variance coordinate.

Decision: no copula repair, tail-knot expansion, seed search or amplitude
correction. Design next: BUNDLE_C_CONDITIONAL_FLOW_DESIGN.md. This diagnosis
does not supply a 1.5 environment posterior or an actual .1875 LG field.
