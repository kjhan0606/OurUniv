# Q1 LOS convolution and LG-observable entry plan — 2026-09-16

Opus5 plan advice selected this as the next substantive bundle. The current
Q0 GH/TSC point-deposit convolution fails its frozen convergence gate, so no
GH order or tolerance is promoted. Implement a NumPy cell-integrated (or
analytic radial-shell) line-of-sight convolution first, using node spacing
Δ/4 then Δ/8, ±5σ truncation, normalized weights and periodic mass
conservation. Keep GH3/7/9/15 only as diagnostic comparisons.

Then implement the matching JAX operator and compare intensity, Poisson,
shared-redshift and joint likelihood values plus finite-difference gradients.
The zero-exposure and observer-coincident numerical guards are now fixed in the
existing JAX kernel. A separate CPU/GPU cost measurement at R2-like N128/3
and N256/1.5 is required before any sampler or production inference.

The LG geometric operator is a later small step: it predicts the contract's
M31/M33 [distance modulus, heliocentric vlos, pmra*, pmdec] from explicitly
supplied host/observer states and solar nuisance terms. It must not assign HOP
groups or activate disabled mass priors; M33 boundness remains deferred.

Q-GOAL: this opens the actual CF4+galaxy z=0 posterior path, while explicitly
retaining MW/M31/M33 observable wiring and unresolved covariance. Q-LEAN: reuse
the frozen Q0 fixture and NumPy oracle; no new simulation, seed selection,
threshold tuning, or RAMSES work. R1 solver diagnostics remain conditional.

Exit: PASS only if Q1 converges at both spacings, conserves mass, agrees with
JAX and has finite gradients and a measured feasible cost. Otherwise NO-GO
and switch to Fourier/radial-shell design; do not relax the gate. No posterior,
LG identity, M33 boundness or ≤0.3 cMpc/h claim is authorized by Q1 alone.

Opus5 planning/code audit (`claude --model opus`, read-only) returned
**CONDITIONAL PASS** after Slurm373164's4/4 fixture tests. The NumPy oracle and
fixture contraction are sound, but the current JAX basis contraction is not a
state-dependent Q1 operator: its gradients are only derivatives with respect
to source masses, and its table lacks an explicit population dimension. The
GH point-deposit path remains unpromoted. Additional findings are that the
observer guard is not guaranteed gradient-safe, zero-intensity Poisson
handling needs support/gradient tests, the Q1 oracle and GH path use different
line-of-sight definitions, frozen sliver/tail tolerances are not enforced, and
no R2-scale cost is measured.

Driver disposition: accept these as real blockers, not audit failure. Keep the
fixture contraction development-only. The next implementation must define one
LOS convention and a population-indexed response contract, enforce Q1
mass/sliver/tail tolerances, add correct zero-support and observer-gradient
tests, then measure N128/3 and N256/1.5 cost. Only after those gates can a
state-dependent JAX operator or R2 entry be considered. The CF4 distance
factor, shared observer/solar nuisance frame, MW/M31/M33 identification and
bound M33 remain separate unresolved work; no HOP identity or posterior is
implied.

## Bundle update — 2026-09-19

The population-indexed fixture and the joint-likelihood smoke test both pass
on Slurm CPU (Q1 fixture: 5/5; joint smoke: including the zero-count,
zero-intensity gradient guard). A first benchmark of the *actual* NumPy
state-dependent cell-integrated oracle was also completed on Slurm job 386723:

| grid | sources | populations | wall time | peak host memory |
|---:|---:|---:|---:|---:|
| 128^3 | 8 | 6 | 13.85 s | 364 MiB |
| 256^3 | 8 | 6 | 34.36 s | 2,572 MiB |

This is a representative eight-source development measurement, not an R2
catalog estimate; it has no JAX state gradient and does not authorize sampler
or production use. The JAX implementation is still only a population-indexed
response-basis contraction bridge, not the physical state-dependent operator.
Therefore Q1 remains **CONDITIONAL / development-only**: the next gate is a
single LOS convention plus a differentiable state-dependent JAX operator,
validated against this NumPy oracle and its gradients. If that cannot meet a
measured cost bound, switch to a radial-shell/Fourier design rather than
relaxing the convergence or science gates. No posterior, MW/M31/M33 identity,
or 0.3 cMpc/h claim follows from this benchmark.

### Bundle-close audit disposition

Fable returned **CONDITIONAL PASS**. The bundle is scientifically aligned and
not over-instrumented, but it is not an R2 entry gate. Mandatory next work is:
(i) freeze one LOS convention (the shifted-position direction used by the
NumPy oracle), including observer-crossing/wrap tests; (ii) implement and
gradient-test a genuinely state-dependent JAX operator against the oracle;
(iii) repeat the benchmark at the real R2 box/spacing and catalog-sized source
counts, timing the JAX path; and (iv) enforce sliver and tail tolerances at
runtime (now fail-closed in the NumPy operator). If the R2 cost is infeasible,
activate the radial-shell/Fourier fallback. Sparse oracle scatter, naming
cleanup, GH diagnostics, and the M31/M33 geometric operator are deferred; no
LG or resolution claim is authorized.
