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
