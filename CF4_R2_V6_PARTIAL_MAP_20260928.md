# R2 v6 live count/singleton MAP — bounded implementation

Order: R1 physics/backend → **R2 actual surroundings (in progress)** → R3
same-state MW/M31/M33 → R4 accurate evolution → R5 phase-consistent zoom IC.
R2 is not complete. The approved full R2 target remains N256/384 = 1.5 cMpc/h;
this N128/384 = 3 cMpc/h calculation is an explicitly partial development fit.

## Corrections before fitting

Job407731 passed 17 wiring tests in9m28s, peak7.59GiB, but source inspection
found the new combined target omitted the count LOS-width argument (default0)
while the mark used100exp(.5t6), and widened the frozen5–180 survey cut to192.
The old finite-gradient test with empty observed counts did not catch either.
Both are corrected, with nonempty-count/reference and radial-edge fixtures.
The count and mark now share the width, luminosity function and radial limits.
GH15 uses a rematerialized scan, not a statically unrolled large gradient
graph. Existing GH15 evidence is fixed-state only; it is not an error bound
over optimized states or proof of continuous mark/count equality at edges.

Fitting accepts no heldout observations. One shared FP zero coordinate has
the existing .004-dex development prior; remaining source-fit covariance is
not thereby calibrated. The 985 grouped one-link rows stay excluded; only
429 catalogue-ungrouped one-point/one-FP-row links enter this partial model.
No TF or other CF4 source is silently claimed to be included.

## One bounded calculation, no sampling escalation

Use v6's47,121 training count points/37,951 keys and population-specific
buffered exposure exclusions. Reuse the PMWD128 forward/adjoint; begin at the
predeclared unranked seed2026092702, not an old fitted/heldout-selected state.
For EVERY evaluation, including optimizer line searches, rebuild a periodic
KD-tree of the current coherently shifted source positions and query the
8-sigma plus TSC radius with the CURRENT width. Fixed padded capacity8192
fails explicitly rather than discarding sources or clipping the nuisance.
This remains approximate Gaussian-tail truncation, not an exact likelihood.

The driver composes the observation VJP with the same PM field VJP. One
initial refreshed-support directional test precedes L-BFGS (128 iterations,
192 evaluations, application45min/Slurm55min). Nine white tracer nuisance
coordinates use a reversible factor100 numerical preconditioner; the shared
zero is separate. No prior is changed by that coordinate scaling. An
optimizer stopping message alone does not establish MAP stationarity.

Save initial/current z=0 density/mean velocity and their white IC/tracer
coordinates; save each accepted optimizer checkpoint, not every trial's
simulation. Planned storage below0.5GiB. Host estimate20GiB including previous
7.6GiB compilation peak, PM adjoint, observation derivatives and L-BFGS history;
request24GiB (~20% margin),2CPU,one typedH100 GPU. No RAMSES run/new snapshot
archive. Corrected focused tests run first; calculation may depend afterok.

## Judgement and following work within R2

Read the optimization trace, true final gradient and field response before
uncertainty or validation. Compare sparse/full factors and quadrature at
the resulting field if necessary, not another arbitrary prior-state sweep.
If support, gradient or cost fails, retain the exact failure and repair it
within scope; do not extend the old identity-mass HMC line. An unconverged
MAP iterate cannot supply a Laplace posterior. Only after a usable local
solution, quantify approximate uncertainty and association sensitivity, then
perform the previously reserved prospective heldout evaluation once. None
of this licenses N256 or full R2 closure without the outstanding science.

Association is assumed field-independent conditional on observed point and
redshift; distance-dependent association/selection, count bias/FoG and shared
survey covariance remain uncalibrated. These must be solved or bounded with
evidence before calibrated R2 delivery; a pleasant MAP image is insufficient.

Q-GOAL: obtain an actual training-conditioned same-state density/velocity
estimate, retaining the latent IC connection. Q-LEAN: reuse frozen data and
existing operators; one bounded fit, not a new simulator or test framework.
MW/M31 remain ambiguous and M33 unresolved in this coarse field. R3 must
identify components in that SAME NEW evolved state and connect their actual
observables, not import native truth IDs or declare them found here.

No emails. Any nonexpert Korean PDF must illustrate each reported test with
an example figure, clearly separating schematic examples from actual results.

## Execution record

Corrected typed-H100 regression407790 COMPLETED/exit0 in9m25s; all17 tests
passed, batch MaxRSS3,366,804KiB (3.21GiB). The nonempty-count/reference test
specifically detects the two repaired width/window errors. This is not a
field-fit or posterior result.

Dependent typed-H100407793 started normally on source645072c. Its55min Slurm
limit includes the first PM/observation adjoint compilation; application
checks a45min cap between evaluations. Read-only terminal summary407795 is
queued afterany:407793 (typedH100,1CPU,2GiB/5min; host estimate<=1.6GiB).
It creates initial/final density and mean-velocity comparison figures only
if a final state exists; no heldout observation is opened or scored. If the
fit fails, it instead records the preserved failure and checkpoint status.
Driver assessment is still required before any uncertainty/validation fit.

### First fit stopped; same-target bounded-step correction

407793 FAILED/exit1 after19m56s (peak batch8.29GiB, Python8.34GiB).
The first value/gradient took737.0s including compilation; warm evaluations
took34–38s. The refreshed-support directional check differs by4.32e-6
relative under its original whole-objective normalization. Six optimizer
steps were accepted before the next trial had a nonfinite value or derivative.
The old combined error omitted which component failed and did not save that
trial, so a zero-support cause is **not established**. Preserve that limitation.
Readout407795 completed4s and correctly produced a failure summary, no final
map. Accepted checkpoint is retained; no heldout data were scored.

The objective fell1,299,959→1,079,270 but the count log factor worsened
−250,478→−275,825 and FP factor−1.15→−233.62, while the prior improved.
This is neither convergence nor an improved observation fit. Do not report
the aggregate descent as scientific reconstruction success.

Same-target retry uses the last accepted coordinates with fresh L-BFGS history,
not a claimed exact continuation. Limit EACH proposed step (not the parameter
domain) to IC-white RMS .1, tracer-white change .1, and zero-white change .5;
backtrack for actual sufficient decrease. A genuine minus-infinite log target
rejects the trial with no floor; a finite value with a nonfinite derivative,
NaN target, or positive-infinite target stops and saves the exact failed
coordinates/current field and component finiteness for diagnosis. The existing
initial derivative check now subtracts the analytic Gaussian-prior tangent
from both sides, without extra PM evaluations. One tiny quadratic/support
test runs inside the same allocation to check the new step controller.

Keep the45min application/55min Slurm cap. A normal time-budget stop now saves
the last accepted current field, not just parameters; it is still explicitly
unconverged. Observed peak8.34GiB plus bounded optimizer/history/diagnostic
buffers gives estimated peak<=10GiB; request12GiB (~20% headroom),2CPU,typedH100.
JAX's ordinary compilation cache is enabled to avoid paying identical future
compilations repeatedly; no filesystem test or diagnostic is introduced.
New output root `r2_v6_partial_map_v2`; v1 results unchanged. R2 remains open.
