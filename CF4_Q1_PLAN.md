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
| 128^3 | 8 | 6 | 13.53 s | 365 MiB |
| 256^3 | 8 | 6 | 33.53 s | 2,573 MiB |

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

Post-audit Slurm rerun at commit `34a5303` passed the joint smoke, the Q1 JAX
fixture (5/5), and the state-cost benchmark (job 386734). These passes verify
the release checks and provenance only; they do not change the conditional
status or promote the response-basis bridge to a physical JAX operator.

### R2-geometry follow-up

At commit `1e77efd`, the JAX stochastic LOS was changed to recompute the
minimum-image direction from the coherent-RSD (shifted) position, matching the
NumPy oracle. The periodic wrap smoke passed on Slurm job 386881, and the joint
smoke passed on job 386883. A geometry-scaled NumPy benchmark at a 384 cMpc/h
box with 32 sources and six populations (job 386882) gave 3.06 s / 365 MiB at
128³ and 31.15 s / 2,573 MiB at 256³. This is a useful oracle reference, but
still not a JAX state-gradient or full-catalog measurement; Q1 therefore
remains conditional and production inference remains closed.

Fable's follow-up close audit is **CONDITIONAL PASS**. It confirms that the
shifted-position LOS fix is correct but that the central gate is still open:
the JAX path has no position/velocity state derivative and the R2 benchmark is
NumPy-only. The next bundle must implement the traceable cell-integrated LOS
response (including periodic wrap and FoG/redshift broadening), compare its
values and position/velocity JVP/VJP against the sealed oracle at x64, then
benchmark JAX forward/gradient memory and wall time at production source
counts. Q2 inference remains closed until those gates pass.

### State-dependent JAX candidate

The next implementation bundle added `src/cf4_q1_state_dependent_jax.py`, a
fixed Gauss--Legendre candidate that is differentiable in source positions,
velocities, masses, and LOS scatter. A Slurm x64 smoke (job 386922) passed
against the NumPy oracle at a small non-boundary fixture with relative L1
`6.82e-5`; its position derivative agreed with central finite difference.
This is a candidate gate, not a promotion: the interval-exact oracle remains
authoritative and boundary/wrap cases still require explicit value and JVP/VJP
coverage.

The first JAX cost run at R2 geometry (384 cMpc/h, 32 sources, six
populations, order-64 quadrature) measured 137.96 s and 3,178 MiB at 128³
(job 386923), and 311.54 s and 4,561 MiB at 256³ (job 386924). These are
forward-only CPU measurements; gradients and production catalog counts are
not yet included. The candidate is therefore computationally plausible for a
bounded pilot but not yet a production inference path. Q1 remains
**CONDITIONAL** until wrap/boundary oracle gates, JVP/VJP gates, and a
gradient-inclusive cost bound pass.

### Boundary/gradient gate result

Slurm job 386928 compared the candidate with the NumPy oracle at a cell
boundary, a periodic seam, and an interior point. Value relative-L1 errors
were `3.73e-5`, `7.09e-5`, and `6.82e-5`; the development value gate passes
at `5e-3`. The interior reverse gradients versus NumPy central differences did
not meet a strict release gate: position absolute error was `7.53e-4` and
velocity absolute error `4.45e-6`. They remain diagnostics, not promotion.

The gradient-inclusive R2 CPU run (job 386929; 384 cMpc/h, 32 sources,
128³) completed in 411.18 s using 14,512 MiB. Uniform exposure makes its
gradient norm a conservation sanity check rather than a science-shaped test.
The cost is already close to the 16 GiB allocation; 256³ gradient cost and a
non-uniform exposure run are deferred until the gradient mismatch is fixed.
Q1 is therefore **NO-GO for production**: reduce the oracle-gradient mismatch
or activate the radial-shell/Fourier fallback before opening any sampler.

### Fable-requested gradient sweep

The float64 step sweep (Slurm job 386931) confirms the position-gradient
discrepancy is not finite-difference truncation: the interior error plateaus at
`7.53e-4` from `h=1e-3` through `1e-6`. Boundary and seam errors are much
smaller, but the interior plateau fails the release gate. The JVP/VJP adjoint
dot-product check passes to `6.94e-17`, so the candidate is internally
differentiable; it is not differentiating the same operator as the NumPy
oracle. The next action is an operator-level gradient fix or a radial-shell /
Fourier redesign, not more cost scaling. Production inference remains closed.

### Quadrature-order resolution

The order gate (job 386935) shows the gradient mismatch is quadrature error:
orders 64, 128, 256, and 512 give position-gradient errors
`7.53e-4`, `1.13e-4`, `1.81e-5`, and `2.15e-6`, while value errors fall to
`1.98e-7` at order 512. The order-512 boundary/seam run (job 386942) also
passes: value errors are `1.25e-7` and `8.52e-8`, position-gradient error
`2.15e-6`, and velocity-gradient error `1.27e-8`. The adjoint check remains at
machine precision. This closes the development accuracy gate, but not the
production cost gate: the measured 14.5 GiB/411 s gradient run used order 64;
order 512 has not yet been costed at R2 geometry and is expected to be much
more expensive. Keep production inference closed until that cost is measured
or the operator is replaced by a lower-cost analytic/radial-shell design.

### Bounded order-512 production-cost gate

The required bounded Slurm measurement (job 386943) completed within its hard
limits (10 minutes, 16 GiB): R2 geometry, 384 cMpc/h box, 32 sources, 128³,
order-512 forward plus position/velocity reverse gradient took `520.58 s` and
used `12,975 MiB`. The run root contains the required owner marker. This
closes the single-case feasibility gate, but the uniform exposure makes the
gradient norm only a conservation sanity check. The measured order-64
256³/gradient scaling and order-512 128³ memory imply that a 256³ order-512
gradient will not fit a single 16 GiB card; tiled/checkpointed execution or a
lower-cost radial-shell/Fourier operator is required for production. Q1 is
therefore **CONDITIONAL**, not a production authorization.

The bounded 256³ order-512 gradient attempt (job 386960) hit the hard
10-minute walltime limit and was cancelled at 10:25, with peak recorded RSS
8,616 MiB and no completed result file. This rejects single-card 256³
order-512 gradient execution under the current implementation. Production
requires tiling/checkpointing or a lower-cost radial-shell/Fourier operator;
the Q1 production gate remains closed.

### Radial-shell/Fourier fallback

The failed tiled route is now superseded by a bounded radial-shell/Fourier
candidate (`src/cf4_q1_radial_shell_fourier_jax.py`). It groups sources into
fixed shells, deposits one shell field, and applies an anisotropic Gaussian
LOS transfer in Fourier space. Slurm smoke job 387184 passed mass/finite-value
and autodiff checks (`mass=6.72`, finite gradient). This is only a numerical
candidate: shell discretization and Fourier-vs-oracle value gates remain to be
measured before any Q1 promotion. The dense and tiled production gates stay
closed.

The first Fourier-vs-oracle check rejects the present transfer convention:
relative L1 is `0.459` at 16³, `0.653` at 32³, and `0.783` at 64³ (jobs
387190, 387192, 387193), despite mass conservation. Refining the grid does
not converge toward the cell-integrated oracle, so this radial-shell candidate
is **NO-GO** as written. Before another production-cost run, the Fourier
kernel must include a verified TSC window/alias treatment or the shell
approximation must be redesigned and pass the oracle value gate.

Per the user's instruction, Grok was consulted after this failure. Its
fact-bounded advice identifies three live causes despite the passing impulse
roll test: kernel origin/half-cell phase, FFT convolution normalization
(including the k=0 mode), and the mismatch between oracle TSC stencil and the
nearest-cell delta source representation. The priority check is a one-source
origin/phase plus k=0 normalization test before any shell grouping. Production
remains closed.

The first oracle-calibrated alias-aware kernel prototype (job 387198) also
failed its cell-centre check with relative L1 `1.20` (mass was conserved).
The likely defect is the discrete kernel origin/translation convention, not
mass normalization. This prototype is rejected and no production work is
opened. The next redesign must explicitly validate kernel origin, FFT shift,
and one-source translation against an impulse response before shell grouping.

Cell-centred source probes (jobs 387195/387196) still give relative L1
`0.552` and `0.677` at 16³/32³. The error is therefore not only sub-cell
phase aliasing; the simple continuous Gaussian Fourier transfer is incompatible
with the periodic cell-integrated TSC oracle. The Fourier candidate is fully
rejected. The replacement design is an alias-aware hybrid: precompute each
shell's exact periodic cell-integrated response kernel from the sealed oracle,
then use FFT only for translation/contraction. It must pass the same oracle
value gate before any cost benchmark.

### Tiled candidate result

The first spatially tiled sparse-TSC candidate matches the dense operator in a
small Slurm smoke (`2.1e-17` maximum absolute difference) and has finite
gradients. However, its 256³ order-256 reverse-mode benchmark (job 387032)
also hit the 10-minute limit and grew to `33,113 MiB` RSS. The unrolled JAX
tile graph is therefore worse than the dense path for production memory. This
tiling implementation is retained only as a correctness reference and is not
a production route. The next implementation must switch to a radial-shell /
Fourier operator with a bounded contraction graph; no further unrolled sparse
tile scaling is authorized.

The lower-order fallback was measured directly: a 256³ order-256 reverse-mode
run (job 386984) also hit the 10-minute hard walltime and was cancelled, with
peak RSS `14,413 MiB` and no result file. Lowering the quadrature order alone
therefore does not make the 256³ path production-feasible. The next design
must use spatial tiling/checkpointed VJP or replace the dense LOS response with
a radial-shell/Fourier operator; no further un-tiled order sweep is warranted.

### Grok replacement audits

At the user's request, Grok replaced both external roles for this bundle using
read-only, fact-bounded prompts. The Fable-role bundle audit returned
**CONDITIONAL PASS**: Q-GOAL is aligned, Q-LEAN is proportionate, and the
mandatory next action is tiling/checkpointing or a radial-shell/Fourier
alternative. The Opus-role technical audit returned **HOLD**: the numerical
and autodiff checks pass, but the 256³ reverse-mode run fails its wall-clock
budget. It recommends keeping the kernel research-only until a budgeted 256³
reverse-mode run fits via lower order, batched LOS, or checkpointed VJP. These
audits do not authorize production inference.

### Fable cross-design after Fourier failure

Fable agrees with Grok that origin/half-cell phase and TSC-versus-delta are
the live causes, but rejects FFT normalization as a primary cause because the
k=0 mass mode is conserved. It also found a harness defect: the alias test
passed pre-RSD positions to the candidate while the oracle used post-RSD
positions. The next diagnostics are therefore a zero-velocity single-source
cell-centre identity, centroid comparison, a consistent post-RSD two-source
test, and only then sub-cell phase/basis tests. Production and LG claims stay
closed.

The corrected alias-aware design now passes the first identity gates. Using the
actual source cell index for the FFT origin (and fixing shell-id wiring), the
single-source error is `3.01e-15` (job 387212) and the zero-velocity
two-source cell-centre comparison is `2.20e-15` with exact mass `6.72` (job
387216). The earlier `1.20` failure was a harness origin/indexing defect. The
next gate is consistent post-RSD positions, followed by sub-cell phase error;
no shell grouping or production cost claim is authorized yet.

The phase-indexed redesign was implemented. A 4³ phase basis reduced the
sub-cell error to about `0.045` but failed the value gate; an 8³ basis (job
387227) reduced the tested phase errors to `0.00248`, `0.00176`, `0.00176`,
and `0.00176`, below the development `5e-3` gate. This closes the small-grid
phase gate for the prototype, but the basis size, post-RSD multi-shell case,
and R2 memory/cost remain unverified; production is still closed.

The subsequent phase/post-RSD gate (job 387219) rejects the single
cell-centre kernel for real states: relative L1 is `1.37e-6` at phase 0 but
`0.107`, `0.275`, `0.567`, and `0.284` at phases 0.1, 0.25, 0.5, and 0.75;
the consistent post-RSD two-source case is `0.563`. Mass remains conserved.
The phase-indexed response basis or exact sub-cell TSC treatment is therefore
mandatory. Grok was queried for the next design after this failure, but the
CLI returned no usable answer in the bounded call; production remains closed.

The first shell-specific phase-basis post-RSD run exposed a second failure
mode. After fixing the shell-id argument order (commit `9754e63`), the p=4
run (job 387242) still gave relative L1 `1.254`; increasing to p=8 (job
387246) improved this only to `0.926`, with source-wise errors `0.646` and
`1.207`. A Slurm diagnostic (job 387255) isolated the cause: translating a
kernel generated at the exact post-RSD source position reproduces the sealed
oracle at `2.6e-15`, while phase-basis interpolation at that same source gives
`0.646` and `1.207`. Thus the FFT origin/translation and mass normalization
are correct; the multi-direction phase-basis interpolation is not a valid
production approximation for the anisotropic cell-integrated TSC response.
The shell phase-basis route is rejected for production and remains a
research diagnostic only. The next validation must use exact per-source
cell-integrated kernels (or a basis indexed jointly by sub-cell phase and
LOS/orientation, with a fresh value gate); no R2 cost or IC inference claim
is authorized from the failed basis.

A bounded exact-kernel scaling check (jobs 387257 and 387260) remains exact
for two sources: n=16/32/64/128/256 took 0.041/0.075/0.170/0.452/1.902 s,
with peak RSS 61/63/78/198/973 MiB and exact mass conservation. This is only
a two-source microbenchmark; it does not authorize large-source R2 inference.
It does show that exact per-source kernels are a viable correctness reference,
while the next production design must reduce source count or batch kernels
before attempting 128^3/256^3 with a realistic catalogue.

The exact-kernel batch gate (job 388101) passed at machine precision for all
catalogue sizes tested: relative L1 `4.3e-15`, `2.4e-15`, `9.0e-15`, and
`5.0e-15` for (64^3,16), (64^3,64), (128^3,16), and (128^3,64), with exact
mass totals. Wall time was 2.8/11.2/8.1/31.7 s respectively. The batched
exact route is therefore a valid correctness fallback; its production cost,
especially at 256^3 and full CF4 source counts, remains to be bounded before
any IC inference is launched.

The 256^3/16-source exact batch (job 388102) also passed: relative L1
`1.82e-14`, mass `16.0`, wall time `38.3 s`. Peak memory stayed within the
8-GiB Slurm request. This validates the exact fallback at the target grid for
a small catalogue; full-catalogue scaling still requires chunking and is not
yet a production IC run.

The bundled chunked exact operator was implemented and gated (job 388106).
For random six-population masses, chunk size 4 reproduced the sealed oracle
with relative L1 `0.0` at both 64^3 and 128^3; maximum per-population mass
errors were `3.6e-15` and `1.8e-15`. Wall times were 14.1 s and 30.8 s.
This validates chunking as an exact memory-bounded fallback, without claiming
that its unaccelerated Python loop is yet production-speed at full CF4 size.

The catalog scaling microbenchmark (job 388100) confirms linear source cost
for exact kernels: 64^3 with 16/64 sources took 1.34/5.19 s at 81 MiB RSS;
128^3 with 16/64 sources took 3.58/14.18 s at 205 MiB RSS, with exact mass
conservation in every case. This is feasible as a correctness/batching
reference, but a full CF4 catalogue still requires source batching and must
not materialize all per-source 256^3 kernels simultaneously.

The 256^3 chunked exact gate also passed for 4 and 8 sources (jobs 388116 and
388117): relative L1 was exactly `0.0`, mass errors were below `5e-16`, and
wall times were 23.8 s and 45.9 s. The measured scaling is linear in source
count, so a full catalogue must be processed in bounded chunks and likely
parallel batches; no unbounded all-source materialization is permitted.

The actual `data/cf4_galaxies.csv` contains 55,877 data rows. A Slurm cost
model (job 388127), using the measured 256^3 rate of 5.73 s/source and the
8-source bounded chunk (about 973 MiB peak), predicts roughly 89.0 h on one
core, 11.1 h on 8 ideal cores, or 2.78 h on 32 ideal cores. These are
optimistic linear estimates and exclude I/O/scheduling overhead. Therefore
the unaccelerated exact operator is a correctness reference, not yet a
reasonable full-catalog production route; a faster response representation or
aggressive source reduction is required before IC inference.

An exact shared-geometry optimization was added: each source's unit geometry
kernel is computed once and contracted with all six population masses. The
Slurm gate (job 388130) reproduced the previous chunked result exactly
(relative L1 `0.0`, mass errors below `9e-16`) and reduced wall time by 5.21x
at 128^3/16 sources and 4.24x at 256^3/8 sources. This lowers the full-catalog
estimate by roughly the same factor, but still requires bounded parallel
chunks; it is now the preferred exact fallback.
