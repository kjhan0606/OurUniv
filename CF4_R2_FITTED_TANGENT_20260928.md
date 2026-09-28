# Fitted-state descent: evidence and driver decision

R1 → **R2 in progress** → R3 same-state LG → R4 → R5 zoom IC.
R2 is not complete. No heldout observation was scored, no posterior promoted.

## Measurements

407840 (5m11s) tested the v2 accepted state, not an unconditional fixture.
Normalized descent reverse derivative −403325.84 versus objective finite
differences −32.82, −4173.25, +105005.17 at eps .01, 1e-4, 1e-6. Both sides
increased the objective at these scales. This initially leaves derivative
error, very strong curvature, and observation discontinuities unresolved.

407881 (3m59s;7.60GiB Python peak) extends the SAME direction/state:

| eps | objective FD | observation VJP with field FD | relative reverse/FD difference |
| --- | ---: | ---: | ---: |
| 1e-7 | −371999.54 | −375423.16 | .07767 |
| 1e-8 | −403443.35 | −403442.77 | .0002913 |
| 1e-9 | −403378.57 | −403378.26 | .0001307 |
| 1e-10 | −403318.55 | −403316.67 | .00001807 |

At1e-7 and below, the plus direction actually decreases the objective.
Occupancy of native cells does not change between the two perturbed fields.
Nearly all the signed velocity-tangent contribution comes from native
rho>=1e-3; bins0<rho<1e-6 contribute~1.8e-7 and1e-6<=rho<1e-3 about−.0012,
versus total~−400000. This signed attribution does not exclude cancellation,
conditioning in count-cell moments, or other directions. It does not support
blaming nearly-empty native nodes as the demonstrated primary cause.

407880 was cancelled immediately to correct Slurm comma-separated export
parsing, replaced by407881 with the epsilon list inherited through ALL.
No output was deleted or treated as a scientific failure.

## External advice and independent driver assessment

Read-only `claude --model claude-fable-5 --permission-mode plan` returned
CONDITIONAL PASS; Q-GOAL and Q-LEAN support same-state diagnosis, not another
blind long fit. No edits, jobs or simulations were delegated. Advice to
separate field and observation tangents and avoid conservation-only claims
is adopted. Proposed direct mass/momentum wiring remains a conditional
alternative, not an evidence-backed fix adopted here.

Specific unsupported inferences are NOT adopted:

- The score reproduction discrepancy6e-10 was **absolute**, not relative;
  the claimed6e-4 objective noise floor and dismissal of eps<=1e-9 do not
  follow. The measured small-epsilon results are informative.
- Equal maximum neighbor counts do not establish identical candidate sets.
- Monotone approach over three eps is suggestive, not proof of a correct
  derivative. The extra measured agreement is stronger local evidence.
- An IC coordinate cannot be identified with the same Eulerian grid cell
  after nonlinear evolution; the proposed location interpretation is invalid.
- A transpose test using mutually derived AD rules cannot by itself prove
  forward correctness. PMWD nbody uses custom VJP, with no direct JVP path.
- Conservation alone does not exclude Jacobian conditioning, but neither
  does a rho*(momentum/rho) intermediate prove the observed discrepancy;
  its algebraic cancellation and fitted-state evidence must be checked.

## Narrow continuation (unchanged physical target)

The existing12-halving search cannot reach the locally valid scale from its
IC RMS .1 trial. There is no evidence here requiring a PM-adjoint patch or a
likelihood floor. Reuse the accepted state with initial coordinate-norm cap
1e-7 (measured descending), grow after successful steps, use safeguarded
quadratic backtracking up to40 trials, and evaluate gradients only at accepted
score-only candidates. Verify score-only and value/gradient agree before
acceptance. This changes optimizer steps, NOT prior support, survey cuts,
likelihood, or fields. Three small optimizer regressions precede the retry.

One20-iteration/25min application/30min Slurm retry, one typedH100,2CPU12GiB
(measured9.52GiB peak plus~20% margin). Initial random-direction derivative
check uses eps1e-8; larger eps at this conditioned state were not reliable.
Report progress and cost; do not keep spending allocations on negligible
descent. No Laplace uncertainty without a usable stationary solution.

Q-GOAL: recover an actual data-conditioned density/velocity state with the
latent-IC connection. Q-LEAN: same target/operator/cache, no new simulator,
new observational data, or validation framework. MW/M31 remain ambiguous,
M33 unresolved on this NEW state; their observables must constrain it in R3,
never native-truth-selected candidates. Surroundings1–2cMpc/h and LG<=.3
remain the final resolutions; N128/384 is only3cMpc/h development. No email.

## Retry disposition and next cause test

407886 passed all3 optimizer tests and the actual-state random-direction
check (prior-subtracted relative difference .0001309). Score-only and full
objectives agree. Accepted decreases then became negligible: total decrease
about.122 from1,039,923.9003, recent decreases~1e-6, maximum gradient still
~11,770. Other tiny candidate steps repeatedly jump upward by~.525. The
driver cancelled this allocation early for scientific stagnation, NOT OOM,
test failure or convergence. Preserve its accepted checkpoints and trace;
no final-state snapshot was promised after forced cancellation. Readout407887
is afterany, so missing final state must not be interpreted as a map success.

407901 is the queued15min/H1002CPU12GiB cause test on the SAME v2 accepted
state used for the derivative comparison. It counts, separately for every
GH15 node, source radial-cut crossings under eps1e-6 and1e-8, and reports
which have nonzero angular exposure. No fitting, changed mask, smoothing,
heldout score or new simulation archive. Zero crossings would exclude this
particular discrete-cut explanation at those trials; nonzero counts alone
would not quantify their contribution to the objective jump. Any subsequent
integral repair must preserve the actual observational cut and be supported
by a count-factor comparison, not an arbitrary likelihood floor.

407901 completed3m29s: at eps1e-6 one sky-active source-node crosses the
radial cut on the plus side and two on the minus side; at1e-8 there are zero
crossings. This identifies actual finite-GH selection switches, but does not
by itself attribute the entire count jump or all field stiffness to them.
407886 was CANCELLED by the driver at14m19s;10 accepted steps, total objective
decrease.121594473, final accepted max gradient11770.6683. Its readout407887
completed4s, explicitly no final map. Its historical status STARTED is a stale
application marker, not a running job; future readouts now label that case
INCOMPLETE_RUN_NO_FINAL_STATE without inventing the scheduler exit cause.

## Boundary-fitted integral repair candidate

`src/cf4_r2_shell_cdf_count.py` integrates the same source-selected LF/K/TSC
integrand over ray intersections with periodic observer shells5–180. Gaussian
probability-coordinate quadrature removes the hard GH radial-selection atoms.
27 disjoint observer images retain wrapped aliases; both signed LOS branches
and the inner5 cut are included. The existing8*sigma_radius<box/2 live-support
bound suffices for those images. Tails beyond8sigma are explicitly omitted
(unconditional Gaussian mass<1.3e-15); probabilities are NOT renormalized to
the selected shell. Positive tails use survival probabilities. Inactive
square roots/quantiles are guarded before nonlinear operations, not hidden
with a likelihood floor. Only identically zero interval weights are skipped.
Integrated Schechter interval fractions are continuous at moving K-bin
intersections; no apparent-K point indicator is newly introduced.

Fable conceptual follow-up returned CONDITIONAL PASS, Q-GOAL/Q-LEAN aligned.
Adopt normalization, tail, signed-image, finite-gradient and numerical-error
controls. Reject its assumed0.3125cMpc/h TSC scale: the current count grid is
3cMpc/h. Also reject globally parameter-independent geometry/smoothness:
post-wrap rhat can depend on velocity; tangencies and other field nonlinearities
remain. Do not claim this fixes every fitted-state pathology. The driver uses
the explicit probability-transform derivation; unverified literature details
suggested by the adviser are not recorded as established citations.

Four focused controls precede any full-state use: analytic cut probability
and derivative, signed inner-exclusion mass, periodic-image tail mass, and
the full LF/K/TSC source integral at16/32 nodes with velocity finite difference.
One H1002CPU6GiB/15min allocation (estimated host<=5GiB with20% margin).
This candidate is not yet wired into a fit or declared production-ready.
