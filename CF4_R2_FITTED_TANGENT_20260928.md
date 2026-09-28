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
