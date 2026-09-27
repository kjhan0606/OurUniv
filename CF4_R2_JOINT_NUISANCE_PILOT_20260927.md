# R2/5 — joint IC, 2M++ response and CF4 mark development pilot

The actual z=0 density/velocity posterior is still the first science
delivery. This bundle is one bounded **partial-target feasibility run**, not
that delivery, N256 production, or LG role inference.

## Decision and target

The fixed-state 406426 study found large count-score response to fixed bias
and FoG choices. Fable5 correctly advised that these are NOT posterior field
sensitivities; a count-unfitted field can favour a flatter tracer response.
The driver adopts one joint nuisance pilot rather than treating external
mock acquisition as a prerequisite to all development sampling. Production
still requires defensible selection/association/source covariance and mock or
independent calibration. Fable's arbitrary short-chain SD/cosine thresholds
and automatic extension are rejected in
`CF4_R2_LIVE_FACTOR_TENSION_20260927.md`.

Use the existing N128/384 coarsened 57,238-galaxy count plus conditional CF4
group-mark partial target from one evolved PMWD IC. Add seven standard-normal
white variables: six log nonlinear count exponents with centre at the
published ARES-derived values and width0.5, and one shared log FoG-width
scale with centre at1 and width0.7. These are **development regularization**,
not calibrated PMWD/RSD uncertainty. The already integrated six Gamma count
rates and existing CF4 hyper/group offsets stay in the SAME target; no old
survival thinning or independent BGc term. Do not seed from the mark-only
endpoints or select based on count score. In empty density cells evaluate
the power-law response with a masked positive log and exact zero response,
with no intensity floor. At zero density for exponent<1 this uses a zero
subgradient; the mathematical non-smoothness is a model limitation.

Run two fresh prior IC/group starts, each64 warmup+64 retained HMC
transitions, random4–8 leapfrog steps, unchanged target acceptance0.8 and
bounded adaptation. Monitor numerical support, finite gradients, acceptance
and divergences. Do not declare convergence from128 transitions. Record
count/mark/white-prior factors at predetermined initial and terminal states;
save scalar nuisance/target traces. Stream z=0 density and mean-velocity
running moments over all retained states for a **development map preview**;
save each chain's mean and SD separately, not a pooled certified posterior.
Keep physical velocity dispersion distinct from uncertainty in mean velocity.
If a chain fails numerically, preserve its partial output and do not extend
or mask the problem.

One typed-H100 Slurm job,4CPU,16GiB host (measured 6.52GiB fixed-state
peak, estimated HMC peak<=13GiB including retained map moments),2h cap.
No new simulation, raw snapshot, external survey download or N256 job.
Run the focused variable-bias empty-cell derivative test in the same job.

Q-GOAL: this tests whether the live actual-data model can update the present
field while carrying tracer-model uncertainty instead of pretending
transferred fixed parameters are known. A successful short calculation is
only a prototype; source-matched calibration, proper selection/inclusion,
heldout prediction and convergence remain before R2 delivery.

Q-LEAN: reuse existing PMWD count/mark functions and HMC chunks, one job and
small scalar/streamed map outputs. No new diagnostic ladder or long-chain
commitment follows automatically.

MW/M31/M33: N128/3 cMpc/h cannot identify separate bound components.
Their future candidates must come from each NEW evolved state without native
truth IDs; retain MW/M31 role ambiguity and resolved/shared/unresolved M33,
and apply their observables to that SAME state through a normalized law.
This bundle reports the LG region only as coarse environment, not as member
identification or <=0.3-cMpc/h information.

## Execution and result

Implementation is in `scripts/cf4_r2_joint_nuisance_pilot.py` with the
variable-exponent empty-cell response in
`src/cf4_r2_continuous_tracer.py`. Typed-H100 job **406431** was submitted
with4CPU/16GiB/2h. The four focused/reused tracer tests passed in its
allocation; the chains are still running. Partial/final result path:
`/gpfs/kjhan/CF4/z0_density/r2_joint_nuisance_pilot_v1/result.json`.
Submission and tests are not a sampling or science pass.
