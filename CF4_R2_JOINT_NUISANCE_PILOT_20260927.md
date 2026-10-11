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

The first typed-H100 Slurm request was4CPU/16GiB host (measured 6.52GiB
fixed-state peak, estimated HMC peak<=13GiB including retained map moments),
2h cap. Its measured Slurm MaxRSS required a22GiB retry request.
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
allocation. Partial result path:
`/gpfs/kjhan/CF4/z0_density/r2_joint_nuisance_pilot_v1/result.json`.
Submission and tests are not a sampling or science pass.

First H100406431 FAILED after11m54s. Chain0 completed64 warmup/64 retained
with64 development maps, acceptance0.7883 and retained divergences0;
chain1 reached warmup28/64 then the generic trace checker rejected a
nonfinite record. The saved chain1 accepted states through step28 and its
gradient were finite; warmup had one recorded divergence and a rapidly
increasing step size. This does not yet prove whether the failing proposal
was an ordinary rejected divergent trajectory or an invalid accepted state.
Preserve the v1 result, scalar traces, chain0 map/state and Slurm logs.
Slurm sampled MaxRSS was~17.49GiB, above the original16GiB request, though
the process-reported peak was10.06GiB and Slurm reported exit1 rather than
OOM. The retry request is22GiB, exceeding sampled peak by>20%.

Scoped v2 recovery keeps identical observations, seeds, target and HMC
settings. It accepts a nonfinite **proposal energy only if that proposal is
flagged divergent/rejected**, counts those events, and still stops on a
nonfinite accepted position, log density, acceptance, step, or state gradient.
One exact synthetic regression runs before the survey arrays. This correction
does not clip likelihood support or silently repair a bad state. The new
result path is
`/gpfs/kjhan/CF4/z0_density/r2_joint_nuisance_pilot_v2/result.json`.
Typed-H100 retry **406462 COMPLETED/exit0 in14m27s**. Both fresh chains
finished64 warmup and64 retained transitions and saved64 present-field maps
each. Their retained acceptance means are0.8155/0.8588, retained divergences
are0/0, and observed occupied-cell unit intensities remain positive at the
terminal states (minimum0.000199/0.000204). The process peak is10.155GiB
under the22GiB request; the Slurm batch MaxRSS is approximately10.03GiB.
The retry counted **zero** rejected nonfinite proposal energies. It crossed
the earlier failure point but does not establish the cause of the v1 failure;
GPU numerical trajectory variation or an unrecorded v1 proposal/record
condition remains possible. Do not label the v2 checker change a demonstrated
physics or sampler fix.

The count log factor improves by about76,191/77,332 nats between the two
predetermined prior starts and their terminal states. This is expected
burn-in movement of an initially count-unfitted IC, not a Bayes factor,
calibrated bias estimate or evidence of equilibrium. The shared FoG white
means1.386/1.458 correspond to development widths~264/~277 km/s, and the
fourth tracer-bias white means-2.146/-1.805 correspond to multipliers
~0.342/~0.405 of the transferred exponent. These visibly differ across
short chains; source-calibrated FoG and bias remain absent. Neither the source
status string nor finite HMC traces certify an R2 posterior.

### Chain/map assessment and science decision

Typed-A100 Slurm **406467 COMPLETED/exit0 in8s**. Its read-only assessment
`/gpfs/kjhan/CF4/z0_density/r2_joint_nuisance_pilot_v2/assessment.json`
compares the exact two saved64-draw traces and chain-wise field moments;
no gravity or new likelihood was run. The implementation is
`scripts/cf4_r2_joint_nuisance_assess.py`.

The retained logtarget first-half to last-half means rise by4,829/4,528
nats, and all six bias-white coordinates drift in both chains. The FoG-white
coordinate moves downward in both. Rank-normalized split Rhat for logtarget
is1.818, for the IC mean square2.630, and for the seven tracer coordinates
1.755–2.437. Approximate split raw ESS is only2.18–3.49 over128 retained
draws. With two short, visibly drifting chains these ESS values are only
warning diagnostics, not precise effective posterior sample counts.

The two chain-wise development mean density fields have Pearson correlation
0.139 in the predeclared15–180 cMpc/h observer shell,0.487 inside15 cMpc/h,
and0.207 after24-cMpc/h box blocking. In the shell their mean-density RMS
difference is2.518 versus each field's spatial RMS~1.89/1.95; mean velocity
vector RMS difference is692 km/s. Even the coarse maps are not mutually
stable. Map differences alone need not mean an incorrect likelihood because
a broad posterior could contain disparate modes; together with clear trace
drift they rule out treating these short chain moments as an equilibrated
posterior mean or uncertainty. The retained within-chain sample SDs are
autocorrelated trajectory variation, not calibrated posterior SDs.

Decision: **NO-GO for R2 posterior/map promotion and NO-GO for a blind
extension or N256 escalation.** The numerical same-state pilot is useful as
an implementation/cost result only. The immediate bottlenecks are both the
nonstationary high-dimensional sampler and the uncalibrated field-dependent
galaxy response/selected-group law. The next substantive bundle must address
an observation-model calibration/heldout mock and sampler equilibration
strategy together, using an explicitly partial target until the group and
selection dependencies are resolved. Do not use these drifted nuisance means
as empirical priors or select a chain/map by how well it resembles LG.
MW/M31/M33 remain latent NEW-state roles, with unresolved M33 preserved;
their observables have not constrained the N128 maps and cannot be inferred
from native truth IDs or these unseparated coarse peaks.
