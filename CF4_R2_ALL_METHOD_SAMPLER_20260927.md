# R2/5 — bounded same-field count+FP+TF equilibration test

The all-method factor has a correct same-IC derivative, but the two old
count+FP chains (64 warmup/64 retained each) drift badly. The new TF-only
source groups are mostly real `Ngal=1` groups, and cross-method source
comparison does not independently calibrate their selection/velocity law.
This bundle tests **sampler equilibration only** on the explicitly partial
N128/384 target. Do not promote its parameter values as calibration.

Start both chains from the two archived count+FP terminal states, with no
selection by TF score or LG appearance. Include all57,238 eligible galaxy
counts, original conditional FP group marks and6,745 disjoint TF-only
training marks from the *same* PMWD-evolved current field. Preserve the old
seven white tracer-nuisance coordinates. Add exactly two provisional standard-
normal white nuisances: TF selected-density exponent `exp(0.5*w)` and
singleton-dominated redshift width `150*exp(0.7*w)` km/s. They expose model
sensitivity, not calibrated source knowledge. The shared TF modulus offset
remains the existing FP-anchor parameter, not a second independent zero.
TF-group redshifts enter only inside the conditional TF mark ratio.

Use 192 warmup and128 retained HMC transitions per chain, with the existing
4–8-step randomized trajectory and target acceptance0.8. Save scalar traces,
accepted terminal white states and one blocked density map per endpoint, not
128 full maps. Inspect between-chain and split-chain drift/ESS/Rhat before any
extension. Preserve finite accepted states and explicitly count rejected
nonfinite proposal energies. One H100/4CPU/30GiB/3h allocation with a bounded
application deadline; the previous HMC process peaked10.16GiB and the
all-method derivative control5.51GiB, so30GiB exceeds their conservative
sum by~90%. GPU-memory use is separately watched by Slurm exit/status.

Q-GOAL: directly tests whether the new all-sky CF4 distance information and
galaxy counts can be sampled as a coherent current-field target. A numerical
pass alone does not deliver R2; selected-group inclusion, source covariance,
tracer response and heldout prediction remain. Q-LEAN: one predeclared pair
of starts, one target, no MAP-selected seed, large map stream or parameter
search. If nonstationarity remains, diagnose mass/temperature geometry
instead of blindly extending the same HMC kernel.

MW/M31/M33 remain R3 latent identities from each NEW state. No native truth
labels or fixed known components seed/select candidates; role ambiguity and
unresolved M33 must survive, and their observables must constrain the same
evolved field. This N128/3-cMpc/h calculation cannot separate those members.

## Execution and result

H100 **406491 was cancelled by the driver after2m18s**, while the source
overlap audit found2,761 TF training groups securely linked to counted
2M++ points. This audit does not prove double counting; it exposes an
unverified count-conditioned group-cz/selection relation in the partial
target. Preserve the job's partial result and logs. No retained samples or
equilibration verdict were produced. Revisit this HMC design only after the
conditional observation relationship is specified or convincingly bounded.

Revised prospective v2 design: reuse this script only after the secure
singleton TF-point connection passes its same-IC numerical control. Sample
all57,238 counts, existing FP groups and **only the3,108 securely
point-matched singleton TF groups** (2,484 train/624 original heldout).
The unmatched/ambiguous5,394 TF groups are omitted from this development
target rather than assigned an invented count-point identity. The matched
TF radial response and redshift width share the six count-population
bias/FoG coordinates; the original seven white tracer nuisance dimensions
remain, and the two provisional TF-only dimensions from the cancelled v1
design are removed. The linked same-IC control406496 passed its vector
tests and derivative; typed-H100 **406501 submitted** with4CPU/30GiB/3h.
Output is isolated at
`/gpfs/kjhan/CF4/z0_density/r2_all_method_sampler_pilot_v2_matched/`.
Submission is not completion or a posterior pass. Source selection and
shared covariance remain unresolved even if HMC traces stabilize.
The matched TF factor also omits the count model's fixed35–50km/s redshift
measurement convolution and the true-distance dependence of count/TF
selection. These are observation-model limits, not HMC acceptance failures.

Read-only scalar/endpoint assessment job **406502** has an Slurm
`afterok:406501` dependency. It will not start on a failed sampler; it
checks exact192+128 transition ownership, split drift/Rhat/ESS for target,
IC variance, shared TF zero and seven count nuisances, plus the two terminal
24-cMpc/h blocked density states. Endpoint agreement is not a posterior
mean check. No automatic posterior promotion or N256 job is chained.

## Matched-subset result and decision

H100 **406501 COMPLETED/exit0 in22m44s**; both chains finished exactly192
warmup and128 retained transitions. Retained acceptance was0.8242/0.8212,
retained divergences0/0; rejected nonfinite proposal energies3/4 did not
produce invalid accepted states. Peak batch RSS was~9.88GiB under30GiB.
A100 **406502 COMPLETED/exit0 in7s**, reading the frozen traces/endpoints.

The chains are **not stationary**: retained log-target half means rise by
1779.84/1558.31 nats, rank-normalized split Rhat is2.064 for the log target,
2.031 for IC white mean-square and1.381–2.871 for the seven count-tracer
coordinates. Approximate split ESS for those scalars is only2.12–5.28 over
256 nominal retained draws. The two terminal24-cMpc/h density blocks correlate
at0.244, with RMS difference0.671 versus spatial RMS0.547/0.543. Endpoint
agreement alone would not prove posterior mixing, but here both scalar drift
and endpoint disagreement independently reject map promotion. The old
count+FP pilot's nonstationarity is **not** cured by matched TF marks or the
extra192 warmup steps. Do not pool these states, use their within-chain spread
as uncertainty, select a visually appealing LG field, blindly extend the
same kernel, or launch N256.

The HMC used an identity inverse mass and only4–8 leapfrog steps with final
step sizes0.00714/0.00791, i.e. trajectory lengths roughly0.029–0.063 in
white coordinates. This short path is a plausible mixing contributor, not a
unique causal proof: the tracer/selection model is also provisional and the
high-dimensional target may be strongly anisotropic. A bounded sampler-
geometry comparison with materially longer, cost-recorded trajectories is
the next computational test, not another arbitrary warmup extension.

The chain code/target is the **pre-redshift-convolution** matched TF version
recorded by its output hashes. A later correction to the TF conditional
kernel cannot retroactively change these traces. Selection, source covariance,
and count/TF redshift/position dependence still require source-backed
calibration before any production posterior claim. Decision:
**NO-GO_R2_POSTERIOR_AND_N256; CLOSE_THIS_MATCHED_HMC_RUN**. Q-GOAL: the
experiment directly tests the actual current-field sampler. Q-LEAN: it
rejected promotion with two bounded runs and no automatic production job.
MW/M31/M33 remain latent roles on a future NEW field, including unresolved
M33; this N128/3-cMpc/h output does not identify those components.

The after-run redshift convolution control406514 passed four focused tests
and changed the TF score by+1.85127nat at the predeclared initial IC, with
TF gradient discrepancy9.81e-8. It is a new partial target, not a repair
of the failed406501 chains or a source-selection calibration.

## One bounded HMC-path geometry comparison (v3)

To test the plausible short-trajectory bottleneck rather than merely add
warmup, v3 starts from the **two unranked406501 terminal white states** under
the newly redshift-error-convolved partial target. It keeps every datum and
prior definition except that scoped numerical correction, then uses96
warmup+64 retained transitions per chain with a state-independent random
32–48 leapfrog-step path and dual-averaged acceptance target0.8. Seeds are
fixed separately from v2. It reports actual integration work, acceptance,
drift, Rhat/ESS and endpoint blocks; it never treats the old target's states
as exact draws from the new one. From the previous22m44s for640 transitions
at4–8 steps, a roughly one-hour H100 runtime is plausible; the Slurm cap
remains3h/30GiB with a155-minute application deadline. This is one geometry
test, **not** a calibrated posterior, chain extension or N256 precursor.

Q-GOAL: isolate whether material path-length increase makes the current
CF4+galaxy field sampler more viable. Q-LEAN: one altered sampler setting,
reused same field/likelihood/operator and existing scalar assessment; no
new gravity data, tuning series or map stream. If drift persists, stop this
identity-mass HMC line and reassess geometry/model rather than extend again.
MW/M31/M33 still require latent assignment from each NEW field with an
unresolved-M33 branch and observations applied to that same field; none of
these coarse states resolves their identities or uses native truth to rank
starts.

Typed-H100 **406521 submitted** with4CPU/30GiB/3h for this one
`long_path_v3` calculation. Read-only A100 assessment **406522** has
`afterok:406521` and no automatic production successor. Submission is not
a mixing or source-calibration pass; result path is
`/gpfs/kjhan/CF4/z0_density/r2_all_method_sampler_pilot_v3_long_path/`.

Literature-scale context, not a transfer prescription: the
[Manticore-Local 2M++ field-level analysis](https://arxiv.org/html/2505.10682)
sampled initial fields with HMC but updated galaxy-bias nuisances in a
separate slice-sampling block. Its five chains used 4,750 burn-in and 2,400
post-burn-in steps each at a coarser 3.9-Mpc inference grid; low-k modes
could remain correlated for hundreds of steps. Their approximate likelihood,
selection, dynamics and computing resources differ from ours, so these counts
are neither a required schedule nor proof that our current target can be
calibrated by longer chains. They do establish that 64 retained transitions
alone cannot plausibly certify the 384-box field posterior. If v3 still
drifts, consider a *different* blocked/conditioned geometry and a calibrated
observation model before any long N128 or N256 production, not a repeated
identity-mass trajectory-length sweep.

## Selection-coordinate cross-check (public source, no email)

The current six-population count operator in
`src/cf4_r2_continuous_tracer.py` multiplies its voxel-integrated radial and
angular exposure **after** displacing model tracers into redshift space. In
contrast, [Nusser (2026), §IV.4.4–5](https://arxiv.org/html/2606.08593v2)
weights each tracer by the survey radial selection at its **model real-space**
distance before redshift-space deposition, to avoid a Kaiser-rocket term.
This is a consequential alternative observation model, not evidence by itself
that our current operator has a software bug: our six absolute-K population
labels and catalogue cuts use distances computed from *observed redshifts*,
and the archived exposure also integrates angular completeness. A direct
replacement of its exposure by a real-space weight would change the datum
definition and potentially mishandle the angular mask.

Before any R2 posterior/N256 promotion, specify the joint generative order of
true distance, apparent magnitude, observed redshift, absolute-K bin and
angular completeness; compare the current post-RSD exposure approximation
with a source-consistent pre-RSD selection construction on matched mock/heldout
predictions. The current HMC v3 remains a sampler-geometry test of its frozen
**partial** target regardless of the outcome. The cited paper's five N128
chains with 5,000 warmup transitions and NUTS show that our 64-retained-step
test cannot certify stationarity, but their survey, dynamics and sampler do not
set a transferable run length or calibration for our CF4+2M++ target.

## Long-path v3 result and disposition

H100 **406521 COMPLETED/exit0 in58m30s** (batch peak RSS
10,406,448 KiB under30 GiB); dependent A100 **406522 COMPLETED/exit0 in3s**.
Both chains finished the exact96 warmup+64 retained transitions under the
redshift-error-convolved *partial* target. The realized sampling steps were
0.005006/0.004491 with randomized32–48 leapfrog steps, retained acceptance
0.869/0.881, retained divergences0/0, and rejected nonfinite proposal
energies3/2. Those are execution checks, not stationarity.

The retained log-target half means rise by **2,963.53/2,183.72 nats**.
Rank-normalized split R-hat is **2.329** for log target, **1.749** for IC
white mean-square, and **1.550–2.346** for the seven count-tracer nuisance
coordinates. The scalar split-ESS approximation is only **2.25–3.99** for
these monitored quantities over128 nominal retained draws. The two terminal
24-cMpc/h density blocks have Pearson correlation **0.1773** and RMS
difference **0.6557**, larger than their individual spatial RMS
0.5322/0.4888. The latter is endpoint disagreement, not a calibrated
posterior-variance statistic. Frozen source artifacts:
`/gpfs/kjhan/CF4/z0_density/r2_all_method_sampler_pilot_v3_long_path/{result,assessment}.json`.

**NO-GO_R2_POSTERIOR_AND_N256; CLOSE_IDENTITY_MASS_HMC_PATH_LENGTH_LINE.**
The longer trajectory did not equilibrate these starts at this budget, even
with high acceptance and no retained divergences. Because v3 also changes the
TF redshift-error convolution from v2, the v2/v3 diagnostic differences are
not a controlled same-target performance ratio, nor proof that long paths
are intrinsically worse. Do not extend these chains again, pool their states,
use their spread as field uncertainty, infer source-calibration parameters
from the drifting trace, or select the better-looking terminal LG. The
count/TF and FP selected-source covariance/inclusion, galaxy bias/FoG and
selection-coordinate law remain uncalibrated independently of sampling.

Next substantial work is a *joint* observation/sampler redesign: specify
the source-consistent generative order for the six galaxy populations and
selected CF4 marks, then test a meaningfully different conditioned/blocked
sampler on a fixed target with independent starts and heldout predictions.
This is not another identity-mass trajectory-length/warmup sweep or a move
to N256 before N128 calibration. Q-GOAL: remove two demonstrated barriers to
an actual CF4+galaxy z=0 posterior. Q-LEAN: reuse the frozen input/source
products and existing partial-target code, with one redesigned algorithm and
one source-model comparison, rather than new gravity or repeated diagnostic
gates. MW/M31/M33 remain latent roles on each NEW state in R3; these coarse
chains do not identify them, and unresolved M33 and same-state LG observables
remain explicit future requirements.
