# C — unresolved LG members connected to the same total field

## Active disposition after Fable5 audit

The audit completed normally (105402ms, is_error=false), DESIGN CONDITIONAL GO.
Raw request/response: `config/cf4_unresolved_member_fable5_plan_audit_v1.*`.
Q-GOAL accepts the missing same-field operator as relevant; Q-LEAN recommends
a much smaller composite-aperture baseline BEFORE high-dimensional decomposition.
Driver adopts that ordering. **The cell-level q_theta below is a DEFERRED
alternative, not the immediate implementation plan.** Next proposed executable
scope is `BUNDLE_C_COMPOSITE_PROXY_PLAN.md`, awaiting implementation approval.

Accept the audit's five conditions with these explicit dispositions:

1. Anchor measure: next baseline uses a finite, explicitly normalized mixture
   of field-only MW/host pairs, not a cell-decomposition anchor measure. The
   latter is still unresolved and deferred. Deterministic reference encoding
   ALONE does not prove a generative density normalized: its support and reverse
   construction must agree. No claim that this harder issue has been solved.
2. Simpler baseline first; no neural training or binary-target pipeline until
   there is evidence it adds information beyond that baseline. This deliberately
   trims the audit's proposed pretraining binary-coordinate work: it is not
   needed to execute or evaluate the composite operator.
3. Existing targets are native SUBFIND-bound disjoint massive members at z=0,
   including the stripped bound M33 remnant, not pre-infall or isolated M200c.
   Future cell partitions inherit exactly that definition. Baseline mass
   likelihoods remain disabled; aperture mass is not a member-mass substitute.
4. Next baseline freezes14 statistical coefficients, one fit and one2-CPU job;
   neural parameters/optimizer updates0. No automatic alternatives on failure.
5. Fixed three-heldout-field cross-scoring and separate conditional-M33 scores
   measure field dependence; they are NOT posterior information gain or proof
   of a good cosmological field prior. Actual LG-on/off maps remain outstanding.

Additional driver correction not established by the audit: if E is a selected
three-member population and q_F is unconditioned, the full target below also
needs p(E|F,O), or instead an explicitly E-conditioned field prior. Writing
q_theta(S|F,O,E) alone does not supply that selection factor. The32 positive
fixtures cannot estimate it. The baseline is a selected-population diagnostic
ONLY; no observed posterior can be promoted while this issue is omitted.

The audit is not authorization for training or a new simulation. No fallback
auditor invoked, and no numerical job launched in this design bundle.

## Original proposal submitted to the auditor (retained with corrections above)

2026-09-09. User approved this DESIGN bundle after identification job337986.
Deliverable: a concrete statistical/implementation design with Fable5 Q-GOAL,
Q-LEAN and explicit MW/M31/M33 identification review, not another mark-only fit.
No numerical implementation, training or actual-data inference is launched by
this document. A subsequent bounded implementation needs bundle approval.

## Evidence and choice

On32 cached native .1875 fields, M31 and M33 share the nearest peak in every
case; distinct three-way matches0/32. The two M33 heldout associations take the
same peak away from M31. This is failure of this field-peak readout, not proof
that M33 is absent or that all reconstruction methods fail. These fixtures were
selected from the satellite branch;32/32 is NOT a cosmic satellite probability.

Choose an explicit latent MEMBER DECOMPOSITION of the supplied total field.
Do not demand a separate M33 density maximum. MW and M31 remain uncertain role
assignments as well, not truth-labelled peaks. Finer local particle/force scales
are ultimately needed for dynamics, but neither a new global fine grid nor a
fresh simulation is necessary to specify this missing observation operator.

The key distinction from old335878 transport is that new-field inference has
NO input native member mask, profile, subhalo ID or true position. Native
members can train/evaluate a conditional model, never seed a new realization.
Do not promote old kinematic transport or mass-only profile regression to this
model. Existing q_F flow morphology failures remain failures.

## Proposed target and exact accounting

Let F be the full present seven-moment field, C its1.5 restriction, O an explicit
observer-location condition and E the declared selected three-member LG model.
Let S contain role-labelled component moments, continuous centers and their
stellar/halo measurement discrepancies. The desired conditional factor is
q_theta(S | F,O,E), not three independent Gaussians around aperture values.

For each fine cell and extensive channel, require

    F = F_MW + F_M31 + F_M33 + F_remainder.

Every component has nonnegative mass and realizable diagonal second moments.
Components partition the same mass budget; their spatial supports may overlap
because different particles share a grid cell. The remainder includes other
halos and non-member matter. Bound-member mass is NOT inclusive host M200c;
never subtract host M200c and then subtract its M33 satellite again.

Proposed normalized coordinates: three sequential binary component splits
(MW versus rest, M31 versus rest, M33 versus remainder), using the existing
lossless mixed atom/continuous moment algebra in `cf4_split_moments.py`.
The ordering defines coordinates, not an assumption of independent roles.
Condition each split on F, prior splits, O and role; retain spatial coupling.
Reuse algebra/tests, NOT the failed spatial-flow checkpoint or an assertion
that a normalized field model already learned this different distribution.

Each present component needs positive total mass. A prospective sampler must
draw role anchors from normalized categorical distributions over positive-mass
cells, allowing co-located anchors, then use positive split support at those
anchors. This is a proposed way to enforce presence constructively, not a
post-sampling rejection with a silently discarded F-dependent normalizer.
Anchor probabilities and conditional split densities must be explicit; an
auxiliary anchor's marginalization must not count one S multiple times without
its specified joint measure. A deterministic reference-cell encoding of native
components or an explicit auxiliary-variable objective must be chosen before
implementation. No uncomputed feasibility/truncation constant is allowed in
an advertised posterior likelihood. F=0 cannot support three nonzero members.

Moment allocation alone does not determine a subcell galaxy center, boundness,
or tidal history. Give each role a continuous center offset relative to its
component mass centroid, calibrated jointly with profile extent/velocity from
native training objects. This is a probabilistic resolution/model discrepancy,
not an invented resolved M33 maximum. Keep M33 distinct from M31 even when
their cell supports overlap. Diagonal velocity variances are physical sigma_v;
uncertainty across S/F draws is a separate quantity. Do not rotate diagonal
tensors as if missing cross moments were measured zero.

## Identification, observational connection and information

Existing field-only MW/M31 candidates may initialize/propose anchors; they must
NOT restrict all prior support or select a best triplet. Preserve alternative
assignments and a route outside the heuristic candidate list. The observer
coarse cell is an external condition, not an exact MW position measurement.
M33 need not attach to a separate peak; satellite-conditional E is explicit.
The separate-primary alternative is deferred, not scientifically excluded.

From each sampled S use centers, component COM velocities and shared solar/
stellar offsets to predict distance moduli, heliocentric LOS and proper motions.
Use the existing observation convention/likelihood code as reference, but do
NOT set `resolved_halos=True` for latent decompositions or disable that guard.
A separately named latent-member adapter must state its weaker physical claim.
No additional derived-distance/velocity constraints or repeated LG likelihood.
Keep published-assumption M200c priors disabled until bound/host/LMC/stellar
mass definitions and covariance are connected. Existing partial covariance is
development-only, with explicit opt-in; zero unknown covariance is not evidence.

    L_LG(F) = integral L(D_LG | S, nuisance) q_theta(S | F,O,E)
                        p(nuisance) dS d(nuisance)

All role assignments/anchors must be integrated, not fixed by native truth or
optimized using the observation and then scored as an independent likelihood.
Exact sky conditioning needs its conditional measure/Jacobian; do not invent
an angular error to make proposals pass. Full-target schematic:

    p(C) q_F(F|C) q_theta(S|F,O,E) p(nuisance)
         L_CF4/counts(D_env|F,nuisance) L_LG(D_LG|S,nuisance).

This is a target specification, NOT an implemented/validated posterior. Survey
selection, shared calibration and overlap cannot be suppressed by writing a
product. The12-grid data posterior is not an independent prior for these same
data. q_F and the global1.5 environment law remain scientifically unready.

LG data can change F only through genuine dependence of q_theta on F: if it
merely changes unobserved member marks while the distribution of F is unchanged,
the desired field reconstruction has NOT advanced. Report field-level LG-on/off
information separately from sharper satellite labels in any subsequent trial.

## Minimal implementation sequence proposed for next approval

ONE bounded member-operator pilot, not a new fine-density generator:

1. Reuse cached total/member fields. Prepare the three binary component targets
   and measured center/COM offsets; retain full joint role information. Use the
   existing geometry-only nonoverlap split for field-conditioned evaluation;
   the old16/16 object labels alone are NOT leak-free voxel holdout. No raw pass.
2. Resolve the anchor/conditional-density measure above; use a small shared
   spatial conditional model and a jointly conditioned center-offset head.
   Freeze architecture, mixed-support objective and finite update/time budget
   before fitting. No independent cell draws that fragment a galaxy, no
   native-template copy and no train-until-it-looks-right loop.
3. In the same job test heldout component mass/COM/profile and ambiguity using
   field-only inputs, plus a fixed noiseless synthetic-observation connection
   control. Native labels are evaluation only. Output component responsibility
   maps and actual total-field dependence, not another total-field roundtrip.

Not yet authorized/executable: architecture/parameter budget, anchor treatment,
center-offset law and a meaningful fixed field-dependence experiment need to be
frozen in the implementation plan. The audit should identify whether these are
manageable essential choices or make this direction disproportionate/infeasible.
Do not claim that31 matched host profiles calibrate a high-dimensional q_theta.
The32 selected sources and one cosmology are a feasibility pilot only. A new
whole-field q_F fit, LG-on/off actual posterior, ICs and simulations are deferred.

Scientific success for that prospective pilot requires more than conservation:
distinguishable latent roles and calibrated-enough heldout joint observables,
without suppressing alternative assignments, plus evidence of field dependence.
Failure closes that model trial and identifies the missing ingredient; no
automatic variants or new audit ladders. A spatial posterior requires q_F to
become adequate separately; this decomposition cannot repair its morphology.

## Plan audit questions

Q-GOAL: does this close the missing same-field LG observation connection toward
CF4/counts -> z=0 posterior -> LCDM IC -> LG zoom (LG<=.3, environment1–2)?
Is it substantive progress beyond known-component transport/mark-only fits?
Q-LEAN: is a new conditional member model essential, or is a simpler physically
honest observable operator sufficient? Cut unnecessary machinery or prerequisites.
Explicitly audit all3 identities, unresolved M33, no oracle leakage, anchor
normalization, member/host mass accounting, conditional selection E and actual
field information. Give GO/CONDITIONAL GO/NO-GO for this design, essential versus
deferred scope, feasibility and the smallest next implementable deliverable.
