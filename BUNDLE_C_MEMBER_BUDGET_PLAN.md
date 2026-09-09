# C — M33 mass/velocity demands against the same local matter budget

2026-09-09: user approved this DESIGN bundle. Proposed next implementation
requires approval. One small physical diagnostic, not another proxy refit,
membership learner, particle simulation or posterior claim.

## Evidence and goal

337991's host discrimination is real at diagnostic level; its M33 increment
is inconsistent. Code inspection matters: `cf4_lg_composite_proxy.py` never
uses aperture mass in its score and has no field readout at the M33 location.
The result therefore does NOT establish that M33 mass/structure cannot inform F.
Do not require M33 alone to improve every pointwise score as a final science goal.

Propose a **necessary same-field moment-budget condition**. It asks whether
specified MW/M31/M33 local member masses and bulk velocities could fit inside
their spatial supports without borrowing the same matter twice. Rejecting an
impossible field would be useful; passing is NOT halo identification, a unique
decomposition, a normalized likelihood, or proof of dynamical compatibility.
No generic validation framework: this is the actual physics operator being tested.

## Identities, positions and mass semantics

The operator accepts F and three explicitly named hypothetical member records
(position, support radius, enclosed MEMBER mass, enclosed-member COM velocity).
It takes NO native membership mask/profile/ID. Positions are observation/state
arguments, not field-peak labels: M33 need not have a third density maximum.
In actual inference distances, MW reference position/common velocity, stellar
COM offsets and assignments must be uncertain and integrated; none is solved
by this conditional diagnostic. Keep MW/M31 proposal machinery reusable but
do not use truth to select a candidate in a new field.

This first test uses explicitly labelled noiseless NATIVE MOCK records; their
positions and BOX-frame velocities are supplied data, not recovered unknowns.
This is a stronger conditioning experiment than available real observations.
Same fixed target data/supports are evaluated on EACH alternative field.
Never recenter on that alternative field's true MW/M31/M33 catalogue.

For each role define Omega_g by native grid CELL CENTERS within R of its supplied
position. Primary R=dx=.1875 cMpc/h, descriptive R=2dx; freeze both, no tuning.
The mock m_g,P_g are sums of that role's cached SUBFIND-bound member moments
inside the SAME Omega_g. They are not whole bound masses, host M200c, isolated
M33 virial masses, or masses within a subcell spherical aperture. Native labels
are used only to construct these mock data, then excluded from the operator.
Report mass retained relative to whole native bound mass for each role/radius.
If zero member mass or incomplete patch support, label unavailable; no filling.

Do NOT activate the existing M33 3e11-Msun assumption as a measurement. The
primary source [Corbelli et al.2014](https://arxiv.org/abs/1409.2665) measures an
extended rotation curve and fits stellar/gas plus an NFW halo model to infer
a total halo mass. Such an inferred halo mass is not automatically the present
stripped bound or grid-aperture member mass. The future observational adapter
must forward-model the appropriate rotation/stellar/gas information and shared
distance/model uncertainty. That is deferred, not silently approximated here.

## Operator: seven unions, no overlap double counting, no solver

For each nonempty subset A of the three roles (seven subsets), form the UNION
U_A of their Omega_g and sum the TOTAL field moments ONCE per cell:
T_A=(M_A,P_A,Q_A), with Q the three raw diagonal second moments.
Subtract only the demanded bulk-motion lower envelope for members in A:

    rM = M_A - sum_g m_g
    rP = P_A - sum_g m_g*v_g
    rQ = Q_A - sum_g m_g*v_g**2.

Necessary conditions: rM>=0 and for EACH axis k the2x2 matrix
[[rM,rP_k],[rP_k,rQ_k]] is positive semidefinite. The rQ budget includes
unconstrained member INTERNAL dispersion plus remaining matter; it is NOT
claimed to be the raw second moment of a unique residual particle component.
When rM=0, rP=0 and positive rQ may be member internal dispersion; do not reject
that valid limiting case by misusing the existing zero-mass-reservoir check.
An empty-demand case still requires original field realizability.

Why necessary: real disjoint members have second moments at least m_g*v_g**2;
the rest of the union has a realizable mass/momentum/second-moment matrix.
Removing the bulk envelope leaves that matrix plus nonnegative internal Q.
The membership supports' union includes every demanded member mass once.
These separate union constraints need NOT imply a simultaneous spatial
allocation, binding or subcell force law; never present sufficiency as proven.

Report signed mass slack and minimum normalized eigenvalue, not fitted Gaussian
weights. Use Mref=max(M_A,sum demanded mass,1 Msun), Vref=300 km/s solely for
unit scaling, and fixed1e-10 relative numerical tolerance. Record raw budgets
too. This is roundoff tolerance, not an astrophysical discrepancy allowance.
Never multiply seven dependent tests as seven independent likelihood factors.
In a future complete model they could be a necessary support condition ONLY;
no observed posterior weighting until target/selection/uncertainty are defined.

## One bounded implementation proposed after approval

Reuse the existing THREE geometry-disjoint retained fields16,17,20. No new fit
or use of heldout results to tune a radius/threshold. Read the cached total
moment patches and sparse member moments from spatial_calibration_v1, not raw
particles. Existing total fine source can supply just the required cells.

For each radius, construct three mock datasets once, then evaluate the fixed
3x3 data/field table. Compare host-only subsets with all-three subsets; separately
report mass-only and mass+momentum/second-moment conditions. All nine cells
receive exactly the same analysis. A new exclusion after adding M33 is the
direct question; no mandatory positive result and no arbitrary score threshold.
It reflects mass/velocity demand at the specified location, not mass alone.

Original native fields should satisfy their own data by construction: these are
positive implementation controls, NOT independent evidence of predictive skill.
Alternative fields probe how informative this weak necessary condition is.
Report unavailable/support-clipped cases separately from incompatibility.
Do not call the three contexts a prior bank or weight them into a fake posterior.

One report/figure: budgets/retained fractions, two3x3 compatibility tables,
which subset/axis causes a violation, and whether M33 adds an exclusion beyond
hosts. Reuse moment routines; two focused tests (overlap-union accounting and
known valid/invalid moment examples including common-boost invariance).
No synthetic inversion, new thresholds, covariance estimation, neural model,
generic gate system, filesystem test or automatic follow-up variant.

Slurm2 CPUs,1200MiB (estimated1000+20%),10min cap, no GPU; allowed CPU requests
on a40/a100/h100/h200, exclude syn06. Small cached cell reads, output<5MiB.
No large3D copies are necessary. Clear implementation bugs may be corrected
within the same scope; physical non-discrimination is a result, not a retry.

## Exit and deferred work

If M33 excludes otherwise allowed contexts, preserve a usable necessary mass/
moment link and then design its observational mass/position/velocity uncertainty
adapter. If everything passes, record that this coarse budget is too weak on
these cases; do not automatically implement a large component network. Both
outcomes are diagnostic. This does not repair q_F, provide p(E|F,O), establish
subcell profiles or build an LG-conditioned field/IC. The missing field prior
and actual z=0 map remain central project work, not things to forget while
adding M33 tests. This is ONE budget test, not an indefinite side programme.

Fable5 questions: Q-GOAL (direct contribution toward CF4/LG z=0 -> IC -> zoom,
LG<=.3, environment1–2), Q-LEAN (is even this diagnostic worthwhile or another
detour?), MW/M31/M33 identity handling and no oracle NEW-field inputs, correctness
of the necessary moment inequalities and enclosed-mass semantics, feasibility,
essential/deferred scope. Give GO/CONDITIONAL GO/NO-GO before implementation.

## Fable5 audit and final driver disposition

`config/cf4_member_budget_fable5_plan_audit_v1.response.json` completed normally:
**CONDITIONAL GO for this one diagnostic**, not posterior/model certification.
Q-GOAL calls it a justified small connection, not a direct solution of the
missing q_F/q_S/actual z=0 inference. Q-LEAN accepts one bounded run with no
automatic variants. The four required in-scope conditions are adopted:

1. A single helper defines Omega_g from cell centers, with distance<=R; use it
   for mock construction AND operator evaluation and check identical cell keys
   in the existing focused tests. No half-cell/rounding convention switch.
2. Normalize each physical2x2 matrix A by D A D with
   D=diag(1/sqrt(Mref),1/(sqrt(Mref)*Vref)). This positive diagonal congruence
   preserves PSD inertia; do not normalize entries inconsistently.
3. Report NOT_EXCLUDED_BY_NECESSARY_TEST / EXCLUDED_BY_NECESSARY_TEST /
   UNAVAILABLE separately. The first is NOT proof of physical compatibility.
   Zero demanded member mass or clipped spatial support is UNAVAILABLE, never
   counted as evidence against a field. Do not turn these into pipeline gates.
4. All moments/COMs are BOX-frame peculiar velocities. For numerical stability
   apply the SAME supplied mock MW COM boost to the target demands and every
   alternative field, without changing positions. The existing boost-invariance
   test verifies necessary pass/fail, not equality of raw uncentered eigenvalues.

Important driver clarification of the audit's anticipated weak-power case:
report the number of available alternative contexts, how many survive host-only
conditions, and how many of THOSE are excluded by adding M33. If none survives
hosts, label **INCONCLUSIVE_NO_HOST_COMPATIBLE_ALTERNATIVES**, not "M33 has no
information" and not "the budget is too weak." If host-compatible alternatives
exist but M33 excludes none, only then report no incremental exclusion in this
limited experiment. No automatic new contexts to chase a positive result.

Two audit-wording corrections are recorded rather than silently inherited:
Corbelli2014's abstract gives an NFW-fitted total mass4.3e11 Msun; the disabled
3e11 assumption in our observation contract is from a different cited dynamical
study. Neither is adopted here as a measured enclosed member mass. Also, this
is not the project's first physical operator: old native-component transport
already conserved moments; the new distinction is a no-membership-input
necessary condition at specified M33/host locations.

Approved DESIGN is complete. No numerical job, source-particle request, new
inference code or refit has run in this turn. Next approval is for the exact
single2-CPU/1200MiB/10min implementation above; no further planning audit of
these accepted four recording requirements is needed.
