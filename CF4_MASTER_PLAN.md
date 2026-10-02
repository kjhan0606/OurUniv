# CF4 / OurUniv — active master plan

Effective 2026-09-07. The user approved registering the goal-alignment review
as the highest-level project plan and proceeding with short, substantive
bundles. This file supersedes CODEX_PLAN.md, PLAN.md, the Hong restart plan,
and cf4_science_route_v3.json as execution/priority authority. Preserve those
files and previous results as history; do not silently revive their routes.
Direct subsequent user instructions take precedence over this file.

## Current R2 continuation — 2026-10-02

Latest bundled continuation (user requests grouping stages): shared-moment
actual-state target implemented in `cf4_r2_moment_target.py`, active8tracer
+15population+3physical closure coordinates, no obsolete sampled global sigma
and no hidden priors.410643 COMPLETED3m41s,10tests passed, including density/
mean/variance/active-nuisance joint finite differences with refreshed support.
Actual saved N128 moment-state/all1414 training rawFP+47121counts attempt
410644 failed at GL1 raw support row2 before likelihood; GL2 retry410645
failed at row76, also24s and before likelihood. No PM evolution, fit or
heldout score. Close blind source-rule escalation; do not floor or inflate
width to obtain a finite target.
All-source localization410646 COMPLETED23s: row2's GL1 untruncated broad
radial weight3.59e-100 becomes.00337527 underGL2. Row76 remains0 at8sigma
and only2.28e-47 without that cutoff underGL2, versus legacyTSC.00176014.
Thus a local support-tree omission is excluded for those source rules, and
removing the Gaussian cutoff alone does not supply useful row76 support.
Finite source-ray/voxel/sphere geometry is implicated but coordinate/measure
semantics and a concrete remedy require verification. Fable5 focused
important-finding advice requested, Q-GOAL/Q-LEAN explicitly; prompt:
`config/cf4_r2_actual_voxel_support_fable_20261002.txt`. Native/PM calibration
uncertainty, missing CF4 methods, actual IC adjoint and stationary posterior
remain unresolved. R2 incomplete; no production target/field promotion.
MW/M31 ambiguous,M33 unresolved; later constraints act on the SAME NEW
field, not native identities. Numerical allocations from this bundle are
terminal; no new gravity simulation or chain is running.
Fable advice is now complete: CONDITIONAL PASS for quadrature relocation.
Driver adopts observed-direction integration but rejects its raw-redshift
ratio as a substitute for actual opticalFP/K/selection normalization,
guaranteed-support claims, ignored coherent periodic angle changes and an
unmeasured full-count runtime guarantee. Within the same bundle, implement a
signed source-cell ray/CDF prototype retaining the FP conditional and explicit
no-wrap sufficiency bound. Registration checks all47121 training points and
1414 links; two actual failed rows get radial4/8 and scalar AD/FD comparisons.
No extra redshift factor, count-target replacement, prior injection or
posterior claim. Sourcef813819 Slurm410664 typedH1002CPU8GiB20min submitted;
earlier pending410662 cancelled before calculation to repair a queued-runner
HEAD-vs-documentation check. Quota QOSMaxGRESPerUser is not bypassed; other
projects are untouched. Await this bounded prototype rather than a simulation.

Next grouped R2 bundle (2026-10-02): resolve observed-ray periodic geometry,
then census the same saved-state training FP factor. Driver source review
confirmed three concrete defects in the first-face prototype: ray intervals
stopped at the first cube face; `chunk_log_terms` reconstructed a minimum-image
radius after wrapping; and per-cell sky completeness was read from the wrapped
cell rather than treated as the fixed observed-direction selection scalar.
The implementation now wraps only field-cell lookup, carries unfolded q into
distance modulus/redshift/eta/K terms, removes the positive direction-constant
selection factor from this conditional numerator/denominator, and extends the
saved 192-cMpc/h distance lookup with a checked local quadratic z(q) extension.
Four focused tests pass, including uniform-field axis/diagonal agreement
across a face and proof that replacing q with minimum-image radius changes the
factor. Commit `f042518` pushed. Q-GOAL: this directly repairs the still-failing
R2 observed FP mechanics. Q-LEAN: one fixed-state training-only cohort census
is proportionate; no count-target wiring, heldout, PM, posterior promotion or
new simulation. Job410733 completed on H100 (`gpu:H100:1`), 2CPU,12GiB,30min.
Both Slurm test suites passed (4 ray tests, 2 raw-mixture regressions). Its
runtime guard stopped after20/1414 order8 rows: all20 were finite and PGC26124
reproduced the prior order8 value to rounding. First-row-per-population JIT
compiles took76.975s total; the14 warmed rows averaged0.04775s. The initial
guard incorrectly divided compile time over all rows and projected6,276.5s;
this was a runtime-estimator false stop, not a likelihood failure. Preserve
v1 at `/gpfs/kjhan/CF4/z0_density/r2_periodic_ray_fp_cohort_20261002_v1`.
The corrected guard now separates six one-time population compiles from warm
throughput and reserves additional convergence/AD compiles. Resubmit the same
bounded cohort under a fresh output path; do not change the scientific target.
First corrected submission410739 failed in one second before tests/calculation:
the new HEAD guard compared the supplied abbreviated revision literally with
the full SHA. No compute/output was produced; preserve the Slurm log. Resolve
the expected revision through `git rev-parse --verify <rev>^{commit}` before
comparing; validate this short-SHA path before another submission.
Corrected job410740 passed preflight and completed the full order8 training
census (1414/1414 finite; no zero-support rows). It found zero q>first-face
radial, numerator, or denominator mass in all1414 links at this saved state;
the earlier39 nonpositive global margins were conservative, not observed
image support. The distance lookup extended from192 to233.581 cMpc/h with
5.42e-10 max local-fit residual and 2.78e-12mag edge mismatch. Its 79-row
order4/8 sample had max |delta logpdf|=.002469 nat (above the predeclared
.001 row tolerance) but summed absolute delta=.005058 nat (within.1). The
initial v3 record omitted the row-level convergence sample; job410793 reran
only the79 order4 values from saved order8 results and persisted the violator.
Only PGC49072/index847 violates the per-row threshold; its first-face margin
is+33.57 cMpc/h and all image fractions are zero. AD/FD passed on PGC26124
(relative error1.29e-11) and conservative-bound-uncertified PGC39989
(7.75e-9); this is not a wrapped-row derivative because no training row has
nonzero q>face weight under the current8sigma state. The synthetic axis/
diagonal uniform-field test separately checks face-crossing invariance and
its AD/FD. Job410793 completed in4m39s, peak MaxRSS5.58GB; future equivalent
jobs request8GiB (>20% margin). Result:
`/gpfs/kjhan/CF4/z0_density/r2_periodic_ray_fp_cohort_20261002_v4`.
Targeted order16 job410816 COMPLETED/exit0 on H100 in2m44s (batch MaxRSS
2.05GiB). For PGC49072/index847/population4, logpdf4=4.4322654342,
logpdf8=4.4297964675, and logpdf16=4.4331565979. Thus order4-vs16 is
.000891 nat (inside .001), but order8-vs16 is .003360 nat (outside); the
order sequence is non-monotonic. All radial and FP numerator/denominator
periodic-image fractions remain zero. This is a one-row fixed-state
quadrature finding, not a periodic-boundary or posterior result. One explicit
diagnostic will compare orders16 and32 on this same row and record split
numerator/denominator terms; it stops there and does not start a blind order
ladder.
The one-row follow-up410826 COMPLETED/exit0 in2m45s (batch MaxRSS2.10GiB).
Order16→32 changes the log numerator by.001332 nat but the log denominator by
only6.63e-7 nat; the resulting logpdf change is.001333 nat, still above
tolerance. All image fractions remain zero. Rather than continue a polynomial
order ladder, the next bounded check compares equal-work layouts8×4,16×2,32×1
on this same row, testing interval-partition sensitivity while recording both
terms. No target wiring or posterior action follows regardless of outcome.
Submission410823 was cancelled during the test suite before the science runner;
to avoid ambiguity in Slurm's comma-delimited export syntax, the corrected
order list was passed by environment inheritance on410826. Test-only job410835
exposed an overstrict assertion comparing distinct quadrature layouts despite
the nonconstant q² integrand; no science result/output was produced. Commit
586fcde narrows that regression fixture to a small Gaussian and checks each
layout against its analytic second moment.
The equal-work comparison job410837 COMPLETED/exit0 in3m05s (batch MaxRSS
2.74GiB; both regression suites passed). On PGC49072, the layouts8×4,16×2,32×1
give logpdfs4.431839080,4.431946422,4.431824091; adjacent differences are
.0001073/.0001223 nat, below.001. Numerator and denominator terms are both
stable across these layouts, and all image fractions remain zero. This
supports interval-partition sensitivity as the cause of the earlier
unsegmented order8 discrepancy, but only on this one training row.
The full composite census410849 COMPLETED/exit0 in10m15s on typed H100, with
both regression suites passing. It evaluated1414/1414 training links at8×4;
the frozen79-row equal-node-budget4×8 comparison had maximum absolute
logpdf difference.000111315 nat and total absolute difference.000186513 nat.
The maximum sampled training row is inside the predeclared.001-nat tolerance.
Two fixed-state AD/FD checks passed at relative errors2.16e-11 and7.02e-12.
All periodic-image numerator/denominator/radial fractions are zero. Result:
`/gpfs/kjhan/CF4/z0_density/r2_periodic_ray_fp_cohort_20261002_v8/result.json`.
Batch MaxRSS was7,636,988K (7.28GiB) under an8GiB request; this job completed,
but future equivalent full-cohort jobs request10GiB to retain>20% margin.
This closes fixed-state training FP quadrature mechanics only. It is not a
count-conditioned joint factor, posterior, heldout score or z=0 field.

Next bundle: reconcile ownership on the predeclared PGC1085367 one-point/
one-mark training control, before any target change. The current raw target
already uses the count voxel and source-selection/RSD kernel in its normalized
FP-mark source measure; the recent periodic observed-ray cohort is a distinct
fixed-state approximation and is not a second likelihood. On the same saved
field and frozen v6 graph, form `w(j,b)=lambda(j,b) K_count(k,z_2M++|j,b)`
from the existing voxel-CDF count kernel at the linked point's individual
observed radius, including its observed-K population, LOS mixture, periodic
images, and source-volume weights. Apply the existing CF4 raw optical
FP/K-selection numerator and denominator once, at the CF4 row's own observed
redshift; do not multiply another count occurrence. Report the 2M++/CF4
observed-radius difference and a control-radius identity against the current
raw target, then compare the point-conditioned source mixture and normalized
mark factor with the periodic observed-ray approximation. Check one same-state
LOS-scale AD/FD direction using a support union rebuilt at the nominal and
two finite-difference states. Do not add `d² rho^b`, extra `f_n`, a floor,
truth-selected groups, or any posterior fit. This remains one conditional
training control, not the all-group law: selection/association and multi-member
covariance remain uncalibrated. Q-GOAL: connect CF4 distance information to the
same z=0 field as counts with explicit catalogue ownership. Q-LEAN: one link,
existing kernels, no new simulation, heldout reads, or target promotion.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must later
constrain the same NEW field, with native IDs calibration/evaluation-only.
First typed-H100 submission410874 failed in the test phase before the science
runner: the new regression tried to subtract the raw `(numerator,denominator)`
tuple as one array. The four observed-ray tests passed; two raw-mixture tests
passed and the new one errored. No science output was written; preserve its
Slurm stdout/stderr and host MaxRSS1732316K. Correct only the tuplewise
assertion and resubmit this same one-link control.
Retry410876 passed all31 focused numerical regressions (4+3+7+14+3) but
failed before creating its output directory or entering the science control:
the batch preflight resolved abbreviated commit `116018a`, while the Python
entrypoint compared that short string literally with full `HEAD`. Batch
MaxRSS was5,642,372K; no scientific result exists. `verify_source_commit`
now resolves both revisions before comparison, with short-SHA and mismatch
regressions. The prior numerical suites are already green at the unchanged
scientific source, so the bounded retry runs only this new guard test before
the same one-link calculation in a fresh output directory. Q-GOAL: restores
the intended ownership diagnostic without changing its target. Q-LEAN: a
two-test guard plus one existing control; no repeated long suite, simulation,
heldout use or posterior promotion.
The corrected retry410882 resolved the commit guard and passed both tests, then
failed a pre-likelihood cohort join: the control imported the old v5 default
split (1417 reconstructed links) while `load_inputs` correctly uses frozen v6
(1414 rows). This is a split-path wiring defect; no likelihood term or science
comparison ran. The control now explicitly binds v6 and adds a regression for
that contract. Retry only the three tiny guard/split tests and this same
training link in a fresh output path; retain the failed JSON/log. Q-GOAL: keeps
the intended frozen training graph consistent. Q-LEAN: no data-law or model
change, long test suite, gravity run or heldout access.
The third retry410883 passed all three guard/split tests and matched the v6
1414-row link table exactly, then failed before the likelihood at support-set
collection: `FreshRawSupport.build` returns `(population_packs, metadata)`, but
the caller had already separated those values and tried to unpack the
six-population pack tuple a second time. No scientific comparison ran. The
support collector now consumes the population-pack tuple directly and has a
focused tuple-shape regression. Next retry uses only these four tiny tests and
the same fixed link in a fresh output directory; no target/input/heldout or
resource scope changes.
410884 COMPLETED/exit0 in1m53s on typed H100; all four focused guard/split/
support tests passed; batch MaxRSS6,638,792K (6.33GiB) under12GiB. For the
predeclared PGC1085367 v6 training link, individual-point and CF4 group radii
are exactly equal (83.923308106 cMpc/h). Existing/default and explicit
same-radius raw log factors both equal2.314581544 nat exactly; the
count-conditioned implementation reproduces that value. The independent
periodic-ray comparator is2.351223827 nat, a fixed-row delta of-.036642283
nat. Count/ray LOS-scale AD-vs-FD relative errors are1.97e-10/6.31e-10;
the rebuilt +/- support union contains nominal/minus support. No ray mass
lies beyond its first face, and the count-voxel geometric margin is108.077
cMpc/h. Result: `/gpfs/kjhan/CF4/z0_density/r2_count_fp_ownership_20261002_v4/result.json`.
Driver accepts fixed-state mechanics only: equal radii mean this row does not
exercise distinct point/group redshift conditioning; one mark is not evidence
for group-selection/covariance calibration or posterior validity. Continue
the same one-point operator check only to its stated fixed-state scope.
410888 passed all five guard/split/support/selection tests, then stopped before
score evaluation because **none** of the1414 v6 one-point training links has
an absolute CF4-group versus linked-2M++ radius difference above1e-8 cMpc/h.
This is a cohort property under the frozen input convention, not a likelihood
result; the median-nonzero selector correctly found no candidate. Its failed
JSON/log are preserved at
`/gpfs/kjhan/CF4/z0_density/r2_count_fp_offset_control_20261003_v1/`.
Do not retry or retain a selector for a nonexistent singleton case. Together,
410884 and410888 close only the one-point count/mark mechanics: a same-radius
identity and its fixed-row ray comparison, with no full-population law,
selection calibration, posterior or heldout evidence. The material remaining
branch is multi-member CF4 groups, where group and member redshifts can differ.
Next bundle: use the source-bound v6 graph to build one normalized shared-
group-latent count/mark factor with member-redshift covariance and explicit
group-inclusion uncertainty. Its first bounded calculation is the source-only
v6 graph census in `cf4_r2_v6_multimember_graph_census.py`: freeze current-v6
member degrees and one training-only multi-member control before reading any
mark score. 410927 COMPLETED/exit0 in4s, two tests pass, MaxRSS3.4MiB. Current
v6 has6,028 training FP source groups:1,833 have one direct secure 2M++ point,
272 have multiple, and3,923 have none. None of the272 has exactly one FP row
and zero anchors, so the originally proposed isolated group control does not
exist; this does not mean multi-member information is absent. No FP mark
scores/values, heldout outcomes, field or likelihood were read. The next
source-only census ranks candidates by fewest anchors, then fewest FP rows,
then lexical source-group label, requiring at least one FP row and all linked
points in training. This is the start of the group-law bundle, not a separate
science gate. Do not multiply member marks independently, score
another count occurrence, or infer group membership from native truth. No
posterior sampling until selection and covariance are defensible. Q-GOAL:
connect actual group-distance information to the same new z=0 field while
preserving 2M++ count ownership. Q-LEAN: one source graph read, no more
singleton/ray ladders, gravity run, heldout values or target promotion.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must later
constrain those roles on the same NEW field at LG<=0.3 cMpc/h, with native
truth IDs only for calibration/evaluation.
Expanded score-blind census410928 (source `e9c06d3`) COMPLETED/exit0 in4s;
two tests pass, MaxRSS3.5MiB. The selected minimum-complexity training control
is Tempel groupT10106, with2 direct secure count members (`recno`52802/52824),
2 FP rows (PGC54049/54054), no anchor, and two distinct population/voxel keys.
Both count points are population3; their radii are125.5622 and121.6602
cMpc/h. The census read no FP mark values/scores, heldout rows, field state
or PM evolution. It defines a concrete control but does not establish physical
group membership or a likelihood. The next bounded source join checks whether
these two count points share their independently defined CF4 `1PGC` and
2M++`GID` groups; catalogue namespaces remain distinct and this cannot
calibrate group inclusion or covariance. Driver audit for this routine,
consecutive-review-avoiding source step: Q-GOAL yes (necessary graph semantics
for a same-field shared-group factor); Q-LEAN yes (one group, IDs only,
no scores, PM, heldout or fit). This follows the user's rule that a duplicated
or consecutive Fable audit is conducted by the driver. R2 remains incomplete;
MW/M31 ambiguous and M33 unresolved on the same NEW field.
Initial job410950 failed in the duplicate-edge regression before source access;
its log is preserved, and corrected retry410952 completed/exit0 in5s on typed
H100 with2 tests passing (MaxRSS3.44MiB/2GiB). Both linked PGCs map to the same
CF4 `1PGC=53982` and same 2M++`GID=2887`; each crossmatch `1PGC` agrees with its
CF4 member-table assignment. This is catalogue concordance for one frozen
training group only, not physical membership/covariance calibration or a
posterior. Full score-blind training identity census410960 COMPLETED/exit0 in5s
on typed H100; one focused test passed, MaxRSS3.44MiB/2GiB. All828 secure edges
agree with the CF4 member table. Of272 eligible multi-link training groups,
167 have one shared assignment within both catalogues;32 have partially
unassigned 2M++ GIDs,54 have all GIDs unassigned,11 span distinct 2M++ GIDs,
and8 span distinct CF4 `1PGC` groups (four with a shared and four with partly
unassigned 2M++ GIDs). Exact cross-tab and limits:
`CF4_R2_V6_GROUP_IDENTITY_CENSUS_20261003.md`. This structural result is not
physical membership, inclusion/covariance calibration or a posterior. Driver
Q-GOAL review: necessary source ownership for the same-field shared factor.
Q-LEAN: proportionate source census; no scores, PM, field or posterior gate.
The following v6-specific group/member `Vcmb` readout (Slurm410985 and410986
failed before science work; corrected H100410988 COMPLETED/exit0 in5s, four
tests, MaxRSS3.29MiB) reproduces272 training groups and828 secure links.
Across167 shared/shared catalogue groups, matched individual velocities differ
by median7 km/s, while CF4 group-to-linked-member differs by205 km/s and
CF4-to-2M++ group velocity differs by79 km/s across148 unique group pairs.
Offsets vary sharply by relation class and are not independent noise or a
covariance estimate. Full table, driver decision and preserved failure details:
`CF4_R2_V6_REDSHIFT_OVERLAP_20261003.md`; result:
`/gpfs/kjhan/CF4/z0_density/r2_v6_redshift_overlap_20261003_v3/result.json`.
The official EDD definitions say CF4 group Vcmb derives from Tully2015
2MASS-group members, while the FP-group table separately exposes `gVcmb`,
`Nest`, and `Ng`. Next bundle links the frozen272 cohort's matched PGCs to the
already archived Tully member list and its parent Nest assignments. This is
source-ownership evidence only; it cannot by itself calibrate the later EDD
revision's covariance or selection. No posterior or IC is promoted. MW/M31
remain role-ambiguous and M33 unresolved on the same NEW field. Under the
user's routing rule this immediate audit is driver-run: Astra is not invoked
for duplicated/consecutive review. R2 remains NO-GO and incomplete.
Job410740 scores all1414 training links at order8, order4/8 on all39
uncertified plus40 deterministic safe rows, and reports per-row
numerator/denominator image fractions. It stops only after a full census if
any row is nonfinite; its corrected first20 runtime guard separates one-time
population compilation from warm throughput.
User reviewer-routing update2026-10-02: Astra handles audits assigned to Fable;
if that would duplicate an auditor or immediately repeat the same audit, the
driver performs it. The current focused numerical follow-up is driver-reviewed
under this rule; it does not trigger a duplicate outside review.

Count/FP periodic-image semantics remain unreconciled and prevent target
wiring regardless of this mechanics result. R2 remains incomplete; heldout
untouched; MW/M31 roles ambiguous and M33 unresolved. Later LG observables must
constrain those roles on the same NEW field.

R2 remains incomplete and is still the current actual-data z=0 field stage;
the N256 grid is1.5 cMpc/h, not the final <=0.3 cMpc/h LG resolution.
409763/409764 at fundamental-mass600 and matched controls410010/410011 at
6000 now form a same-state/same-momentum comparison: exact GL2 target, step.08,
four integrations, same nuisance metric. At600 both proposals were rejected
(deltaH6.5127/19.1332; acceptance.0014845/4.90e-9; zero accepted movement).
At6000 both matched proposals were accepted (deltaH.04123/.19924; acceptance
.9596/.8194; white-field jump RMS.26357/.26354). This identifies the metric
change as the cause of failure for these two tested low-mass proposals and
supports6000/four-step proposal geometry locally. It is not a stationarity,
ESS, or posterior result; no long chain follows automatically.

The trace-label and checkpoint-RNG defects are fixed; focused HMC tests pass
9/9. Maximum observed host RSS was about20.61GiB in A and14.39GiB in B; the24GiB
request remained within limit, but future repeats must request at least32GiB
to retain the20% margin over the new measured peak. No gravity evolution,
heldout score, map promotion, or science-target change occurred. Close the
600-metric tuning branch and return to the unresolved R2 likelihood/calibration
and heldout-prediction requirements before considering posterior sampling.
MW/M31 remain ambiguous and M33 unresolved; later observables must constrain
those roles on this same NEW field. Details:
`CF4_R2_EXACT_GL2_CHAIN_BUNDLE_20261001.md` and
`CF4_R2_METRIC_SENSITIVITY_20261002.md`.

2026-10-02 low-k follow-up: Fable5's second read-only audit recommends
projecting the saved exact-target gradient onto the three N256 fundamental
modes before spending more on chains. The driver adopts this two-checkpoint
diagnostic but rejects using the sampler metric parameter6000 as a Hessian:
source defines `fine_gradient=grad(0.5*q.q-log_likelihood)`, while6000 is only
an inverse-Laplacian momentum-mass proposal parameter, not target curvature.
Job410046 performs four orthonormal FFTs on the saved410010/410011 accepted
checkpoints, reporting prior/likelihood/total gradient decomposition and
signed radial derivatives without a post-hoc threshold. It completed/exit0
in14 seconds. In all six endpoint/mode pairs, the partial-target likelihood
gradient points radially outward more strongly than the Gaussian prior points
inward (three-mode radial total derivatives-8190.7 for A and-2319.9 for B).
This is a model-stress finding, not proof the physical data support those
extreme IC-white modes: the observation law remains incomplete. It is not a
posterior/equilibrium or density-structure result. The driver will not extend
HMC or alter the prior; after advisory review of this important discovery, the
smallest follow-up is same-state attribution between count and CF4 mark
forces, with no new simulation or heldout access. Record:
`CF4_R2_LOWK_GRADIENT_PROJECTION_20261002.md`.

Fable5's focused follow-up returned CONDITIONAL PASS for a single count-vs-raw
FP mark gradient attribution. It confirms the sign and conjugate-pair
normalization, while warning that the force is state-dependent and the A/B
coefficients share one observer-centred phase pattern; this is not independent
evidence or a physical density feature. The driver adopts Fable's numerical
conditions (replay exact energy, reconstruct the saved total gradient, compare
complex/radial/phase mode vectors, all-mode gradient RMS and all24 nuisance
gradients) and rejects a random-phase p-value. The active target has only the
selected 2M++ count and raw FP-mark factors; TF/SNIa/SBF are not attributed.
Because the checkpoints saved only q and the summed gradient, this requires
one deterministic PMWD replay per same accepted state, followed by two
component adjoints. This repeats the forward computation but creates no new IC,
field sample or independent simulation. It is bounded by one H200 Slurm job
(2h30 wall; 2h20 application;32GiB host); no chain extension or target edit.
Job410050 failed after22m23s on a driver bug unpacking three instead of four
component gradient blocks. Preserve its v1 output/logs; no gradient was retained.
The driver fixed the mapping and added two focused block-layout tests. Same-scope
H200 retry410059 (`739fc12`) completed in1h09m42s/exit0; both tests passed.
Saved accepted states A/B reproduced exact energies to7.45e-9/1.68e-8, and
independent component values matched within9.1e-13. The count+raw-FP gradients
reconstructed the saved exact target gradients with relative L2 errors
3.78e-15/3.72e-15. Across the three N256 fundamental IC-white conjugate pairs,
the NLL radial derivative sums were A: count-8425.34, selected raw FP marks
+36.87, total likelihood-8388.47; B: count-2611.38, raw FP+92.29, total
likelihood-2519.09. Thus, at these two fixed nuisance states, the partial
2M++ count factor dominates the outward low-k likelihood force; the selected
raw-FP radial term is small and opposes it. This is evidence of stress in the
current partial observation model, not physical data support, posterior
stationarity/ESS, or a promoted z=0 map. A/B share one short-chain lineage and
are not independent universes. The active target omits TF/SNIa/SBF terms;
heldout data were untouched and no new chain or gravity evolution ran. Next:
audit the count likelihood's exposure/normalization and nuisance coupling
against its actual observation law before modifying the target or sampling
again. Fable5's read-only review of the component result returned CONDITIONAL
PASS: the graph-closed count factor drives this tested force, but this is not
physical-data support. Job410076 then computed the full-field and low-k
geometry from saved gradients only (H100/syn08,10s,exit0,3 tests). The
likelihood NLL radial derivative over the whole IC field is-4128.52(A)/-777.69(B);
within `0<|n|<=8` it is-4972.07/-2102.56. The exact fundamental shell
contributes-8388.47/-2519.09, while other shells within that band contribute
+3416.40/+416.53 against it; modes outside the band add+843.54/+1324.87.
Fundamental-shell score-force monopole/dipole/quadrupole energy shares are
.781/.119/.100(A) and .331/.506/.163(B). Across the full low-k band the
monopole shares are only.435(A)/.059(B); the geometry therefore does **not**
support a robust observer-centred monopole/radial-selection explanation.
The saved gradient contract and centered phase `(-1)^(nx+ny+nz)` were
independently verified; fundamental-mode derivatives match410059 within4e-11.
This remains latent Fourier-gradient geometry, not a physical density peak.
Fable5's second read-only review returned CONDITIONAL PASS. It independently
confirmed the arithmetic, but notes the six-mode fundamental shell is exactly
spanned by l=0/1/2 (zero residual degrees of freedom), its monopole cannot
diagnose radial shape, and A/B's apparent dipole change mainly reflects the
monopole collapsing while a similarly sized dipole rotates. Generic Fourier
anisotropy is not evidence of an angular-selection error; only a data-space
sky residual conditional on radius/population/intensity can localize one.
The speculative “central overdensity 0.5–1” illustration has no reproducible
source and is explicitly rejected for inference. Do not project out mask
multipoles, edit radial selection/prior, or sample another chain from this
geometry. The force is state-dependent, and intrinsic bias masses are
normalized over the full periodic box while counts integrate only graph-closed
training exposure. Job410095 then performed two deterministic saved-q PMWD
forwards and reproduced the exact GL2 training-count scores to2.91e-11; no new
realization, chain, adjoint or heldout outcome. Total observed/expected counts
are47,121/46,810.54(A) and47,121/47,061.09(B), but population x radius
profiles remain structured (e.g. population0 O/E=.886(A)/.872(B) over0–132,
while population4 is1.088/1.079 over108–168 cMpc/h). A descriptive octant
residual persists after normalizing within population/radius strata; it is not
replication or proof of an exposure defect. Driver retracts an exploratory
pooled Q0–Q4 ratio because merged tied cuts make those bins non-comparable.
Fable5 audited read-only and returned CONDITIONAL PASS. The driver verified
both endpoints and records one known limitation: observed counts use NGP
`floor(position/dx)`, while the prediction applies the radial cut before a
TSC deposit that can spill mass across the180 cMpc/h boundary. No operator
correction is made from these nonstationary in-sample states.

Next is Fable's one bounded frozen-field tracer nuisance profile at endpoint
A: recover that same z=0 field with one deterministic PMWD forward because
rho/velocity were not checkpointed; then optimize only the nine count-tracer
coordinates with their Gaussian prior, at most8 value/gradient evaluations,
no PM adjoint, chain, heldout outcome or law edit. Exact initial score and
tracer gradient must reproduce the saved component reference. A closed radial
x population training residual diagnoses nuisance non-equilibration only;
failure to close still does not prove a wrong physical law, and external
bias/survival plus RSD/FoG calibration remains required before sampling.
Q-GOAL: this isolates an R2 obstacle upstream of actual z=0 LG conditioning,
not LG identification. Q-LEAN: one field endpoint, nine nuisance dimensions,
the existing radial/population bins only. R2 remains incomplete at
1.5 cMpc/h; MW/M31 remain role-ambiguous and M33 unresolved. Their observables
must later constrain these roles on the same NEW field at <=0.3 cMpc/h, with
native truth IDs used only for calibration/evaluation. Full records:
`CF4_R2_LOWK_COMPONENT_ATTRIBUTION_20261002.md` and its linked geometry JSON.
The first execution410100 failed before any optimizer trial due to a newly
written JAX auxiliary-return unpack bug; initial count score/gradient checks
passed, but there is no profile result. The unpack contract now has a focused
test. Retry410101 also failed before an optimizer trial (18m04s, host peak
3.82GiB): the finite-output guard attempted to concatenate the two-dimensional
population-by-radius expectation table with scalar/vector arrays using
`np.r_`. Initial score and gradient reproduction again passed; zero optimizer
trials completed and there is no nuisance-profile result. The guard now checks
each output array independently, with a regression covering 2-D finite and
nonfinite tables. Four focused tests pass. The next bounded retry writes only
to a new v3 output directory; no target, prior, data, field, heldout access or
resource cap changes. R2 remains incomplete; MW/M31 are ambiguous and M33
unresolved.

2026-10-02 frozen-field profile410105 completed/exit0 in1h09m13s (H200,
MaxRSS3.89GiB), but stopped at the predeclared eight-evaluation limit:
`FROZEN_FIELD_TRACER_PROFILE_BOUNDED_NOT_CONVERGED`. The v3 JSON corrects an
audit-prompt transcription error: its actual initial saved-score absolute
error is0.0 and tracer-gradient relative error is8.906e-16. Objective improves
168.654nat (count log score+168.352; Gaussian-prior NLL decreases.302), but
population-by-radius L1 only falls.06410→.05854; terminal gradient norm remains
595.8. Expected training counts move46,810.54→47,395.38 against47,121 observed.
Correction to the earlier Pearson threshold label:328.062→291.938 used all59
bins with positive expected count, not the expected>=5 subset. With expected>=5,
the comparison is53 bins,326.197→290.009. For population0, the corresponding
positive-mean calculation is16 bins,144.019→144.773; at expected>=5 it is15
bins,142.229→142.920. These are one fixed, short-lineage field's in-sample
conditional diagnostics, not calibration or posterior evidence.

Fable's read-only important-result audit returned CONDITIONAL PASS. The driver
independently confirmed its bookkeeping and additionally read the saved
endpoint-A raw-FP tracer gradient: its L2 norm is7.37 versus1,952.67 for the
count-plus-prior profile gradient (0.38% at that initial point). The derived
final rate-coordinate gradient is about+549 of the595.8 norm, but the remaining
gradient is still large and the radial/population residual persists. Do not
extend nuisance optimization; it cannot distinguish count-law misspecification
from a nonstationary fixed field and has diminishing gains. Correct the old
profile's final radial-bin label: values clipped to that bin mean>=180, not
180–192; the v3 historical JSON is preserved, and new summaries label the tail.

Closure result410117 completed/exit0 on H200/syn104 in10m28s; the Slurm batch
MaxRSS was3,521,324K. Its12 existing count-operator regression tests and3
diagnostic tests passed. The exact active-TSC reference gate passed: saved-score
absolute error2.91e-11 and max radial x population mean error5.21e-11. At the
same endpoint-A field and initial tracer coordinates, TSC and diagnostic NGP
exposed means are46,810.537 and46,820.573 (difference10.036); maximum single
aggregate mean difference28.854. One seeded NGP Poisson voxel-count mock has
47,103 objects; aggregate Pearson checks are71.715/53 bins under NGP and
72.099/53 under TSC, both inside the approximate one-draw95% interval. This
verifies mechanics only; the one draw is upper-tail and is not validation.

Driver independently recomputed actual TRAINING-only radial x population
aggregates (96 bins; this is not the full voxel likelihood): the un-factorialized
binned score is292,733.987 (TSC) vs292,746.405 (NGP), delta+12.418nat; L1 is
3020.466 vs2903.874; the expected>=5 Pearson statistic is326.197/53 bins vs
305.005/53. NGP's apparent gain is concentrated: the population3 `>=180`
tail has38 observed vs66.685 (TSC) and37.830 (NGP), accounting for12.34 of the
21.19 Pearson-statistic decrease. Excluding that bin leaves313.858 vs305.004
over52 bins. Population0's total is8,859 observed vs9,837.428/9,836.761
expected (TSC/NGP); its expected>=5 Pearson rises142.229→144.012. Across
36–96cMpc/h, observed population0 shares are.116,.152,.218,.350,.560; TSC
shares are.144,.183,.272,.397,.596, and the NGP share shifts by at most.0033.
Thus NGP plausibly accounts for much of the single outer-tail deficit, not the
bright population split. Actual-data voxel score under NGP was not computed;
do not generalize this coarse-bin result to the full voxel likelihood.

Fable5's read-only result audit returned CONDITIONAL PASS and independently
reproduced the coarse aggregate numbers. Driver verified the catalogue's NGP
`floor(pos/dx)` mapping in `scripts/cf4_r2_common_catalogue.py` and the
deposition passthrough through the chunked predictor. Record the Pearson
threshold correction above; these are bins in a same-data, nonstationary
conditional check, not inferential degrees of freedom. Q-GOAL: yes, this
removes one possible R2 operator explanation without promoting a z=0
posterior. Q-LEAN: yes, one field/two operators/one mock was proportionate.

The audit's next suggestion was a field-free K/LF-transfer share calculation.
`scripts/cf4_r2_uniform_transfer_shape.py` implements its bounded form and
records JSON at `/gpfs/kjhan/CF4/z0_density/r2_n256_uniform_transfer_shape_20261002_v1/result.json`: one
uniform-density, zero-velocity field, 24-point radial GL quadrature in the
36–96cMpc/h training shells, and a61-by-61 grid over the existing standard-normal
alpha/mstar coordinates. At the saved initial alpha/mstar coordinates
(.103,-1.608), the field-free pop0 shares are.153,.213,.299,.427,.621, above
the observed.116,.152,.218,.350,.560. A nearby grid point (.2,.1) gives
.104,.154,.230,.353,.557, with conditional-binomial deviance6.35 and only
.025 prior NLL. This is a shape-capacity diagnostic on reused training counts,
not a fit or calibration. The five luminosity-bias coordinates cannot affect
the uniform-field shares: full-box-normalized rho^beta is identically1 at
rho=1. In the actual fixed-field TSC table, population0 shares are instead
.144,.183,.272,.397,.596, so the remaining deficit is coupled to field/bias
weighting and/or the observation law; the field-free result cannot decide it.

Fixed-field prior-centre sensitivity410125 completed/exit0 on H200/syn104 in
3m06s; batch MaxRSS3,409,652K. With only alpha_white/mstar_white set to0,
the active-TSC marked-count mean falls46,810.537→37,944.431 against47,121
training observations. The five bright-population0 shares move closer but
cross to the low side: observed.116,.152,.218,.350,.560; baseline.144,.183,
.272,.397,.596; prior-centre.097,.128,.205,.323,.530. The unprofiled count
score falls1,895.595nat; that raw comparison is dominated by the19.5% mean
deficit. Candidate full table was not saved in v1, so its reported L1/Pearson
cannot be rate-adjusted after the fact. Record:
`/gpfs/kjhan/CF4/z0_density/r2_n256_lf_prior_center_sensitivity_20261002_v1/result.json`.

Fable5's read-only important-result audit returned CONDITIONAL PASS. Driver
verified the source-law scaling: `tracer_masses` adds2*u0 to log rate,
`intrinsic_biased_source_masses` multiplies masses linearly by exp(log-rate),
and the sparse Poisson factor uses training-count sum47121 and the exposed-mean
integral. The Poisson-MLE total-rate multiplier gives s=1.241842,
delta_u0=.108298 for the candidate, and s=1.006632 for baseline. On a
like-for-like rate-profiled comparison, the candidate count score is-136,248.385
vs-135,381.415 baseline (delta-866.970nat); evaluating the Gaussian prior at
these MLE rate points, the count-plus-prior objective is worse by865.707. This
is not a joint MAP over the rate prior. These are one-field,
training-only conditional sensitivities, not a posterior or independent
prediction. The prior-centre share change brackets the observed shares rather
than matching them, and all five density-bias coordinates and sigma_los were
held fixed. The unprofiled candidate's L1/Pearson are not the rate-profiled
statistics. The raw-FP factor was not reevaluated.

Job410128 completed/exit0 on H200/syn104 in5m24s (batch MaxRSS3,507,112K).
Its TSC scalar-rate identity passed (map error3.55e-15; direct-vs-analytic
score error0), and four focused sensitivity tests passed. Fable5's read-only
result audit returned CONDITIONAL PASS. Driver independently reconstructed
the6x16 training-only observed, baseline and candidate tables from the saved
profile, closure and v2 outputs; the historical v2 JSON is preserved. The
audit found two reporting defects: `baseline_rate_profiled` actually held the
unprofiled closure table, and the aggregate score delta compared the candidate
profiled table with the unprofiled baseline while the full score delta used
both profiled scores. The JSON generator and regression test are corrected;
future outputs store the observed table and separately named unprofiled and
profiled baseline metrics.

Corrected derived baseline metrics after profiling its rate are mean47,121,
radialxpopulation L1 2,964.260 and expected>=5 Pearson321.977/53 bins. The
prior-centre candidate has the same profiled total but L1 7,380.534 and
Pearson2,109.859/53. Its population0 share is closer to observed in each of
five36–96cMpc/h shells, but crosses to the low side; this share-only improvement
is not a count fit. The candidate is mainly an Mstar shift: baseline
Mstar≈-23.602, alpha≈-.9368 versus candidate -23.28/-.94. With the five bias
coordinates and sigma_los frozen, count-score delta is-866.970nat, correctly
decomposed as radialxpopulation aggregate-803.099 plus within-bin voxel
allocation-63.871. Aggregate score contributions by population are
pop0-294.774, pop1-4.643, pop2-52.636, pop3-190.900, pop4-96.424,
pop5-163.721nat. Pop0 contributes-337.650nat beyond96cMpc/h; pop3 contributes
-197.310nat beyond96, especially144–180. These are fixed-field in-sample
diagnostics, not a significance test, law rejection, posterior or independent
prediction. The count-plus-prior conditional objective is still865.707nat worse
at count-only MLE rates; it is not a joint MAP and cannot exclude joint fitting
of other tracer coordinates or a corrected observation law.

Read-only source audit confirmed two apparent-K samples x three observed
absolute-K bins, true/observed modulus plus redshift correction and RSD before
TSC deposition, and five true-K bias responses normalized over the full
periodic box (not per chunk). No simple radius-wiring defect was established.
Legacy external CAMELS bias and FoG artifacts do exist, so saying that such
external evidence is wholly absent was too broad. However, the six CAMELS bias
values are ordered by ascending stellar-mass sextile, while the old N32 v8
calibration applied them index-for-index to six observed 2M++ bins ordered by
apparent-K sample and bright-to-faint absolute K. The active R2 target instead
has five latent true-K bins including unbounded tails, transferred into six
observed bins; no validated conditional crosswalk connects these estimands.
Therefore neither unchanged nor reversed injection is justified. The old v8
holdout remains a development-only predictive check under that old mapping,
not a validation of the present R2 bias parameters. The active N256 source
field is 1.5 cMpc/h while its count grid is N128/3 cMpc/h; the old v8 was N32
on a 384 cMpc/h box and used h=.6711, whereas the current fixed-field path uses
h=.746. Current R2 source code uses the H0=100/h-scaled absolute-magnitude
convention consistently for catalogue labels and model transfer; old v8 used
unscaled physical magnitudes and the same numerical edges, so its bins
selected different rows and its Mstar=-23.28 knee was nominally about0.87mag
too faint at h=.6711. This further precludes treating its result as active-R2
calibration; it is not evidence of an h-convention defect in the current
source path. See `CF4_R2_CAMELS_BIAS_CROSSWALK_20261002.md`.
Separate from CAMELS, the six bias exponents in Lavaux & Jasche (2016) Table 1
are same-catalogue plug-in estimates: ARES inferred luminosity-dependent bias
from 2M++ and the authors then fixed those estimates in BORG on the same
catalogue. They are external to CF4 distance marks, but not independent of
2M++ count data; do not use them as an informative prior alongside those same
counts without modelling their dependence. The historical six-observed-bin
partial sampler did center bias nuisances on these values, but remains
explicitly uncalibrated and not an R2 delivery; the current N256 five-true-K
GL2 target uses broad unit-centred regularizers for its **five bias
coordinates** and does not inject the six-vector. The two bin definitions are
not interchangeable. A local heldout
score is prospective only if the bias estimate excluded those rows; that
independence is not established for the paper's full-catalogue ARES estimates.
The paper also maps ARES linear-regime bias to BORG's power-law exponent via
an approximate equality that its authors say is not exact. Fable5 concurs
(CONDITIONAL PASS for documentation only; Q-GOAL/Q-LEAN pass). Exact row-level
overlap with the paper input was not re-derived. Preserve the paper values
only as historical reference points.

The “unit-centred” description applies only to those five bias coordinates,
not every tracer nuisance. The active target's two LF-shape coordinates use
the Lavaux--Hudson (2011) 2M++ row `|b|>10, K<11.5` as their reference:
`Mstar=-23.28 + 0.2*u8` and
`alpha=-1 + 0.06*exp(0.5*u7)`, with `u7,u8 ~ N(0,1)` from the caller's
`.5*q.q` penalty. The log-rate coordinate also has scale 2 and is centered on
the reference-window fraction computed at these default LF values. Thus
`Mstar` has a weak empirical regularizer in physical units (0.2 mag versus
the row's quoted 0.01-mag error), while `alpha` is restricted to `alpha>-1`,
has median -0.94 and local width about 0.03, comparable to the row's quoted
0.02 error. The rate center is broad but also inherits that LF reference.
These references are estimated from a subset of the same 2M++ catalogue used
by the active count likelihood; exact row-by-row overlap was not measured.
They are same-survey empirical regularizers, not independent external LF
calibration.

Fable5's read-only audit returned **CONDITIONAL PASS**. Driver verification
against the primary paper and current target source confirms the lineage and
transforms. The paper also reports a CMB-frame estimate for
`-25<M<-21, 5000<cz<20000 km/s` of `alpha=-0.73, Mstar=-23.17`; the active
alpha coordinate places this alternative about three standard-normal units
from its reference. This does not invalidate the existing fixed-field
diagnostics, and R2 remains NO-GO for posterior promotion. Before any future
sampling, revise the alpha coordinate together with the finite faint-end
population law and radius-dependent selection/transfer. Do not add a second
LF likelihood from the same 2M++ catalogue as if independent. Retaining the
broad Mstar regularizer with explicit sensitivity is provisionally acceptable;
an external LF crosswalk, formal dependence model, and exact row-overlap
accounting are deferred unless needed by the revised likelihood. No fit, job,
or target change was made for this audit. The Lavaux--Jasche six-bias issue
above is distinct: those values are not injected into the active bias target,
whereas the LF-shape coordinates are active through `u7` and `u8`. See
`CF4_R2_CAMELS_BIAS_CROSSWALK_20261002.md` for the detailed disposition.

Follow-up implementation2026-10-02 (`2e6c8aa`, decision record
`CF4_R2_FINITE_LF_PROCESS_20261002.md`) rewrites the active selected count/raw-FP
factor directly as finite `I_selected/I_reference`, algebraically cancelling
the old `I_bin/I_reference * I_selected/I_bin` factorization where alpha>-1.
The active alpha coordinate is now the broad development regularizer
`alpha=-1+.5*u7`; the finite selected process remains defined below -1 without
normalizing an unobserved infinite faint tail. Four focused source/transfer/
count/support test files were submitted as Slurm410407 (H200/H100/A100, 12GiB,
30min). It FAILED after18m34s (exit1, MaxRSS2,628,512K), not OOM: the first
two files passed6/6 and13/13; `test_cf4_r2_shell_cdf_count.py` reached a test
adapter `TypeError` because its direct reference lacked the new optional
`finite_reference_interval` keyword; the final raw-volume file did not run.
Commitcff924a adds the keyword and finite-mode dispatch to the direct helper;
the runner now requests4GiB (above the prior measured2.51GiB by>20%) and45min.
Retry410445 COMPLETED/exit0 on the A100 partition in19m16s: all four files
passed,36 tests (6+13+14+3). Batch MaxRSS4,778,672K (~4.56GiB) exceeded the
4GiB request without an OOM exit; future full-suite runs request6GiB to
restore the20% margin. The earlier dependent throughput job410425 could never
run after410407 failed and was cancelled. Replacement410446 COMPLETED/exit0
in11s. Synthetic finite-transfer forward/gradient median times are1.36/2.26/
4.86ms for4096/65536/262144 sources; this excludes PMWD, full target adjoints
and support construction and cannot authorize sampling. Both historical
submissions used generic GRES; current runners use typed A100 requests, with
future mode selection checked across H200/H100/A100. Output:
`/gpfs/kjhan/CF4/z0_density/r2_lf_transfer_cost_20261002_v1/result.json`.
Old saved tracer coordinates require the mapping in the decision record to
preserve physical alpha and are not directly reusable unchanged. Existing
fixed-field sensitivities remain historical diagnostics, not invalidated
posterior results.

Next bounded calibration-input bundle uses existing TNG100 native K photometry
and the preserved1.5-cMpc/h total-matter moments to measure five true-K proxy
responses and velocity residuals, with train/buffer/test spatial separation.
No new data download or gravity evolution. It resolves whether existing
external information has relevant luminosity-bin support; native K-to-2MASS
Ks, hydro/cosmology, NGP-to-PM scatter and stellar-to-subhalo COM differences
remain explicit before any R2 prior use. No CAMELS mass-sextile injection,
actual CF4 heldout access or posterior promotion. Q-GOAL: supply missing R2
tracer-law evidence upstream of the actual z=0 delivery. Q-LEAN: one448-file
catalogue read plus a7MB preserved moment grid, bounded20min/4GiB Slurm.
MW/M31 remain ambiguous and M33 unresolved on the NEW inferred field; native
IDs label external calibration only. Record:
`CF4_R2_NATIVE_K_RESPONSE_20261002.md`.
Source246f036 Slurm410483 failed before extraction because its Syntax-visible
scratch source was absent on the compute node. I/O-only staging copied146MB
of fixed catalogue fields to the shared project path; retry410484/source78c1d83
COMPLETED/exit0 in3s on typed A100. Five native-K proxy bins have119/637/3181/
4993/318394 objects and native beta1.792/1.364/1.204/1.136/1.035. These cannot
enter R2 as priors yet: the faint tail has only40,223 objects with>=100 stellar
particles, and K-to-Ks/NGP-to-PM/hydro/cosmology correspondence remains open.
Next same-source extension compares finite faint-K slices and stellar
resolution floors with the existing spatial split; no gravity or CF4 holdout.
That extension410505/source dd66f7f COMPLETED/exit0 in5s. Native beta changes
by<=.002 under the100-star cut for[-21,-17), but1.074->1.218 in[-17,-16).
The cut changes populations and does not uniquely diagnose numerical error.
Source-support bundle is closed; next is a native position/velocity/K RSD
observation mock with matched source/count spacings, joint tracer fitting and
heldout mock prediction. This reuses the known matter field, no IC/evolution.
K-to-Ks, NGP-to-PM response and one-box/hydro/cosmology limits must enter the
calibration assessment before any external coefficients become R2 priors.
Next bundle implemented in `CF4_R2_NATIVE_RSD_MOCK_20261002.md`: each native
galaxy/cell occurs once in two disjoint translated source halves, supplying
near/far apparent-K coverage with the SAME density/velocity/mark input.
This is known source-window calibration geometry, not a physical384 mock.
One training-only nine-tracer fit plus fixed source/LOS/deposition controls,
125000 native1.5 source cells,3 count spacing;90min Slurm/75min application,
10GiB host/2CPU/typed GPU. No actual CF4 heldout outcome, PM evolution or
R2 posterior/IC promotion; source K/Ks and native/PM limitations remain.
Source c5f0085 Slurm410544 COMPLETED/exit0 in2m44s on H100/syn08.
All4 observation tests passed, and6050 unique selected galaxies populate all
six bins. Train/test/buffer counts2656/2360/1034; all selected rows have>=300
stellar particles, including156 true-faint-bin migrants. This removes the
unresolved-tail concern for this selected proxy sample, not passband/model
calibration. Derivative error4.45e-6 passes; fit stopped at24 iterations,
gradient_inf2.453, NOT convergence. Development test score improves198.379
but expected counts2503.112 exceed2360 by6.1%, and population/radius L1
worsens222.629 to285.717. GL4/LOS16 intensity changes<.1%; diagnostic NGP
changes36.6% and has two occupied zero-support training cells. No law switch
or calibrated-prior adoption follows. d7d2b25 continuation410551 RUNS through
Slurm on H100/syn08: identical source/model, fresh L-BFGS history, training
rate reprofile, maximum200 iterations/240 evaluations. Stationarity requires
optimizer success AND gradient_inf<=.001. The already inspected development
test is not pristine validation; actual CF4 heldout remains untouched.
410551 now COMPLETED/exit0 in3m57s,48iterations/53evaluations with proper
gradient convergence (.0005615). Test mean remains2504.1049 vs2360;
optimizer extension is closed, NOT an R2 promotion.410555 NGP refinement
COMPLETED54s: GL4/LOS64 leaves one occupied training zero and one4.15e-15.
410561 exact native-row readout COMPLETED9s: these galaxies' local matter-
mean velocity residuals are8.55/9.76 fitted sigmas;8/6050 selected galaxies
exceed8sigma. Native boundary locations and other source contributions
prevent a unique support diagnosis. This important physical closure finding
triggers bounded Fable5 advice on normalized velocity tails, count-operator
mismatch and the minimal decisive computation (Q-GOAL/Q-LEAN). No arbitrary
floor, sigma inflation, NGP adoption or calibration prior injection; native
source roles never label generated field candidates. Request/result scope:
`config/cf4_r2_native_velocity_tail_fable_20261002.txt`.
Fable returned CONDITIONAL PASS. Driver adopts normalized Gaussian-mixture
candidate and interpolation/residual readout, but amends selected-sample
calibration and rejects claims that endpoint checks prove all convergence,
zeros uniquely locate truncation, or correlated-half scatter certifies prior
uncertainty.410569 COMPLETED8s:3785 preselection training galaxies yield
mixture21.13/303.05km/s,weight.4519 versus single Gaussian204.32; TSC
mass/momentum interpolation preserves mixture structure. Edge/interior broad
width400.81/188.39 signals heterogeneity; only ONE rho<1 training galaxy.
The finite-variance t comparison hits its lower nu bound. These are native
development descriptors, not R2 priors or central/satellite labels. Next is
consistent voxel-boundary marked-count integration and training population/
environment support before one mixture count refit. No new simulation or
posterior/IC promotion. Full assessment in the native-RSD bundle record.
410574 scalar marked voxel-CDF probe COMPLETED1m17s,9tests pass and width
derivatives agree~1e-6. Mixture restores support at the two keys but means
remain2.387e-7/1.581e-6; source/mark refinement changes6–8%, not complete
convergence. No claim that the global mixture closes tracer calibration.
Next410575 cheap training-only readout measures conditional broad widths
against existing matter diagonal velocity dispersion, retaining both mean
and variance. Diagonal-LOS proxy excludes missing cross-axis covariance;
native scalar fits remain development evidence, no R2 prior injection or
test-selected law. No full-count refit until closure evidence is adequate.
410575 now COMPLETED20s with9tests pass. Three-parameter conditional
matter-dispersion mixture improves training AIC43968.17 vs46569.18 for
global widths (TSC-moment variant45268.52 vs47833.52). Evidence supports
using field mean AND local dispersion, not an all-space Gaussian width.
Still native training-only, not independent validation or R2 prior. Next
connect conditional variance to boundary-integrated marked count prediction;
preserve diagonal-covariance, K/Ks and full-population calibration limits.
30c77c3 connection410587 COMPLETED1m49s,11tests pass: conditional two-key
means.000304812/.00282951 versus global2.387e-7/1.581e-6. Scale derivatives
agree~1e-7; not population calibration or full quadrature convergence.
63324d6 implements full-grid voxel-boundary partition and marked integration:
all plane crossings covered by the finite8sigma periodic-domain proof,
no point-NGP indicator or TSC smoothing.410592 SUBMITTED typed H100/2CPU/
10GiB/30min; currently PENDING Priority, not failed. One fixed native
conditional forward prediction, sourceGL2/mark4, no fit or CF4 outcomes;
requires scalar/full-grid agreement plus all occupied-key support and
empty-cell-inclusive count readout. Full record remains the native-RSD
bundle document. R2 posterior remains NO-GO; no IC or LG role promotion.

Terminal update:410592 COMPLETED32s and410600 COMPLETED4m34s, both exit0.
GL4/mark8 full native prediction has zero occupied support failures and
training log-dispersion-scale adjoint relative FD error2.59e-8. Expected
development-test counts2500.044 vs2360 retain5.93% excess; this is not an
actual CF4 posterior or untouched validation. Host peak2.573GiB. Close the
two-key/native-forward numerical branch. Next bounded shared-closure bundle
connects the SAME normalized mixture, diagonal matter variance, periodic
voxel radial density and fresh source support to count AND raw-FP interfaces.
Existing production defaults remain unchanged; no native coefficient becomes
an R2 prior, no long chain or new gravity evolution. Small Slurm regressions
precede any actual-field pilot. Q-GOAL: remove the inconsistent omission of
physical dispersion from the joint observation model; Q-LEAN: reuse existing
field moments/operators, no new gate framework. MW/M31 remain ambiguous and
M33 unresolved; their later observables constrain the SAME NEW state, not
native calibration IDs. Full-population calibration, PM/hydro/passband
crosswalk and actual CF4 posterior/heldout requirements remain unresolved.
Shared-interface bundle now numerically complete:410630 COMPLETED4m35s,
17tests pass and dependent410632 COMPLETED39s,2tests pass. Optional count
and raw-FP interfaces share normalized mixture weights, periodic voxel
geometry and physical variances; moment readout preserves mass/momentum/raw
second moments and chunk gradients match unchunked. This is a small-fixture
wiring result, NOT actual-state IC adjoint or R2 posterior/calibration.
Production sampler defaults are unchanged; no bundle job remains running.
Next integrate the same moments in the actual-state joint target with
explicit closure uncertainty before inference, preserving heldout data.

The active fixed-field LF-only sensitivity line is closed: no Mstar scan,
broad optimizer, chain or simulation follows from it. Before another fit, R2
must source and validate the radius-dependent selection/transfer and obtain
genuinely independent or explicitly joint calibration inputs for the actual
five-true-K/six-observed-bin model. The new alpha regularizer remains
development-only pending prior sensitivity and those same calibration
requirements.

The source audit also exposed an exact-tie LF-transfer gradient discrepancy:
before correction, transfer values matched the direct reference but an exact
boundary shift gradient was-2.53536 vs-1.58306. A whole-call reference fallback
was rejected because inactive out-of-table source nodes can tie on every
production chunk. The final fix selects by original magnitude bounds and
averages tied survival values, preserving the nested direct max/min
subgradient elementwise without a slow chunk fallback. `test_cf4_r2_lf_transfer_reuse.py`
passes3/3, including apparent/three-way ties, JIT, shift, Mstar and alpha
gradients; the inactive-clamped-source shell-CDF test, four observed-magnitude
tests and five LF sensitivity tests also pass. Job410128 was forward-only and
is unaffected. No blanket old-gradient rerun is justified; odd-order
GH/zero-velocity gradient results remain tie-sensitive unless their saved-state
weighted tie contribution is shown zero.
Q-GOAL: correct R2 interpretation/gradient mechanics without claiming LG
identification. Q-LEAN: one saved-field replay, exact table decomposition and
one compact derivative fix; no scan or new chain. No heldout outcomes read.
R2 remains NO-GO at1.5cMpc/h. MW/M31 stay role-ambiguous and M33 unresolved;
their observables must constrain the same NEW field at<=0.3cMpc/h, with truth
IDs reserved for calibration/evaluation.
## Historical R2 continuation — 2026-09-25: no new TNG dependency

Latest408447 COMPLETED59m50s,2/2 affine-corrected N256 moves accepted. Fine
gradient/value match, estimated GPU30.56GiB, host18.40GiB. Low-band power
.698109->.700069 still drifts; no stationary posterior or R2 completion.
Next bounded two-chain exploration and its limits are specified in
`CF4_R2_TWO_CHAIN_BUNDLE_20260929.md`. It wires actual field-moment accumulation
and richer traces before up to2x24GPU-hours, not a production science claim.
408499 passed11 sampler/schedule/moment regressions.408500 generated chain B's
independent high-mode initializer.408501/408502 now run the two bounded
chains: each first L4 warmup proposal accepted, low-band powers .704224 and
.702354, still drifting.408504 passed3 scalar-diagnostic tests;408505 runs
after BOTH chains terminate to produce diagnostics and an illustrated report.
408508 passed whole1414-row exact streaming-value equivalence in6m34s,
max batch1.069million of18.843million components, no dropped support.408509
compared GL2/GL4 on two saved pilot fields without new PM evolution: COMPLETED
1h55m17s, corrections+.368041/+.301848nat, change-.066193nat. This is two
nearby-state evidence, not posterior sensitivity or a global error bound.
Its one-page actual-example PDF was viewed.408565 completed a3s read-only
saved-gradient secant: curvature change approximately explains affine-force
residual, but simple shellwise scaling leaves96.5–100% gradient residual;
do not promote a new correction from one direction. Source60fe8b4 pushed.
Chains A/B each reached6 accepted warmup moves and entered L8; low-band power
still drifts (.746770/.744296), no retained posterior claim. R2 remains
incomplete; sampling, numerical/prior sensitivity and untouched prediction
must not be replaced by these implementation successes.
The paragraphs below preserve chronology and may refer to now-terminal jobs.

Current R2 continuation2026-09-29:408353 COMPLETED the entire1414-row live
raw-mark readout (21m47s); packing208.77s, core full26-coordinate value/gradient
226.01s, seven-row gradient agreement2.91e-13, whole directional error3.84e-6.
408357 COMPLETED the source-volume count/native-field adjoint (26m29s).
Source2^3/4^3 gradients cost86.31/685.79s; fine velocity-direction error2.47e-6.
Fine-minus-coarse count log score14.51576 is NOT negligible merely because
expected total counts differ only.068. Do not replace the fine target by the
coarse one.408372 passed the actual-cut comparison;408374 COMPLETED20m47s
with full native-field/24-nuisance adjoints and refreshed support at all FD
states (error3.90e-6). Its legacy1414 readout agrees to6.22e-15; native
gradient220.80s legacy vs236.90s fine-cut shortcut, not a legacy speedup.
408375 corrected-sampler mechanics tests passed.408389 COMPLETED1h43m40s:
bounded N128 joint IC/raw/count pilot,7/8 accepted (3/4 after discarded warmup),
host7.92GiB. Accepted IC/rho/mean velocity/physical dispersion saved; white
mean-square.649291->.698109 is still drifting, not stationary posterior UQ.
The pilot (max8 proposals,120min Slurm)
replaces queued408381 before launch to retain accepted rho/mean velocity/
physical dispersion from the SAME fine-energy forward, no reporting rerun.
Startup PM field agrees with the saved IC state; AD/value primal agrees;
full IC+24nuisance directional error.001421 passes the predefined.002 check.
Acceptance uses fine4^3 energies, forces use2^3; proper priors once, no old eta.
408393 is its dependent actual-example PDF (408390 replaced before execution
to correct native-node pixel origins).408392 TIMEOUT after45m02s, NOT OOM:
lifted N128 field on256 source cells/axis,1/8 cell mass, estimated device
peak5.43GiB and host2.48GiB, but no completed native-gradient/score result.
The retry changes the count readout algebra, not the likelihood or wall cap:
exact Poisson sufficient statistics retain all occupied intensities AND the
integrated intensity over all exposed cells, including empty cells.408406
passed3 compact-deposit tests;408407 passed its volume/LOS/periodic gradient
test but TIMEOUT at30min before the native reference comparison completed.
Do NOT promote the compact backend. Pending408408/408409 were cancelled
before execution. Continue with the already-tested full-grid/source-chunked
law, increasing the source batch from32768 to2097152, with unchanged45min
cap and20% compiled device margin. This reduces4096 volume/source calls to64
without changing the quadrature; runtime improvement is not yet measured.
This is a source-workspace retry, not actual N256 dynamics.
408410 COMPLETED10m17s with the unchanged full-grid law and2097152-source
batches. Full native gradient532.10s, estimated device20.41GiB, host3.26GiB;
physical mass ratio1, rate AD agrees with2*(47121-expected) to1e-12, density
normalization directional derivative-5.68e-13. Small-batch overhead, not
insufficient memory, prevented the preceding45min profile from finishing.
The compact backend is not needed or promoted.408411 is the dependent actual
N256 dynamics initializer;408412 then performs four bounded joint transitions
(source force1/fine2, frozen observed128 keys/exposure, physical rate1/8).
Source4 sensitivity, convergence and posterior UQ remain outstanding.
The actual-example report is now408413: pending408393 replaced before launch
to mask empty velocity cells and clarify sampling vs optimization.
408411 then COMPLETED1m06s: actual N256/1.5 LCDM initialization, not replicated
coarse density. Fourier restriction error2.00e-15; imaginary residual1.25e-15;
rho mean1. Forward plus smooth-probe adjoint7.01s after compilation, estimated
device19.36GiB/host4.13GiB. The overall white mean-square.96272 includes NEW
prior high modes and is NOT evidence that inherited low modes converged.
408412 RUNNING the actual joint pilot; same-IC reconstruction agrees to
rho1.08e-12 and velocity8.12e-10km/s. No N256 accepted proposal yet.
Report408428 corrects an unsupported subscript glyph in408413, and all three
pages were visually checked: `r2_raw_joint_report_v2/` (N128 transition report).
408412 has passed coarse AD/value equality and full PM+24-nuisance FD
(relative6.90e-5). Fine-value memory estimate5.66GiB, first fine value398.16s;
steady coarse gradients147s. First N256 transition is pending. The initial
restricted128 white power.6981094277 reproduces its parent, while total256
power.9627204 includes newly drawn prior modes. Fable's larger-sampling advice
is CONDITIONAL; driver corrections and finite next scope are recorded in
`CF4_R2_N256_SAMPLING_ADVICE_20260929.md`. No large production submitted.
Streaming posterior moments are newly implemented but not wired into current
pilot:408436 passed3 tests (repeated rejected states, occupied-cell velocity
conditioning, checkpoint continuation). Physical and posterior variances are
separate; sample variance is not MC-error evidence.
Terminal408412 COMPLETED59m45s but0/4 proposals accepted: Hamiltonian errors
19.6921,12.8524,4.3137,4.2649. This is a FAILED TRANSPORT PILOT, not R2
posterior success. Fine endpoint cost~289s, steady force~147s, host16.16GiB.
No large two-chain run.408447 is the bounded two-proposal fixed-anchor force
repair, depending on408445 corrected-sampler regressions. It continues from
the saved same-target state, step.0522046,L2,no adaptation; one fine derivative
sets a frozen linear correction to the surrogate, never to the fine target.
408439 passed2 analytic affine-wrapper tests. Actual efficacy is unproven.
408444 rendered the N256/transport/mechanics example PDF, all3 pages viewed:
`r2_n256_pilot_report_v1/`. It shows the0/4 result, not an equilibrium map.
All jobs are Slurm; heldout untouched, no stationary posterior claim.
Corrected3-page actual-example Korean report408370 was rendered and every
page viewed: `/gpfs/kjhan/CF4/z0_density/r2_raw_bundle_report_v2/`.
Finite delivery route and Fable advice disposition are in
`CF4_R2_DELIVERY_ROUTE_20260929.md`: actual conditional N256/1.5 posterior,
numerical/prior sensitivities and current untouched-split prediction remain
required. N128/3 is development, not completion. MW/M31 remain ambiguous,
M33 unresolved; later roles/observables must constrain this SAME new field.
Historical status paragraphs below are not the current job/target state.

Latest R2 work2026-09-29: the raw FP/K observation component is implemented,
not merely proposed.408343 passed6 tests and actual1414-row cut/aperture checks;
408344 exported219401 exact same-field source/bin weights and reproduced the
old FP factors to1.55e-15.408345 jointly fitted15 shared raw-population parameters
at the accepted408337 field in59s;29 iterations, CPU/gradient checks passed.
No data-derived independent prior and no old eta factor were added.408346
completed1000 selected-mark draws per row: four marginal training mean ranks
(.497,.496,.501,.507), no gross discrepancy in these marginal plots. These
are SAME-data, fixed-field model checks, NOT independent calibration/validation
or posterior UQ. All three new Korean PDFs were actually rendered and viewed.
The field target is still unchanged. Next: joint raw-population/LCDM-field
inference with explicit weak priors, rather than freezing fitted calibration
or insisting that only an external zero can make conditional inference possible.
Focused Fable advice returned ADVISE PROCEED with source-cell quadrature first.
Driver adopted that priority but corrected its normalization/gauge claims;
see the recorded disposition.408347 completed all24 centre/2^3/4^3 comparisons
but failed a strict endpoint check;408348 verified the cause was archive
float32 vs promoted geometry (original-precision error1.27e-13) and added
seven8^3 controls. Centre->volume count changes~5–12%,4^3->8^3<.05% on the six
count examples; raw velocity derivatives also change materially. PDFs viewed.
408349/408350 passed live dense/sparse raw-mark value and26-coordinate checks;
sparse core~.25–.34s/row, packing/compilation reported separately.408351's cheap
count-kernel surrogate was rejected (up to6.12% error).408352's unchanged-volume
LOS4x8 control differed from4x32 by<=.0162% on six keys, not a global bound.
408353 now expands the live mark readout to ALL1414 training rows; shared
normalization and priors ONCE, fresh state support, bounded45min/H100/12GiB.
No gravity evolution, fitting, heldout or count-target replacement yet.
This marked cohort is an FP SUBSET, not all CF4 methods or multi-FP groups.
Focused delivery/coverage/cost advice is requested in
`config/cf4_r2_delivery_advice_20260929.md` after the substantive volume finding.
Numerical local controls are not global posterior validation.
See `CF4_R2_RAW_SELECTED_MODEL_20260929.md`. R2 remains open; N256/long HMC
not launched, heldout untouched, MW/M31 ambiguous, M33 unresolved.

Current R2 status2026-09-29:408337 completed52m14s,5/8 fixed L8 proposals
accepted. Numerical checks passed, but white power .336915->.649291 and FP
zero .030946->.035278dex still drift. This is NOT posterior convergence.
408338/408339 illustrated scalar/field readouts completed and were viewed.
No automatic longer chain or N256.408340 joined actual raw training FP/K
observables (8,901 parent rows, same1,414 selected links);408341 PDF rendered
and viewed. Next priority: selected-mark/shared-calibration joint observation
law. A brighter observed subset is not itself proof of a latent-distance bias.
Focused Fable advice was substantively amended/rejected where its proposed
association weighting double-used selection or inferred latent-distance slopes
from apparent-magnitude incidence. See `CF4_R2_SELECTED_FP_ADVICE_20260929.md`
and terminal `CF4_R2_LONG_TRAJECTORY_20260929.md`. R2 incomplete; MW/M31
ambiguous,M33 unresolved. Historical job states below are not current status.

Latest R2 update2026-09-29:408296 completed32 split-HMC feasibility proposals
in58m42s,27 accepted overall and15/16 after discarded warmup; NOT stationary
posterior samples. IC white mean-square .004747->.336915 still drifts.
408305/408331 completed scalar PDF/actual field readouts, visually inspected.
Next bounded bundle:8 fixed-step proposals with8 integrations, same working
target/metric, terminal restart, no new warmup;64 forces,75min application/
90minSlurm,H1002CPU24GiB. Restart/finite-Hamiltonian8 tests passed408336.
No N256, calibration-prior change or heldout scoring. Evidence and scope:
`CF4_R2_LONG_TRAJECTORY_20260929.md`. MW/M31 ambiguous,M33 unresolved;
R2 remains open. Entries below describe their own historical execution states.

Latest R2 outcome2026-09-29:408222 completed32 expanded1414-row joint updates
in42m57s, stopping at its iteration limit, NOT convergence. F146854.44->
144835.84; count log score -141770.42->-139803.43 and FP -49.3064->-31.0739.
408225 independently reproduces the final FP score; training correlation
-.00380, fitted shared zero+.0247702dex (6.19 priorSD), residual mean+.01918dex.
Those plug-in statistics do not demonstrate calibrated distance recovery.
408223 supplies the actual field figure;3cMpc/h development mesh, no named
structure or posterior claim. Corrected terminal count readout408282 passes
4x32/8x32 and scalar-gradient comparisons;408224 had omitted the fine option.
408296 is ONE prior-split HMC feasibility pilot after the completed
readouts:16 discarded warmup+16 fixed-step proposals, same target/priors,
no heldout, no covariance or ESS claim. Mechanics/adapter7 tests passed408273.
Plan/advisory corrections: `CF4_R2_SPLIT_SAMPLING_20260929.md`. R2 remains open;
MW/M31 ambiguous,M33 unresolved. Historical entries below are not live status.

Active R2 bundle2026-09-29: expand the conditional distance working model from
429 to1414 links, retaining exactlyONE FP andONE secure point per source group,
zero anchors and graph-closed training.985 are physically grouped but are NOT
985 multi-FP averages. Their published groupz only defines the FP distance
reference; do not independently score it as extra velocity data. No full
group/shared-source covariance calibration is claimed.408221 reproduces the429
subset and finds a larger conditional FP deficit vs homogeneous benchmark
(63.78log units after same-prior zero profiling); this is not a Bayes factor.
One broadened joint32-step/60min fit is next, with unchanged counts/priors,
fresh optimizer and exact expanded-target startup comparison. R2 incomplete,
heldout untouched,MW/M31 ambiguous,M33 unresolved. Evidence, Fable advice and
driver corrections: `CF4_R2_FIELD_LEVERAGE_20260929.md`.

Latest R2 outcome2026-09-29:408188 completed32 joint steps/59m05s but did NOT
converge. F149990.00->146824.16,count gain2544.82,IC-prior gain~624.08,FP
slightly worse(-18.9454->-19.0348).408191 passes count integration/derivative
checks but radial residuals remain.408216 gives actual429-row training FP
prediction, correlation-.0776 and fitted shared zero+.0203865dex (5.10 prior
SD). This is not positive distance agreement, but noisy nonstationary training
readout alone does not prove model failure. No identical automatic MAP cycle,
posterior promotion or N256. Next address actual CF4 leverage/calibration with
one focused science/cost decision; evidence/plots and workflow correction in
`CF4_R2_FIELD_LEVERAGE_20260929.md`. MW/M31 ambiguous,M33 unresolved; heldout
untouched. Older execution entries below are history, not active job status.

R2 execution-path repair2026-09-29:408185 confirmed that differentiating the
same FP function BEFORE the sole outer JIT restores its exact reference value
and gradient; the nested-JIT derivative was discrepant. Apply flat compilation
to full/conditional targets, keep the likelihood/priors/support unchanged, and
compare actual-size direct/derivative primal components at startup using the
same cached field. Existing restart/adjoint/accepted-step checks remain. Small
target regression is strengthened. This is a measured workaround, not proof of
a particular upstream compiler pass defect. Resume the already-assessed bounded
joint follow-up only; no posterior, heldout or R2 completion claim follows.

Current R2 restart investigation2026-09-29:408163 stopped before any joint
update because its FP value differed from the accepted endpoint by0.13895;
count and prior agreed.408178 verified identical IC/nuisances and only rounding
differences in saved fields.408180 reproduced an isolated compiled FP
value/gradient mismatch, while a flattened full-target derivative reproduced
the accepted score.408184 compact-link values and derivative primals agree.
The exact execution cause is still under investigation; do not widen tolerances,
blame PM float32 (it is float64), consume heldout or claim the fit resumed.
These checks use saved states, not new simulations. Details and job provenance:
`CF4_R2_NUISANCE_BLOCK_ADVICE_20260928.md`. R2 remains incomplete.

Latest R2 follow-up2026-09-29:408154 completed19m13s,5 fixed-field nuisance
updates before application-budget stop;NOT convergence. F151825.92→149990.00,
nuisance max gradient42.68→5.60,IC L2 norm490→684. FP zero grows to5.30 prior
SD and radial residuals remain. Nuisance-only derivatives still cost75–76s,
so the proposed cheap Hessian sweep is not justified. One jointly updated
follow-up will use an SPD optimizer metric from the SAVED conditional secants,
not a Hessian/covariance:32 updates/60min application,75min Slurm,H2002CPU20GiB
(larger observed Slurm RSS~16GiB plus margin),with existing readouts. No
automatic repeated cycles,prior/weight changes,heldout consumption or posterior
promotion. Decision/evidence: `CF4_R2_NUISANCE_BLOCK_ADVICE_20260928.md`.

Latest R2 result2026-09-28:408119 completed64 updates/1h51m08s on iteration
limit,NOT convergence. F159462.61→151825.92,FP−43.95→−25.56;max gradient42.68
still on a luminosity bias. Shared FP zero moved1.67→4.09 prior SD, so the
FP gain is not field-only evidence.408150 passed terminal integration and
scalar-gradient comparisons, but radial shape residuals remain.408121 found
strong step dependence of IC Hessian actions (14% fundamental,102% random),
despite an exact rate-axis control. No Hessian covariance/Laplace promotion.
One bounded fixed-field ten-nuisance optimization is next, following Fable
CONDITIONAL PROCEED with concrete driver amendments in
`CF4_R2_NUISANCE_BLOCK_ADVICE_20260928.md`:no21-evaluation nuisance Hessian,
no conditional-residual=>model-failure claim,mandatory full gradient afterward,
unchanged priors and no automatic alternating cycles. R2 remains incomplete;
heldout untouched,N256 not launched,MW/M31 ambiguous,M33 unresolved.

Active R2 follow-up2026-09-28:408119/source474939c is the single bounded
extension after usable Fable science/cost advice (driver assessment in
`CF4_R2_FIRST_FIELD_ADVICE_20260928.md`). Same partial joint target/data,
v2 accepted state, one exact prior-preserving common-rate warm start, then
up to64 joint updates/120min application,H2002CPU12GiB/130min Slurm.408120
is afterany field readout;408121 is afterok three-direction finite-gradient
curvature feasibility,25min application/30min Slurm. Requested GPU time caps
sum165min including5min readout; no automatic further MAP extension.
This is not posterior sampling, a Hessian certification, N256 or R2 closure.
Heldout data remain untouched. MW/M31 roles ambiguous, M33 unresolved; their
later observables/candidates must belong to the same NEW state, not truth IDs.

408084 completed57m06s,32 updates on iteration limit: objective175535.45→
159462.61, FP−50.94→−43.95, gradient_inf129.91→73.96 with a late rebound
from~31. Not stationary.408093 reproduced the count score and passed
4x32/8x32 comparison (delta.00210,L1=1.437e-6,max occupied log delta.000406)
and scalar derivative(1.56e-6). Its radial-count figure shows shape mismatch,
not only normalization. Expected total50818.60 versus47121 accounts exactly
for the largest gradient in the common rate coordinate. The next scalar
warm start does NOT change the field, likelihood weights or Gaussian prior;
the full target must reproduce its analytic score change before proceeding.
Physical velocity dispersion was saved separately from inference uncertainty.

Latest R2 execution2026-09-28:408032/sourcec8d6af6 completed35m59s,17 accepted
updates, on its application time budget, NOT convergence. Objective
231875.999→175535.453, maximum gradient756.69→129.91; count score improves
−231771.137→−174395.939 while FP score.0859→−50.9406 is worse than its
near-uniform optimizer start. Readout408033 supplies the actual final field
and comparison image. Visible bands/large flows are not validated named
structures. Both counts and field changed; rate normalization increased by
exp(2*.459742)=2.508, and the predicted training count is still40625 vs47121.
The fit used H2002CPU12GiB/45min,20 iterations/35min application. It uses
shell-CDF4x32 counts plus429 strict singleton FP marks on the frozen v6
training graph; no heldout. IC optimizer initialization is.01 times the
predeclared seed draw, NOT a prior realization; cosmological prior/power and
parameter domain unchanged. Do not restart the stagnated407805/407886 states.
The initial same-state adjoint passed (4.39e-5 normalized error); combined
derivative temporary device memory16.82GiB, Python host peak6.71GiB.408078
passed terminal4x32/8x32 comparison (score delta.0008623, exposure L1=1.407e-6)
and scalar derivative (1.93e-6), without new PM evolution or heldout scoring.
Same-target continuation408084/sourceece86ee uses32 updates/70min application,
H2002CPU12GiB/80min Slurm, accepted v1 coordinates/fresh optimizer history,
initial step norm16.408085 is afterany readout. This is NOT N256 or posterior
promotion. Keep physical particle velocity dispersion
separate from posterior uncertainty and phenomenological tracer LOS width.

Preceding numerical repair:407901 identified finite-GH radial-cut switches.
Gaussian-probability integration on physical-distance strata preserves5–180,
negative LOS, periodic images and unnormalized selection, with explicit8sigma
tails. Exact LF-boundary reuse reproduces previous scores.407975 established
4x32/8x32 score delta.00154245, L1=1.494e-6, max occupied log delta=.001073,
but its backward graph requested96.92GiB and OOMed. Streaming408010 then
passed the unchanged value checks and scalar derivative (2.83e-6 relative),
with14.01GiB temporary device estimate.408011 passed4 optimizer tests and the
nonempty coupled count/FP/heldout wiring test. Failures remain recorded, not
relabeled as passes. R2 is NOT complete: calibration, source covariance,
posterior uncertainty, untouched heldout prediction and N256 remain open.
MW/M31 ambiguous, M33 unresolved; future observables constrain the same NEW
state. Details and advisory-review corrections:
`CF4_R2_FITTED_TANGENT_20260928.md`.

Current R2 checkpoint2026-09-28:407805 stopped after4 accepted steps/44
evaluations on insufficient line-search decrease (normal process exit, NOT
convergence). Readout407806 saved the unconverged present field. Fitted-state
derivative separation407840/407881 found strong scale dependence: reverse
−403325.84 agrees with actual objective FD to.029% at1e-8 and.0018% at1e-10,
but not at1e-6 or larger. Thus no PM-adjoint patch is established; the
12-halving search did not reach a measured descending scale. Near-empty
native-cell signed velocity contributions do not dominate that tested
direction. Fable advice was assessed and several incorrect inferences were
rejected explicitly in `CF4_R2_FITTED_TANGENT_20260928.md`.
Same-target retry407886/source52bc3a3 starts with a measured1e-7 coordinate
step norm and adaptive score-only backtracking,20 iterations/25min application,
30min Slurm,H1002CPU12GiB. Three optimizer regressions precede it; readout407887
is afterany. No physical smoothing/floor/prior change, heldout evaluation,
posterior uncertainty or N256/R2 completion. MW/M31 remain ambiguous and
M33 unresolved; same-NEW-state LG identification/constraints belong in R3.
Do not extend negligible descent blindly or call process completion science
completion. The older execution entries below are history.

R2 v6 continuation2026-09-28: wiring-only407731 passed17 tests (9m28s,
7.59GiB RSS), but driver source review found the new linked target omitted
count LOS width and widened the frozen radial180 cut to192. Commit4077f07
fixes both, adds a nonempty-count/radial-edge regression, shares one FP zero,
uses memory-bounded GH15 accumulation, and permits fitting without any
heldout-data arguments. Corrected tests407790 precede typed-H100 partial MAP
407793 via afterok. The N128/384 fit uses only47,121 training counts and429
strict ungrouped linked FP marks, refreshed shifted-source support at EVERY
optimizer evaluation, including trials. Grouped985 links are not treated as
singletons. Source645072c;24GiB/2CPU/55min,45min application bound. No
heldout scoring, Laplace uncertainty, calibration, N256 or R2 completion is
established by submission. Details and next in-R2 decisions:
`CF4_R2_V6_PARTIAL_MAP_20260928.md`. MW/M31 roles remain ambiguous, M33
unresolved; actual LG observables must later constrain the SAME new field.

Follow-up:407790 passed all17 corrections/regressions (9m25s,3.21GiB).
407793 stopped after19m56s on a nonfinite optimizer trial, not OOM; six
accepted states exist but no final/converged map. The combined error did not
identify value versus derivative failure, so its physical/numerical cause
is not yet established. Count and FP scores at the last accepted point
worsened despite a lower total negative log target; do not call prior-driven
descent an improved reconstruction. Readout407795 reported this failure.
Source94a18ef adds bounded trial steps (not parameter-domain restrictions),
rejects genuine zero-support trials without a floor, and preserves exact
state/component evidence for undefined targets or derivative failures.
Same-target407805 restarts the accepted coordinates with fresh L-BFGS history,
typedH100/2CPU/12GiB/55min; ordinary45min-budget termination now saves an
explicitly unconverged field. Heldout data remain unused. See the same R2
v6 partial-MAP record for limits; this is not N256/R2 completion.

R2 LF-shape holdout2026-09-27: typed-H100406570 completed the first actual
2M++ conditional-K check on49,489 eligible source-window rows, fitting seven
predeclared sky octants and holding out8,368 rows in octant3. The imported
low-z bright LF `alpha=-.94,Mstar=-23.28` scores only2.60 log units below
the paper-default `-.73,-23.17` on that one correlated heldout region;
neither is decisively certified, and the training fit is not an independent
prior for the same count data. The source-selected count operator now carries
consistent variable LF shape into its intrinsic fractions and observed-bin
transfer; an unbounded-edge reverse-gradient NaN exposed by406571 was fixed,
then three GPU regressions including alpha finite difference passed in
406573 and the imported near-alpha=-1 shape passed406574. A separate
intrinsic-rate/five-bias mass response and sparse Poisson factor passed the
five-test A100406576 regression. This is observation-model evidence/numerical wiring, not rate,
bias, FoG, CF4 group-selection or sampler calibration. Do not multiply the
diagnostic apparent-class conditional-K score by the six-bin count score:
it would reuse absolute-bin frequencies. R2/N256 posterior remains NO-GO.
Prior joint count pilots consumed all57,238 eligible galaxies; their 2M++
heldout prediction is missing despite frozen split flags. The next field
target must train on a declared subset, scale its thinning intensity, and
close CF4 group/point links across the heldout split to prevent leakage.
No email. Details: `CF4_R2_LF_SHAPE_HELDOUT_20260927.md`.

R2 selection-coordinate control2026-09-27: the observed six K-population
labels use redshift-derived absolute magnitude, while the current count
operator applies six archived radial/angular exposures after redshift-space
deposition. The original fixed-radius transfer406563 omitted the source's
redshift-dependent corrected-K shift. Source-corrected406565 passed4/4 tests
(`CF4_R2_SELECTION_COORDINATE_20260927.md`). Zero displacement reproduces
the old radial fraction; a fixed +3 cMpc/h shift at true30 cMpc/h moves
10.34% of selected LF measure between central absolute-K bins (11.34% of
selected in-range intrinsic sources). The8.79% from faint-side intrinsic
`M_K>-21` is inside the imported LF's fitted `[-25,-17]` magnitude interval
at this radius, but remains model dependent. The bright-side `M_K<-25` tail
is extrapolated and the source notes LF shape systematics near `-21`.
The six-bin LF selection
vector differs from post-RSD exposure by8.64% in L1 for this fixed shift,
of which the corrected-K change itself contributes0.112% L1. This remains
**model sensitivity only**, not a calibrated
actual-data velocity/bias correction. Intrinsic LF/rates, tracer bias, survey
angular/radial selection, source overlap and CF4 grouped marks still require
joint calibration. Do not plug old six published `nbar` values into a new
true-bin process or restart long HMC before its target is fixed. No email was
sent and no R2/N256 posterior was promoted.

R2 source-marked count entry2026-09-27: a new differentiable source-side
five-intrinsic-to-six-observed K operator moves corrected apparent/absolute
marks with each coherent+stochastic LOS displacement, then deposits selected
mass; it does not multiply the old exposure after RSD. A100406567 passed two
reference/mass/velocity-gradient tests. First H100406568 failed before science
on a missing `healpy` import in the GPU environment; two-environment retry
H100406569 COMPLETED/exit0. On one saved **unconditional** N128 PM state,
all45,776 observed population-cells have positive model intensity, compiled
forward time42.13s, host peak2.71GiB. The unit intrinsic rate, fixed b=1,
Schechter within-bin shape and eight-subcell angular map are development
assumptions, not a fit. Source group/point inclusion and shared covariance
remain uncalibrated, and the old nonstationary identity-mass HMC line stays
closed. No heldout predictive pass, R2/N256 posterior, LG role or email.
Details: `CF4_R2_SELECTION_COORDINATE_20260927.md`.

The user directs omitting new TNG data if they are not indispensable. They
are not needed to begin the R2 actual CF4+galaxy present-field inference, so
the proposed hydro-to-Dark matching download and its unsubmitted Slurm job
are withdrawn. Existing TNG products remain optional calibration evidence;
their native identities never enter generated-field candidate selection.
See `CF4_LG_ROLE_DETECTION_CONTRACT_20260925.md` for the R3 limit.

The next critical path is the actual CF4 velocity + galaxy-count likelihood,
its selection/bias limits, and a latent-IC forward state that returns z=0
density/velocity samples. The ARES selection map is a development baseline,
not a calibrated survival/bias likelihood (`config/cf4_selection_limitations_decision_v1.json`).
The historical Q1 `cf4_q1_z0_density_posterior_dev.py` product is **not** a
physical R2 posterior: source inspection shows distance-modulus perturbations
followed by weighted TSC deposition, without an inferred IC, gravity or a
joint CF4+galaxy likelihood. Do not interpolate or rename that proxy into the
required R2 delivery. Start with a bounded same-state observation/forward
feasibility calculation; only launch a larger N256 inference after a viable
state-dependent operator, model calibration and cost are demonstrated.

The first no-TNG same-state wiring control is now complete (Syntax Slurm
403551, 13 s; source `d24aee1`). It read one preserved R1 PM z=0 particle
state and used 64 fixed synthetic tracers to evaluate radial-velocity and
six-population count factors from the **same** positions/velocities. Both
factors respond to a velocity perturbation and have finite local derivatives.
Crucially, the perturbed sparse count model gives positive observed count at
one exactly zero-intensity cell; its huge log-score change is a support
failure, not evidence of information recovery. Output:
`/gpfs/kjhan/CF4/z0_density/r2_same_state_mock_v1/job_403551.json`.
This is a mechanics control only: it uses synthetic observations, an arbitrary
particle-as-tracer assignment, a 12 cMpc/h box and no actual CF4/2M++ rows.
It does not close R1 or deliver R2. Before actual-data sampling, replace the
synthetic tracer law with a calibrated, positive-support galaxy-intensity
model, retain selection/bias uncertainty and show a feasible state-dependent
cost. Do not regularize away zero support with an undocumented likelihood
floor or infer galaxy bias from this toy calculation.

Actual-data tracer identifiability follow-up: Slurm404160 used saved N32/384
CF4+2M++ chains and found all six old-model bias coordinates 2.42–4.95 prior
SD above their imported centres, with posterior SD only 0.11–0.22 prior SD.
This is strong *conditional old-model* information and a model-stress warning,
not a high-resolution bias calibration. The unsupported synthetic count score
was corrected to undefined (`null`) in Slurm404169. Evidence, qualified
external advice, and the next no-TNG 384-box action are in
`CF4_R2_TRACER_IDENTIFIABILITY_20260925.md`. No R2 posterior is delivered.
The follow-up Slurm404189 compared actual 2M++ count factorial moments with
six archived 384-box N32 PM fields under the corrected selection window.
Published-bias PM means are below the observed excess in all six populations;
blindly transferring the old fit biases overshoots dramatically (predicted
excess17.8–69.5). This is a coarse, no-RSD diagnostic, not a bias calibration.
Close the N32 moment check; the next critical work is a positive-support
continuous tracer/RSD operator on a same-state 384-box PM forward with N128
feasibility before N256 inference. Details and limits are in the linked R2
tracer decision record. No TNG, new simulation or posterior was launched.

R2 continuous-tracer mechanics/cost follow-up: Slurm404199 passed two focused
tests and checked a six-population RSD/FoG operator on one archived N32/384
PM state under the corrected actual selection; occupied zero-intensity cells
were 0. An explicit 3% unresolved/contaminant component provides positive
support but is **uncalibrated**, not a numerical likelihood floor or accepted
galaxy model. The N128/384 operator-only analytic-field benchmark and corrected
nonuniform gradient ran in Slurm404201 (GPU warm forward 0.044 s, scalar-
parameter gradient 0.049 s, batch MaxRSS 2.62 GB). This is neither an N128
PM state, full IC adjoint, actual-data likelihood nor R2 posterior. Obtain a
genuine N128/384 PM state and count/selection representation and calibrate
the positive-support response before larger inference. Details:
`CF4_R2_CONTINUOUS_TRACER_FEASIBILITY_20260925.md`.

The subsequent native N128 link is now complete at **wiring/support** level,
not calibration or posterior level. Slurm404203 conservatively aggregated the
existing1.5-cMpc/h 2M++ rows and selection to3 cMpc/h:33,376 galaxies,
24,998 train/8,378 held out, zero occupied selection holes. This corrects
the mistaken claim that only N32 selection existed. Slurm404204 produced one
genuine unconditional384-box N128 PM density/velocity state. Slurm404205 found
zero occupied zero-intensity cells with the arbitrary diffuse fraction **set
to zero**: the fixed3% trial is withdrawn, not calibrated. Slurm404208
connected all19,313 actual CF4 radial rows to the same PM velocity state.
These are not CF4-conditioned phases or a joint likelihood. Independent marks
calibrate survival, not an unclustered fraction or N128 bias. A cosmology
mismatch between the PM control and ARES selection also forbids quantitative
likelihood promotion. Details: `CF4_R2_N128_OBSERVATION_LINK_20260925.md`.

The next R2 datum repair froze the CF4/PM basis (`h=.746`, `Om=.31`) and
rebuilt the 2M++ N128 row catalogue at that same basis (Slurm404255/404257),
retaining33,368 count rows and11,502 disjoint survivor marks. Slurm404259
directly recomputed the six-population/shell N128 selection with order-6
quadrature: no observed zero-exposure cells among29,100 occupied keys.
The published cell-density prior predicts28,574 counts under a homogeneous
field and mean survival against33,368 observed; it is not an exact new-model
rate calibration. This removes the known coordinate/cosmology inconsistency
but **does not** promote the PM control to an actual CF4-conditioned posterior
or certify bias, survival-environment, RSD/FoG and field discrepancy. Details:
`CF4_R2_COMMON_COSMOLOGY_CONTRACT_20260925.md`.
Slurm404269 additionally checked the corrected count/selection response on
the same unconditional N128 PM state: all29,100 occupied cells have positive
finite predicted intensity with zero diffuse component. The random-phase
population totals are not a fit or nuisance calibration.

The first **actual-data joint IC-gradient mechanics** control (N32/12 only)
completed as Syntax Slurm405224 and its same-state step-size diagnosis405225.
It uses24,993 2M++ training counts and15,346 CF4 training radial rows from
one evolved PMWD IC, with both holdouts excluded and four shared CF4 bulk/H0
nuisances analytically marginalized. All occupied cells have positive
intensity. A naive finite difference at epsilon.005 failed badly, but shrinking
to2e-5 gave directional derivative -0.94745 versus autodiff -0.94325 (0.42%
relative difference). Intermediate steps oscillate; this is local derivative
evidence, NOT finite-step HMC stability, N128 gradient readiness, a calibrated
tracer model or an R2 posterior. It motivated a minimal uncertain-rate law
and bounded N128 IC-gradient/cost screen before actual sampling.
Details: `CF4_R2_JOINT_IC_GRADIENT_20260926.md`.

The follow-up uncertain-rate N128 screen (Slurm405229/405230) analytically
marginalized six broad, provisional Gamma count rates and differentiated the
actual CF4+2M++ joint objective through all 2,097,152 IC coordinates at
3 cMpc/h. One small-step directional check differed by 0.022%; warm gradient
evaluation took 0.336 s, without establishing sampler stability. The rate
shape, survival, bias and FoG law remain uncalibrated; no posterior or target
resolution map is delivered. See `CF4_R2_RATE_NUISANCE_N128_SCREEN_20260926.md`.

Slurm405237 tested the disjoint tracer's calibration marks and found strong
population/shell-conditional octant nonexchangeability; its Beta prior-draw
holdout integration collapsed to importance ESS2.76/1.64, so that predictive
number is WITHDRAWN. Slurm405241 attributed3120/3183 (98.0%) failed marks
to CF4–2M++ crossmatch exclusion; the residual noncrossmatched sky test was
much weaker, not certified absent. The current scalar survival-thinned count
likelihood is **NO-GO for production R2 inference**. A marked-point-process
overlap-aware datum is a candidate repair, but the grouped/BGc CF4 velocity
construction and shared redshift must be handled before catalogue replacement
or sampling. Details and driver/Fable dispositions:
`CF4_R2_SURVIVAL_CAUSE_20260926.md`.

The follow-up source-bound group audit finds that only15,211 of the17,007
crossmatched 2M++ targets are in the current N128 eligible parent, including
14,878 secure matches. For eligible secure matches, CF4 *group* V3k differs
from the individual 2M++ Vcmb by median52/p90 437/p99 1,467 km/s, and1,635
CF4 groups have multiple eligible secure members. This makes a literal
restoration of all17,007 objects, or an untested product of inclusive binned
counts and the existing grouped BGc factor, scientifically invalid as a
production repair. The old frozen tracer contract already requires a shared
redshift latent upon reintroduction. No count catalogue, posterior or IC was
changed. Details: `CF4_R2_OVERLAP_GROUP_AUDIT_20260926.md`. The next R2
science step is a bounded inclusive-count **partial-likelihood candidate**
and grouped/member mock control, with metadata anomalies separated. Fable5
gave CONDITIONAL PASS for this lean candidate, but the driver does not accept
its unproved no-bias claim or its angle-only-mask conversion of individual
metadata failures (56 of63 residual calibration failures). If the conditional
mark approximation fails, implement the fuller joint point/mark model. Neither
candidate is a production likelihood yet; do not treat the older
factor-ownership guard as an implementation of the required dependence.
One preserved inclusive *diagnostic* count view now exists (Slurm405248,
COMPLETED/exit0, two seconds):57,238 eligible parent rows,34,301 train,
11,435 heldout,11,502 calibration and15,211 crossmatched; all have positive
integrated N128 cell exposure, and the old disjoint sparse count view is
recovered exactly. There are340 noncrossmatched metadata anomalies in the
full parent, of which63 fall in the calibration slice. Positive *voxel*
exposure does not validate these object-level exceptions or the joint
likelihood. Result and limits are in the overlap group audit record. Next
science action is the one group/member-dependent mock/conditional-mark test,
not immediate actual-data posterior sampling.
Slurm405253 then checked the official angular maps at the *individual*
galaxy positions:442 eligible mark/map mismatches (102 crossmatched,340
nonmatched), with only one zero pointwise map value. Thus an angle-only mask
cannot replace the missing row-level selection/discrepancy model, and positive
integrated-cell support does not settle it. No inclusive likelihood was fit.
The one-state grouped/member **structural stress mock** (Slurm405282 failed
at startup on an erroneous positive-density assumption; fixed retry405294
COMPLETED/exit0) used512 source-bound secure native CF4 group geometries and
4,096 synthetic realizations per arm. Candidate score-variance/curvature
ratios were1.032 for independent reference,1.399 for shared member velocity,
1.294 with distance-dependent group selection (also shifted mean score),
and5.681 for the forbidden per-member duplicate group mark. This demonstrates
that the proposed partial product can overstate information or shift its score
under explicit dependence violations; it **does not** measure their actual
CF4/2M++ amplitude. Preserve the inclusive datum as diagnostic only and do
not rescale a real posterior by these mock ratios. A realistic source-aware
joint survey/group-selection calibration, not more abstract score arms or a
long sampler, is next. Full evidence: `CF4_R2_GROUP_MEMBER_SCORE_STRESS_20260926.md`.
Source aggregation check405317 then found exact `Ngal`/listed-member-count
agreement for38,023 CF4 groups, but group V3k differs from the arithmetic
mean of individual CF4 member Vcmb by median206/p90 620 km/s over all groups.
Thus CF4 distance-contributor membership is available, while a simple member-mean replacement
for the published group redshift is not source-justified. The official EDD
column definition attributes group CMB velocity to the Tully (2015) 2MASS
K<11.75 group catalogue, whose velocity-member population differs from the
CF4 distance contributors. The planned joint observation model must represent
that distinction rather than infer the measured offset as pure redshift error.
No posterior or simulation was run in this check.
The source-only CF4-to-2M++ group-catalogue bridge was first submitted as
typed A100 Slurm405399, then cancelled while still pending at the user's
direction. The same one-CPU/3-GiB/10-min script is now typed H200 Slurm405476
(`--partition=h200 --gres=gpu:H200:1`), initially PENDING(Priority).
H200/H100/A100 preflights passed. Its source-comparison status is recorded in the grouped/member stress
document. Do not equate the Lavaux--Hudson 2M++ `GID`, Tully (2015) group
membership and CF4 `1PGC` without a measured bridge. No source-aware survey
mock or joint likelihood is promoted by this submission.

Update2026-09-26:405476 COMPLETED/exit0;2,524 CF4 groups have one mapped local
2M++ `GID`, but their group-velocity absolute offset has median60 km/s;
159 CF4 groups map to multiple local GIDs. The actual archived Tully (2015)
CDS K<11.75 nest/member/combined tables were then acquired and hash-frozen.
Typed-H200 source bridge405490 COMPLETED/exit0. The archived nest velocity
tracks its own member mean at rounding scale (median0.25 km/s over6,201
multi-member nests), but the9,380 unique CF4 `1PGC`→archived `PGC1`
associations differ in group CMB velocity by median161 km/s. The archived
published 2M++ cross-ID equals the local 2M++ `GID` in only2/3,777 rows with
both IDs. Thus the 2015 archive is source-process evidence, not a drop-in
later-CF4 velocity operator or validated 2M++ join. No R2 likelihood,
posterior or IC was promoted. Detailed evidence and the narrow next decision:
`CF4_R2_GROUP_MEMBER_SCORE_STRESS_20260926.md`.

Next conditional-model entry2026-09-26: source-bound Slurm405550 fitted the
CF4 group CMB redshift conditional on the mean of its eligible secure-matched
individual 2M++ redshifts, separated into one/multiple-member strata and
tested on the frozen native CF4 holdout. Although fixed-df4 Student-t mean
log score exceeds a Gaussian baseline, heldout nominal90/95% coverage for
one-member groups is only82.0/86.1%. This simple empirical conditional is
**NO-GO** as a calibrated group-redshift law. A separate normalized Gaussian
conditional group-Vcmb/distance-modulus kernel passed3/3 numerical tests in
Slurm405551, but its externally supplied joint covariance and selection are
uncalibrated. Neither result licenses the old binned-count/BGc product or an
actual R2 posterior. Details and next full point/mark requirement:
`CF4_R2_CONDITIONAL_GROUP_MARK_20260926.md`.

The next source-bound individual-point/CF4-group contract completed in typed-
H200 Slurm405614 (after a builder-only missing-group correction from failed
405612). It preserves57,238 eligible 2M++ rows,15,211 crossmatched points and
15,239 association edges with exact inclusive-count projection; four secure
edges have no canonical CF4 group row. The pointwise official-map selection
has one observed zero-support point (recno67100, heldout; source mark0.5),
while442 rows have catalogue/map mismatch. A normalized cell-modulated point-
process kernel passed2/2 algebra/support tests, but the full actual-data
likelihood is **NO-GO** pending a source-justified point-selection discrepancy
model and CF4 group-distance/selection conditional. Individual locations do
not add field information beyond counts under a piecewise-constant N128 cell
response; they are retained to model shared CF4 group/member redshifts, not
to claim0.3-cMpc/h information. No posterior/IC promotion. Details:
`CF4_R2_POINT_MARK_CONTRACT_20260926.md`.
Locality check405619 found only98/442 map/catalogue differences agree with an
immediate HEALPix neighbour; a one-pixel shift cannot justify a global repair.
The original2M++ source labels the catalogue completeness columns as
redshift-incompleteness quantities, whereas the ARES example loads separate
HEALPix masks. Their exact pointwise equality is not source-established, so
the442 differences must not automatically be treated as442 bad galaxies or
used as a quality-thinning law. The single literal ARES-map zero-support
point remains; no posterior promotion follows. Source evidence and limits are
recorded in the point/mark contract.

Coarsened-observation bundle 2026-09-26: the pinned ARES `2MPP.txt` has 67,224
rows, of which 67,222 bridge uniquely to the VizieR source under fixed
position/redshift/magnitude tolerances (typed-H200 Slurm405649). The map-zero
recno 67100 is present in ARES's *own* input, so a wrong-catalogue explanation
is rejected. The original ARES/BORG2M++ paper's actual observation model is
voxel-integrated selection and counts, not a full individual-point process.
The source-bound N128 count control 405668 (metadata-corrected version 405676)
counts all 57,238 eligible galaxies in 45,776 occupied population-cells, including the
map-zero point's positive-exposure cell; occupied zero-expected cells=0.
Three numerical tests pass and one published-rate homogeneous log PMF is
finite, but is NOT a field fit/evidence/posterior. Crucially a full joint law
also requires within-cell point-location, CF4/2M++ association/selection and
conditional CF4 group-distance/velocity factors. The old inclusive-count ×
BGc product remains NO-GO. The redshift-space field-to-count mapping is still
uncalibrated. Fable5 gave a conditional development pass; the driver rejected
its unsupported group-redshift determinism and selected-only inclusion
calibration claims. Do not launch N256 sampling before the conditional CF4
group-mark and inclusion law is defensible. See
`CF4_R2_COARSENED_MARKED_COUNTS_20260926.md` and
`CF4_R2_COARSENED_FABLE_DISPOSITION_20260926.md`.

The combined follow-up source/conditional bundle (typed-H200 Slurm405895 and
405896, both exit0; superseding the native-only graph in405888/405893)
finds only1,759 mutually one-to-one CF4-to-2M++ group
pairs, versus3,912 eligible 2M++ GIDs and19,313 native CF4 groups. Of those
GIDs1,430 have no secure *any-CF4* link, but they are **not** known CF4
distance non-detections or an inclusion denominator: group definitions,
crossmatch coverage and distance-method selection differ. One fixed-df4
velocity conditional on the mutual-pair subset has heldout aggregate90%
coverage87.57%, within its weak two-SE screen; at Vcmb>=10,000 km/s the
heldout90% coverage falls to80.99% (two-SE width5.45 points), and FP-only
training90% coverage is83.50%. Do not promote this undercovered restricted
law, retune it on the holdout, or call the apparent aggregate pass a complete
CF4 likelihood. The method-specific CF4 distance-mark selection and
field-linked joint observation law remain the next essential science task;
no actual R2 posterior or IC was produced. Q-GOAL: tests the missing R2
observation dependence. Q-LEAN: two bounded source evaluations, no new PM,
sampler or validation ladder. See
`CF4_R2_GROUP_SELECTION_AND_CONDITIONAL_BUNDLE_20260926.md`.

R2 source-distance implementation follow-up: H200 Slurm405997 completed75s
with two focused tests passed. Official SDSS FP v1.1 source entries join to
10,020 eligible native CF4 FP members in5,450 groups. The source supplies
selection-corrected skew-normal eta summaries with a flat eta prior; use
their likelihood ratios, not another BGc/selection correction. The preferred
richness-corrected columns require group-level shared zero-point treatment.
All rows connect numerically to the SAME unconditional N128 velocity state;
damped inversion plus Newton refinement has maximum redshift residual
1.66e-10 km/s. The source-factor amplitude derivative-35.22357127 agrees
with finite difference-35.22356983. No parameters were fit, no new gravity
run was made, and no R2 posterior or IC was produced. The published FP fit
used its whole sample, so even the group-closed native holdout scores are
not independent end-to-end validation. Next: source-group distance/velocity
and shared-zero-point marginalization, conditional on overlapping observed
redshifts, not more empirical residual tuning or cold-root controls.
TF/6dFGSv remain separate source/calibration tasks. Full details and failed
numerical attempts are preserved in `CF4_R2_SOURCE_DISTANCE_BUNDLE_20260926.md`.

R2 group-distance/shared-zero integration now implemented: H200406005
COMPLETED1m49s, four focused/reused tests pass. All10,020 FP rows become
6,821 source Tempel/singleton groups (not the5,450 CF4 groups);3,062 unique
eligible2M++ member redshifts condition2,413 groups, with no ambiguous
cross-group recnos in this subset. Group distance is integrated once across
all its FP members; correlated redshifts appear in both numerator and
conditional denominator; ONE global calibration is integrated after group
products. Joint distance/zero quadrature refinement changes train log ratios
by at most0.0004234 (all-group0.0025020); amplitude derivative agrees with
finite difference. Accept this numerical observation-model implementation,
not an R2 posterior. COM/member/catalogue covariance and the selected-group
r^2 distance prior remain provisional sensitivity assumptions. Counts alone
still lack the within-cell point/redshift law needed for the complete joint
factorization. Next is source-backed group/selection/covariance specification
or source-mock calibration and that conditional connection, not more generic
residual/root checks or another gravity run. Evidence and explicit limits:
`CF4_R2_GROUP_DISTANCE_MARGINAL_BUNDLE_20260926.md`.

R2 source-membership/conditional-position bundle: H200406012 completed2m11s,
four tests pass. Official Tempel tables connect9,945/10,020 FP rows by exact
photometric ID, with no matched group/richness disagreement;75 unmatched IDs
remain explicit. Saved full30,113-member velocity sample for4,422 source
groups. Published dispersion is reproduced (median ratio1.000000294) but
comes from the SAME redshifts, so it is not independent covariance calibration.
Count plus within-cell conditional position reproduces point-process algebra;
actual recno67100 still has zero literal selection support. Preserve all rows,
no floor or posterior promotion. Next: source-based unmatched-ID resolution,
single group/member/selection law accounting for selected velocity membership,
and justified point-selection support; not another gravity run. Selected-group
distance prior and COM/model discrepancy remain unresolved. Details and result
paths: `CF4_R2_TEMPEL_SOURCE_BUNDLE_20260926.md`.

R2 source semantics resolved in H200406020 (3s, two tests pass): the75
unmatched selected FP rows are all explicitly source-ungrouped; the full
catalogue's418 absent counterparts reproduces Howlett+2022 sec.2.2 exactly.
No missing grouped member/conflict is found. All10,020 FP rows remain in the
new aligned observation artifact with original eta/field/split arrays and
source identities/redshifts/errors. Close this identity concern, not the
physical group/selection model. Published additive-vs-multiplicative redshift
conventions reduce matched individual median cz discrepancy9.084->1.591 km/s
in a fixed diagnostic; do not alter the FP numerator or interpret the original
offset as independent noise calibration. Map-zero recno67100 has no CF4 edge;
its inclusive count is retained, literal full-point support still unresolved.
Next: joint group-distance/selection law, not another matching/residual sweep
or gravity run. See `CF4_R2_SOURCE_SEMANTICS_BUNDLE_20260926.md`.

R2 same-field radial connection406119 completed1m29s; three tests pass.
The group-distance measure can now use dd*d^2*rho_F^b*S_group from the SAME
state as its velocity predictor, in BOTH conditional numerator/denominator.
All10,020 FP rows/6,821 groups evaluated; refined train log-ratio change
0.000225857 and both scalar velocity/radial-bias derivatives agree with finite
differences. Accept this interface, not calibration: b=1,inclusion=1 and the
old covariance are explicit mechanics controls. No fitting, new gravity or
posterior. Next source-constrained group inclusion/tracer/velocity law must
feed this interface and agree with count/shared-redshift ownership; no more
scalar derivative-control series. See
`CF4_R2_SAME_FIELD_RADIAL_BUNDLE_20260926.md`.

R2 official mock bridge:406142 downloaded only5.3MB of the10.6GB archive,
then failed on missing CSV delimiter; corrected406148 and direct-velocity
readout406149 completed1s each from saved source. First fixed mock has33,881
galaxies,14,775 centrals/19,106 satellites. Host/member LOS scatter397.593km/s
is NOT a real group-COM error or replacement for the provisional covariance.
Mock eta uses individual redshift; both measured/truth eta now convert to
the host numerator with residual preserved to5.55e-17. No Tempel recovered
membership or pre-selection parent is provided, so group inclusion and PM
COM discrepancy remain uncalibrated. Do not download the full archive merely
to repeat this schema. Reusable inputs and next model limits:
`CF4_R2_SDSS_MOCK_BRIDGE_20260926.md`. No new gravity or posterior.

R2 selected-mock measurement406150 COMPLETED3s: source skew-normal nominal
68/90/95% truth containment68.540/90.325/95.257%, standardized residual
SD0.99047. Coordinate conversion preserves CDF to2.22e-16. Keep this individual
measurement shape without error inflation, but retain central/satellite mean
eta biases-0.008168/+0.001701dex (larger opposite shifts at low z). One mock
with shared FP fitting is not independent validation or group calibration.
Close this width check; next work is group dependence/selection and latent
population uncertainty, not more individual-PDF diagnostics or truth-based
offset correction. Record: `CF4_R2_FP_MOCK_EVALUATION_20260926.md`.

R2 latent/shared group-mark interface406155 COMPLETED1m49s, six tests pass.
The conditional distance factor now sums no-observed-central/one-central
assignments and integrates one shared group offset after member products;
the existing single global zero is still integrated only after group products.
All10,020 FP rows/6,821 groups connect to the saved same-state density/velocity.
Coarse/fine log-contrast changes are at most0.00036257; exact Gaussian common-
covariance and zero-effect identity checks pass. Accept numerical wiring ONLY:
role probabilities/offsets/scatter are illustrative, not calibrated or inferred
from mock truth. No nuisance fit, new gravity, actual posterior or LG component
identification. Close synthetic covariance variations. Next scientific work
must specify/calibrate the joint selected-group role/FoG/inclusion/source-fit
law, distinguishing available observation information from missing preselection
parent/recovered-group mocks. More rows of the same selected-mock schema do
not solve that absence. Record: `CF4_R2_LATENT_GROUP_MARKS_20260926.md`.

R2 cross-method bundle406182/406183 completed25s/55s; six focused/reused
tests pass. Exact PGC/explicit T17 links attach114 non-FP measurements in96
source groups (SN Ia88, TF12, SBF12, SN II2); combined DM/DMfp are not reused.
Two membership conflicts remain recorded; one FP row moves to holdout after
new-edge closure. Method-specific moduli have free relative zero points.
One training-only Gaussian moment diagnostic estimates total RELATIVE group
excess0.17851mag, but heldout improvement is only0.33092nat over42 rows;
do not interpret/copy this as pure FP common variance or an independent prior.
Actual raw anchors now enter the SAME group-distance integral on the saved
N128 field, with coarse/fine contrast change<=0.00073727. Conditional offset
values in this connection are not a joint calibration posterior. No new
gravity, observed field or IC. Close this scalar diagnostic; next specify
joint method/source-calibration uncertainty and the selected-group radial/
inclusion/FoG law, using each datum once.96 anchored groups do not calibrate
all6,821 groups or latent central roles. Record and preserved artifacts:
`CF4_R2_CROSS_METHOD_ANCHORS_20260926.md`.

R2 joint calibration bundle2026-09-27:406219 passed three tests and saved
the fixed-state geometry but failed on a driver import-path collision before
sampling. Corrected406244 COMPLETED1m27s/exit0,3 tests pass, six shared
nuisances sampled with4x256 retained draws; max split Rhat1.00304, minimum
ESS719.88, zero divergences. Sampled Q257/513 logtarget correction range
0.0003666nat. Original observations enter once; old fitted offsets/scatter
are not priors. Kappa SD1.864 versus prior2 leaves selected-radial shape
weakly informed, not group inclusion identified. Heldout factor integration
has importance ESS5.53/64 and is NOT accepted as reliable predictive evidence.
Close the fixed-state control; reuse its nuisance interface in joint field
updates, not more frozen-field sweeps or a same-data empirical prior.
Selected-group law, full count/point factorization and source/FoG covariance
remain unresolved. No new gravity, field inference or actual R2 posterior.
Evidence, explicit development prior and MW/M31/M33 same-new-field limits:
`CF4_R2_JOINT_CALIBRATION_20260927.md`.

R2 live-field entry2026-09-27 identifies a native PM registration error:
particle_grid uses PMWD node-origin0 scatter while recent source/group
controls read it with half-cell origin.5 (1.5cMpc/h per axis at N128).
Native scatter/gather test confirms this; prior conditional/quadrature passes
do not certify that spatial registration. Preserve their raw results, but do
not transfer their fitted offsets to field inference. New live source-mark
operator explicitly reads origin0 and reconstructs its density measure AND
velocity-dependent kernel from every evolved field. No observer/particle
shift or gravity-kernel change, no inclusive-count/BGc product or production.
Fable returned no verdict within300s; Astra backup CONDITIONALLY supports
the bounded partial-model coupling, with its mark-only adjoint correction
adopted.406349 failed on a radial-constant synthetic fixture assumption;
corrected same-scope H100406350 COMPLETED8m42s/exit0 with4 tests passed.
Two N128 IC/nuisance chains each ran32 warmup+32 retained transitions,
acceptance.9359/.8996, retained divergences0. Mark-only initial adjoint error
0.00252%; last-state Q257/513 sensitivity-0.00233/+0.00115 train/heldout.
Four predetermined IC+same-current-field states and one starting state are
preserved (~432MB including reusable geometry). This is transition/cost
implementation, not convergence, a calibrated full R2 posterior, target1.5
resolution or LG identification. Close the bounded pilot; next selected-
group/shared-error model and legitimate galaxy-data connection must enter
the LIVE operator, not another frozen-field sweep or blind chain extension.
Record:
`CF4_R2_LIVE_FIELD_BUNDLE_20260927.md`.

Next bounded R2 implementation propagates selected-density slope and excess
FP group-offset uncertainty through the SAME live IC/current state, instead
of simply lengthening406350. One common offset per training group, shared
global calibration, raw non-FP marks at the same distance, and member2M++
redshifts inside the joint conditional kernel; no independent count/BGc
product. New bias/tau priors are explicit development regularization, NOT
externally calibrated selection or source-fit covariance. Heldout new-group
offsets are integrated, not fitted. Plan/code/testing scope and unchanged
MW/M31/M33 same-new-state limits:
`CF4_R2_HIERARCHICAL_FIELD_20260927.md`. H100406356 COMPLETED8m54s/exit0:
six tests pass, each chain32 warmup+32 retained, acceptance.8717/.9463,
retained divergences0. Initial mark-only IC+nuisance derivative discrepancy
0.01066%; fixed endpoint Q257/513 training difference.002458nat; heldout new
offset GH9/17 difference at roundoff. One initial +two predetermined final
IC/current states and geometry use~294MB, process peak7.443GiB. Accept bounded
hierarchical wiring only, NOT calibrated bias/shared scatter or a converged
posterior. Endpoint tau~.003dex/b_eff~.65–.74 are not calibration measurements
or future priors. Member2M++ data change the initial IC mark gradient by3.63%
in this conditional model, not certified information gain or use of the full
galaxy counts. Close short pilots; next substantive joint count/point/mark
factorization must state and test retained versus approximated dependence.
No blind chain extension, R2 delivery, N256 or LG resolution promotion.

The next bounded R2 bundle is a single-state live **coarsened 2M++ count +
conditional CF4 mark** control; see `CF4_R2_COARSENED_LIVE_JOINT_20260927.md`.
It scores all57,238 eligible galaxy voxel counts once under integrated ARES
selection and uses linked member redshifts only inside the existing CF4 group
kernel. Native PM node mass/momentum are conservatively read at count-voxel
centres; CF4 rays remain native-origin0. Within-cell coordinates and the
observed association/selected-group graph have an explicitly assumed
field-independent conditional law. Their actual selection probabilities,
six-population nonlinear bias/FoG and source covariance are uncalibrated;
this is a partial development target, not a full R2 posterior or permission
for N256. One prior-state score/adjoint test only, no chain or new simulation.
Q-GOAL: bring the missing bulk galaxy density information into the same
field. Q-LEAN: reuse operators and source products with one short Slurm job.
MW/M31/M33 remain new-state latent roles with unresolved M33 for R3.

Execution2026-09-27: first H100406376 computed the actual joint score,
reverse gradient and positive occupied support but failed only in a diagnostic
JVP call unsupported by PMWD's custom VJP. Same-data/seed/physical-target
retry H100406377 COMPLETED3m53s/exit0: node-to-voxel conservation test passes;
all45,776 occupied cells have positive intensity with no diffuse floor; count
and mark directional reverse-versus-finite-difference discrepancies are
0.0946% and0.00696% at the smaller fixed step. First compiled value+gradient
took75.39s, peak host5.083GiB. At the random prior state, published-rate
prior means predict~48,611 counts versus57,238 observed, not a fit or bias
calibration. Accept numerical **partial-target wiring only**. The within-cell,
association and selected-group laws and six-population nonlinear bias/RSD/FoG
remain uncalibrated. No R2 posterior/map, N256 run or LG identification is
promoted; next quantify actual count/CF4 field tension and observation-model
sensitivity before sampling escalation. Evidence and limits:
`CF4_R2_COARSENED_LIVE_JOINT_20260927.md`.

Fixed-state R2 factor-stress follow-up2026-09-27: first H100406425 failed
only while serializing the open high-k band as JSON Infinity; same-data
retry406426 COMPLETED3m13s/exit0, peak host6.522GiB. On the archived initial
and two predetermined CF4-mark-only endpoints, the count IC-gradient norm
is~94–223 times the mark norm. Initial low-k cosine is-0.372 but the two
mark-only endpoints are near orthogonal overall; these are NOT equilibrated
samples or physical evidence of survey conflict. Fixed-state count log
scores change by +15.6k...+16.6k/-20.6k...-22.0k nats for all six published
bias exponents x0.8/x1.2, and by -664...-689/+2161...+2244 for FoG x0.5/x2.
This is fixed-state model-response sensitivity, **not** evidence of posterior
field sensitivity, a calibrated parameter range or a Bayes factor. Decision:
do not freeze the transferred bias/FoG settings for production R2 sampling,
do not fit them to these three states, and do not launch N256. Important-finding
Fable5 advice CONDITIONALLY supports a bounded live IC + six bias + shared FoG
nuisance pilot before requiring new external mocks; driver adopts that
development order but rejects arbitrary short-chain SD/cosine convergence
gates and retains the missing group-inclusion/association law as a production
barrier. No current-field posterior or LG role was delivered. Record:
`CF4_R2_LIVE_FACTOR_TENSION_20260927.md`.

Joint live-field nuisance pilot2026-09-27: the first H100406431 failed after
one complete and one partial chain on a nonfinite HMC record; its exact
cause is unconfirmed, and failed output remains preserved. Same-target,
same-seed H100406462 COMPLETED14m27s/exit0 with two64-warmup/64-retained
N128/384 chains. Retained acceptance is0.816/0.859, retained divergences0/0,
positive occupied count support, and64 separate development density/velocity
maps per chain. The retry counted no rejected nonfinite energies, so its
success does not establish that the checker amendment caused recovery.
Initial-to-terminal count log factor rises~76k/~77k as count-unfitted prior
starts adapt; this is not convergence, a calibrated tracer response or
evidence for a current-field posterior. A100406467 assessment completes8s:
logtarget split Rhat1.818, IC mean-square2.630, seven nuisance Rhat1.755–2.437;
both logtarget traces rise by~4.5–4.8k nats between retained halves. The
chain-mean density correlation is only0.139 in the15–180-cMpc/h shell and
0.207 on24-cMpc/h box blocks. These values establish **NO-GO for convergence
or map promotion**; short-chain within-trajectory SD is not posterior
uncertainty. Do not blindly extend, pool maps, use nuisance means as priors,
or launch N256. The next science bundle must confront selected-group/galaxy
response calibration and an equilibrating IC sampler jointly, not another
fixed-state sensitivity ladder. MW/M31/M33 remain same-NEW-state latent roles,
including unresolved M33. Record: `CF4_R2_JOINT_NUISANCE_PILOT_20260927.md`.

The next R2 all-sky method connection is a **TF-only source/field control**:
H100406479 COMPLETED25s/exit0, two focused tests pass. From corrected CF4
group `DMtf/e_DMtf`,8,502 native TF-only groups (9,124 summarized TF
measurements;6,745 train/1,757 heldout) are disjoint from the existing FP
source components and non-FP anchors. One unconditional N128/384 state gives
a finite conditional TF group modulus factor with velocity-amplitude
derivative-519.68006577 versus finite difference-519.68006559. Q257/513
training log-ratio changes only-0.0397nat in this state. This is **not** a
TF bias/group-selection/FoG calibration, posterior, independent mock or
permission for N256. The next live partial-target connection must share the
existing TF relative zero point and not count group redshifts twice; actual
source covariance and selected-group dependence remain R2 barriers.
Record: `CF4_R2_TF_SOURCE_LINK_20260927.md`.

The same-state count+FP+TF control H100406482 COMPLETED3m47s/exit0. It
evaluates57,238 2M++ counts, existing FP groups and6,745 disjoint TF-only
training groups from ONE predeclared evolved N128/384 IC. The TF-specific IC
directional reverse gradient agrees with finite difference to relative
6.41e-8; occupied count support is positive. This is a numerical wiring
pass, not a model fit or posterior. TF group inclusion/source covariance and
count bias/FoG remain uncalibrated, and the previous two chains remain
nonstationary. **NO-GO for R2/N256 promotion.** Next, attack observation
calibration and sampler equilibration together, with source/holdout
separation and no truth-selected IC. Record:
`CF4_R2_LIVE_TF_JOINT_20260927.md`.

TF source-consistency screen406487 COMPLETED/exit0 (same A100 request406486
cancelled pending for resources). Across CF4 group-method overlaps,
TF-minus-FP median is+0.016mag/669 groups; TF-minus-SNIa is-0.1375mag/264
groups. The8,502 TF-only groups have median and90th-percentile TF
*distance-contributor* count one; this alone does NOT establish CF4 `Ngal`
group richness. Corrected H100406489 separately checks the source `Ngal`
for the exact TF-only IDs: median/p90=1/1, and95.01% have `Ngal=1`.
Thus these groups really are mostly singletons, but no velocity width is
calibrated by that fact. Shared CF4 calibration and unknown
cross-method covariance mean these internal differences do not provide an
independent zero/error or selection calibration. No parameter was tuned,
sampler run, or R2 posterior promoted. Record:
`CF4_R2_TF_CROSS_METHOD_20260927.md`.

TF-only/2M++ point-ownership check406492 and matched-redshift extension
406493 both completed/exit0. Of6,745 TF training groups,2,761 have a
secure link to an eligible counted 2M++ point (3,686 secure edges over all
TF-only groups). Secure CF4-group vs matched 2M++ point Vcmb absolute
differences have median9/p90 115/p99 420.45 km/s. Overlap is NOT by itself
proof of double counting: a normalized conditional mark given an observed
point can factor from a voxel count under explicit within-cell/selection
assumptions. The present TF factor conditions on group cz and does not
demonstrate equivalence to that count-conditioned law; inclusion/covariance
remain uncalibrated. Driver cancelled long HMC406491 after2m18s rather than
spend up to3h on the potentially changing partial target. No chain result
or posterior. Next: source-linked matched-group conditional or robust model
sensitivity, then sampler work. Record: `CF4_R2_TF_COUNT_OVERLAP_20260927.md`
and `CF4_R2_ALL_METHOD_SAMPLER_20260927.md`.

Secure singleton matched-point TF bridge406494 completed/exit0:3,108
groups (2,484 train/624 heldout) have exactly one secure 2M++ point edge,
catalogue `Ngal=1`, and unique count-point identity. Matched CF4-group vs
point Vcmb difference median5/p90 85.3km/s. The linked control first
failed406495 in a vector-width validity-mask broadcasting unit test before
gravity, then corrected406496 completed3m52s/exit0 with three tests passed.
At the same predefined IC, all-group linked TF score changes-9.3879nat
and TF IC directional gradient agrees to relative3.19e-8; count/FP scores
are unchanged. This closes numerical wiring, **not** source selection,
shared covariance or sampler equilibration. The next bounded sampler tests
only the securely matched TF training subset alongside all counts and FP;
unmatched TF source groups are not silently assigned count identities.
Record: `CF4_R2_MATCHED_TF_CONDITIONAL_20260927.md`.

The bounded matched-subset HMC development run is typed-H100 Slurm406501
(2 archived count+FP terminal starts,192 warmup+128 retained per chain,
all57,238 eligible counts, FP marks and2,484 securely matched TF training
marks). The unlinked5,394 TF groups are not in this target. Typed-A100
read-only assessment406502 depends `afterok:406501` and has no posterior
promotion or N256 follow-on. Submission alone is not a chain or science pass;
source selection/shared covariance remain R2 barriers. Record:
`CF4_R2_ALL_METHOD_SAMPLER_20260927.md`.

Update2026-09-27:406501 and dependent read-only406502 both COMPLETED/exit0.
The two192-warmup/128-retained matched-TF partial-target chains have zero
retained divergences but strong retained drift: logtarget split Rhat2.064,
IC-white-square2.031, seven tracer coordinates1.381–2.871, approximate
ESS2.12–5.28, terminal24-cMpc/h density correlation0.244. **NO-GO for R2
posterior/map and N256**; no blind extension or pooling. The short HMC
trajectory length (about0.029–0.063 in white coordinates) is a plausible
mixing mechanism, not a proved unique cause. TF count-linked redshift-error
convolution was absent in these frozen chains and is being corrected in a
separate same-state numerical control; source selection/covariance and
sampler geometry remain R2 blockers. No LG member identification is claimed.
The scoped matched-TF redshift-error convolution406514 then passed four
tests and a same-IC gradient check (relative discrepancy9.81e-8), changing
the TF log factor by+1.85127nat at that state. It does not reclassify the
frozen406501 chains or resolve the count/TF selection and shared-covariance
law. Record: `CF4_R2_MATCHED_TF_CONDITIONAL_20260927.md`.
One bounded long-path HMC geometry comparison is now H100406521, with
dependent A100 scalar/endpoint assessment406522. It starts from the two
unranked406501 terminal states under the *corrected* partial target and
tests32–48 rather than4–8 leapfrog steps,96 warmup+64 retained per chain.
This is not exact continuation, posterior production, source calibration or
permission for N256. Stop this identity-mass HMC line if nonstationarity
persists rather than sweeping lengths. Q-GOAL: test the specific sampler
obstacle to an actual CF4+galaxy z=0 field. Q-LEAN: one bounded comparison
and reuse of the same assessment; no new gravity simulation or map stream.
MW/M31/M33 remain unresolved latent NEW-state roles. Record:
`CF4_R2_ALL_METHOD_SAMPLER_20260927.md`.

No-email public-source follow-up2026-09-27: CDS J/ApJ/902/145 individual
Kourkchi+2020 TF observables/distances were hash-frozen without contacting
authors. A one-second typed-H200 source bridge406524 finds exact raw-TF
coverage for1,801/2,484 secure matched training singletons and472/624
heldout. Their current CF4 individual `DMtf` versus 2020 `DMbest` absolute
differences have median0.125/0.135mag by split, so these published products
are not interchangeable as written. This is a viable *investigation path*
for a field-dependent conditional raw-TF likelihood, not a selection
calibration, an additional independent TF term or an R2 posterior. No email
was sent; full identity/count/source limitations are in
`CF4_R2_MATCHED_TF_CONDITIONAL_20260927.md`.

Public-method cross-check2026-09-27: the present count operator multiplies
radial/angular exposure after redshift-space displacement, whereas the recent
2MRS reconstruction of Nusser (2026) applies radial selection at model
real-space distance before deposition to avoid a Kaiser-rocket term. This is
not an established software bug because our six population labels/cuts are
based on observed-redshift distances. It is an unresolved generative-selection
choice that must be specified and tested on matched mocks/heldout before R2
or N256 promotion; see `CF4_R2_ALL_METHOD_SAMPLER_20260927.md`. The active
long-path HMC job remains only a geometry comparison of its frozen partial
target. No email or additional simulation was launched for this cross-check.

Long-path HMC result2026-09-27: H100406521 and dependent A100406522 both
COMPLETED/exit0. Both chains ran96 warmup+64 retained transitions at32–48
leapfrog steps, with acceptance0.869/0.881 and no retained divergences, but
retained logtarget halves still rise by2,963.53/2,183.72 nats. Split R-hat
is2.329 for logtarget,1.749 for IC white variance and1.550–2.346 for seven
count-tracer coordinates; their approximate scalar ESS is only2.25–3.99.
Terminal24-cMpc/h density correlation is0.177. **NO-GO R2/N256; close the
identity-mass HMC path-length line.** This is a different partial target from
v2 due to TF redshift-error convolution, so a numerical v2/v3 performance
ratio is not established. Next scientific work must redesign observation
selection/covariance and sampler conditioning together, not lengthen these
chains or promote their maps. See `CF4_R2_ALL_METHOD_SAMPLER_20260927.md`.

For R3, TNG is not a mandatory gate, but omitting it does not make the
MW/M31/M33 assignment, M33 non-detection or galaxy–DMO discrepancy known.
Keep those as latent/nuisance quantities tied to the **same evolved state**;
use project's own resolution/forward controls and independent observations or
mocks to bound them. No actual-data LG weight or seed promotion before a
defensible normalized role law and shared covariance are present. Q-GOAL:
this returns work to the present-field-first science delivery. Q-LEAN: it
removes a 5.3-GB optional download and finder-conversion detour, with no new
gate machinery or simulation authorized by this note.

## Latest execution update — 2026-09-23

Seed40349 remains a **TRACE-ONLY zoom candidate**, not a promoted parent.
GalaxyFinder resolves an L9 pair, while the frozen-threshold HOP regrouping
merges its two distinct raw peaks; M33 is unresolved.  The zoom work therefore
tests whether one conditional fine realization can preserve the coarse LG
candidate, not whether the parent or MW/M31/M33 system is already validated.

The old halo-member-only trace was replaced by a spatial environment trace.
Grammar Slurm1108300 selected all1886 parent particles within5 cMpc/h of the
z=0 pair midpoint, retained all143 pair members, traced them to a
14.13x11.21x14.09 cMpc/h initial footprint, and built a buffered L9 sparse
mask with a42-cell enclosing cube.  Seed40349's measured L9 transfer was
continued with an LCDM-shaped EH98 high-k tail: overlap shape scatter0.10%
and Nyquist join ratio1.00315 (Slurm1108316).  These modes above the parent
Nyquist have random conditional phases and are not claimed as CF4-recovered
information.

Syntax Slurm398944 generated the bounded L9-L12 IC hierarchy with an extended
48-byte GRAFIC header, explicit omega_b=0 DMO metadata, finest particle
spacing0.09375 cMpc/h, runtime ceiling L19 and fine seed403495108.  Grammar
reader gate1108348 and no-output two-step runtime gate1108357 both completed:
138632600 particles, Morton mismatch0, base FFT and all fine MG solves pass,
maximum reported fine residual7.437e-5<1e-4, no boundary/OOM/fatal marker and
no full snapshot.  The two-step gate reached a=0.02455 and used48.72 GiB peak
RSS under a64 GiB request; it did not yet refine above L12.

The bounded early-nonlinear sequence is now complete. Grammar1108467 evolved
normally to a=0.05 but the fail-closed wrapper rejected the gate because no
level above L12 was populated; classify the result as epoch-too-early, not
numerical instability. The first restart attempt
1108478 exposed a missing formal INIT_PARAMS block before evolution and wrote
no output. Corrected same-rank restart1108481 then completed a=0.05->0.10 in
4m42s: transient maximum L17, fine residual<=9.706e-5, no boundary/OOM/fatal/MG
failure,48.89 GiB peak RSS, and one20.31e9-byte final dump. Decision:
TRACE_ONLY_ZOOM_L19_EARLY_NONLINEAR_PASS. This does not promote the parent,
resolve M33, identify a z=0 MW/M31/M33 system, or validate high-k phases.

Do not call this a validated zoom or production IC.  A long z=0 L9/L12/L19
evolution has not yet produced a result.  The early-nonlinear
runtime/refinement gate has passed, and the reviewed full-z0 resource, output,
restart, and science-gate contract is now in
`CF4_S40349_ZOOM_Z0_PLAN.md`.  The actual uniform-L9 reference reaches z=0 in
about27 minutes on16 MPI ranks; the older, much larger L8/L12 mask zoom took
22h56m on8 MPI ranks.  The current L12 mask occupies21.6% of that older mask's
volume, so one 32-rank,96-GiB,48-hour restart with exactly one new final dump is
the bounded next calculation.  Runtime success still requires subsequent
GalaxyFinder/HOP and MW/M31/M33/environment/contamination evaluation.

Per the user's raw-output retention instruction, the superseded19-GiB a=0.05
dump and its historical symlink were removed after the a=0.10 successor was
validated; they are not recoverable.  Logs, namelists, hashes, and decision
records remain.  Keep the a=0.10 restart source until z=0 validation and keep
the z=0 dump until all halo catalogues and science diagnostics are sealed.

Full-z0 restart grammar Slurm1109268 ran from2026-09-23 16:20 to2026-09-24
06:14 KST on grammar089 with32 MPI ranks x2 threads,96 GiB, and completed in
13h54m03s/exit0.  The pinned Intel MPI2021.17 environment loaded, the wrapper
verified the checkpoint writer's `ncpu=32`, and RAMSES evolved a=0.10->1.0.
It populated L19, completed285 coarse steps, reported maximum fine residual
9.992e-5<1e-4, no boundary/OOM/fatal/MG-nonconvergence marker, and wrote
exactly one22.011e9-byte final dump.  Peak RSS was60.34 GiB. Decision:
`TRACE_ONLY_ZOOM_L19_Z0_FORWARD_PASS` in
`config/cf4_lg_s40349_zoom_l19_z0_decision_v1.json`.

This is numerical completion only.  The parent remains trace-only, M33 remains
unresolved, and the single random conditional high-k realization is not
CF4-recovered information. The user now directs using the newer
`GalaxyFinder/NewGalFinder` instead of HOP. Run the RAMSES NewDD/opFoF path,
then NewGalFinder's DMO density-peak/watershed/boundedness/tidal-radius
substructure analysis on output_00003 and evaluate MW/M31/M33, environment
drift, and contamination. HOP is deferred unless NewGalFinder has an
implementation-level failure. Do not promote the seed or call it a validated
production zoom before that science decision.

After the z=0 result and hashes were committed and pushed, the superseded
19-GiB a=0.10 checkpoint and its z=0-run symlink were removed under the user's
raw-output cleanup instruction; they are not recoverable.  Preserve the final
z=0 output_00003 until both halo-finder products and the science decision are
sealed.

NewGalFinder source commit586a62e was fetched without overwriting the user's
modified local GalaxyFinder tree, checked out at the separate GPFS worktree
`GalaxyFinder_newgal_586a62e`, and built successfully with the documented
opFoF ABI. The fixed binary hash and one-run resource plan are recorded in
`CF4_S40349_NEWGALFINDER_PLAN.md`. Grammar job1113459 completed the required
mass-preserving NewDD/opFoF conversion:512 DM slabs,146681 hosts, and a largest
host of165853 particles. This justifies the fixed 8-GB-per-worker allocator
build and a150-GB NewGalFinder request. The first NewGalFinder launch exposed a
DMO ABI mismatch (72-byte member records versus a168-byte hydro build) plus
two periodic-unwrapping defects; it was cancelled after54 seconds without a
usable product. GalaxyFinder branch
`agent/fix-newgalfinder-periodic-unwrapping` commit43046a3 fixes both and adds
a compile-time 72-byte DMO ABI assertion. Corrected-ABI job1113504 then passed
the early hosts and produced DMO subhalos, but was deliberately cancelled after
showing that every zero-star host still paid for an empty stellar FFT. Follow-up
commit96560d9 sends such hosts directly to the identical adaptive DMO finder
and zero-initializes the per-halo state. No HOP job is submitted.

Final NewGalFinder job1113522 completed all146681 FoF hosts on2026-09-24
21:38:44 KST/exit0. Validation1113532 failed at startup because its batch
PATH lacked `python`; the absolute-interpreter retry1114528 completed on
2026-09-25 01:52:23 KST. Its frozen LG cutflow gives0 accepted pairs:
5 eligible hosts,2 separation passes,1 mass-ratio/midpoint pass,0 isolation
passes. The closest candidate misses the3 cMpc/h isolation threshold at
2.7388 cMpc/h. Decision: `NEWGAL_DIAGNOSTIC_PAIR_NO_GO`; Virgo and Coma are
still located. The current trace-only zoom is not a validated MW/M31/M33
realization. Detailed evidence is in `CF4_S40349_NEWGALFINDER_PLAN.md` and
the preserved validation JSON.

The bounded member audit1114563 completed2026-09-25/exit0. It confirms the
isolation intruder is a distinct6.1498e12-Msun/h FoF host2.73884 cMpc/h
from the candidate midpoint. A third LG-scale host of2.5309e12 Msun/h is
only0.8532 cMpc/h from that midpoint. Both nearest-pair hosts contain
well-resolved bound M33-scale components, but there are several competing
assignments. Their FoF and bound-member particle masses show no coarse-particle
contamination. The failure is therefore a real frozen-cut local configuration
at the catalogue level; retain the strict NO-GO for this one conditional
fine-phase realization. No threshold or seed selection follows from it.
The observed-frame cross-check in `CF4_S40349_NEWGALFINDER_PLAN.md` also
finds a minimum57.57-degree MW-to-M31 sky-direction mismatch, a1.522-Mpc
pair separation versus0.761-Mpc observed M31 distance, and candidate MW
hosts3.77/4.57 cMpc/h from the fixed observer. This is an independent
geometric diagnostic, not a calibrated likelihood. Stop seed40349 work;
connect actual LG observables and latent roles to the same evolved IC state
before another zoom or parent promotion.
The first bounded observation-operator bridge now exposes
`predict_candidate` in `src/cf4_lg_observation_contract.py`: for a caller-
supplied resolved MW/M31/M33 role hypothesis it returns the eight predicted
observables **and** both angular mismatches instead of rejecting an off-sky
candidate. The original strict `predict` behavior is unchanged. This is an
interface test, not a likelihood, role enumerator, M33 certification, or an
IC-posterior update. Next, candidate support must come from the NEW evolved
state without truth IDs, with unresolved M33 retained explicitly; a calibrated
joint sky/kinematic likelihood and multiresolution forward connection are
required before an LG-on/off IC comparison can be claimed.
`src/cf4_lg_generated_roles.py` now supplies the corresponding identity-free
NewGalFinder bound-component adapter. On a predeclared generated-state local
support it enumerates every ordered MW/M31 component pair (including two
components sharing one FoF host), every distinct M33 component, and an
explicit unresolved-M33 branch; it does not select the most massive component
or best observed match. For an assigned triple it transforms the saved SG
position/peculiar velocity to the observation contract's ICRS frame and
reports the eight predicted quantities plus both sky offsets. A cap fails
without truncating support. The pure synthetic tests pass. Q-GOAL: this
addresses the missing role/observation wiring on one generated state.
Q-LEAN: no new RAMSES run or arbitrary candidate scoring. These hypotheses
have **no calibrated role prior, detection law, sky covariance or posterior
weight**, and the adapter is not differentiable IC inference. Its M33 branch
labels indicate FoF sharing, not demonstrated association with M31. Do not
call enumeration or a close observed-space match a validated LG realization.
Syntax CPU-only Slurm403323 then read the preserved NewGalFinder z=0
`GALCATALOG.LIST.00003` (6s/exit0) with the fixed 5 cMpc/h observer aperture,
without observational or mass ranking. It contains521 local FoF hosts and579
bound components. Exhaustive ordered MW/M31/M33-or-unresolved enumeration
would generate193434636 hypotheses, so the adapter correctly stopped before
materializing them. The compact result is
`config/cf4_lg_newgal_role_support_403323.json`. An initial job403318 failed
in3s because its runner supplied the particle-data file rather than the
catalogue-list file; it produced no science result and was corrected without
changing the aperture or state. This cardinality is **not evidence that the
observed LG is absent**. It establishes that a best-match shortlist or uniform
explicit triple list is not an acceptable route to posterior conditioning.
The next inference design must specify a normalized generated-state role and
detection law (including missing M33), then preserve its probabilities through
factorized summation or an importance-corrected proposal. Only after that and
the calibrated sky/kinematic likelihood can the same-state LG-on/off IC test
be performed. No extra zoom run or mass-cut retuning follows from this count.
Q-GOAL: the readout tests whether MW/M31/M33 role ambiguity can be retained
when actual generated components, rather than truth IDs, feed the observation
operator. Q-LEAN: one 29-MB catalogue read and a scalar count; no new physics
run, candidate ranking, filesystem diagnostic or validation framework.
The observation-to-role probability boundary after independent Fable/Astra
advice is recorded in `CF4_LG_ROLE_DETECTION_CONTRACT_20260925.md`. No
uniform-role or constant unresolved-M33 weight, immediate R2+LG-dipole run,
or unconditional O(n^2) claim is accepted without calibration and a joint
probability law. This is a design/status record, not a production likelihood
or approval of a new large calculation.

## Latest planning update — 2026-09-13

User requested an end-to-end feasibility/implementation replan with Fable
consultations. [CF4_END_TO_END_REPLAN_20260913.md](CF4_END_TO_END_REPLAN_20260913.md)
and its [advice dispositions](CF4_END_TO_END_ADVICE_20260913.md) record that
work. Three Fable5 consultations were checked against source and literature;
their unsupported claims were not adopted wholesale.

First-scale specialist352623 completed8m21s:16 valid draws, individual1/16,
ensemble2/8; its physical failure persists. CLOSE this ML repair line. No
actual fine CF4/LG posterior has been delivered and no new calculation was
launched for this replan. Historical queued states below are superseded by
[the completion record](BUNDLE_C_FIRST_SCALE_EXPERT.md).

The new recommendation uses a joint present-state/history posterior with
latent ICs and gravity. It is explicitly forward Bayesian IC inference,
with the z=0 marginal delivered first, not a standalone z=0-then-invert
method. **User2026-09-13 explicitly APPROVED this route and R1 execution.**
The R1–R5 replan now supersedes the historical A–D implementation order and
independent-z0-first restriction below. Actual-data z=0 posterior remains
the first science delivery; no best-seed or arbitrary fine-mode substitution.
R1 starts with the bounded implementation in [CF4_R1_RUN.md](CF4_R1_RUN.md).
Its cumulative numerical budget is4 GPU-hours; no new large production.
Preserve all old results, source and failed-model limitations.

R1 entry implemented and submitted as Slurm354568 on2026-09-13 21:34:05 KST,
source8f2aa0d,1GPU/4CPU/10GiB/1h. Initial PENDING(Resources), no numerical
outcome yet. One small periodic PM/aperture/adjoint/HMC mechanics job, not
actual LG inference, nested-gravity validation, R1 closure or R2 entry.
See CF4_R1_RUN.md for the bounded scope and fixed output/log paths.
User2026-09-14 requests a100_pcie. Added it to the SAME pending354568 job's
eligible partitions without cancel/resubmit or altering its source/resources.
Include a100_pcie in subsequent applicable GPU submissions.

Update2026-09-14:354568 COMPLETED5m52s on syn103/a100_pcie,9 tests pass,
likelihood derivative error<=0.0292%. Short-chain mixing remains unestablished
(max unsplit Rhat1.1504/min ESS6.36); mesh32→64 aperture-mass sensitivity up
to40.85%, versus subsequent timestep sensitivity<=3.35%. No R1 closure.
User directs resolving these important issues. Driver implements ONE bounded
same-data HMC comparison (4 chains, fixed8 versus random16–48 trajectories,
matched expected retained gradient work) and a three-seed same-particle force/
time ladder through128^3. CF4_R1_RUN.md records interpretation/resources.
1GPU/4CPU/16GiB/2h within the remaining R1 cumulative4GPU-hour budget. No
mass/noise rescaling, threshold relaxation, actual LG inference or R2 launch.
Physical MW/M31/M33 identification and verified local/coarse dynamics remain
required; successful aperture numerics do not supply them.
Follow-up source32ae8c5 pushed; Slurm358369 started2026-09-14 11:26:08 KST
on syn103/a100_pcie with the above bounded resources. Numerical outcomes
pending; follow the fixed result/logs in CF4_R1_RUN.md, not historical jobs.

358369 subsequently COMPLETED1h17m08s/exit0 at2026-09-14 12:43:16 KST,
tests8/8. Both48-summary mixing criteria pass: fixed8 max Rhat1.0095/min bulk
ESS820 versus random16–48 1.0129/419; no warmup/retained divergences. Keep
adequately sampled fixed8 for the small target; fine-model mixing unverified.
Force differences decrease, but64→128 still changes probe masses up to10.4%,
so accuracy is not closed. R1 consumed1h23m of4GPUh. User approves the next
priority: same interpolated initial displacement/velocity with32³→64³→128³
particles at fixed force128³, then force256³ and timestep halving. One1GPU/
4CPU/16GiB/1h Slurm job, no added high-k or new sampler. Two pre-crossing
plane-wave analytic controls check spatial evolution against known trajectories.
Exact initial/final finest states and common.1875 readout maps are retained;
no actual LG, full-power fine IC, independent collapsed-solver validation or
R1 closure. Details and limits: CF4_R1_RUN.md, config/cf4_r1_particle_resolution_v3.json.
Source62ccea1 pushed; Slurm359000 submitted2026-09-14 13:17:10 KST,
initial PENDING(Resources), with the above1h bound. Fixed result/log paths in
CF4_R1_RUN.md; no numerical pass or independent-solver calibration yet.

359000 failed at startup after5s: the driver basename shadowed its src library,
causing a circular import before numerical work. Driver renames the executable,
updates the Slurm entry and adds a module-resolution regression; same approved
comparison/resources, no physics-model change or external review. Preserve
failed logs; CF4_R1_RUN.md records recovery and subsequent job state.
Recovery90bae45 pushed; retry359203 ran2026-09-14 14:13:01–14:15:04 KST
on syn103/a100_pcie, COMPLETED2m03s/exit0,7 regressions pass. All15 numerical
cases/two planar controls finish. Probe mass sensitivity particles64→128
still15.81%, force128→2562.902%, timestep halving.1240%. CRITICAL unresolved
control: analytic planar velocity relative RMS worsens13.57%→82.68% with finer
force mesh; possible reference/setup or particle-force error must be separated,
not labelled a unique cause yet. HOLD accuracy promotion/R2 despite small
smoothed-aperture residuals and successful implementation tests. Next priority
is bounded planar-reference/initial-force/growth cause separation, not blindly
finer meshes. R1 used5108/14400 GPU-seconds; no next job launched by this record.
User now requests cause confirmation. CF4_R1_PLANAR_DIAGNOSIS.md freezes ONE
15min Slurm diagnostic: independent growth quadrature, pure-mode LPT/units,
original endpoint reproduction, initial-force/growth traces, half-mesh-cell
lattice shift, transverse/axial alias controls and exact planar-sheet force
with the same time integrator. No installed or production gravity changes,
new posterior or mesh escalation.7 controls/1GPU/4CPU/8GiB within R1 budget;
interpretation must distinguish reference error, time integration and spatial
force/discreteness effects rather than presupposing a PMWD defect.
359206 COMPLETED70s/exit0 at2026-09-14 14:36:09 KST on syn103/a100_pcie.
Independent growth/pure-mode IC checks pass and old endpoints reproduce.
Replacing ONLY the spatial force by exact planar sheets reduces velocity RMS
error to2.88e-14 with the SAME integrator; shift/projection/alias controls also
reduce errors. The benchmark discrepancy is localized to particle-mesh spatial
force, with lattice-phase/CIC sampling-alias effects supported. No universal
83% cosmology-error or unique PMWD-line-bug claim. Accuracy/R2 still held.
Details CF4_R1_PLANAR_DIAGNOSIS.md; important-finding Fable advice returned
and supports localization plus an isolated higher-order deposit/gather trial.
Driver rejects calling this single fixture a worst-case bound; no universal
3D accuracy claim or production fix. Suggested corrective experiment is not
launched in this diagnose-only turn. R1 allocation5178/14400 GPU-seconds.
User approves continuation: CF4_R1_TSC_TRIAL.md specifies an isolated,
consistent TSC deposit/gather with unchanged Fourier force/time integration.
Three focused tests and six frozen plane evolutions, one1GPU/4CPU/10GiB/20min
Slurm allocation within R1. No installed/production kernel changes, mode
filtering, real-data posterior or MW/M31/M33 identification claim. Compare
all alignments and preserve residuals; plane improvement alone cannot close
R1 or promote the actual inference backend.
TSC trial359341 COMPLETED68s/exit0 on syn103/a100_pcie2026-09-14 18:15:01 KST;
3 tests pass, six evolutions complete. Nodal final velocity RMS13.57→2.93%
(force128),82.68→6.31% (force256); half-cell4.40→2.42%,25.64→14.15%.
Leading CIC corner artifact is suppressed, but alignment/finite-amplitude
error remains; PARTIAL correction only, no production/R2 promotion. Next
priority is fixed3D and independent force/evolution evidence, not finer-grid/
sampler escalation. Details CF4_R1_TSC_TRIAL.md. No next job launched;
R1 allocation5246/14400 GPU-seconds, all MW/M31/M33 identification limits remain.
User approves next priority. CF4_R1_3D_COMPARISON.md freezes three archived
3D ICs / four CIC-TSC-time arms plus same-state force, field-band, particle
and aperture comparisons and a short trajectory-gradient test. One1GPU/
4CPU/10GiB/30min Slurm job. This is NOT the independent reference: discovered
old RAMSES binaries/source are not verified current, and exact-state input
handoff remains unchecked; user asked for current executable path. No external
project edits or historical launcher reuse. No R2 promotion on code agreement.

## Scientific destination

Use actual CF4 observations, galaxy-density observations, and explicit local
structure constraints to infer a **present-day density/velocity posterior**;
construct compatible LCDM initial conditions; evolve them forward and verify
the observed local environment and MW–M31–M33 system. Deliver phase-consistent
zoom ICs for studying LG formation and evolution, not a unique historical
reconstruction of every small-scale phase.

- Surroundings: 1–2 cMpc/h numerical reconstruction scale is sufficient.
- LG: target <=0.3 cMpc/h numerical reconstruction scale, with explicit LG
  mass, position, distance and relative-velocity uncertainties.
- Finer IC particle/force resolution is a separate zoom requirement. AMR
  alone does not improve particle mass resolution. Fine modes not fixed by
  observations remain conditional/prior content, not recovered observations.
- Retain Virgo/Coma and Local/Bootes Void environment constraints. Separate
  cluster zooms, full-volume 0.3 phase recovery, and RT/stellar/AGN/dust work
  are not prerequisites for the LG deliverable.
- Push defensible observation/structure information toward high k, especially
  in the LG region. Measure information gain; do not arbitrarily choose a
  tiny low-k domain and call the goal achieved. Conversely, global frontier
  certification must not block a useful, explicitly qualified LG prototype.

## Four outcome-based bundles

| Bundle | Deliverable | Exit decision |
| --- | --- | --- |
| A — usable present-field model | One bounded prior comparison, then an actual CF4 + galaxy-data preliminary z=0 density/velocity product with uncertainty and held-out predictions | Declare model limitations and whether it is usable; no indefinite toy-model repair loop |
| B — LG and dynamics connection | Explicit LG constraints in the z=0 route and a small mock z=0 → IC → forward-z=0 demonstration | Check LG/environment conditions and field/velocity consistency after evolution; test this before expensive high resolution |
| C — multiresolution reconstruction | Surroundings at 1–2 and LG at <=0.3 cMpc/h, with coarse/fine coupling and measured observation/structure information | Show LG-conditioned information gain, resolution/cost feasibility and residual uncertainty; recalibrate scale-dependent priors as needed |
| D — zoom IC and evolution | Phase-consistent nested ICs from buffered Lagrangian particle membership and a forward-evolved LG ensemble | Check LG masses/separations/velocities, environment and contamination, then decide on production |

The scientific order remains observations → z=0 posterior → IC → forward
validation. Bundle B is an early end-to-end mock risk test, not permission to
bypass the present-field stage or restart the historical direct-CF4 route.
Do not double-count the same data by treating a data-derived z=0 posterior as
an independent likelihood. Any approximate inversion/proposal must disclose
its target and correction before calling its output an IC posterior.

## Current state and bounded immediate work

**Latest authority2026-09-12, after347085:** user explicitly grants autonomous
continuation without repeated per-bundle approval. Historical "wait for
approval/no automatic next bundle" statements below preserve their original
scope but no longer block driver-designed in-goal follow-up. Keep bounded
experiments, Fable5 advisory planning, source-pinned Slurm, memory sizing,
commit/push and honest outcome reports. Do not resurrect a failed repair line
without new evidence or quietly switch to direct-CF4 IC generation. The
scientific destination above remains unchanged; no promise of a validated
posterior from position-score success.

**Bundle A is closed as a diagnostic delivery, not scientific certification.
Bundle B is closed as a development/diagnostic delivery. Bundle C entry is
approved and implementation is in progress. D requires user approval.**
Current design: [BUNDLE_C_DESIGN.md](BUNDLE_C_DESIGN.md).

Bundle B: environment334398 and four LG interface tests334401 completed;
bounded dynamics retry334402 passed both fixed development cases in4m58s.
Density RMS residuals0.964/0.972 became0.118/0.118; velocity333/300 became
17.8/18.1 km/s, with conservative readout. These are regularized mock IC
candidates, NOT actual-data IC posterior samples. The actual map's Local
Void and precise cluster localization remain unresolved. The source-backed
LG interface still needs a resolved fine-field operator and covariance/model
discrepancy calibration before actual conditioning. No further N32 sampling
or bridge extensions: next work must bring LG observations into the fine z=0
field while retaining its coarse mass/momentum environment. C now begins with
native observations at1.5 cMpc/h, an LG0.1875 layout and conservative coupling;
these are not yet a high-resolution posterior or an evaluated LG operator.
Native-data334408 and conservative/JAX tests passed. Selected actual inputs
contain no direct CF4 or count rows inside LG R2 cMpc/h (CF4 minimum radius
15.8152). LG observations must supply that information explicitly. Selection
integration334409 completed, but order4 missed the positive angular footprint
of one occupied population/cell. Native row positions have valid source-mask
and LF support. Geometry-only repair334508 and preservation check334512
completed: occupied zero-support keys1→0;70260 unoccupied population/cells
also repaired; all prior positive entries unchanged. The v2 selection is a
development input, NOT precision-certified:120 geometry controls still find
33 tiny-support population/cell entries missed by2048-point cubature, with
maximum tested shell-L1 absolute error9.37e-4. Preserve v1 and this limitation.
No high-resolution posterior inference is running. Next substantive C work
is the physically specified fine-field prior and LG halo/subhalo operator,
then the bounded LG-on/off comparison, not another coarse sampler extension.
See the C run record for current products and limits.

Native resolved-operator jobs334521/334522 now completed. A real TNG FoF
fixture with989815 particles supplies mass/mean-velocity/physical-dispersion
fields at0.1875 and1.5, with conservative restriction and particle/SUBFIND
mass/COM agreement. This is a simulated FoF component, NOT a reconstructed
LG or full matter map. The catalogue readout is valid for that unmodified
particle realization; arbitrary changes to fine cells cannot inherit its
halo catalogue. A full-matter conditional fine prior and actual joint LG
conditioning remain outstanding. No new high-resolution posterior is running.

The next C implementation is a finite whole-patch coarse-summary conditional
prior, plus a total-matter source including diffuse/non-FoF matter. It retains
native field/catalogue consistency and separates physical sigma_v from
posterior uncertainty in mean velocity. This finite support baseline does
NOT yet define a continuous LG posterior or exact full-parent conditioning.
Two-file timing334524 and5 regressions passed; full source job334528 submitted
(2 CPUs,9600 MiB,4h cap, estimated2–3h). It builds native400^3/50^3 moments
and18 train/9 nonoverlap check patches from existing TNG. No new simulation,
Hong retraining, actual CF4/LG weighting or automatic downstream inference.
See BUNDLE_C_RUN.md. Do not broaden prior kernels or narrow the science target
just to disguise insufficient support from a small finite patch collection.

Update2026-09-08:334528 completed03:27:20 KST in1h58m04s. All448 files,
11935938442 massive particles/cells processed; conservation passes, absolute
native cosmological mass error4.8093e-5. The18-component prior fails support
on all9 heldout targets (ESS1.00–2.06). This is not a CF4/LG posterior.
User authorized the next C step: ONE bounded support comparison using dense
native translations and24 exact proper rotations, unchanged32-dimensional
conditioning and prior bandwidth, strict heldout spatial exclusion. Report
raw, distinct-anchor and source-spatial-group concentration separately; do
not count correlated/rotated copies as independent universes. If grouped
support still fails, close finite-bank expansion and move to a continuous
joint matter/halo-model design, not repeated kernel/seed tuning.

That bounded comparison335875 has now completed:23275 native anchors and24
proper rotations (558600 correlated hypotheses). Raw component ESS rises to
231.6–2439.3, but native spatial-group concentration ESS is only1.70–5.35;
4/9 heldout targets still fail ESS>=4/max-group-weight<=0.5. Conditions and
bandwidth were unchanged; original results reproduced and the transformed
actual fine field passed its summary check. Status
NO_GO_FINITE_BANK_SUPPORT_CLOSE_THIS_REPAIR. CLOSE this finite-bank expansion.
The next C design must vary matter and halo state jointly and continuously,
with a defined diffuse component and calibrated physical/model discrepancy;
no further bank densification, kernel widening, N32 extension or fake LG map.
This result does not prove that every finite prior fails, nor that a continuous
model will automatically succeed. No actual CF4/LG fine inference is running.

Continuous-state implementation335878 now completed in24s: a native total-
matter patch with three disjoint member components supports21 continuous
position/member-mass/COM-velocity marks, coupled positive remainder mass and
momentum compensation, and correct second-moment transport. Conservation and
one noiseless same-generator inverse control pass. This is a KINEMATIC
OPERATOR, not a calibrated continuous LCDM prior, resolved transformed halo,
independent validation or actual LG posterior. Close the toy-control step.
The next substantive requirement remains the physical joint distribution of
environment, halo marks/profiles and remainder response with COM/model
discrepancy. The21 local marks cannot by themselves fix support on all32
coarse environmental features. Do not promote the precise numerical inverse
error to astronomical accuracy or restart bank expansion. Actual LG-on/off
information comparison remains undelivered; see BUNDLE_C_RUN.md.

Actual LG MARK conditioning has now run (335879,42s):51971 native two-primary/
third-object triples from5601 distinct observer candidates; separate M33
satellite and independent-primary alternatives. A continuous transformed-mark/
shell-summary distribution and measured stellar-minus-halo COM proxy connect
actual distances/LOS/proper motions, including shared MW nuisances and required
probability Jacobians. This is NOT a CF4-conditioned .1875 matter-field prior
or spatial map. Review335880 exposes separate-primary extrapolation despite
high proposal ESS; that alternative is NO-GO for scientific adoption, not
proof of physical impossibility. Gaussian leakage below the1000-DM particle
mass floor was fixed and existing draws restricted/re-normalized in335881,
preserving original outputs. Corrected satellite-conditional marks are usable
as a development input, with broad prior-dominated masses/environment, not a
calibrated LG reconstruction. Current products: `resolved_support_samples.h5`
and `physical_summary_v2.json` in `bundle_c_v1/lg_population_v1`.

Next substantive requirement is the SPATIAL profile/remainder-field model and
its joint connection to actual CF4/galaxy environment, not more mark-only
fits. Native membership/force resolution, missing mass/LMC/stellar-disk priors,
one-box covariance calibration and the32 environmental-feature support problem
are not solved by the marked Gaussian. The desired spatial LG-on/off map and
information gain remain undelivered. No job from this step remains active.

User authorized the next execution bundle, C-spatial, within the unfinished
master C. Scope/first calculation: BUNDLE_C_SPATIAL_DESIGN.md. Start with
native disjoint member profiles plus remaining total matter for16 training
and16 retained spatial cases; then define the continuous spatial distribution
and joint CF4/LG target. Do not reinterpret this as entry to D or completion
of the .1875 spatial deliverable. Source calibration alone does not close the
new execution bundle.

C-spatial first source calculation335916 completed4m08s:93 native member
profiles from16638729 massive rows,32 full member/remainder decompositions
at0.1875 with1.5 restriction and native mass/COM checks passing. These are
spatial CALIBRATION INPUTS, not a fitted conditional field or bundle closure.
Current record BUNDLE_C_SPATIAL_RUN.md. Whole source patches can overlap even
when halo IDs are disjoint; spatial fitting must prevent heldout-voxel leakage.
Next remains the continuous spatial distribution and actual observation-space
connection, not IC generation or another mark-only fit. No current job.

C-spatial conditional remainder candidate335968 completed2m46s, two focused
tests and all12 realizable/conservative draws passed numerical checks. A
geometry-only split provided13 training cubes and three disjoint retained
cubes without training/test voxel leakage. However all12 draws fail the
predeclared small-scale power criterion, both with and without the fixed
native member halos. Status NO_GO_CONDITIONAL_REMAINDER_MORPHOLOGY. This
stationary five-channel Gaussian-copula candidate is CLOSED, not adopted
or patched by amplitude/seed tuning. Native coarse7 moments and halos were
oracle conditions; no actual CF4/LG spatial inference was run. The fitted
profile regression is separate and not a joint halo/field law. Conserving
mass/momentum/second moments does not certify cosmological spatial structure.
See BUNDLE_C_SPATIAL_RUN.md for quantitative results. No active or downstream
job remains. C-spatial is not complete: the next design must supply missing
nonlinear environment/halo–matter spatial dependence before observation
conditioning. Do not close C or enter D on these diagnostic generated fields.

Current execution record: [BUNDLE_C_RUN.md](BUNDLE_C_RUN.md).

User-approved cause separation336263 completed42s; no new fit/draws. Native
density encode/decode is exact to2.6e-16 L1, and quantile+conservation roundtrip
power changes<=0.0944%, unlike the large stochastic generation failure. This
localizes the problem to the generative path/conditioning interaction without
uniquely identifying random phases as cause. Separate confirmed defect: scalar
fine sigma loses directional variance, changing mean velocity by3.4–16.8 km/s
per-axis RMS even for native input. Retain three fine variances in the next
representation. Details BUNDLE_C_SPATIAL_DIAGNOSIS.md and its comparison PNG.
Proposed next implementation BUNDLE_C_CONDITIONAL_FLOW_DESIGN.md: one spatially
conditioned multiscale flow with lossless conservative moment coordinates,
explicitly identifying the missing1.5 environment prior/inference and joint
member-field readout. This is DESIGN, not validated ML or an actual map. Current
approval ends at diagnosis/design; replacement pilot awaits approval. No job.
Subsequent user approval authorizes the implementation/one-GPU4h pilot.
Implementation/source1dc835b submitted as336268: shared multiscale spatial
conditional flow, explicit atom/continuous branches and conservative binary
moment coordinates retaining all directional variances. Frozen configuration,
within-fit spatial split/history limits and gates: BUNDLE_C_FLOW_PILOT_RUN.md.
One job includes native roundtrip, fixed learning and environment/fine mock
generation/evaluation. q_S, actual1.5/CF4/LG posterior and global384 inference
remain outside this pilot; no automatic follow-up science job.
336268 has now completed6000 steps in4m23s: native seven-moment/directional-
variance roundtrip passes, all16 generated cases fail development gates. No
actual-data adoption. User requested continued work2026-09-09. One identified
implementation mismatch is cropped24-parent training versus full64-parent
generation context. BUNDLE_C_FLOW_PILOT_RUN.md now freezes checkpoint-based
paired diagnosis and, ONLY if confirmed, one full-context6000-step correction
with unchanged model/seeds/gates. This is not an automatic longer-training
or prior-family series. All v1 products remain preserved. No actual CF4/LG or
global384 posterior launch; source/representation success is not C completion.
The gated repair337195 stopped after39s before training: crop-context mismatch
is confirmed, but trained-flow inverse/logdet errors exceed numerical limits.
User now authorizes cause separation and necessary correction/reverification:
one frozen same-input default-FP32/strict-FP32/FP64 comparison, not additional
training or relaxed morphology criteria. See BUNDLE_C_FLOW_PILOT_RUN.md.
337268 completed41s: all strict-FP32/FP64 cases pass, all default-FP32 cases
fail. TF32 convolution is supported as the inversion error source, not the
cause of all morphology failures. User now approves strict-FP32 paired gate
and, only on pass, the previously blocked single full-context6000-step fit.
Keep existing numerical/science thresholds and preserve all earlier outputs.
The current repair changes context and precision; no actual CF4/LG inference.
337279 completed9m30s,6000 steps, all numerical gates pass but all16 generated
fields still fail development morphology gates (fine high-band ratio0.235–0.593).
User approved ONE frozen analysis covering before/after quality, train/heldout
fixed likelihood, teacher-parent versus rollout losses, and likelihood/quality
alignment; driver then judges model viability AND actual CF4/LG connection.
Scope BUNDLE_C_FLOW_DECISION.md. No new training or replacement model authorized.
337496 analysis completed2m43s: v2 heldout fine high-band mean0.650 with true
parent versus0.390 in rollout; training0.676 versus0.297. Both one-step and
accumulated deficiencies remain. User accepted withholding current-model
adoption and authorized continued redesign. Candidate proposal:
BUNDLE_C_STRUCTURE_LG_REDESIGN.md, matched-update native-NLL control versus
structural-score repair plus a separately labelled LG member-state readout.
This is DESIGN, not a new fit authorization or a completed q_S/global law.
No actual LG posterior, direct-CF4 IC restart or amplitude repair is allowed.
Subsequent user approval authorizes entry to this next bundle, with a renewed
plan audit FIRST: Fable5 primary, Astra backup only if Fable audit cannot be
completed. Ask Q-GOAL (final-goal contribution) and Q-LEAN (excessive versus
necessary instrumentation/gates), feasibility, and essential/deferred scope.
An adverse scientific verdict is evidence to address, not an invocation failure
to bypass by seeking approval elsewhere. Preserve existing bundle boundaries.
Fable5 plan audit completed normally with CONDITIONAL GO (Q-GOAL conditional,
Q-LEAN broadly proportionate). Driver review found the suggested Gaussian
proxy circular unless its g(F) can infer members from total field alone; native
component inputs are not that operator. See BUNDLE_C_STRUCTURE_LG_PLAN_AUDIT.md.
No Astra fallback/no new fit. Ask before prioritizing the field-only LG proxy
connection ahead of the approved paired learning experiment; do not silently
claim that a Gaussian covariance supplies the missing joint physical q_S.
User now approves prioritizing field-only LG identification/readout before
paired learning. Every future bundle plan AND audit request must explicitly
cover MW/M31/M33 identification, ambiguous/unresolved cases and field-observation
connection. Current bounded implementation plan: BUNDLE_C_LG_IDENTIFICATION.md.
Native labels are for evaluation/calibration after blind candidates are frozen,
not inference inputs. Fable5 plan audit first, Astra fallback on invocation failure.
Fable5 completed the identification plan audit normally: CONDITIONAL GO, with
Q-GOAL direct/Q-LEAN proportionate. All four disclosure conditions (support
source, boundary truth counts, velocity/residual convention, role completeness/
shared peaks/aperture overlap) are implemented in the single CPU job. No extra
auditor/gate, new training or proxy likelihood is authorized by that result.
Identification job337986 completed1m13s, tests2/2, all32 cached native fixtures.
All32 show a shared M31/M33 nearest peak; distinct three-way matching0/32.
Two heldout M33 matches replace M31 on the shared peak, not a third detection.
No boundary truth loss; M33 training calibration unavailable. This closes the
bounded diagnostic, NOT q_S or an observed LG posterior. See the identification
report for counts and limitations. Recommend an explicit unresolved-member/
assignment observation-link design (or justified finer LOCAL information),
with Fable5 plan audit and user bundle approval before implementation. Do not
resume paired learning or new simulations automatically.
User subsequently approved the unresolved-member/observation-link DESIGN bundle.
Current proposal: BUNDLE_C_UNRESOLVED_MEMBER_DESIGN.md, with Fable5 plan audit
requested before any numerical implementation. It must address MW/M31 role
ambiguity as well as M33, normalized same-field component budgets and actual
field information, not repeat native-known-member transport or mark-only fits.
No new training, simulation, actual LG weighting or q_F promotion is implied.
Fable5 returned DESIGN CONDITIONAL GO (normal completion105402ms). Driver
adopts its Q-LEAN recommendation: a14-parameter composite-aperture kinematic
proxy first, not a high-dimensional cell-member learner. Exact finite pair
mixture, joint MW/M31/M33 uncertainty and fixed heldout field cross-scoring are
specified in BUNDLE_C_COMPOSITE_PROXY_PLAN.md. This is the NEXT proposed CPU
implementation, awaiting approval; no job submitted. Cell-member anchor measure
and p(E|F,O)/E-conditioned field prior remain unresolved and explicitly deferred,
not declared solved by conditional covariance or deterministic anchor encoding.
Fable5 follow-up on the concrete14-coefficient composite design returned GO.
Three required disclosures/tests (K spectrum/condition and sample counts,
consistent physical scale determinants, explicit unsupported-context labels)
are incorporated. Design bundle COMPLETE; proposed single2-CPU/1200MiB/10min
implementation now awaits user approval. No active job or new fit from this
design turn. No additional audit micro-stage for these accepted report items.
User now approved the concrete composite-proxy implementation and single Slurm
CPU diagnostic. Sources: src/cf4_lg_composite_proxy.py and
scripts/cf4_bundle_c_composite_proxy.py. Three focused tests run in the same
job before the one14-coefficient fit and frozen13/3 native evaluation. No new
neural training, simulation, actual-data weighting or extra audit is requested.
Composite diagnostic337991 now COMPLETED11s, tests3/3,13 training pairs and all
9 cross-field scores valid. Full/host-only scores prefer the source field in
all3 retained cases; incremental conditional-M33 contrasts versus other-field
means are[+.479,-1.358,-.023] nats and never rank the original field first.
Thus host field dependence is demonstrated diagnostically, but consistent
incremental M33 information is NOT established. No observed posterior/physical
member partition is promoted. See BUNDLE_C_COMPOSITE_PROXY_PLAN.md for full
results/limitations. Close this14-coefficient trial; no automatic refit or
neural model. Next design/approval must address the missing M33 mass/subcell
or finer-local-field connection and retain unresolved q_F/selection caveats.
User approved the next M33 mass/subcell DESIGN. BUNDLE_C_MEMBER_BUDGET_PLAN.md
now specifies one no-fit necessary mass/momentum/diagonal-second-moment budget
test over seven support unions, preventing host/satellite double counting.
Fable5 DESIGN CONDITIONAL GO received normally; four essential conditions are
incorporated (identical cell masks, congruent PSD scaling, unavailable contexts,
common BOX velocity convention). Native supplied mock positions/COMs/masses
are explicit stronger conditioning, not recovered identities or actual masses.
If hosts exclude every alternative, report inconclusive M33 conditional power,
not no M33 information. No posterior/support likelihood or sufficient physical
decomposition is claimed. Design complete; single2-CPU/1200MiB/10min diagnostic
implementation awaits approval. No new numerical job/ML/simulation launched.
User now approves that exact member-budget implementation/one CPU execution.
Sources src/cf4_member_budget.py and scripts/cf4_bundle_c_member_budget.py:
two focused tests then cached fixed16/17/20 mocks and both radii in one job.
No new plan audit, fit, actual-data weighting or automatic downstream task.
Member-budget338040 completed11s, tests2/2,18 comparisons,783 cached cell reads.
Both fixed radii and mass/moment modes: own-field controls3/3 pass, all6
alternatives already fail host-only mass requirements, unavailable0. Decision
INCONCLUSIVE_NO_HOST_COMPATIBLE_ALTERNATIVES, NOT no M33 information. The
necessary union-budget operator is implemented but not a posterior or physical
allocation. Close this one-shot test without further contexts/radii/refits.
Recommend next planning return to the missing present-field prior/inference
with MW/M31 conditions and explicit unresolved M33, rather than more standalone
M33 diagnostic variants. No next bundle is authorized by this result; no job
remains active. See BUNDLE_C_MEMBER_BUDGET_PLAN.md for results and limitations.
User approved the next central-work DESIGN on2026-09-10. Proposed plan:
BUNDLE_C_FIELD_RECOVERY_PLAN.md. It closes standalone member diagnostics and
asks Fable5 whether ONE bounded matched generative-objective repair is the
right immediate bottleneck, explicitly retaining the missing member/selection
and global environment laws. No actual-data fine inference or numerical run
is authorized by this planning approval. Fable5 review completed normally in
137035ms: CONDITIONAL GO, Q-GOAL/Q-LEAN accept one terminal matched experiment.
Three essential corrections are incorporated: training-gradient-based frozen
weight, worst-case full-rollout gradient/memory screen before training, and
one-step/rollout reporting even on failure. Driver does not adopt the audit's
overstrong causal classification or its coarse-posterior/prior-fine-IC fallback
as an automatic route change. See BUNDLE_C_FIELD_RECOVERY_AUDIT.md. DESIGN
complete; one1000-update-per-branch/4h Slurm implementation awaits approval.
No numerical job submitted, no high-resolution observed LG posterior exists.
User now approves the exact field-recovery implementation and one Slurm GPU
experiment. BUNDLE_C_FIELD_RECOVERY_RUN.md records source/configuration:
all-trace energy-score gradient, bounded8-pair screen, equal1000-step branches,
unchanged morphology gates and terminal comparison. No new Fable request or
standalone M33 diagnostic; no actual-data fine posterior or automatic follow-up.
Implementation source43692a1 pushed and submitted as Slurm338194. Initial
state PENDING(Resources), no node/tests/training yet. One job runs the7 focused
regressions, fixed feasibility segment, matched learning and final evaluation.
Execution status belongs in BUNDLE_C_FIELD_RECOVERY_RUN.md, not inferred from
submission success. No new scientific output or next-bundle authorization.
338194 subsequently finished both1000-step branches and saved checkpoints,
but failed at evaluation entry on2026-09-10 02:04:45 KST: FP64 native split
coordinates were passed to FP32 convolution. Seven tests and the operational
screen passed; morphology remains unevaluated. User requests this code fix.
The correction adds the missing network dtype conversion and one regression,
plus evaluation-only resumption using unchanged checkpoints/criteria into a
new output directory. No retraining, new model trial or additional plan audit.
Details BUNDLE_C_FIELD_RECOVERY_RUN.md; evaluation-only execution via Slurm.
Evaluation-only338389 now COMPLETED4m54s,8 tests pass, both saved-model inverse
checks pass, additional training updates0. Intermediate338388 test cleanup
defect is corrected/preserved in the run record. Fixed evaluation source18b84f8.
Both branches pass0/16 original morphology cases. Retained .1875 true-parent
P/native control/repair0.654972/0.664406; rollout0.436332/0.462160. ES modestly
improves both summaries but remains below acceptance: CLOSE_THIS_REPAIR_LINE_
BOTH_FAIL. Close this objective/current-architecture repair without more
steps/seeds/weights. No actual high-resolution LG posterior, no next-bundle
launch or automatic prior-fine-IC fallback. Detailed results and preserved
checkpoints: BUNDLE_C_FIELD_RECOVERY_RUN.md. Evaluation code error is resolved.
User now approves designing the recommended conditional3D U-Net diffusion
replacement, with augmentation and an explicit same-field LG connection plan.
BUNDLE_C_DIFFUSION_PLAN.md is the next proposed implementation, not a launch.
Fable5 plan audit completed normally121178ms: CONDITIONAL GO, Q-GOAL/Q-LEAN
accept ONE bounded field-prior attempt. Five conditions are incorporated:
pre-submission sizing, legal-branch identity before optimization, invalid draws
count as failures, budget-short runs are inconclusive, and no further field
training after failure/inconclusive until native-data member identifiability/
learnability is addressed. Disposition BUNDLE_C_DIFFUSION_AUDIT.md. The member
q_S, selection, global environment and efficient actual-data inference remain
unimplemented; neither morphology nor a normalized diffusion sampler solves
them. No repeated closed member diagnostics or direct-IC fallback. DESIGN
complete; proposed1-GPU/4-CPU/48GiB/24h,30,000-update implementation awaits
user approval at the next bundle boundary. No numerical job submitted.
User now approves this implementation and single Slurm diffusion experiment.
Execution record BUNDLE_C_DIFFUSION_RUN.md; source/model config and static
sizing are implemented.34,095,557 parameters; concrete bounded-cache sizing
reduces host request from provisional48GiB to15GiB (12GiB estimated peak+20%,
rounded), within the approved envelope. One allocation runs8 focused tests,
native identity,32 real largest-scale operational updates, then the same fit
and fixed evaluation. Only the8 original fine draws are comparable here; the
old8 environment draws are explicitly out of scope, not declared passing.
No new audit, q_S learner, observed posterior or automatic follow-up launched.
Implementation341b419 pushed; Slurm338402 submitted2026-09-10 09:03:02 KST.
Initial state PENDING(Priority), no allocation/tests/training yet. One job
will execute the approved test/fit/evaluation sequence when resources arrive.
Actual run state/results belong in BUNDLE_C_DIFFUSION_RUN.md; submission is
not a numerical pass or scientific result. No separate monitoring daemon.
Previous B delivery: [BUNDLE_B_RUN.md](BUNDLE_B_RUN.md).
Update2026-09-10 18:26 KST:338402 COMPLETED/exit0 after9h02m57s;
8 tests passed and30k updates finished, but all8 retained draws are
GENERATION/SUPPORT FAILURES, not valid fields with measured bad morphology.
All12 rollout cases fail first refinement; all native-parent scales fail too.
Continuous loss1.000238 is consistent with a zero predictor, not proof of
its cause. No fine-field or observed-LG promotion. User 'next proceed'
authorizes ONE20min Slurm frozen-checkpoint diagnosis, zero optimizer steps.
Fable5 CONDITIONAL GO; plan and incorporated conditions:
`BUNDLE_C_DIFFUSION_DIAGNOSIS{,_AUDIT}.md`. No verifiable original step0
exists: do not trust seed reconstruction as proof of weight updates.
Check actual normalization, simple denoising baselines, raw/EMA gradients
and unchanged failed reverse chain. No further field training before
member learnability is addressed and a new concrete plan is approved.
Frozen diagnostic implementation3240750 pushed; Slurm338746 submitted
2026-09-10 18:41:37 KST, initial PENDING(Priority), no numerical result yet.
1 GPU/2 CPUs/6GiB host/20min; no automatic additional training or bundle.

338746 completed2026-09-10 18:42:44 KST in36s, zero optimizer steps. On both
native tested scales raw/EMA epsilon RMS~.005, MSE~1 and near-zero correlation;
at t100 the algebraic reference MSE~2.4e-7. Reverse latent RMS rises1->2032 and
fails fraction support. Normalization agrees; continuous gradients are nonzero
and checkpoint/plain backward agrees. Failed denoising is established, but its
unique architectural/optimization cause is not. No new morphology or actual LG
product. Frozen diagnosis CLOSED, not another monitoring/diagnostic series.
User requested next repair PROPOSAL2026-09-11; `BUNDLE_C_REPAIR_PROPOSAL.md`
is design only, with Fable5 plan review. No new implementation/learning job
authorized or submitted. Next bundle requires user approval.
Fable5 repair-plan review returned CONDITIONAL GO111090ms. Current proposed
implementation specification is `BUNDLE_C_REPAIR_DISPOSITION.md`; submitted
plan preserved separately. Next proposed bundle is ONE native field-only
MW/M31/M33/remainder mass-allocation learner,13/3 development fixtures,
<=2000 updates/90min Slurm. No further total-field-prior learning in that job.
Driver corrects the proposed class-balanced CE to per-field/per-role normalized
map-L1 to avoid biasing small-member mass fractions. Pilot is not a calibrated
member posterior or a member-velocity model. Stable v-prediction and separated
category gradients remain a deferred repair hypothesis, not code or a job.
Await user approval; do not revive closed peak/proxy/budget tests or launch a
long density fit automatically. A pass leads to joint-member/denoiser design.
User approved the single native member-mass implementation/pilot. Source and
fixed choices are recorded in `BUNDLE_C_MEMBER_MASS_RUN.md`. One530804-parameter
U-Net,13/3 fixed fixtures, equal-role normalized map-L1, <=2000 updates/70min
within one90min Slurm job,1GPU/2CPU/6GiB. Three focused/reused tests run in the
same allocation; no new audit ladder. Mass-readout only, no member velocities,
observed fine field, diffusion repair/training or automatic next bundle.
Implementationee1c8e7 pushed; Slurm341713 submitted2026-09-11 14:57:49 KST,
initial PENDING(Priority), no allocation or numerical test/fit result yet.
Execution record `BUNDLE_C_MEMBER_MASS_RUN.md`; no separate polling daemon.
341713 completed2026-09-11 15:15:31 KST,17m34s,3 tests passed,2000 updates.
NO_GO_MEMBER_MASS_READOUT: training and all development criteria fail, with
large member mass excess and near-zero native overlap. No accepted mass
readout, member velocities or observed LG; no automatic follow-up. User now
requests independent driver/Fable plans and comparison, DESIGN ONLY. Driver
draft frozen before fresh Fable call in `MEMBER_REPAIR_DRIVER_INDEPENDENT.md`;
Fable receives current failed source/results, not the driver's new proposal.
Independent Fable proposal completed normally123782ms. Original driver/Fable
drafts preserved in `MEMBER_REPAIR_{DRIVER,FABLE}_INDEPENDENT.md`; comparison
and driver recommendation `MEMBER_REPAIR_COMPARISON.md`. Both favor correcting
training conditioning. Driver rejects Fable's unsupported permanent M33 waiver,
exact-baseline-with-epsilon claim and universal learnability failure inference;
also defers the driver's own unnecessary hierarchical architecture change.
Proposed smaller combination: unchanged backbone/four-way output, TRAINING
mass-fraction initialization, separated mass/shape objective, one-field learning
segment then conditional continuation. This is a comparison recommendation,
not Fable approval of a combined execution plan or user launch authorization.
User now approves that final smaller combination. Frozen concrete plan:
`BUNDLE_C_MEMBER_REPAIR_PLAN.md`; implementation/run disposition:
`BUNDLE_C_MEMBER_REPAIR_RUN.md`. Fable5 returned CONDITIONAL GO normally126851ms
(Q-GOAL/Q-LEAN pass), requiring all16 fixtures' four integrated target masses
positive before optimization; incorporated using existing target validation.
Keep flat head/backbone; TRAINING integrated-fraction initialization plus
log-mass-squared/spatial-KL objective. One300-step single-field screen then,
ONLY on pass, the SAME optimizer continues1700 steps on13 fields. Failed
screen is short-budget inconclusive, not fundamental M33 unidentifiability.
1GPU/2CPU/6GiB/90min Slurm, new member_mass_repair_v2 output, preserve v1.
No automatic density fit/IC/next bundle; actual-data member-to-field link
and joint uncertainty remain unimplemented. Launch record belongs in run file.
Implementation8dbaeae pushed; Slurm342013 started2026-09-11 18:26:28 KST on
syn05,1GPU/2CPU/6GiB/90min, initial RUNNING. Numerical/scientific outcomes
not established at submission; single-field continuation is gated inside
that same allocation. No automatic downstream science job.
342013 now closed: COMPLETED3m09s/exit0,5 tests pass,300 single-field updates;
screen failed and no multi-field learning/evaluation ran. Final mass ratios
MW/M31/M33=2.064/1.146/.957; overlap=.915/.838/.789; L1=1.235/.469/.379.
INCONCLUSIVE_SINGLE_FIELD_LEARNING, not fundamental impossibility. User now
explicitly requests5000 cumulative updates on the single field. Resume300
model AND Adam state, add4700, unchanged computation/criteria, fixed endpoint;
no automatic13-field learning afterward. Frozen plan/run:
`BUNDLE_C_MEMBER_5000_{PLAN,RUN}.md`. Fable5 CONDITIONAL GO58623ms, Q-GOAL/
Q-LEAN accepted; hard-abort on incomplete/mismatched resume state incorporated.
One1GPU/2CPU/6GiB/90min Slurm job; source outputs preserved, new directory
member_mass_single5000_v3. Launch/results recorded in its run file.
Implementation787497f pushed; single5000 continuation Slurm342086 started
2026-09-11 23:09:47 KST on syn05 via Slurm, initial RUNNING with no numerical
pass yet. One allocation runs six tests, validates/restores300 then trains
4700 additional updates; report endpoint and wait before another bundle.
342086 completed38m25s/exit0,6 tests pass,5000 cumulative updates. All single-
field criteria pass: MW/M31/M33 L1 .03147/.01031/.01602, mass errors<1.7%,
overlap>=.9886. PASS_SINGLE_FIELD_LEARNING_ONLY, not generalization. User
approves next13-field learning/3-development evaluation. Frozen plan/run:
`BUNDLE_C_MEMBER_MULTI_{PLAN,RUN}.md`. Fable5 CONDITIONAL GO98118ms; primary
training gate excludes pretrained fixture0, existing13-field memory accounting
and48-symmetry tests resolve remaining conditions. One13000-new-update fit
from full5000 model/Adam state,1000 new exposures per training field, unchanged
model/loss/LR, signed augmentations. Same retained development criteria, no
best-epoch choice or actual CF4/LG posterior claim.1GPU/2CPU/6GiB/3h Slurm;
new member_mass_multifield_v4, no automatic downstream science bundle.
Implementation20253e1 pushed; Slurm342129 submitted2026-09-12 00:30:11 KST,
initial PENDING(Resources), no numerical tests/learning yet. Allocation will
execute the full one-fit/evaluation sequence; run record contains actual state.

342129 COMPLETED1h44m53s/exit0,7 tests pass,13000 new updates (1000/field).
NO_GO_MEMBER_MASS_READOUT: training/development criteria fail; formerly fitted
fixture0 also degrades. Same field/symmetry log comparisons nevertheless show
learning improvement, not total learning failure. Source random companion
selection is not encoded in field-only inputs, but neither unique cause nor
duplicate-input contradictions are established. User approves ONE frozen
evaluation-only diagnosis and observation-aware redesign, no retraining:
`BUNDLE_C_MEMBER_DIAGNOSIS_PLAN.md`. Fable5 plan audit first; two saved models,
six fixed fields/all48 orientations and existing small selection tables in
one15min Slurm GPU allocation. Do not revive closed prior/peak/proxy repairs
or claim a high-resolution observed field; next training awaits approval.
Fable5 normal78825ms CONDITIONAL GO; incorporated tolerance/material-spread
criteria and positive finite mass-denominator guard plus absolute errors.
Frozen diagnostic sourcedd006b5 committed/pushed, Slurm343469 submitted,
initial PENDING(Priority). Runtime results belong in BUNDLE_C_MEMBER_DIAGNOSIS_RUN.md.

343469 completed3m24s on syn07/A40,8 tests pass,576 frozen forwards and no
saved-model updates. Both identity results reproduce. Orientation sensitivity
pre-exists the multi-field fit; final multi model remains sensitive in6/6
tested fields.12/16 source cases have companion alternatives, but unique-choice
development16 also fails; no identical-origin/different-label pair found.
Diagnosis closed, no unique failure-cause or accepted member/field claim.
User requests next correction. Proposed `BUNDLE_C_ROLE_LOCATION_PLAN.md`
replaces deterministic mass allocation with an autoregressive distribution
over MW/M31/M33 center cells and structurally cubic-equivariant scalar kernels.
It is LOCATION ONLY, not q_S masses/COM or an observed .1875 field posterior.
Fable returned a read-intention preamble without a verdict; not an audit pass.
Driver backup review `BUNDLE_C_ROLE_LOCATION_REVIEW.md` is conditional feasibility
GO, with explicit probability/selection/teacher-forcing/center-label caveats.
User clarifies audits are advisory: driver verifies and decides with reasons;
independent evidence and separate user bundle approval remain required.
User now explicitly approves implementation and ONE1GPU/2CPU/6GiB/4h pilot,
6500 joint updates (500 per13 fields). Code implements the fresh72,417-parameter
location law, sequential factor gradients, native center-cell references,
nonoracle autoregressive draws and before/after48-view probability tests.
Static syntax checks pass; numerical tests and fit run only in the SAME Slurm
allocation. Job347007 submitted from pushed source50841a2 and RUNNING on
syn05 from2026-09-12 15:52:02 KST. See `BUNDLE_C_ROLE_LOCATION_RUN.md`.
347007 has now COMPLETED2h36m33s/exit0 at18:28:35 KST,6500 updates and final
evaluation complete. Tests4/4 and symmetry before/after pass, but
NO_GO_ROLE_LOCATION_AT_FIXED_BUDGET. Mean train joint NLL .0000387 versus
development63.0029 (density reference10.5120). Autonomous positions fail too.
Close the13-field U-Net pilot; no longer-training continuation or promotion.

User approved the recommended next bundle: population-weighted multiple
native center labels using existing total field/catalogue, ONE small density-
anchored location model and spatially separated comparison. Concrete plan:
`BUNDLE_C_POPULATION_LOCATION_PLAN.md`. Fable5 returned conditional approval
normally138976ms; driver disposition in the corresponding REVIEW document.
The adviser incorrectly added nested time caps; driver rejects that arithmetic,
keeps110min TOTAL including70min-capped learning inside120min Slurm. Empty-
cell/periodic-overlap/early-count conditions are incorporated without extra
audit stages. Model21 ridge-regularized linear coefficients plus one scalar
calibration weight,12 epochs with all alternatives and uniform-observer/
M31/M33 hierarchical weights. Fixed spatial slabs include feature halos;
old three development volumes excluded from new test. No new raw snapshots,
profile extraction, q_F learning, actual-data posterior or IC job. Execution
record: `BUNDLE_C_POPULATION_LOCATION_RUN.md`. Approval covers this single
comparison; next bundle still requires its concrete result and user direction.
Slurm347085 started19:08:47 KST2026-09-12 on syn05/A40, pushed sourceaa024b4,
1GPU/2CPU/10GiB/2h. Four numerical regressions pass. Fixed split counts are
1388 training /87 calibration /127 test observers, with5006/261/449 native
triples and no cross-split native-ID overlap. Population scope passes.
347085 now COMPLETED/exit0 in3m11s at19:11:58 KST;12 epochs/2088 updates,
all tests/symmetry checks and all five fixed location-feasibility criteria
pass. Calibration alpha=1. Mean test joint NLL9.9064 versus density11.2234;
M33 NLL3.4396 versus density3.8976/shared-cell3.7562. Training joint10.6850
versus density12.0591; no comparable extreme train/test score gap here.
Autonomous512 triples complete, with32 distinct draws per observer/law; broad
weighted-target distances remain (e.g. M331.214–3.603 cMpc/h over8 testcases).
.1875 cell size is NOT .1875 position accuracy. Driver accepts and CLOSES the
bounded position comparison, not a physical halo/mass/COM model or q_F.
These are correlated one-box samples, not independent universes or observed
LG fields. Next proposed outcome: connect LG position observations to a
SAME-field likelihood and the remaining field/mass/COM model, working toward
an LG-on/off field response, not another location-score repair loop. No new
bundle is launched; user direction is required. See the run record for full
proper scores, autoregressive caveats and preserved artifacts.

Autonomous follow-up now implements BUNDLE_C_POSITION_LINK_PLAN.md: frozen
location law -> joint LG distance/sky observation density on the SAME field,
with shared distance covariance, within-cell integration and no native
candidate parents supplied to inference. Fable5 PROCEED; driver disposition
uses exact exponential tilting/cell-face intervals instead of GH aliasing,
and finite differences at an interior conservative mixture to avoid cold/
empty boundary nondifferentiability. One GPU/2CPU/6GiB/30min, eight fixed native
test fields/all archived alternatives, actual-data interface scores and one
field-sensitivity projection. No training, field selection, density painting,
posterior or IC. Missing selection, offsets, physical field/mass/COM joint law
remain explicit. Execution/results: BUNDLE_C_POSITION_LINK_RUN.md. Autonomous
approval replaces another wait; closed model repairs remain closed.
Position-link347086 completed14s but an unweighted worst-case quadrature error
estimate was too loose. Correction01971b8 weights each rectangle error by the
same qA*qT used in the integral; old worst-case estimate remains visible,
threshold unchanged, numerical estimates distinguished from exact tail bounds.
347087 COMPLETED15s/exit0, all3 tests and integrals/conservative derivatives
pass. All208 mock +8 actual-data logL values unchanged EXACTLY. Eight mock
source fields rank first both in joint score and incremental M33 score;
development evidence only. CLOSE position diagnostics: usable observation-
likelihood component, NOT field selection, masses/COM or observed posterior.
Next autonomous design targets a physically conservative joint-field prior;
old field diffusion already used dense random native translations (unlike
13-field member fits), so more data alone is not a justified repair.

Autonomous next generator experiment: BUNDLE_C_STABLE_FIELD_PLAN.md, Fable5
GO with driver corrections in REVIEW. Fresh1,490,406-parameter full-resolution
v-parameterized joint moment model, independent continuous/category branches,
explicit observer conditioning and matched archive-E native fields. This
changes the unstable reverse reference and gradient paths, not just length.
80^3 generated buffered field /inner64 at.1875, native1.5 parent, three scales;
all physical seven moments retained. Source-backed v parameterization is a
plausible remedy, not a proven cosmological reconstruction. One24k-step fit,
1GPU/2CPU/6GiB/4h with tests and fixed evaluation in the same job. Sixteen
draws/eight parents plus frozen field-only LG role/observation machinery;
no true fine buffer or native role parents passed to generated-field readout.
Conditional-E/native-coarse development only; no observed posterior, IC or
automatic continuation of this fit after a miss. Run record:
BUNDLE_C_STABLE_FIELD_RUN.md.
Generator source0bfd832 pushed, Slurm347088 started2026-09-12 20:19:41 KST
on syn05/A40,1GPU/2CPU/6GiB/4h. All3 tests pass;64 normalization observers
and192 native roundtrips pass,1388 training observers retained. Training400/
24000 at application131s, early v-loss improves on its zero-v training
reference but NO validation/generated-field result yet. Host3.030GiB,
GPU reserved1.176GiB. Rough initial total runtime80–95min; hard Slurm limit
00:19:41 KST next day. The same allocation automatically proceeds through
checkpoint, heldout denoising,16 generated fields and role readout. Next
driver action: read fixed347088/final artifacts and judge, continuing under
autonomous authority without another routine approval request.

Previous diagnostic delivery: [BUNDLE_A_RUN.md](BUNDLE_A_RUN.md).

Completed A history, not instructions to rerun: the four-fit comparison
failed the frozen quantile-prior adoption rule. That repair series closed.
The original24-nuisance PM-calibrated model then supplied an actual-data
model-stress diagnostic, followed by the approved observation-model correction
and length check below. None scientifically promotes the baseline.

Actual-data preflight333862 passed; fit333872 and aggregation333990 completed.
The actual-data result is NO_GO_SAMPLER_NOT_VALIDATED (max Rhat1.325,
minimum bulk ESS11.1), not a usable posterior. Saved-chain diagnostic334240
completed without refitting. It confirms field/H0 coupling and identifies a
physical-Mpc versus Mpc/h magnitude convention mismatch in the imported
2M++ population/selection setup. See
[ACTUAL_PREVIEW_DIAGNOSIS.md](ACTUAL_PREVIEW_DIAGNOSIS.md).
The user subsequently approved the focused observation-model correction,
field-aware sampler adjustment and ONE corrected actual-data refit.
Active correction plan: `config/cf4_actual_data_corrected_v2.json`, documented
in [ACTUAL_CORRECTION_RUN.md](ACTUAL_CORRECTION_RUN.md). Preserve the old fit.
The disjoint sample is retained; calibrate its survival from a separate20%
parent mark sample, retain the old20% heldout, and use60% for field fitting.
This supersedes the earlier one-fit cap only for this approved refit, not for
further prior-family experiments. No Bundle B execution is active.

Corrected fit334345 completed in22m22s. Max Rhat1.034 passes, but H0 and
its field-dependent conditional mean have bulk ESS97.1/85.1 (<100); retain
NO_GO_SAMPLER_NOT_VALIDATED. The user now approved ONE sampling-length check:
`config/cf4_actual_data_longer_v3.json`, four fresh chains with2048 samples
each,512 warmup, unchanged model/data/gates. Reuse frozen actual inputs,
preserve old chains, and do not claim exact continuation or pool the runs.
No automatic further extension or Bundle B launch.

Final length-check job334358 completed2026-09-07 22:24:15 KST in36m33s;
aggregation334359 completed22:24:23. All predefined sampler gates pass:
max Rhat1.03183, min bulk ESS113.59, min tail ESS316.88, divergences0.
Heldout count/velocity moment-residual SDs .99628/1.11173 are diagnostic,
not scientific calibration. Product remains12 cMpc/h and contains no
resolved LG. Stop adding samples. The user accepted preparing the next
bundle: actual environment comparison, explicit LG observational likelihood
contract, and a bounded coarse z=0→IC→forward-z=0 mock bridge. No B jobs yet.

Z4–Z11 were N32, 12 cMpc/h development experiments, not actual-data maps.
Z8's amplitude interpretation was erroneous and has been withdrawn. Z10
fixed-field and Z11 prior-compatible controls recover the injected tracer
curvature; Z9 free-field PM cases do not. This supports a field/observation
model mismatch but neither isolates its sole cause nor proves a quantile
transform will fix it. Z11 completed all four fits; its results are at
`/gpfs/kjhan/CF4/z0_density/z11_prior_control_v1/comparison.json`.

The following is the historical Bundle A scope, now completed at diagnostic
level; it does not authorize another comparison or actual-data fit:

1. Finish ONE training-only non-Gaussian-prior comparison against the saved
   native-PM controls, with at most four new mock fits. Reuse existing
   simulations, likelihood, sampler and reports. Require spatial field
   recovery and held-out predictions, not just a nicer density histogram or
   recovered nuisance coefficient. Exact cases/resources/decision rule:
   `config/cf4_bundle_a_prior_to_data_v1.json`.
2. Driver judges the comparison. No automatic additional prior families,
   nuisance terms or new simulations if it fails. State the model limitation
   and decide its suitability for a preliminary observation-space diagnosis.
3. Within A, implement and run at most one actual-data preliminary four-chain
   fit after its input semantics and chosen model are recorded. Use actual
   CF4 velocity data and prepared 2M++ counts, preserving selection, errors,
   overlap treatment and heldout separation. No pseudo-truth or truth-based
   accuracy/coverage numbers for the real universe. If model suitability
   fails, the product is a model-stress diagnostic, not a validated posterior.
4. Deliver actual-data mean/sample density maps, mean velocity and posterior
   velocity uncertainty, predictive residuals and limitations. Distinguish
   uncertainty in the mean field from physical velocity dispersion/FoG;
   current population FoG parameters are not a resolved sigma_v(x) field.
   Close A with a concrete Bundle B design and request approval.

The real-data preview remains 12 cMpc/h and cannot certify the target LG
scale. Independent galaxy/selection mocks and physically relevant validation
are required before scientific promotion, but not an unbounded prerequisite
to viewing a clearly labelled diagnostic map. Reused development fields and
shared generator/inference code are never independent validation.

Update2026-09-12 21:57 KST:347088 completed24000 updates/final checkpoint,
then FAILED at evaluation entry (duplicate status keyword in driver code).
No generated quality verdict exists. Autonomous technical recovery fixes the
assignment and adds evaluation-only final-EMA loading with identical saved
normalization/config/seeds, zero optimizer steps, original outputs preserved.
New stable_field_eval_v2, one GPU/2CPU/6GiB/30min Slurm allocation runs four
tests and the original fixed endpoint. No new model/advisory audit required
for this same-bundle technical correction. See BUNDLE_C_STABLE_FIELD_RUN.md.

Update2026-09-13:347108 COMPLETED2m23s, tests4/4, final EMA evaluation with
zero optimizer steps. All16 draws valid/conservative, terminal latent RMS
.926–1.077, but individual morphology0/16 and two-draw summary means2/8 pass.
Boundary-gradient ratio fails16/16; power5/16; connected fraction2/16.
NO_GO_E_CONDITIONAL_FIELD_GENERATOR: stable generation is progress, not
scientific adoption. User approves saved-field boundary/metric diagnosis;
no retraining, new samples or threshold change. BUNDLE_C_BOUNDARY_DIAGNOSIS.md.

Boundary diagnosis347159 completed8s with two controls and all eight stored
first draws; original24 axis-ratios reproduce. Raw squared-gradient boundary
energy is dominated by10 edges (median85–87%); native ratios themselves vary
widely. Log-density still shows modest grid-phase roughness: median boundary/
internal .994 native versus1.141 generated. Mixed metric sensitivity and
generated hierarchy artifact; not a pure code bug or significance claim.
Close this diagnosis, retain original NO-GO/power failures, no blind longer
training or current-model actual-LG promotion. BUNDLE_C_BOUNDARY_DIAGNOSIS.md
records evidence and next model-design target; no active follow-up job.

Latest audit policy2026-09-13: external review only for an important discovery,
goal revision or large calculation. Driver handles all remaining evaluations
and routine plans. This supersedes historical every-bundle external reviews;
retain scientific goal/lean reasoning without another gate framework.

User reconfirms continuous autonomous work without routine approval2026-09-13.
Next bounded correction-localization uses saved rollout restrictions at three
scales and sixteen frozen-model native-parent one-step draws; no fit, new
full rollout, threshold change or actual posterior. It separates inherited
parent error from one-step field/velocity-variance partition error before
selecting one correction. BUNDLE_C_SCALE_LINK.md, driver review only,
1GPU/2CPU/4GiB/10min Slurm. No further standalone role-identification checks.

347160 now COMPLETED57s on2026-09-13, tests4/4 and sixteen native-parent
draws/physical variance identities pass. Both one-step error and propagation:
at.1875 bulk RMS ratios teacher1.185/rollout1.567; sigma .949/.799. Already
at.75 bulk budget fraction .235 versus native.094. Grid roughness persists
with true parents. Close diagnosis. User requests correction; driver implements
ONE matched6000-update-per-arm comparison from final EMA, original continuous
model versus fine-lattice residual path + parent-only mass/variance-weighted
v loss. Freeze category in BOTH arms; same noise/data/optimizer/normalization,
unchanged physical decoder/sampler/criteria and generated-field LG readout.
BUNDLE_C_FIELD_LINK_REPAIR.md;1GPU/2CPU/6GiB/2h, routine driver review,
no new posterior or automatic repair series. Do not claim correction efficacy
before final physical fields are compared.

## Working rules

User2026-09-13 preapproves continuous multiple in-goal bundles without routine
approval stops. Next BUNDLE_C_FIRST_SCALE_EXPERT.md: one first-scale-only
6000-update expert, original finer networks frozen, existing sixteen-draw
physical/LG readout comparison in one1GPU/2CPU/6GiB/30min job. Scale competition
is a hypothesis, not diagnosed fact. No output amplitude adjustment, no
actual posterior promotion or blind subsequent specialist/epoch sweep.

347264 completed42m56s, both6000-update arms and7 tests complete. Individual
quality0/16 versus1/16; ensemble0/8 both. Physical partition/grid improvement
small, no accepted field prior or observed posterior. Close this extension.
User approves next bundle: BUNDLE_C_SPLIT_ATTRIBUTION.md, one saved-field
first-split decoder intervention (native/generated codes versus values),
2CPU/3GiB/5min Slurm, no training or new stochastic draws. Oracle hybrids
are diagnostics only, not inference inputs or proof of unique neural cause.
Next model choice requires this concrete evidence, not more blind steps.
352595 now COMPLETED26s:64 rows/32 roundtrips pass, zero learning/draws.
All first-split legal category arrays match native EXACTLY in both arms.
Swapping categories does nothing; native continuous values recover native
physics. Error resides in continuous output values at this split; no claim
of unique training cause or exclusion of sampling-category history/finer
scale effects. Close diagnostic; no category-only repair justified. Next
design targets continuous physical-budget distribution, not amplitude repair.

- Prefer substantive scientific outputs over more generic validation code.
  Reuse tests; add only checks necessary for the changed computation.
- A clean sampler is necessary, not scientific success. Separate posterior
  mean smoothing, individual-draw structure, uncertainty and phase recovery.
- Syntax is a Slurm login server. Numerical jobs use Slurm; no manual syn101
  or login-node calculations. Current GPU partitions: a100_pcie,a40,a100,h100,h200;
  exclude syn06 for these fits. Request estimated peak memory plus ~20%.
- GPFS is ordinary shared storage. Read/write scoped project artifacts;
  do not implement storage/inode/renameat2 probes or process-scan monitoring.
- Use fixed job IDs and final artifacts for bounded checks; no pgrep loops.
  - The driver plans, implements, runs, evaluates and commits. External review
  is reserved for important discoveries, goal revisions or large calculations;
  routine work is driver-reviewed. A Fable-assigned audit is performed by
  Astra; if auditors would be duplicated or immediately consecutive, the driver
  performs that audit instead of stacking another reviewer. Reviews answer
  Q-GOAL and Q-LEAN plus feasibility
  and essential/deferred scope. Do not turn these two questions into a new
  gate framework or per-step audit series. Prior Astra closure-audit waiver
  remains unless changed by the user; this instruction concerns plan audits.
  Explicitly audit the MW/M31/M33 identification plan in every bundle; require
  no oracle component access on new fields and honest unresolved-M33 handling.
- Commit/push coherent changes. Preserve unrelated user work and all failed
  scientific results. Keep run summaries current; do not confuse submission,
  sampler pass, scientific acceptance and final-goal completion.
- At every bundle boundary report the result, goal contribution, unresolved
  risk and next deliverable. When approval is needed say **승인해주세요**.

Update2026-09-27 R2: a source-bound spatial count holdout and graph-closed
CF4/FP mark roles are frozen in `r2_sky_closed_split_v2` (Slurm406580).
Octant5 withholds3,400/57,238 2M++ points; 1,877 CF4 groups are sky
validation and23 are boundary buffer; FP has no sky-validation groups in
this footprint. The count kernel now integrates only its selected sky
window. This fixes direct count/group train-test leakage for a future fit,
not the missing selected-group law, sampler stationarity, independent mock
coverage or R2 posterior. See CF4_R2_SKY_CLOSED_HOLDOUT_20260927.md.

Same-day correction before field fitting: octant5 had no FP validation.
Footprint-only choice406581 selected octant2 without viewing field/mark
scores; anchor-inclusive graph closure406584 froze split v4 with9,696
2M++ heldout points,2,867 native CF4 and764 FP sky-validation groups.
Intermediate v3 omitted114 cross-method anchor links; its first same-field
job406583 was cancelled before scores. Direct secure 2M++→FP member closure
406586 found3,062 edges (395 heldout) and changed zero group roles; **v5**
is the authoritative prospective split. Source-selected count+FP same-prior-IC
gradient/support control406585 COMPLETED4m32s on role-identical v4: all
occupied sky count cells have positive intensity and IC directional gradient
agrees to9.10e-6 relative, with5.40GiB host peak. This is a numerical
partial-target bridge, **not** a fitted heldout prediction, calibrated
source/group law, converged sampler or R2/N256 posterior. Old identity-mass
HMC remains closed. No email. Details in the split record.

R2 observable-rate repair2026-09-27: the source-selected count control's
all-faint Schechter rate makes the predicted `[-25,-21]` bright fraction fall
from0.09596 at alpha=-.94 to0.01634 at alpha=-.99, creating artificial
rate/shape coupling near the unobserved faint-tail divergence. The new joint
target uses a finite `[-25,-21]` intrinsic reference-rate coordinate, with
the old centre's physical intensity preserved. Typed-H100406587
COMPLETED/exit0 with7/7 focused tests, including JIT and old/new equivalence.
This exact parameterization
repair does not calibrate the faint LF, survey/group selection, bias/FoG or
sampler; no R2 posterior or N256 map follows. See
`CF4_R2_LF_RATE_IDENTIFIABILITY_20260927.md`. No email.

Train-only FP singleton prior sensitivity2026-09-27: typed-H100406598
COMPLETED/exit0, comparing selected d²ρ, geometrical d² and log-distance-flat
radial weights on3,535 overlap-free training FP groups in one unconditioned
N128 PM state. Per-group score differences are generally small, but this is
neither a fitted/posterior result nor a selected-group calibration; heldout
marks were untouched. Do not promote a singleton-only model or use cross-state
scores as information fractions. Next observation-law work must jointly own
linked count/FP marks and address selected-group inclusion/shared covariance
with source-backed evidence before a heldout fit. Details in
`CF4_R2_FP_SINGLETON_PRIOR_SENSITIVITY_20260927.md`. No email.

Public SDSS FP radial-source check2026-09-27: typed-H100406602
COMPLETED/exit0, streaming and hash-verifying4,000,000 official randoms
without retaining the284MB raw file. Their relative observed-z shell density
falls to0.614 at z=.050-.055 and0.239 at z=.095-.100 versus z=.030-.035.
The8,708 in-mask CF4-eligible FP sky-training rows are sharply cut near
z=.061 by the **known observed-group-redshift** d_z<180-cMpc/h eligibility;
the full SDSS random n(z) cannot be inserted as a true-distance group prior.
The source FP PDF already includes its own f_n selection correction. No
heldout distance marks or field scores were used; full-source redshifts/randoms
are not an independent sky holdout. Group incidence, count/FP joint ownership
and sampler calibration remain open. See
`CF4_R2_SDSS_RANDOM_RADIAL_20260927.md`. No email.

R2 direct-link ownership check2026-09-27: the frozen v5 sky graph has3,062
secure counted-point→FP-source-group edges. Training FP groups split into
1,836 with one direct counted point,273 with multiple, and3,925 without a
direct point; heldout roles remain separate. A normalized one-point FP mark
given its observed count key and redshift requires source-to-key intensity
contributions from the SAME count transfer, not a separately assumed `d²ρ^b`
distance prior. Multi-point groups require one shared group latent; unlinked
groups require their own selected-group law. The present count×FP product is
a qualified partial target, not automatically literal double counting or a
complete joint likelihood. No FP heldout mark, field fit, N256, email or R2
posterior. Next implementation ownership and limits:
`CF4_R2_LINKED_POINT_MARK_OWNERSHIP_20260927.md`.

R2 one-link source-kernel entry2026-09-27: typed-H100406603/406605 passed
the source-to-count-key TSC/K/RSD identity and smooth-case continuous radial
normalization controls. The actual-source one-group H100406607
COMPLETED/exit0: nine tests pass, full count-key mean and its five-K/source
sum agree at0.01243817430227125; the conditional FP mark score on the saved
**unconditional** N128/384 state is-0.3097834, with velocity-scale derivative
-0.4686775166 versus finite difference-0.4686775148. This selected training
group was frozen by identities/geometry before scores, not chosen by fit.
Association/FP/group redshift covariance and inclusion remain uncalibrated,
GH3-vs-continuous error at actual keys is not measured, and the all-source
single-key reference is not scalable to full HMC. No heldout prediction,
field fit, R2/N256 posterior, LG identity or email. Source and limits:
`CF4_R2_LINKED_POINT_MARK_OWNERSHIP_20260927.md`.
Post-control signed-LOS-branch completion406610 passed all nine tests; the
actual one-group control406607 preceded this small addition, whose reverse
branch underflows at that fixed83.923-cMpc/h point. No score or posterior
promotion follows from the branch test.

R2 count-kernel quadrature2026-09-27: frozen training split v5 has47,542
count points/38,194 population-voxel keys. On one saved **unconditional**
N128/384 state, dynamic one-GH-node accumulation passed9/9 focused tests and
GH3/9/15; GH21 was compared to the frozen means without repeating those
orders. Relative to GH21, GH15 has p95 key change0.0475%, p99 0.315%, and
maximum20.0%; its total training Poisson score differs by only+0.028. The
largest relative errors are count-1, low-predicted-rate population3 keys at
the180-cMpc/h radial-selection edge. GH3 differs by-4.615 in total score and
has p95 0.616%; its extreme tail is not uniformly negligible. Preserve the
per-key edge qualification; do not infer the continuous individual-redshift
law from a high-order count integral. The R2 posterior is still NO-GO: no
field fit, held-out prediction, calibrated association/group-selection law,
shared multi-member covariance, or N256/LG-resolution result. MW/M31/M33 stay
latent on a new field with M33 unresolved; native truth IDs were not used.
The next R2 calculation is the actual-source continuous-radius linked-mark
comparison and scalable key support, with held-out marks untouched. Details,
including preserved OOM and reporting failures: `CF4_R2_COUNT_QUADRATURE_20260927.md`.
No email.

Update2026-09-28 R2 Tempel parent source/denominator check: Slurm407002
completed5m14s/exit0. Exact SDSS-PV↔Tempel joins match33,641/34,059 rows,
including9,945/10,020 currently eligible FP rows; no matched group/richness
conflicts. An independent position/redshift-only Tempel-member↔2M++ bridge
has17,714 reciprocal one-to-one member links. This is catalogue identity
evidence only. The p99 nearest-random mask proxy has98.998% sensitivity but
43.390% false-positive rate on native `in_mask=0`; therefore the68,103
proxy-footprint denominator and its derived rates are **withdrawn**. Howlett
et al. describe `in_mask` as membership in the NYU-VAGC DR7 polygon mask, not a
complete DR8–DR14 selection function. The output JSON's Tempel/FP row-count
fields are wrong (`len(dict)` gives7/8); parser gates establish584,449 and
34,059, and the script is corrected without rerunning the preserved job.
Six mixed-role Tempel parents invalidate v5 split closure for this expanded
graph; five are within the now-withdrawn proxy-mask/redshift window, while
one is outside it. No heldout field score, fit, posterior, map or gravity run.
R2 remains NO-GO. A bounded exact-name search did not recover the official
DR7 MANGLE polygon or a verifiably identical mirror, and the NYU host fails
strict TLS validation after redirect; no verification bypass was used.
Close this denominator subroute descriptively, not R2. Next build a v6
source-graph-closed split that buffers mixed train/heldout components and
their aggregated count keys before further fitting. Details, Q-GOAL/Q-LEAN
and MW/M31/M33 limits:
`CF4_R2_TEMPEL_PARENT_DENOMINATOR_20260928.md`. No email.

R2 source-graph closure2026-09-28: source-pinned H100 Slurm407069 completed
after three preserved pre-output submission/schema failures. The new v6 split
closes count points/keys, M2++ GIDs, Tempel parents, CF4 groups and FP source
groups under frozen observed links. It preserves an exact count projection:
47,121 train,8,475 heldout and1,642 buffered points;879 keys are withheld from
their respective count exposure windows. The 1,642-row buffer includes421
formerly training and1,221 formerly heldout points. Twenty-seven graph
components mixed train/heldout roles; existing buffers propagated to74 total
components. CF4/FP buffer roles increase by131/36. This is leakage control
under a conservative, uncalibrated catalogue-association graph—not physical
membership, a field likelihood or a posterior. No heldout score, field fit,
gravity, sampler or N256 run. Exact NYU mask denominator route is closed
descriptively; R2 remains NO-GO. Use
`/gpfs/kjhan/CF4/z0_density/r2_sky_closed_split_v6/split.npz` for prospective
expanded-graph fitting, excluding its train/heldout buffered exposure keys.
Q-GOAL: enables clean future same-field heldout assessment. Q-LEAN: one source
identity closure, no new simulation or calibration ladder. MW/M31 remain
ambiguous latent roles, M33 remains unresolved when unsupported; all three
observables must act on that same NEW evolved field and native identities are
evaluation-only. Details and limits: `CF4_R2_GRAPH_CLOSURE_20260928.md`.

R2 linked-FP live-support geometry check2026-09-28: the proposed static
original-direction ray list is not exact under the current periodic RSD
operator. A 20 cMpc/h coherent displacement of a source at minimum-image
relative position (190,30,0) in the 384 cMpc/h box wraps x and changes the
post-RSD radial direction; at observed radius177.37 cMpc/h (inside the180 cut)
the current operator contributes to a TSC voxel missed by both signed
original rays. A focused regression test covers this failure. Do not adopt a
no-wrap posterior restriction or change the operator silently. The existing
8-sigma neighborhood is built from actual shifted positions at one saved
field, so it is a frozen-state control, not live-field support. Typed-H100
Slurm407082 then compared all1,414 v6 secure one-point/one-FP-row training
links against the full2,097,152-source calculation: max normalized log-factor
error1.78e-15 and max relative five-bin density-sum error1.74e-13. The v6
training link set removes v5 groups T60475/T66514/T81318 and adds none. A
focused two-source wrap regression confirms the dynamically shifted-position
neighborhood matches the full factor while the original-ray list omits the
wrapped source. These are fixed-state/support mechanics only: the 8-sigma
truncation remains an approximation, no live-field posterior, sampler,
heldout score or N256 result exists. R2 still requires defensible
selection/association and shared group-redshift/FP covariance, followed by a
stationary live-field posterior and untouched heldout prediction; do not
advance to R3 or label R2 complete without them. MW/M31/M33 remain
latent same-new-field roles (MW/M31 ambiguity; M33 may be unresolved); native
truth IDs remain evaluation-only. Q-GOAL: this validates one necessary
same-field likelihood factor and preserves graph-closed heldout separation.
Q-LEAN: one full-source reference comparison, no new simulation/archive/search
ladder or further threshold sweep. Details:
`CF4_R2_LIVE_SUPPORT_GEOMETRY_20260928.md`. No email.

R2 bounded follow-up Slurm407305 COMPLETED/exit0 in2m51s on source commit
`7e0071670d342d8d8ee6c0a61254a9da8c421051`. The v6 membership screen parsed
8,901 training eta rows. Matched observed-cz/magnitude contrasts are
descriptive only and do not calibrate true-distance inclusion; heldout eta
values were not read. A mark-blind 48-link dynamic-neighborhood stress at
velocity scales0.5/1/1.5 matches full-source factors to floating-point
precision for those fixed-state copies, not arbitrary live posterior states.
The existing one-box TNG TSC residual screen gives isotropic 1D sigma
122.3km/s for resolved centrals and307.0km/s for satellites. This is auxiliary
scale evidence, not CF4/2M++ calibration or a role classifier. Result files
and limits are recorded in `CF4_R2_SDSS_MOCK_DISPOSITION_20260928.md`.
R2 remains NO-GO: no field fit, posterior, or heldout prediction. The next
candidate is the explicitly partial v6 N128 fit/evaluation; it must preserve
the buffered-key exclusion and untouched heldout count/singleton-FP data, and
must not be promoted as a calibrated posterior while selection, association,
and shared covariance remain unresolved. No N256 or R3 escalation follows.

R2 fit-readiness repair2026-09-28: commit68b0de9 allows the continuous linked
FP radial factor to differentiate a traced positive LOS-width nuisance and
adds a population-specific flattened exposure-mask regression for the v6
buffered-key exclusions. All12 focused marked-tracer tests pass in the
project's `circle` environment (161.0s, two CPU threads); no field fit or
heldout value was evaluated. Fixed-state neighborhoods remain invalid as a
live optimizer support. The next required action remains wiring refreshed or
provably bounded singleton support into one v6 training objective, with the
985 Tempel-grouped links kept out of the singleton term. R2 remains NO-GO;
preserve untouched heldout counts/marks, association and shared-covariance
limits, and latent MW/M31/M33 same-field treatment. Details:
`CF4_R2_SDSS_MOCK_DISPOSITION_20260928.md`. No N256, R3, or email.
