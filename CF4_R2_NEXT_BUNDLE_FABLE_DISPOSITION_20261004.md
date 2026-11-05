# R2 bundle disposition — Fable5 applicability audit, 2026-10-04

## Verdict and driver decision

Fable5 returned **CONDITIONAL PASS** for the proposed next R2 bundle. The
driver adopts its applicability correction and bounded shared-factor checks,
but not a claim that the joint count/mark likelihood is already available.
The ray exposure calculation read the predecessor disjoint catalogue
(`r2_common_catalogue_128_v1`: 24,993 training and 8,375 held-out counts),
while the active target is the inclusive v6 split (47,121 training counts)
whose count path uses the shell-CDF/TSC source-to-key kernel. The ray operator
is not wired into that active path. Its NSIDE pairwise statistics therefore
remain frozen historical diagnostics, not a v6 resolution, likelihood, or
posterior gate. No v6 ray rerun or NSIDE4096 escalation is adopted.

## Independently checked evidence

- `CF4_R2_COMMON_COSMOLOGY_CONTRACT_20260925.md` documents the predecessor
  disjoint counts (24,993 train / 8,375 held out).
- `scripts/cf4_r2_raw_joint_pilot.py` reads
  `r2_sky_closed_split_v6/split.npz` and its 47,121 training counts.
- `src/cf4_r2_raw_volume_target.py` routes the active count term through
  `count_field_loglike` and `predict_source_volume_intensity`, i.e. the
  shell-CDF/TSC source-to-key path; it does not consume the ray exposure
  artifact.
- `conditional_mark_log_likelihood_from_shared_latent` in
  `src/cf4_2mpp_joint_likelihood_local.py` implements the normalized
  log-sum-exp identity, but is a generic reference primitive. The current
  `joint_log_likelihood` is explicitly additive/development-only and does not
  pass a common latent into the count kernel.
- `scripts/cf4_r2_count_fp_ownership_control.py` is a one-point saved-field
  control; it checks count-key training ownership and a conditional raw-FP
  score, but does not implement a common latent for the two T10106 members.
  The score-blind identity control records T10106 as 2M++ recnos52802/52824,
  FP rows PGC54049/54054, CF4 1PGC53982, and 2M++ GID2887. These catalog
  associations do not establish physical group membership.

## Adopted bounded continuation (A2–A4)

1. Build one source-only v6 training ownership ledger: each count key/count is
   owned once by the 2M++ count factor; each secure or ambiguous crossmatch is
   classified; each CF4 group mark is identified; and groups are separated
   into count-only, linked-singleton, multi-member/shared-latent, and
   unanchored-conditional classes. A collision/ambiguous association must be
   represented as a mixture or flagged unresolved, never silently duplicated.
   Do not read held-out values or score a field.
2. Inspect the T10106 saved-state mechanics against actual kernel interfaces.
   A shared-latent calculation may proceed only if a well-defined latent node
   controls both the source-to-count-key factor and both group marks with
   compatible measures and a normalized denominator. Do not substitute a
   radius-conditioning argument for a derived common latent or invent a
   covariance, selection, or velocity model.
3. Test zero-information and one-member reductions, compare a real
   two-member shared-latent conditional to the present additive comparator,
   and run one AD/FD check only if the actual common-latent path exists. If it
   does not, report a precise technical NO-GO and missing contract rather
   than a synthetic “joint score.”

The driver rejects repeating any source census already present in the v6
identity/overlap artifacts, re-running the ray exposure on v6 keys, an NSIDE
sweep, another PM evolution, posterior fitting, or held-out scoring. After
this bounded factor-ownership decision, continue to the next material R2
blocker: external or otherwise identifiable calibration of selection,
group inclusion, member covariance and bias/RSD-FoG. The selected mock lacks
the recovered-group/preselection parent needed to estimate those terms;
state them as unidentified rather than tuning them in-sample.

## Goal checks

- **Q-GOAL:** This bundle is necessary observation-law plumbing for the
  CF4-conditioned z=0 field, but is not itself a field result and cannot
  satisfy R2. MW/M31 remain role-ambiguous and M33 unresolved; their
  observables must constrain those same roles in the same NEW evolved LG field
  at `<=0.3 cMpc/h`. Truth identities are calibration/evaluation-only.
- **Q-LEAN:** The source ledger plus one saved-state T10106 mechanics decision
  is proportionate. Stop at a technical NO-GO if the real shared latent is not
  defined by the current interfaces. Do not add another census, a synthetic
  covariance, ray-resolution escalation, PM run, fit, or posterior gate.

Fable's review is advisory. Its useful applicability finding was adopted
because the split counts and active source code independently confirm it; any
future shared-latent factor must likewise be supported by explicit kernel and
measure checks before adoption.

## Execution outcome — A2 and actual-kernel A3/A4

The source-only ledger completed as typed-H100 job412227 (4 seconds,
MaxRSS3,520KiB; three focused tests passed) at
`/gpfs/kjhan/CF4/z0_density/r2_v6_factor_ownership_ledger_20261004_v2/`.
It exactly reconstructs the47,121 v6 training count total using37,951 unique
count-cell factors; every key has one count owner. It records6,028 training
CF4 groups and8,901 FP rows, each with one group owner. Crossmatch classes are
2,682 secure,6,181 unmatched,30 extended-review candidates and8 coordinate/
redshift conflicts. Twenty-one secure links reference point recnos absent
from the point manifest;2,661 are direct secure training links. Count-cell
classes are35,535 with no secure linked CF4 group,2,287 with one, and129 with
multiple. Group classes are3,871 unanchored selected-group conditionals,21
anchor-marked/no-direct-count groups,31 ambiguous-only groups,27 secure-plus-
unresolved groups,1,820 clean one-point groups and258 clean multi-member
candidates. There is no cross-group secure-recno collision.
The earlier v1 ledger is retained; its overbroad ambiguity label was corrected
in v2 and must not be used for interpretation.

The proposed real T10106 control is a **technical NO-GO** with existing
interfaces. `predict_source_volume_intensity` accepts source positions,
velocities, rates and angular completeness, but no group IDs/shared latent;
`count_field_loglike` only scores sparse counts against that global intensity.
The raw-FP `source_conditioning_radius_cMpc_h` argument is a supplied per-row
conditioning value, not a shared inferred latent with a probability measure.
`conditional_group_mark_logpdf` requires a calibrated covariance on group
velocity, group distance modulus and member velocities; no such calibrated
T10106 covariance is available. The generic shared-latent log-sum-exp helper
and its synthetic tests do not couple these real kernels. Accordingly, no
T10106 joint score, one-member reduction, or AD/FD check was claimed. The
failure mode is missing model/input contract, not an optimizer or numerical
failure. A3/A4 remain open until a common latent is derived and both real
kernels consume it.

R2 remains NO-GO. The next advisory should choose between recovering a
preselection/recovered-group parent and its selection process, or explicitly
restricting the estimand to a conditional target that does not claim
unidentified group inclusion. Do not run another catalogue census, ray
resolution sweep, PM evolution, field fit or posterior while that choice is
being assessed. Q-GOAL and Q-LEAN, plus same-field ambiguous MW/M31 and
unresolved M33 handling, remain mandatory for the next plan.

## Follow-up plan audit and driver adjudication — 2026-10-04

Fable5's read-only plan audit returned **CONDITIONAL PASS** for a narrow
conditional estimand using the active v6 count factors and admitted FP rows.
The driver adopts the narrow scope, the exact-cohort reconciliation, and the
recommendation not to invent group inclusion, shared-member covariance or
FoG calibration. The driver does **not** adopt the audit's claim that the
active target already conditions on each linked count point: source inspection
and the active call path showed the opposite.

`src/cf4_r2_raw_volume_target.py::raw_field_logpdf` forwards the optional
`source_conditioning_radius_cMpc_h`, but both active callers in
`scripts/cf4_r2_raw_field_profile.py` and
`scripts/cf4_r2_raw_joint_pilot.py` omitted it. The low-level default in
`src/cf4_r2_raw_live_mark.py::chunk_log_terms` then used
`observation['radius']`, which is the CF4 group radius. The v6 loader did
compute the linked 2M++ point radius, but previously used it only for the
observed voxel/selection. This is a substantive target-definition gap, not a
wording difference. The conditional Route2 target is therefore not yet
validated until the new fixed-state profile succeeds.

Typed-H100 Slurm job412308 performed the promised source-only join. It
reconstructed the active cohort in exact PGC order (1,414/1,414) and found
1,414 `one_linked_count_point` rows, 1,414 secure direct-edge statuses and
zero unresolved associations in this cohort. No raw mark values, held-out
values, field, score, fit or PM evolution was read. Fifteen focused tests
passed; MaxRSS was4,249,636KiB against8GiB requested. The full row-level
reconciliation and hashes are in
`/gpfs/kjhan/CF4/z0_density/r2_v6_active1414_association_reconciliation_20261004_v1/`.
The previous failed attempts412296 and412307 stopped in tests before the
data join; their logs and failure artifacts are preserved. The first exposed
an incorrect selected-PGC test fixture; the second exposed an overmodified
legacy support-test fixture. Both were corrected before412308.

The implementation now computes a PGC-order-checked vector of individual
2M++ point radii, fails closed on unresolved group associations, and carries
that vector through `FreshRawSupport` and `raw_field_logpdf` into the source
selection kernel. The frozen v6 cohort remains all1,414 rows because none is
unresolved; the exclusion is a future-proof conditional-scope rule, not a
post-hoc cut on current scores. The47,121 training count factors are not
changed. Current fixed-state profile is a separate new v2 result and must not
compare against the old v1 CF4-radius readout.

Fable's proposed saved-state citations 408412/409024 were not valid for this
mechanics check: 408412 accepted 0/4 proposals and 409024 was a re-anchored
proposal diagnostic. The driver instead used the saved N128 terminal state
from job 408337 (accepted proposal 7, nonstationary) only for a fixed-state
target/gradient profile. No extra PM evolution was needed to produce that
state. The proposed option to admit one chosen member from multi-member
groups is not adopted; the actual count term aggregates the source set, so
picking one would require an explicit joint ownership model.

### Driver's current bundle checks

- **Q-GOAL:** exact linkage radius conditioning is necessary for the active
  CF4-conditioned same-field z=0 observation law. This is a mechanics repair,
  not a delivered map or evidence that the conditional model captures all
  CF4 galaxies. MW/M31 remain role-ambiguous and M33 unresolved; later
  observables must constrain those same roles on the same NEW evolved LG
  field at `<=0.3 cMpc/h`, with truth IDs used only for calibration/evaluation.
- **Q-LEAN:** one exact 1,414-row ownership join, four small regression suites
  and one corrected fixed-state profile; no new ray sweep, census or PM run
  for mechanics. No held-out score or posterior promotion.
- Fixed-state profile job 412315 completed/exit 0 on H100 in 10:02. It used
  all 1,414 rows, point radii 22.301–179.868 cMpc/h, and fresh support of
  19,348,137 components. Fast/legacy maximum raw-row difference was
  3.84e-6; full directional AD-vs-FD relative error was 3.98e-6. Device
  temporary memory was 3.11/3.87 GiB; host MaxRSS was 4,271,420 KiB under
  the10GiB request. The score is a finite fixed-state diagnostic, not a field
  posterior or z=0 delivery. Its historical v1 reference was not reused.
- Next bundle: one eight-proposal PM joint-target pilot at the same N128
  saved IC, with all47,121 count factors and1,414 linked-point FP factors.
  PM is evaluated only as the forward map inside that inference target, not
  as a standalone gravity simulation. Prior same-run measured MaxRSS was
  7.92GiB against24GiB; lower request10GiB leaves >20% observed headroom.
  Driver-led audit (rather than consecutive Fable review) confirms Q-GOAL:
  this is the minimum dynamic test that count and linked-point marks act on
  one field, but it is not the delivered N256/1.5 map. Q-LEAN: one existing
  bounded eight-proposal pilot, no separate PM comparison, ray sweep or
  sampler grid. It must stop on restart, finiteness, exact primal, AD/FD or
  memory failure; short transitions cannot establish stationarity. MW/M31
  stay role-ambiguous and M33 unresolved; their observables must later
  constrain the same NEW evolved LG field at `<=0.3 cMpc/h`.

## Joint-pilot outcome and corrected N256 handoff

Typed-H100 Slurm job412327 completed the corrected N128/3 joint target in
32:23. It made8 proposals,7 accepted overall and3/4 after warmup; maximum
absolute Hamiltonian error was1.075. Restart field identity and the exact
same-state primal passed. The joint PM/raw/count directional AD-vs-FD relative
error was1.33e-3 against the predeclared2e-3 gate. Slurm MaxRSS was7,070,068KiB
under10GiB; the estimated GPU peak18.36GiB was within its69.81GiB device
limit. White mean-square changed0.6493->0.6981. This short trace establishes
that the corrected joint target can execute bounded transitions, not
stationarity, mixing, posterior UQ, heldout performance or a z=0 map. The
machine-readable record is
`/gpfs/kjhan/CF4/z0_density/r2_raw_joint_pilot_v2_linked_point_radius/result.json`.

During continuation, driver source comparison found that the earlier N256
pilot (`e36daa8`) and its48-proposal chains A/B (`4af620d`) did not attach
`source_conditioning_radius_cMpc_h` in their `load_inputs`; `raw_field_logpdf`
therefore fell back to the CF4 group radius. Those outputs are preserved for
their historical target but are not samples under the corrected linked-point
target. Their low-band white-power traces also drifted roughly0.70->0.97 and
their last accepted proposals were iterations39/48 and43/48, respectively.
The corrected N256 target now requires a finite positive radius vector aligned
to every mark, passes it explicitly into the raw factor, and records the
radius range and association-ledger provenance. A focused regression suite
passes5/5.

The CF4 paper describes55,877 galaxy distances from eight methods and notes
large potential systematic uncertainty; a 2026 analysis further shows that
selection/distance-prior choices matter for CF4 and that a principled model
needs the detailed heterogeneous survey selection. This is supporting
context, not a claim that the present raw FP factor is disproven: the active
target explicitly models its known magnitude and velocity-dispersion cuts,
but remains conditional on its selected source/type/link graph and does not
calibrate every inclusion process ([Tully et al. 2023](https://arxiv.org/abs/2209.11238),
[Desmond & Stiskalek 2026](https://academic.oup.com/mnras/article/550/2/stag1144/8709283)).

**Next bounded bundle:** revalidate the corrected target on the approved
N256/384=1.5 source field using the preserved N256 dynamics initializer, all
47,121 training count factors on the frozen observed N128 keys, and the
1,414 reconciled linked-point FP/K rows. Use the existing force-GL1/fine-GL2
four-proposal pilot; do not read heldout outcomes or reuse old N256 chain
draws as posterior samples. Record the linked-radius range, exact source
commit, target/primal/gradient checks, device and host memory, and all
transitions. Q-GOAL: this repairs target identity at the actual global
1.5-cMpc/h grid but is not a map or LG delivery. Q-LEAN: one corrected-target
revalidation, no duplicated exposure/census, new simulation, or sampler
ladder. If it passes, proceed to a corrected-target sampling plan without
waiting for a routine turn approval. R2 remains NO-GO pending credible
stationarity/UQ, a frozen untouched predictive assessment, and a delivered
conditional z=0 map; selection/inclusion limits must remain explicit. MW/M31
remain ambiguous and M33 unresolved; their observables must constrain the
same NEW evolved LG field at `<=0.3 cMpc/h`, truth IDs evaluation-only.
